"""Stallion assessed on OBSERVED burn severity, from CIMSS BRISK.

Until now every Stallion product assumed the whole perimeter burned at its
regionally calibrated severity quantile, because the fire is still burning and
the authoritative severity products (BAER SBS, then MTBS) arrive weeks to a
year late. ``stormscape.burn`` reads the BRISK near-real-time dNBR composite,
which is updated daily, so the same hazard chain can now run on severity that
was actually measured.

Three things to keep straight:

* **Units.** BRISK dNBR is unscaled (about -0.3 to 1.0); pfdf's severity and
  M1 code want dNBR x1000. Converted here once, at the boundary.
* **Maturity.** BRISK's own guidance is that a composite younger than ~14 days
  has the right *pattern* but under-reads *magnitude* until it ingests a clear
  Landsat/Sentinel overpass. Stallion's is a day old, so treat the severity --
  and therefore the hazard -- as a floor.
* **dNBR is vegetation change, not soil burn severity.** The USGS models are
  calibrated on BAER SBS, which adjusts dNBR for hydrophobicity and ground
  cover. This is an interim answer, to be superseded when BAER lands.

Writes the observed-severity assessment beside the pre-fire forecast so the
two can be differenced.
"""
import datetime as dt
import json
import sys
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np

from firescape import assess, config, paths, statewide

FIRE = sys.argv[1] if len(sys.argv) > 1 else "Stallion"
CAL = sys.argv[2] if len(sys.argv) > 2 else "statewide_v1_1"
OUT = paths.products_dir("forecast", f"{FIRE.lower()}_observed")
TILE_DIR = paths.cache_root() / "3dep_tiles"

per_all = gpd.read_file(
    sorted(paths.raw_dir("perimeters").glob("wfigs_current_*.geojson"))[-1])
ncol = next(c for c in per_all.columns if c.endswith("IncidentName"))
per = per_all[per_all[ncol].str.fullmatch(FIRE, case=False, na=False)].to_crs(4326)
if per.empty:
    sys.exit(f"{FIRE} not in the current perimeter file")

# --- observed severity ------------------------------------------------------
from stormscape import burn

res = burn.burn_severity(tuple(per.total_bounds), fires=[FIRE], scheme="usgs")
if res is None:
    sys.exit(f"BRISK has no scene intersecting {FIRE}")
meta = res["meta"]
print(f"BRISK {meta['scene_dates']} over {meta['fires']}: "
      f"{meta['burned_px']:,} burned px, dNBR p98 {meta['dnbr_p98']:.3f}",
      flush=True)
print("class fractions:", json.dumps(meta.get("class_fraction", {}), indent=2),
      flush=True)

OUT.mkdir(parents=True, exist_ok=True)
sev_tif = OUT / f"{FIRE.lower()}_brisk_dnbr.tif"
import rasterio

prof = dict(res["profile"])
prof.update(dtype="float32", nodata=np.nan)
with rasterio.open(sev_tif, "w", **prof) as ds:
    ds.write(res["fields"]["dnbr"].astype("float32"), 1)
paths.write_provenance(
    sev_tif, url="https://bin.ssec.wisc.edu/pub/realearth/brisk",
    note=(f"CIMSS BRISK near-real-time dNBR composite, scenes "
          f"{meta['scene_dates']}; UNSCALED dNBR (not x1000). Interim product: "
          "vegetation change, not soil burn severity."))

# x1000 for pfdf, written as the raster the hazard chain reads
dnbr_x1000 = OUT / f"{FIRE.lower()}_brisk_dnbr_x1000.tif"
with rasterio.open(dnbr_x1000, "w", **prof) as ds:
    ds.write((res["fields"]["dnbr"] * 1000.0).astype("float32"), 1)
print(f"wrote {sev_tif.name} and {dnbr_x1000.name}", flush=True)

# --- hazard chain on observed severity --------------------------------------
reg = gpd.read_file("/Users/scottmccoy/git/code/firescape/firescape/data/"
                    "regions/nv_prefire_regions.geojson").to_crs(4326)
pt = per.geometry.union_all().representative_point()
hit = reg.loc[reg.contains(pt), "region"]
region = str(hit.iloc[0]) if len(hit) else "Central Basin and Range"
slug = region.lower().replace(" ", "_").replace("&", "and")
cal = config.packaged_calibration(CAL).region(slug)
# cal.barc_breaks IS the BARC4 threshold triple already -- config.py,
# severity.classify_barc4 and calibrate.py all define it as
# (unburned-low, low-moderate, moderate-high), and calibrate writes it as
# [125, break_lowmod, 500]. This line used to rebuild it as
# [low, (low+mod)/2, mod], which treats the *low-moderate* break as if it were
# moderate-high and invents a low-moderate at the midpoint. That drops the
# boundary M1 actually pools on (moderate+high) from 325 to 225 x1000 and
# inflated the moderate-or-high burned area 1.9x on Hawk, 2.4x on Stallion and
# 4.7x on Bug -- straight into the F term, so every observed likelihood ran
# high. The pre-fire path (prefire.run_unit) always passed it through
# unchanged; this is now consistent with it.
breaks = list(cal.barc_breaks)             # unburned/low/moderate/high, x1000
print(f"{region}: BARC breaks {breaks} (x1000 dNBR)", flush=True)

b5070 = tuple(per.to_crs("EPSG:5070").total_bounds)
pad = 3500.0
dem = statewide.dem_for_unit((b5070[0] - pad, b5070[1] - pad,
                              b5070[2] + pad, b5070[3] + pad), TILE_DIR)

result = assess.run_observed(
    FIRE.lower(), dem=dem, out_dir=OUT,
    kf_polygons=paths.raw_dir("ssurgo") / "nv_ssurgo_kf.gpkg",
    perimeter=per.to_crs(dem.crs), dnbr=str(dnbr_x1000), barc_breaks=breaks)
print(f"{result.get('n_segments', 'n/a')} segments -> {OUT}", flush=True)

# Age from the scene itself, not a constant: the first run happened to be a
# day old and the 1 got frozen into every summary written since, including
# ones fetched days later. Age is the whole basis for trusting the magnitude.
_scenes = [dt.date.fromisoformat(str(s)[:10]) for s in meta["scene_dates"]]
_age = (dt.date.today() - max(_scenes)).days

summary = {"fire": FIRE, "severity_source": "CIMSS BRISK",
           "scene_dates": meta["scene_dates"],
           "composite_age_days": _age,
           "magnitude_caveat": (
               "under BRISK's ~14 d maturity mark: pattern reliable, magnitude "
               "may still rise" if _age < 14 else "past BRISK's ~14 d mark"),
           "class_fraction": meta.get("class_fraction", {}),
           "barc_breaks_x1000": breaks, "region": region,
           "calibration_for_breaks": CAL}
(OUT / "observed_severity_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
