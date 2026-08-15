"""Triggering-rainfall forecast for the Stallion fire, currently burning.

An emergency assessment normally runs on OBSERVED burn severity from a BARC or
MTBS product. Stallion is still burning, so no severity product exists yet --
which is precisely the gap the pre-fire method fills: every vegetation class is
assumed to burn at its regionally calibrated severity quantile, giving the
hazard the fire *would* pose once contained.

Read the output as a pre-fire expectation for the burned footprint, not as an
emergency assessment. Two consequences:

* it assumes the whole perimeter burns, whereas real fires leave unburned
  islands and a mosaic of severity, so segment thresholds here are on the
  conservative (low) side wherever the fire actually burned lightly;
* the perimeter is today's, and an active fire grows.

Rerun with ``assess.run_observed`` the moment a BARC product lands -- that
comparison is also the M6 validation this project has been waiting for.

Output: per-segment triggering intensity I15 (mm/h) at a 50% modelled
debris-flow likelihood, plus likelihood and hazard class at the 24 mm/h
reference storm.
"""
import json
import sys
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd

from firescape import config, paths, prefire, severity, statewide

FIRE = sys.argv[1] if len(sys.argv) > 1 else "Stallion"
CAL = sys.argv[2] if len(sys.argv) > 2 else "statewide_v1_1"
PERIM = sorted(paths.raw_dir("perimeters").glob("wfigs_current_*.geojson"))[-1]
OUT = paths.products_dir("forecast", f"{FIRE.lower()}_{CAL}")
TILE_DIR = paths.cache_root() / "3dep_tiles"
EVT_DIR = paths.raw_dir("landfire", "LF2025_EVT_nv")   # pre-fire vintage

per = gpd.read_file(PERIM)
name_col = next(c for c in per.columns if c.endswith("IncidentName"))
hit = per[per[name_col].str.fullmatch(FIRE, case=False, na=False)]
if hit.empty:
    sys.exit(f"{FIRE} not in {PERIM.name}")
geom = hit.geometry.union_all()
acres = float(hit["poly_GISAcres"].iloc[0])
print(f"{FIRE}: {acres:,.0f} acres ({acres*0.00404686:.1f} km2) from {PERIM.name}",
      flush=True)

# which calibration region does the fire sit in?
reg = gpd.read_file(
    "/Users/scottmccoy/git/code/firescape/firescape/data/regions/"
    "nv_prefire_regions.geojson").to_crs(per.crs)
pt = gpd.GeoSeries([geom], crs=per.crs).representative_point().iloc[0]
region = reg.loc[reg.contains(pt), "region"]
region = str(region.iloc[0]) if len(region) else "Central Basin and Range"
slug = region.lower().replace(" ", "_").replace("&", "and")
cal = config.packaged_calibration(CAL).region(slug)
print(f"region: {region} -> pdsim {cal.pdsim}, breaks {cal.barc_breaks}", flush=True)

legend_csv = paths.interim_dir("statewide") / "evt_legend_union.csv"
mapping, _ = severity.remap_crosswalk(pd.read_csv(legend_csv))
cfg = prefire.PreFireConfig(
    pdsim=cal.pdsim,
    barc_breaks=cal.barc_breaks,
    evt_path=legend_csv,
    evt_legend_path=legend_csv,
    crosswalk=mapping,
    dem_path=paths.interim_dir("pilot") / "pilot_dem.tif",   # unused (injected)
    region=f"{CAL}:{slug}",
    severity_sigma=0.91,
    emit_ranges_topo=True,
    cdf_table="nv_merged" if CAL == "statewide_v1_1" else "nv_statewide",
    kf_polygons=paths.raw_dir("ssurgo") / "nv_ssurgo_kf.gpkg",
)

b5070 = tuple(gpd.GeoSeries([geom], crs=per.crs).to_crs("EPSG:5070").total_bounds)
pad = 2000.0
b5070 = (b5070[0] - pad, b5070[1] - pad, b5070[2] + pad, b5070[3] + pad)
print("staging DEM + EVT...", flush=True)
dem = statewide.dem_for_unit(b5070, TILE_DIR)
evt = statewide.evt_for_unit(b5070, EVT_DIR)

print("routing flow and delineating segments...", flush=True)
result = prefire.run_unit(geom, cfg, unit_key=FIRE.lower(), crs=per.crs,
                          dem=dem, evt=evt)
prefire.save_unit(result, OUT)
print(f"{result['n_segments']:,} segments -> {OUT}", flush=True)

seg = gpd.read_file(OUT / f"{FIRE.lower()}_segments.gpkg")
thr = seg["I15_50"].to_numpy() if "I15_50" in seg else seg["thresh_i15"].to_numpy()
p24 = seg["P_24mmh"].to_numpy()
ok = np.isfinite(thr)
q = np.percentile(thr[ok], [5, 25, 50, 75, 95])
summary = {
    "fire": FIRE, "perimeter_file": PERIM.name, "acres": round(acres),
    "calibration": CAL, "region": region, "pdsim": cal.pdsim,
    "segments": int(len(seg)),
    "segments_with_threshold": int(ok.sum()),
    "triggering_i15_mmh": {"p05": round(q[0], 1), "p25": round(q[1], 1),
                           "median": round(q[2], 1), "p75": round(q[3], 1),
                           "p95": round(q[4], 1)},
    "likelihood_at_24mmh": {
        "median": round(float(np.nanmedian(p24)), 3),
        "segments_over_0.5": int((p24 > 0.5).sum()),
        "segments_over_0.75": int((p24 > 0.75).sum())},
    "most_responsive_segments_i15_under_20mmh": int((thr[ok] < 20).sum()),
}
if "H_24mmh" in seg:
    h = seg["H_24mmh"].to_numpy()
    summary["hazard_class_at_24mmh"] = {
        str(int(c)): int((h == c).sum()) for c in sorted(set(h[np.isfinite(h)]))}
print(json.dumps(summary, indent=2), flush=True)
(OUT / "forecast_summary.json").write_text(json.dumps(summary, indent=2))
