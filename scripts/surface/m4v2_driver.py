"""Regenerate the pilot pre-fire surface with pilot_v2 (nv_merged, P_dsim 0.51)
and real SSURGO soils. Resumable/time-budgeted; exit 42 = budget hit."""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
from shapely.geometry import box

from firescape import config, paths, prefire, severity
from firescape.regions import PILOT_BBOX

T0 = time.time()
BUDGET = float(os.environ.get("FIRESCAPE_M4_BUDGET", 3300))
ONLY = os.environ.get("FIRESCAPE_ONLY")
OUT = paths.products_dir("prefire", "pilot_v2")

cal = config.packaged_calibration("pilot_v2").region("pilot")
lf = paths.raw_dir("landfire") / "LF2025_EVT_pilot"     # present-day fuels for a pre-fire map
legend = lf / "LF2025_EVT_pilot.tif.vat.dbf"
mapping, _ = severity.remap_crosswalk(gpd.read_file(legend))

cfg = prefire.PreFireConfig(
    pdsim=cal.pdsim,
    barc_breaks=cal.barc_breaks,
    evt_path=lf / "LF2025_EVT_pilot.tif",
    evt_legend_path=legend,
    crosswalk=mapping,
    dem_path=paths.interim_dir("pilot") / "pilot_dem.tif",
    region="pilot",
    cdf_table="nv_merged",
    kf_polygons=paths.raw_dir("ssurgo") / "pilot_ssurgo_kf.gpkg",
)
print(f"pilot_v2: pdsim={cfg.pdsim}, table={cfg.cdf_table}, breaks={cfg.barc_breaks}",
      flush=True)

hu = gpd.read_file(paths.interim_dir("pilot") / "pilot_hu10.geojson")
hu["geometry"] = hu.geometry.intersection(box(*PILOT_BBOX))
hu = hu[~hu.geometry.is_empty].sort_values("clip_km2").reset_index(drop=True)
if ONLY:
    hu = hu[hu["huc10"] == ONLY]

done, pending = [], []
for _, row in hu.iterrows():
    key = str(row["huc10"])
    if (OUT / f"{key}_meta.json").exists():
        done.append(key)
        continue
    if time.time() - T0 > BUDGET:
        pending.append(key)
        continue
    t1 = time.time()
    try:
        result = prefire.run_unit(row.geometry, cfg, unit_key=key, crs=hu.crs)
        prefire.save_unit(result, OUT)
        m = result["meta"]
        print(f"{key} {str(row.get('name'))[:24]:<24} {row['clip_km2']:6.0f} km2 -> "
              f"{result['n_segments']:5d} segs, kf={str(m.get('kf_source'))[:16]}, "
              f"{time.time()-t1:.0f}s", flush=True)
        done.append(key)
    except Exception as e:
        print(f"{key} FAILED: {type(e).__name__}: {str(e)[:150]}", flush=True)

print(f"\nprogress: {len(done)} done, {len(pending)} pending, {time.time()-T0:.0f}s", flush=True)
sys.exit(42 if pending else 0)
