"""M4 driver: pre-fire hazard surface over the pilot AOI, one HU10 at a time.

Resumable and time-budgeted, like the M3 driver. Exit 42 = budget hit, rerun.
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
from shapely.geometry import box

from firescape import config, prefire, paths, severity
from firescape.regions import PILOT_BBOX

T0 = time.time()
BUDGET = float(os.environ.get("FIRESCAPE_M4_BUDGET", 480))
ONLY = os.environ.get("FIRESCAPE_ONLY")
OUT = paths.products_dir("prefire", "pilot_v1")

cal = config.packaged_calibration("pilot_v1").region("pilot")
lf_dir = paths.raw_dir("landfire") / "LF2025_EVT_pilot"
legend_path = lf_dir / "LF2025_EVT_pilot.tif.vat.dbf"
mapping, _audit = severity.remap_crosswalk(gpd.read_file(legend_path))

cfg = prefire.PreFireConfig(
    pdsim=cal.pdsim,
    barc_breaks=cal.barc_breaks,
    evt_path=lf_dir / "LF2025_EVT_pilot.tif",
    evt_legend_path=legend_path,
    crosswalk=mapping,
    dem_path=paths.interim_dir("pilot") / "pilot_dem.tif",
    region="pilot",
)
print(f"calibration: pdsim={cfg.pdsim}, breaks={cfg.barc_breaks}", flush=True)

hu = gpd.read_file(paths.interim_dir("pilot") / "pilot_hu10.geojson")
pilot = box(*PILOT_BBOX)
hu["geometry"] = hu.geometry.intersection(pilot)
hu = hu[~hu.geometry.is_empty].sort_values("clip_km2").reset_index(drop=True)
if ONLY:
    hu = hu[hu["huc10"] == ONLY]

done, pending, failed = [], [], []
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
        print(f"{key} {str(row.get('name'))[:26]:<26} {row['clip_km2']:6.0f} km2 -> "
              f"{result['n_segments']:5d} segs (of {m.get('segments_delineated', 0)}), "
              f"kf={m.get('kf_source', '?')[:18]}, "
              f"masked={m.get('mask_fraction', {}).get('exclude', 0):.2f}, "
              f"{time.time()-t1:.0f}s", flush=True)
        done.append(key)
    except Exception as e:
        print(f"{key} FAILED: {type(e).__name__}: {str(e)[:160]}", flush=True)
        failed.append(key)

print(f"\nprogress: {len(done)} done, {len(pending)} pending, {len(failed)} failed, "
      f"{time.time()-T0:.0f}s", flush=True)
sys.exit(42 if pending else 0)
