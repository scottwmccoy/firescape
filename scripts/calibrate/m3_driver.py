"""M3 calibration driver: resumable, time-budgeted. Exit 42 = budget hit, rerun."""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from firescape import calibrate, mtbs, paths, severity
from firescape.regions import PILOT_BBOX

T0 = time.time()
BUDGET = float(os.environ.get("FIRESCAPE_CALIB_BUDGET", 480))
ONLY = os.environ.get("FIRESCAPE_ONLY")

fires = pd.read_csv(
    "/Users/scottmccoy/git/code/firescape/firescape/data/calibration/fire_sets/pilot_v1.csv"
)
usable = fires[fires["include"] & (fires["mod_t"] > 0) & (fires["mod_t"] < 2000)].copy()
usable = usable[usable["event_id"] != "CA3959712021620240902"]  # Bear: BAER SBS thresholds, not MTBS

# overlap with pilot bbox from the downloaded burn_area shapefiles
pilot_geom = box(*PILOT_BBOX)
keep_rows = []
for _, row in usable.iterrows():
    try:
        b = mtbs.fire_bundle(row["event_id"])
    except FileNotFoundError:
        continue
    g = gpd.read_file(b["burn_area"]).to_crs("EPSG:4326").union_all()
    frac = g.intersection(pilot_geom).area / g.area
    if frac >= 0.5:
        keep_rows.append({**row, "pilot_frac": frac})
usable = pd.DataFrame(keep_rows).sort_values("km2")
regional_break = float(np.median(usable["mod_t"]))  # from the FULL usable set
if ONLY:
    usable = usable[usable["event_id"] == ONLY]
print(f"usable fires: {len(usable)}; regional low-mod break = {regional_break:.0f}", flush=True)

vat = gpd.read_file(paths.raw_dir("landfire") / "LF2025_EVT_pilot" / "LF2025_EVT_pilot.tif.vat.dbf")
mapping, _audit = severity.remap_crosswalk(vat)
evt_path = paths.raw_dir("landfire") / "LF2025_EVT_pilot" / "LF2025_EVT_pilot.tif"

done, pending = [], []
for _, row in usable.iterrows():
    eid = row["event_id"]
    if calibrate.load_cached(eid) is not None:
        done.append(eid)
        continue
    if time.time() - T0 > BUDGET:
        pending.append(eid)
        continue
    t1 = time.time()
    try:
        fc = calibrate.fire_calibration(
            eid, (float(row["low_t"]), float(row["mod_t"])), regional_break,
            evt_path=evt_path, crosswalk=mapping)
        print(f"{eid} {row['incid_name']:<14} {row['km2']:7.1f} km2 -> "
              f"pdsim={fc.pdsim if fc.pdsim is None else round(fc.pdsim, 2)} "
              f"(n_sel={fc.n_selected}/{fc.n_delineated}, dT={fc.validation_dT:.4f}, "
              f"dF={fc.validation_dF:.4f}, {time.time()-t1:.0f}s)", flush=True)
        done.append(eid)
    except Exception as e:
        print(f"{eid} FAILED: {type(e).__name__}: {str(e)[:150]}", flush=True)

print(f"\nprogress: {len(done)} done, {len(pending)} pending, "
      f"{time.time()-T0:.0f}s elapsed", flush=True)
sys.exit(42 if pending else 0)
