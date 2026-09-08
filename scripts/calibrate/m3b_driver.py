"""Recalibrate P_dsim against nv_merged, era-matched, with Staley as control.

Both CDF tables are solved in the same pass per fire, so the ONLY difference
between them is the per-class dNBR vector: delineation, terrain, observed T/F
and the catchment class weights are shared. The regional break is held at 281
(the 34-fire median from pilot_v1) so the table is the only variable.
"""
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
BUDGET = float(os.environ.get("FIRESCAPE_CALIB_BUDGET", 3300))
TAG = "lf2016"
BREAK = 281.0
MIN_YEAR = 2017          # era-matched: LF2016 vegetation predates these fires

EVT_DIR = paths.raw_dir("landfire") / "LF2016_EVT_pilot"
EVT_TIF = EVT_DIR / "LF2016_EVT_pilot.tif"
mapping, _ = severity.remap_crosswalk(gpd.read_file(EVT_DIR / "LF2016_EVT_pilot.tif.vat.dbf"))

TABLES = {"staley2018": severity.load_cdf_table(table="staley2018"),
          "nv_merged": severity.load_cdf_table(table="nv_merged")}

fires = pd.read_csv(paths.package_data("calibration", "fire_sets", "pilot_v1.csv"))
use = fires[fires["include"] & (fires["mod_t"] > 0) & (fires["mod_t"] < 2000)
            & (fires["ig_year"] >= MIN_YEAR)
            & (fires["event_id"] != "CA3959712021620240902")].copy()   # Bear: BAER
pilot = box(*PILOT_BBOX)
keep = []
for _, row in use.iterrows():
    try:
        b = mtbs.fire_bundle(row["event_id"])
    except FileNotFoundError:
        continue
    g = gpd.read_file(b["burn_area"]).to_crs("EPSG:4326").union_all()
    if g.intersection(pilot).area / g.area >= 0.5:
        keep.append(row)
use = pd.DataFrame(keep).sort_values("km2")
print(f"era-matched calibration fires (>= {MIN_YEAR}): {len(use)}; break held at {BREAK:.0f}",
      flush=True)

done, pending = [], []
for _, row in use.iterrows():
    eid = row["event_id"]
    if time.time() - T0 > BUDGET:
        pending.append(eid)
        continue
    t1 = time.time()
    try:
        res = calibrate.fire_calibration(
            eid, (float(row["low_t"]), float(row["mod_t"])), BREAK,
            evt_path=EVT_TIF, crosswalk=mapping, cdf_tables=TABLES, tag=TAG)
        st = res["staley2018"].pdsim
        nv = res["nv_merged"].pdsim
        print(f"{eid} {row['incid_name'][:16]:<16} {row['km2']:6.1f} km2 -> "
              f"staley={st if st is None else round(st, 2)}  "
              f"nv_merged={nv if nv is None else round(nv, 2)}  "
              f"(n_sel={res['nv_merged'].n_selected}, {time.time()-t1:.0f}s)", flush=True)
        done.append(eid)
    except Exception as e:
        print(f"{eid} FAILED: {type(e).__name__}: {str(e)[:140]}", flush=True)

print(f"\nprogress: {len(done)} done, {len(pending)} pending, {time.time()-T0:.0f}s", flush=True)
sys.exit(42 if pending else 0)
