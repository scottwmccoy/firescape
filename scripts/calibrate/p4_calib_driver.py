"""Statewide per-region P_dsim recalibration under DISPERSED severity (sigma=0.91).

Era-matched: each fire pairs with the newest LANDFIRE EVT strictly predating
its ignition year (LF2016/LF2020/LF2022 chunk stores). DEM comes from the
statewide 3DEP tile cache per fire (fixes the pilot DEM-edge fires). Breaks
are per-region medians of the analyst mod_t thresholds (Rossi's rule).
Both CDF tables are solved in the same pass (exact comparison).
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd

from firescape import calibrate, mtbs, paths, severity, statewide

T0 = time.time()
BUDGET = float(os.environ.get("FIRESCAPE_CAL_BUDGET", 520))
SET = (paths.package_data("calibration", "fire_sets", "statewide_v1.csv"))
TILE_DIR = paths.cache_root() / "3dep_tiles"
TABLES = {"staley2018": severity.load_cdf_table(),
          "nv_merged": severity.load_cdf_table(table="nv_merged")}

df = pd.read_csv(SET)
use = df[df["include"]].copy()
# mod_t=9999 = MTBS "no threshold" sentinel; keep it out of the break
# median (it inflated v1/v1_1/v1_2: central 325->307.5, mojave 388->345,
# northern 355->317.5). Sentinel fires stay in the P_dsim pool.
breaks = (use[(use["mod_t"] > 0) & (use["mod_t"] < 2000)]
          .groupby("region")["mod_t"].median())
print("per-region low-moderate breaks (median analyst mod_t):")
print(breaks.round(0).to_string(), flush=True)

_xwalk: dict = {}


def crosswalk_for(vintage):
    if vintage in _xwalk:
        return _xwalk[vintage]
    short = vintage.split("_")[0]
    dirn = paths.raw_dir("landfire", f"{short}_EVT_nv")
    vats = sorted(dirn.glob("*/evt_*.tif.vat.dbf"))
    if not vats:
        _xwalk[vintage] = None
        return None
    legend = pd.concat([gpd.read_file(v) for v in vats],
                       ignore_index=True).drop_duplicates(subset="Value")
    legend.drop(columns=[c for c in legend.columns if c == "geometry"],
                errors="ignore").to_csv(
        paths.interim_dir("statewide") / f"evt_legend_{short}.csv", index=False)
    mapping, _ = severity.remap_crosswalk(legend)
    _xwalk[vintage] = (mapping, dirn)
    return _xwalk[vintage]


done, pending, failed = [], [], []
for _, row in use.sort_values("km2", ascending=False).iterrows():
    eid = row["event_id"]
    vintage = row["evt_vintage"]
    brk = float(breaks[row["region"]])
    tag = f"{vintage.split('_')[0].lower()}_b{brk:.0f}_d91"
    if calibrate.load_cached(eid, tag, table="nv_merged_disp") is not None:
        done.append(eid)
        continue
    if time.time() - T0 > BUDGET:
        pending.append((eid, "budget"))
        continue
    xw = crosswalk_for(vintage)
    if xw is None:
        pending.append((eid, f"evt:{vintage}"))
        continue
    mapping, chunk_dir = xw
    try:
        bundle = mtbs.fire_bundle(eid)
    except FileNotFoundError:
        pending.append((eid, "bundle"))
        continue
    t1 = time.time()
    try:
        perim = gpd.read_file(bundle["burn_area"]).to_crs("EPSG:5070")
        w, s, e, n = perim.union_all().buffer(
            calibrate.PERIMETER_BUFFER_M).bounds
        dem = statewide.dem_for_unit((w, s, e, n), TILE_DIR)
        evt = statewide.evt_for_unit((w, s, e, n), chunk_dir)
        res = calibrate.fire_calibration(
            eid, (float(row["low_t"]), float(row["mod_t"])), brk,
            evt=evt, crosswalk=mapping, dem=dem, cdf_tables=TABLES,
            dispersed_sigma=0.91, tag=tag)
        fc = res["nv_merged_disp"]
        ps = {k: (f"{v.pdsim:.2f}" if v.ok else "-") for k, v in res.items()}
        print(f"{eid} {str(row['incid_name'])[:16]:<16} {row['region'][:22]:<22} "
              f"{row['km2']:6.0f} km2 {vintage[:6]} -> pdsim {ps} "
              f"(n={fc.n_selected}, {time.time()-t1:.0f}s)", flush=True)
        done.append(eid)
    except Exception as exc:
        print(f"{eid} FAILED: {type(exc).__name__}: {str(exc)[:140]}", flush=True)
        failed.append(eid)

print(f"\ncalib: {len(done)} done, {len(pending)} pending, {len(failed)} failed "
      f"({time.time()-T0:.0f}s)", flush=True)
by_reason = pd.Series([r for _, r in pending]).value_counts() if pending else None
if by_reason is not None:
    print("pending by reason:")
    print(by_reason.to_string(), flush=True)

# interim per-region medians over whatever is cached so far
rows = []
for _, row in use.iterrows():
    brk = float(breaks[row["region"]])
    tag = f"{row['evt_vintage'].split('_')[0].lower()}_b{brk:.0f}"
    for table in list(TABLES) + [f"{t}_disp" for t in TABLES]:
        fc = calibrate.load_cached(row["event_id"], tag, table=table)
        if fc is not None and fc.ok:
            rows.append({"region": row["region"], "table": table,
                         "pdsim": fc.pdsim})
if rows:
    med = (pd.DataFrame(rows).groupby(["region", "table"])["pdsim"]
           .agg(["median", "count"]))
    print("\nregional P_dsim so far:")
    print(med.to_string(), flush=True)

sys.exit(42 if pending else 0)
