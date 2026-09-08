"""statewide_v1_4: the v1_3 calibration with the BARC256 threshold scale fixed.

Two things change relative to ``p7_calib_driver`` (v1_2/v1_3):

1. The fire set's BAER-programme thresholds are on the dNBR scale
   (``p2b_fireset_barc_scale.py``), so the Sierra Nevada break median is
   350, not 312. Every Sierra fire re-solves from its cached decomposition at
   the new break (~2 s each); the other three regions' breaks are unchanged
   and their fires are found already cached.
2. Fires whose bundle is NOT an MTBS product (no ``dnbr6``) build their
   observed classes from the analyst's own thresholds instead of pfdf's fixed
   125/250/500 -- the same thing ``dnbr6`` gives MTBS fires. Their
   decompositions are rebuilt (``force_decomp``): Davis, Bear, Broom Canyon
   (BAER), Stockade Canyon (RAVG) and Earthstone (an MTBS bundle delivered
   without dnbr6). Their v1_3 caches were moved to
   ``interim/calib/_superseded_v1_3/`` rather than deleted.

    FIRESCAPE_CAL_BUDGET=3000 python scripts/calibrate/p8_calib_driver.py
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
          "nv_statewide": severity.load_cdf_table(table="nv_statewide")}
TAG_SUFFIX = "_d91sw"          # same cache namespace as v1_2/v1_3

df = pd.read_csv(SET)
if "threshold_scale" not in df.columns:
    raise SystemExit("run scripts/stage/p2b_fireset_barc_scale.py first")
use = df[df["include"]].copy()
breaks = (use[(use["mod_t"] > 0) & (use["mod_t"] < 2000)]
          .groupby("region")["mod_t"].median())
print("per-region low-moderate breaks (median analyst mod_t, BARC256-corrected):")
print(breaks.round(1).to_string(), flush=True)

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
    mapping, _ = severity.remap_crosswalk(legend)
    _xwalk[vintage] = (mapping, dirn)
    return _xwalk[vintage]


def observed_breaks_for(row):
    """Analyst (low, mod, high) on the dNBR scale with the sentinel guards
    ``calibrate`` applies elsewhere; None when the fire has no usable mod_t."""
    lo, mo, hi = (float(row[c]) for c in ("low_t", "mod_t", "high_t"))
    if not (0.0 < mo < 2000.0):
        return None
    lo = lo if 0.0 < lo < min(mo, 2000.0) else 125.0
    hi = hi if mo < hi < 2000.0 else 500.0
    return (lo, mo, hi)


done, pending, failed, rebuilt = [], [], [], []
for _, row in use.sort_values("km2", ascending=True).iterrows():
    eid = row["event_id"]
    vintage = row["evt_vintage"]
    brk = float(breaks[row["region"]])
    tag = f"{vintage.split('_')[0].lower()}_b{brk:.0f}{TAG_SUFFIX}"
    if calibrate.load_cached(eid, tag, table="nv_statewide_disp") is not None:
        done.append(eid)
        continue
    if time.time() - T0 > BUDGET:
        pending.append((eid, "budget"))
        continue
    try:
        bundle = mtbs.fire_bundle(eid)
    except FileNotFoundError:
        pending.append((eid, "bundle"))
        continue
    programme = mtbs.bundle_programme(bundle)
    # The rule is about dnbr6, not the programme: one MTBS bundle in this set
    # (Earthstone, 2017) ships no dnbr6 either and had fallen back to pfdf's
    # 125/250/500 exactly like the BAER/RAVG ones.
    non_mtbs = "dnbr6" not in bundle
    obs = observed_breaks_for(row) if non_mtbs else None
    cached = (not non_mtbs) and calibrate._find_decomp(eid, tag) is not None
    mapping, chunk_dir, dem, evt = {}, None, None, None
    if not cached:
        xw = crosswalk_for(vintage)
        if xw is None:
            pending.append((eid, f"evt:{vintage}"))
            continue
        mapping, chunk_dir = xw
    t1 = time.time()
    try:
        if not cached:
            perim = gpd.read_file(bundle["burn_area"]).to_crs("EPSG:5070")
            w, s, e, n = perim.union_all().buffer(calibrate.PERIMETER_BUFFER_M).bounds
            dem = statewide.dem_for_unit((w, s, e, n), TILE_DIR)
            evt = statewide.evt_for_unit((w, s, e, n), chunk_dir)
        res = calibrate.fire_calibration(
            eid, (float(row["low_t"]), float(row["mod_t"])), brk,
            evt=evt, crosswalk=mapping, dem=dem, cdf_tables=TABLES,
            dispersed_sigma=0.91, observed_breaks=obs, force_decomp=non_mtbs, tag=tag)
        fc = res["nv_statewide_disp"]
        ps = {k: (f"{v.pdsim:.2f}" if v.ok else "-") for k, v in res.items()}
        note = f" [{programme} bundle, observed at {obs}]" if non_mtbs else ""
        print(f"{eid} {str(row['incid_name'])[:16]:<16} {row['region'][:22]:<22} "
              f"{row['km2']:6.0f} km2 b{brk:.0f} -> pdsim {ps} (n={fc.n_selected}, "
              f"{time.time()-t1:.0f}s){note}", flush=True)
        done.append(eid)
        if non_mtbs:
            rebuilt.append(eid)
    except Exception as exc:
        print(f"{eid} FAILED: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
        failed.append(eid)

print(f"\ncalib: {len(done)} done ({len(rebuilt)} rebuilt on analyst thresholds), "
      f"{len(pending)} pending, {len(failed)} failed ({time.time()-T0:.0f}s)", flush=True)
if pending:
    print("pending by reason:")
    print(pd.Series([r for _, r in pending]).value_counts().to_string(), flush=True)
sys.exit(42 if pending or failed else 0)
