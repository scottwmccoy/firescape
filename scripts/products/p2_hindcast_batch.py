"""Run observed-severity hindcasts + figures for a list of MTBS fires.

Usage: python p2_hindcast_batch.py EID [EID ...]
Skips fires whose assess products already exist; appends one summary row per
fire to products/assess/hindcast_summary.csv.
"""
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import geopandas as gpd
import pandas as pd

from firescape import assess, calibrate, mtbs, paths, statewide

FIG = str(Path(__file__).parent / "p2_hindcast_fig.py")
PY = paths.python_executable()

tgt = pd.read_csv(paths.products_dir("prefire", "statewide_v0")
                  / "validation_targets.csv")
names = dict(zip(tgt["event_id"],
                 zip(tgt["incid_name"], tgt["ig_year"])))

rows = []
for eid in sys.argv[1:]:
    d = paths.products_dir("assess", eid)
    nm, yr = names.get(eid, (eid, "?"))
    label = f"{str(nm).title()}, {yr}"
    if not (d / f"{eid}_basins.gpkg").exists():
        bundle = mtbs.fire_bundle(eid)
        perim = gpd.read_file(bundle["burn_area"]).to_crs("EPSG:5070")
        b = perim.union_all().buffer(calibrate.PERIMETER_BUFFER_M).bounds
        dem = statewide.dem_for_unit(tuple(b), paths.cache_root() / "3dep_tiles")
        assess.run_observed(eid, dem=dem,
                            kf_polygons=paths.raw_dir("ssurgo") / "nv_ssurgo_kf.gpkg")
        print(f"{label}: hindcast done", flush=True)
    r = subprocess.run([PY, FIG, eid, label], capture_output=True, text=True)
    print(f"{label}: figure rc={r.returncode} "
          f"{r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-200:]}",
          flush=True)
    bas = gpd.read_file(d / f"{eid}_basins.gpkg")
    rows.append({
        "event_id": eid, "fire": nm, "ig_year": yr, "n_basins": len(bas),
        "n_modplus": int((bas["H_24mmh"] >= 2).sum()),
        "n_high": int((bas["H_24mmh"] == 3).sum()),
        "max_P": round(float(bas["P_24mmh"].max()), 2),
        "min_thresh_mmh": round(float(bas["I15_50"].min()), 1),
        "max_V_m3": int(bas["V_24mmh"].max()),
    })

out = paths.products_dir("assess") / "hindcast_summary.csv"
new = pd.DataFrame(rows)
if out.exists():
    prev = pd.read_csv(out)
    new = pd.concat([prev[~prev["event_id"].isin(new["event_id"])], new],
                    ignore_index=True)
new.to_csv(out, index=False)
print("\n" + new.to_string(index=False))
