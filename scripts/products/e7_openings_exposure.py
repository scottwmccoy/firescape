"""Statewide exposure for underground workings -> products/exposure/openings_v1.

The companion asset class to ``e2_aml_exposure``. USMIN maps a dump or
tailings **extent** only where a topographer drew one, which is 8% of the 5 km
cells in Nevada that hold a mine opening — so the waste ranking is blank over
most of the state's mining country, Perry Canyon included. Adits and shafts
are the other half of the record: 38,712 of them, every one the anchor of a
portal dump that was never mapped.

Same chain as the waste product, with the two lessons Perry Canyon taught:

* **Deduplicate across quadrangles.** USMIN is digitized per sheet, and
  neighbouring quads overlap, so one adit appears two or three times a few
  metres apart under different sheet vintages. At Perry Canyon that inflated
  50 workings to 64.
* **This orders exposure, not source size.** Nothing in USMIN records how
  much rock came out of a working, and at Perry Canyon neither the mine name
  nor the size of the receiving channel predicted where the large pile was.
  The ranking says which workings sit where a debris flow is likely to reach
  them; it does not say which holds the most material.
"""
import json
import subprocess
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely.geometry import box as sbox

from firescape import exposure, paths

V12 = paths.products_dir("prefire", "statewide_v1_2")
OUT = paths.products_dir("exposure", "openings_v1")
CRS = "EPSG:5070"
PAD, PAD_M, NEAR_M, RECEPTOR_M, DEDUPE_M = 2000.0, 25.0, 1000.0, 10_000.0, 30.0

mines = gpd.read_file(paths.raw_dir("usmin") / "nv_mines.gpkg")
work = mines[mines["group"] == "openings"].reset_index(drop=True).to_crs(CRS)
xy = np.c_[work.geometry.x, work.geometry.y]
pairs = list(cKDTree(xy).query_pairs(r=DEDUPE_M))
m = sp.coo_matrix((np.ones(len(pairs)),
                   ([p[0] for p in pairs], [p[1] for p in pairs])),
                  shape=(len(work), len(work)))
work["working"] = connected_components(m, directed=False)[1]
print(f"{len(work):,} USMIN opening records -> "
      f"{work['working'].nunique():,} distinct workings", flush=True)
work.index = pd.Index([f"op{i:05d}" for i in range(len(work))], name="site_id")
work["named"] = work["name"].notna()

cachef = paths.interim_dir("exposure") / "unit_bounds_v1_2.json"
ubounds = json.loads(cachef.read_text())
units = sorted(V12.glob("[0-9]*_segments.gpkg"))
SEGCOLS = ["Segment_ID", "Area_km2", "P_24mmh", "V_24mmh", "H_24mmh", "I15_50"]
frames, touched = [], 0
for k, path in enumerate(units):
    huc = path.name.split("_")[0]
    w, s, e, n = ubounds[huc]
    idx = work.sindex.query(sbox(w - PAD, s - PAD, e + PAD, n + PAD),
                            predicate="intersects")
    if not len(idx):
        continue
    sub = work.iloc[sorted(idx)]
    bb = sub.total_bounds
    segs = pyogrio.read_dataframe(
        path, columns=SEGCOLS,
        bbox=(bb[0] - PAD, bb[1] - PAD, bb[2] + PAD, bb[3] + PAD))
    if not len(segs):
        continue
    segs.index = pd.Index([f"{huc}-{i}" for i in segs["Segment_ID"]])
    frames.append(exposure.hazard_at_assets(
        sub, segs, pad_m=PAD_M, near_max_m=NEAR_M,
        cols=("P_24mmh", "V_24mmh", "H_24mmh", "I15_50", "Area_km2")))
    touched += 1
    if touched % 50 == 0:
        print(f"  {touched} units joined ({k + 1}/{len(units)} scanned)",
              flush=True)
print(f"{touched} units carried workings", flush=True)

hz = exposure.combine_best(frames).reindex(work.index)
hz["exposed"] = hz["exposed"].fillna(False).astype(bool)
hz["n_deliver"] = hz["n_deliver"].fillna(0).astype(int)

basins = pyogrio.read_dataframe(V12 / "statewide_v1_2_basins.gpkg",
                                columns=["I15_1yr", "I15_50yr", "P_F"])
ann = exposure.site_annual(hz, work, basins)
receptors = gpd.read_file(
    paths.interim_dir("exposure") / "nhd_receptors.gpkg").to_crs(CRS)
rec = exposure.nearest_receptor(work, receptors, cols=("name", "kind"),
                                max_m=RECEPTOR_M)
ann["water_dist_m"] = rec["water_dist_m"]
ann["water_name"] = rec["name"]
ann["water_kind"] = rec["kind"]
blm = gpd.read_file(paths.raw_dir("blm") / "nv_blm_sma.gpkg").to_crs(CRS)
ann["on_blm"] = exposure.within_any(work, blm)

keep = ["name", "ftr_type", "county", "topo_name", "topo_date", "named",
        "working"]
full = gpd.GeoDataFrame(ann.join(work[keep + ["geometry"]]), geometry="geometry",
                        crs=CRS)
order = full.sort_values(["exposed", "named", "P_annual_site", "dist_m"],
                         ascending=[False, False, False, True],
                         na_position="last")
best = order[~order["working"].duplicated(keep="first")].copy()
best["n_quad_records"] = full.groupby("working").size().reindex(
    best["working"]).to_numpy()
out = exposure.rank(best).rename(columns={"Area_km2": "seg_area_km2"})
out["volume_rank"] = out["V_24mmh"].rank(ascending=False,
                                         method="min").astype("Int64")
print(f"collapsed {len(full):,} records -> {len(out):,} workings")

OUT.mkdir(parents=True, exist_ok=True)
out.to_file(OUT / "openings_exposure.gpkg", layer="workings", driver="GPKG")
ll = out.geometry.to_crs(4326)
out.drop(columns="geometry").assign(lon=ll.x.round(5), lat=ll.y.round(5)) \
   .to_csv(OUT / "openings_ranked.csv")

sha = subprocess.run(["git", "-C", str(paths.Path(__file__).resolve().parents[2]),
                      "rev-parse", "--short", "HEAD"],
                     capture_output=True, text=True).stdout.strip()
near = (~out["exposed"]) & out["dist_m"].notna()
(OUT / "run_meta.json").write_text(json.dumps({
    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "git_sha": sha,
    "assets": "USMIN group=openings (adits, shafts) as portal-dump proxies",
    "orders": "exposure, NOT source size — USMIN records no measure of how "
              "much rock a working produced",
    "dedupe": {"records": int(len(full)), "workings": int(len(out)),
               "join_distance_m": DEDUPE_M},
    "params": {"pad_m": PAD_M, "near_max_m": NEAR_M,
               "receptor_max_m": RECEPTOR_M},
    "n_workings": int(len(out)), "n_near_channel": int(out["exposed"].sum()),
    "n_within_1km": int(near.sum()), "n_on_blm": int(out["on_blm"].sum()),
}, indent=2) + "\n")

print(f"\nwithin 30–55 m of a channel: {int(out['exposed'].sum()):,}   "
      f"within 1 km: {int(near.sum()):,}   "
      f"beyond: {int(out['dist_m'].isna().sum()):,}")
print(f"on BLM-managed land: {int(out['on_blm'].sum()):,} of {len(out):,}")
top = out.head(10).copy()
top["site"] = top["name"].fillna(top["ftr_type"])
top["RI_yr"] = (1.0 / top["P_annual_site"]).round(0)
print(top[["site", "ftr_type", "county", "dist_m", "P_annual_site", "RI_yr",
           "on_blm"]].to_string())
print(f"\nwrote {OUT}")
