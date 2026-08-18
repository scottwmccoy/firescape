"""Perry Canyon pilot: underground workings as the waste-rock proxy.

USMIN maps no dump or tailings **extent** anywhere near Perry Canyon, so the
statewide waste ranking is silent there — yet the canyon holds 34 adits and 30
shafts, and Scott confirms the named mines carry real waste-rock piles. That
is the general case, not a local quirk: 90% of the 5 km cells in Nevada that
hold a mine opening hold no mapped waste at all, because a topographer drew a
dump as an extent only where it was large enough to warrant one.

So this runs the same exposure chain over a different asset class — every adit
and shaft in the canyon, treated as the anchor of a portal dump. Two honest
differences from the waste product:

* **The asset is a point, not a footprint.** A portal dump spills downslope
  from the opening and is not mapped, so the distance reported is to the
  opening, and a dump can reach the channel from an opening that does not.
  ``dist_m`` is carried for every working, not just the near-channel ones, so
  the margin is visible rather than hidden behind a yes/no.
* **Nothing here predicts the size of the pile, and two guesses died proving
  it.** Scott's ground truth is that the largest waste-rock pile in the
  district is at rank 7 — an unnamed in-channel adit at (-119.60253,
  39.85373). The first guess was that named mines were the producers: wrong,
  and the ranking had already put the named Jones Kincaid shaft 47th of 50.
  The second was that the pile would sit low in the drainage on a big
  channel: also wrong — rank 7 is 31st of 50 by contributing area (0.11 km²,
  against 1.46 km² for the most-downstream adit) and 23rd by predicted
  volume. Pile size is a function of production history, which USMIN does not
  record in any form. This product can therefore order **exposure**, and
  cannot order **source size**; nothing in it should be read as the latter.
* **Two orderings, because likelihood is not consequence.** ``rank`` sorts by
  annual hit probability — where a flow is most likely to occur at all. The
  delivering segment's contributing area and RANGES volume say something
  different: how big a channel the working sits on and how much material a
  flow there would move. They disagree strongly (the top-volume workings sit
  in the 17th-44th places by probability), so both are written out and the
  field to sort on depends on the question being asked. Neither is a stand-in
  for how much waste rock is actually sitting at the portal.

USMIN is digitized **per quadrangle**, and three quads overlap here — Sutcliffe
(1957), Moses Rock (1980) and Fraser Flat (1980) — so the same adit appears two
or three times, 3-30 m apart under different sheet vintages. Left alone that
inflates a 50-working district to 64 and double-counts the exposed ones, so
records within :data:`DEDUPE_M` are collapsed to one working, keeping the
record that scores highest (a working counts as near-channel if any of its
sheets puts it there).

Writes products/exposure/perry_canyon_v1/.
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
from shapely.geometry import LineString, box as sbox

from firescape import exposure, paths

V12 = paths.products_dir("prefire", "statewide_v1_2")
OUT = paths.products_dir("exposure", "perry_canyon_v1")
CRS = "EPSG:5070"
BOX = (-119.68, 39.78, -119.49, 39.91)
#: GNIS "Perry Canyon" (Washoe Co., valley): mouth and head.
AXIS = LineString([(-119.60927, 39.86253), (-119.56104, 39.82705)])
PAD, PAD_M, NEAR_M, RECEPTOR_M = 2000.0, 25.0, 1000.0, 10_000.0
#: Records this close are the same working seen on two overlapping quads.
DEDUPE_M = 30.0

mines = gpd.read_file(paths.raw_dir("usmin") / "nv_mines.gpkg")
work = mines[(mines["group"] == "openings")
             & mines.geometry.within(sbox(*BOX))].reset_index(drop=True)
work = work.to_crs(CRS)
work.index = pd.Index([f"pc{i:03d}" for i in range(len(work))], name="site_id")
axis = gpd.GeoSeries([AXIS], crs=4326).to_crs(CRS).iloc[0]
work["dist_axis_m"] = work.distance(axis).round(0)
work["named"] = work["name"].notna()

# One id per physical working, across the overlapping quad sheets.
import scipy.sparse as _sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

_xy = np.c_[work.geometry.x, work.geometry.y]
_pairs = list(cKDTree(_xy).query_pairs(r=DEDUPE_M))
_m = _sp.coo_matrix((np.ones(len(_pairs)),
                     ([p[0] for p in _pairs], [p[1] for p in _pairs])),
                    shape=(len(work), len(work)))
work["working"] = connected_components(_m, directed=False)[1]
n_work = work["working"].nunique()
print(f"{len(work)} USMIN records -> {n_work} distinct workings "
      f"(quads: {', '.join(sorted(set(work['topo_name'].dropna())))})")
print(work["ftr_type"].value_counts().to_string())
named = sorted(set(work.loc[work["named"], "name"].astype(str)))
print(f"named: {len(named)} — {', '.join(named)}")

units = sorted(V12.glob("[0-9]*_segments.gpkg"))
ub = json.loads(
    (paths.interim_dir("exposure") / "unit_bounds_v1_2.json").read_text())
SEGCOLS = ["Segment_ID", "Area_km2", "P_24mmh", "V_24mmh", "H_24mmh", "I15_50"]
bb = work.total_bounds
frames = []
for path in units:
    huc = path.name.split("_")[0]
    b = ub.get(huc)
    if b is None or (b[2] < bb[0] - PAD or b[0] > bb[2] + PAD
                     or b[3] < bb[1] - PAD or b[1] > bb[3] + PAD):
        continue
    segs = pyogrio.read_dataframe(
        path, columns=SEGCOLS,
        bbox=(bb[0] - PAD, bb[1] - PAD, bb[2] + PAD, bb[3] + PAD))
    if len(segs):
        segs.index = pd.Index([f"{huc}-{i}" for i in segs["Segment_ID"]])
        frames.append(exposure.hazard_at_assets(
            work, segs, pad_m=PAD_M, near_max_m=NEAR_M,
            cols=("P_24mmh", "V_24mmh", "H_24mmh", "I15_50", "Area_km2")))
        print(f"  unit {huc}: {len(segs):,} segments", flush=True)
if not frames:
    raise SystemExit("no modelled channels near Perry Canyon")

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
        "dist_axis_m", "working"]
full = gpd.GeoDataFrame(ann.join(work[keep + ["geometry"]]),
                        geometry="geometry", crs=CRS)

# Collapse the quad duplicates: best-scoring record wins, so a working counts
# as near-channel if any sheet puts it there. The dropped records stay in the
# GPKG as a second layer rather than vanishing.
order = full.sort_values(["exposed", "named", "P_annual_site", "dist_m"],
                         ascending=[False, False, False, True],
                         na_position="last")
best = order[~order["working"].duplicated(keep="first")].copy()
best["n_quad_records"] = full.groupby("working").size().reindex(
    best["working"]).to_numpy()
best["quads"] = full.groupby("working")["topo_name"].apply(
    lambda s: ", ".join(sorted(set(s.dropna().astype(str))))
).reindex(best["working"]).to_numpy()
print(f"collapsed {len(full)} records -> {len(best)} workings")
out = exposure.rank(best)
out = out.rename(columns={"Area_km2": "seg_area_km2"})
# The consequence ordering, kept beside the likelihood one.
out["volume_rank"] = out["V_24mmh"].rank(ascending=False,
                                         method="min").astype("Int64")

OUT.mkdir(parents=True, exist_ok=True)
out.to_file(OUT / "perry_openings.gpkg", layer="workings", driver="GPKG")
full.to_file(OUT / "perry_openings.gpkg", layer="usmin_records", driver="GPKG")
ll = out.geometry.to_crs(4326)
out.drop(columns="geometry").assign(lon=ll.x.round(5), lat=ll.y.round(5)) \
   .to_csv(OUT / "perry_openings.csv")

sha = subprocess.run(["git", "-C", str(paths.Path(__file__).resolve().parents[2]),
                      "rev-parse", "--short", "HEAD"],
                     capture_output=True, text=True).stdout.strip()
(OUT / "run_meta.json").write_text(json.dumps({
    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "git_sha": sha,
    "aoi": {"box4326": list(BOX), "gnis_feature": "Perry Canyon (Washoe Co.)",
            "axis4326": list(AXIS.coords)},
    "assets": "USMIN group=openings (adits + shafts) as portal-dump proxies; "
              "USMIN maps NO waste extent in this area",
    "caveat": "the asset is the opening, not the dump: a portal dump spills "
              "downslope and is unmapped, so a dump may reach a channel from "
              "an opening that does not",
    "params": {"pad_m": PAD_M, "near_max_m": NEAR_M,
               "receptor_max_m": RECEPTOR_M},
    "dedupe": {"records": int(len(full)), "workings": int(len(out)),
               "join_distance_m": DEDUPE_M,
               "why": "USMIN is digitized per quadrangle and three overlap "
                      "here, so one adit appears on two or three sheets"},
    "n_workings": int(len(out)), "n_named": int(out["named"].sum()),
    "n_near_channel": int(out["exposed"].sum()),
    "n_on_blm": int(out["on_blm"].sum()),
}, indent=2) + "\n")

near = (~out["exposed"]) & out["dist_m"].notna()
print(f"\nwithin 30–55 m of a channel: {int(out['exposed'].sum())}   "
      f"within 1 km: {int(near.sum())}   beyond: {int(out['dist_m'].isna().sum())}")
print(f"on BLM-managed land: {int(out['on_blm'].sum())} of {len(out)}")
print(f"distance to channel (m): median {out['dist_m'].median():.0f}, "
      f"min {out['dist_m'].min():.0f}")
show = out.copy()
show["site"] = show["name"].fillna(show["ftr_type"])
show["RI_yr"] = (1.0 / show["P_annual_site"]).round(0)
cols = ["site", "ftr_type", "dist_m", "P_24mmh", "P_annual_site", "RI_yr",
        "water_name", "water_dist_m", "on_blm"]
print("\n=== named workings (Scott: these carry the real waste-rock piles) ===")
print(show[show["named"]][cols].to_string())
print("\n=== top 12 by annual hit probability ===")
print(show.head(12)[cols].to_string())
print(f"\nwrote {OUT}")
