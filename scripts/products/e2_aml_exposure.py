"""AML waste exposure vs the statewide v1.2 network -> products/exposure/aml_v2.

Joins the staged USMIN waste footprints (dumps, tailings, ponds) against the
per-unit v1.2 segment products with exposure.hazard_at_assets (corridor-width
delivery, nearest-within-1-km otherwise), computes per-site annual rates
through the merged-basin rain climatology, names each site's nearest NHD
receptor water (perennial or GNIS-named, 10 km question radius; needs
e1c_stage_nhd_receptors.py), flags sites on BLM-managed land (needs
e1b_stage_blm.py), and writes the ranked site table.

v1 (NE rivers/lakes receptors, no manager flag) is preserved in
products/exposure/aml_v1 as run at commit 57af036.
"""
import json
import subprocess
import sys
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from shapely.geometry import box as sbox

from firescape import exposure, paths

V12 = paths.products_dir("prefire", "statewide_v1_2")
OUT = paths.products_dir("exposure", "aml_v2")
CRS = "EPSG:5070"
PAD = 2000.0        # m around assets when reading a unit; > near_max + w_max/2
PAD_M, NEAR_M = 25.0, 1000.0
RECEPTOR_M = 10_000.0   # beyond this, "no mapped receptor" is the answer

mines = gpd.read_file(paths.raw_dir("usmin") / "nv_mines.gpkg")
waste = mines[mines["group"] == "waste"].reset_index(drop=True).to_crs(CRS)
waste.index = pd.Index([f"aml{i:04d}" for i in range(len(waste))], name="site_id")
waste["footprint_m2"] = waste.geometry.area
print(f"{len(waste)} waste assets "
      f"({int((waste.geom_kind == 'area').sum())} polygons)", flush=True)

# per-unit bounds, cached (591 gpkg header reads on first run only)
units = sorted(V12.glob("[0-9]*_segments.gpkg"))
cachef = paths.interim_dir("exposure") / "unit_bounds_v1_2.json"
if cachef.exists():
    ubounds = json.loads(cachef.read_text())
else:
    ubounds = {p.name.split("_")[0]: list(pyogrio.read_info(p)["total_bounds"])
               for p in units}
    cachef.write_text(json.dumps(ubounds))
print(f"{len(units)} network units", flush=True)

SEGCOLS = ["Segment_ID", "Area_km2", "P_24mmh", "V_24mmh", "H_24mmh", "I15_50"]
frames, touched = [], 0
for k, path in enumerate(units):
    huc = path.name.split("_")[0]
    w, s, e, n = ubounds[huc]
    idx = waste.sindex.query(sbox(w - PAD, s - PAD, e + PAD, n + PAD),
                             predicate="intersects")
    if not len(idx):
        continue
    sub = waste.iloc[sorted(idx)]
    bb = sub.total_bounds
    segs = pyogrio.read_dataframe(
        path, columns=SEGCOLS,
        bbox=(bb[0] - PAD, bb[1] - PAD, bb[2] + PAD, bb[3] + PAD))
    if not len(segs):
        continue
    segs.index = pd.Index([f"{huc}-{i}" for i in segs["Segment_ID"]])
    frames.append(exposure.hazard_at_assets(
        sub, segs, pad_m=PAD_M, near_max_m=NEAR_M))
    touched += 1
    if touched % 50 == 0:
        print(f"  {touched} units joined ({k + 1}/{len(units)} scanned)",
              flush=True)
if not frames:
    sys.exit("no unit reached any asset")
print(f"{touched} units carried assets", flush=True)

hz = exposure.combine_best(frames).reindex(waste.index)
hz["exposed"] = hz["exposed"].fillna(False).astype(bool)
hz["n_deliver"] = hz["n_deliver"].fillna(0).astype(int)

basins = pyogrio.read_dataframe(
    V12 / "statewide_v1_2_basins.gpkg",
    columns=["I15_1yr", "I15_50yr", "P_F"])
print(f"{len(basins)} basins for site climatology", flush=True)
ann = exposure.site_annual(hz, waste, basins)

receptors = gpd.read_file(
    paths.interim_dir("exposure") / "nhd_receptors.gpkg").to_crs(CRS)
print(f"{len(receptors)} NHD receptor features", flush=True)
rec = exposure.nearest_receptor(waste, receptors, cols=("name", "kind"),
                                max_m=RECEPTOR_M)
ann["water_dist_m"] = rec["water_dist_m"]
ann["water_name"] = rec["name"]
ann["water_kind"] = rec["kind"]

blm = gpd.read_file(paths.raw_dir("blm") / "nv_blm_sma.gpkg").to_crs(CRS)
ann["on_blm"] = exposure.within_any(waste, blm)
print(f"{int(ann['on_blm'].sum())} sites on BLM-managed land", flush=True)

ranked = exposure.rank(ann)
keep = ["name", "ftr_type", "county", "topo_name", "topo_date", "geom_kind",
        "footprint_m2"]
out = gpd.GeoDataFrame(ranked.join(waste[keep + ["geometry"]]),
                       geometry="geometry", crs=CRS)

OUT.mkdir(parents=True, exist_ok=True)
out.to_file(OUT / "aml_exposure.gpkg", layer="waste_sites", driver="GPKG")
ll = out.geometry.representative_point().to_crs(4326)
csv = out.drop(columns="geometry").assign(lon=ll.x.round(5), lat=ll.y.round(5))
csv.to_csv(OUT / "aml_ranked.csv")

sha = subprocess.run(["git", "-C", str(paths.Path(__file__).resolve().parents[2]),
                      "rev-parse", "--short", "HEAD"],
                     capture_output=True, text=True).stdout.strip()
(OUT / "run_meta.json").write_text(json.dumps({
    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "git_sha": sha,
    "inputs": {"network": "products/prefire/statewide_v1_2 (per-unit segments)",
               "basins": "statewide_v1_2_basins.gpkg",
               "assets": "raw/usmin/nv_mines.gpkg (group=waste)",
               "assets_sha256": json.loads(
                   (paths.raw_dir("usmin") / "nv_mines.gpkg.provenance.json")
                   .read_text()).get("sha256"),
               "waters": "interim/exposure/nhd_receptors.gpkg (NHDPlus-HR "
                         "perennial or GNIS-named flowlines + named "
                         "waterbodies; straight-line, not routed)",
               "lands": "raw/blm/nv_blm_sma.gpkg (BLM SMA, ~30 m "
                        "simplification)"},
    "params": {"pad_m": PAD_M, "near_max_m": NEAR_M, "unit_pad_m": PAD,
               "receptor_max_m": RECEPTOR_M,
               "width_law": "w_min 9 + 12*sqrt(A) capped 60"},
    "n_assets": int(len(out)), "n_exposed": int(out["exposed"].sum()),
}, indent=2) + "\n")

near = (~out["exposed"]) & out["dist_m"].notna()
print(f"\nexposed (corridor hit): {int(out['exposed'].sum())}   "
      f"near (<{NEAR_M:.0f} m): {int(near.sum())}   "
      f"clear: {int(out['dist_m'].isna().sum())}")
exp_blm = out["exposed"] & out["on_blm"]
print(f"on BLM-managed land: {int(out['on_blm'].sum())} of {len(out)} "
      f"({int(exp_blm.sum())} of the exposed)")
top = out.head(15).copy()
top["site"] = top["name"].fillna(top["ftr_type"])
top["RI_yr"] = (1.0 / top["P_annual_site"]).round(0)
top["water_km"] = (top["water_dist_m"] / 1000).round(1)
top["receptor"] = top["water_name"].fillna("(" + top["water_kind"] + ")") \
    .fillna("none<10km")
print(top[["site", "county", "P_24mmh", "P_annual_site", "RI_yr",
           "n_deliver", "receptor", "water_km", "on_blm"]].to_string())
print(f"\nwrote {OUT}")
