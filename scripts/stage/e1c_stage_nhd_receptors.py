"""NHD receptor waters around the AML sites -> interim/exposure/nhd_receptors.gpkg.

Receptors are the waters a manager would name in a pollution conversation:
NHDPlus-HR flowlines that are perennial or carry a GNIS name (streams,
rivers, named canals, the named artificial paths that thread reservoirs),
plus named waterbodies. Ephemeral unnamed washes are transport, not
receptors, and stay out — that is also what makes a statewide pull feasible,
since the filter runs server-side.

Fetched per 0.5-degree cell around the staged waste sites with 0.15-degree
pad, so any receptor within the 10 km question radius of a site is present.
Cells cache as geojson and re-runs skip them (a regeneratable query product,
so it lives in interim/, not raw/).
"""
import json
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd
from stormscape import refdata

from firescape import paths

CELL, PADD = 0.5, 0.15
FIELDS = "permanent_identifier,gnis_name,fcode"
W_FL = "fcode = 46006 OR (gnis_name IS NOT NULL AND gnis_name <> '')"
W_WB = "gnis_name IS NOT NULL AND gnis_name <> ''"
OUTDIR = paths.interim_dir("exposure")
NHDDIR = OUTDIR / "nhd"
NHDDIR.mkdir(parents=True, exist_ok=True)

mines = gpd.read_file(paths.raw_dir("usmin") / "nv_mines.gpkg")
pts = mines[mines["group"] == "waste"].geometry.representative_point()
cells = sorted({(float(np.floor(x / CELL) * CELL),
                 float(np.floor(y / CELL) * CELL))
                for x, y in zip(pts.x, pts.y)})
print(f"{len(cells)} cells for {len(pts)} sites", flush=True)


def kind_of(fcode, src):
    fc = int(fcode) if pd.notna(fcode) else 0
    if src == "wb":
        return "waterbody"
    if fc == 46006:
        return "perennial stream"
    if fc in (46000, 46003, 46007):
        return "named stream"
    if 33600 <= fc <= 33699:
        return "canal/ditch"
    return "river path"          # named artificial path / connector


def fetch_cell(cell):
    """Fill this cell's two caches (skip when present); thread-safe."""
    cx, cy = cell
    tag = f"{cx:+08.2f}_{cy:+07.2f}"
    b = (cx - PADD, cy - PADD, cx + CELL + PADD, cy + CELL + PADD)
    for src, layer, where in (("fl", 3, W_FL), ("wb", 9, W_WB)):
        f = NHDDIR / f"{src}_{tag}.geojson"
        if not f.exists():
            g = refdata.arcgis_query(refdata.NHDPLUS_HR, layer, b,
                                     out_fields=FIELDS, where=where,
                                     timeout=180, what=f"NHD {src} {tag}")
            f.write_text(g.to_json() if len(g)
                         else '{"type": "FeatureCollection", "features": []}')
    return tag


from concurrent.futures import ThreadPoolExecutor, as_completed

with ThreadPoolExecutor(max_workers=5) as pool:
    futs = [pool.submit(fetch_cell, c) for c in cells]
    for k, fut in enumerate(as_completed(futs)):
        fut.result()
        print(f"  cell {k + 1}/{len(cells)} done", flush=True)

frames = []
for cx, cy in cells:
    tag = f"{cx:+08.2f}_{cy:+07.2f}"
    for src in ("fl", "wb"):
        f = NHDDIR / f"{src}_{tag}.geojson"
        if json.loads(f.read_text()).get("features"):
            g = gpd.read_file(f)
            g["src"] = src
            frames.append(g)

rec = pd.concat(frames, ignore_index=True)
rec = rec[~rec.duplicated(["src", "permanent_identifier"])].reset_index(drop=True)
rec["kind"] = [kind_of(fc, s) for fc, s in zip(rec.get("fcode"), rec["src"])]
rec["name"] = rec.get("gnis_name")
rec = gpd.GeoDataFrame(rec[["name", "kind", "fcode", "src", "geometry"]],
                       crs="EPSG:4326")
rec.to_file(OUTDIR / "nhd_receptors.gpkg", layer="receptors", driver="GPKG")
print(f"{len(rec)} receptor features "
      f"({(rec['kind'] == 'perennial stream').sum()} perennial, "
      f"{(rec['src'] == 'wb').sum()} waterbodies)")
print(f"wrote {OUTDIR / 'nhd_receptors.gpkg'}")
