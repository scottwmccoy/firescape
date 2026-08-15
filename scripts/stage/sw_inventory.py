"""Statewide inventory: NV boundary, full HU10 units intersecting NV, and the
3DEP 1-degree tile list those units require."""
import json
import math
import warnings
import zipfile

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from firescape import paths, wbd

OUT = paths.interim_dir("statewide")

# ---- Nevada boundary (TIGER cartographic 1:500k) ----------------------------
nv_path = paths.raw_dir("boundaries") / "nv_state.geojson"
if not nv_path.exists():
    url = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_500k.zip"
    z = paths.raw_dir("boundaries") / "cb_2023_us_state_500k.zip"
    paths.download(url, z)
    paths.write_provenance(z, url=url, note="Census cartographic state boundaries 1:500k")
    states = gpd.read_file(f"zip://{z}")
    nv = states[states["STUSPS"] == "NV"].to_crs("EPSG:4326")
    nv.to_file(nv_path, driver="GeoJSON")
nv = gpd.read_file(nv_path)
nv_geom = nv.union_all()
print(f"NV boundary loaded; bounds {[round(v, 2) for v in nv.total_bounds]}")

# ---- HU10 units intersecting NV, fetched in 2-degree tiles ------------------
hu_path = OUT / "nv_hu10.geojson"
if hu_path.exists():
    hu = gpd.read_file(hu_path)
    print(f"HU10 inventory already present: {len(hu)} units")
else:
    w0, s0, e0, n0 = nv.total_bounds
    frames = []
    for x in np.arange(math.floor(w0), math.ceil(e0), 2.0):
        for y in np.arange(math.floor(s0), math.ceil(n0), 2.0):
            try:
                t = wbd.units((x, y, x + 2, y + 2))
            except Exception as exc:
                print(f"  tile ({x},{y}) retrying once: {type(exc).__name__}")
                t = wbd.units((x, y, x + 2, y + 2))
            frames.append(t)
            print(f"  tile ({x:.0f},{y:.0f}): {len(t)} units", flush=True)
    allhu = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    allhu = allhu.drop_duplicates(subset="huc10")
    hu = allhu[allhu.intersects(nv_geom)].reset_index(drop=True)
    # fraction of each unit inside NV, for reporting/clipping decisions later
    a5070 = hu.to_crs("EPSG:5070")
    nv5070 = nv.to_crs("EPSG:5070").union_all()
    hu["nv_frac"] = (a5070.geometry.intersection(nv5070).area / a5070.geometry.area).to_numpy()
    hu["areasqkm"] = pd.to_numeric(hu["areasqkm"], errors="coerce")
    hu.to_file(hu_path, driver="GeoJSON")
print(f"HU10 units intersecting Nevada: {len(hu)}; "
      f"total {hu['areasqkm'].sum():,.0f} km2 "
      f"(inside-NV fraction mean {hu['nv_frac'].mean():.2f})")

# ---- 3DEP 1-degree tile list covering every unit ----------------------------
tiles = set()
for w, s, e, n in hu.geometry.bounds.itertuples(index=False):
    for lon in range(math.floor(w), math.ceil(e)):
        for lat in range(math.floor(s), math.ceil(n)):
            tiles.add((lat + 1, -lon))          # n = top edge, w = positive west
tile_list = sorted(tiles)
(OUT / "dem_tiles.json").write_text(json.dumps(
    [{"n": t[0], "w": t[1],
      "url": f"https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/"
             f"current/n{t[0]}w{t[1]:03d}/USGS_13_n{t[0]}w{t[1]:03d}.tif"}
     for t in tile_list], indent=1))
print(f"3DEP tiles required: {len(tile_list)} "
      f"(n{min(t[0] for t in tile_list)}-n{max(t[0] for t in tile_list)}, "
      f"w{min(t[1] for t in tile_list)}-w{max(t[1] for t in tile_list)})")
