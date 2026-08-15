"""Phase-2 staging: era-matched EVT chunks under calibration fires + DEM gaps.

Only LFPS 2-degree chunks intersecting each vintage's fires are fetched —
calibration/refit need pre-fire vegetation under fire footprints, not
statewide (the v1 surface keeps using the already-staged LF2025).
Resumable: exit 42 = more to do, relaunch.
"""
import math
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import requests
from shapely.geometry import box

from firescape import paths

T0 = time.time()
BUDGET = float(os.environ.get("FIRESCAPE_STAGE_BUDGET", 520))
# LFPS EVT catalog: LF2016/LF2022/LF2023/LF2024/LF2025 only (no LF2020).
VINTAGES = ("LF2016_EVT", "LF2022_EVT", "LF2023_EVT")

fires = gpd.read_file(paths.interim_dir("statewide") / "calib_fires_v1.geojson")
fires = fires[fires["include"]]
hu = gpd.read_file(paths.interim_dir("statewide") / "nv_hu10.geojson")
w0, s0, e0, n0 = hu.total_bounds


def out_of_time():
    return time.time() - T0 > BUDGET


# ---- DEM gap fill: 1-degree 3DEP tiles under fire footprints ---------------
TILE_DIR = paths.cache_root() / "3dep_tiles"
need_tiles = set()
for geom in fires.geometry.buffer(0.03):     # ~3 km margin for basin capture
    fw, fs, fe, fn = geom.bounds
    for x in range(math.floor(fw), math.ceil(fe)):
        for y in range(math.floor(fs), math.ceil(fn)):
            need_tiles.add((y + 1, -x))      # 3DEP tiles named by NW corner
dem_pending = []
for n, w in sorted(need_tiles):
    name = f"USGS_13_n{n}w{w:03d}.tif"
    dest = TILE_DIR / name
    if dest.exists() and dest.stat().st_size > 1e6:
        continue
    if out_of_time():
        dem_pending.append(name)
        continue
    url = (f"https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/"
           f"TIFF/current/n{n}w{w:03d}/USGS_13_n{n}w{w:03d}.tif")
    t1 = time.time()
    try:
        paths.download(url, dest, timeout=900)
        print(f"DEM {name}: {dest.stat().st_size/1e6:.0f} MB in "
              f"{time.time()-t1:.0f}s", flush=True)
    except requests.HTTPError as e:
        print(f"DEM {name}: HTTP {getattr(e.response, 'status_code', '?')} "
              "(no such tile?)", flush=True)
    except Exception as e:
        print(f"DEM {name}: {type(e).__name__}", flush=True)
        dem_pending.append(name)
print(f"DEM tiles: {len(need_tiles)} needed under fires, "
      f"{len(dem_pending)} pending", flush=True)

# ---- era-matched EVT chunks -------------------------------------------------
from pfdf.data import landfire as lf
from pfdf.projection import BoundingBox

evt_pending = []
for vintage in VINTAGES:
    sub = fires[fires["evt_vintage"] == vintage]
    if not len(sub):
        continue
    OUT = paths.raw_dir("landfire", f"{vintage.split('_')[0]}_EVT_nv")
    hull = sub.buffer(0.02)
    chunks = []
    for x in np.arange(math.floor(w0), math.ceil(e0), 2.0):
        for y in np.arange(math.floor(s0), math.ceil(n0), 2.0):
            if hull.intersects(box(x, y, x + 2, y + 2)).any():
                chunks.append((x, y))
    print(f"{vintage}: {len(sub)} fires -> {len(chunks)} chunks", flush=True)
    for x, y in chunks:
        name = f"evt_{int(-x):03d}w_{int(y):02d}n"
        if (OUT / name / f"{name}.tif").exists():
            continue
        if out_of_time():
            evt_pending.append(f"{vintage}/{name}")
            continue
        t1 = time.time()
        try:
            lf.download(vintage, BoundingBox(x, y, x + 2, y + 2, crs=4326),
                        os.environ["FIRESCAPE_LFPS_EMAIL"], parent=OUT,
                        name=name, max_job_time=900)
            print(f"{vintage}/{name}: ok in {time.time()-t1:.0f}s", flush=True)
        except Exception as e:
            print(f"{vintage}/{name}: {type(e).__name__}: {str(e)[:120]}",
                  flush=True)
            evt_pending.append(f"{vintage}/{name}")

pending = dem_pending + evt_pending
print(f"\nstaging pass done in {time.time()-T0:.0f}s; pending: {len(pending)}",
      flush=True)
sys.exit(42 if pending else 0)
