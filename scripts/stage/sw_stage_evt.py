"""Stage LF2025 EVT over all statewide units via LFPS 2-degree chunks."""
import math
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from firescape import paths

T0 = time.time()
BUDGET = float(os.environ.get("FIRESCAPE_STAGE_BUDGET", 480))
OUT = paths.raw_dir("landfire", "LF2025_EVT_nv")

hu = gpd.read_file(paths.interim_dir("statewide") / "nv_hu10.geojson")
w0, s0, e0, n0 = hu.total_bounds
chunks = []
for x in np.arange(math.floor(w0), math.ceil(e0), 2.0):
    for y in np.arange(math.floor(s0), math.ceil(n0), 2.0):
        cell = box(x, y, x + 2, y + 2)
        if hu.intersects(cell).any():
            chunks.append((x, y))
print(f"{len(chunks)} LFPS chunks needed", flush=True)

from pfdf.data import landfire as lf
from pfdf.projection import BoundingBox

done, pending = [], []
for x, y in chunks:
    name = f"evt_{int(-x):03d}w_{int(y):02d}n"
    if (OUT / name / f"{name}.tif").exists():
        done.append(name)
        continue
    if time.time() - T0 > BUDGET:
        pending.append(name)
        continue
    t1 = time.time()
    try:
        lf.download("LF2025_EVT", BoundingBox(x, y, x + 2, y + 2, crs=4326),
                    landfire.delivery_email(), parent=OUT, name=name,
                    max_job_time=900)
        print(f"{name}: ok in {time.time()-t1:.0f}s", flush=True)
        done.append(name)
    except Exception as e:
        print(f"{name}: {type(e).__name__}: {str(e)[:100]}", flush=True)

if not pending and done:
    # VRT across chunk tifs for windowed statewide reads
    tifs = sorted(str(p) for p in OUT.glob("*/evt_*.tif"))
    try:
        from osgeo import gdal

        gdal.BuildVRT(str(OUT / "LF2025_EVT_nv.vrt"), tifs)
        print(f"VRT built over {len(tifs)} chunks")
    except Exception:
        import subprocess

        r = subprocess.run(["rio", "stack"], capture_output=True)
        print("osgeo gdal unavailable; VRT deferred to statewide module", flush=True)
print(f"\nEVT: {len(done)} done, {len(pending)} pending, {time.time()-T0:.0f}s", flush=True)
sys.exit(42 if pending else 0)
