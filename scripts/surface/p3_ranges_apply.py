"""Per-basin RANGES topo predictors statewide: frac_north + mean slope (deg).

Zonal over each unit's 10 m DEM window (native RANGES convention). Writes one
parquet part per HU10 to interim/statewide/ranges_parts/ (resumable, exit 42);
p3_ranges_final.py assembles the volume columns.
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd
from pyogrio import read_dataframe
from rasterio.features import rasterize

from firescape import paths, statewide

T0 = time.time()
BUDGET = float(os.environ.get("FIRESCAPE_RANGES_BUDGET", 520))
OUT = paths.products_dir("prefire", "statewide_v1")
PARTS = paths.interim_dir("statewide") / "ranges_parts"
PARTS.mkdir(exist_ok=True)
TILE_DIR = paths.cache_root() / "3dep_tiles"

basins = read_dataframe(OUT / "statewide_v1_basins.gpkg",
                        columns=["huc10", "Segment_ID"])
print(f"{len(basins):,} basins loaded", flush=True)
units = sorted(basins["huc10"].unique())

done, pending = [], []
for huc in units:
    part = PARTS / f"{huc}.parquet"
    if part.exists():
        done.append(huc)
        continue
    if time.time() - T0 > BUDGET:
        pending.append(huc)
        continue
    t1 = time.time()
    sub = basins[basins["huc10"] == huc]
    g5070 = sub.to_crs("EPSG:5070")
    b = g5070.total_bounds
    try:
        dem = statewide.dem_for_unit((b[0] - 300, b[1] - 300,
                                      b[2] + 300, b[3] + 300), TILE_DIR)
        z = np.asarray(dem.values, dtype="float64")
        gr, gc = np.gradient(z, 10.0)
        slope_deg = np.degrees(np.arctan(np.hypot(gr, gc)))
        aspect = np.degrees(np.arctan2(-gc, gr)) % 360.0
        north = (aspect >= 315.0) | (aspect < 45.0)
        idx = rasterize(((geom, i + 1) for i, geom in
                         enumerate(g5070.geometry)),
                        out_shape=z.shape, transform=dem.transform.affine
                        if hasattr(dem.transform, "affine") else dem.transform,
                        fill=0, dtype="int32")
        flat = idx.ravel()
        ok = flat > 0
        nb = len(sub)
        cnt = np.bincount(flat[ok], minlength=nb + 1)[1:]
        vals_ok = np.isfinite(slope_deg.ravel())
        sel = ok & vals_ok
        s_sum = np.bincount(flat[sel], weights=slope_deg.ravel()[sel],
                            minlength=nb + 1)[1:]
        n_sum = np.bincount(flat[sel], weights=north.ravel()[sel].astype(float),
                            minlength=nb + 1)[1:]
        cnt2 = np.bincount(flat[sel], minlength=nb + 1)[1:]
        with np.errstate(all="ignore"):
            slope_b = np.where(cnt2 > 0, s_sum / cnt2, np.nan)
            north_b = np.where(cnt2 > 0, n_sum / cnt2, np.nan)
        pd.DataFrame({"huc10": huc,
                      "Segment_ID": sub["Segment_ID"].to_numpy(),
                      "SlopeDeg": slope_b.astype("float32"),
                      "FracNorth": north_b.astype("float32"),
                      "n_px": cnt.astype("int32")}).to_parquet(part, index=False)
        print(f"{huc}: {nb:5d} basins in {time.time()-t1:.0f}s", flush=True)
        done.append(huc)
    except Exception as e:
        print(f"{huc} FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)
        pending.append(huc)

print(f"\nranges zonal: {len(done)} done, {len(pending)} pending "
      f"({time.time()-T0:.0f}s)", flush=True)
sys.exit(42 if pending else 0)
