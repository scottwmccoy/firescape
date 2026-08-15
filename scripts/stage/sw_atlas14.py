"""Statewide Atlas 14 grids + the sw/inw seam check for northeast Nevada."""
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np

from firescape import annualprob, paths

hu = gpd.read_file(paths.interim_dir("statewide") / "nv_hu10.geojson")
w, s, e, n = hu.total_bounds
bbox = (float(w) - 0.05, float(s) - 0.05, float(e) + 0.05, float(n) + 0.05)
print(f"statewide bounds: {[round(v, 2) for v in bbox]}")

grids = {}
for ari in (1, 50):
    arr, res = annualprob.climatology_i15(bbox, ari=ari, region="sw")
    grids[ari] = (arr, res)
    nodata_frac = float(np.mean(~np.isfinite(arr)))
    print(f"sw ARI {ari:>2}: shape {arr.shape}, "
          f"range {np.nanmin(arr):.1f}-{np.nanmax(arr):.1f} mm/h, "
          f"NODATA {nodata_frac:.1%}")

# seam check: NODATA inside the northeastern units?
ne = hu[(hu.geometry.centroid.x > -116.5) & (hu.geometry.centroid.y > 40.5)]
arr, res = grids[1]
tr = res["transform"]
cent = ne.geometry.centroid
cols = ((cent.x - tr.c) / tr.a).astype(int).to_numpy()
rows = ((cent.y - tr.f) / tr.e).astype(int).to_numpy()
inb = (rows >= 0) & (rows < arr.shape[0]) & (cols >= 0) & (cols < arr.shape[1])
vals = np.full(len(ne), np.nan)
vals[inb] = arr[rows[inb], cols[inb]]
n_bad = int((~np.isfinite(vals)).sum())
print(f"NE-Nevada units ({len(ne)}): {n_bad} with NODATA 1-yr I15 "
      f"{'-> need inw mosaic' if n_bad else '-> sw covers the seam, no inw needed'}")

np.savez(paths.interim_dir("statewide") / "atlas14_i15.npz",
         i1=grids[1][0], i50=grids[50][0],
         transform=np.array(grids[1][1]["transform"]).reshape(-1)[:6])
print("saved statewide Atlas 14 grids")
