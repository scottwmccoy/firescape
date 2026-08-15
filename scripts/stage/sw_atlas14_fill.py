"""Mosaic Atlas 14 volumes (sw + inw + ca) onto the statewide grid."""
import warnings; warnings.filterwarnings("ignore")
import geopandas as gpd
import numpy as np
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject
from firescape import annualprob, paths

hu = gpd.read_file(paths.interim_dir("statewide") / "nv_hu10.geojson")
w, s, e, n = hu.total_bounds
bbox = (float(w) - 0.05, float(s) - 0.05, float(e) + 0.05, float(n) + 0.05)

base = {}
for ari in (1, 50):
    arr, res = annualprob.climatology_i15(bbox, ari=ari, region="sw")
    base[ari] = (arr.copy(), res)
tr = Affine(*np.array(base[1][1]["transform"]).reshape(-1)[:6].tolist()) \
    if not hasattr(base[1][1]["transform"], "a") else base[1][1]["transform"]

for region in ("inw", "ca"):
    for ari in (1, 50):
        try:
            arr_r, res_r = annualprob.climatology_i15(bbox, ari=ari, region=region)
        except Exception as e:
            print(f"{region} ARI{ari}: unavailable ({type(e).__name__})")
            continue
        dst, meta = base[ari]
        fill = np.full_like(dst, np.nan)
        reproject(arr_r, fill,
                  src_transform=res_r["transform"], src_crs=res_r["crs"],
                  src_nodata=np.nan,
                  dst_transform=meta["transform"], dst_crs=meta["crs"],
                  dst_nodata=np.nan, resampling=Resampling.bilinear)
        need = ~np.isfinite(dst) & np.isfinite(fill)
        dst[need] = fill[need]
        print(f"{region} ARI{ari}: filled {int(need.sum())} cells")

for ari in (1, 50):
    arr, _ = base[ari]
    print(f"final ARI{ari}: NODATA {float(np.mean(~np.isfinite(arr))):.1%}")

# per-unit centroid audit
cent = hu.geometry.centroid
arr, meta = base[1]
t = meta["transform"]
cols = ((cent.x - t.c) / t.a).astype(int).to_numpy()
rows = ((cent.y - t.f) / t.e).astype(int).to_numpy()
inb = (rows >= 0) & (rows < arr.shape[0]) & (cols >= 0) & (cols < arr.shape[1])
vals = np.full(len(hu), np.nan); vals[inb] = arr[rows[inb], cols[inb]]
bad = hu.loc[~np.isfinite(vals), ["huc10", "name"]]
print(f"units with NODATA centroid after mosaic: {len(bad)}")
if len(bad):
    print(bad.head(10).to_string(index=False))

np.savez(paths.interim_dir("statewide") / "atlas14_i15.npz",
         i1=base[1][0], i50=base[50][0],
         transform=np.array(base[1][1]["transform"]).reshape(-1)[:6])
print("saved mosaicked statewide Atlas 14 grids")
