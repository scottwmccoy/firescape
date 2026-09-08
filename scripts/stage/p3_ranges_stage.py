"""Stage the RANGES PGA predictor: NSHM-2023 PGA (2%/50yr, BC) smoothed to
a 25-km-radius mean on a 1-km EPSG:5070 statewide grid."""
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.signal import fftconvolve

from firescape import paths

SHP = (paths.research_root() / "Volume_debrisFlows" / "NSHM2023_SeismicHazard" / "US_PGA_2Pct50Yrs_BC_poly.shp")
RES = 1000.0
RADIUS_M = 25_000.0

hu = gpd.read_file(paths.interim_dir("statewide") / "nv_hu10.geojson").to_crs("EPSG:5070")
w, s, e, n = hu.total_bounds
w, s, e, n = w - 60_000, s - 60_000, e + 60_000, n + 60_000
W, H = int((e - w) / RES), int((n - s) / RES)
tr = from_origin(w, n, RES, RES)

pga = gpd.read_file(SHP)
pga["pga_g"] = (pga.low_cont.astype(float) + pga.high_cont.astype(float)) / 2.0
pga = pga.to_crs("EPSG:5070").cx[w:e, s:n]
pga["geometry"] = pga.geometry.make_valid()
print(f"{len(pga)} PGA contour polygons in domain, "
      f"{pga.pga_g.min():.2f}-{pga.pga_g.max():.2f} g")

arr = rasterize(((g, v) for g, v in zip(pga.geometry, pga.pga_g)),
                out_shape=(H, W), transform=tr, fill=np.nan, dtype="float32")
print(f"grid {W}x{H} @ 1 km; coverage {np.isfinite(arr).mean():.1%}")

r_px = int(RADIUS_M / RES)
yy, xx = np.mgrid[-r_px:r_px + 1, -r_px:r_px + 1]
disk = ((xx ** 2 + yy ** 2) <= r_px ** 2).astype("float64")
valid = np.isfinite(arr).astype("float64")
num = fftconvolve(np.nan_to_num(arr, nan=0.0), disk, mode="same")
den = fftconvolve(valid, disk, mode="same")
sm = np.where(den > 0, num / den, np.nan).astype("float32")

out = paths.interim_dir("statewide") / "pga25_g.tif"
with rasterio.open(out, "w", driver="GTiff", height=H, width=W, count=1,
                   dtype="float32", crs="EPSG:5070", transform=tr,
                   nodata=np.nan, compress="deflate") as ds:
    ds.write(sm, 1)
paths.write_provenance(out, url="local: Volume_debrisFlows/NSHM2023_SeismicHazard",
                       note=("USGS NSHM 2023 PGA 2% in 50 yr site class BC, "
                             "contour-polygon midpoints rasterized at 1 km and "
                             "averaged in a 25 km radius (RANGES 'G' term)"))
print("wrote", out, f"| smoothed range {np.nanmin(sm):.3f}-{np.nanmax(sm):.3f} g")
