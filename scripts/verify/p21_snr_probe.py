"""Per-block snap-SNR distributions on WV-Dolan and Hidden Valley.

No gate, wider search (+/-12 px), raw peak table -- the data that sets
min_snr instead of a synthetic. Prints, per site: SNR quantiles, and the
(snr, dy, dx) of every block, accepted or not.
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
from firescape import change as ch, corridor as co, epochs, paths, sources

S = Path(sys.argv[1])
sys.path.insert(0, str(Path(__file__).parent))
from p20_snap_rescore import maxar_z, BOX4326, WV, INV  # reuse builders
import pyogrio
from rasterio.warp import transform_bounds
from shapely.geometry import box


def probe(name, z, labels, block, res):
    dy, dx, info = co.local_offsets(labels, z, block=block, max_off=12,
                                    min_snr=0.0, smooth=False,
                                    with_info=True)
    s = info["snr"][np.isfinite(info["snr"])]
    print(f"--- {name}: {s.size} estimable blocks; SNR quantiles "
          f"p25={np.percentile(s,25):.1f} p50={np.percentile(s,50):.1f} "
          f"p75={np.percentile(s,75):.1f} p90={np.percentile(s,90):.1f}")
    ys, xs = np.nonzero(np.isfinite(info["snr"]))
    rows = sorted(zip(info["snr"][ys, xs], info["raw_dy"][ys, xs],
                      info["raw_dx"][ys, xs]), reverse=True)
    for snr, ddy, ddx in rows[:12]:
        print(f"    snr={snr:5.1f}  dy={ddy:+3d} dx={ddx:+3d} px "
              f"({ddy*res:+.0f},{ddx*res:+.0f} m)")

# WV Dolan
ref = epochs.grid(BOX4326, "EPSG:32610", 2.0)
z = maxar_z(WV / "2020_11_29", [WV / "2021_04_19", WV / "2021_05_08"], ref)
seg = pyogrio.read_dataframe(INV, columns=["UpArea_km2", "Confidence"]).to_crs(ref["crs"])
seg = seg[seg.intersects(box(*transform_bounds("EPSG:4326", ref["crs"], *BOX4326)))].reset_index(drop=True)
labels = co.corridor_raster(seg, ref, area_col="UpArea_km2")
probe("WV2 Dolan", z, labels, 750, 2.0)

# Hidden Valley
import geopandas as gpd, pandas as pd
AOI = (-119.72, 39.45, -119.63, 39.53)
pre, _, _, _, hvref = epochs.build(paths.raw_dir("planet", "hidden_valley", "pre_20260619"),
                                   indices=("brightness",), rgb=False, keep_nir=False, verbose=False)
post, _, _, _, _ = epochs.build(paths.raw_dir("planet", "hidden_valley", "post_20260619"), hvref,
                                indices=("brightness",), rgb=False, keep_nir=False, verbose=False)
zhv = ch.robust_z(pre["brightness"][0], post["brightness"][0], pre["brightness"][1])
zhv = zhv - np.ma.median(zhv)
prod = paths.products_dir("prefire", "statewide_v1_2")
bbox5070 = transform_bounds("EPSG:4326", "EPSG:5070", *AOI)
segs = gpd.GeoDataFrame(pd.concat(
    [pyogrio.read_dataframe(prod / f"{h}_segments.gpkg", bbox=bbox5070, columns=["Area_km2"])
     for h in ["1605010203", "1605010205", "1605010206"]], ignore_index=True),
    geometry="geometry", crs="EPSG:5070").to_crs(hvref["crs"])
hvlab = co.corridor_raster(segs, hvref)
probe("Hidden Valley", zhv, hvlab, 500, 3.0)
