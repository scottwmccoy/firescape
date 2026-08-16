"""Re-score all three Dolan pairs with label-free local corridor snapping.

For each pair (Planet 3 m; WV-2 2 m; GE-1 x WV-2 2 m): corridor AUCs raw,
then with the block-wise network-to-imagery offset field estimated by
cross-correlating the corridor mask against |z| (no truth labels touched).
Prints the offset-field statistics so the spatial structure of the
misalignment is on the record.
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pyogrio
import rasterio
from rasterio.merge import merge
from rasterio.warp import Resampling, reproject, transform_bounds
from scipy.stats import mannwhitneyu
from shapely.geometry import box

from firescape import change as ch, corridor as co, epochs, sources

BOX4326 = (-121.50, 36.00, -121.32, 36.18)
DOLAN = Path("/Users/scottmccoy/Library/CloudStorage/Box-Box/SWMresearch/"
             "PostFireDebrisFlows/2020_Dolan")
WV = DOLAN / "WorldviewImagery"
INV = DOLAN / "confidence_map/Dolan_ROC_BasinOverride.shp"
MIN_PIX = 12
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()


def auc(hi, lo):
    hi, lo = hi.dropna(), lo.dropna()
    if not len(hi) or not len(lo):
        return float("nan")
    return float(mannwhitneyu(hi, lo, alternative="greater").statistic
                 / (len(hi) * len(lo)))


def dem_pieces(ref):
    h, w = ref["height"], ref["width"]
    b4326 = transform_bounds(ref["crs"], "EPSG:4326",
                             *rasterio.transform.array_bounds(
                                 h, w, ref["transform"]))
    want = (b4326[0]-.02, b4326[1]-.02, b4326[2]+.02, b4326[3]+.02)
    srcs = [rasterio.open(t) for t in sorted(DOLAN.glob("USGS_13_*.tif"))]
    dem, tr_nat = merge(srcs, bounds=want, nodata=np.nan)
    crs_nat = srcs[0].crs
    [s.close() for s in srcs]
    dem = dem[0].astype(np.float32)
    lat = (want[1]+want[3])/2
    gy, gx = np.gradient(dem, abs(tr_nat.e)*111132.0,
                         abs(tr_nat.a)*111320.0*np.cos(np.radians(lat)))
    slope, aspect = np.arctan(np.hypot(gx, gy)), np.arctan2(-gx, gy)

    def cosi(az, el):
        zz, aa = np.radians(90-el), np.radians(az)
        return (np.cos(zz)*np.cos(slope) + np.sin(zz)*np.sin(slope)
                * np.cos(aa-aspect)).astype(np.float32)

    def togrid(arr):
        dst = np.full((h, w), np.nan, np.float32)
        reproject(arr, dst, src_transform=tr_nat, src_crs=crs_nat,
                  dst_transform=ref["transform"], dst_crs=ref["crs"],
                  resampling=Resampling.bilinear, src_nodata=np.nan,
                  dst_nodata=np.nan)
        return dst
    return cosi, togrid


def maxar_z(pre_dirs, post_dirs, ref):
    pre_src = sources.MaxarStrips(pre_dirs)
    post_src = sources.MaxarStrips(post_dirs)
    az0, el0 = pre_src.sun(); az1, el1 = post_src.sun()
    pre = pre_src.read_into(ref); post = post_src.read_into(ref)
    h, w = ref["height"], ref["width"]
    both0 = ~(np.ma.getmaskarray(pre["nir"]) | np.ma.getmaskarray(post["nir"]))
    ys, xs = np.nonzero(both0[::8, ::8])
    cy, cx = int(np.median(ys))*8, int(np.median(xs))*8
    sl = (slice(max(cy-1024, 0), cy+1024), slice(max(cx-1024, 0), cx+1024))
    dy, dx = ch.scene_offset(pre["nir"][sl], post["nir"][sl])
    from scipy.ndimage import shift as ndshift
    if max(abs(dy), abs(dx)) > 0.6:
        for k in post:
            post[k] = np.ma.masked_invalid(
                ndshift(post[k].filled(np.nan), (-dy, -dx), order=1,
                        mode="constant", cval=np.nan))
    both = ~(np.ma.getmaskarray(pre["nir"]) | np.ma.getmaskarray(post["nir"]))

    def nb(b):
        v = ch.brightness(b["blue"], b["green"], b["red"], b["nir"])
        return v / np.ma.median(v)

    d = nb(post) - nb(pre)
    cosi, togrid = dem_pieces(ref)
    ci0, ci1 = togrid(cosi(az0, el0)), togrid(cosi(az1, el1))
    dci = ci1 - ci0
    sh = (ci0 < 0.08) | (ci1 < 0.08)
    m = ~np.ma.getmaskarray(d) & np.isfinite(dci) & ~sh & both
    x, y = dci[m], np.ma.filled(d, np.nan)[m]
    sub = np.random.default_rng(1).choice(len(x), min(2_000_000, len(x)), False)
    x, y = x[sub], y[sub]
    keep = np.abs(y - np.median(y)) < 8*np.std(y)
    for _ in range(3):
        b, a = np.polyfit(x[keep], y[keep], 1)
        keep = np.abs(y - (a + b*x)) < 2.5*np.std(y[keep] - (a + b*x[keep]))
    z = np.ma.masked_where(sh | ~np.isfinite(dci) | ~both, d - (a + b*dci))
    z = z - np.ma.median(z)
    return z / (1.4826 * np.ma.median(np.ma.abs(z)))


def planet_z(ref):
    d = np.load(OUT / "dolan_z_cache.npz")
    return np.ma.masked_invalid(d["zb"].astype(np.float32))


def run_pair(name, zfun, res, block):
    ref = epochs.grid(BOX4326, "EPSG:32610", res)
    z = zfun(ref)
    seg = pyogrio.read_dataframe(INV, columns=["UpArea_km2", "Confidence"]) \
        .to_crs(ref["crs"])
    gbox = box(*transform_bounds("EPSG:4326", ref["crs"], *BOX4326))
    seg = seg[seg.intersects(gbox)].reset_index(drop=True)
    labels = co.corridor_raster(seg, ref, area_col="UpArea_km2")
    t_all = seg["Confidence"].replace({2: 3})

    def scored(lab):
        st = co.segment_stats(lab, {"z": z}).reindex(range(len(seg)))
        ok = st["n_pixels"].fillna(0) >= MIN_PIX
        v, t = st.loc[ok, "z_mean"], t_all[ok]
        return (round(auc(v[t > 0], v[t == 0]), 3),
                round(auc(v[t == 3], v[t == 1]), 3), int(ok.sum()))

    raw = scored(labels)
    dyf, dxf = co.local_offsets(labels, z, block=block)
    snapped = scored(co.apply_offsets(labels, dyf, dxf, block=block))
    off = {"median_dy_m": float(np.median(dyf)) * res,
           "median_dx_m": float(np.median(dxf)) * res,
           "spread_dy_px": [int(dyf.min()), int(dyf.max())],
           "spread_dx_px": [int(dxf.min()), int(dxf.max())]}
    print(f"{name:14s} raw resp={raw[0]:.3f} DF|fl={raw[1]:.3f} n={raw[2]}"
          f"  ->  snapped resp={snapped[0]:.3f} DF|fl={snapped[1]:.3f} "
          f"n={snapped[2]}  offsets(px): dy{off['spread_dy_px']} "
          f"dx{off['spread_dx_px']}", flush=True)
    return {"raw": raw, "snapped": snapped, "offsets": off}


if __name__ == "__main__":
    res = {}
    res["planet_3m"] = run_pair("planet 3m", planet_z, 3.0, 500)
    res["wv2_2m"] = run_pair("WV2 2m", lambda r: maxar_z(
        WV / "2020_11_29", [WV / "2021_04_19", WV / "2021_05_08"], r), 2.0, 750)
    res["ge1xwv2_2m"] = run_pair("GE1xWV2 2m", lambda r: maxar_z(
        [WV / "2020_11_16", WV / "2020_10_22"],
        [WV / "2021_04_19", WV / "2021_05_08"], r), 2.0, 750)
    (OUT / "dolan_snap_summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
