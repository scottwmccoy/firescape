"""Dolan, second epoch pair: GeoEye-1 pre x WV-2 post (coverage widener).

CROSS-SENSOR pair: differencing per-epoch-normalized indices (unitless)
rather than radiance; band-pass differences between GE-1 and WV-2 remain
a disclosed first-order residual. Purpose: score the inland segments the
Nov-29 WV-2 pre strips never covered (p17 joint coverage was 5.1%).

Same protocol as p15/p15c (their network, their classes, corridor stats,
AUC), different source: WV-2 M2AS strips, pre 2020-11-29 (sun 32 deg) vs
post 2021-04-19 (sun 63 deg), radiance via IMD calibration, one composite
per epoch (spatial mosaic -- no temporal stack, so z uses a spatial robust
scale after the DEM illumination regression). Global registration is
checked and, if needed, corrected with a single shift; terrain-dependent
ortho parallax is beyond v0 and gets reported, not fixed.

Run:  /opt/anaconda3/envs/FireMan/bin/python p17_dolan_worldview.py [outdir]
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

ref = epochs.grid(BOX4326, "EPSG:32610", 2.0)
print(f"grid {ref['width']}x{ref['height']} @2 m", flush=True)

pre_src = sources.MaxarStrips([WV / "2020_11_16", WV / "2020_10_22"])  # GE01 4-band
post_src = sources.MaxarStrips([WV / "2021_04_19", WV / "2021_05_08"])  # WV02
print(f"pre: {len(pre_src.tifs)} strips, sun {pre_src.sun()}", flush=True)
print(f"post: {len(post_src.tifs)} strips, sun {post_src.sun()}", flush=True)
pre = pre_src.read_into(ref)
print("pre mosaicked", flush=True)
post = post_src.read_into(ref)
print("post mosaicked", flush=True)

# ---- global registration check/fix -----------------------------------------
h, w = ref["height"], ref["width"]
both0 = ~(np.ma.getmaskarray(pre["nir"]) | np.ma.getmaskarray(post["nir"]))
ys, xs = np.nonzero(both0[::8, ::8])
cy, cx = int(np.median(ys)) * 8, int(np.median(xs)) * 8   # joint-coverage centre
sl = (slice(max(cy - 1024, 0), cy + 1024), slice(max(cx - 1024, 0), cx + 1024))
dy, dx = ch.scene_offset(pre["nir"][sl], post["nir"][sl])
print(f"registration pre->post: dy={dy:.2f}, dx={dx:.2f} px", flush=True)
if max(abs(dy), abs(dx)) > 0.6:
    from scipy.ndimage import shift as ndshift
    for k in post:
        filled = post[k].filled(np.nan)
        post[k] = np.ma.masked_invalid(
            ndshift(filled, (-dy, -dx), order=1, mode="constant",
                    cval=np.nan))
    print("applied global shift correction", flush=True)

both = ~(np.ma.getmaskarray(pre["nir"]) | np.ma.getmaskarray(post["nir"]))
print(f"joint coverage: {both.mean():.1%} of window", flush=True)

# ---- per-epoch indices, illumination regression, spatial z -----------------
def norm_bright(b):
    v = ch.brightness(b["blue"], b["green"], b["red"], b["nir"])
    return v / np.ma.median(v)          # radiance -> unitless per epoch

d_bright = norm_bright(post) - norm_bright(pre)
d_ndvi = ch.ndvi(post["nir"], post["red"]) - ch.ndvi(pre["nir"], pre["red"])

# DEM cos(i) per epoch (same machinery as p15c)
b4326 = transform_bounds(ref["crs"], "EPSG:4326",
                         *rasterio.transform.array_bounds(h, w,
                                                          ref["transform"]))
want = (b4326[0] - .02, b4326[1] - .02, b4326[2] + .02, b4326[3] + .02)
srcs = [rasterio.open(t) for t in sorted(DOLAN.glob("USGS_13_*.tif"))]
dem, tr_nat = merge(srcs, bounds=want, nodata=np.nan)
crs_nat = srcs[0].crs
[s.close() for s in srcs]
dem = dem[0].astype(np.float32)
lat = (want[1] + want[3]) / 2
gy, gx = np.gradient(dem, abs(tr_nat.e) * 111_132.0,
                     abs(tr_nat.a) * 111_320.0 * np.cos(np.radians(lat)))
slope, aspect = np.arctan(np.hypot(gx, gy)), np.arctan2(-gx, gy)

def cosi(az, el):
    z, a = np.radians(90 - el), np.radians(az)
    return (np.cos(z) * np.cos(slope)
            + np.sin(z) * np.sin(slope) * np.cos(a - aspect)).astype(np.float32)

def to_grid(a):
    dst = np.full((h, w), np.nan, np.float32)
    reproject(a, dst, src_transform=tr_nat, src_crs=crs_nat,
              dst_transform=ref["transform"], dst_crs=ref["crs"],
              resampling=Resampling.bilinear, src_nodata=np.nan,
              dst_nodata=np.nan)
    return dst

az0, el0 = pre_src.sun(); az1, el1 = post_src.sun()
ci0, ci1 = to_grid(cosi(az0, el0)), to_grid(cosi(az1, el1))
dci = ci1 - ci0
shadowed = (ci0 < 0.08) | (ci1 < 0.08)

def spatial_z(delta, name):
    m = ~np.ma.getmaskarray(delta) & np.isfinite(dci) & ~shadowed & both
    x = dci[m]; y = np.ma.filled(delta, np.nan)[m]
    sub = np.random.default_rng(1).choice(len(x), min(2_000_000, len(x)),
                                          replace=False)
    x, y = x[sub], y[sub]
    keep = np.abs(y - np.median(y)) < 8 * np.std(y)
    for _ in range(3):
        b, a = np.polyfit(x[keep], y[keep], 1)
        r = y - (a + b * x)
        keep = np.abs(r) < 2.5 * np.std(r[keep])
    r2 = 1 - np.var(y[keep] - (a + b * x[keep])) / np.var(y[keep])
    resid = delta - (a + b * dci)
    resid = np.ma.masked_where(shadowed | ~np.isfinite(dci) | ~both, resid)
    resid = resid - np.ma.median(resid)
    mad = np.ma.median(np.ma.abs(resid))
    print(f"{name}: illum fit R2={r2:.2f}; spatial MAD={float(mad):.4f}",
          flush=True)
    return resid / (1.4826 * mad)

z = {"z_brightness": spatial_z(d_bright, "brightness"),
     "z_ndvi": spatial_z(d_ndvi, "ndvi")}

# ---- score on their network -------------------------------------------------
seg = pyogrio.read_dataframe(INV, columns=["Segment_ID", "UpArea_km2",
                                           "Confidence"]).to_crs(ref["crs"])
gbox = box(*transform_bounds("EPSG:4326", ref["crs"], *BOX4326))
seg = seg[seg.intersects(gbox)].reset_index(drop=True)
labels = co.corridor_raster(seg, ref, area_col="UpArea_km2")
stats = co.segment_stats(labels, z).reindex(range(len(seg)))
seg = seg.join(stats)
ok = seg["n_pixels"].fillna(0) >= MIN_PIX
dd = seg[ok]
t3 = dd["Confidence"].replace({2: 3})
print(f"{int(ok.sum()):,} segments scored", flush=True)

def auc(hi, lo):
    hi, lo = hi.dropna(), lo.dropna()
    if not len(hi) or not len(lo):
        return float("nan")
    return float(mannwhitneyu(hi, lo, alternative="greater").statistic
                 / (len(hi) * len(lo)))

res = {}
for col in ("z_brightness_mean", "z_brightness_p90",
            "z_ndvi_mean", "z_ndvi_p90"):
    res[col] = {"auc_responded": round(auc(dd.loc[t3 > 0, col],
                                           dd.loc[t3 == 0, col]), 3),
                "auc_df_vs_fluvial": round(auc(dd.loc[t3 == 3, col],
                                               dd.loc[t3 == 1, col]), 3)}
    print(f"{col:20s} responded={res[col]['auc_responded']:.3f}  "
          f"DF|fluvial={res[col]['auc_df_vs_fluvial']:.3f}", flush=True)

# ---- Santa Lucia chip -------------------------------------------------------
import pyproj
t = pyproj.Transformer.from_crs("EPSG:4326", ref["crs"], always_xy=True)
x, y = t.transform(-121.415, 36.115)
tr = ref["transform"]
c, r = int((x - tr.c) / tr.a), int((y - tr.f) / tr.e)
H = 900
if not both[max(r - H, 0):r + H, max(c - H, 0):c + H].any():
    r, c = cy, cx                      # chip target outside joint coverage
    print("chip recentred to joint-coverage centroid", flush=True)
win = (slice(max(r - H, 0), r + H), slice(max(c - H, 0), c + H))
ext = (tr.c + tr.a * win[1].start, tr.c + tr.a * win[1].stop,
       tr.f + tr.e * win[0].stop, tr.f + tr.e * win[0].start)
truth_df = seg[seg["Confidence"].isin([2, 3])]
truth_fl = seg[seg["Confidence"] == 1]
fig, axes = plt.subplots(1, 3, figsize=(17, 6.5), dpi=140)
for ax, (arr, cmap, vlim, title) in zip(axes, [
        (norm_bright(pre)[win], "gray", None, "pre (WV-2 2020-11-29)"),
        (norm_bright(post)[win], "gray", None, "post (WV-2 2021-04-19)"),
        (z["z_brightness"][win], "RdBu_r", (-6, 6), "z(ΔBrightness), 2 m")]):
    if vlim:
        ax.imshow(arr, cmap=cmap, vmin=vlim[0], vmax=vlim[1], extent=ext)
    else:
        v = np.ma.compressed(arr)
        lo, hi = (np.percentile(v, [2, 98]) if v.size else (0, 1))
        ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi, extent=ext)
    truth_fl.plot(ax=ax, color="#F5D000", linewidth=0.5, alpha=0.8)
    truth_df.plot(ax=ax, color="#C1272D", linewidth=1.0, alpha=0.9)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_axis_off(); ax.set_title(title, fontsize=10)
fig.suptitle("Dolan on WorldView-2 — Santa Lucia / Rattlesnake "
             "(truth: red DF, yellow fluvial)", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "dolan_ge_chip.png", bbox_inches="tight")
plt.close(fig)

summary = {
    "pre": "2020-11-16+10-22 GE01 M2AS", "post": "2021-04-19+05-08 WV02 M2AS",
    "sun": {"pre": [az0, el0], "post": [az1, el1]},
    "registration_px": [round(dy, 2), round(dx, 2)],
    "joint_coverage": round(float(both.mean()), 3),
    "segments_scored": int(ok.sum()),
    "auc": res,
    "planet_baseline": {"responded": 0.574, "df_vs_fluvial": 0.531},
}
(OUT / "dolan_ge_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
