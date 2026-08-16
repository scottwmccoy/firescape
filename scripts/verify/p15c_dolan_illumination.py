"""Dolan: remove the illumination artifact, then re-score.

p15b showed the z field organized by hillslope aspect, not channels: the
sun climbed ~6-8 deg between the January pre epoch and the February post
epoch, and on 30-60 deg Big Sur slopes that alone moves brightness by
whole sigmas. Model: per-epoch mean solar incidence on the DEM,
cos(i)_post - cos(i)_pre as the illumination-change regressor, robust
linear fit of z on it, residual re-scored through the same corridor
machinery. Deep self-shadow (cos i < 0.08 in either epoch) is masked --
cast shadows (horizon occlusion) are NOT modelled; that is the remaining
gap if this correction under-delivers.

Run:  /opt/anaconda3/envs/FireMan/bin/python p15c_dolan_illumination.py [outdir]
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
from rasterio.warp import Resampling, reproject, transform_bounds
from scipy.stats import mannwhitneyu
from shapely.geometry import box

from firescape import corridor as co, epochs, paths

BOX4326 = (-121.50, 36.00, -121.32, 36.18)
DOLAN = Path("/Users/scottmccoy/Library/CloudStorage/Box-Box/SWMresearch/"
             "PostFireDebrisFlows/2020_Dolan")
INV = DOLAN / "confidence_map/Dolan_ROC_BasinOverride.shp"
TILES = sorted(DOLAN.glob("USGS_13_*.tif"))
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
MIN_PIX = 12

ref = epochs.grid(BOX4326, "EPSG:32610", 3.0)
d = np.load(OUT / "dolan_z_cache.npz")
zb = np.ma.masked_invalid(d["zb"].astype(np.float32))

# ---- per-epoch mean sun geometry -------------------------------------------
def epoch_sun(tag):
    az, el = [], []
    for f in paths.raw_dir("planet", "dolan", tag).glob("*_metadata.json"):
        p = json.loads(f.read_text())["properties"]
        az.append(p["sun_azimuth"]); el.append(p["sun_elevation"])
    return float(np.mean(az)), float(np.mean(el)), len(az)

az0, el0, n0 = epoch_sun("pre_20210127")
az1, el1, n1 = epoch_sun("post_20210127")
print(f"pre sun: az {az0:.1f}, el {el0:.1f} ({n0} scenes); "
      f"post sun: az {az1:.1f}, el {el1:.1f} ({n1} scenes)", flush=True)

# ---- cos(incidence) on the native DEM grid, warped once --------------------
import rasterio
from rasterio.merge import merge

b4326 = transform_bounds(ref["crs"], "EPSG:4326",
                         *rasterio.transform.array_bounds(
                             ref["height"], ref["width"], ref["transform"]))
pad = 0.02
want = (b4326[0] - pad, b4326[1] - pad, b4326[2] + pad, b4326[3] + pad)
srcs = [rasterio.open(t) for t in TILES]
dem_nat, tr_nat = merge(srcs, bounds=want, nodata=np.nan)
crs_nat = srcs[0].crs
for s in srcs:
    s.close()
dem_nat = dem_nat[0].astype(np.float32)
lat = (want[1] + want[3]) / 2
dy_m, dx_m = abs(tr_nat.e) * 111_132.0, \
    abs(tr_nat.a) * 111_320.0 * np.cos(np.radians(lat))
gy, gx = np.gradient(dem_nat, dy_m, dx_m)
slope = np.arctan(np.hypot(gx, gy))
aspect = np.arctan2(-gx, gy)          # 0 = north, clockwise positive

def cos_i(az_deg, el_deg):
    z = np.radians(90.0 - el_deg)
    a = np.radians(az_deg)
    return (np.cos(z) * np.cos(slope)
            + np.sin(z) * np.sin(slope) * np.cos(a - aspect)).astype(np.float32)

def to_grid(arr):
    dst = np.full((ref["height"], ref["width"]), np.nan, np.float32)
    reproject(arr, dst, src_transform=tr_nat, src_crs=crs_nat,
              dst_transform=ref["transform"], dst_crs=ref["crs"],
              resampling=Resampling.bilinear,
              src_nodata=np.nan, dst_nodata=np.nan)
    return dst

ci0, ci1 = to_grid(cos_i(az0, el0)), to_grid(cos_i(az1, el1))
dci = ci1 - ci0
shadowed = (ci0 < 0.08) | (ci1 < 0.08)
print(f"self-shadow masked: {np.nanmean(shadowed):.1%} of window", flush=True)

# ---- robust regression z ~ dcos(i), sigma-clipped OLS ----------------------
m = ~np.ma.getmaskarray(zb) & np.isfinite(dci) & ~shadowed
x, y = dci[m], np.ma.filled(zb, np.nan)[m]
sub = np.random.default_rng(0).choice(len(x), size=min(2_000_000, len(x)),
                                      replace=False)
x, y = x[sub], y[sub]
keep = np.abs(y) < 10
for _ in range(3):
    b, a = np.polyfit(x[keep], y[keep], 1)
    r = y - (a + b * x)
    s = np.std(r[keep])
    keep = np.abs(r) < 2.5 * s
ss_res = np.var(y[keep] - (a + b * x[keep]))
ss_tot = np.var(y[keep])
r2 = 1 - ss_res / ss_tot
print(f"illumination fit: z = {a:.2f} + {b:.2f}*dcos(i), "
      f"R^2 = {r2:.2f} (sigma-clipped)", flush=True)

z_corr = zb - (a + b * dci)
z_corr = np.ma.masked_where(shadowed | ~np.isfinite(dci), z_corr)
z_corr = z_corr - np.ma.median(z_corr)

# ---- re-score --------------------------------------------------------------
seg = pyogrio.read_dataframe(INV, columns=["Segment_ID", "UpArea_km2",
                                           "Confidence"]).to_crs(ref["crs"])
gbox = box(*transform_bounds("EPSG:4326", ref["crs"], *BOX4326))
seg = seg[seg.intersects(gbox)].reset_index(drop=True)
labels = co.corridor_raster(seg, ref, area_col="UpArea_km2")
stats = co.segment_stats(labels, {"z_corr": z_corr}).reindex(range(len(seg)))
seg = seg.join(stats)
ok = seg["n_pixels"].fillna(0) >= MIN_PIX
dd = seg[ok]
truth3 = dd["Confidence"].replace({2: 3})
print(f"{int(ok.sum()):,} segments scored after shadow mask", flush=True)

def auc(hi, lo):
    hi, lo = hi.dropna(), lo.dropna()
    if not len(hi) or not len(lo):
        return float("nan")
    return float(mannwhitneyu(hi, lo, alternative="greater").statistic
                 / (len(hi) * len(lo)))

res = {}
for col in ("z_corr_mean", "z_corr_p90", "z_corr_frac"):
    res[col] = {
        "auc_responded": round(auc(dd.loc[truth3 > 0, col],
                                   dd.loc[truth3 == 0, col]), 3),
        "auc_df_vs_fluvial": round(auc(dd.loc[truth3 == 3, col],
                                       dd.loc[truth3 == 1, col]), 3)}
    print(f"{col:14s} responded AUC={res[col]['auc_responded']:.3f}  "
          f"DF|fluvial AUC={res[col]['auc_df_vs_fluvial']:.3f}", flush=True)

# ---- before/after figure ---------------------------------------------------
tr = ref["transform"]
extent = (tr.c, tr.c + tr.a * ref["width"],
          tr.f + tr.e * ref["height"], tr.f)
truth_df = seg[seg["Confidence"].isin([2, 3])]
truth_fl = seg[seg["Confidence"] == 1]
fig, axes = plt.subplots(1, 2, figsize=(17, 10), dpi=130)
for ax, fld, title in ((axes[0], zb, "z(ΔBrightness), uncorrected"),
                       (axes[1], z_corr,
                        f"illumination-corrected (R²={r2:.2f} removed), "
                        "self-shadow masked")):
    im = ax.imshow(fld, cmap="RdBu_r", vmin=-6, vmax=6, extent=extent)
    truth_fl.plot(ax=ax, color="#F5D000", linewidth=0.4, alpha=0.7)
    truth_df.plot(ax=ax, color="#C1272D", linewidth=0.8, alpha=0.9)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_axis_off(); ax.set_title(title, fontsize=11)
fig.colorbar(im, ax=axes, shrink=0.5)
fig.suptitle("Dolan — illumination correction (truth: red DF, yellow fluvial)",
             fontsize=13)
fig.savefig(OUT / "dolan_illumination_correction.png", bbox_inches="tight")
plt.close(fig)

summary = {
    "sun": {"pre": [az0, el0], "post": [az1, el1]},
    "fit": {"a": round(float(a), 3), "b": round(float(b), 3),
            "r2": round(float(r2), 3)},
    "shadow_masked_frac": round(float(np.nanmean(shadowed)), 4),
    "segments_scored": int(ok.sum()),
    "auc": res,
}
(OUT / "dolan_illumination_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
