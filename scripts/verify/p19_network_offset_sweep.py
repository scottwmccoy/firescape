"""Is the inventory network misaligned with the imagery? Measure it.

Scott's observation: truth centerlines plot slightly off the visible
channels. If the DEM-delineated network is shifted relative to the
orthoimagery, corridors sample off-channel and every AUC we computed is an
underestimate. Test: shift the z field on a +/-16 m grid (2 m steps in
imagery pixels), recompute corridor-mean z each time against the FIXED
network labels, and map responded-AUC over offset. A peak away from (0,0)
both confirms the misalignment and measures it.

Runs on the same-sensor WV pair (p17 config, illumination-corrected).
"""
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
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()

ref = epochs.grid(BOX4326, "EPSG:32610", 2.0)
pre = sources.MaxarStrips(WV / "2020_11_29")
post = sources.MaxarStrips([WV / "2021_04_19", WV / "2021_05_08"])
az0, el0 = pre.sun(); az1, el1 = post.sun()
pre = pre.read_into(ref); post = post.read_into(ref)
print("mosaicked", flush=True)

h, w = ref["height"], ref["width"]
both0 = ~(np.ma.getmaskarray(pre["nir"]) | np.ma.getmaskarray(post["nir"]))
ys, xs = np.nonzero(both0[::8, ::8])
cy, cx = int(np.median(ys)) * 8, int(np.median(xs)) * 8
sl = (slice(max(cy - 1024, 0), cy + 1024), slice(max(cx - 1024, 0), cx + 1024))
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

# illumination correction (p15c/p17 machinery, condensed)
b4326 = transform_bounds(ref["crs"], "EPSG:4326",
                         *rasterio.transform.array_bounds(h, w, ref["transform"]))
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
    z, a = np.radians(90-el), np.radians(az)
    return (np.cos(z)*np.cos(slope)
            + np.sin(z)*np.sin(slope)*np.cos(a-aspect)).astype(np.float32)

def togrid(a):
    dst = np.full((h, w), np.nan, np.float32)
    reproject(a, dst, src_transform=tr_nat, src_crs=crs_nat,
              dst_transform=ref["transform"], dst_crs=ref["crs"],
              resampling=Resampling.bilinear, src_nodata=np.nan,
              dst_nodata=np.nan)
    return dst

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
z = (z - np.ma.median(z)) / (1.4826 * np.ma.median(np.ma.abs(z - np.ma.median(z))))
zf = z.filled(np.nan)
print("z built", flush=True)

seg = pyogrio.read_dataframe(INV, columns=["UpArea_km2", "Confidence"]) \
    .to_crs(ref["crs"])
gbox = box(*transform_bounds("EPSG:4326", ref["crs"], *BOX4326))
seg = seg[seg.intersects(gbox)].reset_index(drop=True)
labels = co.corridor_raster(seg, ref, area_col="UpArea_km2")
t_all = seg["Confidence"].replace({2: 3})

def auc(hi, lo):
    hi, lo = hi.dropna(), lo.dropna()
    if not len(hi) or not len(lo):
        return float("nan")
    return float(mannwhitneyu(hi, lo, alternative="greater").statistic
                 / (len(hi)*len(lo)))

def score(off_y, off_x):
    zz = np.ma.masked_invalid(np.roll(zf, (off_y, off_x), axis=(0, 1)))
    st = co.segment_stats(labels, {"z": zz}).reindex(range(len(seg)))
    ok = st["n_pixels"].fillna(0) >= 12
    v, t = st.loc[ok, "z_mean"], t_all[ok]
    return (auc(v[t > 0], v[t == 0]), auc(v[t == 3], v[t == 1]),
            int(ok.sum()))

print("offset sweep (imagery px @2 m; +y = network effectively moves north):",
      flush=True)
best = (None, -1)
for oy in range(-8, 9, 2):
    row = []
    for ox in range(-8, 9, 2):
        r, s, n = score(oy, ox)
        row.append(f"{r:.3f}")
        if r > best[1]:
            best = ((oy, ox, s, n), r)
    print(f"  dy={oy:+d}: " + " ".join(row), flush=True)
(oy, ox, s_best, n) = best[0]
print(f"\nPEAK responded AUC {best[1]:.3f} at (dy={oy}, dx={ox}) px "
      f"= ({oy*2}, {ox*2}) m; DF|fluvial there {s_best:.3f}; n={n}")
print(f"centre (0,0) for reference: {score(0, 0)[0]:.3f}")
