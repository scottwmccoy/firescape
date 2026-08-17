"""Robustified shadow-alignment probe for Dolan.

Many candidate windows ranked by relief content; per-window phase
correlation gated on (a) |shift| <= 30 px sanity bound, (b) skimage
lock error, (c) valid fraction. Reported: per-window table and the
median +/- MAD of accepted windows -- the quotable DEM->imagery offset.
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation

sys.path.insert(0, str(Path(__file__).parent))
from p20_snap_rescore import BOX4326, WV, DOLAN
from firescape import change as ch, epochs, sources
import rasterio
from rasterio.merge import merge
from rasterio.warp import Resampling, reproject, transform_bounds
from shapely.geometry import box

ref = epochs.grid(BOX4326, "EPSG:32610", 2.0)
pre_src = sources.MaxarStrips(WV / "2020_11_29")
az0, el0 = pre_src.sun()
pre = pre_src.read_into(ref)
obs = ch.brightness(pre["blue"], pre["green"], pre["red"], pre["nir"])
print("pre mosaicked", flush=True)

b = transform_bounds("EPSG:32610", "EPSG:4326",
                     *box(*transform_bounds("EPSG:4326", "EPSG:32610",
                                            *BOX4326)).bounds)
want = (b[0]-.03, b[1]-.03, b[2]+.03, b[3]+.03)
srcs = [rasterio.open(t) for t in sorted(DOLAN.glob("USGS_13_*.tif"))]
dem, tr_nat = merge(srcs, bounds=want, nodata=np.nan)
crs_nat = srcs[0].crs
[s.close() for s in srcs]
dem = dem[0].astype(np.float32)
lat = (want[1]+want[3])/2
gy, gx = np.gradient(dem, abs(tr_nat.e)*111132.0,
                     abs(tr_nat.a)*111320.0*np.cos(np.radians(lat)))
sl, asp = np.arctan(np.hypot(gx, gy)), np.arctan2(-gx, gy)
z_, a_ = np.radians(90-el0), np.radians(az0)
cosi = (np.cos(z_)*np.cos(sl)+np.sin(z_)*np.sin(sl)*np.cos(a_-asp)).astype(np.float32)
mod = np.full((ref["height"], ref["width"]), np.nan, np.float32)
reproject(cosi, mod, src_transform=tr_nat, src_crs=crs_nat,
          dst_transform=ref["transform"], dst_crs=ref["crs"],
          resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)

def hp(x, sig=25):
    f = np.ma.filled(np.ma.masked_invalid(np.ma.asarray(x)), np.nan)
    good = np.isfinite(f)
    f = np.where(good, f, np.nanmean(f))
    return np.where(good, f - gaussian_filter(f, sig), 0.0), good

O, vO = hp(obs)
M, vM = hp(mod)
valid = vO & vM
H, W = valid.shape
WIN, MAX_SHIFT = 768, 30

cands = []
for cy in range(WIN, H-WIN, WIN//2):
    for cx in range(WIN, W-WIN, WIN//2):
        s = (slice(cy-WIN//2, cy+WIN//2), slice(cx-WIN//2, cx+WIN//2))
        vf = valid[s].mean()
        if vf < 0.9:
            continue
        relief = float(np.std(M[s]))
        cands.append((relief, vf, s, cy, cx))
cands.sort(reverse=True)
print(f"{len(cands)} valid windows; probing top 12 by relief", flush=True)

acc = []
print(f"{'win':>4} {'relief':>7} {'dy':>7} {'dx':>7} {'err':>6}  verdict")
for i, (relief, vf, s, cy, cx) in enumerate(cands[:12]):
    shift, err, _ = phase_cross_correlation(M[s], O[s], upsample_factor=10,
                                            normalization="phase")
    dy, dx = -float(shift[0]), -float(shift[1])
    ok = (abs(dy) <= MAX_SHIFT and abs(dx) <= MAX_SHIFT and err < 0.72)
    print(f"{i:>4} {relief:>7.4f} {dy:>7.1f} {dx:>7.1f} {err:>6.3f}  "
          f"{'ACCEPT' if ok else 'reject'}", flush=True)
    if ok:
        acc.append((dy, dx))
if acc:
    a = np.array(acc)
    med = np.median(a, axis=0)
    mad = np.median(np.abs(a - med), axis=0)
    print(f"\nACCEPTED {len(acc)}/12 windows")
    print(f"DEM->imagery offset: dy={med[0]:+.1f}±{mad[0]:.1f} px, "
          f"dx={med[1]:+.1f}±{mad[1]:.1f} px  "
          f"= ({med[0]*2:+.0f}±{mad[0]*2:.0f}, {med[1]*2:+.0f}±{mad[1]*2:.0f}) m")
    print(f"network->imagery was (+6,+50) m  =>  network->DEM residual "
          f"≈ ({6-med[0]*2:+.0f}, {50-med[1]*2:+.0f}) m")
else:
    print("no window passed the gate")
