"""Global network-to-imagery offset for WV-Dolan: one FFT, full region.

Per-block correlation was noise-limited; integrating the whole joint
window multiplies the corridor support ~20x. Reports the correlation peak
(the offset estimate), its prominence, and the AUC at that offset vs (0,0).
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pyogrio
from rasterio.warp import transform_bounds
from scipy.stats import mannwhitneyu
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).parent))
from p20_snap_rescore import maxar_z, BOX4326, WV, INV
from firescape import corridor as co, epochs

S = Path(sys.argv[1])
ref = epochs.grid(BOX4326, "EPSG:32610", 2.0)
z = maxar_z(WV / "2020_11_29", [WV / "2021_04_19", WV / "2021_05_08"], ref)
seg = pyogrio.read_dataframe(INV, columns=["UpArea_km2", "Confidence"]).to_crs(ref["crs"])
seg = seg[seg.intersects(box(*transform_bounds("EPSG:4326", ref["crs"], *BOX4326)))].reset_index(drop=True)
labels = co.corridor_raster(seg, ref, area_col="UpArea_km2")
t_all = seg["Confidence"].replace({2: 3})

dy, dx, info = co.local_offsets(labels, z, block=10**6, max_off=30,
                                min_snr=0.0, smooth=False, with_info=True)
print(f"global correlation peak: dy={info['raw_dy'][0,0]:+d} "
      f"dx={info['raw_dx'][0,0]:+d} px "
      f"({info['raw_dy'][0,0]*2:+d},{info['raw_dx'][0,0]*2:+d} m), "
      f"SNR={info['snr'][0,0]:.1f}", flush=True)

def auc(hi, lo):
    hi, lo = hi.dropna(), lo.dropna()
    return float(mannwhitneyu(hi, lo, alternative="greater").statistic
                 / (len(hi)*len(lo))) if len(hi) and len(lo) else float("nan")

zf = z.filled(np.nan)
def score(oy, ox):
    zz = np.ma.masked_invalid(np.roll(zf, (oy, ox), axis=(0, 1)))
    st = co.segment_stats(labels, {"z": zz}).reindex(range(len(seg)))
    ok = st["n_pixels"].fillna(0) >= 12
    v, t = st.loc[ok, "z_mean"], t_all[ok]
    return auc(v[t > 0], v[t == 0]), auc(v[t == 3], v[t == 1]), int(ok.sum())

r0 = score(0, 0)
rp = score(int(info['raw_dy'][0, 0]), int(info['raw_dx'][0, 0]))
print(f"(0,0):  resp={r0[0]:.3f} DF|fl={r0[1]:.3f} n={r0[2]}")
print(f"peak :  resp={rp[0]:.3f} DF|fl={rp[1]:.3f} n={rp[2]}")
