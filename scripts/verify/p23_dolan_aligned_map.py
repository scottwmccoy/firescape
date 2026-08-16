"""The aligned side-by-side Dolan map: their inventory vs our prediction.

WV-2 pair, network shifted onto the imagery frame by the measured global
offset, prevalence-matched thresholds, both panels cropped to the scored
(joint-coverage) region, hillshade backdrop, Dolan-figure styling. The
network-shift sign is chosen empirically: both signs are scored and the
one reproducing the validated AUC (~0.80) is drawn.
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyogrio
from matplotlib.lines import Line2D
from rasterio.warp import transform_bounds
from scipy.stats import mannwhitneyu
from shapely.geometry import box
from shapely import affinity

sys.path.insert(0, str(Path(__file__).parent))
from p20_snap_rescore import maxar_z, BOX4326, WV, INV, DOLAN
from firescape import corridor as co, epochs, relief

S = Path(sys.argv[1])
OFF_PX = (3, 25)          # |dy|, |dx| from the p22 global estimate
ref = epochs.grid(BOX4326, "EPSG:32610", 2.0)
z = maxar_z(WV / "2020_11_29", [WV / "2021_04_19", WV / "2021_05_08"], ref)
seg = pyogrio.read_dataframe(INV, columns=["UpArea_km2", "Confidence"]).to_crs(ref["crs"])
seg = seg[seg.intersects(box(*transform_bounds("EPSG:4326", ref["crs"], *BOX4326)))].reset_index(drop=True)
t_all = seg["Confidence"].replace({2: 3})

def auc(hi, lo):
    hi, lo = hi.dropna(), lo.dropna()
    return float(mannwhitneyu(hi, lo, alternative="greater").statistic
                 / (len(hi)*len(lo))) if len(hi) and len(lo) else float("nan")

def run(sign):
    dy, dx = sign*OFF_PX[0], sign*OFF_PX[1]
    s2 = seg.copy()
    # shifting geometry east = +x when label-roll dx>0 (2 m px)
    s2["geometry"] = s2.geometry.apply(
        lambda g: affinity.translate(g, xoff=dx*2.0, yoff=-dy*2.0))
    lab = co.corridor_raster(s2, ref, area_col="UpArea_km2")
    st = co.segment_stats(lab, {"z": z}).reindex(range(len(seg)))
    ok = st["n_pixels"].fillna(0) >= 12
    v, t = st.loc[ok, "z_mean"], t_all[ok]
    return auc(v[t > 0], v[t == 0]), (s2, st, ok)

best = None
for sign in (+1, -1):
    r, pack = run(sign)
    print(f"sign {sign:+d}: responded AUC {r:.3f}", flush=True)
    if best is None or r > best[0]:
        best = (r, sign, pack)
resp_auc, sign, (s2, st, ok) = best
print(f"using sign {sign:+d} -> network shift ({sign*OFF_PX[0]*2:+d},"
      f"{sign*OFF_PX[1]*2:+d}) m in (y,x)", flush=True)

v, t = st.loc[ok, "z_mean"], t_all[ok]
p_resp, p_df = float((t > 0).mean()), float((t == 3).mean())
fl_t = float(np.nanquantile(v, 1 - p_resp))
df_t = float(np.nanquantile(v, 1 - p_df))
import pandas as pd
pred = pd.Series(0, index=v.index)
pred[v >= fl_t] = 1
pred[v >= df_t] = 3
acc = float((pred == t).mean())
df_rec = float((pred[t == 3] == 3).mean())
df_prec = float((t[pred == 3] == 3).mean()) if (pred == 3).any() else np.nan
dffl = auc(v[t == 3], v[t == 1])
print(f"resp AUC {resp_auc:.3f} | DF|fl {dffl:.3f} | 3-class acc {acc:.0%} "
      f"| DF recall {df_rec:.0%} prec {df_prec:.0%}", flush=True)

s2 = s2.assign(pred=-1)
s2.loc[v.index, "pred"] = pred
scored = s2[s2["pred"] >= 0]
xb, yb = scored.total_bounds[[0, 2]], scored.total_bounds[[1, 3]]
pad = 500
crop = (xb[0]-pad, yb[0]-pad, xb[1]+pad, yb[1]+pad)

tr = ref["transform"]
c0 = max(int((crop[0]-tr.c)/tr.a), 0); c1 = min(int((crop[2]-tr.c)/tr.a), ref["width"])
r1 = max(int((crop[3]-tr.f)/tr.e), 0); r0 = min(int((crop[1]-tr.f)/tr.e), ref["height"])
import rasterio
sub_tr = rasterio.transform.from_origin(tr.c + c0*tr.a, tr.f + r1*tr.e, tr.a, -tr.e)
hs = relief.shaded_relief(sorted(DOLAN.glob("USGS_13_*.tif")), sub_tr,
                          (r0-r1, c1-c0), crs=str(ref["crs"]), shade_res_m=10.0)
ext = (crop[0], crop[2], crop[1], crop[3])

STYLE = {0: ("white", 0.6, 0.6), 1: ("#F5D000", 1.6, 0.95),
         2: ("#F28C28", 2.0, 1.0), 3: ("#C1272D", 2.0, 1.0)}
asp = (crop[3]-crop[1]) / (crop[2]-crop[0])
fig, axes = plt.subplots(1, 2, figsize=(4.6*2 + 1.2, min(4.6*asp + 1.8, 13)),
                         dpi=150)
s2["scored"] = s2["pred"] >= 0
for ax, frame, col, title in (
        (axes[0], s2, "Confidence",
         "Cavagnaro & McCoy inventory\n(faded = outside imaged window)"),
        (axes[1], scored, "pred",
         f"firescape corridor z on WV-2, network-aligned\n"
         f"prevalence-matched: fluvial ≥ {fl_t:.2f}, DF ≥ {df_t:.2f}")):
    ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=ext, zorder=0)
    for cls, (color, lw, al) in STYLE.items():
        sub = frame[frame[col] == cls]
        if not len(sub):
            continue
        if col == "Confidence":
            dim = sub[~sub["scored"]]
            lit = sub[sub["scored"]]
            if len(dim):
                dim.plot(ax=ax, color=color, linewidth=lw*0.5, alpha=al*0.3,
                         zorder=3 + (cls > 0))
            if len(lit):
                lit.plot(ax=ax, color=color, linewidth=lw, alpha=al,
                         zorder=4 + (cls > 0) + (cls > 1))
        else:
            sub.plot(ax=ax, color=color, linewidth=lw, alpha=al,
                     zorder=4 + (cls > 0) + (cls > 1))
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_axis_off(); ax.set_title(title, fontsize=11)
axes[0].legend(handles=[
    Line2D([], [], color="white", lw=1.6, label="no erosion"),
    Line2D([], [], color="#F5D000", lw=2.4, label="fluvial erosion"),
    Line2D([], [], color="#F28C28", lw=2.8, label="DF (remotely mapped)"),
    Line2D([], [], color="#C1272D", lw=2.8, label="debris flow"),
], loc="upper left", bbox_to_anchor=(0.0, -0.01), ncol=4,
    framealpha=0.92, fontsize=8)
fig.suptitle(f"Dolan Fire, storm of 2021-01-27 — {int(ok.sum()):,} scored "
             f"segments · responded AUC {resp_auc:.2f} · DF-vs-fluvial AUC "
             f"{dffl:.2f} · 3-class agreement {acc:.0%}", fontsize=13)
fig.tight_layout()
fig.savefig(S / "dolan_aligned_side_by_side.png", bbox_inches="tight")
fig.savefig(S / "dolan_aligned_side_by_side.pdf", bbox_inches="tight")
print("map written", flush=True)
