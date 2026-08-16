"""Dolan Fire, storm of 2021-01-27: score the detector against the inventory.

The Cavagnaro/McCoy Dolan inventory (USGS release 10.5066/P13ZGR6F; local
shapefile copy) classified every channel segment from imagery + field work:
``Confidence`` 0 = no erosion, 1 = fluvial erosion, 2 = remotely-mapped
debris flow, 3 = field-verified debris flow. This driver runs the
stack--z--corridor chain over the inland half of the fire (Santa Lucia /
Rattlesnake / Wizard country; the coastal redwood basins occlude the
channels) on THEIR network, then asks the calibration questions:

1. Does corridor-integrated z separate responded from unresponded reaches?
2. Does it separate debris flow from fluvial erosion -- i.e. is Scott's
   three-class severe/slight/none reading of the evidence supported?
3. What thresholds should Hidden Valley (and future events) use?

Scoring drops segments with fewer than MIN_PIX usable corridor pixels
(outside the imaged bbox, cloud-masked, or sliver corridors): absence of
imagery must never score as a "no erosion" prediction.

Run:  /opt/anaconda3/envs/FireMan/bin/python p15_dolan_comparison.py [outdir]
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
import pandas as pd
import pyogrio
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu
from shapely.geometry import box

import geopandas as gpd

from firescape import change as ch, corridor as co, epochs, paths

EVENT = "2021-01-27"
BOX4326 = (-121.50, 36.00, -121.32, 36.18)
INV = ("/Users/scottmccoy/Library/CloudStorage/Box-Box/SWMresearch/"
       "PostFireDebrisFlows/2020_Dolan/confidence_map/"
       "Dolan_ROC_BasinOverride.shp")
MIN_PIX = 12
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()

# ---- epochs (lite: the grid is ~40M px x 30 frames) ------------------------
# The grid comes from the ordered AOI: Dolan deliveries are strip TILES, so
# adopting the first scene's clip grid confines everything to its corner.
print("=== epochs ===", flush=True)
ref = epochs.grid(BOX4326, "EPSG:32610", 3.0)
print(f"grid {ref['width']}x{ref['height']} EPSG:32610", flush=True)
LITE = dict(indices=("brightness", "msavi2"), rgb=False,
            keep_nir=False, dtype=np.float16, verbose=True)
pre, _, _, pre_ids, _ = epochs.build(
    paths.raw_dir("planet", "dolan", "pre_20210127"), ref, **LITE)
post, _, _, post_ids, _ = epochs.build(
    paths.raw_dir("planet", "dolan", "post_20210127"), ref, **LITE)
z = {}
for k in ("brightness", "msavi2"):
    zz = ch.robust_z(pre[k][0].astype(np.float32),
                     post[k][0].astype(np.float32),
                     pre[k][1].astype(np.float32))
    z[f"z_{k}"] = zz - np.ma.median(zz)
print("z fields built", flush=True)

# ---- their network + truth --------------------------------------------------
seg = pyogrio.read_dataframe(INV, columns=["Segment_ID", "UpArea_km2",
                                           "Confidence", "Mapped_DF"])
seg = seg.to_crs(ref["crs"])
from rasterio.warp import transform_bounds
gbox = box(*transform_bounds("EPSG:4326", ref["crs"], *BOX4326))
seg = seg[seg.intersects(gbox)].reset_index(drop=True)
print(f"{len(seg):,} inventory segments in the imaged window", flush=True)

labels = co.corridor_raster(seg, ref, area_col="UpArea_km2")
stats = co.segment_stats(labels, z).reindex(range(len(seg)))
seg = seg.join(stats)
ok = seg["n_pixels"].fillna(0) >= MIN_PIX
print(f"{int(ok.sum()):,} segments with >= {MIN_PIX} usable pixels", flush=True)

d = seg[ok].copy()
truth3 = d["Confidence"].replace({2: 3})          # collapse remote-DF into DF

# ---- separability -----------------------------------------------------------
def auc(a, b):
    """P(sample from a > sample from b) via Mann-Whitney."""
    a, b = a.dropna(), b.dropna()
    if not len(a) or not len(b):
        return float("nan")
    u = mannwhitneyu(a, b, alternative="greater").statistic
    return float(u / (len(a) * len(b)))

metrics = {}
for col in ("z_brightness_mean", "z_brightness_p90", "z_msavi2_mean",
            "z_msavi2_p90"):
    resp = auc(d.loc[truth3 > 0, col], d.loc[truth3 == 0, col])
    sever = auc(d.loc[truth3 == 3, col], d.loc[truth3 == 1, col])
    metrics[col] = {"auc_responded": round(resp, 3),
                    "auc_df_vs_fluvial": round(sever, 3)}
    print(f"{col:22s} responded AUC={resp:.3f}  DF|fluvial AUC={sever:.3f}",
          flush=True)

# ---- calibrate thresholds by prevalence matching ---------------------------
FIELD = "z_brightness_mean"
v = d[FIELD]
p_resp = float((truth3 > 0).mean())
p_df = float((truth3 == 3).mean())
fluvial_t = float(np.nanquantile(v, 1 - p_resp))
df_t = float(np.nanquantile(v, 1 - p_df))
print(f"prevalence-matched thresholds on {FIELD}: "
      f"fluvial >= {fluvial_t:.2f}, DF >= {df_t:.2f}", flush=True)

pred = pd.Series(0, index=d.index)
pred[v >= fluvial_t] = 1
pred[v >= df_t] = 3
cm = pd.crosstab(truth3.rename("truth"), pred.rename("pred"))
print("confusion (rows=truth, cols=pred):")
print(cm.to_string(), flush=True)
acc = float((pred == truth3).mean())
acc_bin = float(((pred > 0) == (truth3 > 0)).mean())
recall_df = float((pred[truth3 == 3] == 3).mean()) if (truth3 == 3).any() else np.nan
prec_df = float((truth3[pred == 3] == 3).mean()) if (pred == 3).any() else np.nan

seg["pred"] = np.where(ok, pred.reindex(seg.index).fillna(-1), -1).astype(int)
seg.to_file(OUT / "dolan_comparison_segments.gpkg", driver="GPKG")

# ---- figures ----------------------------------------------------------------
# 1. distributions by their class
fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
groups = [(0, "no erosion", "0.55"), (1, "fluvial", "#D9A400"),
          (3, "debris flow", "#C1272D")]
bins = np.linspace(np.nanpercentile(v, 0.5), np.nanpercentile(v, 99.5), 60)
for cls, name, color in groups:
    ax.hist(v[truth3 == cls], bins=bins, density=True, histtype="step",
            lw=2.2, color=color, label=f"{name} (n={int((truth3==cls).sum()):,})")
for t, lab in ((fluvial_t, "fluvial threshold"), (df_t, "DF threshold")):
    ax.axvline(t, color="k", lw=0.8, ls="--")
    ax.text(t, ax.get_ylim()[1] * 0.95, f" {lab}", fontsize=8, va="top")
ax.set_xlabel("corridor-mean z(ΔBrightness), background-centred")
ax.set_ylabel("density")
ax.legend(fontsize=9)
ax.set_title(f"Dolan {EVENT}: does corridor z separate the inventory classes?\n"
             f"responded AUC {metrics[FIELD]['auc_responded']:.2f} · "
             f"DF vs fluvial AUC {metrics[FIELD]['auc_df_vs_fluvial']:.2f}",
             fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "dolan_separability.png", bbox_inches="tight")
plt.close(fig)

# 2. side-by-side maps on their network
STYLE = {0: ("white", 0.4, 0.5), 1: ("#F5D000", 1.1, 0.9),
         2: ("#F28C28", 1.6, 1.0), 3: ("#C1272D", 1.6, 1.0)}
fig, axes = plt.subplots(1, 2, figsize=(16, 9), dpi=150)
for ax, col, title in ((axes[0], "Confidence",
                        "Cavagnaro & McCoy inventory (truth)"),
                       (axes[1], "pred", "firescape corridor z, "
                        "prevalence-matched thresholds")):
    shown = seg[seg["pred"] >= 0] if col == "pred" else seg
    for cls, (c, lw, al) in STYLE.items():
        sub = shown[shown[col] == cls]
        if len(sub):
            sub.plot(ax=ax, color=c, linewidth=lw, alpha=al,
                     zorder=4 + (cls > 0) + (cls > 1))
    ax.set_facecolor("0.35")
    ax.set_xlim(gbox.bounds[0], gbox.bounds[2])
    ax.set_ylim(gbox.bounds[1], gbox.bounds[3])
    ax.set_axis_off()
    ax.set_title(title, fontsize=11)
axes[0].legend(handles=[
    Line2D([], [], color="white", lw=1.5, label="no erosion"),
    Line2D([], [], color="#F5D000", lw=2.2, label="fluvial erosion"),
    Line2D([], [], color="#F28C28", lw=2.6, label="DF (remote; truth only)"),
    Line2D([], [], color="#C1272D", lw=2.6, label="debris flow"),
], loc="lower left", framealpha=0.9, fontsize=9)
fig.suptitle(f"Dolan Fire {EVENT} storm — inland window, "
             f"{int(ok.sum()):,} scored segments · 3-class acc {acc:.0%} · "
             f"responded acc {acc_bin:.0%} · DF recall {recall_df:.0%} / "
             f"precision {prec_df:.0%}", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "dolan_side_by_side.png", bbox_inches="tight")
plt.close(fig)

summary = {
    "event": EVENT, "window": BOX4326,
    "segments_in_window": int(len(seg)), "segments_scored": int(ok.sum()),
    "pre_scenes": len(pre_ids), "post_scenes": len(post_ids),
    "separability_auc": metrics,
    "thresholds": {"field": FIELD, "fluvial_t": round(fluvial_t, 3),
                   "df_t": round(df_t, 3), "method": "prevalence-matched"},
    "scores": {"acc_3class": round(acc, 3), "acc_responded": round(acc_bin, 3),
               "df_recall": round(recall_df, 3),
               "df_precision": round(prec_df, 3)},
    "confusion": {str(k): {str(kk): int(vv) for kk, vv in r.items()}
                  for k, r in cm.to_dict("index").items()},
}
(OUT / "dolan_comparison_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary["scores"], indent=2))
