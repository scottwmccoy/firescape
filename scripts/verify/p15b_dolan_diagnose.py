"""Why is Dolan corridor z at chance? Look before concluding.

Saves the z fields (npz) so later iterations skip the 8-minute rebuild,
renders z with the truth network on top, chips two field-verified inland
drainages, and prints the mechanical suspects: corridor pixel counts,
UpArea sanity, epoch-median registration, mask fractions.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyogrio
from rasterio.warp import transform_bounds
from shapely.geometry import box

from firescape import change as ch, corridor as co, epochs, paths

BOX4326 = (-121.50, 36.00, -121.32, 36.18)
INV = ("/Users/scottmccoy/Library/CloudStorage/Box-Box/SWMresearch/"
       "PostFireDebrisFlows/2020_Dolan/confidence_map/"
       "Dolan_ROC_BasinOverride.shp")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
CACHE = OUT / "dolan_z_cache.npz"

ref = epochs.grid(BOX4326, "EPSG:32610", 3.0)

if CACHE.exists():
    d = np.load(CACHE)
    zb = np.ma.masked_invalid(d["zb"].astype(np.float32))
    pre_med = d["pre_med"].astype(np.float32)
    post_med = d["post_med"].astype(np.float32)
    pre_count, post_count = d["pre_count"], d["post_count"]
else:
    LITE = dict(indices=("brightness", "msavi2"), rgb=False, keep_nir=False,
                dtype=np.float16, verbose=False)
    pre, _, _, _, _ = epochs.build(
        paths.raw_dir("planet", "dolan", "pre_20210127"), ref, **LITE)
    post, _, _, _, _ = epochs.build(
        paths.raw_dir("planet", "dolan", "post_20210127"), ref, **LITE)
    zb = ch.robust_z(pre["brightness"][0].astype(np.float32),
                     post["brightness"][0].astype(np.float32),
                     pre["brightness"][1].astype(np.float32))
    zb = zb - np.ma.median(zb)
    pre_med = pre["brightness"][0].astype(np.float32).filled(np.nan)
    post_med = post["brightness"][0].astype(np.float32).filled(np.nan)
    pre_count, post_count = pre["brightness"][2], post["brightness"][2]
    np.savez_compressed(
        CACHE, zb=np.ma.filled(zb, np.nan).astype(np.float16),
        pre_med=pre_med.astype(np.float16),
        post_med=post_med.astype(np.float16),
        pre_count=pre_count.astype(np.uint8),
        post_count=post_count.astype(np.uint8))
    print("z cached", flush=True)

print("=== mask / coverage ===")
print(f"pre usable-count: median {np.median(pre_count)}, "
      f"px with <3 obs: {(pre_count < 3).mean():.1%}")
print(f"post usable-count: median {np.median(post_count)}, "
      f"px with <3 obs: {(post_count < 3).mean():.1%}")
print(f"z valid fraction: {(~np.ma.getmaskarray(zb)).mean():.1%}")

print("=== inter-epoch registration (median vs median, centre 1024²) ===")
h, w = ref["height"], ref["width"]
sl = (slice(h // 2 - 512, h // 2 + 512), slice(w // 2 - 512, w // 2 + 512))
try:
    dy, dx = ch.scene_offset(np.ma.masked_invalid(pre_med[sl]),
                             np.ma.masked_invalid(post_med[sl]))
    print(f"pre->post epoch shift: dy={dy:.2f}, dx={dx:.2f} px")
except Exception as e:
    print("offset check failed:", e)

seg = pyogrio.read_dataframe(INV, columns=["UpArea_km2", "Confidence"])
seg = seg.to_crs(ref["crs"])
gbox = box(*transform_bounds("EPSG:4326", ref["crs"], *BOX4326))
seg = seg[seg.intersects(gbox)].reset_index(drop=True)
print("=== corridor mechanics ===")
print("UpArea_km2:", seg["UpArea_km2"].describe()[["min", "50%", "max"]].to_dict())
labels = co.corridor_raster(seg, ref, area_col="UpArea_km2")
stats = co.segment_stats(labels, {"zb": zb})
print("corridor n_pixels: median "
      f"{stats['n_pixels'].median():.0f}, p10 {stats['n_pixels'].quantile(.1):.0f}")

# ---- figures ---------------------------------------------------------------
tr = ref["transform"]
extent = (tr.c, tr.c + tr.a * w, tr.f + tr.e * h, tr.f)
truth_df = seg[seg["Confidence"].isin([2, 3])]
truth_fl = seg[seg["Confidence"] == 1]

fig, axes = plt.subplots(1, 2, figsize=(17, 10), dpi=130)
axes[0].imshow(np.ma.masked_invalid(post_med), cmap="gray",
               vmin=np.nanpercentile(post_med, 2),
               vmax=np.nanpercentile(post_med, 98), extent=extent)
axes[0].set_title("post-epoch median brightness (what the imagery shows)")
im = axes[1].imshow(zb, cmap="RdBu_r", vmin=-6, vmax=6, extent=extent)
axes[1].set_title("z(ΔBrightness), centred")
fig.colorbar(im, ax=axes[1], shrink=0.6)
for ax in axes:
    truth_fl.plot(ax=ax, color="#F5D000", linewidth=0.5, alpha=0.8, zorder=5)
    truth_df.plot(ax=ax, color="#C1272D", linewidth=1.0, alpha=0.9, zorder=6)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_axis_off()
fig.suptitle("Dolan diagnostics — truth network over imagery and z "
             "(red = inventory DF, yellow = fluvial)", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "dolan_diagnose_overview.png", bbox_inches="tight")
plt.close(fig)

# chips at two field-verified inland systems (Santa Lucia, Mill)
import pyproj
t = pyproj.Transformer.from_crs("EPSG:4326", ref["crs"], always_xy=True)
CHIPS = {"Santa Lucia / Rattlesnake": (-121.415, 36.115),
         "Mill Creek headwaters": (-121.445, 36.005)}
H = 500
fig, axes = plt.subplots(2, 3, figsize=(16, 11), dpi=130)
for row, (name, (lon, lat)) in enumerate(CHIPS.items()):
    x, y = t.transform(lon, lat)
    c = int((x - tr.c) / tr.a); r = int((y - tr.f) / tr.e)
    win = (slice(max(r - H, 0), r + H), slice(max(c - H, 0), c + H))
    ext = (tr.c + tr.a * win[1].start, tr.c + tr.a * win[1].stop,
           tr.f + tr.e * win[0].stop, tr.f + tr.e * win[0].start)
    for col, (arr, cmap, vlim, title) in enumerate([
            (pre_med[win], "gray", None, f"{name} — pre"),
            (post_med[win], "gray", None, f"{name} — post"),
            (np.ma.masked_invalid(np.ma.filled(zb, np.nan)[win]), "RdBu_r",
             (-6, 6), f"{name} — z")]):
        ax = axes[row, col]
        if vlim:
            ax.imshow(arr, cmap=cmap, vmin=vlim[0], vmax=vlim[1], extent=ext)
        else:
            a = np.ma.masked_invalid(arr)
            ax.imshow(a, cmap=cmap, vmin=np.nanpercentile(arr, 2),
                      vmax=np.nanpercentile(arr, 98), extent=ext)
        truth_df.plot(ax=ax, color="#C1272D", linewidth=1.2, alpha=0.9)
        truth_fl.plot(ax=ax, color="#F5D000", linewidth=0.6, alpha=0.8)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_axis_off()
        ax.set_title(title, fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "dolan_diagnose_chips.png", bbox_inches="tight")
plt.close(fig)
print("figures written", flush=True)
