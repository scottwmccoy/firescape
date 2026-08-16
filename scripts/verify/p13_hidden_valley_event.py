"""Hidden Valley 2026-06-19 debris-flow event: first real epoch pair.

Two fan sites flag the event reach (Scott's field knowledge, WGS84):
south fan (-119.702819, 39.495294), north fan (-119.682656, 39.515572).
This driver runs the v0 stack--z slice of the verification design over the
staged pre/post epochs and asks one question: do the tracks and deposits
between those points light up?

v0 honesty list (each is a later design stage, not an oversight):
no AROSICS (a phase-correlation check REPORTS misregistration instead),
no per-scene normalization, no cast-shadow mask (June/July sun is high),
no corridor conditioning -- the z maps are unconditioned, so expect
false positives the pfdf-corridor stage exists to kill.

Run:  /opt/anaconda3/envs/FireMan/bin/python p13_hidden_valley_event.py
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

from firescape import change as ch, epochs, paths

EVENT = "2026-06-19"
POINTS = {"south fan": (-119.702819, 39.495294),
          "north fan": (-119.682656, 39.515572)}
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()


def epoch_stacks(epoch, ref=None):
    return epochs.build(paths.raw_dir("planet", "hidden_valley", epoch), ref)


def px(ref, lon, lat):
    import pyproj
    x, y = pyproj.Transformer.from_crs("EPSG:4326", ref["crs"],
                                       always_xy=True).transform(lon, lat)
    c, r = ~ref["transform"] * (x, y)
    return int(r), int(c)


def stretch(rgb, lo=2, hi=98):
    v = rgb[rgb > 0]
    a, b = np.percentile(v, [lo, hi])
    return np.clip((rgb - a) / (b - a), 0, 1) ** 0.9


print("=== PRE epoch ===", flush=True)
pre, pre_rgb, pre_nirs, pre_ids, ref = epoch_stacks("pre_20260619")
print("=== POST epoch ===", flush=True)
post, post_rgb, post_nirs, post_ids, _ = epoch_stacks("post_20260619", ref)

print("=== registration check (vs first pre scene NIR, centre window) ===")
w = ref["height"] // 2 - 512, ref["width"] // 2 - 512
sl = (slice(w[0], w[0] + 1024), slice(w[1], w[1] + 1024))
offsets = {}
for sid, nir in zip(pre_ids + post_ids, pre_nirs + post_nirs):
    if sid == pre_ids[0]:
        continue
    dy, dx = ch.scene_offset(pre_nirs[0][sl], nir[sl])
    offsets[sid] = (round(dy, 2), round(dx, 2))
mags = [np.hypot(*v) for v in offsets.values()]
print(f"|shift| median {np.median(mags):.2f} px, max {np.max(mags):.2f} px")

z = {k: ch.robust_z(pre[k][0], post[k][0], pre[k][1])
     for k in ("brightness", "msavi2", "redness")}
rd = ch.rdndvi(pre["ndvi"][0], post["ndvi"][0])

# ---------------------------------------------------------------- figures --
pts = {name: px(ref, lon, lat) for name, (lon, lat) in POINTS.items()}


def mark(ax):
    for name, (r, c) in pts.items():
        ax.plot(c, r, "o", ms=11, mfc="none", mec="#00A0B0", mew=2.2)
        ax.annotate(name, (c, r), xytext=(8, -12),
                    textcoords="offset points", color="#00A0B0", fontsize=9,
                    fontweight="bold")
    ax.set_axis_off()


fig, axes = plt.subplots(2, 2, figsize=(15, 15), dpi=140)
axes[0, 0].imshow(stretch(pre_rgb))
axes[0, 0].set_title(f"pre median ({len(pre_ids)} scenes, May 9 – Jun 12)")
axes[0, 1].imshow(stretch(post_rgb))
axes[0, 1].set_title(f"post median ({len(post_ids)} scenes, Jun 20 – Jul 3)")
im = axes[1, 0].imshow(z["brightness"], cmap="RdBu_r", vmin=-8, vmax=8)
axes[1, 0].set_title("z(ΔBrightness) — red = brightened (fresh sediment)")
fig.colorbar(im, ax=axes[1, 0], shrink=0.7)
im = axes[1, 1].imshow(z["msavi2"], cmap="PuOr", vmin=-8, vmax=8)
axes[1, 1].set_title("z(ΔMSAVI2) — brown = vegetation stripped")
fig.colorbar(im, ax=axes[1, 1], shrink=0.7)
for ax in axes.ravel():
    mark(ax)
fig.suptitle(f"Hidden Valley event {EVENT} — stack–z first look "
             "(unconditioned; corridor stage pending)", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "hv_event_overview.png", bbox_inches="tight")
plt.close(fig)

H = 300                                                    # ~900 m half-chip
fig, axes = plt.subplots(2, 3, figsize=(15, 10.5), dpi=140)
for row, (name, (r, c)) in enumerate(pts.items()):
    win = (slice(max(r - H, 0), r + H), slice(max(c - H, 0), c + H))
    axes[row, 0].imshow(stretch(pre_rgb[win]))
    axes[row, 0].set_title(f"{name} — pre")
    axes[row, 1].imshow(stretch(post_rgb[win]))
    axes[row, 1].set_title(f"{name} — post")
    im = axes[row, 2].imshow(z["brightness"][win], cmap="RdBu_r",
                             vmin=-8, vmax=8)
    axes[row, 2].set_title(f"{name} — z(ΔBrightness)")
    fig.colorbar(im, ax=axes[row, 2], shrink=0.8)
    for ax in axes[row]:
        ax.plot(min(c, H), min(r, H), "o", ms=12, mfc="none",
                mec="#00A0B0", mew=2.2)
        ax.set_axis_off()
fig.suptitle(f"Hidden Valley {EVENT} — the two reported fan sites", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "hv_event_chips.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- summary --
def at(name, arr, half=17):                                # ~100 m window
    r, c = pts[name]
    w = arr[max(r - half, 0):r + half, max(c - half, 0):c + half]
    return round(float(np.ma.median(w)), 2)

summary = {
    "event": EVENT, "points": POINTS,
    "pre_scenes": pre_ids, "post_scenes": post_ids,
    "registration_px": {"median": round(float(np.median(mags)), 2),
                        "max": round(float(np.max(mags)), 2),
                        "per_scene": offsets},
    "at_points": {name: {"z_brightness": at(name, z["brightness"]),
                         "z_msavi2": at(name, z["msavi2"]),
                         "z_redness": at(name, z["redness"]),
                         "rdndvi_pct": at(name, rd)}
                  for name in pts},
    "background": {
        "z_brightness_p50": round(float(np.ma.median(z["brightness"])), 2),
        "z_brightness_p99": round(float(np.percentile(
            np.ma.masked_invalid(z["brightness"]).compressed(), 99)), 2)},
}
(OUT / "hv_event_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
