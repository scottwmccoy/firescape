"""Hidden Valley 2026-06-19: corridor-conditioned per-segment response map.

The stack--z--segment chain end to end on the first real event: epoch z maps
(p13's machinery, now firescape.epochs) -> pfdf segment corridors ->
corridor-integrated z -> three-class response, drawn Dolan-inventory style
(white / yellow / red on hillshade).

The z fields are background-centred before integration (median over all
valid pixels subtracted) -- the v0 spatial control for the scene-wide
June->July cheatgrass-curing brightening. Class thresholds are PROVISIONAL
(df >= 2.5, fluvial >= 1.0 on centred corridor-mean z_brightness) until the
Dolan calibration lands.

Known v0 gap, by design: pfdf segments stop at the range front (valley
mask), so fan deposits beyond the network -- including the north-fan lobe at
the canal -- are outside every corridor. The fan-zone stage exists for that;
this map is the TRACK inventory.

Run:  /opt/anaconda3/envs/FireMan/bin/python p14_hv_corridor.py [outdir]
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
from rasterio.warp import transform_bounds
from shapely.geometry import box

import geopandas as gpd

from firescape import change as ch, corridor as co, epochs, paths

EVENT = "2026-06-19"
AOI = (-119.72, 39.45, -119.63, 39.53)
HUCS = ["1605010203", "1605010205", "1605010206"]
POINTS = {"south fan": (-119.702819, 39.495294),
          "north fan": (-119.682656, 39.515572)}
VERSION = "statewide_v1_2"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()

# ---- epochs -> centred z fields -------------------------------------------
print("=== building epochs ===", flush=True)
pre, pre_rgb, _, pre_ids, ref = epochs.build(
    paths.raw_dir("planet", "hidden_valley", "pre_20260619"))
post, post_rgb, _, post_ids, _ = epochs.build(
    paths.raw_dir("planet", "hidden_valley", "post_20260619"), ref)

z = {}
for k in ("brightness", "msavi2"):
    zz = ch.robust_z(pre[k][0], post[k][0], pre[k][1])
    z[f"z_{k}"] = zz - np.ma.median(zz)          # centre: v0 spatial control
print(f"z fields centred; grid {ref['width']}x{ref['height']} "
      f"({ref['crs']})", flush=True)

# ---- segments -> corridors -------------------------------------------------
prod = paths.products_dir("prefire", VERSION)
bbox5070 = transform_bounds("EPSG:4326", "EPSG:5070", *AOI)
parts = [pyogrio.read_dataframe(prod / f"{h}_segments.gpkg", bbox=bbox5070,
                                columns=["Area_km2"]) for h in HUCS]
seg = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True),
                       geometry="geometry", crs=parts[0].crs)
seg = seg.to_crs(ref["crs"])
seg = seg[seg.intersects(box(*transform_bounds("EPSG:4326", ref["crs"],
                                               *AOI)))].reset_index(drop=True)
print(f"{len(seg):,} pfdf segments in the window", flush=True)

labels = co.corridor_raster(seg, ref)
stats = co.segment_stats(labels, z)
stats = stats.reindex(range(len(seg)))
seg = seg.join(stats)
raw_cls = co.classify(stats)
seg["response_raw"] = raw_cls
seg["response"] = co.continuity_demote(seg, raw_cls)
demoted = int((seg["response"] != seg["response_raw"]).sum())
print(f"continuity vote demoted {demoted} isolated segments", flush=True)
counts = seg["response"].value_counts().to_dict()
print("classes:", {co.CLASSES[k]: int(v) for k, v in sorted(counts.items())},
      flush=True)

gpkg = OUT / f"hv_segments_response_{EVENT}.gpkg"
seg.to_file(gpkg, driver="GPKG")

# ---- Dolan-style map -------------------------------------------------------
from firescape import relief
import rasterio

with rasterio.open(paths.raw_dir("planet", "hidden_valley", "pre_20260619")
                   .glob("*_SR_*.tif").__next__()) as s0:
    pass  # grid == ref already

tr = ref["transform"]
extent = (tr.c, tr.c + tr.a * ref["width"],
          tr.f + tr.e * ref["height"], tr.f)
hs = relief.shaded_relief(
    sorted((paths.cache_root() / "3dep_tiles").glob("USGS_13_*.tif")),
    tr, (ref["height"], ref["width"]), crs=str(ref["crs"]), shade_res_m=10.0)

STYLE = {0: dict(color="white", lw=0.5, z=4),
         1: dict(color="#F5D000", lw=1.6, z=5),
         3: dict(color="#C1272D", lw=2.2, z=6)}
fig, ax = plt.subplots(figsize=(11, 11), dpi=150)
ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=extent, zorder=0)
for cls, st in STYLE.items():
    sub = seg[seg["response"] == cls]
    if len(sub):
        sub.plot(ax=ax, color=st["color"], linewidth=st["lw"],
                 zorder=st["z"],
                 alpha=0.9 if cls else 0.55)
import pyproj
t = pyproj.Transformer.from_crs("EPSG:4326", ref["crs"], always_xy=True)
for name, (lon, lat) in POINTS.items():
    x, y = t.transform(lon, lat)
    ax.plot(x, y, "o", ms=13, mfc="none", mec="#00A0B0", mew=2.5, zorder=9)
    ax.annotate(name, (x, y), xytext=(10, -14), textcoords="offset points",
                color="#00A0B0", fontsize=11, fontweight="bold", zorder=9)
ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
ax.set_axis_off()
ax.legend(handles=[
    Line2D([], [], color="white", lw=1.5, label=f"no erosion ({counts.get(0, 0):,})"),
    Line2D([], [], color="#F5D000", lw=2.5, label=f"fluvial erosion ({counts.get(1, 0):,})"),
    Line2D([], [], color="#C1272D", lw=3.0, label=f"debris flow ({counts.get(3, 0):,})"),
], loc="lower left", framealpha=0.9, fontsize=10)
ax.set_title(f"Hidden Valley {EVENT} — per-segment response from corridor-"
             "integrated z(ΔBrightness)\nprovisional thresholds "
             "(fluvial ≥ 1.0σ, debris flow ≥ 2.5σ); pfdf network stops at "
             "the valley mask, so fans are not decision units yet",
             fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "hv_segment_response_map.png", bbox_inches="tight")
plt.close(fig)

# ---- summary ---------------------------------------------------------------
summary = {
    "event": EVENT, "n_segments": int(len(seg)),
    "classes": {co.CLASSES[k]: int(v) for k, v in sorted(counts.items())},
    "thresholds": {"fluvial_t": 1.0, "df_t": 2.5,
                   "field": "z_brightness_mean (background-centred)"},
    "pre_scenes": pre_ids, "post_scenes": post_ids,
    "corridor_pixels": int((labels > 0).sum()),
    "outputs": {"gpkg": gpkg.name, "map": "hv_segment_response_map.png"},
}
(OUT / "hv_corridor_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
