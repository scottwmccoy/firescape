"""Hidden Valley 2026-06-19: fan-deposit detection beyond the network.

The fan stage's first real run. Acceptance test is concrete: the two fan
sites Scott reported must fall inside (or within ~100 m of) detected
polygons. Everything else the detector returns is either more of the event
or a false positive the connectivity stage will need to handle -- both
worth seeing.

Run:  /opt/anaconda3/envs/FireMan/bin/python p16_hv_fans.py [outdir]
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
from rasterio.warp import transform_bounds
from shapely.geometry import Point, box

import geopandas as gpd

from firescape import change as ch, corridor as co, epochs, fans, paths, relief

EVENT = "2026-06-19"
AOI = (-119.72, 39.45, -119.63, 39.53)
HUCS = ["1605010203", "1605010205", "1605010206"]
POINTS = {"south fan": (-119.702819, 39.495294),
          "north fan": (-119.682656, 39.515572)}
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()

print("=== epochs ===", flush=True)
pre, pre_rgb, _, _, ref = epochs.build(
    paths.raw_dir("planet", "hidden_valley", "pre_20260619"),
    indices=("brightness",), rgb=False, keep_nir=False, verbose=False)
post, _, _, _, _ = epochs.build(
    paths.raw_dir("planet", "hidden_valley", "post_20260619"), ref,
    indices=("brightness",), rgb=False, keep_nir=False, verbose=False)
z = ch.robust_z(pre["brightness"][0], post["brightness"][0],
                pre["brightness"][1])
z = z - np.ma.median(z)
print("z built", flush=True)

print("=== terrain ===", flush=True)
tiles = sorted((paths.cache_root() / "3dep_tiles").glob("USGS_13_*.tif"))
dem, slope = fans.dem_and_slope(tiles, ref)

prod = paths.products_dir("prefire", "statewide_v1_2")
bbox5070 = transform_bounds("EPSG:4326", "EPSG:5070", *AOI)
seg = gpd.GeoDataFrame(pd.concat(
    [pyogrio.read_dataframe(prod / f"{h}_segments.gpkg", bbox=bbox5070,
                            columns=["Area_km2"]) for h in HUCS],
    ignore_index=True), geometry="geometry", crs="EPSG:5070").to_crs(ref["crs"])

channels = fans.channel_mask(seg, ref)
hand, dist = fans.hand_lite(dem, channels, pixel_m=abs(ref["transform"].a))
zone = fans.fan_zone(hand, slope, dist)
print(f"fan zone: {zone.mean():.1%} of window", flush=True)

det = fans.detect_fans(z, zone, ref, z_t=2.5, min_area_px=30)
print(f"{len(det)} candidate deposits", flush=True)

# acceptance: known fans recovered?
import pyproj
t = pyproj.Transformer.from_crs("EPSG:4326", ref["crs"], always_xy=True)
hits = {}
if len(det):
    u = det.union_all()
    for name, (lon, lat) in POINTS.items():
        p = Point(*t.transform(lon, lat))
        hits[name] = round(float(p.distance(u)), 1)
        print(f"  {name}: {hits[name]} m from nearest detected deposit")

det.to_file(OUT / f"hv_fans_{EVENT}.gpkg", driver="GPKG")

# ---- map -------------------------------------------------------------------
tr = ref["transform"]
extent = (tr.c, tr.c + tr.a * ref["width"],
          tr.f + tr.e * ref["height"], tr.f)
hs = relief.shaded_relief(tiles, tr, (ref["height"], ref["width"]),
                          crs=str(ref["crs"]), shade_res_m=10.0)
fig, ax = plt.subplots(figsize=(11, 11), dpi=150)
ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=extent, zorder=0)
zm = np.ma.masked_where(~zone, np.ones_like(np.asarray(z), np.float32))
ax.imshow(zm.filled(np.nan), cmap="Blues", vmin=0, vmax=3, alpha=0.25,
          extent=extent, zorder=1)
seg.plot(ax=ax, color="white", linewidth=0.4, alpha=0.6, zorder=2)
if len(det):
    det.plot(ax=ax, facecolor="#C1272D", edgecolor="#7A0D12",
             linewidth=0.6, alpha=0.75, zorder=5)
for name, (lon, lat) in POINTS.items():
    x, y = t.transform(lon, lat)
    ax.plot(x, y, "o", ms=13, mfc="none", mec="#00A0B0", mew=2.5, zorder=9)
    ax.annotate(f"{name} ({hits.get(name, float('nan')):.0f} m)", (x, y),
                xytext=(10, -14), textcoords="offset points",
                color="#00A0B0", fontsize=11, fontweight="bold", zorder=9)
ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
ax.set_axis_off()
ax.set_title(f"Hidden Valley {EVENT} — fan-stage v0: detected deposits (red) "
             "in the fan zone (blue tint)\nHAND-lite < 5 m ∧ slope 1.5–15° ∧ "
             "≤ 500 m of a channel; z ≥ 2.5, ≥ 270 m²", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "hv_fans_map.png", bbox_inches="tight")
plt.close(fig)

summary = {
    "event": EVENT, "n_deposits": int(len(det)),
    "total_deposit_area_m2": float(det["area_m2"].sum()) if len(det) else 0.0,
    "largest_m2": float(det["area_m2"].max()) if len(det) else 0.0,
    "distance_to_known_fans_m": hits,
    "fan_zone_fraction": round(float(zone.mean()), 4),
    "params": {"z_t": 2.5, "min_area_px": 30, "hand_max": fans.HAND_MAX_M,
               "slope_deg": [fans.SLOPE_MIN_DEG, fans.SLOPE_MAX_DEG],
               "dist_max_m": fans.CHANNEL_DIST_MAX_M},
}
(OUT / "hv_fans_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
