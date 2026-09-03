"""Statewide pre-fire hazard with the observed debris-flow inventory on top.

Answers one question at state scale: do the places where a debris flow has
actually happened sit on ground firescape already predicts as elevated
hazard? ``p12_urban_zoom_forecast --key`` answers the same question at
channel scale for the five windows dense enough to be worth it (55 points,
26 of them fall inside those five); this sheet is the one panel wide enough
to show all 55 at once, so a reader can see the state-scale pattern before
going to a zoom for the ground-truth-vs-model detail.

Combined hazard class, not likelihood or the threshold: it is what the five
zoom sheets show in their own second panel, and it is the one field simple
enough to read at state scale without a legend for every shade.

    python scripts/figures/p16_debrisflow_inventory_statewide.py [calibration]
"""
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from pyogrio import read_dataframe

import geopandas as gpd

from firescape import paths, plotting as mc

VERSION = sys.argv[1] if len(sys.argv) > 1 else "statewide_v1_2"
OUT = paths.products_dir("prefire", VERSION)
INVENTORY_PATH = paths.raw_dir("debrisflow_inventory") / "nv_debrisflow_inventory.geojson"

basins = read_dataframe(OUT / f"{VERSION}_basins.gpkg",
                        columns=["H_24mmh"]).to_crs("EPSG:4326")
nv = gpd.read_file(paths.raw_dir("boundaries") / "nv_state.geojson").to_crs("EPSG:4326")
inv = gpd.read_file(INVENTORY_PATH)
print(f"{len(basins):,} basins · {len(inv)} inventory points "
      f"({int((inv['category'] == 'debris_flow').sum())} debris flow, "
      f"{int((inv['category'] != 'debris_flow').sum())} other process)", flush=True)

transform, shape, extent = mc.grid(mc.domain_bounds4326())
print(f"display grid {shape[1]}x{shape[0]} @ {mc.RES_DEG:g} deg", flush=True)
hs = mc.hillshade(transform, shape)
ctx = mc.fetch_context()
print("context layers ready", flush=True)

HAZ = ListedColormap(["#4B9B6E", "#E8A33D", "#C1272D"])   # low / mod / high
NORM = BoundaryNorm([0.5, 1.5, 2.5, 3.5], HAZ.N)
im_extent = tuple(extent)

fig, ax = plt.subplots(figsize=(13, 11), dpi=150)
ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=im_extent)
arr = mc.burn(basins, "H_24mmh", transform, shape)
ax.imshow(np.ma.masked_invalid(arr), cmap=HAZ, norm=NORM, extent=im_extent,
          alpha=0.6, interpolation="nearest")
mc.draw_context(ax, ctx, extent=extent)
nv.boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=8)

df_pts = inv[inv["category"] == "debris_flow"]
other_pts = inv[inv["category"] != "debris_flow"]
# Same white triangle the zoom sheets use, at a size that survives the
# 4x coarser statewide display grid without disappearing into the hillshade.
if len(df_pts):
    ax.scatter(df_pts.geometry.x, df_pts.geometry.y, marker="^", s=48,
              facecolor="white", edgecolor="black", linewidth=0.8, zorder=9.6)
if len(other_pts):
    ax.scatter(other_pts.geometry.x, other_pts.geometry.y, marker="^", s=48,
              facecolor="none", edgecolor="black", linewidth=1.0, zorder=9.6)

ax.legend(handles=[
    *(Line2D([0], [0], marker="s", color="none", markersize=11,
            markerfacecolor=c, label=l)
      for c, l in zip(HAZ.colors, ["low", "moderate", "high"])),
    Line2D([0], [0], marker="^", color="none", markerfacecolor="white",
          markeredgecolor="black", markersize=9,
          label=f"observed debris flow (n={len(df_pts)})"),
    Line2D([0], [0], marker="^", color="none", markerfacecolor="none",
          markeredgecolor="black", markersize=9,
          label=f"observed, other process (n={len(other_pts)})"),
], title="Hazard class · inventory", loc="lower left", fontsize=8, title_fontsize=8)
mc.style_axes(ax, extent)

dated = pd.to_datetime(inv["obs_date"], errors="coerce").dropna()
year_span = f"{dated.dt.year.min()}–{dated.dt.year.max()}" if len(dated) else "undated"

# Margins first, then the caption: caption() reserves its own space by
# growing the bottom margin from wherever it already is, so it has to run
# AFTER the baseline is set or its reservation gets overwritten (see p15).
fig.subplots_adjust(top=0.97, bottom=0.02, left=0.02, right=0.98)
mc.caption(fig, (
    f"Combined hazard class (Cannon et al. 2010) at the 24 mm/h design storm, conditional on a "
    f"basin burning, against the statewide {VERSION} pre-fire surface ({len(basins):,} basins, "
    f"591 HU10 watersheds). Triangles are the hand-digitized Nevada debris-flow inventory "
    f"(nv_debrisflow_inventory, {year_span}) -- not the model's own output, the closest thing to "
    f"ground truth available at this scale. Five clusters dense enough for a channel-scale look "
    f"sit inside existing zoom windows (Las Vegas, Lincoln-Caliente, Ely-White Pine, "
    f"Winnemucca-Battle Mountain, and Railroad Valley, added for this inventory) -- see "
    f"zoom_<key>_forecast_{VERSION}.pdf for those. A point landing on green ground is a debris "
    f"flow the pre-fire surface would not have flagged; that is information about the surface, "
    f"not an error in the inventory."
), label="Nevada debris-flow inventory.", fontsize=9.5)

mc.save(fig, f"debrisflow_inventory_{VERSION}")
plt.close(fig)
print(f"wrote figures/debrisflow_inventory_{VERSION}.pdf", flush=True)
