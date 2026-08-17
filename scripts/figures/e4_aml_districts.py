"""District sheets for the AML exposure ranking -> figures/aml_districts_v2.pdf.

Two data-driven windows around the top-ranked clusters (split at 40 N: the
Winnemucca-area districts vs the Comstock-Virginia Range), each showing the
mechanism the statewide dots compress: the hazard network colored by
likelihood, the corridors delivering into named NHD receptor waters, BLM
holdings shaded, and the ranked waste sites numbered.
"""
import json
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke
from rasterio.warp import transform_bounds

from firescape import paths
from firescape import plotting as mc

plt.rcParams["image.cmap"] = "viridis"
V12 = paths.products_dir("prefire", "statewide_v1_2")
TOPN, PADDEG = 20, 0.05

sites = gpd.read_file(
    paths.products_dir("exposure", "aml_v2") / "aml_exposure.gpkg"
).to_crs(4326)
sites["rep"] = sites.geometry.representative_point()
receptors = gpd.read_file(paths.interim_dir("exposure") / "nhd_receptors.gpkg")
blm = gpd.read_file(paths.raw_dir("blm") / "nv_blm_sma.gpkg").to_crs(4326)
ubounds = json.loads((paths.interim_dir("exposure")
                      / "unit_bounds_v1_2.json").read_text())

top = sites.nsmallest(TOPN, "rank")
vmax = float(np.nanpercentile(
    sites.loc[sites["exposed"].astype(bool), "P_annual_site"], 99))

panels = []
for label, sel in (("North-central Nevada districts", top["rep"].y >= 40.0),
                   ("Comstock–Virginia Range", top["rep"].y < 40.0)):
    members = top[sel]
    if not len(members):
        continue
    w, s, e, n = members.total_bounds
    w, s, e, n = w - PADDEG, s - PADDEG, e + PADDEG, n + PADDEG
    dx, dy = e - w, n - s                       # roughly square windows
    if dx < dy:
        w, e = (w + e - dy) / 2, (w + e + dy) / 2
    else:
        s, n = (s + n - dx) / 2, (s + n + dx) / 2
    panels.append((label, (w, s, e, n)))

fig, axes = plt.subplots(1, len(panels), figsize=(8.2 * len(panels), 9),
                         dpi=150)
axes = np.atleast_1d(axes)
for ax, (label, win) in zip(axes, panels):
    tr, shape, extent = mc.grid(win, res=(win[2] - win[0]) / 1100)
    ax.imshow(mc.hillshade(tr, shape), cmap="gray", vmin=0, vmax=1,
              extent=extent)
    blm.clip(tuple(win)).plot(ax=ax, facecolor="#D9C98C", edgecolor="none",
                              alpha=0.28, zorder=1)

    # hazard network from the per-unit v1.2 segments intersecting the window
    win5070 = transform_bounds("EPSG:4326", "EPSG:5070", *win)
    segs = []
    for huc, b in ubounds.items():
        if not (b[2] < win5070[0] or b[0] > win5070[2]
                or b[3] < win5070[1] or b[1] > win5070[3]):
            g = pyogrio.read_dataframe(V12 / f"{huc}_segments.gpkg",
                                       columns=["P_24mmh", "Area_km2"],
                                       bbox=tuple(win5070))
            if len(g):
                segs.append(g)
    segs = pd.concat(segs).to_crs(4326).explode(index_parts=False)
    segs = segs[segs.geometry.type == "LineString"]
    # only the consequential part of the network, or the terrain drowns
    segs = segs[segs["P_24mmh"] >= 0.10].sort_values("P_24mmh")
    net = ax.add_collection(matplotlib.collections.LineCollection(
        [np.asarray(g.coords) for g in segs.geometry],
        array=segs["P_24mmh"].to_numpy(), cmap="magma_r", clim=(0.1, 0.6),
        linewidths=np.clip(0.2 + 0.3 * np.sqrt(segs["Area_km2"]), 0.2, 1.1),
        alpha=0.75, zorder=3))

    rec = receptors.cx[win[0]:win[2], win[1]:win[3]]
    wbod = rec[rec["src"] == "wb"]
    per = rec[(rec["src"] == "fl") & (rec["kind"] == "perennial stream")]
    oth = rec[(rec["src"] == "fl") & (rec["kind"] != "perennial stream")]
    if len(wbod):
        wbod.plot(ax=ax, facecolor="#9ECAE1", edgecolor="#0072B2",
                  linewidth=0.5, alpha=0.75, zorder=4)
    if len(oth):
        oth.plot(ax=ax, color="#56B4E9", linewidth=0.9,
                 linestyle=(0, (4, 2)), zorder=5)
    if len(per):
        per.plot(ax=ax, color="#0072B2", linewidth=1.6, zorder=5)
    named = rec[rec["name"].notna()].copy()
    if len(named):
        named["L"] = named.geometry.length
        for nm, grp in sorted(named.groupby("name"), key=lambda kv: -kv[1]["L"].sum())[:5]:
            p = grp.loc[grp["L"].idxmax()].geometry.representative_point()
            ax.annotate(nm, (p.x, p.y), fontsize=7, style="italic",
                        color="#0072B2", zorder=11,
                        path_effects=[withStroke(linewidth=2,
                                                 foreground="white")])

    stat = {"clear": sites["dist_m"].isna(),
            "near": (~sites["exposed"].astype(bool)) & sites["dist_m"].notna()}
    box = sites.cx[win[0]:win[2], win[1]:win[3]]
    for key, mk, sz in (("clear", ".", 6), ("near", "o", 16)):
        ss = box[stat[key].reindex(box.index)]
        ax.scatter([p.x for p in ss["rep"]], [p.y for p in ss["rep"]], s=sz,
                   marker=mk, facecolors="none" if key == "near" else "#777777",
                   edgecolors="#5D5D5D" if key == "near" else "none",
                   linewidths=0.7, zorder=9)
    ex = box[box["exposed"].astype(bool)]
    ax.scatter([p.x for p in ex["rep"]], [p.y for p in ex["rep"]], s=34,
               c=ex["P_annual_site"], vmin=0, vmax=vmax, marker="o",
               edgecolors="white", linewidths=0.5, zorder=10)
    for _, r in ex[ex["rank"] <= TOPN].iterrows():
        ax.annotate(f"{int(r['rank'])}", (r["rep"].x, r["rep"].y),
                    xytext=(6, 5), textcoords="offset points", fontsize=9,
                    fontweight="bold", color="#C1272D", zorder=12,
                    path_effects=[withStroke(linewidth=2.2,
                                             foreground="white")])
    mc.style_axes(ax, extent, step=0.2)
    ax.set_title(f"{label} — {int(ex['on_blm'].sum())} of "
                 f"{len(ex)} exposed sites on BLM land", fontsize=10)

cb = fig.colorbar(net, ax=axes[-1], shrink=0.4, pad=0.01)
cb.set_label("segment debris-flow likelihood (I15 = 24 mm/h)", fontsize=8)
cb.ax.tick_params(labelsize=7)
axes[0].legend(handles=[
    Line2D([0], [0], color="#0072B2", lw=1.6, label="perennial stream (NHD)"),
    Line2D([0], [0], color="#56B4E9", lw=1.0, linestyle=(0, (4, 2)),
           label="named intermittent / canal"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#3B528B",
           markeredgecolor="white", markersize=7, label="waste site in a corridor"),
    matplotlib.patches.Patch(facecolor="#D9C98C", alpha=0.5,
                             label="BLM-managed land"),
], loc="lower left", fontsize=7.5, framealpha=0.85)
fig.suptitle("Where the top-ranked AML sites meet the water: hazard network, "
             "delivery corridors, and named receptors", fontsize=12, y=0.98)
fig.tight_layout(rect=(0, 0, 1, 0.97))
mc.save(fig, "aml_districts_v2")
print("saved figures/aml_districts_v2.pdf")
