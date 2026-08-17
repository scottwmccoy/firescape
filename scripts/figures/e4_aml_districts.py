"""District sheets for the AML exposure ranking -> figures/aml_districts_v2.pdf.

Two data-driven windows around the top-ranked clusters (split at 40 N), each
showing the mechanism the statewide dots compress: the hazard network colored
by likelihood, the corridors delivering into named NHD receptor waters, BLM
holdings shaded, and the ranked waste sites numbered.

Three sheet-scale rules this file needs, all of which now live in the
libraries rather than here:

1. **Shade finer than the display grid** — ``plotting.hillshade`` sizes the
   shading resolution to the window (relief.py rule 3) and keys its cache on
   the grid origin, so two panels cannot swap terrain.

2. **Rasterize the dense layers.** The network is ~380k vector line segments
   in the northern panel. Left as vectors the sheet was 57 MB and slow to
   open; rasterized at the save dpi it is 4 MB, and the linework a reader
   zooms into (text, markers, leaders) stays vector. One keyword per artist,
   so it stays a convention rather than a wrapper.

3. **Place labels against a collision list** — ``stormscape.plot.Labeller``,
   which moves a label that would land on another one or run off the frame
   and connects it with a leader once it has moved far enough to be
   ambiguous.
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
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterio.warp import transform_bounds
from stormscape.plot import Labeller, interior_point, shorten

from firescape import paths
from firescape import plotting as mc

plt.rcParams["image.cmap"] = "viridis"
V12 = paths.products_dir("prefire", "statewide_v1_2")
TOPN, PADDEG, NPX = 20, 0.05, 1900
P_FLOOR, P_CEIL = 0.10, 0.60      # network color range; below the floor is quiet
CROWD_PX = 26.0                   # a neighbour this close makes a label ambiguous


sites = gpd.read_file(
    paths.products_dir("exposure", "aml_v2") / "aml_exposure.gpkg").to_crs(4326)
sites["rep"] = sites.geometry.representative_point()
receptors = gpd.read_file(paths.interim_dir("exposure") / "nhd_receptors.gpkg")
blm = gpd.read_file(paths.raw_dir("blm") / "nv_blm_sma.gpkg").to_crs(4326)
ctx = mc.fetch_context()          # statewide cache; clipped per window below
ubounds = json.loads(
    (paths.interim_dir("exposure") / "unit_bounds_v1_2.json").read_text())

top = sites.nsmallest(TOPN, "rank")
vmax = float(np.nanpercentile(
    sites.loc[sites["exposed"].astype(bool), "P_annual_site"], 99))

panels = [(title, mc.square_window(top[sel].total_bounds, pad=PADDEG))
          for title, sel in (("North-central Nevada districts", top["rep"].y >= 40),
                           ("Comstock–Virginia Range", top["rep"].y < 40))
          if len(top[sel])]

PANEL_IN, LEFT, GAP, RIGHT, TOPM, BOTM = 6.2, 0.42, 0.30, 0.42, 0.62, 1.25
FW = LEFT + len(panels) * PANEL_IN + (len(panels) - 1) * GAP + RIGHT
FH = TOPM + PANEL_IN + BOTM
fig = plt.figure(figsize=(FW, FH), dpi=150)
gs = fig.add_gridspec(1, len(panels), left=LEFT / FW, right=1 - RIGHT / FW,
                      top=1 - TOPM / FH, bottom=BOTM / FH,
                      wspace=GAP / PANEL_IN)

net = None
for i, (label, win) in enumerate(panels):
    ax = fig.add_subplot(gs[0, i])
    tr, shape, extent = mc.grid(win, res=(win[2] - win[0]) / NPX)
    hs = mc.hillshade(tr, shape)
    print(f"{label}: {shape[1]}x{shape[0]} px, "
          f"{mc.grid_res_m(tr, shape):.0f} m/px, shaded at "
          f"{mc.shade_resolution(tr, shape):.0f} m", flush=True)
    ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=extent, zorder=0)
    bl = blm.clip(tuple(win))
    if len(bl):
        bl.plot(ax=ax, facecolor="#D9C98C", edgecolor="none", alpha=0.28,
                zorder=1, rasterized=True)
    cc = ctx["counties"].clip(tuple(win))
    if len(cc):
        cc.boundary.plot(ax=ax, color="0.35", linewidth=0.35, alpha=0.7,
                         zorder=2, rasterized=True)

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
    segs = segs[segs["P_24mmh"] >= P_FLOOR].sort_values("P_24mmh")
    print(f"  {len(segs)} network segments drawn (rasterized)", flush=True)
    net = ax.add_collection(LineCollection(
        [np.asarray(g.coords) for g in segs.geometry],
        array=segs["P_24mmh"].to_numpy(), cmap="magma_r",
        clim=(P_FLOOR, P_CEIL),
        linewidths=np.clip(0.2 + 0.3 * np.sqrt(segs["Area_km2"]), 0.2, 1.1),
        alpha=0.75, zorder=3, rasterized=True))

    # Receptors: at or below the weight of the widest hazard channel (1.1),
    # so the water reads as context and the hazard stays the subject.
    rec = receptors.cx[win[0]:win[2], win[1]:win[3]]
    wbod = rec[rec["src"] == "wb"]
    per = rec[(rec["src"] == "fl") & (rec["kind"] == "perennial stream")]
    oth = rec[(rec["src"] == "fl") & (rec["kind"] != "perennial stream")]
    if len(wbod):
        wbod.plot(ax=ax, facecolor="#9ECAE1", edgecolor="#0072B2",
                  linewidth=0.4, alpha=0.75, zorder=4, rasterized=True)
    if len(oth):
        oth.plot(ax=ax, color="#56B4E9", linewidth=0.55,
                 linestyle=(0, (4, 2)), zorder=5, rasterized=True)
    if len(per):
        per.plot(ax=ax, color="#0072B2", linewidth=1.0, zorder=5,
                 rasterized=True)

    box = sites.cx[win[0]:win[2], win[1]:win[3]]
    clear = box[box["dist_m"].isna()]
    near = box[(~box["exposed"].astype(bool)) & box["dist_m"].notna()]
    ex = box[box["exposed"].astype(bool)]
    ax.scatter([p.x for p in clear["rep"]], [p.y for p in clear["rep"]],
               s=6, marker=".", color="#777777", zorder=9, linewidths=0)
    ax.scatter([p.x for p in near["rep"]], [p.y for p in near["rep"]], s=16,
               marker="o", facecolors="none", edgecolors="#5D5D5D",
               linewidths=0.7, zorder=9)
    ax.scatter([p.x for p in ex["rep"]], [p.y for p in ex["rep"]], s=34,
               c=ex["P_annual_site"], vmin=0, vmax=vmax, marker="o",
               edgecolors="white", linewidths=0.5, zorder=10)
    places = ctx["places"].clip(tuple(win))
    if len(places):
        ax.scatter(places.geometry.x, places.geometry.y, s=11, color="black",
                   edgecolor="white", linewidth=0.5, zorder=9)

    mc.style_axes(ax, extent, step=mc.tick_step(win[2] - win[0]))
    if i:
        # Degree labels on the outside edge, so the panels can sit close
        # together without the right panel's labels landing on the left
        # panel's frame.
        ax.yaxis.tick_right()
    ax.set_title(f"{label} — {int(ex['on_blm'].sum())} of {len(ex)} "
                 f"exposed sites on BLM land", fontsize=10)
    fig.canvas.draw()          # transforms must be final before labels land

    # Labels: towns, then the waters they sit on, then the ranked sites, all
    # sharing one collision list so a number cannot land on a river name.
    lab = Labeller(ax)
    lab.block_many([p.x for p in box["rep"]], [p.y for p in box["rep"]])
    for _, row in places.iterrows():
        lab.label(row.geometry.x, row.geometry.y, shorten(row["name"]),
                  fontsize=7.0, color="black", weight="bold", zorder=11)
    named = rec[rec["name"].notna()].copy()
    if len(named):
        named["L"] = named.geometry.length
        ranked = sorted(named.groupby("name"), key=lambda kv: -kv[1]["L"].sum())
        for nm, grp in ranked[:8]:
            p = interior_point(grp.loc[grp["L"].idxmax()].geometry, win)
            if p:
                lab.label(p[0], p[1], shorten(nm), fontsize=6.5,
                          color="#0072B2", style="italic", zorder=11)
    dropped = 0
    for _, r in ex[ex["rank"] <= TOPN].sort_values("rank").iterrows():
        if not lab.label(r["rep"].x, r["rep"].y, f"{int(r['rank'])}",
                         fontsize=8.5, color="#C1272D", weight="bold",
                         zorder=12,
                         force_leader=lab.crowded(r["rep"].x, r["rep"].y,
                                                  radius_px=CROWD_PX)):
            dropped += 1
    if dropped:
        print(f"  {dropped} site labels had nowhere to go", flush=True)

cax = fig.add_axes([0.5 - 0.13, (BOTM - 0.42) / FH, 0.26, 0.012])
cb = fig.colorbar(net, cax=cax, orientation="horizontal")
cb.set_label("segment debris-flow likelihood (I15 = 24 mm/h)", fontsize=8,
             labelpad=2)
cb.ax.tick_params(labelsize=7, pad=1.5)
fig.legend(handles=[
    Line2D([0], [0], color="#0072B2", lw=1.0, label="perennial stream (NHD)"),
    Line2D([0], [0], color="#56B4E9", lw=0.55, linestyle=(0, (4, 2)),
           label="named intermittent / canal"),
    Patch(facecolor="#9ECAE1", edgecolor="#0072B2", label="named waterbody"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#3B528B",
           markeredgecolor="white", markersize=7,
           label="waste site in a delivery corridor"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
           markeredgecolor="#5D5D5D", markersize=6,
           label="within 1 km of one"),
    Patch(facecolor="#D9C98C", alpha=0.5, label="BLM-managed land"),
], loc="lower center", bbox_to_anchor=(0.5, 0.022), ncol=6, fontsize=8,
    frameon=False)
fig.suptitle("Where the top-ranked AML sites meet the water: hazard network, "
             "delivery corridors, and named receptors", fontsize=12, y=0.975)
fig.text(0.5, 0.006,
         "Site color = annual hit probability; numbers are the statewide "
         "exposure rank. Delivery is a proximity proxy (corridor width "
         "9–60 m from contributing area), not a runout model.",
         ha="center", fontsize=6.5, color="#444444")
mc.save(fig, "aml_districts_v2")
print("saved figures/aml_districts_v2.pdf")
