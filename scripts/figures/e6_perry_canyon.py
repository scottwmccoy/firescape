"""Perry Canyon pilot sheet -> figures/perry_canyon_openings.pdf.

The district the statewide waste ranking cannot see. USMIN maps no dump or
tailings extent here, so this draws the underground workings instead — adits
as triangles, shafts as squares — over the modelled debris-flow network.

Workings are labelled by **rank, not by name**. The obvious guess was that the
named mines carry the waste-rock piles; Scott's ground knowledge falsified it
(the largest pile is at an unnamed adit low in the drainage, and the named
Jones Kincaid shaft has little), so privileging names on the sheet would draw
the eye to the wrong workings. Names are still drawn where they exist, in
grey, as an attribute rather than as evidence.

Same grammar as the district zooms (``e4_aml_zooms``): named water first, the
modelled channels over it, BLM shaded, labels placed against a collision list.
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
from shapely.geometry import LineString
from stormscape.plot import Labeller, interior_point, shorten

from firescape import paths
from firescape import plotting as mc

plt.rcParams["image.cmap"] = "viridis"
V12 = paths.products_dir("prefire", "statewide_v1_2")
EXPO = paths.products_dir("exposure", "perry_canyon_v1")
AXIS = LineString([(-119.60927, 39.86253), (-119.56104, 39.82705)])
P_FLOOR, P_CEIL = 0.05, 0.40
NPX, MIN_M_PER_PX = 1900, 8.0
NLABEL = 15
MARK = {"Adit": "^", "Mine Shaft": "s"}

work = gpd.read_file(EXPO / "perry_openings.gpkg", layer="workings").to_crs(4326)
receptors = gpd.read_file(paths.interim_dir("exposure") / "nhd_receptors.gpkg")
blm = gpd.read_file(paths.raw_dir("blm") / "nv_blm_sma.gpkg").to_crs(4326)
ctx = mc.fetch_context()
ub = json.loads(
    (paths.interim_dir("exposure") / "unit_bounds_v1_2.json").read_text())
PER_NAMES = set(receptors.loc[(receptors["src"] == "fl")
                              & (receptors["kind"] == "perennial stream"),
                              "name"].dropna())

b = work.total_bounds
pad = 1200.0 / 111_320.0
win = mc.square_window((b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad))
span_m = (win[2] - win[0]) * 111_320 * np.cos(np.radians((win[1] + win[3]) / 2))
npx = int(min(NPX, max(600, span_m / MIN_M_PER_PX)))
tr, shape, extent = mc.grid(win, res=(win[2] - win[0]) / npx)

PANEL, LEFT, RIGHT, TOPM, BOTM = 7.0, 0.55, 0.30, 0.92, 1.60
FW, FH = LEFT + PANEL + RIGHT, TOPM + PANEL + BOTM
fig = plt.figure(figsize=(FW, FH), dpi=150)
ax = fig.add_axes([LEFT / FW, BOTM / FH, PANEL / FW, PANEL / FH])
ax.imshow(mc.hillshade(tr, shape), cmap="gray", vmin=0, vmax=1, extent=extent,
          zorder=0)
print(f"{shape[1]}x{shape[0]} px, {mc.grid_res_m(tr, shape):.0f} m/px, "
      f"shaded at {mc.shade_resolution(tr, shape):.0f} m", flush=True)

bl = blm.clip(tuple(win))
if len(bl):
    bl.plot(ax=ax, facecolor="#D9C98C", edgecolor="none", alpha=0.30, zorder=1,
            rasterized=True)

rec = receptors.cx[win[0]:win[2], win[1]:win[3]]
fl = rec[rec["src"] == "fl"]
is_per = (fl["kind"] == "perennial stream") | fl["name"].isin(PER_NAMES)
if len(rec[rec["src"] == "wb"]):
    rec[rec["src"] == "wb"].plot(ax=ax, facecolor="#9ECAE1",
                                 edgecolor="#0072B2", linewidth=0.5,
                                 alpha=0.75, zorder=2.0, rasterized=True)
if (~is_per).any():
    fl[~is_per].plot(ax=ax, color="#56B4E9", linewidth=0.9,
                     linestyle=(0, (4, 2)), zorder=2.2, rasterized=True)
if is_per.any():
    fl[is_per].plot(ax=ax, color="#0072B2", linewidth=1.8, zorder=2.4,
                    rasterized=True)

win5070 = transform_bounds("EPSG:4326", "EPSG:5070", *win)
segs = []
for huc, bb in ub.items():
    if not (bb[2] < win5070[0] or bb[0] > win5070[2]
            or bb[3] < win5070[1] or bb[1] > win5070[3]):
        g = pyogrio.read_dataframe(V12 / f"{huc}_segments.gpkg",
                                   columns=["P_24mmh", "Area_km2"],
                                   bbox=tuple(win5070))
        if len(g):
            segs.append(g)
segs = pd.concat(segs).to_crs(4326).explode(index_parts=False)
segs = segs[(segs.geometry.type == "LineString")
            & (segs["P_24mmh"] >= P_FLOOR)].sort_values("P_24mmh")
net = ax.add_collection(LineCollection(
    [np.asarray(g.coords) for g in segs.geometry],
    array=segs["P_24mmh"].to_numpy(), cmap="magma_r", clim=(P_FLOOR, P_CEIL),
    linewidths=np.clip(0.5 + 0.9 * np.sqrt(segs["Area_km2"]), 0.5, 2.4),
    alpha=0.9, zorder=3.5, rasterized=True))

gpd.GeoSeries([AXIS], crs=4326).plot(ax=ax, color="#222222", linewidth=1.0,
                                     linestyle=(0, (6, 3)), zorder=4,
                                     alpha=0.65)

vmax = float(np.nanpercentile(work["P_annual_site"], 98))
for kind, marker in MARK.items():
    k = work[work["ftr_type"] == kind]
    if not len(k):
        continue
    sc = ax.scatter(k.geometry.x, k.geometry.y, s=95, marker=marker,
                    c=k["P_annual_site"], vmin=0, vmax=vmax,
                    edgecolors="white", linewidths=0.8, zorder=10)
near = work[work["exposed"].astype(bool)]
ax.scatter(near.geometry.x, near.geometry.y, s=210, marker="o",
           facecolors="none", edgecolors="#C1272D", linewidths=1.1, zorder=9.5)

mc.style_axes(ax, extent, step=mc.tick_step(win[2] - win[0]))
fig.canvas.draw()
lab = Labeller(ax)
lab.block_many(work.geometry.x, work.geometry.y, radius_px=7.0)
named = rec[rec["name"].notna()].copy()
if len(named):
    named["L"] = named.geometry.length
    for nm, grp in sorted(named.groupby("name"),
                          key=lambda kv: -kv[1]["L"].sum())[:6]:
        p = interior_point(grp.loc[grp["L"].idxmax()].geometry, win)
        if p:
            lab.label(p[0], p[1], shorten(nm), fontsize=7.5, color="#0072B2",
                      style="italic", zorder=11)
lab.label(AXIS.interpolate(0.5, normalized=True).x,
          AXIS.interpolate(0.5, normalized=True).y, "Perry Canyon",
          fontsize=9, color="#222222", style="italic", weight="bold",
          zorder=11)
drop = 0
for _, r in work.sort_values("rank").head(NLABEL).iterrows():
    if not lab.label(r.geometry.x, r.geometry.y, f"{int(r['rank'])}",
                     fontsize=9, color="#C1272D", weight="bold", zorder=12,
                     force_leader=lab.crowded(r.geometry.x, r.geometry.y,
                                              radius_px=26)):
        drop += 1
for _, r in work[work["named"].astype(bool)].sort_values("rank").iterrows():
    lab.label(r.geometry.x, r.geometry.y, shorten(str(r["name"]), 22),
              fontsize=7, color="#555555", style="italic", zorder=11.5)
if drop:
    print(f"{drop} rank labels had nowhere to go", flush=True)

cby = (BOTM - 0.46) / FH
cb = fig.colorbar(net, cax=fig.add_axes([0.10, cby, 0.34, 0.013]),
                  orientation="horizontal")
cb.set_label("channel debris-flow likelihood (I15 = 24 mm/h)", fontsize=8,
             labelpad=2)
cb.ax.tick_params(labelsize=7, pad=1.5)
cb2 = fig.colorbar(sc, cax=fig.add_axes([0.56, cby, 0.34, 0.013]),
                   orientation="horizontal")
cb2.set_label("annual hit probability at the working  "
              "(labelled as a return interval)", fontsize=8, labelpad=2)
# Probabilities of 0.002 need five significant figures to separate, which
# collides at this width and is not how the number gets quoted anyway.
ticks = [t for t in np.linspace(0, vmax, 5)[1:]]
cb2.set_ticks(ticks)
cb2.set_ticklabels([f"1 in {1/t:,.0f} yr" for t in ticks])
cb2.ax.tick_params(labelsize=7, pad=1.5)

fig.legend(handles=[
    Line2D([0], [0], marker="^", color="none", markerfacecolor="#3B528B",
           markeredgecolor="white", markersize=9, label="adit"),
    Line2D([0], [0], marker="s", color="none", markerfacecolor="#3B528B",
           markeredgecolor="white", markersize=8, label="shaft"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
           markeredgecolor="#C1272D", markersize=11,
           label="within 30–55 m of a channel"),
    Line2D([0], [0], color="#0072B2", lw=1.8, label="perennial stream (NHD)"),
    Line2D([0], [0], color="#56B4E9", lw=0.9, linestyle=(0, (4, 2)),
           label="named intermittent / canal"),
    Patch(facecolor="#D9C98C", alpha=0.5, label="BLM-managed land"),
], loc="lower center", bbox_to_anchor=(0.5, 0.020), ncol=3, fontsize=8,
    frameon=False)

meta = json.loads((EXPO / "run_meta.json").read_text())
n_near = int(work["exposed"].astype(bool).sum())
fig.suptitle("Perry Canyon, Washoe County — underground workings on the "
             "debris-flow network", fontsize=13, y=0.985)
fig.text(0.5, 0.952,
         f"USMIN maps no dump or tailings extent here, so the statewide waste "
         f"ranking is blank: these are the {len(work)} adits and shafts "
         f"instead, {meta['dedupe']['records']} quad records collapsed",
         ha="center", fontsize=9.5, color="#333333")
fig.text(0.5, 0.929,
         f"{n_near} lie within 30–55 m of a modelled channel · "
         f"{int(work['on_blm'].sum())} of {len(work)} on BLM-managed land · "
         f"most frequent 1 in {1/work['P_annual_site'].max():,.0f} yr",
         ha="center", fontsize=9.5, color="#333333")
fig.text(0.5, 0.006,
         "The asset is the opening, not the dump: a portal dump spills "
         "downslope and USMIN never mapped it, so a pile can reach a channel "
         "from an opening that does not. Numbers rank annual hit probability, "
         "which is likelihood and not consequence — the delivering channel's "
         "size and predicted volume are in the CSV and point elsewhere. Mine "
         "names are shown where USMIN carries one; they do not predict which "
         "working holds the larger pile.",
         ha="center", fontsize=6.5, color="#444444")
mc.save(fig, "perry_canyon_openings")
print("saved figures/perry_canyon_openings.pdf")
