"""Statewide AML compilation: mapped waste **and** underground workings.

``e3_aml_map`` draws the 1,131 USMIN waste extents — dumps, tailings, ponds.
That layer is only as complete as the topographer's pen: a dump was drawn as
an extent where it was large enough to warrant one, so 90% of the 5 km cells
in Nevada holding a mine opening hold no mapped waste at all, and whole
districts (Perry Canyon among them) come back empty. This sheet adds the
32,582 adits and shafts as the second asset class, each the anchor of a portal
dump that was never mapped, and puts the two on one frame.

The point of the compilation is coverage: the waste layer says *where the
material was drawn*, the workings layer says *where mining happened*, and the
gap between them is the part of the problem the first layer cannot see.

Reading the sheet: colour is the annual hit probability, and only assets
within 30-55 m of a modelled channel carry it. Everything further out is grey,
because a grey dot is still a place someone dug — and the sheet would lie by
omission if it dropped them. Circles are mapped waste, triangles are workings.

Neither ordering is a source-size ordering: USMIN records nothing about how
much rock a working produced, and at Perry Canyon neither the mine name nor
the size of the receiving channel predicted where the large pile actually was.
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
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter
from scipy.cluster.hierarchy import fcluster, linkage
from stormscape.plot import Labeller

from firescape import paths
from firescape import plotting as mc

plt.rcParams["image.cmap"] = "viridis"
EXPO = paths.products_dir("exposure")

waste = gpd.read_file(EXPO / "aml_v2" / "aml_exposure.gpkg").to_crs(4326)
work = gpd.read_file(EXPO / "openings_v1" / "openings_exposure.gpkg").to_crs(4326)
waste["kind"] = "waste"
work["kind"] = "working"
for g in (waste, work):
    g["rep"] = g.geometry.representative_point()
    g["exposed"] = g["exposed"].astype(bool)
both = pd.concat([waste, work], ignore_index=True)
print(f"{len(waste):,} mapped waste sites + {len(work):,} workings "
      f"= {len(both):,} assets", flush=True)

nv = gpd.read_file(paths.raw_dir("boundaries") / "nv_state.geojson").to_crs(4326)
blm = gpd.read_file(paths.raw_dir("blm") / "nv_blm_sma.gpkg").to_crs(4326)
transform, shape, extent = mc.grid(mc.domain_bounds4326())
hs = mc.hillshade(transform, shape)
ctx = mc.fetch_context()

near = both[both["exposed"]]
far = both[~both["exposed"]]
p = near["P_annual_site"].dropna()
vmin, vmax = float(np.nanpercentile(p, 2)), float(np.nanpercentile(p, 99.5))
norm = LogNorm(vmin=vmin, vmax=vmax)
print(f"near-channel: {int((waste['exposed']).sum()):,} waste, "
      f"{int((work['exposed']).sum()):,} workings", flush=True)

# Explicit placement rather than tight_layout: the bottom note is added
# after the layout is final, and tight_layout reserves nothing for it — it
# printed straight through the degree labels and the legend box.
FW, FH = 9.6, 12.4
fig = plt.figure(figsize=(FW, FH), dpi=150)
ax = fig.add_axes([0.055, 0.088, 0.80, 0.855])
ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=extent)
blm.plot(ax=ax, facecolor="#D9C98C", edgecolor="none", alpha=0.26, zorder=1,
         rasterized=True)
mc.draw_context(ax, ctx, extent=extent)
nv.boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=8)

# Everything not near a channel, faint: still a place someone dug.
ax.scatter([p.x for p in far["rep"]], [p.y for p in far["rep"]], s=1.6,
           marker=".", color="#8A8A8A", alpha=0.5, zorder=8.5, linewidths=0,
           rasterized=True)
sc = None
for kind, marker, size in (("working", "^", 8), ("waste", "o", 30)):
    k = near[near["kind"] == kind]
    if not len(k):
        continue
    sc = ax.scatter([p.x for p in k["rep"]], [p.y for p in k["rep"]], s=size,
                    c=k["P_annual_site"], norm=norm, marker=marker,
                    edgecolors="white" if kind == "working" else "#111111",
                    linewidths=0.25 if kind == "working" else 0.45,
                    zorder=9 + size / 100, rasterized=True)
cb = fig.colorbar(sc, cax=fig.add_axes([0.875, 0.40, 0.016, 0.30]),
                  extend="both")
cb.set_label("annual chance a debris flow reaches the site\n"
             "P(F) × P(R>T) × P(DF), as a return interval", fontsize=8)
ticks = [t for t in (1/30, 1/100, 1/300, 1/1000, 1/3000) if vmin <= t <= vmax]
cb.ax.yaxis.set_major_locator(FixedLocator(ticks))
cb.ax.yaxis.set_minor_locator(FixedLocator([]))
cb.ax.yaxis.set_major_formatter(FuncFormatter(
    lambda x, _: f"1 in {round(1/x, -1):,.0f} yr" if x > 0 else ""))
cb.ax.tick_params(labelsize=7)

# One entry per district: the plain top twelve were all the same Elko
# cluster, which made a starburst of leaders in one corner and told a
# reader nothing about the rest of the state.
n5070 = near.to_crs(5070)
xy = np.c_[n5070.geometry.centroid.x, n5070.geometry.centroid.y]
grp = (fcluster(linkage(xy, method="single"), t=25_000, criterion="distance")
       if len(xy) > 1 else np.ones(len(xy), int))
near = near.assign(district=grp)
top = (near.sort_values("P_annual_site", ascending=False)
           .drop_duplicates("district").head(12).reset_index(drop=True))
print(f"{near['district'].nunique()} districts with near-channel assets",
      flush=True)
rows = []
for i, r in top.iterrows():
    nm = next((str(v) for v in (r["name"], r["ftr_type"])
               if isinstance(v, str) and v.strip()), "unnamed")[:20]
    county = str(r["county"])[:9] if isinstance(r["county"], str) else ""
    # every row the same width: the block is right-aligned, so ragged
    # lines would step in and out instead of forming a table
    rows.append(f"{i+1:>2} {'W' if r['kind']=='waste' else 'A'} {nm:<20} "
                f"{county:<9} 1 in {1/r['P_annual_site']:>5,.0f} yr "
                f"{'BLM' if r['on_blm'] else '   '}")
ax.text(0.985, 0.015, "highest-ranked district, one entry each\n"
        "(W = mapped waste, A = adit/shaft)\n" + "\n".join(rows),
        transform=ax.transAxes, ha="right", va="bottom", fontsize=6.4,
        family="monospace",
        bbox=dict(facecolor="white", alpha=0.87, edgecolor="#666666",
                  linewidth=0.5, boxstyle="round,pad=0.4"), zorder=13)

nw, nk = int(waste["exposed"].sum()), int(work["exposed"].sum())
ax.legend(handles=[
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#3B528B",
           markeredgecolor="white", markersize=7,
           label=f"mapped waste, near a channel ({nw:,} of {len(waste):,})"),
    Line2D([0], [0], marker="^", color="none", markerfacecolor="#3B528B",
           markeredgecolor="white", markersize=7,
           label=f"adit / shaft, near a channel ({nk:,} of {len(work):,})"),
    Line2D([0], [0], marker=".", color="none", markerfacecolor="#8A8A8A",
           markeredgecolor="none", markersize=7,
           label=f"further than 30–55 m ({len(far):,})"),
    Patch(facecolor="#D9C98C", alpha=0.5, label="BLM-managed land"),
], loc="lower left", fontsize=7.2, title="USMIN mine features",
    title_fontsize=8, framealpha=0.87)

mc.style_axes(ax, extent)
ax.set_title("Abandoned-mine material on Nevada's post-fire debris-flow "
             "network\nmapped waste and underground workings, "
             "v1.2 pre-fire hazard surface", fontsize=11)
mc.footnote(fig,
    "Near-channel means within 30–55 m of a modelled channel — half a corridor "
    "9–60 m wide that widens with contributing area, plus 25 m for "
    "registration. Proximity to the network, not a runout model. USMIN maps a "
    "dump as an extent only where a topographer drew one, which is why the "
    "workings are here: they mark mining in the 90% of Nevada's mine-bearing "
    "5 km cells that carry no mapped waste at all. Records are deduplicated "
    "across overlapping quadrangles. This orders exposure, not source size — "
    "nothing in USMIN records how much rock a working produced.")

fig.canvas.draw()
lab = Labeller(ax)
lab.block_many([p.x for p in top["rep"]], [p.y for p in top["rep"]],
               radius_px=4.0)
for i, r in top.iterrows():
    lab.label(r["rep"].x, r["rep"].y, f"{i+1}", fontsize=8, weight="bold",
              color="#C1272D", zorder=12,
              force_leader=lab.crowded(r["rep"].x, r["rep"].y, radius_px=14))

mc.save(fig, "aml_compilation_v1")
(EXPO / "compilation_summary.json").write_text(json.dumps({
    "waste_sites": int(len(waste)), "waste_near_channel": nw,
    "workings": int(len(work)), "workings_near_channel": nk,
    "assets_total": int(len(both)),
    "near_channel_total": int(len(near)),
    "on_blm_near_channel": int(near["on_blm"].sum()),
}, indent=2) + "\n")
print("saved figures/aml_compilation_v1.pdf")
