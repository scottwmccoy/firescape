"""Statewide AML waste exposure map -> figures/aml_exposure_v1.pdf.

One panel: every USMIN waste site over the house hillshade+context frame,
status by marker (dot = clear, open circle = near-corridor, filled = in a
delivery corridor) with the exposed sites colored by their annual PFDF hit
rate; the top ten ranked sites numbered, with a side table.
"""
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from pyogrio import read_dataframe

from firescape import paths
from firescape import plotting as mc

plt.rcParams["image.cmap"] = "viridis"

sites = read_dataframe(
    paths.products_dir("exposure", "aml_v1") / "aml_exposure.gpkg"
).to_crs("EPSG:4326")
sites["rep"] = sites.geometry.representative_point()
nv = gpd.read_file(paths.raw_dir("boundaries") / "nv_state.geojson").to_crs(4326)

transform, shape, extent = mc.grid(mc.domain_bounds4326())
hs = mc.hillshade(transform, shape)
ctx = mc.fetch_context()

exp = sites[sites["exposed"].astype(bool)]
near = sites[(~sites["exposed"].astype(bool)) & sites["dist_m"].notna()]
clear = sites[sites["dist_m"].isna()]
vmax = float(np.nanpercentile(exp["P_annual_site"], 99))

fig, ax = plt.subplots(figsize=(9, 11.5), dpi=150)
ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=extent)
mc.draw_context(ax, ctx)
nv.boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=8)

ax.scatter([p.x for p in clear["rep"]], [p.y for p in clear["rep"]],
           s=4, c="#888888", marker=".", zorder=9, linewidths=0)
ax.scatter([p.x for p in near["rep"]], [p.y for p in near["rep"]],
           s=12, facecolors="none", edgecolors="#5D5D5D", marker="o",
           linewidths=0.6, zorder=9)
sc = ax.scatter([p.x for p in exp["rep"]], [p.y for p in exp["rep"]],
                s=22, c=exp["P_annual_site"], vmin=0, vmax=vmax,
                marker="o", edgecolors="white", linewidths=0.35, zorder=10)
cb = fig.colorbar(sc, ax=ax, shrink=0.35, pad=0.01)
cb.set_label("annual hit probability  P(F) x P(R>T) x P(DF)", fontsize=8)
cb.ax.tick_params(labelsize=7)

top = sites.nsmallest(10, "rank")
rows = []
for _, r in top.iterrows():
    ax.annotate(f"{int(r['rank'])}", (r["rep"].x, r["rep"].y),
                xytext=(5, 4), textcoords="offset points", fontsize=8,
                fontweight="bold", color="#C1272D", zorder=12,
                path_effects=[matplotlib.patheffects.withStroke(
                    linewidth=2, foreground="white")])
    name = next((str(v) for v in (r["name"], r["ftr_type"])
                 if isinstance(v, str) and v.strip()), "unnamed")[:24]
    county = str(r["county"])[:9] if isinstance(r["county"], str) else ""
    rows.append(f"{int(r['rank']):>2}  {name:<24} {county:<9} "
                f"1-in-{1 / r['P_annual_site']:,.0f} yr")
ax.text(0.985, 0.015,
        "highest-exposure sites\n" + "\n".join(rows),
        transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8,
        family="monospace",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="#666666",
                  linewidth=0.5, boxstyle="round,pad=0.4"), zorder=13)

ax.legend(handles=[
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#3B528B",
           markeredgecolor="white", markersize=7,
           label=f"in a delivery corridor ({len(exp)})"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
           markeredgecolor="#5D5D5D", markersize=6,
           label=f"within 1 km of one ({len(near)})"),
    Line2D([0], [0], marker=".", color="none", markerfacecolor="#888888",
           markeredgecolor="none", markersize=6,
           label=f"clear (> 1 km) ({len(clear)})"),
], loc="lower left", fontsize=7.5, title="USMIN mine-waste sites",
    title_fontsize=8, framealpha=0.85)

mc.style_axes(ax, extent)
ax.set_title("Abandoned-mine waste in post-fire debris-flow corridors\n"
             "statewide exposure ranking on the v1.2 pre-fire hazard surface",
             fontsize=11)
fig.text(0.01, 0.005,
         "Delivery = site touches a segment corridor (width 9-60 m from "
         "contributing area, +25 m pad): a proximity proxy, not a runout "
         "model. Assets: USMIN topo-sheet symbols (historical, not the "
         "operational AML inventory). Annual rate at the delivering "
         "segment's threshold through the basin rain climatology.",
         fontsize=6, color="#444444")
fig.tight_layout()
mc.save(fig, "aml_exposure_v1")
print("saved figures/aml_exposure_v1.pdf")
