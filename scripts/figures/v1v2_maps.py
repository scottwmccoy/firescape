"""pilot_v1 vs pilot_v2 hazard maps + the change the Nevada refit produces."""
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from cycler import cycler

OK = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000']
plt.rcParams['axes.prop_cycle'] = cycler(color=OK)

from datetime import date

import geopandas as gpd
import numpy as np
import rioxarray as rxr

from firescape import paths

hs = rxr.open_rasterio(paths.interim_dir("pilot") / "pilot_hillshade.tif",
                       masked=True).squeeze()[::5, ::5]
v1 = gpd.read_file(paths.products_dir("prefire", "pilot_v1") / "pilot_v1_basins.gpkg").to_crs(hs.rio.crs)
v2 = gpd.read_file(paths.products_dir("prefire", "pilot_v2") / "pilot_v2_basins.gpkg").to_crs(hs.rio.crs)
fires = gpd.read_file(
    paths.raw_dir("perimeters") / f"wfigs_current_{date(2026, 8, 12):%Y%m%d}.geojson"
).to_crs(hs.rio.crs)

HAZ = ListedColormap(["#4B9B6E", "#E8A33D", "#C1272D"])
NORM = BoundaryNorm([0.5, 1.5, 2.5, 3.5], HAZ.N)


def base(ax):
    hs.plot.imshow(ax=ax, cmap="gray", vmin=-20, vmax=255, add_colorbar=False)


def finish(ax, title):
    fires.boundary.plot(ax=ax, color="#56B4E9", linewidth=1.1)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")


# ---- figure 1: the v2 product ----------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(19, 8), dpi=140)
base(axes[0])
v2.plot(ax=axes[0], column="H_24mmh", cmap=HAZ, norm=NORM, edgecolor="none", alpha=0.85)
axes[0].legend(handles=[Line2D([0], [0], marker="s", color="none", markersize=11,
                               markerfacecolor=c, label=l)
                        for c, l in zip(HAZ.colors, ["low", "moderate", "high"])],
               title="Hazard class", loc="lower left", fontsize=8, title_fontsize=8)
finish(axes[0], "Combined hazard class")
for ax, col, cmap, lbl in ((axes[1], "P_24mmh", "magma",
                            "Debris-flow likelihood (I15 = 24 mm/h)"),
                           (axes[2], "P_RgtT", "viridis",
                            "P(R>T): annual chance of threshold-exceeding rain")):
    base(ax)
    v2.plot(ax=ax, column=col, cmap=cmap, vmin=0,
            vmax=float(np.nanpercentile(v2[col], 99)), edgecolor="none", alpha=0.85,
            legend=True, legend_kwds={"shrink": 0.55, "label": lbl})
    finish(ax, lbl)
fig.suptitle(
    "firescape pilot pre-fire postfire-debris-flow hazard (pilot_v2) — north of Reno, Nevada\n"
    f"{len(v2):,} basins · Nevada-refit EVT–dNBR CDFs at P$_{{dsim}}$=0.51 · SSURGO soils · "
    "blue = Bug and Stallion fires, Aug 2026", y=1.04, fontsize=12)
fig.tight_layout()
p2 = paths.figures_dir() / "pilot_prefire_hazard_v2.png"
fig.savefig(p2, bbox_inches="tight")
plt.close(fig)

# ---- figure 2: what the refit changed ---------------------------------------
assert (v1["Segment_ID"].to_numpy() == v2["Segment_ID"].to_numpy()).all(), "basin order differs"
dv = v2["P_24mmh"].to_numpy() - v1["P_24mmh"].to_numpy()
fig, axes = plt.subplots(1, 3, figsize=(19, 8), dpi=140)
for ax, df, ttl in ((axes[0], v1, "pilot_v1 — Staley 2018 CDFs, P$_{dsim}$=0.48"),
                    (axes[1], v2, "pilot_v2 — Nevada-refit CDFs, P$_{dsim}$=0.51")):
    base(ax)
    df.plot(ax=ax, column="H_24mmh", cmap=HAZ, norm=NORM, edgecolor="none", alpha=0.85)
    n2 = int((df["H_24mmh"] == 2).sum())
    finish(ax, f"{ttl}\nmoderate-hazard basins: {n2:,}")
axes[0].legend(handles=[Line2D([0], [0], marker="s", color="none", markersize=11,
                               markerfacecolor=c, label=l)
                        for c, l in zip(HAZ.colors, ["low", "moderate", "high"])],
               title="Hazard class", loc="lower left", fontsize=8, title_fontsize=8)
base(axes[2])
lim = float(np.nanpercentile(np.abs(dv), 99))
v2.assign(_d=dv).plot(ax=axes[2], column="_d", cmap="RdBu_r",
                      norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim),
                      edgecolor="none", alpha=0.9, legend=True,
                      legend_kwds={"shrink": 0.55, "label": "Δ likelihood (v2 − v1)"})
finish(axes[2], f"Change in likelihood\nmean {np.nanmean(dv):+.3f}, "
                f"{100*np.nanmean(dv > 0):.0f}% of basins increase")
fig.suptitle("Effect of the Nevada EVT–dNBR refit on the pilot pre-fire hazard surface\n"
             "basins with zero predicted volume: 83.5% → 24.7%", y=1.04, fontsize=12)
fig.tight_layout()
p3 = paths.figures_dir() / "pilot_v1_v2_comparison.png"
fig.savefig(p3, bbox_inches="tight")
print("wrote", p2)
print("wrote", p3)
