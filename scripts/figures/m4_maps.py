"""Pilot pre-fire product maps: hazard class, likelihood, and P(R>T)."""
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from cycler import cycler

OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000']
plt.rcParams['axes.prop_cycle'] = cycler(color=OKABE_ITO)

import geopandas as gpd
import numpy as np
import rioxarray as rxr

from firescape import paths
from datetime import date

OUT = paths.products_dir("prefire", "pilot_v1")
basins = gpd.read_file(OUT / "pilot_v1_basins.gpkg")
hs = rxr.open_rasterio(paths.interim_dir("pilot") / "pilot_hillshade.tif",
                       masked=True).squeeze()
basins = basins.to_crs(hs.rio.crs)
fires = gpd.read_file(
    paths.raw_dir("perimeters") / f"wfigs_current_{date(2026, 8, 12):%Y%m%d}.geojson"
).to_crs(hs.rio.crs)
hs_small = hs[::5, ::5]

panels = [
    ("H_24mmh", "Combined hazard class", None),
    ("P_24mmh", "Debris-flow likelihood (I15 = 24 mm/h)", "magma"),
    ("P_RgtT", "P(R>T): annual chance of threshold-exceeding rain", "viridis"),
]
fig, axes = plt.subplots(1, 3, figsize=(19, 8), dpi=140)
for ax, (col, title, cmap) in zip(axes, panels):
    hs_small.plot.imshow(ax=ax, cmap="gray", vmin=-20, vmax=255, add_colorbar=False)
    if col == "H_24mmh":
        haz_cmap = ListedColormap(["#4B9B6E", "#E8A33D", "#C1272D"])
        norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], haz_cmap.N)
        basins.plot(ax=ax, column=col, cmap=haz_cmap, norm=norm, edgecolor="none",
                    alpha=0.85)
        ax.legend(handles=[Line2D([0], [0], marker="s", color="none", markersize=11,
                                  markerfacecolor=c, label=l)
                           for c, l in zip(haz_cmap.colors, ["low", "moderate", "high"])],
                  title="Hazard class", loc="lower left", fontsize=8, title_fontsize=8)
    else:
        vmax = float(np.nanpercentile(basins[col], 99))
        basins.plot(ax=ax, column=col, cmap=cmap, vmin=0, vmax=vmax, edgecolor="none",
                    alpha=0.85, legend=True,
                    legend_kwds={"shrink": 0.55, "label": title})
    fires.boundary.plot(ax=ax, color="#56B4E9", linewidth=1.1)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")

fig.suptitle(
    "firescape pilot pre-fire postfire-debris-flow hazard — north of Reno, Nevada\n"
    f"{len(basins):,} basins over 35 HU10 watersheds · simulated severity at "
    "P$_{dsim}$=0.48 (Nevada calibration, 34 MTBS fires) · blue = Bug and Stallion fires, Aug 2026",
    y=1.005, fontsize=12)
fig.tight_layout()
png = paths.figures_dir() / "pilot_prefire_hazard.png"
fig.savefig(png, bbox_inches="tight")
print("wrote", png)
