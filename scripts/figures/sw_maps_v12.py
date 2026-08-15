"""Statewide v0 hazard maps — WGS84 axes, stormscape context, PNG+PDF."""
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

from datetime import date

import geopandas as gpd
import numpy as np
from pyogrio import read_dataframe

from firescape import plotting as mc
from firescape import paths

OUT = paths.products_dir("prefire", "statewide_v1_2")

basins = read_dataframe(OUT / "statewide_v1_2_basins.gpkg",
                        columns=["H_24mmh", "P_24mmh", "P_RgtT"]).to_crs("EPSG:4326")
nv = gpd.read_file(paths.raw_dir("boundaries") / "nv_state.geojson").to_crs("EPSG:4326")
fires = gpd.read_file(
    paths.raw_dir("perimeters") / f"wfigs_current_{date(2026, 8, 12):%Y%m%d}.geojson"
).to_crs("EPSG:4326")

transform, shape, extent = mc.grid(mc.domain_bounds4326())
print(f"display grid {shape[1]}x{shape[0]} @ {mc.RES_DEG:g} deg")
hs = mc.hillshade(transform, shape)
ctx = mc.fetch_context()
print("context layers ready")

HAZ = ListedColormap(["#4B9B6E", "#E8A33D", "#C1272D"])
NORM = BoundaryNorm([0.5, 1.5, 2.5, 3.5], HAZ.N)
im_extent = (extent[0], extent[1], extent[2], extent[3])
panels = [("H_24mmh", None, None, "Combined hazard class"),
          ("P_24mmh", "magma", (0, float(np.nanpercentile(basins["P_24mmh"], 99))),
           "Debris-flow likelihood (I15 = 24 mm/h)"),
          ("P_RgtT", "viridis", (0, float(np.nanpercentile(basins["P_RgtT"], 99))),
           "P(R>T): annual chance of threshold rain")]

fig, axes = plt.subplots(1, 3, figsize=(20, 10), dpi=140)
for ax, (col, cmap, clim, title) in zip(axes, panels):
    ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=im_extent)
    arr = mc.burn(basins, col, transform, shape)
    if col == "H_24mmh":
        ax.imshow(np.ma.masked_invalid(arr), cmap=HAZ, norm=NORM,
                  extent=im_extent, alpha=0.6, interpolation="nearest")
        ax.legend(handles=[Line2D([0], [0], marker="s", color="none",
                                  markersize=11, markerfacecolor=c, label=l)
                           for c, l in zip(HAZ.colors,
                                           ["low", "moderate", "high"])],
                  title="Hazard class", loc="lower left", fontsize=8,
                  title_fontsize=8)
    else:
        im = ax.imshow(np.ma.masked_invalid(arr), cmap=cmap, vmin=clim[0],
                       vmax=clim[1], extent=im_extent, alpha=0.6,
                       interpolation="antialiased")
        fig.colorbar(im, ax=ax, shrink=0.5, label=title)
    mc.draw_context(ax, ctx)
    nv.boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=8)
    fires.boundary.plot(ax=ax, zorder=8.5, **mc.fire_style("current"))
    mc.style_axes(ax, extent)
    ax.set_title(title, fontsize=11)

fig.suptitle(
    "firescape statewide v1.2 — pre-fire postfire-debris-flow hazard, Nevada\n"
    f"{len(basins):,} basins · 591 HU10 watersheds · Nevada-refit CDFs · "
    "per-region dispersed-severity calibration ($\\sigma_z$=0.91) · RANGES volume · SSURGO soils · "
    "blue = Bug and Stallion fires", y=0.99, fontsize=13)
fig.tight_layout()
mc.save(fig, "statewide_v1_2_hazard")
