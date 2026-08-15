"""Statewide simulated burn-severity map — WGS84 axes, context, PNG+PDF."""
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
import rasterio
from rasterio.warp import Resampling, reproject

from firescape import plotting as mc
from firescape import paths

OUT = paths.products_dir("prefire", "statewide_v0")

nv = gpd.read_file(paths.raw_dir("boundaries") / "nv_state.geojson").to_crs("EPSG:4326")
fires = gpd.read_file(
    paths.raw_dir("perimeters") / f"wfigs_current_{date(2026, 8, 12):%Y%m%d}.geojson"
).to_crs("EPSG:4326")

transform, shape, extent = mc.grid(mc.domain_bounds4326())
hs = mc.hillshade(transform, shape)
ctx = mc.fetch_context()


def to_display(path, resampling, fill=np.nan):
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        out = np.full(shape, fill, dtype="float32")
        reproject(arr, out, src_transform=ds.transform, src_crs=ds.crs,
                  src_nodata=ds.nodata, dst_transform=transform,
                  dst_crs="EPSG:4326", dst_nodata=fill, resampling=resampling)
    return out


dnbr = to_display(OUT / "statewide_v0_simdnbr_120m.tif", Resampling.bilinear)
barc = to_display(OUT / "statewide_v0_simbarc_120m.tif", Resampling.nearest, 0)
barc[barc == 0] = np.nan

BARC_CMAP = ListedColormap(["#009E73", "#F0E442", "#E69F00", "#D55E00"])
BARC_NORM = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], BARC_CMAP.N)
im_extent = (extent[0], extent[1], extent[2], extent[3])

fig, axes = plt.subplots(1, 2, figsize=(15, 10.5), dpi=140)
for ax in axes:
    ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=im_extent)

im = axes[0].imshow(np.ma.masked_invalid(dnbr), cmap="magma", vmin=0,
                    vmax=float(np.nanpercentile(dnbr, 99.5)), extent=im_extent,
                    alpha=0.6, interpolation="nearest")
fig.colorbar(im, ax=axes[0], shrink=0.55, label="simulated dNBR (x1000)")
axes[0].set_title("Simulated dNBR — every pixel burns at P$_{dsim}$=0.51",
                  fontsize=11)

axes[1].imshow(np.ma.masked_invalid(barc), cmap=BARC_CMAP, norm=BARC_NORM,
               extent=im_extent, alpha=0.6, interpolation="nearest")
axes[1].legend(handles=[Line2D([0], [0], marker="s", color="none",
                               markersize=11, markerfacecolor=c, label=l)
                        for c, l in zip(BARC_CMAP.colors,
                                        ["unburned/very low", "low",
                                         "moderate", "high"])],
               title="simulated BARC class", loc="lower left", fontsize=8,
               title_fontsize=8)
axes[1].set_title("Simulated BARC class (breaks 125 / 281 / 500)", fontsize=11)

for ax in axes:
    mc.draw_context(ax, ctx)
    nv.boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=8)
    fires.boundary.plot(ax=ax, zorder=8.5, **mc.fire_style("current"))
    mc.style_axes(ax, extent)

fig.suptitle(
    "firescape intermediate product — statewide simulated burn severity, Nevada\n"
    "LF2025 EVT + Nevada-refit EVT-dNBR CDFs (nv_merged) at the calibrated "
    "P$_{dsim}$=0.51 · 120 m product, 500 m display · blue = Bug and Stallion",
    y=0.98, fontsize=12.5)
fig.tight_layout()
mc.save(fig, "statewide_v0_severity")
