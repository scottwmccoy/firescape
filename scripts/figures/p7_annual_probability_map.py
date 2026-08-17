"""Annual postfire-debris-flow probability: P(F) x P(R>T), and its two factors."""
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator, FuncFormatter
from pyogrio import read_dataframe

import geopandas as gpd
from firescape import paths, plotting as mc

VERSION = sys.argv[1] if len(sys.argv) > 1 else "statewide_v1_1"
OUT = paths.products_dir("prefire", VERSION)

g = read_dataframe(OUT / f"{VERSION}_basins.gpkg",
                   columns=["P_F", "P_RgtT", "P_annual"]).to_crs("EPSG:4326")
nv = gpd.read_file(paths.raw_dir("boundaries") / "nv_state.geojson").to_crs("EPSG:4326")

# where is P(F) missing? BP_NV.tif is clipped to the state, and HU10 units
# cross the border -- so the gaps should be out-of-state basins, not holes.
miss = g["P_F"].isna()
inside = g[miss].geometry.representative_point().within(nv.union_all().buffer(0.01))
print(f"{miss.sum():,} basins without P(F); "
      f"{int(inside.sum()):,} of them inside Nevada "
      f"({100*inside.mean():.1f}% -- the rest are out-of-state)", flush=True)

tr, shape, extent = mc.grid(mc.domain_bounds4326())
hs = mc.hillshade(tr, shape)
ctx = mc.fetch_context()
im_extent = (extent[0], extent[1], extent[2], extent[3])

panels = [
    ("P_F", "magma", None, "P(F) — annual burn probability\nFSim / Wildfire Risk to Communities"),
    ("P_RgtT", "viridis", None, "P(R>T) — annual chance of threshold rain\ngiven the basin has burned"),
    ("P_annual", "inferno", None, "P(F) × P(R>T) — annual probability\nof a postfire debris flow"),
]

fig, axes = plt.subplots(1, 3, figsize=(21, 11), dpi=140)
for ax, (col, cmap, _, title) in zip(axes, panels):
    ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=im_extent)
    arr = mc.burn(g, col, tr, shape)
    v = arr[np.isfinite(arr) & (arr > 0)]
    lo, hi = np.percentile(v, [5, 99.5])
    im = ax.imshow(np.ma.masked_invalid(arr), cmap=cmap,
                   norm=LogNorm(vmin=max(lo, 1e-6), vmax=hi),
                   extent=im_extent, alpha=mc.MAX_LAYER_ALPHA,
                   interpolation="antialiased")
    cb = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.02)
    cb.ax.tick_params(labelsize=8)
    if col == "P_RgtT":
        # This panel lives between about 0.01 and 0.9, where LogNorm's default
        # scientific notation (2x10^-1) is harder to read than 0.2.
        cand = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        cb.ax.yaxis.set_major_locator(
            FixedLocator([t for t in cand if lo <= t <= hi]))
        cb.ax.yaxis.set_minor_locator(FixedLocator([]))
        cb.ax.yaxis.set_major_formatter(FuncFormatter(
            lambda x, _: f"{x:.3f}".rstrip("0").rstrip(".") if x > 0 else ""))
    mc.draw_context(ax, ctx, label_rivers=False, extent=extent)
    nv.boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=8)
    mc.style_axes(ax, extent)
    ax.set_title(title, fontsize=10)

med = float(np.nanmedian(g["P_annual"]))
fig.suptitle(
    f"Annual probability of a postfire debris flow — firescape {VERSION}\n"
    f"median basin {med:.2e}/yr (~1 in {1/med:,.0f} years) · log colour scales · "
    "assumes fire and storm occurrence are independent; P(F) is a 2020-fuels "
    "steady-state rate", y=0.985, fontsize=12.5)
fig.tight_layout(rect=(0, 0, 1, 0.94))
mc.save(fig, f"{VERSION}_annual_probability")
