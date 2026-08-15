"""Regional zooms over Nevada -- annualized triptych.

The statewide sheet answers "where in the state", which is the wrong question
for a planner in Washoe or White Pine County. These render the same v1.2
product at a scale where an individual drainage above a subdivision is
legible: a ~50 m display grid, hillshade shaded at 30 m, sub-degree graticule
ticks, and every incorporated place labelled rather than the statewide city
whitelist. The windows themselves live in :mod:`_corridors`.

The same triptych as the statewide sheet -- the two factors and their product.
Probability bars are labelled as **return intervals** ("1 in 300 yr"), the form
the number gets quoted in; P(R>T) keeps plain decimals, since it lives between
about 0.01 and 0.9. Hazard-class counts still go to the summary JSON.

The companion sheet is :mod:`p12_urban_zoom_forecast`, which drops the annual
rate and asks the fire-forecast question instead. Both share their window,
labels and line styles through :mod:`_corridors`.

Several windows deliberately run past the state line: the drainages above the
Sierra front, the Spring Mountains, the Snake Range and the Jarbidge country
do not stop at a border, P(F) now covers the neighbouring states, and the Lake
Tahoe / Truckee River units on the California side were added so that basin is
whole rather than clipped. Coverage still thins outside Nevada, because the
unit inventory is "every HU10 that touches the state" -- that visible edge is
the modelled domain, not a change in the hazard.
"""
import json
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

import _corridors as cor
from firescape import paths, plotting as mc, relief

VERSION = sys.argv[2] if len(sys.argv) > 2 else "statewide_v1_2"

name = sys.argv[1] if len(sys.argv) > 1 else None
todo = [name] if name else list(cor.ZOOMS)
GP = paths.products_dir("prefire", VERSION) / f"{VERSION}_basins.gpkg"

#: Return intervals a planner actually quotes, as annual probabilities.
RI_TICKS = [1 / 30, 1 / 100, 1 / 300, 1 / 1000, 1 / 3000, 1 / 10000, 1 / 30000]

for key in todo:
    z = cor.ZOOMS[key]
    bounds = z["bounds"]
    print(f"\n=== {key}: {z['label']} ===", flush=True)
    # pyogrio's bbox is in the LAYER's CRS, and the product is EPSG:5070 --
    # passing lon/lat silently returns nothing.
    from rasterio.warp import transform_bounds
    bbox5070 = transform_bounds("EPSG:4326", "EPSG:5070", *bounds)
    basins = read_dataframe(
        GP, bbox=bbox5070,
        columns=["H_24mmh", "P_24mmh", "P_annual", "P_F", "P_RgtT",
                 "I15_50", "kf_filled"]).to_crs("EPSG:4326")
    if basins.empty:
        sys.exit(f"no basins within {bounds}")
    print(f"{len(basins):,} basins in the window", flush=True)

    tr, shape, extent = mc.grid(bounds, res=z["res"])
    print(f"display grid {shape[1]}x{shape[0]} @ {z['res']:g} deg", flush=True)
    hs = relief.shaded_relief(
        sorted((paths.cache_root() / "3dep_tiles").glob("USGS_13_*.tif")),
        tr, shape, crs="EPSG:4326", shade_res_m=30.0)
    C = cor.context(key)
    print(f"{len(C['rivers'])} reaches of "
          f"{sorted(C['river_labels']['name'].astype(str))}", flush=True)
    im_extent = (extent[0], extent[1], extent[2], extent[3])

    counts = basins["H_24mmh"].value_counts()
    med = float(np.nanmedian(basins["P_annual"]))
    fig, axes = plt.subplots(1, 3, figsize=cor.figure_size(bounds, 3), dpi=140)

    # Same triptych as the statewide sheet: the two factors, then the product.
    spec = [("P_F", "magma", "P(F) — annual burn probability\n"
             "FSim / Wildfire Risk to Communities"),
            ("P_RgtT", "viridis", "P(R>T) — annual chance of threshold rain\n"
             "given the basin has burned"),
            ("P_annual", "YlOrRd", "P(F) × P(R>T) — annual probability\n"
             f"of a postfire debris flow · median ≈ 1 in {1/med:,.0f} yr")]
    for ax, (col, cmap, title) in zip(axes, spec):
        ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=im_extent, zorder=0)
        arr = mc.burn(basins, col, tr, shape)
        v = arr[np.isfinite(arr) & (arr > 0)]
        lo, hi = np.percentile(v, [2, 99.5])
        im = ax.imshow(np.ma.masked_invalid(arr), cmap=cmap,
                       norm=LogNorm(vmin=lo, vmax=hi), extent=im_extent,
                       alpha=mc.MAX_LAYER_ALPHA, zorder=2,
                       interpolation="antialiased")
        cb = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.02)
        cb.ax.tick_params(labelsize=8)
        cb.ax.yaxis.set_minor_locator(FixedLocator([]))
        if col == "P_RgtT":            # decimals, as on the statewide sheet
            cand = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
            cb.ax.yaxis.set_major_locator(
                FixedLocator([x for x in cand if lo <= x <= hi]))
            cb.ax.yaxis.set_major_formatter(FuncFormatter(
                lambda x, _: f"{x:.3f}".rstrip("0").rstrip(".") if x > 0 else ""))
        else:                          # return intervals, the quoted form
            cb.ax.yaxis.set_major_locator(
                FixedLocator([x for x in RI_TICKS if lo <= x <= hi]))
            cb.ax.yaxis.set_major_formatter(FuncFormatter(
                lambda x, _: f"1 in {round(1/x, -1):,.0f} yr" if x > 0 else ""))
        ax.set_title(title, fontsize=10.5)
        cor.decorate(ax, C, extent, step=z["step"])

    kf_frac = float(basins["kf_filled"].astype(bool).mean())
    fig.tight_layout(w_pad=0.4)
    cor.suptitle(fig, [line for line in (
        f"{z['label']} — firescape {VERSION} pre-fire debris-flow hazard",
        z["blurb"],
        f"{len(basins):,} basins · median ≈ 1 in {1/med:,.0f} yr · "
        f"{int(counts.get(2, 0)):,} moderate, {int(counts.get(3, 0)):,} high "
        "at the 24 mm/h reference storm",
        cor.kf_note(kf_frac, "basins"),
    ) if line])
    mc.save(fig, f"zoom_{key}_{VERSION}")
    plt.close(fig)

    stats = {
        "zoom": key, "label": z["label"], "version": VERSION,
        "bounds": list(bounds), "basins": int(len(basins)),
        "hazard_class": {str(int(k)): int(v) for k, v in counts.items()},
        "median_threshold_mmh": round(float(np.nanmedian(basins["I15_50"])), 1),
        "median_P_24mmh": round(float(np.nanmedian(basins["P_24mmh"])), 3),
        "median_P_annual": round(med, 8),
        "median_return_interval_yr": round(1 / med),
        "basins_over_1_in_100_yr": int((basins["P_annual"] > 0.01).sum()),
        "basins_over_1_in_300_yr": int((basins["P_annual"] > 1 / 300).sum()),
    }
    out = paths.products_dir("prefire", VERSION) / f"zoom_{key}_summary.json"
    out.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)
