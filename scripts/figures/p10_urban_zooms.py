"""Regional zooms over Nevada's two urban corridors.

The statewide sheet answers "where in the state", which is the wrong question
for a planner in Washoe or Clark County. These render the same v1.2 product at
a scale where an individual drainage above a subdivision is legible: a ~50 m
display grid, hillshade shaded at 30 m, sub-degree graticule ticks, and every
incorporated place labelled rather than the statewide city whitelist.

Two panels per corridor -- the operational hazard class at the reference storm,
and the annualised probability. The probability colour bar is labelled as a
**return interval** ("1 in 300 yr") rather than a decimal, because that is the
form the number gets used in.

Both corridors deliberately run past the state line: the drainages above the
Sierra front and the Spring Mountains do not stop at a border, and P(F) now
covers the neighbouring states so the basins there carry real values.
"""
import json
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter
from pyogrio import read_dataframe

import geopandas as gpd
from firescape import paths, plotting as mc, relief

VERSION = sys.argv[2] if len(sys.argv) > 2 else "statewide_v1_2"

#: (west, south, east, north), tick step in degrees, display resolution.
ZOOMS = {
    "reno_carson": dict(
        bounds=(-120.45, 38.72, -119.15, 40.42), step=0.25, res=0.0005,
        label="Reno–Carson corridor",
        blurb="north Pyramid Lake through Reno–Sparks to Carson City, "
              "Minden and Gardnerville, with the Sierra front"),
    "las_vegas": dict(
        bounds=(-116.35, 34.90, -113.90, 36.95), step=0.5, res=0.0006,
        label="Las Vegas / Clark County",
        blurb="Clark County with the Spring Mountains north of Pahrump, "
              "to the southern and eastern state lines"),
}

name = sys.argv[1] if len(sys.argv) > 1 else None
todo = [name] if name else list(ZOOMS)
GP = paths.products_dir("prefire", VERSION) / f"{VERSION}_basins.gpkg"

HAZ = ListedColormap(["#4B9B6E", "#E8A33D", "#C1272D"])
NORM = BoundaryNorm([0.5, 1.5, 2.5, 3.5], HAZ.N)
#: Return intervals a planner actually quotes, as annual probabilities.
RI_TICKS = [1 / 30, 1 / 100, 1 / 300, 1 / 1000, 1 / 3000, 1 / 10000, 1 / 30000]


#: Places to label even though GNIS does not call them incorporated. Minden,
#: Gardnerville and Carson City are exactly the towns these corridors are
#: about, and an "incorporated only" filter drops all three.
NOTABLE = {
    "Carson City", "Minden", "Gardnerville", "Gardnerville Ranchos",
    "Incline Village", "Dayton", "Virginia City", "Stateline", "Verdi",
    "Boulder City", "Pahrump", "Mesquite", "Indian Springs", "Searchlight",
    "Laughlin", "Overton", "Logandale", "Bunkerville", "Blue Diamond",
    "Mount Charleston", "Sandy Valley", "Primm", "Moapa Valley", "Jean",
}
_PREFIX = ("City of ", "Town of ", "Village of ", "Township of ")


def places_for(bounds):
    """Labelled places in the window.

    The statewide whitelist is tuned for a statewide sheet; at corridor scale
    it drops the towns the map is about. GNIS prefixes its civil names ("City
    of Reno"), which is noise on a figure, so those are stripped.
    """
    from stormscape import refdata
    g = refdata.places(bounds).to_crs("EPSG:4326").copy()
    g["geometry"] = g.geometry.representative_point()
    nm = g["name"].astype(str)
    for p in _PREFIX:
        nm = nm.str.removeprefix(p)
    g["name"] = nm.str.strip()
    inc = (g["kind"].astype(str).str.contains("incorporated", case=False, na=False)
           if "kind" in g.columns else False)
    g = g[inc | g["name"].isin(NOTABLE)]
    g = g[~g["name"].str.contains("Township|County", case=False, na=False)]
    return g.drop_duplicates("name")


for key in todo:
    z = ZOOMS[key]
    bounds = z["bounds"]
    print(f"\n=== {key}: {z['label']} ===", flush=True)
    # pyogrio's bbox is in the LAYER's CRS, and the product is EPSG:5070 --
    # passing lon/lat silently returns nothing.
    from rasterio.warp import transform_bounds
    bbox5070 = transform_bounds("EPSG:4326", "EPSG:5070", *bounds)
    basins = read_dataframe(
        GP, bbox=bbox5070,
        columns=["H_24mmh", "P_24mmh", "P_annual", "I15_50"]).to_crs("EPSG:4326")
    if basins.empty:
        sys.exit(f"no basins within {bounds}")
    print(f"{len(basins):,} basins in the window", flush=True)

    tr, shape, extent = mc.grid(bounds, res=z["res"])
    print(f"display grid {shape[1]}x{shape[0]} @ {z['res']:g} deg", flush=True)
    hs = relief.shaded_relief(
        sorted((paths.cache_root() / "3dep_tiles").glob("USGS_13_*.tif")),
        tr, shape, crs="EPSG:4326", shade_res_m=30.0)
    ctx = mc.fetch_context(bounds, cache_dir=paths.interim_dir("zooms", key))
    pl = places_for(bounds)
    im_extent = (extent[0], extent[1], extent[2], extent[3])

    counts = basins["H_24mmh"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(20, 11.5), dpi=140)

    ax = axes[0]
    ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=im_extent, zorder=0)
    arr = mc.burn(basins, "H_24mmh", tr, shape)
    imh = ax.imshow(np.ma.masked_invalid(arr), cmap=HAZ, norm=NORM,
                    extent=im_extent, alpha=mc.MAX_LAYER_ALPHA, zorder=2,
                    interpolation="nearest")
    # A colour bar rather than a legend, so both panels give up the same width
    # to their bar and end up the same size; the counts ride on the ticks.
    cbh = fig.colorbar(imh, ax=ax, shrink=0.55, pad=0.02, ticks=[1, 2, 3])
    cbh.ax.set_yticklabels([f"{l}\n({int(counts.get(i, 0)):,})" for i, l in
                            enumerate(["low", "moderate", "high"], 1)],
                           fontsize=8)
    cbh.set_label("combined hazard class", fontsize=9)
    ax.set_title("Combined hazard class at the 24 mm/h reference storm\n"
                 "(≈1-year, 15-minute intensity)", fontsize=10.5)

    ax = axes[1]
    ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=im_extent, zorder=0)
    arr = mc.burn(basins, "P_annual", tr, shape)
    v = arr[np.isfinite(arr) & (arr > 0)]
    lo, hi = np.percentile(v, [2, 99.5])
    # light-to-dark warm ramp: over a pale hillshade the most hazardous basins
    # need to be the darkest thing on the panel, and dark red matches the
    # hazard-class legend beside it
    im = ax.imshow(np.ma.masked_invalid(arr), cmap="YlOrRd",
                   norm=LogNorm(vmin=lo, vmax=hi), extent=im_extent,
                   alpha=mc.MAX_LAYER_ALPHA, zorder=2,
                   interpolation="antialiased")
    cb = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.02)
    cb.ax.yaxis.set_major_locator(
        FixedLocator([t for t in RI_TICKS if lo <= t <= hi]))
    cb.ax.yaxis.set_minor_locator(FixedLocator([]))
    cb.ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"1 in {round(1/x, -1):,.0f} yr" if x > 0 else ""))
    cb.ax.tick_params(labelsize=8)
    med = float(np.nanmedian(basins["P_annual"]))
    ax.set_title("Annual probability of a postfire debris flow\n"
                 f"P(F) × P(R>T) · median basin ≈ 1 in {1/med:,.0f} yr",
                 fontsize=10.5)

    for ax in axes:
        mc.draw_context(ax, ctx, label_cities=False, label_rivers=False)
        if len(pl):
            ax.scatter(pl.geometry.x, pl.geometry.y, s=16, color="black",
                       edgecolor="white", linewidth=0.6, zorder=9)
            for _, r in pl.iterrows():
                ax.annotate(str(r["name"]), (r.geometry.x, r.geometry.y),
                            xytext=(4, 3), textcoords="offset points",
                            fontsize=7.5, zorder=9.1, color="black",
                            path_effects=mc.fire_style("calibration")["path_effects"])
        mc.style_axes(ax, extent, step=z["step"])

    fig.suptitle(
        f"{z['label']} — firescape {VERSION} pre-fire debris-flow hazard\n"
        f"{z['blurb']} · {len(basins):,} basins · "
        f"{int(counts.get(2, 0)):,} moderate, {int(counts.get(3, 0)):,} high",
        y=0.98, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
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
