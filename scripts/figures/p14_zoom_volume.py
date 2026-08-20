"""Zoom volume sheets -> figures/zoom_<key>_volume_<version>.pdf.

The third sheet over the same windows as :mod:`p10_urban_zooms` (annualized
triptych) and :mod:`p12_urban_zoom_forecast` (triggering intensity). Those two
answer *how often* and *how hard must it rain*; this one answers **how much
material arrives**, which is the number a debris basin is sized on.

**Two views, same numbers.** Default is outlet-basin polygons, rasterized the
way the annualized triptych does it. ``--segments`` draws the stream network
instead, the way the forecast sheet does, which is the view at which a single
channel above a subdivision can be pointed at.

RANGES is legitimate on either. Its predictors are catchment quantities, and
``prefire`` emits exactly those per segment -- ``Area_km2`` is
``segments.area()``, ``SlopeDeg``/``FracNorth`` come from pfdf's
``catchment_summary``/``catchment_ratio`` -- so every segment already carries
the statistics of the basin draining to *its* outlet, the same footing Gartner
uses there. Both layers are written by ``sw_ranges_segments``/
``sw_repair_ranges``; nothing is recomputed here.

**Both models, and their ratio.** RANGES became the headline volume in v1.1,
but Gartner-14 is what every earlier figure showed and what the USGS emergency
assessments use, so a reader needs to see the size of the change and where it
falls rather than take a median on faith. Statewide the medians are 67 m3
(RANGES) against 225 m3 (Gartner) -- but the ratio is not a constant, and the
third panel is where that shows.

Uncertainty is deliberately NOT a panel. ``volume_ranges`` returns bounds at a
single global RMSE (1.136 ln-units, x0.32 to x3.11), so a Vmin or Vmax map is
the V map times a constant -- an identical picture under a relabelled bar.
The band belongs in the subtitle, where it is read once and applies to every
basin, and it is wide: a factor of ten separates the lower bound from the
upper.

**Read the ratio panel before reading the medians.** RANGES ÷ Gartner is not a
constant -- it climbs with catchment area and crosses 1 near 0.8 km2, because
RANGES carries area at an exponent of 1.052 while Gartner reaches it through a
burned-area term. So RANGES is *smaller* in most basins (which are small) and
*larger* over the big ones (which fill most of the map). A median over basins
and the impression the eye takes off the map therefore point in opposite
directions, both correctly. The subtitle leads with the crossover for that
reason: an earlier draft led with the medians and read as though the two
volume panels had been swapped.

    python scripts/figures/p14_zoom_volume.py                  # every window
    python scripts/figures/p14_zoom_volume.py hidden_valley
    python scripts/figures/p14_zoom_volume.py hidden_valley --segments
    python scripts/figures/p14_zoom_volume.py hidden_valley statewide_v1_2
"""
import json
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, TwoSlopeNorm
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter
from pyogrio import read_dataframe
from rasterio.warp import transform_bounds

import _corridors as cor
from firescape import hazard as hz
from firescape import paths, plotting as mc

SEGMENTS = "--segments" in sys.argv
_pos = [a for a in sys.argv[1:] if not a.startswith("-")]
name = _pos[0] if _pos else None
VERSION = _pos[1] if len(_pos) > 1 else "statewide_v1_2"
todo = [name] if name else list(cor.ZOOMS)
PROD = paths.products_dir("prefire", VERSION)
GP = PROD / f"{VERSION}_basins.gpkg"
HU = paths.interim_dir("statewide") / "nv_hu10.geojson"

HAZ = ListedColormap(["#4B9B6E", "#E8A33D", "#C1272D"])   # low / mod / high
HNORM = BoundaryNorm([0.5, 1.5, 2.5, 3.5], HAZ.N)

#: One-RMSE ln-space band on every RANGES volume (hazard.RANGES_RMSE_LN).
RMSE_LO, RMSE_HI = np.exp(-hz.RANGES_RMSE_LN), np.exp(hz.RANGES_RMSE_LN)

#: Volume ticks a reader thinks in: truckloads, then debris-basin capacities.
V_TICKS = [1, 3, 10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000]

REQUIRED = ["V_24mmh", "Vg14_24mmh", "H_24mmh", "P_24mmh", "Area_km2"]


#: Segment width scaled to the window, as p12_urban_zoom_forecast does -- a
#: constant tuned on a 2-degree corridor is invisible on a 0.14-degree zoom.
SEG_LW_REF, SPAN_REF, SEG_LW_MAX = 0.4, 1.21, 2.2


def seg_linewidth(bounds):
    span = max(bounds[2] - bounds[0], 1e-6)
    return float(np.clip(SEG_LW_REF * np.sqrt(SPAN_REF / span),
                         SEG_LW_REF, SEG_LW_MAX))


def segments_in(bounds):
    """Modelled stream segments in the window, as one lon/lat frame."""
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box

    hu = gpd.read_file(HU).to_crs("EPSG:4326")
    keys = hu[hu.intersects(box(*bounds))]["huc10"].astype(str)
    bbox5070 = transform_bounds("EPSG:4326", "EPSG:5070", *bounds)
    parts = []
    for k in sorted(keys):
        f = PROD / f"{k}_segments.gpkg"
        if not f.exists():          # all-playa units write nothing
            continue
        g = read_dataframe(f, bbox=bbox5070)
        if len(g):
            parts.append(g)
    if not parts:
        sys.exit(f"no segment files with data inside {bounds}")
    seg = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True),
                           geometry="geometry", crs=parts[0].crs)
    return seg.to_crs("EPSG:4326").explode(index_parts=False)


def line_panel(ax, fig, seg, col, extent, hs, *, title, cmap, vlo, vhi, lw,
               norm=None, label="volume at the 24 mm/h reference storm (m$^3$)",
               ticks=V_TICKS, fmt=None):
    """One network panel. Drawn small-to-large so the biggest volumes end up
    on top rather than buried under their neighbours at a confluence."""
    from matplotlib.collections import LineCollection

    ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=extent, zorder=0)
    d = seg[np.isfinite(seg[col])].sort_values(col)
    blank = seg[~np.isfinite(seg[col])]
    if len(blank):
        ax.add_collection(LineCollection(
            [np.asarray(g.coords) for g in blank.geometry],
            colors="#7A7A7A", linewidths=lw, zorder=4.5, rasterized=True))
    im = ax.add_collection(LineCollection(
        [np.asarray(g.coords) for g in d.geometry],
        array=d[col].to_numpy(), cmap=cmap,
        norm=norm or LogNorm(vmin=vlo, vmax=vhi),
        linewidths=lw, zorder=5, rasterized=True))
    cb = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.02, extend="both")
    cb.ax.tick_params(labelsize=8)
    cb.ax.yaxis.set_minor_locator(FixedLocator([]))
    cb.ax.yaxis.set_major_locator(
        FixedLocator([t for t in ticks if (norm or LogNorm(vmin=vlo, vmax=vhi)
                                           ).vmin <= t <= (norm or LogNorm(
                                               vmin=vlo, vmax=vhi)).vmax]))
    cb.ax.yaxis.set_major_formatter(FuncFormatter(
        fmt or (lambda x, _: f"{x:,.0f}" if x >= 1 else f"{x:g}")))
    cb.set_label(label, fontsize=8.5)
    ax.set_title(title, fontsize=10.5)


def volume_panel(ax, fig, arr, extent, hs, *, title, cmap, vlo, vhi):
    ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=extent, zorder=0)
    im = ax.imshow(np.ma.masked_invalid(arr), cmap=cmap,
                   norm=LogNorm(vmin=vlo, vmax=vhi), extent=extent,
                   alpha=mc.MAX_LAYER_ALPHA, zorder=2,
                   interpolation="antialiased")
    cb = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.02, extend="both")
    cb.ax.tick_params(labelsize=8)
    cb.ax.yaxis.set_minor_locator(FixedLocator([]))
    cb.ax.yaxis.set_major_locator(
        FixedLocator([t for t in V_TICKS if vlo <= t <= vhi]))
    cb.ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x:,.0f}" if x >= 1 else f"{x:g}"))
    cb.set_label("volume at the 24 mm/h reference storm (m$^3$)", fontsize=8.5)
    ax.set_title(title, fontsize=10.5)


for key in todo:
    z = cor.ZOOMS[key]
    bounds = z["bounds"]
    print(f"\n=== {key}: {z['label']} (volume) ===", flush=True)

    bbox5070 = transform_bounds("EPSG:4326", "EPSG:5070", *bounds)
    if SEGMENTS:
        basins = segments_in(bounds)
        noun, fixer = "segments", "sw_ranges_segments.py"
    else:
        basins = read_dataframe(GP, bbox=bbox5070).to_crs("EPSG:4326")
        noun, fixer = "basins", "sw_repair_ranges.py"
    if basins.empty:
        sys.exit(f"no {noun} within {bounds}")
    missing = [c for c in REQUIRED if c not in basins.columns]
    if missing:
        sys.exit(f"{VERSION} {noun} lack {missing} -- run "
                 f"scripts/surface/{fixer} {VERSION}")
    print(f"{len(basins):,} {noun} in the window", flush=True)

    tr, shape, extent = mc.grid(bounds, res=z["res"])
    print(f"display grid {shape[1]}x{shape[0]} @ {z['res']:g} deg", flush=True)
    hs = mc.hillshade(tr, shape)
    C = cor.context(key)

    V = basins["V_24mmh"].to_numpy()          # RANGES (headline since v1.1)
    G = basins["Vg14_24mmh"].to_numpy()       # Gartner-14, soft-Bmh
    both = np.isfinite(V) & np.isfinite(G) & (V > 0) & (G > 0)
    basins["V_ratio"] = np.where(both, V / np.where(G > 0, G, np.nan), np.nan)

    # One log scale across both volume panels, or the eye reads a colour
    # difference that is only a difference of bars.
    #
    # Limits come from the RASTERIZED arrays, not from the basin attribute
    # list. Volume goes as area^1.05, so the few largest basins hold both the
    # largest volumes and nearly all the drawn pixels: percentiles over 325
    # basin *rows* put the 98th at a value that a handful of basins exceed
    # over most of the panel, and both maps came out uniformly saturated.
    # Percentiles over what is actually displayed is the same rule
    # p10_urban_zooms uses.
    RTICKS = (0.1, 0.2, 0.33, 0.5, 1, 2, 3, 5, 10)
    RFMT = lambda x, _: (f"{x:g}x" if x >= 1 else f"1/{1/x:.3g}")
    T_RANGES = "RANGES volume\nMcCoy 5-variable, headline since v1.1"
    T_G14 = "Gartner-14 volume\nthe emergency-assessment model, for comparison"
    # Colour sense verified against the colormap, not guessed: PuOr runs
    # orange (#7f3b08) at its LOW end to purple (#2d004b) at its HIGH end, so
    # low ratio -- RANGES smaller -- is orange. The first draft of this caption
    # had it backwards, which put a second contradictory cue on a sheet that
    # already reads counterintuitively, and made the volume panels look swapped.
    T_RATIO = ("Where the two models disagree\n"
               "orange = RANGES smaller · purple = RANGES larger")
    fig, axes = plt.subplots(1, 3, figsize=cor.figure_size(bounds, 3), dpi=140)

    if SEGMENTS:
        # Percentiles over segment rows, not over pixels: a line's drawn area
        # goes with its LENGTH, and length is near-independent of catchment
        # volume, so rows and pixels weight the distribution the same way here.
        # (For the basin polygons below they do not, which is the whole reason
        # those limits are taken off the rasterized arrays.)
        pool = np.concatenate([basins[c][np.isfinite(basins[c])
                                         & (basins[c] > 0)].to_numpy()
                               for c in ("V_24mmh", "Vg14_24mmh")])
        vlo, vhi = np.percentile(pool, [2, 98])
        lw = seg_linewidth(bounds)
        print(f"segments at {lw:.2f} pt", flush=True)
        line_panel(axes[0], fig, basins, "V_24mmh", extent, hs, title=T_RANGES,
                   cmap="cividis", vlo=vlo, vhi=vhi, lw=lw)
        line_panel(axes[1], fig, basins, "Vg14_24mmh", extent, hs, title=T_G14,
                   cmap="cividis", vlo=vlo, vhi=vhi, lw=lw)
        rr = basins["V_ratio"][np.isfinite(basins["V_ratio"])]
        span = max(float(np.percentile(rr, 98)),
                   1.0 / max(float(np.percentile(rr, 2)), 1e-6)) if len(rr) else 3.0
        line_panel(axes[2], fig, basins, "V_ratio", extent, hs, title=T_RATIO,
                   cmap="PuOr", vlo=1.0 / span, vhi=span, lw=lw,
                   norm=LogNorm(vmin=1.0 / span, vmax=span),
                   label="RANGES ÷ Gartner-14", ticks=RTICKS, fmt=RFMT)
    else:
        arr_v = mc.burn(basins, "V_24mmh", tr, shape)
        arr_g = mc.burn(basins, "Vg14_24mmh", tr, shape)
        pool = np.concatenate([a[np.isfinite(a) & (a > 0)]
                               for a in (arr_v, arr_g)])
        vlo, vhi = np.percentile(pool, [2, 98])
        volume_panel(axes[0], fig, arr_v, extent, hs, title=T_RANGES,
                     cmap="cividis", vlo=vlo, vhi=vhi)
        volume_panel(axes[1], fig, arr_g, extent, hs, title=T_G14,
                     cmap="cividis", vlo=vlo, vhi=vhi)

        ax = axes[2]
        ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=extent, zorder=0)
        arr = mc.burn(basins, "V_ratio", tr, shape)
        finite = arr[np.isfinite(arr) & (arr > 0)]
        hi = float(np.nanpercentile(finite, 98)) if finite.size else 3.0
        lo = float(np.nanpercentile(finite, 2)) if finite.size else 0.3
        span = max(hi, 1.0 / max(lo, 1e-6))
        im = ax.imshow(np.ma.masked_invalid(arr), cmap="PuOr",
                       norm=LogNorm(vmin=1.0 / span, vmax=span), extent=extent,
                       alpha=mc.MAX_LAYER_ALPHA, zorder=2,
                       interpolation="antialiased")
        cb = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.02, extend="both")
        cb.ax.tick_params(labelsize=8)
        cb.ax.yaxis.set_minor_locator(FixedLocator([]))
        cb.ax.yaxis.set_major_locator(
            FixedLocator([t for t in RTICKS if 1.0 / span <= t <= span]))
        cb.ax.yaxis.set_major_formatter(FuncFormatter(RFMT))
        cb.set_label("RANGES ÷ Gartner-14", fontsize=8.5)
        ax.set_title(T_RATIO, fontsize=10.5)

    for a in axes:
        cor.decorate(a, C, extent, step=z["step"])

    med_v, med_g = float(np.nanmedian(V)), float(np.nanmedian(G))
    ratio = basins["V_ratio"].to_numpy()
    ratio = ratio[np.isfinite(ratio)]
    counts = basins["H_24mmh"].value_counts()
    hg = hz.combined_c10(basins["P_24mmh"].to_numpy(), G)
    nan_v = int((~np.isfinite(V)).sum())

    # Where the two models change places, measured rather than asserted.
    area = basins["Area_km2"].to_numpy()
    fit_ok = np.isfinite(basins["V_ratio"]) & (area > 0)
    if fit_ok.sum() > 10:
        la, lr = np.log(area[fit_ok]), np.log(basins["V_ratio"][fit_ok])
        slope, icpt = np.polyfit(la, lr, 1)
        cross = float(np.exp(-icpt / slope)) if slope else np.nan
        corr = float(np.corrcoef(la, lr)[0, 1])
    else:
        cross, corr, slope = np.nan, np.nan, np.nan
    X_SMALL, X_BIG = 0.1, 2.0

    # The crossover line comes FIRST, and the medians hang off it. Led the
    # other way round, the subtitle said "RANGES median 166 vs Gartner 409"
    # over a map whose large catchments are visibly brighter under RANGES,
    # and the honest reading was that the panels had been mislabelled. Both
    # facts are true at once; only the order makes them legible.
    small = area < X_SMALL
    big = area > X_BIG
    r_small = (float(np.nanmedian(basins["V_ratio"][small]))
               if small.sum() else np.nan)
    r_big = (float(np.nanmedian(basins["V_ratio"][big]))
             if big.sum() else np.nan)

    fig.tight_layout(w_pad=0.4)
    cor.suptitle(fig, [line for line in (
        f"{z['label']} — firescape {VERSION} debris-flow volume "
        f"at the 24 mm/h reference storm",
        z["blurb"],
        f"RANGES ÷ Gartner-14 climbs with catchment area — {r_small:.2f}× below "
        f"{X_SMALL:g} km², {r_big:.2f}× above {X_BIG:g} km², crossing 1× near "
        f"{cross:.2g} km² (corr of logs {corr:+.2f}). So RANGES is smaller in "
        f"{100*(ratio < 1).mean():.0f}% of {noun} — which are the small ones — "
        f"and larger over the big ones that cover most of the map.",
        f"{len(basins):,} {noun} · medians (per {noun[:-1]}, so weighted to the "
        f"small): RANGES {med_v:,.0f} m³ vs Gartner-14 {med_g:,.0f} m³",
        f"RANGES ±1 RMSE spans ×{RMSE_LO:.2f} to ×{RMSE_HI:.2f} — a factor of "
        f"{RMSE_HI/RMSE_LO:.0f} — and no Nevada fire is in its training set; "
        f"Utah, the nearest Great Basin analog, is overpredicted ~2.8×",
        (f"{nan_v:,} {noun} have no RANGES volume (off the Atlas 14 grid) and "
         f"are drawn grey" if nan_v else ""),
    ) if line])
    mc.save(fig, f"zoom_{key}_volume{'_segments' if SEGMENTS else ''}_{VERSION}")
    plt.close(fig)

    stats = {
        "zoom": key, "label": z["label"], "version": VERSION,
        "bounds": list(bounds), "view": noun, "n": int(len(basins)),
        "volume_model": "RANGES (headline); Gartner-14 as Vg14_24mmh",
        "median_V_ranges_m3": round(med_v, 1),
        "median_V_gartner_m3": round(med_g, 1),
        "p90_V_ranges_m3": round(float(np.nanpercentile(V, 90)), 1),
        "ratio_median": round(float(np.median(ratio)), 3),
        "ratio_p05": round(float(np.percentile(ratio, 5)), 3),
        "ratio_p95": round(float(np.percentile(ratio, 95)), 3),
        "without_ranges_volume": nan_v,
        "ratio_vs_area_crossover_km2": (round(cross, 3)
                                        if np.isfinite(cross) else None),
        "ratio_vs_area_logcorr": (round(corr, 3)
                                  if np.isfinite(corr) else None),
        "hazard_class_ranges": {str(int(k)): int(v) for k, v in counts.items()},
        "hazard_class_gartner": {str(c): int((hg == c).sum()) for c in (1, 2, 3)},
        "rmse_band": [round(float(RMSE_LO), 3), round(float(RMSE_HI), 3)],
    }
    suffix = "_segments" if SEGMENTS else ""
    out = (paths.products_dir("prefire", VERSION)
           / f"zoom_{key}_volume{suffix}_summary.json")
    out.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)
