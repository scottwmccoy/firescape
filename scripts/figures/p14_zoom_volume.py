"""Zoom volume sheets -> figures/zoom_<key>_volume_<version>.pdf.

The third sheet over the same windows as :mod:`p10_urban_zooms` (annualized
triptych) and :mod:`p12_urban_zoom_forecast` (triggering intensity). Those two
answer *how often* and *how hard must it rain*; this one answers **how much
material arrives**, which is the number a debris basin is sized on.

**Basins, not segments.** RANGES is a basin-scale regression -- its predictors
are basin area, basin mean slope and basin north-facing fraction, and it was
fit to basin-integrated deposit volumes. The segment files carry a Gartner
volume per segment, and the same trick does not transfer: a segment's "volume"
under a basin model is a category error, not merely a smaller number. So this
sheet rasterizes the outlet-basin polygons the way the annualized triptych
does, and the forecast sheet's network view has no volume counterpart.

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

    python scripts/figures/p14_zoom_volume.py                  # every window
    python scripts/figures/p14_zoom_volume.py hidden_valley
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

VERSION = sys.argv[2] if len(sys.argv) > 2 else "statewide_v1_2"
name = sys.argv[1] if len(sys.argv) > 1 else None
todo = [name] if name else list(cor.ZOOMS)
GP = paths.products_dir("prefire", VERSION) / f"{VERSION}_basins.gpkg"

HAZ = ListedColormap(["#4B9B6E", "#E8A33D", "#C1272D"])   # low / mod / high
HNORM = BoundaryNorm([0.5, 1.5, 2.5, 3.5], HAZ.N)

#: One-RMSE ln-space band on every RANGES volume (hazard.RANGES_RMSE_LN).
RMSE_LO, RMSE_HI = np.exp(-hz.RANGES_RMSE_LN), np.exp(hz.RANGES_RMSE_LN)

#: Volume ticks a reader thinks in: truckloads, then debris-basin capacities.
V_TICKS = [1, 3, 10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000]

REQUIRED = ["V_24mmh", "Vg14_24mmh", "H_24mmh", "P_24mmh", "Area_km2"]


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
    basins = read_dataframe(GP, bbox=bbox5070).to_crs("EPSG:4326")
    if basins.empty:
        sys.exit(f"no basins within {bounds}")
    missing = [c for c in REQUIRED if c not in basins.columns]
    if missing:
        sys.exit(f"{GP.name} lacks {missing} -- run "
                 f"scripts/surface/sw_repair_ranges.py {VERSION}")
    print(f"{len(basins):,} basins in the window", flush=True)

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
    arr_v = mc.burn(basins, "V_24mmh", tr, shape)
    arr_g = mc.burn(basins, "Vg14_24mmh", tr, shape)
    pool = np.concatenate([a[np.isfinite(a) & (a > 0)] for a in (arr_v, arr_g)])
    vlo, vhi = np.percentile(pool, [2, 98])

    fig, axes = plt.subplots(1, 3, figsize=cor.figure_size(bounds, 3), dpi=140)
    volume_panel(axes[0], fig, arr_v, extent, hs,
                 title="RANGES volume\nMcCoy 5-variable, headline since v1.1",
                 cmap="cividis", vlo=vlo, vhi=vhi)
    volume_panel(axes[1], fig, arr_g, extent, hs,
                 title="Gartner-14 volume\nthe emergency-assessment model, "
                       "for comparison", cmap="cividis", vlo=vlo, vhi=vhi)

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
    ticks = [t for t in (0.1, 0.2, 0.33, 0.5, 1, 2, 3, 5, 10)
             if 1.0 / span <= t <= span]
    cb.ax.yaxis.set_major_locator(FixedLocator(ticks))
    cb.ax.yaxis.set_major_formatter(FuncFormatter(
        lambda x, _: (f"{x:g}x" if x >= 1 else f"1/{1/x:.3g}")))
    cb.set_label("RANGES ÷ Gartner-14", fontsize=8.5)
    ax.set_title("Where the two models disagree\n"
                 "purple = RANGES smaller · orange = RANGES larger",
                 fontsize=10.5)

    for a in axes:
        cor.decorate(a, C, extent, step=z["step"])

    med_v, med_g = float(np.nanmedian(V)), float(np.nanmedian(G))
    ratio = basins["V_ratio"].to_numpy()
    ratio = ratio[np.isfinite(ratio)]
    counts = basins["H_24mmh"].value_counts()
    hg = hz.combined_c10(basins["P_24mmh"].to_numpy(), G)
    nan_v = int((~np.isfinite(V)).sum())

    fig.tight_layout(w_pad=0.4)
    cor.suptitle(fig, [line for line in (
        f"{z['label']} — firescape {VERSION} debris-flow volume "
        f"at the 24 mm/h reference storm",
        z["blurb"],
        f"{len(basins):,} basins · RANGES median {med_v:,.0f} m³ vs "
        f"Gartner-14 {med_g:,.0f} m³ · ratio median {np.median(ratio):.2f}× "
        f"(5–95%: {np.percentile(ratio, 5):.2f}–{np.percentile(ratio, 95):.2f}×)",
        f"RANGES ±1 RMSE spans ×{RMSE_LO:.2f} to ×{RMSE_HI:.2f} — a factor of "
        f"{RMSE_HI/RMSE_LO:.0f} — and no Nevada fire is in its training set; "
        f"Utah, the nearest Great Basin analog, is overpredicted ~2.8×",
        (f"{nan_v:,} basins have no RANGES volume (missing a predictor) and "
         f"are drawn blank" if nan_v else ""),
    ) if line])
    mc.save(fig, f"zoom_{key}_volume_{VERSION}")
    plt.close(fig)

    stats = {
        "zoom": key, "label": z["label"], "version": VERSION,
        "bounds": list(bounds), "basins": int(len(basins)),
        "volume_model": "RANGES (headline); Gartner-14 as Vg14_24mmh",
        "median_V_ranges_m3": round(med_v, 1),
        "median_V_gartner_m3": round(med_g, 1),
        "p90_V_ranges_m3": round(float(np.nanpercentile(V, 90)), 1),
        "ratio_median": round(float(np.median(ratio)), 3),
        "ratio_p05": round(float(np.percentile(ratio, 5)), 3),
        "ratio_p95": round(float(np.percentile(ratio, 95)), 3),
        "basins_without_ranges_volume": nan_v,
        "hazard_class_ranges": {str(int(k)): int(v) for k, v in counts.items()},
        "hazard_class_gartner": {str(c): int((hg == c).sum()) for c in (1, 2, 3)},
        "rmse_band": [round(float(RMSE_LO), 3), round(float(RMSE_HI), 3)],
    }
    out = (paths.products_dir("prefire", VERSION)
           / f"zoom_{key}_volume_summary.json")
    out.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)
