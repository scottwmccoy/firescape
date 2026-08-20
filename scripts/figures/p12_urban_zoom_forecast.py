"""Regional zooms in the fire-forecast format: the drainage network itself.

:mod:`p10_urban_zooms` answers "how often should this happen", which is the
right question for a hazard rate but the wrong one for anybody deciding where
to put a debris basin. This sheet asks the question the post-fire assessments
ask -- **if this drainage burned today, how hard does it have to rain, and how
bad is it at the reference storm** -- over the same two windows, in the same
two-panel form as ``stallion_forecast`` and ``bug_forecast``.

Two things change from the triptych, and both are deliberate:

* **Segments, not basins.** The annualized sheets rasterize the *outlet* basin
  polygons -- 33k of them in the Reno window. This draws the full stream
  network beneath them: 307k segments in Reno, 604k in Las Vegas. That is what
  a USGS assessment sheet shows, and it is the only view at which a single
  channel above a subdivision can be picked out and pointed at.
* **Conditional on burning.** No P(F), no annual rate. Triggering intensity
  and hazard class are what they are the day after a fire.

Everything else -- window, place names, watercourses, line weights -- is
shared with the triptych through :mod:`_corridors`, so the two sheets can be
laid side by side and any difference between them is a difference in the data.

Segment collections are drawn ``rasterized=True``: at 600k lines a vector PDF
would be several hundred megabytes to no purpose, since each segment is a few
pixels long at print scale. Labels, boundaries and axes stay vector.
"""
import json
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from rasterio.warp import transform_bounds
from shapely.geometry import box

import geopandas as gpd

import _corridors as cor
from firescape import hazard as hz, paths, plotting as mc

VERSION = sys.argv[2] if len(sys.argv) > 2 else "statewide_v1_2"
name = sys.argv[1] if len(sys.argv) > 1 else None
todo = [name] if name else list(cor.ZOOMS)

PROD = paths.products_dir("prefire", VERSION)
HU = paths.interim_dir("statewide") / "nv_hu10.geojson"

HAZ = ListedColormap(["#4B9B6E", "#E8A33D", "#C1272D"])   # low / mod / high
NORM = BoundaryNorm([0.5, 1.5, 2.5, 3.5], HAZ.N)
UNSUPPORTED = "#7A7A7A"   # neutral: a channel the model has no input for

#: Segment line width, tuned on the corridor windows (1.2-2.5 deg across,
#: 300-600k segments crowding one panel). A fixed value does not survive a
#: change of scale: the Hidden Valley window is 0.14 deg with 3.4k segments --
#: two orders of magnitude fewer over the same physical panel -- and 0.4 pt
#: hairlines disappeared into the hillshade, on the one sheet whose entire
#: subject is the network. Same failure as the hillshade constant that
#: `plotting.shade_resolution` now derives per window, so derive this one too.
#:
#: Square-root of the span ratio, not linear: halving the window doubles a
#: segment's drawn length but only thins it by the same factor in one
#: dimension, and a linear rule turned the Hidden Valley network into ribbons
#: wider than the drainages they trace.
#:
#: ``SPAN_REF`` is the narrowest window that existed when the rule was written
#: (reno_carson, 1.21 deg), so every corridor sheet clips to exactly
#: ``SEG_LW_REF`` and re-renders unchanged. Setting it to a round 1.5 would
#: have quietly thickened Reno's network by 11% -- a restyle of a published
#: sheet, smuggled in as a fix for a different window.
SEG_LW_REF, SPAN_REF, SEG_LW_MAX = 0.4, 1.21, 2.2


def seg_linewidth(bounds):
    span = max(bounds[2] - bounds[0], 1e-6)
    return float(np.clip(SEG_LW_REF * np.sqrt(SPAN_REF / span),
                         SEG_LW_REF, SEG_LW_MAX))


def statewide_median_kf():
    """The KF the merge used to fill SSURGO gaps.

    The per-unit segment files were written before that fill, so ~0.6% of them
    carry a NaN ``Soil_M1`` and therefore no likelihood at all. Filling them
    the same way the published basin layer was filled keeps the two corridor
    sheets consistent with each other; reading the column back (no geometry,
    ~1 s over 412k rows) keeps it consistent with the product rather than with
    a number pasted into this script.
    """
    g = pyogrio.read_dataframe(PROD / f"{VERSION}_basins.gpkg",
                               columns=["Soil_M1"], read_geometry=False)
    return float(np.nanmedian(g["Soil_M1"]))


def segments_in(bounds):
    """Every modelled stream segment in the window, as one lon/lat frame."""
    hu = gpd.read_file(HU).to_crs("EPSG:4326")
    keys = hu[hu.intersects(box(*bounds))]["huc10"].astype(str)
    bbox5070 = transform_bounds("EPSG:4326", "EPSG:5070", *bounds)
    parts, missing = [], []
    for k in sorted(keys):
        f = PROD / f"{k}_segments.gpkg"
        if not f.exists():
            # Not a failure: a unit that is all playa (Smoke Creek Desert, in
            # the Reno window) is masked out entirely and writes nothing.
            missing.append(k)
            continue
        g = pyogrio.read_dataframe(
            f, bbox=bbox5070,
            columns=["Area_km2", "Terrain_M1", "Fire_M1", "Soil_M1",
                     "V_24mmh", "I15_50", "H_24mmh"])
        if len(g):
            parts.append(g)
    if not parts:
        sys.exit(f"no segment files with data inside {bounds}")
    if missing:
        print(f"  {len(missing)} unit(s) modelled to nothing (all valley or "
              f"playa): {', '.join(missing)}", flush=True)
    seg = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True),
                           geometry="geometry", crs=parts[0].crs)
    return seg.to_crs("EPSG:4326")


for key in todo:
    z = cor.ZOOMS[key]
    bounds = z["bounds"]
    print(f"\n=== {key}: {z['label']} (forecast format) ===", flush=True)

    seg = segments_in(bounds)
    print(f"{len(seg):,} segments in the window", flush=True)

    # Same gap fill and same recomputation as the statewide merge.
    S = seg["Soil_M1"].to_numpy().copy()
    gap = ~np.isfinite(S)
    if gap.any():
        S[gap] = statewide_median_kf()
        print(f"KF gaps filled: {int(gap.sum()):,} segments "
              f"({gap.mean():.2%}) at {S[gap][0]:.3f}", flush=True)
    T, F = seg["Terrain_M1"].to_numpy(), seg["Fire_M1"].to_numpy()
    seg["P_24mmh"] = hz.likelihood_m1(T, F, S)
    seg["I15_50"] = hz.threshold_i15(T, F, S, p=0.5)
    seg["H_24mmh"] = hz.combined_c10(seg["P_24mmh"].to_numpy(),
                                     seg["V_24mmh"].to_numpy())

    # Segments M1 has no real input for. The terrain term is exactly zero
    # where no upslope cell reaches 23 degrees, and over the Test Site SSURGO
    # never mapped the soil, so on that ground the likelihood is computed from
    # a filled statewide-median K and a simulated severity alone -- the soil
    # term carries ~91% of it. Those are drawn grey rather than coloured: the
    # channel is still real, the number over it is not supported by anything
    # measured there. Data-driven, so it is 15% of the NNSS window and ~0% of
    # Elko, without the sheet special-casing a region.
    seg["unsupported"] = gap & (T == 0)
    uns, sup = seg[seg["unsupported"]], seg[~seg["unsupported"]]
    uns_km = float(uns.to_crs("EPSG:5070").length.sum() / 1e3)
    if len(uns):
        print(f"no model input: {len(uns):,} segments ({len(uns)/len(seg):.1%}, "
              f"{uns_km:,.0f} km) drawn grey", flush=True)

    tr, shape, extent = mc.grid(bounds, res=z["res"])
    seg_lw = seg_linewidth(bounds)
    print(f"display grid {shape[1]}x{shape[0]} @ {z['res']:g} deg, "
          f"segments at {seg_lw:.2f} pt", flush=True)
    hs = mc.hillshade(tr, shape)   # shade resolution follows the zoom
    C = cor.context(key)
    im_extent = (extent[0], extent[1], extent[2], extent[3])

    # Every reported number is over the supported segments only, so the colour
    # scale is not stretched by ground the model cannot speak about.
    thr = sup["I15_50"].to_numpy()
    lo, hi = np.percentile(thr[np.isfinite(thr)], [2, 98])
    counts = sup["H_24mmh"].value_counts()
    med_thr = float(np.nanmedian(thr))
    net_km = float(seg.to_crs("EPSG:5070").length.sum() / 1e3)

    fig, axes = plt.subplots(1, 2, figsize=cor.figure_size(bounds, 2), dpi=140)

    for ax, mode in zip(axes, ("threshold", "hazard")):
        ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=im_extent, zorder=0)
        # Lines are drawn opaque. The alpha ceiling exists so terrain reads
        # through a data layer; the line covers so little of the canvas
        # that the hillshade is never buried, and washing it out would cost
        # the colour resolution the panel is for.
        if len(uns):
            uns.plot(ax=ax, color=UNSUPPORTED, linewidth=seg_lw, zorder=4.5,
                     rasterized=True)
        if mode == "threshold":
            # descending, so the channels that respond to the SMALLEST storm
            # end up drawn on top rather than buried under their neighbours
            d = sup.sort_values("I15_50", ascending=False)
            d.plot(ax=ax, column="I15_50", cmap="plasma_r", linewidth=seg_lw,
                   vmin=lo, vmax=hi, zorder=5, legend=True, rasterized=True,
                   legend_kwds={"shrink": 0.55, "pad": 0.02, "extend": "both",
                                "label": "triggering $I_{15}$ (mm/h) "
                                         "at 50% likelihood"})
            fig.axes[-1].tick_params(labelsize=8)     # geopandas' colourbar
            ax.set_title("Rainfall intensity that triggers a debris flow\n"
                         "lower = responds to a smaller storm", fontsize=10.5)
        else:
            d = sup.sort_values("H_24mmh")          # high class drawn last
            d.plot(ax=ax, column="H_24mmh", cmap=HAZ, norm=NORM,
                   linewidth=seg_lw, zorder=5, rasterized=True)
            # A discrete colourbar rather than the in-map legend the
            # single-fire sheets carry. Both panels are aspect-locked, so a
            # colourbar on only one of them steals width from that panel
            # alone and the pair stops being the same size -- which is
            # exactly what a reader compares them by.
            sm = plt.cm.ScalarMappable(cmap=HAZ, norm=NORM)
            cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.02, ticks=[1, 2, 3])
            cb.ax.set_yticklabels(["low", "moderate", "high"], fontsize=8)
            cb.set_label("combined hazard class (Cannon et al. 2010)")
            ax.set_title("Combined hazard class at the 24 mm/h reference storm\n"
                         "(≈1-year, 15-minute intensity)", fontsize=10.5)
        cor.decorate(ax, C, extent, step=z["step"], extra_handles=(
            [Line2D([0], [0], color=UNSUPPORTED, lw=1.4,
                    label="no model input (see note)")] if len(uns) else []))

    q = np.nanpercentile(thr, [5, 95])
    fig.tight_layout(w_pad=0.4)
    cor.suptitle(fig, [line for line in (
        f"{z['label']} — firescape {VERSION} pre-fire debris-flow forecast",
        f"{z['blurb']} · {len(seg):,} stream segments, "
        f"{net_km:,.0f} km of channel",
        f"median triggering $I_{{15}}$ {med_thr:.0f} mm/h "
        f"(5–95%: {q[0]:.0f}–{q[1]:.0f}) · "
        f"{int(counts.get(2, 0)):,} moderate, {int(counts.get(3, 0)):,} high "
        "— conditional on the basin burning",
        cor.kf_note(float(gap.mean()), "segments"),
        cor.unsupported_note(len(uns), uns_km, len(seg)),
    ) if line])
    mc.save(fig, f"zoom_{key}_forecast_{VERSION}")
    plt.close(fig)

    stats = {
        "zoom": key, "label": z["label"], "version": VERSION,
        "format": "forecast (segment network, conditional on burning)",
        "bounds": list(bounds), "segments": int(len(seg)),
        "network_length_km": round(net_km, 1),
        "hazard_class": {str(int(k)): int(v) for k, v in counts.items()},
        "kf_gap_filled": int(gap.sum()),
        "unsupported_segments": int(len(uns)),
        "unsupported_length_km": round(uns_km, 1),
        "stats_over": "supported segments only (unsupported drawn grey)",
        "I15_50_mmh": {"p05": round(float(q[0]), 1),
                       "median": round(med_thr, 1),
                       "p95": round(float(q[1]), 1)},
        "median_P_24mmh": round(float(np.nanmedian(sup["P_24mmh"])), 3),
        "segments_under_20_mmh": int((thr < 20).sum()),
    }
    out = PROD / f"zoom_{key}_forecast_summary.json"
    out.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)
