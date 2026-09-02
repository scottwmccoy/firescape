"""Post-fire debris-flow hazard on OBSERVED severity -> figures/<fire>_postfire_hazard.pdf.

The emergency-assessment sheet, in the form the USGS publishes it: what burned,
how likely a debris flow is under the **design storm**, and how hard it has to
rain before one is as likely as not. All three panels are conditional on the
burn that BRISK actually measured -- no simulated severity, and no observed
rainfall.

**Deliberately not a storm-response map.** Its predecessor,
``p9_fire_observed_map``, drew likelihood under the 12-14 Aug rainfall pulled
from MRMS; that field was wrong over these fires, and a hazard map inherits
every error in its rainfall. The design storm (24 mm/h I15, the USGS reference
and about a 1-year intensity here) is a *climatological* input, so this sheet
cannot be wrong in that particular way. What it gives up is the answer to
"what should have happened last week"; what it buys is a product that stays
valid until the severity changes.

Panels:

1. **Observed severity** -- BRISK dNBR classed at the run's OWN calibrated BARC
   breaks, in the four published BAER colours, so a hillside that reads
   "moderate" here is moderate to the model that used it.
2. **Likelihood at the design storm** -- P(debris flow) at I15 = 24 mm/h, in
   the five USGS likelihood classes. Same breaks and colours as the CalTopo
   field layers, so the map and the phone agree.
3. **Triggering intensity** -- I15 at which modelled likelihood reaches 50%,
   from inverting the same M1 fit. Low = responds to a small storm. Its colour
   bar carries a **second scale**: the annual chance of a storm that big,
   from the fire's own NOAA Atlas 14 intensity-frequency fit. Atlas 14 makes
   I15 log-linear in recurrence interval, so that scale is monotone in the
   colour ramp and its decades land evenly along the bar -- one ramp, two
   readings of the same segment, and no fourth panel needed. What it is NOT
   is a per-segment annual probability: the fit is the fire median, and the
   spread it hides is measured and quoted in the caption. A per-segment
   P(R>T) map is a different product (see ``annualprob``/M5).

Panels 2 and 3 are two readings of one model, not two models: 24 mm/h into M1
gives panel 2, and p=0.5 back out of M1 gives panel 3.

**No figure title.** Everything that used to sit in the suptitle -- run,
calibration, statistics, provenance, caveats -- is set as a caption under the
panels (``plotting.caption``), the way a journal sets one. Panel titles stay
short. See ``plotting.caption`` for why.

    python scripts/figures/p15_fire_postfire_hazard.py [Fire] [calibration]
"""
import json
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator
from rasterio.warp import Resampling, reproject

import geopandas as gpd
from firescape import paths, plotting as mc
from stormscape import burn, relief

#: ``--prefire`` swaps the severity source and nothing else. The two products
#: MUST stay visually identical: the whole value of a pre-fire sheet is that it
#: can be laid beside the post-fire one for the same ground, and any difference
#: in layout would be read as a difference in the hazard.
PREFIRE = "--prefire" in sys.argv
_pos = [a for a in sys.argv[1:] if not a.startswith("-")]
FIRE = _pos[0] if _pos else "Stallion"
#: Only the pre-fire product is *keyed* by calibration, so only it needs a
#: default here. An observed run is keyed by fire, and the calibration that
#: set its breaks is recorded in its own summary -- read the label from there
#: rather than from this default, which went stale the moment the breaks were
#: recalibrated and put "statewide_v1_2" over a v1_3 map.
_CAL_ARG = _pos[1] if len(_pos) > 1 else None
CAL = _CAL_ARG or "statewide_v1_2"
OBS = paths.products_dir(
    "forecast", f"{FIRE.lower()}_{CAL}" if PREFIRE else f"{FIRE.lower()}_observed")

#: USGS likelihood classes -- the breaks every emergency assessment reports,
#: and the ones already exported to CalTopo for the field team. ColorBrewer
#: RdYlBu reversed: colourblind-safe, and blue reads "cool/unlikely" without
#: anyone needing the legend.
P_BREAKS = (0.2, 0.4, 0.6, 0.8)
P_COLORS = ("#2C7BB6", "#ABD9E9", "#FFFFBF", "#FDAE61", "#D7191C")
P_LABELS = ("< 20%", "20–40%", "40–60%", "60–80%", "≥ 80%")

#: BAER's four published severity colours (teal / cyan / yellow / dark red).
#: ``unburned`` is drawn transparent (see ``_severity_rgba``), so its
#: swatch says so rather than showing a teal no pixel carries.
SEV_LABELS = ("unburned", "low", "moderate", "high")
SEV_LEGEND = ("unburned (not drawn)", "low", "moderate", "high")

DESIGN_I15 = 24.0        # mm/h, the USGS reference storm (~1-yr, 15-minute)

#: Opacity of the severity wash. The BARC classes are categorical -- they have
#: to stay *identifiable*, not exact -- and at full opacity they paint out the
#: terrain that explains where debris flows come from, which is the one thing
#: this panel shares with its two neighbours. 0.60 is the house-style ceiling
#: for data over terrain (repo CLAUDE.md) and every class still reads against
#: the others at it, while ridges and drainages show through.
SEV_ALPHA = 0.60

#: Annual-exceedance ticks for the second scale on the triggering-intensity
#: bar. Atlas 14 makes I15 log-linear in recurrence interval, and P = 1 -
#: exp(-1/RI) is ~1/RI once P is small, so these land at even spacing along
#: the bar -- one decade of rarity per fixed number of mm/h.
P_AXIS_TICKS = (0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)

#: BLM surface-management wash (matches the AML district sheets).
BLM_FILL, BLM_EDGE = "#D9C98C", "#8C7B45"


def _severity_rgba(dnbr, breaks_x1000, shape, dst_transform, dst_crs, src):
    """BRISK dNBR on the display grid, painted in the BAER class colours.

    Reprojected with **nearest**: the array is about to be binned into four
    classes, and bilinear would invent dNBR values on class boundaries that no
    pixel ever held. Unburned is left transparent -- inside a perimeter it
    means nothing happened there, and the terrain underneath is worth more
    than flat paint.
    """
    dst = np.full(shape, np.nan, dtype="float32")
    reproject(dnbr, dst, src_transform=src["transform"], src_crs=src["crs"],
              dst_transform=dst_transform, dst_crs=dst_crs,
              resampling=Resampling.nearest, src_nodata=np.nan,
              dst_nodata=np.nan)
    rgba = np.zeros(shape + (4,), dtype="uint8")
    a = int(round(255 * SEV_ALPHA))
    # BRISK ships raw dNBR (~-0.3 to 1.0); the simulated raster is already
    # x1000, the MTBS convention the breaks are stated in. One scale factor,
    # applied at the one place the two sources meet.
    idx = np.digitize(dst * src["scale"], list(breaks_x1000))
    for i, (r, g, b) in enumerate(burn.BAER_CLASS_COLORS):
        m = np.isfinite(dst) & (idx == i)
        rgba[m] = (r, g, b, 0 if i == 0 else a)
    return rgba, dst


def climatology_fit(seg):
    """Fire-median Atlas 14 I15-vs-recurrence fit, or ``None`` if unavailable.

    NOAA Atlas 14 anchors the log-linear intensity-frequency line per point
    (``annualprob`` Eqns 6-8); sampling it at every segment and taking the
    median gives ONE line for the fire. That is the honest resolution for a
    scale drawn beside a colourbar: within a single perimeter the 1-year I15
    varies by a couple of mm/h, so a per-segment axis would be false
    precision, but the *spread* it hides is measured here and quoted in the
    caption rather than dropped.

    Returns the median slope/intercept, the per-segment pair, and the
    fractional 5-95% spread of P at the fire's median triggering intensity.
    """
    from rasterio.transform import Affine

    from firescape import annualprob as ap
    try:
        z = np.load(paths.interim_dir("statewide") / "atlas14_i15.npz")
    except (FileNotFoundError, OSError) as exc:
        print(f"Atlas 14 grids unavailable ({type(exc).__name__}) — "
              "triggering-intensity bar gets no probability scale", flush=True)
        return None
    i1g, i50g, tr = z["i1"], z["i50"], Affine(*z["transform"])
    cent = seg.geometry.centroid.to_crs("EPSG:4269")
    cols = ((cent.x - tr.c) / tr.a).astype(int).to_numpy()
    rows = ((cent.y - tr.f) / tr.e).astype(int).to_numpy()
    ok = ((rows >= 0) & (rows < i1g.shape[0])
          & (cols >= 0) & (cols < i1g.shape[1]))
    i1 = np.full(len(seg), np.nan)
    i50 = np.full(len(seg), np.nan)
    i1[ok], i50[ok] = i1g[rows[ok], cols[ok]], i50g[rows[ok], cols[ok]]
    m, b = ap.fit_log_linear(i1, i50)
    fin = np.isfinite(m) & np.isfinite(b)
    if fin.mean() < 0.5:
        print(f"Atlas 14 covers only {fin.mean():.0%} of segments — "
              "no probability scale", flush=True)
        return None
    m_med, b_med = float(np.median(m[fin])), float(np.median(b[fin]))
    t_med = float(np.nanmedian(seg["I15_50"].to_numpy()))
    p_seg = ap.annual_probability(
        ap.recurrence_interval(np.full(int(fin.sum()), t_med), m[fin], b[fin]))
    p5, p50, p95 = np.percentile(p_seg, [5, 50, 95])
    return {"m": m_med, "b": b_med, "m_seg": m, "b_seg": b, "covered": float(fin.mean()),
            "i15_1yr_med": float(np.nanmedian(i1)), "i15_50yr_med": float(np.nanmedian(i50)),
            "p_at_median_T": float(p50), "p_spread_frac": float((p95 - p5) / 2 / p50),
            "p_at_median_T_p05": float(p5), "p_at_median_T_p95": float(p95)}


def i15_of_p(p, fit):
    """The 15-minute intensity whose annual exceedance probability is ``p``."""
    ri = -1.0 / np.log(1.0 - np.asarray(p, dtype=float))
    return (np.log10(ri) - fit["b"]) / fit["m"]


def p_of_i15(i15, fit):
    """Annual exceedance probability of a 15-minute intensity."""
    from firescape import annualprob as ap
    return ap.annual_probability(ap.recurrence_interval(i15, fit["m"], fit["b"]))


def inset_colorbar(ax, mappable, label, *, corner="lower left", pad=0.018,
                   w=0.30, h=0.086, nticks=5, fontsize=8, twin=None):
    """A colorbar that sits INSIDE its map, in the corner the panel legends use.

    An external colorbar steals width from the axes it is attached to, so the
    one panel carrying one drew a visibly smaller map than its two neighbours
    -- the same ground at two different scales on one sheet, which invites the
    eye to read a difference that is not in the data. Insetting keeps every map
    box identical and puts the key where a reader is already looking for it.

    Sized in axes fractions, which is safe here because all three panels share
    one data extent and are aspect-locked by ``plotting.style_axes``, so their
    axes boxes are identical. ``corner`` is resolved AFTER the box is sized --
    a right-hand corner has to know the final width, which ``twin`` grows.
    """
    # A second scale needs two more rows of type (its ticks and its label)
    # above the bar, so the box grows and the bar sits lower inside it.
    if twin is not None:
        h, w = max(h, 0.160), max(w, 0.36)
    x, y = mc.corner_xy(corner, (w, h), pad)
    ax.add_patch(Rectangle((x, y), w, h, transform=ax.transAxes,
                           facecolor="white", alpha=0.80, edgecolor="0.8",
                           linewidth=0.8, zorder=11))
    # Horizontal inset leaves room for the extend arrows, which overhang the
    # cax and would otherwise touch the frame.
    bar_h = 0.017
    bar_y = y + (0.030 if twin is None else 0.062)   # room for a label BELOW
    cax = ax.inset_axes([x + 0.045, bar_y, w - 0.090, bar_h], zorder=12)
    cb = ax.figure.colorbar(mappable, cax=cax, orientation="horizontal",
                            extend="both")
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(labelsize=fontsize, length=2.5, width=0.6, pad=1.5)
    cb.locator = MaxNLocator(nbins=nticks - 1)
    cb.update_ticks()
    if twin is None:
        cax.set_title(label, fontsize=fontsize, pad=3.5)
        return cb
    # The second scale is drawn as its own inset on the SAME parent axes, not
    # as cax.twiny(): a twin is placed in figure coordinates at the moment it
    # is made and would drift off the bar as soon as the layout moves (which
    # it does -- the caption reserves its space last). An inset tracks its
    # parent, so the two scales stay locked to one another.
    positions, labels, twin_label = twin
    tax = ax.inset_axes([x + 0.045, bar_y, w - 0.090, bar_h], zorder=12)
    tax.set_xlim(cax.get_xlim())
    tax.patch.set_visible(False)
    tax.set_yticks([])
    for sp in tax.spines.values():
        sp.set_visible(False)
    tax.xaxis.set_ticks_position("top")
    tax.xaxis.set_label_position("top")
    tax.set_xticks(list(positions))
    tax.set_xticklabels(list(labels))
    tax.tick_params(axis="x", labelsize=fontsize, length=2.5, width=0.6, pad=1.5)
    tax.set_title(twin_label, fontsize=fontsize, pad=10.0)
    cax.set_xlabel(label, fontsize=fontsize, labelpad=1.5)
    return cb


if PREFIRE:
    meta = json.loads((OBS / "forecast_summary.json").read_text())
    seg_path = OBS / f"{FIRE.lower()}_segments.gpkg"
    sev_path = OBS / f"{FIRE.lower()}_sim_dnbr_x1000.tif"
    SEV_SCALE = 1.0                      # already x1000
else:
    meta = json.loads((OBS / "observed_severity_summary.json").read_text())
    CAL = _CAL_ARG or meta.get("calibration_for_breaks") or CAL
    seg_path = OBS / f"{FIRE.upper()}_segments.gpkg"
    sev_path = OBS / f"{FIRE.lower()}_brisk_dnbr.tif"
    SEV_SCALE = 1000.0                   # BRISK is raw dNBR
breaks = meta["barc_breaks_x1000"]
seg = gpd.read_file(seg_path).to_crs("EPSG:4326")
per = gpd.read_file(
    sorted(paths.raw_dir("perimeters").glob("wfigs_current_*.geojson"))[-1])
ncol = next(c for c in per.columns if c.endswith("IncidentName"))
per = per[per[ncol].str.fullmatch(FIRE, case=False, na=False)].to_crs("EPSG:4326")

w, s, e, n = per.total_bounds
pad = 0.035
bounds = (w - pad, s - pad, e + pad, n + pad)
tr, shape, extent = mc.grid(bounds, res=0.0002)          # ~20 m display grid
print(f"{len(seg):,} segments · display grid {shape[1]}x{shape[0]}", flush=True)
hs = relief.shaded_relief(
    sorted((paths.cache_root() / "3dep_tiles").glob("USGS_13_*.tif")),
    tr, shape, crs="EPSG:4326", shade_res_m=20.0)
im_extent = tuple(extent)

with rasterio.open(sev_path) as ds:
    _arr = ds.read(1).astype("float64")
    if ds.nodata is not None:            # sim raster carries -9999, not NaN
        _arr[_arr == ds.nodata] = np.nan
    sev_rgba, sev_on_grid = _severity_rgba(
        _arr, breaks, shape, tr, "EPSG:4326",
        {"transform": ds.transform, "crs": ds.crs, "scale": SEV_SCALE})

# The simulated raster covers the whole analysis domain -- the perimeter plus a
# 2 km pad, which in EPSG:5070 renders as a rotated rectangle of "severity"
# over ground that is not burning. Clip it to the perimeter: outside it the
# number is a hypothetical about unburnt hillsides, and drawing it invites
# exactly the wrong reading. Observed severity is NOT clipped -- BRISK is a
# measurement, its footprint is data, and a WFIGS perimeter disagreeing with it
# at the edge is information rather than error.
if PREFIRE:
    from rasterio.features import geometry_mask
    _outside = geometry_mask([per.union_all()], out_shape=shape, transform=tr,
                             invert=False)
    sev_rgba[_outside] = 0
    sev_on_grid = np.where(_outside, np.nan, sev_on_grid)

#: BLM surface-management polygons. The staged file is national in extent
#: (-124.7 to -109.0), not Nevada-only as its name suggests, which matters
#: here: Bug burns on the California side of the line and a state-clipped
#: layer would have drawn nothing over 78% of it.
try:
    from shapely.geometry import box as _box
    blm = gpd.read_file(paths.raw_dir("blm") / "nv_blm_sma.gpkg").to_crs("EPSG:4326")
    blm = gpd.clip(blm, _box(*bounds))
    blm = blm[~blm.geometry.is_empty]
    _fire_blm = gpd.clip(blm, per.union_all())
    BLM_FRAC = float(_fire_blm.to_crs("EPSG:5070").area.sum()
                     / max(per.to_crs("EPSG:5070").area.sum(), 1e-9))
    print(f"BLM: {len(blm)} feature(s) in window, "
          f"{BLM_FRAC:.0%} of the perimeter is BLM land", flush=True)
except Exception as exc:
    print(f"BLM layer unavailable: {type(exc).__name__}: {exc}", flush=True)
    blm, BLM_FRAC = None, None

try:
    from stormscape import refdata
    roads = refdata.roads(bounds)
except Exception as exc:                                   # context is a bonus
    print(f"roads unavailable: {type(exc).__name__}: {exc}", flush=True)
    roads = None

P = seg["P_24mmh"].to_numpy()
T = seg["I15_50"].to_numpy()
lo, hi = np.percentile(T[np.isfinite(T)], [2, 98])
# One key placement for all three panels, chosen from the perimeter: pinning
# every sheet's key to the lower left eventually parks it on the fire (Bug's
# southwestern lobe reaches into that corner, and 31% of the box would land on
# the burn). Same corner on all three panels -- a reader who has found the key
# once should not have to hunt for it again on the panel beside it.
KEY = mc.clear_corner(tuple(extent), per.geometry, size=(0.36, 0.22))
print("key corner: " + KEY["loc"] + " · perimeter overlap "
      + ", ".join(f"{k} {v:.0%}" for k, v in KEY["overlap"].items()), flush=True)

FIT = climatology_fit(seg)
if FIT is not None:
    print(f"Atlas 14 (fire median): 1-yr {FIT['i15_1yr_med']:.1f}, "
          f"50-yr {FIT['i15_50yr_med']:.1f} mm/h · P at the median trigger "
          f"{FIT['p_at_median_T']:.3f} (5-95% across segments "
          f"{FIT['p_at_median_T_p05']:.3f}-{FIT['p_at_median_T_p95']:.3f}, "
          f"±{FIT['p_spread_frac']:.0%})", flush=True)
PCMAP = ListedColormap(P_COLORS)
PNORM = BoundaryNorm([0.0, *P_BREAKS, 1.0], PCMAP.N)

fig, axes = plt.subplots(1, 3, figsize=(25, 10.5), dpi=140)
for ax, mode in zip(axes, ("severity", "likelihood", "threshold")):
    ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=im_extent, zorder=0)
    if roads is not None and len(roads):
        roads.to_crs("EPSG:4326").plot(ax=ax, color="0.25", linewidth=0.5,
                                       alpha=0.7, zorder=2)
    # A translucent fill, not outlines. BLM here is the railroad checkerboard
    # -- hundreds of section-sized parcels -- so stroking every boundary drew
    # a white grid across all three panels and buried the data under its own
    # casing. A wash reads ownership at a glance and competes with nothing.
    # Same tan the AML district sheets use, and drawn UNDER the severity
    # raster and the segments so it can never tint a class colour.
    if blm is not None and len(blm):
        blm.plot(ax=ax, facecolor=BLM_FILL, edgecolor=BLM_EDGE, linewidth=0.35,
                 alpha=0.30, zorder=1.5, rasterized=True)
    if mode == "severity":
        ax.imshow(sev_rgba, extent=im_extent, zorder=3,
                  interpolation="nearest")
        # The context lines ride in this legend rather than getting one of
        # their own on each panel: they are identical on all three, and a
        # repeated key is three chances to read it as three different things.
        _ctx = [Line2D([0], [0], color="#56B4E9", lw=1.8, label="fire perimeter")]
        if blm is not None and len(blm):
            _ctx.append(Patch(facecolor=BLM_FILL, edgecolor=BLM_EDGE,
                              alpha=0.30, label="BLM land"))
        ax.legend(handles=[Patch(facecolor=np.array(c) / 255.0, alpha=SEV_ALPHA,
                                 edgecolor="0.3", label=l)
                           for c, l in zip(burn.BAER_CLASS_COLORS, SEV_LEGEND)]
                          + _ctx,
                  title=f"BARC class (breaks {'/'.join(f'{b/1000:g}' for b in breaks)} dNBR)",
                  loc=KEY["loc"], fontsize=8, title_fontsize=8)
        ax.set_title("Simulated burn severity" if PREFIRE else
                     "Observed burn severity", fontsize=13)
    elif mode == "likelihood":
        # Ascending, so the most likely segments finish on top rather than
        # being overdrawn by a quiet neighbour at a confluence.
        seg.sort_values("P_24mmh").plot(ax=ax, column="P_24mmh", cmap=PCMAP,
                                        norm=PNORM, linewidth=1.6, zorder=5)
        ax.legend(handles=[Line2D([0], [0], color=c, lw=3, label=l)
                           for c, l in zip(P_COLORS, P_LABELS)],
                  title=f"P(debris flow) at {DESIGN_I15:g} mm/h",
                  loc=KEY["loc"], fontsize=8, title_fontsize=8)
        ax.set_title("Debris-flow likelihood at the design storm",
                     fontsize=13)
    else:
        seg.sort_values("I15_50", ascending=False).plot(
            ax=ax, column="I15_50", cmap="plasma_r", linewidth=1.6,
            vmin=lo, vmax=hi, zorder=5, legend=False)
        _sm = ScalarMappable(norm=Normalize(vmin=lo, vmax=hi), cmap="plasma_r")
        _sm.set_array([])
        # Same bar, two readings: how hard it has to rain, and how often it
        # rains that hard here. Both are properties of the segment under the
        # colour, so they belong on one ramp -- a second colour ramp would
        # imply a second field that does not exist.
        _twin = None
        if FIT is not None:
            _pt = [(i15_of_p(q, FIT), f"{q*100:g}%") for q in P_AXIS_TICKS]
            _pt = [(v, lab) for v, lab in _pt if lo <= v <= hi]
            # Ticks are evenly spaced in mm/h (the fit is log-linear), so a
            # wide-range fire like Bug qualifies for all nine and they collide.
            # Thinning by stride keeps them evenly spaced AND keeps the
            # sequence decade-like (50/10/2/0.5/0.1%) rather than ragged.
            if len(_pt) > 5:
                _pt = _pt[::-(-len(_pt) // 5)]
            if len(_pt) >= 2:
                _twin = ([v for v, _ in _pt], [lab for _, lab in _pt],
                         "annual chance of that intensity")
        inset_colorbar(ax, _sm, "triggering $I_{15}$ (mm/h)", twin=_twin,
                       corner=KEY["loc"])
        ax.set_title("Rainfall intensity that triggers a debris flow",
                     fontsize=13)
    per.boundary.plot(ax=ax, color="#56B4E9", linewidth=1.8, zorder=6,
                      path_effects=mc.fire_style("current")["path_effects"])
    mc.style_axes(ax, extent, step=0.1)

age = meta.get("composite_age_days")
if PREFIRE:
    _kind = "Pre-fire debris-flow forecast on simulated severity."
    _sev = ("severity SIMULATED from LANDFIRE vegetation — no burn-severity "
            "imagery exists for this fire — with the whole perimeter burning "
            f"at the {meta.get('region','?')} calibrated quantile "
            f"P$_{{dsim}}$ = {meta.get('pdsim','?')}")
    _tail = ("Real fires leave unburned islands and a severity mosaic, so these "
             "are an upper expectation for the footprint, not an emergency "
             "assessment; the perimeter is today's and an uncontained fire "
             "grows. Supersede with a BRISK or BAER run the moment one lands.")
else:
    _kind = "Post-fire debris-flow hazard on the severity BRISK measured."
    _sev = (f"burn severity from the CIMSS BRISK dNBR composite of "
            f"{', '.join(meta['scene_dates'])}, which maps vegetation change "
            "rather than soil burn severity")
    _stale = ("posted today" if age == 0 else
              f"{age} d old" if age is not None else "of unknown age")
    _tail = (f"The composite is {_stale}; {meta.get('magnitude_caveat', '')}."
             ).replace("; .", ".")

# Class fractions computed from the SAME array the map paints, at the SAME
# breaks. The summary JSON also carries a class_fraction, but it is BRISK's
# own five-class 'usgs' scheme (unburned/low/moderate-low/moderate-high/high),
# not the four BARC classes these breaks define -- quoting it under this legend
# put "14% moderate or high" in the title over a map showing 21%. One binning
# per sheet.
# SEV_SCALE, not a literal: this line carried a hardcoded x1000 from when
# BRISK was the only source, and the pre-fire raster is already x1000 --
# so the map painted correctly while the title claimed 100% high. Same
# number-vs-picture split as the class_fraction bug above, reintroduced
# by generalising the source and not its twin. One scale, one place.
_v = sev_on_grid[np.isfinite(sev_on_grid)] * SEV_SCALE
_idx = np.digitize(_v, list(breaks))
frac = {lab: float((_idx == i).mean()) for i, lab in enumerate(SEV_LABELS)}
modhigh = frac["moderate"] + frac["high"]
print("BARC class fractions on the display grid: "
      + ", ".join(f"{k} {v:.1%}" for k, v in frac.items()), flush=True)
_blm_txt = ("" if BLM_FRAC is None else
            "; tan wash BLM-administered land (none inside the perimeter)"
            if BLM_FRAC < 0.005 else
            f"; tan wash BLM-administered land ({BLM_FRAC:.0%} of the perimeter)")
_clim_txt = (
    "; the upper scale on its colour bar converts that intensity to the annual "
    "chance of such a storm, from the fire-median NOAA Atlas 14 15-minute "
    f"intensity–frequency fit (per-segment fits shift it by ±"
    f"{FIT['p_spread_frac']:.0%} of its value)" if FIT is not None else "")
CAPTION = (
    f"{_kind} {len(seg):,} stream segments; {CAL} calibration, "
    f"{meta.get('region', '?')} region. "
    f"Left: {_sev}, binned at that region's calibrated BARC breaks "
    f"({'/'.join(f'{b/1000:g}' for b in breaks)} dNBR); {modhigh:.0%} of pixels "
    "burned moderate or high. "
    "Middle: modelled likelihood of a debris flow (USGS M1) under the design "
    f"storm, $I_{{15}}$ = {DESIGN_I15:g} mm/h (≈1-year, 15-minute) — median "
    f"{np.nanmedian(P):.2f}, {int((P >= 0.5).sum()):,} segments at or above 50%. "
    "Right: the 15-minute intensity at which that same fit reaches 50% "
    f"likelihood (median {np.nanmedian(T):.1f} mm/h, 5–95% "
    f"{np.nanpercentile(T, 5):.0f}–{np.nanpercentile(T, 95):.0f}), so a low "
    f"value is a channel segment that responds to a small storm{_clim_txt}. "
    f"Hillshade 3DEP 1/3-arcsec{_blm_txt}. {_tail}")

# Margins first, then the caption: it is wrapped to where the panels
# actually are, and it grows the bottom margin to fit itself.
fig.subplots_adjust(top=0.945, bottom=0.055, left=0.03, right=0.97)
mc.caption(fig, CAPTION, label=f"{FIRE} fire.", fontsize=10)
mc.save(fig, f"{FIRE.lower()}_{'prefire' if PREFIRE else 'postfire'}_hazard")
plt.close(fig)

summary = {
    "fire": FIRE, "calibration": CAL,
    "severity_source": ("simulated (LANDFIRE EVT + Staley 2018 CDFs)"
                        if PREFIRE else "CIMSS BRISK"),
    "scene_dates": meta.get("scene_dates"),
    "pdsim": meta.get("pdsim"), "region": meta.get("region"),
    "composite_age_days": age, "barc_breaks_x1000": breaks,
    "design_storm_i15_mmh": DESIGN_I15,
    "segments": int(len(seg)),
    "P_design": {"median": round(float(np.nanmedian(P)), 3),
                 "p90": round(float(np.nanpercentile(P, 90)), 3),
                 "n_ge_0.5": int((P >= 0.5).sum()),
                 "n_ge_0.8": int((P >= 0.8).sum())},
    "I15_50_mmh": {"median": round(float(np.nanmedian(T)), 1),
                   "p05": round(float(np.nanpercentile(T, 5)), 1),
                   "p95": round(float(np.nanpercentile(T, 95)), 1),
                   "n_below_design": int((T < DESIGN_I15).sum())},
    "key_corner": KEY["loc"],
    "key_corner_overlap": {k: round(v, 3) for k, v in KEY["overlap"].items()},
    "blm_fraction_of_perimeter": (round(BLM_FRAC, 4)
                                  if BLM_FRAC is not None else None),
    "class_fraction_barc": {k: round(v, 4) for k, v in frac.items()},
    "annual_exceedance": (None if FIT is None else {
        "source": "NOAA Atlas 14 sw, 15-min, 1-yr & 50-yr (interim/statewide)",
        "fit": "fire-median log10(RI) = m*I15 + b",
        "m": round(FIT["m"], 5), "b": round(FIT["b"], 4),
        "i15_1yr_mmh": round(FIT["i15_1yr_med"], 1),
        "i15_50yr_mmh": round(FIT["i15_50yr_med"], 1),
        "P_at_median_threshold": round(FIT["p_at_median_T"], 4),
        "P_at_median_threshold_p05_p95": [round(FIT["p_at_median_T_p05"], 4),
                                          round(FIT["p_at_median_T_p95"], 4)],
        "segment_coverage": round(FIT["covered"], 3),
    }),
    "caption": CAPTION,
    "class_fraction_brisk_usgs_scheme": meta.get("class_fraction", {}),
}
(OBS / f"{'prefire' if PREFIRE else 'postfire'}_hazard_summary.json"
 ).write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
