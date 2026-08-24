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
   from inverting the same M1 fit. Low = responds to a small storm.

Panels 2 and 3 are two readings of one model, not two models: 24 mm/h into M1
gives panel 2, and p=0.5 back out of M1 gives panel 3.

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

FIRE = sys.argv[1] if len(sys.argv) > 1 else "Stallion"
CAL = sys.argv[2] if len(sys.argv) > 2 else "statewide_v1_2"
OBS = paths.products_dir("forecast", f"{FIRE.lower()}_observed")

#: USGS likelihood classes -- the breaks every emergency assessment reports,
#: and the ones already exported to CalTopo for the field team. ColorBrewer
#: RdYlBu reversed: colourblind-safe, and blue reads "cool/unlikely" without
#: anyone needing the legend.
P_BREAKS = (0.2, 0.4, 0.6, 0.8)
P_COLORS = ("#2C7BB6", "#ABD9E9", "#FFFFBF", "#FDAE61", "#D7191C")
P_LABELS = ("< 20%", "20–40%", "40–60%", "60–80%", "≥ 80%")

#: BAER's four published severity colours (teal / cyan / yellow / dark red).
SEV_LABELS = ("unburned", "low", "moderate", "high")

DESIGN_I15 = 24.0        # mm/h, the USGS reference storm (~1-yr, 15-minute)


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
    idx = np.digitize(dst * 1000.0, list(breaks_x1000))
    for i, (r, g, b) in enumerate(burn.BAER_CLASS_COLORS):
        m = np.isfinite(dst) & (idx == i)
        rgba[m] = (r, g, b, 0 if i == 0 else 255)
    return rgba, dst


def inset_colorbar(ax, mappable, label, *, x=0.018, y=0.018, w=0.30,
                   h=0.086, nticks=5, fontsize=8):
    """A colorbar that sits INSIDE its map, lower-left, where the other panels
    keep their legends.

    An external colorbar steals width from the axes it is attached to, so the
    one panel carrying one drew a visibly smaller map than its two neighbours
    -- the same ground at two different scales on one sheet, which invites the
    eye to read a difference that is not in the data. Insetting keeps every map
    box identical and puts the key where a reader is already looking for it.

    Sized in axes fractions, which is safe here because all three panels share
    one data extent and are aspect-locked by ``plotting.style_axes``, so their
    axes boxes are identical.
    """
    ax.add_patch(Rectangle((x, y), w, h, transform=ax.transAxes,
                           facecolor="white", alpha=0.80, edgecolor="0.8",
                           linewidth=0.8, zorder=11))
    # Horizontal inset leaves room for the extend arrows, which overhang the
    # cax and would otherwise touch the frame.
    cax = ax.inset_axes([x + 0.045, y + 0.030, w - 0.090, 0.017], zorder=12)
    cb = ax.figure.colorbar(mappable, cax=cax, orientation="horizontal",
                            extend="both")
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(labelsize=fontsize, length=2.5, width=0.6, pad=1.5)
    cb.locator = MaxNLocator(nbins=nticks - 1)
    cb.update_ticks()
    cax.set_title(label, fontsize=fontsize, pad=3.5)
    return cb


meta = json.loads((OBS / "observed_severity_summary.json").read_text())
breaks = meta["barc_breaks_x1000"]
seg = gpd.read_file(OBS / f"{FIRE.upper()}_segments.gpkg").to_crs("EPSG:4326")
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

with rasterio.open(OBS / f"{FIRE.lower()}_brisk_dnbr.tif") as ds:
    sev_rgba, sev_on_grid = _severity_rgba(
        ds.read(1).astype("float64"), breaks, shape, tr, "EPSG:4326",
        {"transform": ds.transform, "crs": ds.crs})

try:
    from stormscape import refdata
    roads = refdata.roads(bounds)
except Exception as exc:                                   # context is a bonus
    print(f"roads unavailable: {type(exc).__name__}: {exc}", flush=True)
    roads = None

P = seg["P_24mmh"].to_numpy()
T = seg["I15_50"].to_numpy()
lo, hi = np.percentile(T[np.isfinite(T)], [2, 98])
PCMAP = ListedColormap(P_COLORS)
PNORM = BoundaryNorm([0.0, *P_BREAKS, 1.0], PCMAP.N)

fig, axes = plt.subplots(1, 3, figsize=(25, 10.5), dpi=140)
for ax, mode in zip(axes, ("severity", "likelihood", "threshold")):
    ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=im_extent, zorder=0)
    if roads is not None and len(roads):
        roads.to_crs("EPSG:4326").plot(ax=ax, color="0.25", linewidth=0.5,
                                       alpha=0.7, zorder=2)
    if mode == "severity":
        ax.imshow(sev_rgba, extent=im_extent, zorder=3,
                  interpolation="nearest")
        ax.legend(handles=[Patch(facecolor=np.array(c) / 255.0,
                                 edgecolor="0.3", label=l)
                           for c, l in zip(burn.BAER_CLASS_COLORS, SEV_LABELS)],
                  title=f"BARC class (breaks {'/'.join(f'{b/1000:g}' for b in breaks)} dNBR)",
                  loc="lower left", fontsize=8, title_fontsize=8)
        ax.set_title("Observed burn severity — CIMSS BRISK dNBR\n"
                     f"scene {', '.join(meta['scene_dates'])} · "
                     "vegetation change, not soil burn severity", fontsize=10.5)
    elif mode == "likelihood":
        # Ascending, so the most likely segments finish on top rather than
        # being overdrawn by a quiet neighbour at a confluence.
        seg.sort_values("P_24mmh").plot(ax=ax, column="P_24mmh", cmap=PCMAP,
                                        norm=PNORM, linewidth=1.6, zorder=5)
        ax.legend(handles=[Line2D([0], [0], color=c, lw=3, label=l)
                           for c, l in zip(P_COLORS, P_LABELS)],
                  title=f"P(debris flow) at {DESIGN_I15:g} mm/h",
                  loc="lower left", fontsize=8, title_fontsize=8)
        ax.set_title("Debris-flow likelihood at the design storm\n"
                     f"$I_{{15}}$ = {DESIGN_I15:g} mm/h (≈1-year, 15-minute)",
                     fontsize=10.5)
    else:
        seg.sort_values("I15_50", ascending=False).plot(
            ax=ax, column="I15_50", cmap="plasma_r", linewidth=1.6,
            vmin=lo, vmax=hi, zorder=5, legend=False)
        _sm = ScalarMappable(norm=Normalize(vmin=lo, vmax=hi), cmap="plasma_r")
        _sm.set_array([])
        inset_colorbar(ax, _sm, "triggering $I_{15}$ (mm/h)")
        ax.set_title("Rainfall intensity that triggers a debris flow\n"
                     "inverting the same fit at p = 0.5 · lower = smaller storm",
                     fontsize=10.5)
    per.boundary.plot(ax=ax, color="#56B4E9", linewidth=1.8, zorder=6,
                      path_effects=mc.fire_style("current")["path_effects"])
    mc.style_axes(ax, extent, step=0.1)

age = meta.get("composite_age_days")

# Class fractions computed from the SAME array the map paints, at the SAME
# breaks. The summary JSON also carries a class_fraction, but it is BRISK's
# own five-class 'usgs' scheme (unburned/low/moderate-low/moderate-high/high),
# not the four BARC classes these breaks define -- quoting it under this legend
# put "14% moderate or high" in the title over a map showing 21%. One binning
# per sheet.
_v = sev_on_grid[np.isfinite(sev_on_grid)] * 1000.0
_idx = np.digitize(_v, list(breaks))
frac = {lab: float((_idx == i).mean()) for i, lab in enumerate(SEV_LABELS)}
modhigh = frac["moderate"] + frac["high"]
print("BARC class fractions on the display grid: "
      + ", ".join(f"{k} {v:.1%}" for k, v in frac.items()), flush=True)
fig.suptitle(
    f"{FIRE} fire — post-fire debris-flow hazard on observed severity · "
    f"{len(seg):,} segments · {CAL} calibration, {meta.get('region','?')}\n"
    f"median P at {DESIGN_I15:g} mm/h = {np.nanmedian(P):.2f} · "
    f"{int((P >= 0.5).sum()):,} segments at or above 50% · "
    f"median triggering $I_{{15}}$ {np.nanmedian(T):.1f} mm/h "
    f"(5–95%: {np.nanpercentile(T, 5):.0f}–{np.nanpercentile(T, 95):.0f})\n"
    f"BRISK composite {', '.join(meta['scene_dates'])}, {age} d old — "
    f"{meta.get('magnitude_caveat','')}; {modhigh:.0%} of burned pixels "
    "moderate or high at these breaks",
    y=0.985, fontsize=12)


fig.subplots_adjust(top=0.90, bottom=0.06, left=0.03, right=0.97)
mc.save(fig, f"{FIRE.lower()}_postfire_hazard")
plt.close(fig)

summary = {
    "fire": FIRE, "calibration": CAL,
    "severity_source": "CIMSS BRISK", "scene_dates": meta["scene_dates"],
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
    "class_fraction_barc": {k: round(v, 4) for k, v in frac.items()},
    "class_fraction_brisk_usgs_scheme": meta.get("class_fraction", {}),
}
(OBS / "postfire_hazard_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
