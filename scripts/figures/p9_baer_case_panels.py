"""Two classified maps of the same fire, side by side, for the fires where our
fixed regional dNBR break and BAER's field map disagree hardest.

Left: MTBS (or BAER) dNBR classified at the calibrated regional break -- the
map firescape actually uses.  Right: BAER's published Soil Burn Severity,
pulled on the SAME grid and LOCKED to the matched tile (never omit
``lock_raster_id``; see the ``firescape.baer`` docstring).  Underneath, the
fire's own dNBR histogram with both lines drawn on it, which is where the
disagreement is actually legible.

Reads ``products/calibration/baer_outlier_diagnostics.csv`` -- run
``scripts/analysis/p9_baer_outlier_diagnostics.py`` first.

    python scripts/figures/p9_baer_case_panels.py [FIRE NAME ...]
"""
import glob, pathlib, re, sys, warnings
warnings.filterwarnings("ignore")

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from firescape import baer, paths, severity

FIGS = paths.figures_dir("baer_outliers")
FIGS.mkdir(parents=True, exist_ok=True)

BARC = ListedColormap(["#009E73", "#F0E442", "#E69F00", "#D55E00"])
NORM = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], BARC.N)
LABELS = ["unburned / very low", "low", "moderate", "high"]
ACC, BLU = "#8C2318", "#2A5A8C"

CASES = [c.strip() for c in (sys.argv[1:] or
         ["DAVIS", "TABOOSE", "MOUNTAIN VIEW", "SLATER", "CHERRY", "HUGHES"])]

diag = pd.read_csv(paths.products_dir("calibration") / "baer_outlier_diagnostics.csv")
ca_set = pd.read_csv(pathlib.Path(paths.__file__).parent / "data" / "calibration" /
                     "fire_sets" / "ca_baer_candidates.csv")
oid_by_event = dict(zip(ca_set.event_id, ca_set.baer_oid))


def nv_oid(event_id, ig_year, perim):
    """NV fires aren't in the CA candidate CSV -- ask the mosaic catalog directly."""
    m = baer.match_perimeters(perim, min_frac=0.90, fire_years=[int(ig_year)])
    return None if not len(m) else int(m.iloc[0]["baer_oid"])


for name in CASES:
    sub = diag[diag.incid_name == name]
    if not len(sub):
        print(f"!! {name} not in diagnostics"); continue
    # Seven fires sit in both sets; take the NEVADA row where there is one --
    # these are the fires the Nevada chart shows, so the Nevada regional break
    # is the number the reader is comparing against.
    f = sub.sort_values("state", ascending=False).iloc[0]
    bundle = paths.raw_dir("mtbs", "fires", f["event_id"])
    tif = sorted(glob.glob(str(bundle / "*_dnbr.tif")))[0]
    with rasterio.open(tif) as ds:
        dnbr = ds.read(1).astype("float64")
        nodata, crs, tr, bounds = ds.nodata, ds.crs, ds.transform, ds.bounds
    perim = gpd.read_file(sorted(bundle.glob("*_burn_area.shp"))[0]).to_crs(crs)

    oid = oid_by_event.get(f["event_id"])
    if pd.isna(oid) if oid is not None else True:
        oid = nv_oid(f["event_id"], f["ig_year"], perim.to_crs(4326))
    bcls = baer.fetch_aligned((bounds.left, bounds.bottom, bounds.right, bounds.top),
                              dnbr.shape, crs.to_epsg(), lock_raster_id=oid)

    valid = severity.valid_dnbr(dnbr, nodata) & (bcls >= 1) & (bcls <= 4)
    ours_break = float(f["our_break"])
    ours = np.where(valid, severity.classify_barc4(dnbr, (125.0, ours_break, 500.0)), 0)
    theirs = np.where(valid, bcls, 0)
    d = dnbr[valid]
    implied = float(f["implied"])

    # crop to the data, with a small margin
    rr, cc = np.where(valid)
    r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
    pad = max(4, int(0.02 * max(r1 - r0, c1 - c0)))
    r0, c0 = max(0, r0 - pad), max(0, c0 - pad)
    r1, c1 = min(dnbr.shape[0], r1 + pad), min(dnbr.shape[1], c1 + pad)
    sl = (slice(r0, r1), slice(c0, c1))
    ext = (bounds.left + c0 * tr.a, bounds.left + c1 * tr.a,
           bounds.top + r1 * tr.e, bounds.top + r0 * tr.e)

    m_ours = np.ma.masked_where(ours[sl] == 0, ours[sl])
    m_thrs = np.ma.masked_where(theirs[sl] == 0, theirs[sl])
    modhi_o = float((ours[valid] >= 3).mean())
    modhi_b = float((theirs[valid] >= 3).mean())

    w, h = (c1 - c0), (r1 - r0)
    map_h = 5.4 * h / max(w, 1) * 1.0
    map_h = float(np.clip(map_h, 3.4, 7.0))
    fig = plt.figure(figsize=(11.0, map_h + 2.9), dpi=150)
    gs = fig.add_gridspec(2, 2, height_ratios=[map_h, 1.85], hspace=0.20, wspace=0.06,
                          left=0.035, right=0.985, top=0.855, bottom=0.095)
    axL, axR = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axH = fig.add_subplot(gs[1, :])

    for ax, arr, title in ((axL, m_ours, f"ours — MTBS dNBR at the regional break ({ours_break:g})"),
                           (axR, m_thrs, "BAER — field-verified Soil Burn Severity")):
        ax.imshow(arr, cmap=BARC, norm=NORM, extent=ext, interpolation="nearest")
        perim.boundary.plot(ax=ax, color="#111111", linewidth=0.7, zorder=5)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#BBBBBB")
        ax.set_title(title, fontsize=10.5, pad=6)
    for ax, v in ((axL, modhi_o), (axR, modhi_b)):
        ax.text(.985, .975, f"moderate + high  {v:.0%}", transform=ax.transAxes,
                fontsize=10.5, va="top", ha="right", color="#111",
                bbox=dict(fc="white", ec="#CCCCCC", alpha=.92, pad=4))
    # scale bar on the left panel
    span = ext[1] - ext[0]
    bar = 10 ** np.floor(np.log10(span / 4)); bar = bar * (5 if span / 4 / bar > 5 else (2 if span / 4 / bar > 2 else 1))
    x0 = ext[0] + 0.05 * span; y0 = ext[2] + 0.055 * (ext[3] - ext[2])
    axL.add_patch(Rectangle((x0, y0), bar, 0.008 * (ext[3] - ext[2]), fc="#111", ec="none", zorder=6))
    axL.text(x0 + bar / 2, y0 + 0.018 * (ext[3] - ext[2]), f"{bar/1000:g} km",
             ha="center", fontsize=8, color="#111", zorder=6)

    axL.legend(handles=[Line2D([0], [0], marker="s", color="none", markersize=9,
                               markerfacecolor=c, label=l) for c, l in zip(BARC.colors, LABELS)],
               loc="upper center", bbox_to_anchor=(1.03, -0.015), ncol=4, frameon=False,
               fontsize=9, handletextpad=.4, columnspacing=1.4)

    hi = np.percentile(d, 99.5)
    axH.hist(d, bins=np.linspace(min(-100, d.min()), max(hi, implied, ours_break) * 1.05, 90),
             color="#C9C6BF", edgecolor="none")
    ymax = axH.get_ylim()[1]
    pts = sorted(((ours_break, ACC), (implied, BLU)))
    close = abs(pts[1][0] - pts[0][0]) < 0.07 * (axH.get_xlim()[1] - axH.get_xlim()[0])
    for i, (x, c) in enumerate(pts):
        axH.axvline(x, color=c, lw=2.0, zorder=4)
        axH.text(x, ymax * (0.995 if (i == 0 or not close) else 0.86), f"{x:g}", color=c,
                 fontsize=10, ha="center", va="top", fontweight="bold", zorder=5,
                 bbox=dict(fc="white", ec="none", alpha=.85, pad=1.5))
    axH.legend(handles=[Line2D([0], [0], color=ACC, lw=2.4, label="our regional break"),
                        Line2D([0], [0], color=BLU, lw=2.4, label="where BAER's line falls")],
               loc="upper left" if np.median(d) > np.mean(axH.get_xlim()) else "upper right",
               frameon=False, fontsize=9.5, handlelength=1.5)
    axH.set_xlabel("dNBR × 1000, inside the fire perimeter", fontsize=9.5)
    axH.set_yticks([])
    for s in ("top", "right", "left"):
        axH.spines[s].set_visible(False)
    axH.spines["bottom"].set_edgecolor("#BBBBBB")
    axH.tick_params(labelsize=9)

    mm = re.match(r"(mtbs|baer|ravg|provisional)_", pathlib.Path(tif).name)
    prog = mm.group(1).upper() if mm else "unknown"
    sub2 = (f"{f['region_name']} · {f['km2']:.0f} km² · {prog} dNBR, "
            f"{f['assess_type'] if isinstance(f['assess_type'],str) else 'initial'} assessment "
            f"({f['pre_date']} → {f['post_date']}) · κ={f['kappa_4']:.2f}, "
            f"moderate-or-higher agreement {f['modhi_agree']:.2f}")
    fig.suptitle(f"{name.title()} Fire ({int(f['ig_year'])})", fontsize=15, y=0.975, x=0.035, ha="left")
    fig.text(0.035, 0.905, sub2, fontsize=9, color="#5D6467", ha="left")
    gap = modhi_o - modhi_b
    fig.text(0.985, 0.972,
             ("our map runs HOTTER by " if gap > 0 else "our map runs COOLER by ")
             + f"{abs(gap)*100:.0f} points of burned area",
             fontsize=11, color=ACC if gap > 0 else "#2E6E74", ha="right", va="top",
             fontweight="bold")

    stem = name.lower().replace(" ", "_")
    fig.savefig(FIGS / f"{stem}.png", dpi=150, facecolor="white")
    plt.close(fig)
    print(f"{name:16s} ours modhi={modhi_o:.0%}  BAER modhi={modhi_b:.0%}  "
          f"break {ours_break:g} vs implied {implied:g}  -> {stem}.png", flush=True)
print("done")
