"""Why does P(R>T) rise toward southeast Nevada?

P(R>T) is built from three per-basin numbers: the M1 rainfall threshold T (the
15-min intensity giving a 50% debris-flow likelihood) and the Atlas 14 1-year
and 50-year 15-min intensities, which anchor a log-linear intensity-frequency
line. Rossi et al. (2025) Eqns 6-9 collapse to

    log10(RI) = log10(50) * (T - I1) / (I50 - I1)          P = 1 - exp(-1/RI)

so the recurrence interval depends only on **where the threshold sits on the
local IDF curve**, in units of the 1-to-50-year spread. That framing makes the
southeast gradient a testable question rather than an impression:

* if the whole IDF curve scaled up while T scaled with it, the normalised
  position -- and therefore P -- would not move at all;
* P rises only where T fails to keep pace with the climatology.

Two independent tests are run: rank correlations of P and its inputs against a
northwest-to-southeast axis, and counterfactual surfaces that freeze either the
climatology or the threshold at its statewide median and recompute P. Whichever
freeze flattens the gradient is the cause.

Writes figures/prt_decomposition.{png,pdf} and
products/prefire/statewide_v1_1/prt_decomposition.json.
"""
import json
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyogrio import read_dataframe
from scipy.stats import spearmanr

from firescape import annualprob, paths, plotting as mc

OUT = paths.products_dir("prefire", "statewide_v1_1")
BASINS = OUT / "statewide_v1_1_basins.gpkg"
COLS = ["I15_50", "I15_1yr", "I15_50yr", "P_RgtT"]

g = read_dataframe(BASINS, columns=COLS).to_crs("EPSG:4326")
pt = g.geometry.representative_point()
g["lon"], g["lat"] = pt.x.values, pt.y.values
g = g.dropna(subset=COLS).reset_index(drop=True)

T = g["I15_50"].to_numpy()          # M1 threshold at p=0.5, mm/h
I1 = g["I15_1yr"].to_numpy()        # Atlas 14 1-year 15-min intensity
I50 = g["I15_50yr"].to_numpy()      # Atlas 14 50-year
P = g["P_RgtT"].to_numpy()
spread = I50 - I1
position = (T - I1) / spread        # threshold's place on the IDF curve

# Northwest -> southeast axis, 0 at the NW corner and 1 at the SE corner.
lon, lat = g["lon"].to_numpy(), g["lat"].to_numpy()
se = 0.5 * ((lon - lon.min()) / np.ptp(lon) + (lat.max() - lat) / np.ptp(lat))
g["se"] = se

# --- counterfactuals: freeze one input at its statewide median --------------
med_T, med_I1, med_I50 = np.median(T), np.median(I1), np.median(I50)


def prob(t, i1, i50):
    m, b = annualprob.fit_log_linear(i1, i50)
    return annualprob.annual_probability(annualprob.recurrence_interval(t, m, b))


P_check = prob(T, I1, I50)                                  # reproduces P_RgtT
P_clim_fixed = prob(T, np.full_like(I1, med_I1), np.full_like(I50, med_I50))
P_thresh_fixed = prob(np.full_like(T, med_T), I1, I50)
assert np.nanmax(np.abs(P_check - P)) < 1e-9, "recomputation disagrees with the product"

# --- how much of the gradient survives each freeze --------------------------
lo, hi = se < np.percentile(se, 20), se > np.percentile(se, 80)


def gradient(v):
    """Ratio of SE-quintile median to NW-quintile median."""
    return float(np.median(v[hi]) / np.median(v[lo]))


stats = {
    "n_basins": int(len(g)),
    "spearman_vs_se_axis": {k: round(float(spearmanr(se, v).statistic), 3) for k, v in
                            [("P_RgtT", P), ("threshold_T", T), ("atlas14_I1", I1),
                             ("atlas14_I50", I50), ("spread_I50_minus_I1", spread),
                             ("threshold_position", position)]},
    "nw_to_se_ratio": {k: round(gradient(v), 3) for k, v in
                       [("P_RgtT", P), ("threshold_T", T), ("atlas14_I1", I1),
                        ("atlas14_I50", I50), ("spread_I50_minus_I1", spread)]},
    "counterfactual_nw_to_se_ratio": {
        "actual": round(gradient(P), 3),
        "climatology_frozen": round(gradient(P_clim_fixed), 3),
        "threshold_frozen": round(gradient(P_thresh_fixed), 3),
    },
    "medians": {"threshold_T_mmh": round(float(med_T), 2),
                "atlas14_I1_mmh": round(float(med_I1), 2),
                "atlas14_I50_mmh": round(float(med_I50), 2),
                "threshold_position": round(float(np.median(position)), 3),
                "RI_thresh_yr": round(float(np.median(10 ** (np.log10(50) * position))), 2)},
}
print(json.dumps(stats, indent=2))
(OUT / "prt_decomposition.json").write_text(json.dumps(stats, indent=2))

# --- figure -----------------------------------------------------------------
tr, shape, extent = mc.grid(mc.domain_bounds4326())
hs = mc.hillshade(tr, shape)
ctx = mc.fetch_context()
im_extent = (extent[0], extent[1], extent[2], extent[3])
plim = (0.0, float(np.percentile(P, 99)))

g["P_clim_fixed"], g["P_thresh_fixed"] = P_clim_fixed, P_thresh_fixed
panels = [
    ("I15_50", "cividis", (float(np.percentile(T, 2)), float(np.percentile(T, 98))),
     "M1 rainfall threshold\n$I_{15}$ at 50% likelihood (mm/h)"),
    ("I15_1yr", "viridis", (float(np.percentile(I1, 2)), float(np.percentile(I1, 98))),
     "Atlas 14 climatology\n1-year $I_{15}$ (mm/h)"),
    ("P_RgtT", "viridis", plim, "P(R>T) as modelled\nannual chance of threshold rain"),
    ("P_clim_fixed", "viridis", plim,
     "Counterfactual: climatology frozen\n(only the threshold varies)"),
    ("P_thresh_fixed", "viridis", plim,
     "Counterfactual: threshold frozen\n(only the climatology varies)"),
]

fig = plt.figure(figsize=(21, 11.5), dpi=140)
for i, (col, cmap, clim, title) in enumerate(panels):
    ax = fig.add_subplot(2, 3, i + 1)
    ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=im_extent)
    arr = mc.burn(g, col, tr, shape)
    im = ax.imshow(np.ma.masked_invalid(arr), cmap=cmap, vmin=clim[0], vmax=clim[1],
                   extent=im_extent, alpha=mc.MAX_LAYER_ALPHA,
                   interpolation="antialiased")
    fig.colorbar(im, ax=ax, shrink=0.62, pad=0.02)
    mc.draw_context(ax, ctx, label_rivers=False, extent=extent)
    mc.style_axes(ax, extent)
    ax.set_title(title, fontsize=9.5)

# profile panel: everything normalised to its northwest value
ax = fig.add_subplot(2, 3, 6)
edges = np.linspace(se.min(), se.max(), 13)
mid = 0.5 * (edges[:-1] + edges[1:])
series = [("Atlas 14 $I_{15}$, 50-yr", I50, "#0072B2", "-"),
          ("Atlas 14 $I_{15}$, 1-yr", I1, "#56B4E9", "-"),
          ("M1 threshold", T, "#D55E00", "-"),
          ("P(R>T)", P, "black", "--")]
for label, v, colour, ls in series:
    med = np.array([np.median(v[(se >= a) & (se < b)]) if ((se >= a) & (se < b)).any()
                    else np.nan for a, b in zip(edges[:-1], edges[1:])])
    ax.plot(mid, med / med[0], color=colour, ls=ls, lw=2.0, label=label)
ax.axhline(1.0, color="0.6", lw=0.8, zorder=0)
ax.set_xlabel("northwest  →  southeast", fontsize=9)
ax.set_ylabel("median, relative to the northwest end", fontsize=9)
ax.set_title("The threshold does not keep pace with the storms\n"
             f"climatology ×{stats['nw_to_se_ratio']['atlas14_I1']:.2f}, "
             f"threshold ×{stats['nw_to_se_ratio']['threshold_T']:.2f} "
             f"→ P(R>T) ×{stats['nw_to_se_ratio']['P_RgtT']:.2f}", fontsize=9.5)
ax.legend(fontsize=8, frameon=False)
ax.tick_params(labelsize=8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.suptitle(
    "What drives the southeast rise in P(R>T)? — firescape statewide v1.1\n"
    "$\\log_{10}RI = \\log_{10}50 \\cdot (T - I_1)/(I_{50} - I_1)$: P depends on where the "
    "threshold sits on the local intensity-frequency curve, not on rainfall alone",
    y=0.985, fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.955))
mc.save(fig, "prt_decomposition")
