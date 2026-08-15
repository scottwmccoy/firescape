"""Figure: what dispersion and RANGES do to the hazard chain (12 units)."""
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from firescape import paths

OK = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00',
      '#CC79A7', '#000000']

df = pd.read_csv(paths.products_dir("calibration") / "dist_experiment_units.csv")
df = df.sort_values("det_zeroV_g14", ascending=False).reset_index(drop=True)
short = df["name"].str.replace("-Frontal Lake Tahoe", "", regex=False) \
    .str.replace("-Las Vegas Wash", "", regex=False) \
    .str.replace("-Truckee River", "", regex=False) \
    .str.replace("-White River", "", regex=False)
y = np.arange(len(df))

fig, axes = plt.subplots(1, 3, figsize=(17, 7.5), dpi=140,
                         gridspec_kw={"width_ratios": [1, 1.25, 1]})

ax = axes[0]
ax.barh(y - 0.2, df["det_zeroV_g14"] * 100, height=0.38, color=OK[5],
        label="deterministic severity")
ax.barh(y + 0.2, df["soft_zeroV_g14"] * 100, height=0.38, color=OK[2],
        label="dispersed severity ($\\sigma$=0.91)")
ax.set_yticks(y)
ax.set_yticklabels(short, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("basins with ZERO Gartner volume (%)")
ax.legend(fontsize=8, loc="lower right")
ax.set_title("Dispersion dissolves the zero-volume collapse", fontsize=10.5)

ax = axes[1]
ax.barh(y - 0.27, df["det_mod+_g14"], height=0.24, color=OK[5],
        label="Gartner, deterministic")
ax.barh(y, df["soft_mod+_g14"], height=0.24, color=OK[2],
        label="Gartner, dispersed")
ax.barh(y + 0.27, df["det_mod+_ranges"], height=0.24, color=OK[4],
        label="RANGES")
ax.set_yticks(y)
ax.set_yticklabels([""] * len(df))
ax.invert_yaxis()
ax.set_xscale("symlog", linthresh=10)
ax.set_xlabel("moderate+ hazard basins (symlog)")
ax.legend(fontsize=8, loc="lower right")
ax.set_title("...and the hazard class responds by volume model", fontsize=10.5)

ax = axes[2]
ax.scatter(df["det_medP"], df["soft_medP"], s=55, color=OK[0],
           edgecolor="black", linewidth=0.5, zorder=5)
lim = (0.15, 0.40)
ax.plot(lim, lim, color="#888888", linewidth=0.8, zorder=1)
for _, r in df.iterrows():
    ax.annotate(r["region"].split()[0][:3], (r["det_medP"], r["soft_medP"]),
                xytext=(4, 3), textcoords="offset points", fontsize=6.5)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("median likelihood, deterministic")
ax.set_ylabel("median likelihood, dispersed")
ax.set_title("Calibrated likelihood is preserved\n(max shift 0.022)",
             fontsize=10.5)
ax.set_aspect("equal")

fig.suptitle(
    "Distributional severity experiment - 12 HU10 units, statewide_v1 regional calibration\n"
    "measured quantile dispersion $\\sigma_z$=0.91 (analytic exceedance validates at r=0.995) · "
    "statewide: zero-volume 93.0% (Gartner-det) $\\rightarrow$ 1.7% (RANGES)",
    y=1.0, fontsize=12)
fig.tight_layout()
out = paths.figures_dir() / "dist_severity_experiment.pdf"   # see plotting.save
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
