"""M3 summary: regional calibration TOML + Rossi-style diagnostics figure."""
import json
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler

OKABE_ITO = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000']
plt.rcParams['axes.prop_cycle'] = cycler(color=OKABE_ITO)

import numpy as np
import pandas as pd

from firescape import calibrate, paths
from firescape.hazard import logistic_m1_reference

fires = pd.read_csv(
    "/Users/scottmccoy/git/code/firescape/firescape/data/calibration/fire_sets/pilot_v1.csv"
).set_index("event_id")

calibs, rows = [], []
calib_root = paths.interim_dir("calib")
for d in sorted(calib_root.iterdir()):
    if not (d / "fire.json").exists():
        continue
    fc = calibrate.load_cached(d.name)
    meta = json.loads((d / "fire.json").read_text())
    if fc is None or not fc.ok:
        print(f"  {d.name}: no usable calibration (n_sel={meta['n_selected']})")
        continue
    calibs.append(fc)
    frow = fires.loc[d.name]
    rows.append({
        "event_id": d.name, "name": frow["incid_name"], "year": frow["ig_year"],
        "km2": frow["km2"], "pdsim": fc.pdsim, "n_sel": fc.n_selected,
        "mod_t": meta["mod_t"],
    })
tab = pd.DataFrame(rows).sort_values("year")
print(tab.to_string(index=False))

regional = calibrate.regional_summary(
    calibs, region="pilot",
    fire_breaks={r["event_id"]: r["mod_t"] for r in rows},
    toml_path=paths.products_dir("calibration") / "pilot_v1.toml",
)
print(f"\nREGIONAL: pdsim={regional['pdsim']:.2f} "
      f"(IQR {regional['pdsim_iqr'][0]:.2f}-{regional['pdsim_iqr'][1]:.2f}), "
      f"break={regional['break_lowmod']:.0f}, n={regional['n_fires']} fires")

# fire-wide mean DFL, sim (at regional pdsim) vs obs — placeholder S=0.25
# (KF pending: ScienceBase outage; exact S splices in later from curves.npz)
S_PLACEHOLDER = 0.25
grid = calibrate.PDSIM_GRID
k = int(np.argmin(np.abs(grid - regional["pdsim"])))
means = []
for fc in calibs:
    z = np.load(fc.cache_dir / "curves.npz")
    sel = z["sel"]
    if sel.sum() < 3:
        continue
    p_obs = logistic_m1_reference(z["T_obs"][sel], z["F_obs"][sel], S_PLACEHOLDER, 6.0)
    p_sim = logistic_m1_reference(z["T_sim"][sel, k], z["F_sim"][sel, k], S_PLACEHOLDER, 6.0)
    means.append({"event_id": fc.event_id, "obs": float(np.nanmean(p_obs)),
                  "sim": float(np.nanmean(p_sim)), "n": int(sel.sum())})
mdf = pd.DataFrame(means)
resid = mdf["sim"] - mdf["obs"]
nse = 1 - (resid ** 2).sum() / ((mdf["obs"] - mdf["obs"].mean()) ** 2).sum()
rmse = float(np.sqrt((resid ** 2).mean()))
print(f"fire-wide mean DFL (placeholder S): NSE={nse:.2f}, RMSE={rmse:.3f} "
      f"(Rossi regional benchmark: NSE 0.57, fire-mean RMSE 0.09)")

fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
order = tab.sort_values("pdsim")
axes[0].barh(np.arange(len(order)), order["pdsim"], color="#56B4E9", alpha=0.85)
axes[0].axvline(regional["pdsim"], color="#D55E00", linewidth=2,
                label=f"pilot regional median = {regional['pdsim']:.2f}")
axes[0].axvline(0.38, color="#009E73", linestyle="--", linewidth=1.5,
                label="Rossi CA MBD = 0.38")
axes[0].axvline(0.49, color="#CC79A7", linestyle=":", linewidth=1.5,
                label="Rossi CA NSN = 0.49")
axes[0].set_yticks(np.arange(len(order)))
axes[0].set_yticklabels([f"{r['name'][:14]} '{str(r['year'])[-2:]}" for _, r in order.iterrows()],
                        fontsize=7)
axes[0].set_xlabel("fire-calibrated P$_{dsim}$")
axes[0].set_title(f"Per-fire P$_{{dsim}}$ ({len(order)} pilot-area fires)")
axes[0].legend(fontsize=8, loc="lower right")

axes[1].scatter(mdf["obs"], mdf["sim"], s=np.sqrt(mdf["n"]) * 6, alpha=0.7, color="#0072B2")
axes[1].plot([0, 0.8], [0, 0.8], "k--", linewidth=1)
axes[1].set_xlabel("fire-wide mean observed DFL")
axes[1].set_ylabel("fire-wide mean simulated DFL (regional P$_{dsim}$)")
axes[1].set_title(f"Fire means at regional P$_{{dsim}}$={regional['pdsim']:.2f}\n"
                  f"NSE={nse:.2f}, RMSE={rmse:.3f} (placeholder S; Rossi bench NSE 0.57)")
axes[1].set_xlim(0, 0.8); axes[1].set_ylim(0, 0.8); axes[1].set_aspect("equal")
fig.suptitle("firescape M3 — Nevada pilot P$_{dsim}$ calibration (Rossi 2025 DFL method)", y=1.02)
fig.tight_layout()
png = paths.figures_dir() / "pilot_m3_calibration.png"
fig.savefig(png, bbox_inches="tight")
print("wrote", png)
tab.to_csv(paths.products_dir("calibration") / "fire_pdsim_table.csv", index=False)
mdf.to_csv(paths.products_dir("calibration") / "fire_mean_dfl.csv", index=False)
