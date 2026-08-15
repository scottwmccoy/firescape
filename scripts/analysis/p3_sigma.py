"""Estimate the within-fire severity quantile dispersion sigma.

Model: pixel quantile q_i = Phi(z0 + eps_i), z0 = Phi^-1(P_dsim),
eps ~ N(0, sigma^2) spatially correlated. sigma is measurable: convert each
observed era-matched pixel dNBR to its class-CDF quantile, probit-transform,
and take the within-(fire x class) standard deviation. Validation: the
analytic exceedance 1 - Phi((Phi^-1(F_c(B)) - z_med)/sigma) must reproduce
each fire x class's observed break exceedance.
"""
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import norm

from firescape import paths, severity

BREAK = 325.0     # CBR regional break for the validation check

samp = pd.read_parquet(paths.interim_dir("calib") / "refit_samples_pilot.parquet")
table = severity.load_cdf_table(table="nv_merged")

lam = table["Weibull_Lambda_Scale"]
kap = table["Weibull_Kappa_Shape"]

rows = []
for (eid, code), g in samp.groupby(["event_id", "code"], observed=True):
    if len(g) < 500 or code not in table.index:
        continue
    z = (g["dnbr"].to_numpy(dtype=float) + 1000.0) / 2000.0
    q = 1.0 - np.exp(-np.power(np.clip(z, 1e-9, None) / lam[code], kap[code]))
    q = np.clip(q, 1e-4, 1 - 1e-4)
    zz = norm.ppf(q)
    qB = 1.0 - np.exp(-((BREAK + 1000.0) / 2000.0 / lam[code]) ** kap[code])
    z_med, z_sd = float(np.median(zz)), float(np.std(zz))
    pred_exc = 1.0 - norm.cdf((norm.ppf(qB) - z_med) / max(z_sd, 1e-6))
    rows.append({"event_id": eid, "code": int(code), "n": len(g),
                 "q_med": float(norm.cdf(z_med)), "sigma_z": z_sd,
                 "obs_exc": float((g["dnbr"] >= BREAK).mean()),
                 "pred_exc": pred_exc})

df = pd.DataFrame(rows)
sigma = float(df["sigma_z"].median())
w_sigma = float(np.average(df["sigma_z"], weights=df["n"]))
r = float(np.corrcoef(df["obs_exc"], df["pred_exc"])[0, 1])
mae = float((df["obs_exc"] - df["pred_exc"]).abs().mean())

print(f"fire x class cells (n>=500): {len(df)}")
print(f"sigma_z: median {sigma:.3f}, n-weighted {w_sigma:.3f}, "
      f"IQR [{df.sigma_z.quantile(.25):.3f}, {df.sigma_z.quantile(.75):.3f}]")
print(f"per-fire median quantile spread (the P_dsim dial): "
      f"sd {df.groupby('event_id', observed=True)['q_med'].median().std():.3f}")
print(f"validation at break {BREAK:.0f}: obs vs analytic exceedance "
      f"r={r:.3f}, MAE={mae:.3f}")
print("\ndeterministic simulator exceedance at this break would be 0 or 1 per "
      "class; observed cells span:")
print(df["obs_exc"].describe().round(3).to_string())

out = {"sigma_z_median": sigma, "sigma_z_weighted": w_sigma,
       "n_cells": len(df), "validation_r": r, "validation_mae": mae,
       "break_used": BREAK, "cdf_table": "nv_merged",
       "note": ("within-(fire x class) probit-quantile sd from the 2.0M-px "
                "era-matched sample; q_i = Phi(Phi^-1(P) + eps), "
                "eps ~ N(0, sigma^2)")}
p = paths.products_dir("calibration") / "severity_dispersion.json"
p.write_text(json.dumps(out, indent=2))
df.to_csv(paths.products_dir("calibration") / "severity_dispersion_cells.csv",
          index=False)
print("\nwrote", p)
