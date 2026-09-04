"""Can anything we have predict WHICH WAY a fire misses BAER?

Pools every per-fire covariate the other p9-p11 scripts produce -- where the
fixed break lands in the fire's own dNBR, post-fire image delay, ignition
month, perimeter elongation, dNBR roughness, and first-week gridMET fire
weather -- into one linear model of the signed moderate+ area gap, scored
leave-one-fire-out.  Result (2026-09-03, 124 fires): raster + metadata
explain 18% in-sample and 9% out of sample; adding fire weather raises the
former to 26% and LOWERS the latter to 8%.  The sign is not predictable from
these inputs.  Runs after p11_baer_patchiness.py and p11_baer_fireweather.py.
"""
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import numpy.linalg as la
import pandas as pd
from scipy import stats

from firescape import paths

CAL = paths.products_dir("calibration")
w = pd.read_csv(CAL / "baer_fireweather.csv")
p = pd.read_csv(CAL / "baer_patchiness.csv")
w = w.merge(p[["event_id", "elong", "roughness", "ig_month", "our_modhi",
               "hot_speckle", "cool_speckle"]], on="event_id")
w["gap"] = w.modhi_gap_ours_minus_baer * 100
w = w.dropna(subset=["wind_max", "vpd_max", "erc_mean", "bi_max"])


def partial_spearman(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return stats.pearsonr(ex, ey)


def loo_r2(cols):
    X = w[cols].copy()
    if "days_post" in X:
        X["days_post"] = np.log10(X.days_post.clip(lower=1))
    X = (X - X.mean()) / X.std()
    X["1"] = 1
    X, y = X.values, w.gap.values
    b, *_ = la.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    e = []
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        bb, *_ = la.lstsq(X[m], y[m], rcond=None)
        e.append(y[i] - X[i] @ bb)
    return r2, 1 - np.sum(np.square(e)) / ((y - y.mean()) ** 2).sum()


print(f"n={len(w)}")
print("\nspeckle vs gap, raw and controlling for class balance (our moderate+ fraction):")
for c in ("hot_speckle", "cool_speckle"):
    raw = stats.spearmanr(w[c], w.gap); pr = partial_spearman(w[c], w.gap, w.our_modhi)
    print(f"  {c:12s} rho={raw.statistic:+.3f} (p={raw.pvalue:.3f}) -> partial r={pr.statistic:+.3f} (p={pr.pvalue:.3f})")
print("\nsingle covariates vs gap:")
for c in ("pct_rank_ours", "days_post", "elong", "ig_month", "wind_max", "wind_mean",
          "vpd_max", "erc_mean", "bi_max"):
    r = stats.spearmanr(w[c], w.gap)
    print(f"  {c:14s} rho={r.statistic:+.3f}  p={r.pvalue:.3f}")
base = ["pct_rank_ours", "days_post", "elong", "ig_month", "roughness"]
wx = ["wind_max", "vpd_max", "erc_mean", "bi_max"]
print("\npooled linear model of the signed gap, leave-one-fire-out:")
for lab, cols in (("raster + metadata", base), ("+ fire weather", base + wx), ("fire weather alone", wx)):
    r2, l = loo_r2(cols)
    print(f"  {lab:22s} in-sample r2={r2:.2f}   LOO r2={l:.2f}")
