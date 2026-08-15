"""Should the statewide refit replace nv_merged? A held-out answer.

Two things make a naive comparison worthless:

* **Quantile mismatch.** Comparing ``weibull_dnbr(P_dsim)`` against an observed
  *median* measures the gap between P_dsim and 0.5, not the quality of the fit.
  Predictions here are read at the same quantile as the observation.
* **Circularity.** A table scored on the pixels it was fit to always wins.
  Fires are split into k folds; each fold is predicted by a table refit from
  the other folds only, so every number below is out-of-sample.

Tables compared: Staley 2018 (published western US), nv_merged (the committed
table, refit from 16 pilot-area fires), and the statewide refit. Error is
|simulated - observed| dNBR at the 25th, 50th, 75th and 90th percentiles,
weighted by held-out sample size -- the upper quantiles matter because the
BARC break test lives in the tail.
"""
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from firescape import paths, severity

CACHE = paths.interim_dir("calib", "refit_samples_statewide")
OUT = paths.products_dir("calibration")
QUANTILES = (0.25, 0.50, 0.75, 0.90)
K_FOLDS = 5
MIN_TRAIN, MIN_TEST = 2000, 500
MIN_FIRES = 5                 # independent fires required to override a class
RNG = np.random.default_rng(11)

files = sorted(CACHE.glob("NV*.npz")) + sorted(CACHE.glob("CA*.npz")) \
    + sorted(CACHE.glob("ID*.npz")) + sorted(CACHE.glob("OR*.npz")) \
    + sorted(CACHE.glob("UT*.npz")) + sorted(CACHE.glob("AZ*.npz"))
files = sorted(set(files))
per_fire = []
for f in files:
    z = np.load(f, allow_pickle=True)
    if "codes" not in z:
        continue
    per_fire.append({int(c): z[f"c{c}"] for c in z["codes"]})
print(f"{len(per_fire)} fires with samples")

staley = severity.load_cdf_table()
nv_merged = severity.load_cdf_table(table="nv_merged")

fold_of = RNG.permutation(len(per_fire)) % K_FOLDS
records = []
for k in range(K_FOLDS):
    train = [s for i, s in enumerate(per_fire) if fold_of[i] != k]
    test = [s for i, s in enumerate(per_fire) if fold_of[i] == k]

    pooled_train: dict[int, list] = {}
    fires_per_class: dict[int, int] = {}
    for s in train:
        for c, v in s.items():
            pooled_train.setdefault(c, []).append(v)
            if v.size >= 500:
                fires_per_class[c] = fires_per_class.get(c, 0) + 1
    pooled_train = {c: np.concatenate(v) for c, v in pooled_train.items()}
    refit = severity.refit_table(pooled_train, min_n=MIN_TRAIN)

    # Conservative variant: a Nevada fit only replaces the published one where
    # enough INDEPENDENT fires back it. Pixel count alone is not evidence --
    # 22 classes clear 2,000 px on two fires or fewer, and a class defined by
    # one fire's weather is not a class model.
    gated = nv_merged.copy()
    for c in refit.index:
        if fires_per_class.get(c, 0) >= MIN_FIRES:
            gated.loc[c, refit.columns] = refit.loc[c]

    pooled_test: dict[int, list] = {}
    for s in test:
        for c, v in s.items():
            pooled_test.setdefault(c, []).append(v)

    for c, chunks in pooled_test.items():
        obs = np.concatenate(chunks)
        if obs.size < MIN_TEST or c not in refit.index:
            continue
        if c not in staley.index or c not in nv_merged.index:
            continue          # score only where all three can predict
        for q in QUANTILES:
            o = float(np.quantile(obs, q))
            for label, tab in (("staley2018", staley), ("nv_merged", nv_merged),
                               ("statewide_refit", refit),
                               ("statewide_gated", gated)):
                pred = float(severity.weibull_dnbr(
                    q, tab.at[c, "Weibull_Lambda_Scale"],
                    tab.at[c, "Weibull_Kappa_Shape"]))
                records.append({"fold": k, "code": c, "n": obs.size, "q": q,
                                "table": label, "obs": o, "pred": pred,
                                "abs_err": abs(pred - o)})

df = pd.DataFrame(records)
df.to_csv(OUT / "cdf_statewide_holdout.csv", index=False)

print(f"\nout-of-sample error over {df['code'].nunique()} classes, "
      f"{K_FOLDS} folds by fire (weighted by held-out pixels)")
print(f"{'table':>18} " + "".join(f"{f'q{int(q*100)}':>9}" for q in QUANTILES)
      + f"{'all':>10}")
summary = {}
for label in ("staley2018", "nv_merged", "statewide_refit", "statewide_gated"):
    d = df[df.table == label]
    cells = []
    for q in QUANTILES:
        dq = d[d.q == q]
        cells.append(np.average(dq.abs_err, weights=dq.n))
    overall = np.average(d.abs_err, weights=d.n)
    summary[label] = {**{f"q{int(q*100)}": round(float(c), 2)
                         for q, c in zip(QUANTILES, cells)},
                      "all": round(float(overall), 2)}
    print(f"{label:>18} " + "".join(f"{c:9.1f}" for c in cells) + f"{overall:10.1f}")

best = min(summary, key=lambda k: summary[k]["all"])
print(f"\nbest out-of-sample: {best}")

# per-class, where the statewide refit changes the answer most
piv = (df[df.q == 0.5].pivot_table(index="code", columns="table",
                                   values="abs_err", aggfunc="mean")
       .join(df.groupby("code")["n"].max().rename("n")))
piv["gain_vs_nv_merged"] = piv["nv_merged"] - piv["statewide_refit"]
piv = piv.sort_values("n", ascending=False)
names = {c: str(nv_merged.at[c, "CLASSNAME"])[:38] if c in nv_merged.index else ""
         for c in piv.index}
piv.insert(0, "class", [names[c] for c in piv.index])
piv.to_csv(OUT / "cdf_statewide_holdout_by_class.csv")
pd.set_option("display.width", 200)
print("\nmedian-quantile error by class (largest classes; + = refit is better):")
print(piv.head(14).round(1).to_string())

(OUT / "cdf_statewide_holdout.json").write_text(json.dumps(
    {"folds": K_FOLDS, "quantiles": list(QUANTILES), "n_classes":
     int(df["code"].nunique()), "weighted_abs_error_dnbr": summary,
     "best": best}, indent=2))
print(f"\nwrote {OUT/'cdf_statewide_holdout.json'}")
