"""Merge pilot_v2 units, add P(R>T), and compare against pilot_v1."""
import json
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.transform import Affine

from firescape import annualprob as ap
from firescape import paths

OUT = paths.products_dir("prefire", "pilot_v2")
units = sorted(p for p in OUT.glob("*_basins.gpkg") if p.name[0].isdigit())
frames = []
for p in units:
    g = gpd.read_file(p)
    g["huc10"] = p.name.split("_")[0]
    frames.append(g)
basins = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True),
                          geometry="geometry", crs=frames[0].crs)
print(f"merged {len(units)} units -> {len(basins)} basins")

z = np.load(paths.interim_dir("pilot") / "atlas14_i15.npz")
i1g, i50g, tr = z["i1"], z["i50"], Affine(*z["transform"])
cent = basins.geometry.centroid.to_crs("EPSG:4269")
cols = ((cent.x - tr.c) / tr.a).astype(int).to_numpy()
rows = ((cent.y - tr.f) / tr.e).astype(int).to_numpy()
ok = (rows >= 0) & (rows < i1g.shape[0]) & (cols >= 0) & (cols < i1g.shape[1])
i1 = np.full(len(basins), np.nan); i50 = np.full(len(basins), np.nan)
i1[ok] = i1g[rows[ok], cols[ok]]; i50[ok] = i50g[rows[ok], cols[ok]]
basins["I15_1yr"], basins["I15_50yr"] = i1, i50

p_rt, m, b = ap.p_exceed_threshold(basins["I15_50"].to_numpy(), i1, i50)
basins["RI_thresh_yr"] = ap.recurrence_interval(basins["I15_50"].to_numpy(), m, b)
basins["P_RgtT"] = p_rt
basins["P_annual"] = ap.combined(p_rt)
basins.to_file(OUT / "pilot_v2_basins.gpkg", driver="GPKG")

v1 = gpd.read_file(paths.products_dir("prefire", "pilot_v1") / "pilot_v1_basins.gpkg")


def stats(df, label):
    h = df["H_24mmh"]
    return {
        "label": label, "n_basins": int(len(df)),
        "mean_likelihood": float(df["P_24mmh"].mean()),
        "median_likelihood": float(df["P_24mmh"].median()),
        "p90_likelihood": float(df["P_24mmh"].quantile(0.9)),
        "median_threshold_mmh": float(df["I15_50"].median()),
        "median_P_RgtT": float(df["P_RgtT"].median()),
        "zero_volume_frac": float((df["V_24mmh"] <= 0).mean()),
        "hazard": {int(c): int((h == c).sum()) for c in (1, 2, 3)},
    }


s1, s2 = stats(v1, "pilot_v1 (Staley, P=0.48)"), stats(basins, "pilot_v2 (nv_merged, P=0.51)")
print(f"\n{'metric':<26}{'pilot_v1':>22}{'pilot_v2':>22}")
for k in ("n_basins", "mean_likelihood", "median_likelihood", "p90_likelihood",
          "median_threshold_mmh", "median_P_RgtT", "zero_volume_frac"):
    f = "{:>22.3f}" if isinstance(s1[k], float) else "{:>22}"
    print(f"{k:<26}" + f.format(s1[k]) + f.format(s2[k]))
for c in (1, 2, 3):
    print(f"{'hazard class ' + str(c):<26}{s1['hazard'][c]:>22}{s2['hazard'][c]:>22}")

(OUT / "pilot_v2_summary.json").write_text(json.dumps({"v1": s1, "v2": s2}, indent=2))
print(f"\nwrote {OUT / 'pilot_v2_basins.gpkg'}")
