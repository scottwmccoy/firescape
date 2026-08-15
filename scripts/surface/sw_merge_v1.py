"""Merge the 591-unit statewide surface, add P(R>T), summarize."""
import json
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.transform import Affine

from firescape import annualprob as ap
from firescape import hazard as hz
from firescape import paths

OUT = paths.products_dir("prefire", "statewide_v1")
units = sorted(p for p in OUT.glob("*_basins.gpkg") if p.name[0].isdigit())
print(f"merging {len(units)} unit files...", flush=True)
frames = []
for i, p in enumerate(units):
    g = gpd.read_file(p)
    g["huc10"] = p.name.split("_")[0]
    frames.append(g)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(units)}", flush=True)
basins = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True),
                          geometry="geometry", crs=frames[0].crs)
del frames
print(f"{len(basins):,} basins statewide", flush=True)

# fill SSURGO gaps with the statewide median KF (flagged), recompute chain
miss = basins["Soil_M1"].isna()
med = float(np.nanmedian(basins["Soil_M1"]))
S = basins["Soil_M1"].to_numpy().copy()
S[miss.to_numpy()] = med
basins["Soil_M1"] = S
basins["kf_filled"] = miss
T, F = basins["Terrain_M1"].to_numpy(), basins["Fire_M1"].to_numpy()
basins["P_24mmh"] = hz.likelihood_m1(T, F, S)
basins["I15_50"] = hz.threshold_i15(T, F, S, p=0.5)
basins["H_24mmh"] = hz.combined_c10(basins["P_24mmh"].to_numpy(),
                                    basins["V_24mmh"].to_numpy())
print(f"KF gaps filled: {int(miss.sum()):,} basins ({miss.mean():.1%}) at median {med:.3f}",
      flush=True)

z = np.load(paths.interim_dir("statewide") / "atlas14_i15.npz")
i1g, i50g, tr = z["i1"], z["i50"], Affine(*z["transform"])
cent = basins.geometry.centroid.to_crs("EPSG:4269")
cols = ((cent.x - tr.c) / tr.a).astype(int).to_numpy()
rows = ((cent.y - tr.f) / tr.e).astype(int).to_numpy()
ok = (rows >= 0) & (rows < i1g.shape[0]) & (cols >= 0) & (cols < i1g.shape[1])
i1 = np.full(len(basins), np.nan)
i50 = np.full(len(basins), np.nan)
i1[ok] = i1g[rows[ok], cols[ok]]
i50[ok] = i50g[rows[ok], cols[ok]]
basins["I15_1yr"], basins["I15_50yr"] = i1, i50
p_rt, m, b = ap.p_exceed_threshold(basins["I15_50"].to_numpy(), i1, i50)
basins["RI_thresh_yr"] = ap.recurrence_interval(basins["I15_50"].to_numpy(), m, b)
basins["P_RgtT"] = p_rt
basins["P_annual"] = ap.combined(p_rt)
no_clim = int((~np.isfinite(p_rt)).sum())

merged = OUT / "statewide_v1_basins.gpkg"
basins.to_file(merged, driver="GPKG")

h = basins["H_24mmh"]
summary = {
    "n_basins": int(len(basins)),
    "n_units": len(units),
    "total_area_km2": float(basins["Area_km2"].sum()),
    "calibration": "statewide_v1 per-region (CBR 0.52/325, NBR 0.54/355, MBR 0.55/388, SN 0.43/312)",
    "cdf_table": "nv_merged",
    "kf": "SSURGO (statewide), gaps median-filled and flagged",
    "likelihood": {k: float(v) for k, v in basins["P_24mmh"].describe().items()},
    "hazard_class": {int(c): int((h == c).sum()) for c in (1, 2, 3)},
    "zero_volume_frac": float((basins["V_24mmh"] <= 0).mean()),
    "median_threshold_mmh": float(basins["I15_50"].median()),
    "median_P_RgtT": float(np.nanmedian(basins["P_RgtT"])),
    "basins_without_atlas14": no_clim,
    "note": "first surface on the 123-fire per-region calibration (era-matched)",
}
(OUT / "statewide_v1_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print("wrote", merged)
