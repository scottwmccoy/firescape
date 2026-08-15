"""M4/M5 merge: stitch HU10 units into the pilot product and add P(R>T)."""
import json
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine

from firescape import annualprob as ap
from firescape import paths

OUT = paths.products_dir("prefire", "pilot_v1")
units = sorted(p for p in OUT.glob("*_basins.gpkg") if p.name[0].isdigit())
print(f"merging {len(units)} HU10 units")

frames = []
for p in units:
    g = gpd.read_file(p)
    g["huc10"] = p.name.split("_")[0]
    frames.append(g)
basins = pd.concat(frames, ignore_index=True)
basins = gpd.GeoDataFrame(basins, geometry="geometry", crs=frames[0].crs)
print(f"{len(basins)} basins, CRS {basins.crs}")

# --- Atlas 14 sampling (basins are typically smaller than one 800 m cell, so
# centroid sampling approximates Rossi's zonal mean) -------------------------
z = np.load(paths.interim_dir("pilot") / "atlas14_i15.npz")
i1_grid, i50_grid = z["i1"], z["i50"]
tr = Affine(*z["transform"])
cent = basins.geometry.centroid.to_crs("EPSG:4269")
cols = ((cent.x - tr.c) / tr.a).astype(int).to_numpy()
rows = ((cent.y - tr.f) / tr.e).astype(int).to_numpy()
valid = ((rows >= 0) & (rows < i1_grid.shape[0]) & (cols >= 0) & (cols < i1_grid.shape[1]))
i1 = np.full(len(basins), np.nan)
i50 = np.full(len(basins), np.nan)
i1[valid] = i1_grid[rows[valid], cols[valid]]
i50[valid] = i50_grid[rows[valid], cols[valid]]
basins["I15_1yr"] = i1
basins["I15_50yr"] = i50

p_rt, m, b = ap.p_exceed_threshold(basins["I15_50"].to_numpy(), i1, i50)
basins["RI_thresh_yr"] = ap.recurrence_interval(basins["I15_50"].to_numpy(), m, b)
basins["P_RgtT"] = p_rt
basins["P_annual"] = ap.combined(p_rt)  # x P(F) once FSim lands

merged = OUT / "pilot_v1_basins.gpkg"
basins.to_file(merged, driver="GPKG")

summary = {
    "n_basins": int(len(basins)),
    "n_units": len(units),
    "area_km2_total": float(basins["Area_km2"].sum()),
    "likelihood_24mmh": {k: float(v) for k, v in
                          basins["P_24mmh"].describe().items()},
    "volume_24mmh_m3": {k: float(v) for k, v in
                         basins["V_24mmh"].describe().items()},
    "hazard_class_counts": {int(k): int(v) for k, v in
                             basins["H_24mmh"].value_counts().sort_index().items()},
    "threshold_i15_mmh": {k: float(v) for k, v in basins["I15_50"].describe().items()},
    "atlas14_i15_1yr_mmh": {k: float(v) for k, v in basins["I15_1yr"].describe().items()},
    "P_RgtT": {k: float(v) for k, v in basins["P_RgtT"].describe().items()},
    "threshold_RI_yr_median": float(np.nanmedian(basins["RI_thresh_yr"])),
    "kf_note": "see per-unit *_meta.json kf_source",
}
(OUT / "pilot_v1_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2)[:1400])
print("\nwrote", merged)
