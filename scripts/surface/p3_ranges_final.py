"""Assemble the statewide RANGES volume columns onto statewide_v1 basins.

Joins the zonal parts (FracNorth, SlopeDeg), samples 25-km PGA and the
Atlas 14 1-yr i15 at basin centroids, computes V_ranges (+ bounds) and the
Cannon class under the RANGES volume, rewrites the merged GPKG, and updates
the summary with a Gartner-vs-RANGES comparison. Likelihood stays the v1
deterministic-severity M1 (the soft-severity fleet is a separate decision).
"""
import json
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from firescape import paths
from firescape import hazard as hz

OUT = paths.products_dir("prefire", "statewide_v1")
PARTS = paths.interim_dir("statewide") / "ranges_parts"

parts = sorted(PARTS.glob("*.parquet"))
print(f"{len(parts)} zonal parts")
topo = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
topo["huc10"] = topo["huc10"].astype(str)

basins = gpd.read_file(OUT / "statewide_v1_basins.gpkg")
basins["huc10"] = basins["huc10"].astype(str)
n0 = len(basins)
basins = basins.merge(topo[["huc10", "Segment_ID", "SlopeDeg", "FracNorth"]],
                      on=["huc10", "Segment_ID"], how="left", validate="1:1")
assert len(basins) == n0
print(f"joined topo for {basins['SlopeDeg'].notna().sum():,}/{n0:,} basins")

cent = basins.geometry.centroid
with rasterio.open(paths.interim_dir("statewide") / "pga25_g.tif") as pg:
    arr, tr = pg.read(1), pg.transform
cols = ((cent.x - tr.c) / tr.a).astype(int).to_numpy()
rows = ((cent.y - tr.f) / tr.e).astype(int).to_numpy()
ok = (rows >= 0) & (rows < arr.shape[0]) & (cols >= 0) & (cols < arr.shape[1])
pga = np.full(n0, np.nan)
pga[ok] = arr[rows[ok], cols[ok]]
basins["PGA25_g"] = pga

ratio = 24.0 / basins["I15_1yr"].to_numpy()
V, Vmin, Vmax = hz.volume_ranges(basins["Area_km2"].to_numpy(),
                                 basins["SlopeDeg"].to_numpy(), ratio, pga,
                                 basins["FracNorth"].to_numpy())
basins["i15_ratio"] = ratio
basins["Vr_24mmh"], basins["Vrmin_24mmh"], basins["Vrmax_24mmh"] = V, Vmin, Vmax
basins["Hr_24mmh"] = hz.combined_c10(basins["P_24mmh"].to_numpy(), V)

basins.to_file(OUT / "statewide_v1_basins.gpkg", driver="GPKG")
print("rewrote statewide_v1_basins.gpkg with RANGES columns")

hG = basins["H_24mmh"]
hR = basins["Hr_24mmh"]
finite = np.isfinite(V)
cmp = {
    "volume_model": "RANGES (McCoy Volume Playground, by-fire R2 0.742, RMSE 1.136 ln)",
    "predictors": "Area, SlopeDeg(10 m zonal), 24/I15_1yr, NSHM2023 PGA 25 km, FracNorth",
    "basins_with_V": int(finite.sum()),
    "basins_without_V": int((~finite).sum()),
    "median_V_m3": float(np.nanmedian(V)),
    "p90_V_m3": float(np.nanpercentile(V, 90)),
    "zero_volume_frac_gartner": float((basins["V_24mmh"] <= 0).mean()),
    "zero_volume_frac_ranges": float((~finite | (V <= 0)).mean()),
    "hazard_gartner": {int(c): int((hG == c).sum()) for c in (1, 2, 3)},
    "hazard_ranges": {int(c): int((hR == c).sum()) for c in (1, 2, 3)},
    "caveats": ["no NV fires in RANGES training set (Gorr 2026 inventory)",
                "UT (n=3) OOF bias -1.03: nearest Great Basin analog overpredicted ~2.8x",
                "likelihood term remains deterministic-severity v1 M1"],
}
summ_path = OUT / "statewide_v1_summary.json"
summ = json.loads(summ_path.read_text())
summ["ranges_volume"] = cmp
summ_path.write_text(json.dumps(summ, indent=2))
print(json.dumps(cmp, indent=2))
