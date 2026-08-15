"""P(F) x P(R>T): the annual probability of a postfire debris flow.

Everything upstream in this project is *conditional on the basin burning*.
P(R>T) asks how often the triggering storm arrives; multiplying by the annual
burn probability P(F) turns "hazard if it burns" into an expected annual rate,
which is what a planner or a hazard map legend actually needs.

    P_annual = P(F) x P(R>T)          Rossi et al. (2025), and the seam that
                                      firescape.annualprob.combined has held
                                      open since M5.

Two honest caveats travel with this layer:

* **Independence is assumed.** The product treats fire occurrence and storm
  occurrence as independent. They are not perfectly so -- monsoon convection
  both ignites fires and delivers the triggering rain -- so the product is a
  first-order estimate. It is also a *steady-state* rate: P(F) reflects fuels
  as of 2020, and the debris-flow response is only elevated for the few years
  after a fire, which this formulation does not resolve.
* **P(F) is sampled, not catchment-averaged.** Basins here nest (each segment
  carries its full upstream catchment), so a rasterize-and-bincount zonal mean
  would assign each cell to a single basin and silently corrupt the overlaps.
  Instead BP is aggregated to 270 m -- its true FSim resolution, the 30 m
  product being an upsample -- and sampled at each basin's representative
  point. For basins of 0.025-8 km2 against a 270 m field this is a small
  approximation. The exact fix is to catchment-summarise BP inside run_unit,
  where the flow network is already in hand; that is the right move for the
  next fleet, not a reason to hold this layer.
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import rasterio
from pyogrio import read_dataframe, write_dataframe
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject

from firescape import annualprob, paths

VERSION = sys.argv[1] if len(sys.argv) > 1 else "statewide_v1_1"
OUT = paths.products_dir("prefire", VERSION)
GPKG = OUT / f"{VERSION}_basins.gpkg"
BP_TIF = paths.raw_dir("wrc") / "BP_NV.tif"
BP_RES = 270.0        # FSim's real cell size; the 30 m raster is an upsample

if not GPKG.exists():
    sys.exit(f"no basins at {GPKG}")

print(f"reading {GPKG.name}", flush=True)
# Write back to the layer we read. pyogrio's write_dataframe ADDS a layer
# rather than replacing the file, so naming it anything else leaves the
# original layer in place, silently unchanged, and doubles the file.
import pyogrio
LAYER = str(pyogrio.list_layers(GPKG)[0][0])
g = read_dataframe(GPKG, layer=LAYER)
print(f"  {len(g):,} basins (layer {LAYER!r})", flush=True)

# --- BP onto a 270 m grid, warped exactly once from native ------------------
with rasterio.open(BP_TIF) as src:
    w, s, e, n = src.bounds
    width = int(np.ceil((e - w) / BP_RES))
    height = int(np.ceil((n - s) / BP_RES))
    tr = from_origin(w, n, BP_RES, BP_RES)
    bp = np.full((height, width), np.nan, dtype="float32")
    arr = src.read(1).astype("float32")
    if src.nodata is not None:
        arr[arr == np.float32(src.nodata)] = np.nan
    arr[arr < 0] = np.nan
    reproject(arr, bp, src_transform=src.transform, src_crs=src.crs,
              src_nodata=np.nan, dst_transform=tr, dst_crs=src.crs,
              dst_nodata=np.nan, resampling=Resampling.average)
    del arr
print(f"BP at {BP_RES:g} m: {bp.shape}, "
      f"{np.isfinite(bp).mean():.1%} covered, "
      f"mean {np.nanmean(bp):.5f}, max {np.nanmax(bp):.5f}", flush=True)

# --- sample at basin representative points ---------------------------------
pts = g.geometry.representative_point()
col = ((pts.x.to_numpy() - tr.c) / BP_RES).astype(int)
row = ((tr.f - pts.y.to_numpy()) / BP_RES).astype(int)
inside = (col >= 0) & (col < bp.shape[1]) & (row >= 0) & (row < bp.shape[0])
pf = np.full(len(g), np.nan, dtype="float64")
pf[inside] = bp[row[inside], col[inside]]

# A basin whose point lands on a nodata cell (water, a data hole) takes the
# mean of whatever is finite in a 3x3 neighbourhood before giving up.
gap = inside & ~np.isfinite(pf)
if gap.any():
    for i in np.flatnonzero(gap):
        r0, c0 = row[i], col[i]
        win = bp[max(0, r0 - 1):r0 + 2, max(0, c0 - 1):c0 + 2]
        if np.isfinite(win).any():
            pf[i] = np.nanmean(win)
print(f"P(F) sampled: {np.isfinite(pf).mean():.2%} of basins "
      f"({int((~np.isfinite(pf)).sum()):,} without a value)", flush=True)

g["P_F"] = pf
g["P_annual"] = annualprob.combined(g["P_RgtT"].to_numpy(), pf)
g["RI_annual_yr"] = np.where(g["P_annual"] > 0, 1.0 / g["P_annual"], np.nan)

ok = np.isfinite(g["P_annual"])
summary = {
    "version": VERSION,
    "basins": int(len(g)),
    "basins_with_P_annual": int(ok.sum()),
    "P_F": {"mean": round(float(np.nanmean(pf)), 6),
            "median": round(float(np.nanmedian(pf)), 6),
            "p95": round(float(np.nanpercentile(pf, 95)), 6),
            "max": round(float(np.nanmax(pf)), 6),
            "implied_mean_fire_return_yr": round(float(1 / np.nanmean(pf)), 1)},
    "P_RgtT_median": round(float(np.nanmedian(g["P_RgtT"])), 4),
    "P_annual": {
        "mean": round(float(np.nanmean(g["P_annual"][ok])), 6),
        "median": round(float(np.nanmedian(g["P_annual"][ok])), 6),
        "p95": round(float(np.nanpercentile(g["P_annual"][ok], 95)), 6),
        "max": round(float(np.nanmax(g["P_annual"][ok])), 6)},
    "median_return_interval_yr": round(
        float(np.nanmedian(g["RI_annual_yr"][ok])), 1),
    "basins_over_1_in_100_yr": int((g["P_annual"] > 0.01).sum()),
    "basins_over_1_in_500_yr": int((g["P_annual"] > 0.002).sum()),
}
print(json.dumps(summary, indent=2), flush=True)
(OUT / "annual_probability.json").write_text(json.dumps(summary, indent=2))

tmp = GPKG.with_suffix(".tmp.gpkg")
write_dataframe(g, tmp, layer=LAYER)
os.replace(tmp, GPKG)          # atomic: never leave a half-written product
print(f"wrote P_F, P_annual, RI_annual_yr -> {GPKG} (layer {LAYER!r})", flush=True)
