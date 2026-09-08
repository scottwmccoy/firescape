"""Which Stallion segments should have produced debris flows, 12-14 Aug 2026?

The pre-fire forecast gives each segment a triggering intensity. This drives
the same M1 model with the rainfall that actually fell -- MRMS peak 15-minute
intensity from the storm composite -- and returns, per segment, the modelled
likelihood *at the observed storm* rather than at a design storm. That turns a
forecast into a testable prediction with a date attached.

Rainfall is averaged over each segment's own catchment rather than sampled at a
point: M1's R is basin rainfall, and these basins nest, so a rasterize-once
zonal mean would misassign cells. The grid is small (41x99), so an honest
per-basin mask is cheap.

Caveats that bound how hard to push the result:

* **Severity is simulated, not observed.** Stallion was still burning through
  this storm window (discovered 6 Aug), so parts of the perimeter had not
  burned when the rain fell, and the real severity mosaic is unknown. Segments
  flagged here are where a *fully burned* landscape would respond.
* **MRMS at 1 km under-resolves convective cores**, which are often smaller
  than a grid cell; peak intensities in small basins can be higher than shown.
  RQI is carried through so low-confidence radar coverage is visible.

Output: per-segment observed I15, exceedance ratio, likelihood at the observed
storm, and which day drove the peak -- a field list, ordered.
"""
import json
import sys
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask

from firescape import hazard, paths

FIRE = sys.argv[1] if len(sys.argv) > 1 else "Stallion"
CAL = sys.argv[2] if len(sys.argv) > 2 else "statewide_v1_1"
STORM = (paths.research_root() / "2026_Bug_Stalion" / "storms" / "composite_20260812-20260814")
SRC = paths.products_dir("forecast", f"{FIRE.lower()}_{CAL}")
DAYS = ["S12Aug", "S13Aug", "S14Aug"]

seg = gpd.read_file(SRC / f"{FIRE.lower()}_segments.gpkg")
print(f"{len(seg):,} segments", flush=True)
thr_col = "I15_50" if "I15_50" in seg.columns else "thresh_i15"

# The saved basins layer holds only the 254 outlet catchments, not one per
# segment, so rainfall is averaged over each segment's own channel cells
# instead. At MRMS's 1 km grid against basins of 0.025-8 km2 the two differ
# little -- a basin spans a handful of cells at most.
BUF_M = 250.0


def raster_means(path, geoms_4326):
    """Mean of a raster over each geometry (all_touched, so short segments in
    a 1 km grid still pick up their cell)."""
    with rasterio.open(path) as ds:
        arr = ds.read(1, masked=True).filled(np.nan).astype("float64")
        tr, crs, shape = ds.transform, ds.crs, ds.shape
    g = geoms_4326.to_crs(crs)
    out = np.full(len(g), np.nan)
    for i, geom in enumerate(g.geometry):
        try:
            m = geometry_mask([geom], out_shape=shape, transform=tr,
                              invert=True, all_touched=True)
        except Exception:
            continue
        v = arr[m]
        v = v[np.isfinite(v)]
        if v.size:
            out[i] = v.mean()
    return out


print("averaging storm rainfall along each segment...", flush=True)
buffered = seg.to_crs("EPSG:5070").buffer(BUF_M).to_frame("geometry")
buffered = gpd.GeoDataFrame(buffered, crs="EPSG:5070")
i15 = {d: raster_means(f"{STORM}/per_storm/{d}_i15max.tif", buffered)
       for d in DAYS}
comp = raster_means(f"{STORM}/rasters/BSE3day_i15max.tif", buffered)
rqi = np.nanmax(np.stack([raster_means(f"{STORM}/per_storm/{d}_rqi.tif",
                                       buffered) for d in DAYS]), axis=0)

stack = np.stack([i15[d] for d in DAYS])
peak_day = np.array(DAYS)[np.nanargmax(np.where(np.isfinite(stack), stack, -1),
                                       axis=0)]
obs = np.nanmax(np.stack([comp, np.nanmax(stack, axis=0)]), axis=0)

seg["i15_obs"], seg["rqi"], seg["peak_day"] = obs, rqi, peak_day
for d in DAYS:
    seg[f"i15_{d}"] = i15[d]

thr = seg[thr_col].to_numpy()
seg["exceed_ratio"] = obs / thr

# Likelihood at the rainfall that actually fell. pfdf's s17.likelihood
# broadcasts an array R against the variables into a cross product, so the
# logistic is evaluated directly on matched per-segment values instead --
# validated against hazard.likelihood_m1 at the 24 mm/h design storm below.
from pfdf.models import s17

B, Ct, Cf, Cs = (float(np.ravel(x)[0]) for x in s17.M1.parameters(durations=[15]))
T = seg["Terrain_M1"].to_numpy(dtype=float)
F = seg["Fire_M1"].to_numpy(dtype=float)
S = seg["Soil_M1"].to_numpy(dtype=float)


def m1_likelihood(i15_mmh):
    R = np.asarray(i15_mmh, dtype=float) * 0.25        # mm accumulated in 15 min
    X = B + (Ct * T + Cf * F + Cs * S) * R
    return 1.0 / (1.0 + np.exp(-X))


check = m1_likelihood(np.full(len(seg), 24.0))
ref = seg["P_24mmh"].to_numpy(dtype=float)
good = np.isfinite(check) & np.isfinite(ref)
assert np.nanmax(np.abs(check[good] - ref[good])) < 1e-6, "M1 reimplementation disagrees"
print(f"M1 check against stored P_24mmh: max diff "
      f"{np.nanmax(np.abs(check[good] - ref[good])):.2e}", flush=True)

p_obs = np.where(np.isfinite(obs) & (obs > 0), m1_likelihood(obs), np.nan)
seg["P_observed"] = p_obs

fired = seg["exceed_ratio"] >= 1.0
summary = {
    "fire": FIRE, "calibration": CAL,
    "storm_window": "2026-08-12 to 2026-08-14",
    "segments": int(len(seg)),
    "observed_i15_mmh": {
        "median": round(float(np.nanmedian(obs)), 1),
        "p90": round(float(np.nanpercentile(obs, 90)), 1),
        "max": round(float(np.nanmax(obs)), 1)},
    "threshold_i15_mmh_median": round(float(np.nanmedian(thr)), 1),
    "segments_exceeding_threshold": int(fired.sum()),
    "pct_exceeding": round(100 * float(fired.mean()), 1),
    "segments_P_observed_over_0.5": int((p_obs > 0.5).sum()),
    "segments_P_observed_over_0.75": int((p_obs > 0.75).sum()),
    "segments_P_observed_over_0.9": int((p_obs > 0.9).sum()),
    "peak_day_of_exceeding": {d: int((peak_day[fired.to_numpy()] == d).sum())
                              for d in DAYS} if fired.any() else {},
    "min_rqi_among_exceeding": (round(float(np.nanmin(seg.loc[fired, "rqi"])), 2)
                                if fired.any() else None),
}
print(json.dumps(summary, indent=2), flush=True)
(SRC / "storm_response_summary.json").write_text(json.dumps(summary, indent=2))

seg.to_file(SRC / f"{FIRE.lower()}_storm_response.gpkg", driver="GPKG")

key = "Segment_ID" if "Segment_ID" in seg.columns else seg.columns[0]
top = seg.loc[p_obs > 0.5, [key, thr_col, "i15_obs", "exceed_ratio",
                            "P_observed", "peak_day", "rqi"]]
top = top.sort_values("P_observed", ascending=False)
top.to_csv(SRC / "priority_segments.csv", index=False)
print(f"\n{len(top):,} segments with likelihood > 0.5 at the observed storm "
      f"-> priority_segments.csv", flush=True)
pd.set_option("display.width", 200)
print(top.head(15).round(3).to_string(index=False), flush=True)
