"""Statewide Nevada EVT-dNBR CDF refit (resumable, exit 42 = run me again).

The committed ``nv_merged`` table refit 29 Staley classes from 16 pilot-area
fires. That left the two regions farthest from the pilot leaning on borrowed
western-US fits: Mojave Mid-Elevation Mixed Desert Scrub (8.4% of calibration
basin area) and the Columbia Plateau steppe/juniper group (~7.3%) had no
Nevada data at all, and even the refit classes rested on Sierra-front pixels.

This refits from the full statewide era-matched set: every calibration fire
paired with the newest LANDFIRE EVT strictly predating its ignition
(LF2016 for 2017-22, LF2022 for 2023, LF2023 for 2024), sampled inside the
MTBS perimeter with greening/non-mapping pixels dropped.

Pass 1 caches a per-fire sample; pass 2 pools and fits. Sampling is the long
pole, so the cache makes the run restartable and the fit instant to redo.

Per-fire-per-class sampling is capped (PER_FIRE_CAP) so a handful of very
large fires cannot define a class on their own -- pooling raw would let one
880 km2 fire outvote thirty small ones in the same vegetation.
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
from rasterio.warp import Resampling, reproject

from firescape import mtbs, paths, severity, statewide

T0 = time.time()
BUDGET = float(os.environ.get("FIRESCAPE_REFIT_BUDGET", 900))
SET = (paths.package_data("calibration", "fire_sets", "statewide_v1.csv"))
CACHE = paths.interim_dir("calib", "refit_samples_statewide")
CACHE.mkdir(parents=True, exist_ok=True)
PER_FIRE_CAP = 120_000        # pixels per class per fire
PER_CLASS_CAP = 4_000_000     # pixels per class, pooled
MIN_N = 2000                  # below this a class keeps its existing fit
RNG = np.random.default_rng(17)

use = pd.read_csv(SET).query("include").copy()
print(f"statewide era-matched fire set: {len(use)} fires", flush=True)

_xwalk: dict = {}


def crosswalk_for(vintage):
    """(code mapping, EVT chunk dir) for a LANDFIRE vintage."""
    if vintage not in _xwalk:
        short = vintage.split("_")[0]
        dirn = paths.raw_dir("landfire", f"{short}_EVT_nv")
        vats = sorted(dirn.glob("*/evt_*.tif.vat.dbf"))
        if not vats:
            _xwalk[vintage] = None
        else:
            legend = pd.concat([gpd.read_file(v) for v in vats],
                               ignore_index=True).drop_duplicates(subset="Value")
            mapping, _ = severity.remap_crosswalk(legend)
            names = dict(zip(legend["Value"].astype(int),
                             legend["EVT_NAME"].astype(str)))
            _xwalk[vintage] = (mapping, dirn, names)
    return _xwalk[vintage]


def sample_fire(eid, vintage):
    """dNBR values by crosswalked EVT class inside one fire's perimeter."""
    xw = crosswalk_for(vintage)
    if xw is None:
        raise FileNotFoundError(f"no EVT chunks for {vintage}")
    mapping, chunk_dir, names = xw
    bundle = mtbs.fire_bundle(eid)

    dnbr = rxr.open_rasterio(bundle["dnbr"], masked=True).squeeze()
    perim = gpd.read_file(bundle["burn_area"]).to_crs(dnbr.rio.crs)
    inside = dnbr.rio.clip(perim.geometry, drop=False)
    d = inside.values.astype("float32")

    # era-matched EVT over the fire, warped once onto the dNBR grid (nearest --
    # these are class codes, never interpolate them)
    p5070 = gpd.read_file(bundle["burn_area"]).to_crs("EPSG:5070")
    out = statewide._mosaic_to_grid(
        sorted(chunk_dir.glob("*/evt_*.tif")), p5070.union_all().bounds,
        resolution=30.0, resampling="nearest", dtype="int32")
    if out is None:
        raise FileNotFoundError("no EVT chunk intersects the perimeter")
    src, src_tr = out
    e = np.zeros(d.shape, dtype="int32")
    reproject(src, e, src_transform=src_tr, src_crs="EPSG:5070", src_nodata=0,
              dst_transform=inside.rio.transform(), dst_crs=inside.rio.crs,
              dst_nodata=0, resampling=Resampling.nearest)

    ok = np.isfinite(d) & (d > -1000) & (d < 1000) & (e > 0)
    if "dnbr6" in bundle:                      # drop greening (5), non-map (6)
        d6 = rxr.open_rasterio(bundle["dnbr6"], masked=False).squeeze()
        d6_on = d6.rio.reproject_match(
            inside, resampling=Resampling.nearest).values
        ok &= np.isin(d6_on, (1, 2, 3, 4))

    codes = severity.apply_crosswalk(e[ok], mapping)
    vals = d[ok]
    per_class = {}
    for c in np.unique(codes):
        v = vals[codes == c]
        if v.size > PER_FIRE_CAP:
            v = RNG.choice(v, PER_FIRE_CAP, replace=False)
        per_class[int(c)] = v.astype("float32")
    return per_class, names


# --- pass 1: sample each fire once -----------------------------------------
pending = []
for _, row in use.sort_values("km2", ascending=False).iterrows():
    eid = row["event_id"]
    dest = CACHE / f"{eid}.npz"
    if dest.exists():
        continue
    if time.time() - T0 > BUDGET:
        pending.append((eid, "budget"))
        continue
    try:
        per_class, names = sample_fire(eid, row["evt_vintage"])
    except Exception as exc:
        print(f"  {eid} {str(row['incid_name'])[:18]:<18} SKIP "
              f"{type(exc).__name__}: {str(exc)[:80]}", flush=True)
        np.savez_compressed(dest, empty=True)      # do not retry every pass
        continue
    np.savez_compressed(dest, codes=np.array(sorted(per_class), dtype="int32"),
                        **{f"c{c}": v for c, v in per_class.items()})
    print(f"  {eid} {str(row['incid_name'])[:18]:<18} "
          f"{sum(v.size for v in per_class.values()):>9,} px, "
          f"{len(per_class)} classes ({time.time()-T0:.0f}s)", flush=True)

cached = sorted(CACHE.glob("NV*.npz"))
print(f"\nsampled {len(cached)}/{len(use)} fires", flush=True)
if pending:
    print(f"{len(pending)} pending -- rerun", flush=True)
    sys.exit(42)

# --- pass 2: pool and fit ---------------------------------------------------
pooled: dict[int, list] = {}
for f in cached:
    z = np.load(f, allow_pickle=True)
    if "codes" not in z:
        continue
    for c in z["codes"]:
        pooled.setdefault(int(c), []).append(z[f"c{c}"])
pooled = {c: np.concatenate(v) for c, v in pooled.items()}
for c, v in pooled.items():
    if v.size > PER_CLASS_CAP:
        pooled[c] = RNG.choice(v, PER_CLASS_CAP, replace=False)
print(f"pooled {sum(v.size for v in pooled.values()):,} pixels over "
      f"{len(pooled)} EVT classes "
      f"({sum(1 for v in pooled.values() if v.size >= MIN_N)} above n={MIN_N:,})",
      flush=True)

# class names, from whichever vintage legends were loaded
names: dict[int, str] = {}
for vintage in use["evt_vintage"].unique():
    xw = crosswalk_for(vintage)
    if xw is None:
        continue
    mapping, _, nm = xw
    for src, tgt in mapping.items():
        names.setdefault(int(tgt), nm.get(int(src), ""))

new = severity.refit_table(pooled, min_n=MIN_N, classnames=names)
staley = severity.load_cdf_table()
current = severity.load_cdf_table(table="nv_merged")

OUT = paths.products_dir("calibration")
new.reset_index().to_csv(OUT / "CDFParameters_NV2_refit.csv", index=False)

merged = staley.copy()
for code in new.index:
    merged.loc[code, new.columns] = new.loc[code]
merged = merged.sort_index()
merged.reset_index().to_csv(OUT / "CDFParameters_NV2_merged.csv", index=False)
print(f"refit {len(new)} classes; merged table {len(merged)} classes -> {OUT}",
      flush=True)

# --- how the three tables differ where it matters ---------------------------
PD, BREAK = 0.54, 340.0
rows = []
for code in new.index:
    obs = float(np.median(pooled[code]))
    row = {"code": code, "n": int(new.at[code, "N"]),
           "class": str(new.at[code, "CLASSNAME"])[:40],
           "obs_median": round(obs, 1),
           "new": round(float(severity.weibull_dnbr(
               PD, new.at[code, "Weibull_Lambda_Scale"],
               new.at[code, "Weibull_Kappa_Shape"])), 1),
           "r2": round(float(new.at[code, "Weibull_R2"]), 3)}
    for label, tab in (("staley", staley), ("nv_merged", current)):
        row[label] = (round(float(severity.weibull_dnbr(
            PD, tab.at[code, "Weibull_Lambda_Scale"],
            tab.at[code, "Weibull_Kappa_Shape"])), 1)
            if code in tab.index else np.nan)
    # was this class already carrying a Nevada fit, or is this its first?
    row["first_nv_fit"] = bool(
        code not in current.index
        or (code in staley.index and np.isclose(
            current.at[code, "Weibull_Lambda_Scale"],
            staley.at[code, "Weibull_Lambda_Scale"])))
    rows.append(row)
comp = pd.DataFrame(rows).sort_values("n", ascending=False)
comp.to_csv(OUT / "cdf_refit_statewide_comparison.csv", index=False)

# weighted absolute error against observed class medians, per table
w = comp["n"].to_numpy(dtype=float)
print(f"\nweighted |simulated - observed| median dNBR at P_dsim={PD} "
      f"(n-weighted over {len(comp)} refit classes):")
for label in ("staley", "nv_merged", "new"):
    v = comp[label].to_numpy(dtype=float)
    m = np.isfinite(v)
    print(f"  {label:>10}: {np.average(np.abs(v[m] - comp['obs_median'].to_numpy()[m]), weights=w[m]):6.2f} dNBR")

pd.set_option("display.width", 220)
print("\ntop classes by sample size:")
print(comp[["code", "n", "class", "obs_median", "staley", "nv_merged", "new", "r2"]]
      .head(18).to_string(index=False), flush=True)
