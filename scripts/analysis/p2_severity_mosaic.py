"""Statewide simulated-severity intermediates: the M2 layer, mapped.

Simulated dNBR is a pure per-pixel function of EVT class at fixed P_dsim, so
the statewide field comes straight from the LF2025 chunks — no hydrology.
Products (120 m, EPSG:5070, statewide_v0 config: nv_merged CDFs, P_dsim 0.51):
  statewide_v0_simdnbr_120m.tif   float32 dNBR(x1000), NaN = nodata/water
  statewide_v0_simbarc_120m.tif   uint8 BARC4 (0 nodata)
  statewide_v0_sevsrc_120m.tif    uint8 QA: 1 = class in CDF table, 2 = barren fallback
  severity_class_table.csv        per-EVT-class LUT + statewide areas
"""
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from firescape import config, delineate, paths, severity, statewide

RES = 120.0
OUT = paths.products_dir("prefire", "statewide_v0")
EVT_DIR = paths.raw_dir("landfire", "LF2025_EVT_nv")

cal = config.packaged_calibration("pilot_v2").region("pilot")
table = severity.load_cdf_table(table="nv_merged")
refit = severity.load_cdf_table(table="nv_refit")
legend = pd.read_csv(paths.interim_dir("statewide") / "evt_legend_union.csv")
mapping, _ = severity.remap_crosswalk(legend)
water = set(delineate.water_codes_from_legend(legend))
names = dict(zip(legend["Value"].astype(int), legend["EVT_NAME"].astype(str)))

hu = gpd.read_file(paths.interim_dir("statewide") / "nv_hu10.geojson").to_crs("EPSG:5070")
chunks = sorted(EVT_DIR.glob("*/evt_*.tif"))
print(f"mosaicking {len(chunks)} EVT chunks at {RES:.0f} m ...", flush=True)
arr, transform = statewide._mosaic_to_grid(chunks, tuple(hu.total_bounds),
                                           resolution=RES, resampling="nearest",
                                           dtype="int32")
print(f"grid {arr.shape[1]}x{arr.shape[0]}", flush=True)

mapped = severity.apply_crosswalk(arr, mapping)
dnbr, src = severity.simulate_dnbr(mapped, cal.pdsim, table, evt_nodata=0)
is_water = np.isin(arr, list(water))
dnbr[is_water] = np.nan
src[is_water] = 0
barc = severity.classify_barc4(dnbr, cal.barc_breaks)

prof = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
            crs="EPSG:5070", transform=transform, compress="deflate",
            tiled=True, blockxsize=512, blockysize=512)
for name, data, dtype, nodata in (
        ("statewide_v0_simdnbr_120m.tif", dnbr, "float32", np.nan),
        ("statewide_v0_simbarc_120m.tif", barc, "uint8", 0),
        ("statewide_v0_sevsrc_120m.tif", src, "uint8", 0)):
    with rasterio.open(OUT / name, "w", dtype=dtype, nodata=nodata, **prof) as ds:
        ds.write(data.astype(dtype), 1)
    print(f"wrote {OUT / name}", flush=True)

# ---- per-class table --------------------------------------------------------
vals, counts = np.unique(arr[arr > 0], return_counts=True)
km2 = counts * (RES * RES) / 1e6
rows = []
for v, a in zip(vals.tolist(), km2.tolist()):
    code = mapping.get(v, v)
    in_table = code in table.index
    lam = table.at[code, "Weibull_Lambda_Scale"] if in_table else np.nan
    kap = table.at[code, "Weibull_Kappa_Shape"] if in_table else np.nan
    if v in water:
        sim, cls, source = np.nan, 0, "water/masked"
    elif in_table:
        sim = severity.weibull_dnbr(cal.pdsim, lam, kap)
        cls = int(severity.classify_barc4(np.array([sim]), cal.barc_breaks)[0])
        source = "nv_refit" if code in refit.index else "staley2018"
    else:
        fb = severity.BARREN_EVT_CODE
        sim = severity.weibull_dnbr(cal.pdsim,
                                    table.at[fb, "Weibull_Lambda_Scale"],
                                    table.at[fb, "Weibull_Kappa_Shape"])
        cls = int(severity.classify_barc4(np.array([sim]), cal.barc_breaks)[0])
        source = "fallback_barren"
    rows.append({"evt_code": v, "evt_name": names.get(v, ""), "cdf_code": code,
                 "area_km2": round(a, 1), "source": source,
                 "lambda": lam, "kappa": kap,
                 "simdnbr_p51": None if not np.isfinite(sim) else round(float(sim), 1),
                 "barc4_p51": cls})
tab = pd.DataFrame(rows).sort_values("area_km2", ascending=False)
tab.to_csv(OUT / "severity_class_table.csv", index=False)

valid = src > 0
n = valid.sum()
print(f"\n=== statewide simulated severity (P_dsim={cal.pdsim}, nv_merged) ===")
print(f"domain pixels: {n:,} ({n * RES * RES / 1e6:,.0f} km2)")
for c, lbl in ((1, "unburned/very low"), (2, "low"), (3, "moderate"), (4, "high")):
    frac = (barc[valid] == c).mean()
    print(f"  BARC {c} {lbl:<18} {frac:6.1%}")
print(f"  fallback-barren share: {(src[valid] == 2).mean():.2%}")
print(f"  water/snow masked: {is_water.sum() * RES * RES / 1e6:,.0f} km2")
print("\ntop classes by area:")
print(tab.head(12)[["evt_code", "evt_name", "area_km2", "source",
                    "simdnbr_p51", "barc4_p51"]].to_string(index=False))
