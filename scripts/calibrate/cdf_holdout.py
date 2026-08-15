"""Leave-one-fire-out test: refit WITHOUT Loyalton, then predict Loyalton.

The in-sample evaluation is circular because Loyalton contributes pixels to the
refit. This holds it out entirely, which is the honest test of whether the
Nevada refit generalizes.
"""
import json
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray as rxr
from pfdf import severity as pfsev
from pfdf.projection import BoundingBox
from pfdf.raster import Raster
from rasterio.enums import Resampling

from firescape import assess, hazard as hz, mtbs, paths, severity
from firescape.delineate import match_grid, network, usgs_filter

HOLDOUT = "CA3968112017120200814"          # Loyalton
NAN32 = np.float32(np.nan)
BREAK = 281.0
PD_GRID = np.round(np.arange(0.20, 0.96, 0.02), 2)
OUTC = paths.products_dir("calibration")

fires = pd.read_csv("/Users/scottmccoy/git/code/firescape/firescape/data/"
                    "calibration/fire_sets/pilot_v1.csv")
use = fires[fires["include"] & (fires["ig_year"] >= 2017)
            & (fires["event_id"] != HOLDOUT)]
print(f"refitting from {len(use)} fires, Loyalton held out")

EVT_DIR = paths.raw_dir("landfire") / "LF2016_EVT_pilot"
vat = gpd.read_file(EVT_DIR / "LF2016_EVT_pilot.tif.vat.dbf")
mapping, _ = severity.remap_crosswalk(vat)
names = dict(zip(vat["Value"].astype(int), vat["EVT_NAME"].astype(str)))
evt_full = rxr.open_rasterio(EVT_DIR / "LF2016_EVT_pilot.tif", masked=False).squeeze()

samples: dict[int, list] = {}
for _, row in use.iterrows():
    try:
        bundle = mtbs.fire_bundle(row["event_id"])
    except FileNotFoundError:
        continue
    dnbr = rxr.open_rasterio(bundle["dnbr"], masked=True).squeeze()
    evt_on = evt_full.rio.reproject_match(dnbr, resampling=Resampling.nearest)
    perim = gpd.read_file(bundle["burn_area"]).to_crs(dnbr.rio.crs)
    d = dnbr.rio.clip(perim.geometry, drop=False).values.astype("float64")
    e = evt_on.values
    ok = np.isfinite(d) & (d > -1000) & (d < 1000) & (e > 0)
    if "dnbr6" in bundle:
        d6 = rxr.open_rasterio(bundle["dnbr6"], masked=False).squeeze()
        ok &= np.isin(d6.rio.reproject_match(dnbr, resampling=Resampling.nearest).values,
                      (1, 2, 3, 4))
    codes = severity.apply_crosswalk(e[ok], mapping)
    vals = d[ok]
    for c in np.unique(codes):
        samples.setdefault(int(c), []).append(vals[codes == c])
    print(f"  {row['event_id']} {row['incid_name'][:16]:<16} {int(ok.sum()):>9,} px", flush=True)

pooled = {c: np.concatenate(v) for c, v in samples.items()}
rev: dict[int, str] = {}
for src, tgt in mapping.items():
    rev.setdefault(int(tgt), names.get(int(src), ""))
refit = severity.refit_table(pooled, min_n=2000, classnames=rev)
staley = severity.load_cdf_table()
merged = staley.copy()
for code in refit.index:
    if code in merged.index:
        for col in ("Weibull_Lambda_Scale", "Weibull_Kappa_Shape"):
            merged.at[code, col] = refit.at[code, col]
    else:
        merged.loc[code] = refit.loc[code].reindex(merged.columns)
print(f"holdout refit: {len(refit)} classes ({sum(v.size for v in pooled.values()):,} px)")

# ---- predict the held-out fire ---------------------------------------------
bundle = mtbs.fire_bundle(HOLDOUT)
dem_path = paths.interim_dir("pilot") / "pilot_dem.tif"
with rasterio.open(dem_path) as src:
    dem_crs = src.crs
perim_gdf = gpd.read_file(bundle["burn_area"]).to_crs(dem_crs)
w, s, e, n = perim_gdf.union_all().buffer(assess.PERIMETER_BUFFER_M).bounds
dem = Raster.from_file(dem_path, bounds=BoundingBox(w, s, e, n, crs=dem_crs))
res = dem.resolution("meters")
out_dir = paths.products_dir("assess", HOLDOUT)
domain = match_grid(Raster.from_polygons(out_dir / "_domain.geojson", bounds=dem, resolution=res), dem)
perim = match_grid(Raster.from_polygons(bundle["burn_area"], bounds=dem, resolution=res), dem)
segments, terr = network(dem, domain)
dnbr6 = match_grid(Raster.from_file(bundle["dnbr6"]), dem, resampling="nearest")
barc4 = Raster.from_array(mtbs.dnbr6_to_barc4(dnbr6.values), spatial=dem, nodata=0)
usgs_filter(segments, burned=pfsev.mask(barc4, ["low", "moderate", "high"]),
            perimeter=perim, slopes=terr.slopes, dem_conditioned=terr.conditioned)
saved = gpd.read_file(out_dir / f"{HOLDOUT}_segments.gpkg").set_index("Segment_ID").loc[segments.ids]
S = saved["Soil_M1"].to_numpy()
p_obs = saved["P_24mmh"].to_numpy()

evt_loy = severity.apply_crosswalk(
    match_grid(Raster.from_file(EVT_DIR / "LF2016_EVT_pilot.tif", bounds=dem.bounds),
               dem, resampling="nearest").values, mapping)
kf_dummy = Raster.from_array(np.full(dem.shape, 0.25, "float32"), spatial=dem, nodata=NAN32)

rows = []
for label, table in (("Staley 2018", staley), ("NV refit (holdout)", merged)):
    for pd_ in PD_GRID:
        sim, _ = severity.simulate_dnbr(evt_loy, pd_, table)
        barc_sim = Raster.from_array(severity.classify_barc4(sim, (125., BREAK, 500.)),
                                     spatial=dem, nodata=0)
        T, F, _ = hz.m1_inputs(segments, barc_sim, terr.slopes,
                               Raster.from_array(sim.astype("float32"), spatial=dem, nodata=NAN32),
                               kf_dummy, omitnan=True)
        p_sim = hz.likelihood_m1(T, F, S)
        ok = np.isfinite(p_obs) & np.isfinite(p_sim)
        r = p_sim[ok] - p_obs[ok]
        rows.append({"table": label, "pdsim": pd_, "mean_sim": float(p_sim[ok].mean()),
                     "rmse": float(np.sqrt((r ** 2).mean())),
                     "nse": float(1 - (r ** 2).sum() / ((p_obs[ok] - p_obs[ok].mean()) ** 2).sum())})
sweep = pd.DataFrame(rows)
sweep.to_csv(OUTC / "cdf_holdout_sweep.csv", index=False)

print(f"\nHELD-OUT Loyalton (obs mean DFL {np.nanmean(p_obs):.3f}, {segments.size} segments)")
res_out = {}
for label in sweep["table"].unique():
    sub = sweep[sweep["table"] == label]
    b = sub.loc[sub["nse"].idxmax()]
    at48 = sub[sub["pdsim"] == 0.48].iloc[0]
    res_out[label] = {"best_pdsim": float(b["pdsim"]), "best_nse": float(b["nse"]),
                      "best_rmse": float(b["rmse"]),
                      "nse_at_0.48": float(at48["nse"]), "rmse_at_0.48": float(at48["rmse"])}
    print(f"  {label:>20}: best NSE {b['nse']:.3f} @ P_dsim {b['pdsim']:.2f} "
          f"(RMSE {b['rmse']:.3f}) | at 0.48: NSE {at48['nse']:.3f}, RMSE {at48['rmse']:.3f}")
(OUTC / "cdf_holdout_eval.json").write_text(json.dumps(res_out, indent=2))
