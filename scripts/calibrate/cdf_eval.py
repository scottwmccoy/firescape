"""Does the Nevada refit beat the Staley 2018 baseline?

Builds the production table (Staley 2018 with Nevada refits overriding where
we have >=2000 era-matched pixels), then evaluates both tables on the Loyalton
fire across a P_dsim sweep, holding the segment network and soil term fixed.
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

from firescape import assess, hazard as hz, mtbs, paths, severity
from firescape.delineate import match_grid, network, usgs_filter

NAN32 = np.float32(np.nan)
PD_GRID = np.round(np.arange(0.20, 0.96, 0.02), 2)
BREAK = 281.0
OUTC = paths.products_dir("calibration")

# ---- merged production table ------------------------------------------------
staley = severity.load_cdf_table()
refit = pd.read_csv(OUTC / "CDFParameters_NV_refit.txt").set_index("EVT_Code")
merged = staley.copy()
merged["source"] = "staley2018"
for code in refit.index:
    if code in merged.index:
        for col in ("N", "Weibull_Lambda_Scale", "Weibull_Kappa_Shape",
                    "Weibull_R2", "Weibull_RMSE"):
            merged.at[code, col] = refit.at[code, col]
        merged.at[code, "source"] = "nv_refit"
    else:
        row = refit.loc[code]
        merged.loc[code] = {**{c: row.get(c) for c in merged.columns if c in row.index},
                            "source": "nv_refit"}
merged_path = OUTC / "CDFParameters_NV_merged.txt"
merged.reset_index().to_csv(merged_path, index=False)
n_over = int((merged["source"] == "nv_refit").sum())
print(f"merged table: {len(merged)} classes, {n_over} overridden by NV refit")

# ---- pilot-wide simulated severity ------------------------------------------
lf = paths.raw_dir("landfire") / "LF2025_EVT_pilot"
vat25 = gpd.read_file(lf / "LF2025_EVT_pilot.tif.vat.dbf")
map25, _ = severity.remap_crosswalk(vat25)
evt_pilot = severity.apply_crosswalk(
    rxr.open_rasterio(lf / "LF2025_EVT_pilot.tif", masked=False).squeeze().values, map25)
print("\npilot area simulating at moderate+ severity (break 281):")
for label, table in (("Staley 2018", staley), ("NV merged", merged)):
    for pd_ in (0.43, 0.48, 0.53):
        sim, _ = severity.simulate_dnbr(evt_pilot, pd_, table)
        print(f"  {label:>12} P_dsim {pd_}: {np.nanmean(sim >= BREAK):5.1%}")

# ---- Loyalton: rebuild the network once, sweep P_dsim for both tables -------
EVENT = "CA3968112017120200814"
bundle = mtbs.fire_bundle(EVENT)
dem_path = paths.interim_dir("pilot") / "pilot_dem.tif"
with rasterio.open(dem_path) as src:
    dem_crs = src.crs
perim_gdf = gpd.read_file(bundle["burn_area"]).to_crs(dem_crs)
w, s, e, n = perim_gdf.union_all().buffer(assess.PERIMETER_BUFFER_M).bounds
dem = Raster.from_file(dem_path, bounds=BoundingBox(w, s, e, n, crs=dem_crs))
res = dem.resolution("meters")
out_dir = paths.products_dir("assess", EVENT)
domain = match_grid(Raster.from_polygons(out_dir / "_domain.geojson", bounds=dem, resolution=res), dem)
perim = match_grid(Raster.from_polygons(bundle["burn_area"], bounds=dem, resolution=res), dem)
segments, terr = network(dem, domain)
dnbr6 = match_grid(Raster.from_file(bundle["dnbr6"]), dem, resampling="nearest")
barc4 = Raster.from_array(mtbs.dnbr6_to_barc4(dnbr6.values), spatial=dem, nodata=0)
usgs_filter(segments, burned=pfsev.mask(barc4, ["low", "moderate", "high"]),
            perimeter=perim, slopes=terr.slopes, dem_conditioned=terr.conditioned)
saved = gpd.read_file(out_dir / f"{EVENT}_segments.gpkg").set_index("Segment_ID").loc[segments.ids]
S = saved["Soil_M1"].to_numpy()
p_obs = saved["P_24mmh"].to_numpy()
print(f"\nLoyalton: {segments.size} segments, mean observed DFL {np.nanmean(p_obs):.3f}")

# EVT for the Loyalton domain, era-matched (LF2016 = pre-fire for a 2020 fire)
lf16 = paths.raw_dir("landfire") / "LF2016_EVT_pilot"
vat16 = gpd.read_file(lf16 / "LF2016_EVT_pilot.tif.vat.dbf")
map16, _ = severity.remap_crosswalk(vat16)
evt_loy = severity.apply_crosswalk(
    match_grid(Raster.from_file(lf16 / "LF2016_EVT_pilot.tif", bounds=dem.bounds),
               dem, resampling="nearest").values, map16)

rows = []
for label, table in (("Staley 2018", staley), ("NV merged", merged)):
    for pd_ in PD_GRID:
        sim_dnbr, _ = severity.simulate_dnbr(evt_loy, pd_, table)
        barc_sim = Raster.from_array(severity.classify_barc4(sim_dnbr, (125., BREAK, 500.)),
                                     spatial=dem, nodata=0)
        dnbr_sim = Raster.from_array(sim_dnbr.astype("float32"), spatial=dem, nodata=NAN32)
        T, F, _ = hz.m1_inputs(segments, barc_sim, terr.slopes, dnbr_sim,
                               Raster.from_array(np.full(dem.shape, 0.25, "float32"),
                                                 spatial=dem, nodata=NAN32), omitnan=True)
        p_sim = hz.likelihood_m1(T, F, S)
        ok = np.isfinite(p_obs) & np.isfinite(p_sim)
        r = p_sim[ok] - p_obs[ok]
        rows.append({"table": label, "pdsim": pd_,
                     "mean_sim": float(p_sim[ok].mean()),
                     "rmse": float(np.sqrt((r ** 2).mean())),
                     "nse": float(1 - (r ** 2).sum() / ((p_obs[ok] - p_obs[ok].mean()) ** 2).sum())})
sweep = pd.DataFrame(rows)
sweep.to_csv(OUTC / "cdf_refit_loyalton_sweep.csv", index=False)

print(f"\n{'table':>12}  best P_dsim   RMSE     NSE   mean sim (obs {np.nanmean(p_obs):.3f})")
best = {}
for label in ("Staley 2018", "NV merged"):
    sub = sweep[sweep["table"] == label]
    b = sub.loc[sub["nse"].idxmax()]
    best[label] = b.to_dict()
    print(f"{label:>12}      {b['pdsim']:.2f}    {b['rmse']:.3f}   {b['nse']:.3f}   {b['mean_sim']:.3f}")
    at48 = sub[sub["pdsim"] == 0.48].iloc[0]
    print(f"{'':>12}  (at 0.48: RMSE {at48['rmse']:.3f}, NSE {at48['nse']:.3f})")

(OUTC / "cdf_refit_eval.json").write_text(json.dumps(
    {"n_classes_overridden": n_over, "best": best,
     "observed_mean_dfl": float(np.nanmean(p_obs))}, indent=2))
print("\nwrote", merged_path)
