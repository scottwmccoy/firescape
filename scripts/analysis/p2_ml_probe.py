"""ML severity feasibility probe: does conditioning beyond EVT class help?

Question (framed by the lit survey): the Staley/Rossi simulator draws dNBR
from per-EVT-class marginals; published RF models (Wells 2023/24, Klimas
2025) condition on terrain/fuels too and report leave-fire-out R2 0.25-0.41.
Probe: leave-one-fire-out over the era-matched pilot fires, per-class-median
baseline (the marginal analog) vs gradient boosting on class + terrain.
"""
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
from rasterio.enums import Resampling

from firescape import mtbs, paths, severity

COV = paths.interim_dir("calib") / "refit_samples_cov.parquet"

if COV.exists():
    df = pd.read_parquet(COV)
    print(f"loaded {len(df):,} cached covariate samples")
else:
    FIRE_SET = ("/Users/scottmccoy/git/code/firescape/firescape/data/"
                "calibration/fire_sets/pilot_v1.csv")
    EVT_DIR = paths.raw_dir("landfire") / "LF2016_EVT_pilot"
    fires = pd.read_csv(FIRE_SET)
    use = fires[fires["include"] & (fires["ig_year"] >= 2017)]
    vat = gpd.read_file(EVT_DIR / "LF2016_EVT_pilot.tif.vat.dbf")
    mapping, _ = severity.remap_crosswalk(vat)
    evt_full = rxr.open_rasterio(EVT_DIR / "LF2016_EVT_pilot.tif",
                                 masked=False).squeeze()
    dem_full = rxr.open_rasterio(paths.interim_dir("pilot") / "pilot_dem.tif",
                                 masked=True).squeeze()
    parts = []
    for _, row in use.iterrows():
        eid = row["event_id"]
        try:
            bundle = mtbs.fire_bundle(eid)
        except FileNotFoundError:
            continue
        dnbr = rxr.open_rasterio(bundle["dnbr"], masked=True).squeeze()
        evt_on = evt_full.rio.reproject_match(dnbr, resampling=Resampling.nearest)
        elev = dem_full.rio.reproject_match(dnbr, resampling=Resampling.bilinear)
        gy, gx = np.gradient(elev.values.astype("float64"), 30.0)
        slope = np.hypot(gx, gy)
        asp = np.arctan2(-gx, gy)
        perim = gpd.read_file(bundle["burn_area"]).to_crs(dnbr.rio.crs)
        inside = dnbr.rio.clip(perim.geometry, drop=False)
        d = inside.values.astype("float64")
        e = evt_on.values
        z = elev.values
        ok = (np.isfinite(d) & (d > -1000) & (d < 1000) & (e > 0)
              & np.isfinite(z) & np.isfinite(slope))
        if "dnbr6" in bundle:
            d6 = rxr.open_rasterio(bundle["dnbr6"], masked=False).squeeze()
            ok &= np.isin(d6.rio.reproject_match(
                dnbr, resampling=Resampling.nearest).values, (1, 2, 3, 4))
        codes = severity.apply_crosswalk(e[ok], mapping)
        parts.append(pd.DataFrame({
            "event_id": eid, "code": codes.astype("int32"),
            "dnbr": d[ok].astype("float32"),
            "elev": z[ok].astype("float32"),
            "slope": slope[ok].astype("float32"),
            "northness": np.cos(asp[ok]).astype("float32"),
            "eastness": np.sin(asp[ok]).astype("float32")}))
        print(f"  {eid}: {int(ok.sum()):,} px", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df["event_id"] = df["event_id"].astype("category")
    df.to_parquet(COV, index=False)
    print(f"cached {len(df):,} -> {COV}")

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

rng = np.random.default_rng(11)
counts = df.groupby("event_id", observed=True).size().sort_values(ascending=False)
folds = counts.head(6).index.tolist()
FEATS = ["code", "elev", "slope", "northness", "eastness"]

rows = []
for held in folds:
    tr = df[df["event_id"] != held]
    te = df[df["event_id"] == held]
    if len(tr) > 500_000:
        tr = tr.iloc[rng.choice(len(tr), 500_000, replace=False)]
    med = tr.groupby("code")["dnbr"].median()
    gmed = float(tr["dnbr"].median())
    base_pred = te["code"].map(med).fillna(gmed).to_numpy()

    X = tr[FEATS].copy(); X["code"] = X["code"].astype("category")
    Xt = te[FEATS].copy(); Xt["code"] = Xt["code"].astype("category")
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08,
                                      categorical_features=["code"],
                                      random_state=0)
    m.fit(X, tr["dnbr"])
    hgb_pred = m.predict(Xt)

    y = te["dnbr"].to_numpy()
    rows.append({
        "held_out": held, "n_test": len(te),
        "r2_classmed": r2_score(y, base_pred),
        "r2_hgb": r2_score(y, hgb_pred),
        "mae_classmed": mean_absolute_error(y, base_pred),
        "mae_hgb": mean_absolute_error(y, hgb_pred)})
    print(f"{held}: class-median R2={rows[-1]['r2_classmed']:+.3f} "
          f"HGB R2={rows[-1]['r2_hgb']:+.3f}  "
          f"MAE {rows[-1]['mae_classmed']:.0f} -> {rows[-1]['mae_hgb']:.0f}",
          flush=True)

res = pd.DataFrame(rows)
res.to_csv(paths.products_dir("calibration") / "ml_probe_lofo.csv", index=False)
print("\nmeans: class-median R2 %.3f | HGB R2 %.3f | MAE %.0f -> %.0f"
      % (res["r2_classmed"].mean(), res["r2_hgb"].mean(),
         res["mae_classmed"].mean(), res["mae_hgb"].mean()))
print("wrote", paths.products_dir("calibration") / "ml_probe_lofo.csv")
