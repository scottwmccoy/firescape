"""Nevada EVT-dNBR CDF refit, era-matched.

LANDFIRE EVT postdating a fire describes POST-fire vegetation, so a refit must
pair each fire with vegetation mapped BEFORE it burned. LFPS serves LF2016 as
its oldest product, so the defensible sample is fires that burned 2017+ with
LF2016 as their pre-fire vegetation.
"""
import json
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
from rasterio.enums import Resampling

from firescape import mtbs, paths, severity

FIRE_SET = ("/Users/scottmccoy/git/code/firescape/firescape/data/calibration/"
            "fire_sets/pilot_v1.csv")
EVT_DIR = paths.raw_dir("landfire") / "LF2016_EVT_pilot"
EVT_TIF = EVT_DIR / "LF2016_EVT_pilot.tif"
MIN_YEAR = 2017          # EVT vintage LF2016 -> only later fires are era-matched
OUT = paths.products_dir("calibration")

fires = pd.read_csv(FIRE_SET)
use = fires[fires["include"] & (fires["ig_year"] >= MIN_YEAR)].copy()
print(f"era-matched fires (LF2016 EVT, ignition >= {MIN_YEAR}): {len(use)}")
print(use[["event_id", "incid_name", "ig_year", "km2"]].to_string(index=False))

vat = gpd.read_file(EVT_DIR / "LF2016_EVT_pilot.tif.vat.dbf")
mapping, audit = severity.remap_crosswalk(vat)
names = dict(zip(vat["Value"].astype(int), vat["EVT_NAME"].astype(str)))
evt_full = rxr.open_rasterio(EVT_TIF, masked=False).squeeze()

samples: dict[int, list] = {}
for _, row in use.iterrows():
    eid = row["event_id"]
    try:
        bundle = mtbs.fire_bundle(eid)
    except FileNotFoundError:
        print(f"  {eid}: no bundle, skipped")
        continue
    dnbr = rxr.open_rasterio(bundle["dnbr"], masked=True).squeeze()
    evt_on = evt_full.rio.reproject_match(dnbr, resampling=Resampling.nearest)
    perim = gpd.read_file(bundle["burn_area"]).to_crs(dnbr.rio.crs)
    inside = dnbr.rio.clip(perim.geometry, drop=False)
    d = inside.values.astype("float64")
    e = evt_on.values
    ok = np.isfinite(d) & (d > -1000) & (d < 1000) & (e > 0)
    if "dnbr6" in bundle:   # drop greening (5) and non-mapping (6)
        d6 = rxr.open_rasterio(bundle["dnbr6"], masked=False).squeeze()
        d6_on = d6.rio.reproject_match(dnbr, resampling=Resampling.nearest).values
        ok &= np.isin(d6_on, (1, 2, 3, 4))
    codes = severity.apply_crosswalk(e[ok], mapping)
    vals = d[ok]
    for c in np.unique(codes):
        samples.setdefault(int(c), []).append(vals[codes == c])
    print(f"  {eid} {row['incid_name'][:18]:<18} {int(ok.sum()):>9,} px", flush=True)

pooled = {c: np.concatenate(v) for c, v in samples.items()}
print(f"\npooled {sum(v.size for v in pooled.values()):,} pixels across "
      f"{len(pooled)} EVT classes")

# name lookup back through the crosswalk
rev: dict[int, str] = {}
for src, tgt in mapping.items():
    rev.setdefault(int(tgt), names.get(int(src), ""))
new = severity.refit_table(pooled, min_n=2000, classnames=rev)
old = severity.load_cdf_table()

new_path = paths.products_dir("calibration") / "CDFParameters_NV_refit.txt"
new.reset_index().to_csv(new_path, index=False)
print(f"\nrefit {len(new)} classes -> {new_path}")

BREAK, PD = 281.0, 0.48
rows = []
for code in new.index:
    if code not in old.index:
        continue
    dn_new = severity.weibull_dnbr(PD, new.at[code, "Weibull_Lambda_Scale"],
                                   new.at[code, "Weibull_Kappa_Shape"])
    dn_old = severity.weibull_dnbr(PD, old.at[code, "Weibull_Lambda_Scale"],
                                   old.at[code, "Weibull_Kappa_Shape"])
    obs_med = float(np.median(pooled[code]))
    rows.append({"code": code, "n": int(new.at[code, "N"]),
                 "class": str(new.at[code, "CLASSNAME"])[:38],
                 "obs_median_dnbr": obs_med,
                 "staley_dnbr@0.48": dn_old, "refit_dnbr@0.48": dn_new,
                 "delta": dn_new - dn_old,
                 "lam_old": old.at[code, "Weibull_Lambda_Scale"],
                 "lam_new": new.at[code, "Weibull_Lambda_Scale"],
                 "kap_old": old.at[code, "Weibull_Kappa_Shape"],
                 "kap_new": new.at[code, "Weibull_Kappa_Shape"],
                 "r2_new": new.at[code, "Weibull_R2"]})
comp = pd.DataFrame(rows).sort_values("n", ascending=False)
comp.to_csv(OUT / "cdf_refit_comparison.csv", index=False)
pd.set_option("display.width", 200)
print("\n=== refit vs Staley 2018 (simulated dNBR at P_dsim=0.48, break=281) ===")
print(comp[["code", "n", "class", "obs_median_dnbr", "staley_dnbr@0.48",
            "refit_dnbr@0.48", "delta", "r2_new"]].head(16).to_string(index=False))

# how much of the pilot would simulate moderate+ under each table?
evt_pilot = severity.apply_crosswalk(
    rxr.open_rasterio(EVT_TIF, masked=False).squeeze().values, mapping)
for label, table in (("Staley 2018", old), ("NV refit", new)):
    sim, _ = severity.simulate_dnbr(evt_pilot, PD, table)
    print(f"{label:>12}: {np.nanmean(sim >= BREAK):5.1%} of pilot at moderate+ "
          f"(P_dsim {PD}, break {BREAK:.0f})")
