"""Diff the Sierra test units (sw_sierra_v14_test) against shipped statewide_v1_2."""
import json
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd

from firescape import paths

pd.set_option("display.width", 220)
regions = pd.read_csv(paths.interim_dir("statewide") / "hu10_regions.csv", dtype={"huc10": str})
sierra = sorted(regions.loc[regions.region == "Sierra Nevada", "huc10"])
BASE = paths.products_dir("prefire", "statewide_v1_2")
TEST = paths.products_dir("prefire", "_sierra_v1_4_test")


def load(d, units):
    fr = []
    for h in units:
        p = d / f"{h}_basins.gpkg"
        if not p.exists():                       # test units are keyed "<variant>_<huc10>"
            alt = sorted(d.glob(f"*_{h}_basins.gpkg"))
            p = alt[0] if alt else p
        if p.exists():
            g = gpd.read_file(p); g["huc10"] = h; fr.append(g)
    return pd.concat(fr, ignore_index=True) if fr else None


base = load(BASE, sierra)
out = {}
for variant in ("v1_4", "break_only"):
    t = load(TEST / variant, sierra)
    if t is None:
        print(f"{variant}: nothing yet"); continue
    units = sorted(set(t.huc10))
    b = base[base.huc10.isin(units)]
    # basins are delineated identically (same DEM, same masks) -> join on unit + Segment_ID
    m = b.merge(t, on=["huc10", "Segment_ID"], suffixes=("_v12", "_new"))
    print(f"\n=== {variant}: {len(units)} units, {len(m):,} matched basins of {len(b):,} ===")
    for col, lab in (("Bmh_km2", "moderate+ burned area per basin (km2)"), ("P_24mmh", "M1 likelihood at 24 mm/h"),
                     ("V_24mmh", "volume (m3)"), ("I15_50", "rainfall threshold I15 at P=0.5 (mm/h)")):
        a, n = m[f"{col}_v12"], m[f"{col}_new"]
        print(f"  {lab:42s} v1_2 mean {a.mean():9.3f} -> {n.mean():9.3f}  ({(n.mean()/a.mean()-1)*100:+.0f}%)   "
              f"median {a.median():8.3f} -> {n.median():8.3f}")
    hc = pd.crosstab(m["H_24mmh_v12"], m["H_24mmh_new"])
    print("  hazard class, shipped (rows) vs new (cols):"); print("  " + hc.to_string().replace("\n", "\n  "))
    dP = m["P_24mmh_new"] - m["P_24mmh_v12"]
    print(f"  ΔP: mean {dP.mean():+.3f}, 5-95% [{dP.quantile(.05):+.3f}, {dP.quantile(.95):+.3f}], "
          f"basins with |ΔP| > 0.05: {(dP.abs() > .05).mean()*100:.0f}%")
    zero = (m["Bmh_km2_v12"] > 0) & (m["Bmh_km2_new"] == 0)
    print(f"  basins whose moderate+ area went to zero: {int(zero.sum())} of {int((m['Bmh_km2_v12']>0).sum())}")
    out[variant] = {"units": len(units), "basins": int(len(m)),
                    "Bmh_change_pct": float((m["Bmh_km2_new"].mean() / m["Bmh_km2_v12"].mean() - 1) * 100),
                    "P_mean_v12": float(m["P_24mmh_v12"].mean()), "P_mean_new": float(m["P_24mmh_new"].mean()),
                    "class2_v12": int((m["H_24mmh_v12"] >= 2).sum()), "class2_new": int((m["H_24mmh_new"] >= 2).sum()),
                    "I15_median_v12": float(m["I15_50_v12"].median()), "I15_median_new": float(m["I15_50_new"].median())}
(TEST / "comparison.json").write_text(json.dumps(out, indent=2))
print(f"\n-> {TEST/'comparison.json'}")
