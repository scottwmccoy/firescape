"""Assemble statewide_v1.toml from per-fire calibration caches (all regions).

Run when the calibration driver reports zero pending. Writes the four-region
TOML (nv_merged table), a per-fire CSV, and prints staley-control medians.
"""
import warnings

warnings.filterwarnings("ignore")

from datetime import date

import numpy as np
import pandas as pd

from firescape import calibrate, paths

SET = (paths.package_data("calibration", "fire_sets", "statewide_v1.csv"))
OUT_TOML = (paths.package_data("calibration", "statewide_v1_1.toml"))

df = pd.read_csv(SET)
use = df[df["include"]].copy()
# mod_t=9999 = MTBS "no threshold" sentinel; keep it out of the break
# median (it inflated v1/v1_1/v1_2: central 325->307.5, mojave 388->345,
# northern 355->317.5). Sentinel fires stay in the P_dsim pool.
breaks = (use[(use["mod_t"] > 0) & (use["mod_t"] < 2000)]
          .groupby("region")["mod_t"].median())

rows = []
for _, row in use.iterrows():
    brk = float(breaks[row["region"]])
    tag = f"{row['evt_vintage'].split('_')[0].lower()}_b{brk:.0f}_d91"
    for table in ("staley2018_disp", "nv_merged_disp"):
        fc = calibrate.load_cached(row["event_id"], tag, table=table)
        if fc is not None and fc.ok:
            rows.append({"event_id": row["event_id"], "region": row["region"],
                         "table": table, "pdsim": fc.pdsim,
                         "n_selected": fc.n_selected,
                         "ig_year": row["ig_year"], "km2": row["km2"]})
cal = pd.DataFrame(rows)
if cal.empty:
    raise SystemExit("no cached calibrations yet")
nv = cal[cal["table"] == "nv_merged_disp"]
missing = set(use["event_id"]) - set(nv["event_id"])
print(f"calibrated fires: {nv['event_id'].nunique()}/{len(use)} "
      f"({len(missing)} missing)")

summary = nv.groupby("region")["pdsim"].agg(["median", "count",
                                             lambda s: s.quantile(.25),
                                             lambda s: s.quantile(.75)])
summary.columns = ["pdsim", "n", "q25", "q75"]
print(summary.round(3).to_string())
st = cal[cal["table"] == "staley2018_disp"].groupby("region")["pdsim"].median()
print("\nstaley2018 control medians:")
print(st.round(2).to_string())

def slug(r):
    return r.lower().replace(" ", "_").replace("&", "and")

lines = [
    "# Nevada statewide per-region P_dsim calibration (v1).",
    "# Era-matched: each fire paired with the newest LANDFIRE EVT strictly",
    "# predating its ignition year (LFPS catalog: LF2016/LF2022/LF2023 -",
    "# no LF2020 exists, so 2021-22 fires carry LF2016). Regions = EPA L3",
    "# ecoregions folded to four (data/regions/nv_prefire_regions.geojson);",
    "# breaks = per-region median analyst mod_t (Rossi 2025 rule).",
    "",
    "[meta]",
    'name = "statewide_v1_1"',
    f"created = {date.today().isoformat()}",
    'method = "rossi2025-dfl-calibration (logit-space, per-region, DISPERSED severity sigma=0.91)"',
    'cdf_source = "nv_merged (Staley 2018 + Nevada era-matched refit)"',
    'evt = "era-matched (LF2016/LF2022/LF2023)"',
    f"fires_used = {nv['event_id'].nunique()}",
]
for region in sorted(summary.index):
    s = summary.loc[region]
    fires = nv.loc[nv["region"] == region, "event_id"]
    lines += [
        "",
        f"[regions.{slug(region)}]",
        f"pdsim = {s['pdsim']:.2f}",
        f"barc_breaks = [125.0, {breaks[region]:.0f}, 500.0]",
        f"# n={int(s['n'])}, IQR=[{s['q25']:.2f}, {s['q75']:.2f}], "
        f"staley_control={st.get(region, float('nan')):.2f}",
        "fires = [" + ", ".join(f'"{e}"' for e in fires) + "]",
    ]
with open(OUT_TOML, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nwrote {OUT_TOML}")
cal.to_csv(paths.products_dir("calibration") / "statewide_v1_1_fires.csv", index=False)
print("wrote products/calibration/statewide_v1_fires.csv")
