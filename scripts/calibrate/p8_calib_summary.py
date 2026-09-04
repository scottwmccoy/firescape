"""Assemble statewide_v1_4.toml from the per-fire caches (BARC256-corrected).

Run after ``p8_calib_driver`` reports zero pending. Writes the four-region
TOML (nv_statewide table), ``products/calibration/statewide_v1_4_fires.csv``,
and prints the v1_3 -> v1_4 deltas so the change is visible in the log.
"""
import tomllib
import warnings

warnings.filterwarnings("ignore")

from datetime import date

import pandas as pd

from firescape import calibrate

SET = ("/Users/scottmccoy/git/code/firescape/firescape/data/calibration/"
       "fire_sets/statewide_v1.csv")
CAL_DIR = "/Users/scottmccoy/git/code/firescape/firescape/data/calibration/"
OUT_TOML = CAL_DIR + "statewide_v1_4.toml"
PREV = tomllib.loads(open(CAL_DIR + "statewide_v1_3.toml", "rb").read().decode())

df = pd.read_csv(SET)
use = df[df["include"]].copy()
brk_pool = use[(use["mod_t"] > 0) & (use["mod_t"] < 2000)]
breaks = brk_pool.groupby("region")["mod_t"].median()
n_thr = brk_pool.groupby("region").size()
n_conv = brk_pool[brk_pool["threshold_scale"] == "barc256->dnbr"].groupby("region").size()

rows = []
for _, row in use.iterrows():
    brk = float(breaks[row["region"]])
    tag = f"{row['evt_vintage'].split('_')[0].lower()}_b{brk:.0f}_d91sw"
    for table in ("staley2018_disp", "nv_statewide_disp"):
        fc = calibrate.load_cached(row["event_id"], tag, table=table)
        if fc is not None and fc.ok:
            rows.append({"event_id": row["event_id"], "incid_name": row["incid_name"],
                         "region": row["region"], "table": table, "pdsim": fc.pdsim,
                         "n_selected": fc.n_selected, "ig_year": row["ig_year"],
                         "km2": row["km2"], "map_prog": row.get("map_prog"),
                         "threshold_scale": row.get("threshold_scale")})
cal = pd.DataFrame(rows)
nv = cal[cal["table"] == "nv_statewide_disp"]
missing = sorted(set(use["event_id"]) - set(nv["event_id"]))
print(f"calibrated fires: {nv['event_id'].nunique()}/{len(use)} ({len(missing)} missing)")
if missing:
    raise SystemExit(f"not all fires calibrated: {missing[:6]} ...")

summary = nv.groupby("region")["pdsim"].agg(["median", "count",
                                             lambda s: s.quantile(.25),
                                             lambda s: s.quantile(.75)])
summary.columns = ["pdsim", "n", "q25", "q75"]
st = cal[cal["table"] == "staley2018_disp"].groupby("region")["pdsim"].median()


def slug(r):
    return r.lower().replace(" ", "_").replace("&", "and")


print("\nv1_3 -> v1_4:")
for region in sorted(summary.index):
    old = PREV["regions"][slug(region)]
    print(f"  {region:26s} break {old['barc_breaks'][1]:>6} -> {breaks[region]:<6.1f} "
          f"pdsim {old['pdsim']:.2f} -> {summary.loc[region, 'pdsim']:.2f}   (n={int(summary.loc[region, 'n'])})")

lines = [
    "# Nevada statewide per-region P_dsim calibration (v1.4).",
    "# v1_3 with the BARC256 threshold scale fixed. Three 2024 fires whose",
    "# bundles are BAER products (Davis, Bear, Broom Canyon) carried GTAC's",
    "# 0-255 BARC256 thresholds into the fire set as if they were dNBR",
    "# (158/120/111 -> 515/325/280 after dNBR = 5*B - 275). Two of them sat in",
    "# the seven-fire Sierra Nevada break median: 312 -> 350. Also, fires whose",
    "# bundle has no dnbr6 -- those three, Stockade Canyon (RAVG) and Earthstone",
    "# (an MTBS bundle delivered without one) -- now build their observed",
    "# classes from the analyst's own thresholds, as dnbr6 does for every other",
    "# fire, instead of pfdf's fixed 125/250/500; their five decompositions were",
    "# rebuilt (v1_3 caches kept in interim/calib/_superseded_v1_3/). The other",
    "# 116 caches are carried unchanged; the seven Sierra fires re-solved at the",
    "# new break. P_dsim moved 0.47 -> 0.48 in Sierra Nevada only: raising the",
    "# break cools simulated severity, so the untouched Sierra fires rose",
    "# 0.01-0.02 while Davis fell 0.78 -> 0.74. Fire-set rule otherwise as v1_3.",
    "",
    "[meta]",
    'name = "statewide_v1_4"',
    f"created = {date.today().isoformat()}",
    'method = "rossi2025-dfl-calibration (logit-space, per-region, DISPERSED severity sigma=0.91, sentinel-guarded + BARC256-corrected breaks)"',
    'cdf_source = "nv_statewide (Staley 2018 + Nevada era-matched refit)"',
    'evt = "era-matched (LF2016/LF2022/LF2023)"',
    f"fires_used = {nv['event_id'].nunique()}",
    'supersedes = "statewide_v1_3"',
]
for region in sorted(summary.index):
    s = summary.loc[region]
    fires = nv.loc[nv["region"] == region, "event_id"]
    conv = int(n_conv.get(region, 0))
    lines += [
        "",
        f"[regions.{slug(region)}]",
        f"pdsim = {s['pdsim']:.2f}",
        f"barc_breaks = [125.0, {breaks[region]:g}, 500.0]",
        f"# n={int(s['n'])} ({int(n_thr[region])} with analyst mod_t, {conv} BARC256-converted), "
        f"IQR=[{s['q25']:.2f}, {s['q75']:.2f}], staley_control={st.get(region, float('nan')):.2f}",
        "fires = [" + ", ".join(f'"{e}"' for e in fires) + "]",
    ]
with open(OUT_TOML, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nwrote {OUT_TOML}")
out_csv = ("/Users/scottmccoy/Library/CloudStorage/Box-Box/SWMresearch/PostFireDebrisFlows/"
           "PreFireAssessment/products/calibration/statewide_v1_4_fires.csv")
cal.to_csv(out_csv, index=False)
print(f"wrote {out_csv}")
