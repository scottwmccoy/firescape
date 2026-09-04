"""Patch the existing statewide_v1 fire set for the BARC256 threshold scale.

Membership and every other column are left exactly as staged on 2026-08-13;
this only (1) fetches ``map_prog`` / ``asmnt_type`` for the set's own
event_ids from the MTBS WFS, (2) runs ``mtbs.normalize_thresholds`` so BAER
records with BARC256-scale thresholds land on the dNBR scale, and (3) writes
the CSV back with the three new columns and a ``region`` column preserved.
Prints every row it changed. Re-running is a no-op.

    python scripts/stage/p2b_fireset_barc_scale.py
"""
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from firescape import mtbs

SET = ("/Users/scottmccoy/git/code/firescape/firescape/data/calibration/"
       "fire_sets/statewide_v1.csv")

df = pd.read_csv(SET)
if "threshold_scale" in df.columns and (df["threshold_scale"] == "barc256->dnbr").any():
    raise SystemExit("already patched -- nothing to do")

ids = sorted(df["event_id"])
recs = []
for i in range(0, len(ids), 40):
    g = mtbs.fire_records(event_id_like=None, event_ids=ids[i:i + 40], min_acres=None)
    recs.append(g.drop(columns="geometry"))
recs = mtbs.dedupe_records(pd.concat(recs, ignore_index=True))   # keeps the row with thresholds
recs = recs.set_index("event_id")
missing = [e for e in ids if e not in recs.index]
print(f"WFS records: {len(recs)} of {len(ids)} ({len(missing)} missing: {missing[:5]})")

df["map_prog"] = df["event_id"].map(recs["map_prog"])
df["asmnt_type"] = df["event_id"].map(recs["asmnt_type"])
# sanity: the thresholds the WFS serves now must be what the set was staged with
chk = df.join(recs[["low_t", "mod_t", "high_t"]], on="event_id", rsuffix="_wfs")
drift = chk[(chk["mod_t"].fillna(-1) != chk["mod_t_wfs"].fillna(-1))]
if len(drift):
    print("WARNING -- WFS thresholds differ from the staged set for:")
    print(drift[["event_id", "incid_name", "mod_t", "mod_t_wfs", "map_prog"]].to_string(index=False))

before = df[["low_t", "mod_t", "high_t"]].copy()
df = mtbs.normalize_thresholds(df)
changed = df[df["threshold_scale"] == "barc256->dnbr"]
print(f"\nconverted {len(changed)} BAER-programme rows from BARC256 to dNBR:")
for i, r in changed.iterrows():
    b = before.loc[i]
    print(f"  {r['incid_name']:16s} {r['region']:26s} {b['low_t']:.0f}/{b['mod_t']:.0f}/{b['high_t']:.0f}"
          f"  ->  {r['low_t']:.0f}/{r['mod_t']:.0f}/{r['high_t']:.0f}")
print("\nby programme:", df["map_prog"].value_counts(dropna=False).to_dict())

cols = ["event_id", "incid_name", "ig_date", "ig_year", "incid_type", "km2",
        "map_prog", "asmnt_type", "low_t", "mod_t", "high_t", "threshold_scale",
        "dnbr_offst", "evt_vintage", "have_bundle", "thresholds_ok", "include",
        "lon", "lat", "region"]
df[cols].to_csv(SET, index=False)
print(f"\nwrote {SET}")
use = df[df["include"] & (df["mod_t"] > 0) & (df["mod_t"] < 2000)]
print("\nregional low/moderate break = median analyst mod_t, now:")
print(use.groupby("region")["mod_t"].agg(["median", "count"]).to_string())
