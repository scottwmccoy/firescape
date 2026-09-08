"""Summarize the era-matched recalibration and write the pilot_v2 TOML."""
import json
import warnings
from datetime import date

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from firescape import paths

TAG = "lf2016"
BREAK = 281.0
fires = pd.read_csv(paths.package_data("calibration", "fire_sets", "pilot_v1.csv")).set_index("event_id")

rows = []
for d in sorted(paths.interim_dir("calib").iterdir()):
    if not d.name.endswith(f"__{TAG}") or not (d / "fire.json").exists():
        continue
    m = json.loads((d / "fire.json").read_text())
    by = m.get("pdsim_by_table", {})
    eid = m["event_id"]
    if by.get("nv_merged") is None or by.get("staley2018") is None:
        print(f"  {eid}: incomplete ({by})")
        continue
    f = fires.loc[eid]
    rows.append({"event_id": eid, "name": f["incid_name"], "year": int(f["ig_year"]),
                 "km2": f["km2"], "n_sel": m["n_selected"],
                 "staley2018": by["staley2018"], "nv_merged": by["nv_merged"]})
tab = pd.DataFrame(rows).sort_values("km2", ascending=False)
print(f"{len(tab)} era-matched fires\n")
print(tab[["name", "year", "km2", "n_sel", "staley2018", "nv_merged"]].to_string(index=False))

out = {}
for col in ("staley2018", "nv_merged"):
    v = tab[col].to_numpy(dtype=float)
    out[col] = {"median": float(np.median(v)),
                "iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))],
                "min": float(v.min()), "max": float(v.max()), "n_fires": len(v)}
    print(f"\n{col:>12}: regional P_dsim = {out[col]['median']:.2f} "
          f"(IQR {out[col]['iqr'][0]:.2f}-{out[col]['iqr'][1]:.2f}, "
          f"range {v.min():.2f}-{v.max():.2f})")

# spread comparison: a table that describes local fuels better should need
# less fire-to-fire adjustment of the single free parameter
for col in ("staley2018", "nv_merged"):
    v = tab[col].to_numpy(dtype=float)
    out[col]["iqr_width"] = float(np.percentile(v, 75) - np.percentile(v, 25))
    out[col]["std"] = float(v.std(ddof=1))
print(f"\nfire-to-fire spread in P_dsim (lower = table needs less per-fire adjustment):")
for col in ("staley2018", "nv_merged"):
    print(f"  {col:>12}: IQR width {out[col]['iqr_width']:.3f}, sd {out[col]['std']:.3f}")

C = paths.products_dir("calibration")
tab.to_csv(C / "pilot_v2_fire_pdsim.csv", index=False)
(C / "pilot_v2_summary.json").write_text(json.dumps(
    {"regional": out, "break_lowmod": BREAK, "evt": "LF2016 (era-matched, fires >= 2017)",
     "note": "break held at the pilot_v1 34-fire median so the CDF table is the "
             "only variable; P_dsim solved for both tables in one pass per fire"},
    indent=2))

pdsim = out["nv_merged"]["median"]
toml = C / "pilot_v2.toml"
toml.write_text("\n".join([
    "# Nevada pilot calibration v2 - recalibrated against the nv_merged CDF table.",
    "# Era-matched: LANDFIRE LF2016 vegetation with fires that ignited 2017+, so",
    "# the vegetation predates each fire (post-fire EVT would leak severity).",
    "# The low/moderate break is held at the pilot_v1 34-fire median (281) so the",
    "# CDF table is the only thing that changed between v1 and v2.",
    "",
    "[meta]",
    'name = "pilot_v2"',
    f"created = {date.today().isoformat()}",
    'method = "rossi2025-dfl-calibration (logit-space solve)"',
    'cdf_source = "nv_merged (Staley 2018 + Nevada era-matched refit)"',
    'evt = "LF2016"',
    f"fires_used = {len(tab)}",
    f'staley_control_pdsim = {out["staley2018"]["median"]:.2f}',
    "",
    "[regions.pilot]",
    f"pdsim = {pdsim:.2f}",
    f"barc_breaks = [125.0, {BREAK:.0f}, 500.0]",
    "fires = [" + ", ".join(f'"{e}"' for e in tab["event_id"]) + "]",
]) + "\n")
print(f"\nwrote {toml}")
