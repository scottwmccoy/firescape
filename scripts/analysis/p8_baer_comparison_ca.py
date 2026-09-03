"""CA companion to ``p7_baer_comparison.py``: how much does the official BAER
soil burn severity map differ from MTBS dNBR classified at ROSSI ET AL.
(2025)'s own calibrated regional break -- the CA analogue of "the map we're
using," since firescape has no independent CA calibration of its own.

Fire set: ``firescape/data/calibration/fire_sets/ca_baer_candidates.csv``,
built 2026-09-03 by (1) querying the MTBS WFS for CA fires >=~2000 acres,
ig_year 2017+ (BAER SBS's own coverage start), (2) matching each against the
national BAER SBS mosaic (``firescape.baer.match_perimeters``, >=90% of the
fire perimeter covered by one BAER raster), (3) assigning each fire's centroid
into one of Rossi's 8 ``ca_prefire_regions`` (Zenodo 10.5281/zenodo.15313560)
via the SAME within-region-else-nearest rule ``scripts/stage/p2_regions.py``
uses for NV. 140 fires matched, spread across all 8 regions (thinnest: Central
Coast Ranges, n=2).

Needs the fires' MTBS dNBR bundles in ``raw/mtbs/fires/<event_id>/`` -- ordered
via ``firescape.mtbs.order_bundles()`` on 2026-09-03 (140 fires, links emailed
to the project inbox, ~1 h turnaround); download the emailed bsp_*.zip into
raw/mtbs/ and extract into raw/mtbs/fires/<EVENT_ID>/ (same layout the NV
bundles already have) before running this.

The delivered bundle is NOT all MTBS -- the portal serves whatever exists
per fire, and this order came back 115 mtbs / 20 baer / 3 ravg / 2
provisional. That is tracked per fire (``dnbr_programme``) and broken out in
the summary, because the 20 BAER ones are a different and cleaner experiment:
BAER's SBS against a break on BAER's OWN dNBR isolates the field adjustment,
same sensor and pass, while against MTBS's dNBR it is that plus a different
pass, date and processing chain.

Otherwise identical methodology to p7: Youden's-J dNBR threshold that best
reproduces BAER's low/moderate and moderate/high split, BARC4 class-fraction
comparison over the same jointly-valid pixels (BAER's 1..4 codes match
``classify_barc4``'s numbering, no remap needed).

    python scripts/analysis/p8_baer_comparison_ca.py
"""
import glob
import json
import pathlib
import re
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import rasterio

from firescape import baer, paths, severity

LABELS = ("unburned", "low", "moderate", "high")
_PKG = pathlib.Path(paths.__file__).resolve().parent


def dnbr_programme(path):
    """Which programme produced this fire's continuous dNBR.

    Material to what the comparison means, not bookkeeping. The portal serves
    whatever exists for a fire, and for 20 of these 140 that is BAER's own
    dNBR -- the BARC the field team started from. Comparing BAER's SBS
    against a break applied to BAER's OWN dNBR isolates the field
    adjustment: same sensor, same pass, the only difference is the ground
    visit. Against MTBS's dNBR it is that PLUS a different sensor pass,
    date and processing chain. Pooling the two silently would blur the
    cleanest sub-experiment in the set into the noisier one.
    """
    m = re.match(r"(mtbs|baer|ravg|provisional)_", pathlib.Path(path).name)
    return m.group(1).upper() if m else "unknown"


fs = pd.read_csv(_PKG / "data" / "calibration" / "fire_sets" / "ca_baer_candidates.csv")
print(f"{len(fs)} candidate CA fires ({fs['rossi_region'].nunique()} Rossi regions)", flush=True)

have_bundle, missing = [], []
for _, row in fs.iterrows():
    d = paths.raw_dir("mtbs", "fires", row["event_id"])
    if sorted(glob.glob(str(d / "*_dnbr.tif"))):
        have_bundle.append(row["event_id"])
    else:
        missing.append(row["event_id"])
print(f"{len(have_bundle)}/{len(fs)} have a local dNBR bundle already "
      f"({len(missing)} missing -- see module docstring for the addQueue order)",
      flush=True)
if not have_bundle:
    raise SystemExit(
        "No CA dNBR bundles staged yet. Order status: check the project inbox "
        "for the addQueue.php email (placed 2026-09-03, ~1 h turnaround), "
        "download the bsp_*.zip, and extract into raw/mtbs/fires/<EVENT_ID>/ "
        "(matching the NV bundle layout) before re-running this script.")

fs = fs.set_index("event_id").loc[have_bundle].reset_index()

rows, class_rows = [], []
for _, f in fs.iterrows():
    event_id = f["event_id"]
    dnbr_path = sorted(glob.glob(str(paths.raw_dir("mtbs", "fires", event_id) / "*_dnbr.tif")))[0]
    with rasterio.open(dnbr_path) as ds:
        dnbr = ds.read(1).astype("float64")
        nodata, crs_epsg, bounds = ds.nodata, ds.crs.to_epsg(), ds.bounds

    # Locked to the matched tile -- see firescape.baer module docstring.
    baer_cls = baer.fetch_aligned((bounds.left, bounds.bottom, bounds.right, bounds.top),
                                  dnbr.shape, crs_epsg,
                                  lock_raster_id=f.get("baer_oid"))
    valid = severity.valid_dnbr(dnbr, nodata) & (baer_cls >= 1) & (baer_cls <= 4)
    if valid.sum() < 200:
        print(f"{f['incid_name']:20s} SKIP -- only {int(valid.sum())} jointly-valid px", flush=True)
        continue
    d_v, b_v = dnbr[valid], baer_cls[valid].astype(np.uint8)

    prog = dnbr_programme(dnbr_path)
    rossi_break = (125.0, float(f["rossi_break"]), 500.0)
    our = severity.classify_barc4(d_v, rossi_break)
    t_lowmod, j_lowmod = severity.youden_threshold(d_v, b_v >= 3)
    t_modhigh, j_modhigh = severity.youden_threshold(d_v, b_v >= 4)

    rows.append({
        "event_id": event_id, "incid_name": f["incid_name"], "rossi_region": f["rossi_region"],
        "dnbr_programme": prog,
        "ig_year": f["ig_year"], "mtbs_km2": f["mtbs_km2"],
        "analyst_mod_t": f["mod_t"], "baer_coverage_frac": f["coverage_frac"],
        "n_valid_px": int(valid.sum()), "rossi_break": rossi_break[1],
        "baer_implied_lowmod": t_lowmod, "youdenJ_lowmod": j_lowmod,
        "baer_implied_modhigh": t_modhigh, "youdenJ_modhigh": j_modhigh,
        "class_agree_4": float((our == b_v).mean()),
        "kappa_4": severity.confusion_kappa(our, b_v),
        "modhi_agree": float(((our >= 3) == (b_v >= 3)).mean()),
    })
    for lab, code in zip(LABELS, (1, 2, 3, 4)):
        class_rows.append({"event_id": event_id, "incid_name": f["incid_name"],
                           "rossi_region": f["rossi_region"], "class": lab,
                           "baer_frac": float((b_v == code).mean()),
                           "our_frac": float((our == code).mean())})
    print(f"{f['incid_name']:20s} {prog:5s} {f['rossi_region'][:22]:22s} n={int(valid.sum()):8,d}  "
          f"rossi={rossi_break[1]:6.1f}  BAER-lowmod={t_lowmod:6.1f} (J={j_lowmod:.2f})  "
          f"modhi_agree={rows[-1]['modhi_agree']:.2f} kappa4={rows[-1]['kappa_4']:.2f}", flush=True)

res = pd.DataFrame(rows)
cls = pd.DataFrame(class_rows)
OUT = paths.products_dir("calibration")
res.to_csv(OUT / "baer_vs_mtbs_rossi_ca.csv", index=False)
cls.to_csv(OUT / "baer_vs_mtbs_rossi_ca_class_fractions.csv", index=False)

modhi = (cls[cls["class"].isin(["moderate", "high"])]
         .groupby(["event_id", "incid_name", "rossi_region"])[["baer_frac", "our_frac"]]
         .sum().reset_index())
summary = {
    "source": "Rossi et al. 2025 ca_prefire_regions (zenodo 10.5281/zenodo.15313560)",
    "n_fires": int(len(res)),
    "n_regions": int(res["rossi_region"].nunique()),
    "baer_implied_lowmod_break": {"median": float(res["baer_implied_lowmod"].median()),
                                  "iqr": [float(res["baer_implied_lowmod"].quantile(.25)),
                                          float(res["baer_implied_lowmod"].quantile(.75))],
                                  "range": [float(res["baer_implied_lowmod"].min()),
                                            float(res["baer_implied_lowmod"].max())]},
    "rossi_break_minus_baer_implied": {"median": float((res["rossi_break"] - res["baer_implied_lowmod"]).median()),
                                       "std": float((res["rossi_break"] - res["baer_implied_lowmod"]).std())},
    "median_youdenJ_lowmod": float(res["youdenJ_lowmod"].median()),
    "median_kappa_4class": float(res["kappa_4"].median()),
    "median_modhigh_binary_agreement": float(res["modhi_agree"].median()),
    "pct_moderate_or_high": {"baer_median": float(modhi["baer_frac"].median()),
                             "ours_median": float(modhi["our_frac"].median()),
                             "diff_ours_minus_baer_median": float((modhi["our_frac"] - modhi["baer_frac"]).median())},
    "by_dnbr_programme": {
        prog: {"n": int(len(g)),
               "baer_implied_median": float(g["baer_implied_lowmod"].median()),
               "rossi_minus_implied_median": float((g["rossi_break"] - g["baer_implied_lowmod"]).median()),
               "modhi_agree_median": float(g["modhi_agree"].median()),
               "kappa_median": float(g["kappa_4"].median())}
        for prog, g in res.groupby("dnbr_programme")
    },
    "by_region": {
        region: {
            "n": int(len(g)),
            "rossi_break": float(g["rossi_break"].iloc[0]),
            "baer_implied_median": float(g["baer_implied_lowmod"].median()),
            "modhi_agree_median": float(g["modhi_agree"].median()),
            "kappa_median": float(g["kappa_4"].median()),
        }
        for region, g in res.groupby("rossi_region")
    },
}
(OUT / "baer_vs_mtbs_rossi_ca_summary.json").write_text(json.dumps(summary, indent=2))
print(f"\n{len(res)} fires compared -> {OUT}", flush=True)
print(json.dumps(summary, indent=2))
