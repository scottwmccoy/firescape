"""How much does the official (BAER) soil burn severity map differ from the
map we're actually USING -- MTBS dNBR classified at our calibrated regional
BARC break?

Restricted to the ``statewide_v1`` calibration fires with a clean single-tile
BAER SBS match (>=90% of the fire perimeter covered by one BAER raster,
found via ``firescape.baer.match_perimeters``). MTBS dNBR bundles for these
fires are already staged locally from calibration; nothing is re-downloaded
except the BAER rasters themselves, pulled aligned to each fire's own dNBR
grid so no separate resampling step is needed.

Two questions, both from the SAME per-fire pixel comparison:

1. **Where does BAER draw its low/moderate and moderate/high line, on OUR
   dNBR raster?** Not read from BAER metadata (it doesn't publish one) --
   empirically, as the dNBR value that best separates BAER's classes
   (Youden's J-optimal cutoff, the standard ROC threshold statistic).
   Comparable directly to ``our_break`` (the regional calibrated value) and
   to this fire's own MTBS analyst threshold (``mod_t``, already in the fire
   set CSV).
2. **How do the class fractions compare?** BAER's four published classes
   vs. ours (MTBS dNBR at ``our_break``), over the same jointly-valid pixels.
   Because BARC4 and BAER's SBS use the SAME 1..4 numbering
   (unburned/low/moderate/high), the two class arrays need no remap.

    python scripts/analysis/p7_baer_comparison.py [calibration]
"""
import glob
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import tomllib

from firescape import baer, paths, severity

CAL = sys.argv[1] if len(sys.argv) > 1 else "statewide_v1_3"
FIRE_SET = "statewide_v1"          # the fire_sets/*.csv the calibration was built from
LABELS = ("unburned", "low", "moderate", "high")

# Calibration TOMLs ship inside the package (firescape/data/calibration/),
# not under the Box data root.
_PKG = pathlib.Path(paths.__file__).resolve().parent
with open(_PKG / "data" / "calibration" / f"{CAL}.toml", "rb") as fh:
    cal = tomllib.load(fh)
REGION_BREAK = {name: cfg["barc_breaks"] for name, cfg in cal["regions"].items()}
# TOML region keys are snake_case; the fire-set CSV's `region` column is the
# human-readable EPA level-3 name -- map explicitly rather than guess a
# .title()-based slug (it mangles "and").
SLUG_TO_NAME = {
    "central_basin_and_range": "Central Basin and Range",
    "mojave_basin_and_range": "Mojave Basin and Range",
    "northern_basin_and_range": "Northern Basin and Range",
    "sierra_nevada": "Sierra Nevada",
}
REGION_BREAK = {SLUG_TO_NAME[k]: v for k, v in REGION_BREAK.items() if k in SLUG_TO_NAME}

fs = pd.read_csv(_PKG / "data" / "calibration" / "fire_sets" / f"{FIRE_SET}.csv")
fs = fs[fs["include"] == True].copy()
print(f"{len(fs)} fires in {FIRE_SET} (include=True)", flush=True)

perim_rows = []
for i, row in fs.iterrows():
    shp = sorted(glob.glob(str(paths.raw_dir("mtbs", "fires", row["event_id"]) / "*_burn_area.shp")))
    if not shp:
        continue
    g = gpd.read_file(shp[0]).to_crs("EPSG:4326")
    perim_rows.append({"event_id": row["event_id"], "geometry": g.geometry.union_all()})
    if len(perim_rows) % 20 == 0:
        print(f"  {len(perim_rows)} perimeters loaded", flush=True)
perims = gpd.GeoDataFrame(perim_rows, geometry="geometry", crs="EPSG:4326").set_index("event_id")
print(f"{len(perims)} perimeters loaded locally", flush=True)

# Year-aligned (see baer.match_perimeters): spatial overlap alone matched 29
# of 140 CA fires to another fire's assessment. Nevada happens to be clean --
# all 19 matches are same-year -- but the join has to be correct by rule, not
# by luck.
_years = fs.set_index("event_id")["ig_year"].reindex(perims.index)
match = baer.match_perimeters(perims, min_frac=0.90, fire_years=_years)
match = match.join(fs.set_index("event_id"), how="left")
print(f"{len(match)} clean BAER SBS matches "
      f"({match.index.to_series().str[:2].value_counts().to_dict()})", flush=True)


rows, class_rows = [], []
for event_id, f in match.iterrows():
    dnbr_path = sorted(glob.glob(str(paths.raw_dir("mtbs", "fires", event_id) / "*_dnbr.tif")))[0]
    with rasterio.open(dnbr_path) as ds:
        dnbr = ds.read(1).astype("float64")
        nodata, crs_epsg, bounds = ds.nodata, ds.crs.to_epsg(), ds.bounds

    # Locked to the tile match_perimeters chose: the mosaic otherwise
    # composites overlapping assessments by its own rule and can hand back a
    # different fire's map (see firescape.baer module docstring).
    baer_cls = baer.fetch_aligned((bounds.left, bounds.bottom, bounds.right, bounds.top),
                                  dnbr.shape, crs_epsg,
                                  lock_raster_id=f.get("baer_oid"))
    valid = severity.valid_dnbr(dnbr, nodata) & (baer_cls >= 1) & (baer_cls <= 4)
    if valid.sum() < 200:
        print(f"{f['incid_name']:16s} SKIP -- only {int(valid.sum())} jointly-valid px", flush=True)
        continue
    d_v, b_v = dnbr[valid], baer_cls[valid].astype(np.uint8)

    breaks = REGION_BREAK[f["region"]]
    our = severity.classify_barc4(d_v, breaks)
    t_lowmod, j_lowmod = severity.youden_threshold(d_v, b_v >= 3)
    t_modhigh, j_modhigh = severity.youden_threshold(d_v, b_v >= 4)

    rows.append({
        "event_id": event_id, "incid_name": f["incid_name"], "region": f["region"],
        "ig_year": f["ig_year"], "mtbs_km2": f.get("km2"),
        "analyst_mod_t": f.get("mod_t"), "baer_coverage_frac": f["coverage_frac"],
        "n_valid_px": int(valid.sum()), "our_break": breaks[1],
        "baer_implied_lowmod": t_lowmod, "youdenJ_lowmod": j_lowmod,
        "baer_implied_modhigh": t_modhigh, "youdenJ_modhigh": j_modhigh,
        "class_agree_4": float((our == b_v).mean()),
        "kappa_4": severity.confusion_kappa(our, b_v),
        "modhi_agree": float(((our >= 3) == (b_v >= 3)).mean()),
    })
    for lab, code in zip(LABELS, (1, 2, 3, 4)):
        class_rows.append({"event_id": event_id, "incid_name": f["incid_name"],
                           "region": f["region"], "class": lab,
                           "baer_frac": float((b_v == code).mean()),
                           "our_frac": float((our == code).mean())})
    print(f"{f['incid_name']:16s} {f['region'][:22]:22s} n={int(valid.sum()):8,d}  "
          f"our={breaks[1]:6.1f}  BAER-lowmod={t_lowmod:6.1f} (J={j_lowmod:.2f})  "
          f"modhi_agree={rows[-1]['modhi_agree']:.2f} kappa4={rows[-1]['kappa_4']:.2f}", flush=True)

res = pd.DataFrame(rows)
cls = pd.DataFrame(class_rows)
OUT = paths.products_dir("calibration")
res.to_csv(OUT / f"baer_vs_mtbs_{CAL}.csv", index=False)
cls.to_csv(OUT / f"baer_vs_mtbs_{CAL}_class_fractions.csv", index=False)

modhi = (cls[cls["class"].isin(["moderate", "high"])]
         .groupby(["event_id", "incid_name", "region"])[["baer_frac", "our_frac"]]
         .sum().reset_index())
summary = {
    "calibration": CAL, "fire_set": FIRE_SET, "n_fires": int(len(res)),
    "baer_implied_lowmod_break": {"median": float(res["baer_implied_lowmod"].median()),
                                  "iqr": [float(res["baer_implied_lowmod"].quantile(.25)),
                                          float(res["baer_implied_lowmod"].quantile(.75))],
                                  "range": [float(res["baer_implied_lowmod"].min()),
                                            float(res["baer_implied_lowmod"].max())]},
    "our_break_minus_baer_implied": {"median": float((res["our_break"] - res["baer_implied_lowmod"]).median()),
                                     "std": float((res["our_break"] - res["baer_implied_lowmod"]).std())},
    "median_youdenJ_lowmod": float(res["youdenJ_lowmod"].median()),
    "median_kappa_4class": float(res["kappa_4"].median()),
    "median_modhigh_binary_agreement": float(res["modhi_agree"].median()),
    "pct_moderate_or_high": {"baer_median": float(modhi["baer_frac"].median()),
                             "ours_median": float(modhi["our_frac"].median()),
                             "diff_ours_minus_baer_median": float((modhi["our_frac"] - modhi["baer_frac"]).median())},
}
import json
(OUT / f"baer_vs_mtbs_{CAL}_summary.json").write_text(json.dumps(summary, indent=2))
print(f"\n{len(res)} fires compared -> {OUT}", flush=True)
print(json.dumps(summary, indent=2))
