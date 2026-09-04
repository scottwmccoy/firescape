"""Why does the BAER-implied dNBR break scatter so hard around our single
regional number?  Diagnostics for the outliers in ``p7`` (NV) and ``p8`` (CA).

Everything here is local -- no BAER re-fetch.  For each of the 124 distinct
fires already compared it computes:

* the fire's own dNBR distribution inside its MTBS/BAER perimeter, and the
  PERCENTILE RANK at which the BAER-implied break and our fixed regional
  break land in it.  This is the whole argument: a field crew splits what is
  in front of them, so BAER's line sits near the middle of whatever
  distribution that fire produced (median 56th percentile, range 19-83),
  while a fixed number lands wherever it lands (9th to 99th).
* the AREA-matched break -- the dNBR value at which our classification would
  call exactly as much of the fire moderate-or-high as BAER did.  Youden's J
  finds the ROC-optimal separator, which is NOT the prevalence quantile when
  the classes overlap heavily; the area version is the one comparable to
  what the two maps look like.
* assessment type and the pre/post image dates, parsed from the bundle's ISO
  metadata (BAER bundles carry the dates in the filename instead), turned into
  TWO different delays that must not be confused:
  ``days_post`` = ignition -> post-fire image (2-462 d here; this is "how long
  after the fire was it looked at"), and ``pre_lead`` = pre-fire image ->
  ignition (up to 855 d; how stale the baseline is).  The pre-to-post SPAN is
  the sum and is NOT a post-fire delay -- reading it as one invents a
  "two-to-three-seasons-later" group that does not exist in this set.
  On the correct variable the timing effect is real but modest: fires imaged
  within 60 days run a median 10 pp hotter than BAER, against 1 pp for the
  rest (Mann-Whitney p=0.009), and 13 of the 14 large under-calls are on
  delayed imagery -- but large OVER-calls split 8/8 across the two groups.
* pre-fire LANDFIRE LF2016 lifeform fractions, for the fires the staged NV
  EVT tiles cover.  LF2016 because every fire here ignited 2017+, so it is
  pre-fire for all of them.  Result is a NEGATIVE one worth keeping: tree
  fraction has no relationship with the implied break (r = -0.01), only with
  the dNBR ceiling (p90, r = +0.50).

    python scripts/analysis/p9_baer_outlier_diagnostics.py
"""
import glob
import pathlib
import re
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.windows
from rasterio.warp import transform_bounds
from shapely.geometry import box

from firescape import paths, severity

CAL = paths.products_dir("calibration")
EVT_SET = "LF2016_EVT_nv"           # pre-fire for every 2017+ fire in the set


def _fire_table() -> pd.DataFrame:
    """The p7 (NV) and p8 (CA) result tables, stacked on common column names.

    Seven fires are in BOTH sets -- Slinkard, Boot, Mountain View, Tamarack,
    Loyalton, Caldor and Sugar are CA-located but calibrate Nevada regions.
    They are kept here (each carries its own regional break) and deduplicated
    by ``event_id`` wherever a per-fire statistic is pooled.
    """
    nv = pd.read_csv(CAL / "baer_vs_mtbs_statewide_v1_3.csv").rename(
        columns={"region": "region_name"})
    nv["state"], nv["dnbr_programme"] = "NV", np.nan
    ca = pd.read_csv(CAL / "baer_vs_mtbs_rossi_ca.csv").rename(
        columns={"rossi_region": "region_name", "rossi_break": "our_break"})
    ca["state"] = "CA"
    return pd.concat([nv, ca], ignore_index=True)


def _meta(bundle: pathlib.Path) -> dict:
    out = {"assess_type": None, "pre_date": None, "post_date": None}
    xmls = list(bundle.glob("*metadata.xml"))
    if xmls:
        t = max(xmls, key=lambda p: p.stat().st_size).read_text(errors="ignore")
        m = re.search(r"Type of Assessment:\s*([A-Za-z ]+?)<", t)
        if m:
            out["assess_type"] = m.group(1).strip()
        for tag, key in (("Pre-Fire", "pre"), ("Post-Fire", "post")):
            m = re.search(tag + r" Sensor, Date, Scene ID:\s*[^,]+,\s*([0-9\-]+)", t)
            if m:
                out[f"{key}_date"] = m.group(1).strip()
    if out["post_date"] is None:                      # BAER bundles name their dates
        m = re.search(r"_(\d{8})_(\d{8})_dnbr", sorted(bundle.glob("*_dnbr.tif"))[0].name)
        if m:
            for key, g in (("pre", 1), ("post", 2)):
                d = m.group(g)
                out[f"{key}_date"] = f"{d[:4]}-{d[4:6]}-{d[6:]}"
    return out


def _perimeter_dnbr(bundle: pathlib.Path):
    """Valid dNBR inside the fire perimeter, plus the open dataset's grid."""
    tif = sorted(glob.glob(str(bundle / "*_dnbr.tif")))[0]
    with rasterio.open(tif) as ds:
        dnbr = ds.read(1).astype("float64")
        ok = severity.valid_dnbr(dnbr, ds.nodata)
        shp = sorted(bundle.glob("*_burn_area.shp"))
        if shp:
            per = gpd.read_file(shp[0]).to_crs(ds.crs)
            ok &= rasterio.features.geometry_mask(per.geometry, dnbr.shape,
                                                  ds.transform, invert=True)
        return dnbr[ok], (per if shp else None)


def _evt_tiles():
    tiles = []
    for t in sorted(paths.raw_dir("landfire", EVT_SET).glob("evt_*/*.tif")):
        with rasterio.open(t) as ds:
            tiles.append({"path": t, "name": t.parent.name,
                          "geometry": box(*transform_bounds(ds.crs, "EPSG:4326", *ds.bounds))})
    return gpd.GeoDataFrame(tiles, crs="EPSG:4326")


def main():
    fires = _fire_table()
    print(f"{len(fires)} rows ({fires.event_id.nunique()} distinct fires)", flush=True)

    # Ignition dates, for the only delay that means "how long after the fire".
    fsets = pathlib.Path(paths.__file__).parent / "data" / "calibration" / "fire_sets"
    ig = pd.concat([pd.read_csv(fsets / "statewide_v1.csv")[["event_id", "ig_date"]],
                    pd.read_csv(fsets / "ca_baer_candidates.csv")[["event_id", "ig_date"]]])
    ig = ig.drop_duplicates("event_id")

    # BAER's own moderate+ area fraction, from the class-fraction tables.
    cf = pd.concat([pd.read_csv(CAL / "baer_vs_mtbs_statewide_v1_3_class_fractions.csv"),
                    pd.read_csv(CAL / "baer_vs_mtbs_rossi_ca_class_fractions.csv")])
    cf = cf.drop_duplicates(subset=["event_id", "class"])   # the seven shared fires
    modhi = (cf[cf["class"].isin(["moderate", "high"])]
             .groupby("event_id")[["baer_frac", "our_frac"]].sum())
    assert modhi.max().max() <= 1 + 1e-9, "class fractions doubled -- check the dedup"

    tiles = _evt_tiles()
    rows = []
    for _, f in fires.iterrows():
        bundle = paths.raw_dir("mtbs", "fires", f["event_id"])
        if not sorted(glob.glob(str(bundle / "*_dnbr.tif"))):
            print(f"  no bundle: {f['incid_name']}", flush=True)
            continue
        d, per = _perimeter_dnbr(bundle)
        if d.size < 200:
            continue
        ours, implied = float(f["our_break"]), float(f["baer_implied_lowmod"])
        b = float(modhi.loc[f["event_id"], "baer_frac"]) if f["event_id"] in modhi.index else np.nan

        r = {"event_id": f["event_id"], "incid_name": f["incid_name"], "state": f["state"],
             "region_name": f["region_name"], "ig_year": f["ig_year"], "km2": f["mtbs_km2"],
             "programme": f["dnbr_programme"], "our_break": ours, "implied": implied,
             "implied_minus_ours": implied - ours, "kappa_4": f["kappa_4"],
             "modhi_agree": f["modhi_agree"], "youdenJ": f["youdenJ_lowmod"],
             "analyst_mod_t": f["analyst_mod_t"], "n_px": int(d.size),
             "dnbr_mean": float(d.mean()),
             "pct_rank_implied": float((d < implied).mean() * 100),
             "pct_rank_ours": float((d < ours).mean() * 100),
             "baer_modhi": b,
             "our_modhi": float(modhi.loc[f["event_id"], "our_frac"]) if b == b else np.nan,
             # the dNBR value that reproduces BAER's moderate+ AREA (not its ROC point)
             "implied_area": float(np.quantile(d, 1 - b)) if 0 < (b if b == b else -1) < 1 else np.nan}
        r["modhi_gap_ours_minus_baer"] = r["our_modhi"] - r["baer_modhi"]
        for q in (10, 25, 50, 75, 90, 95):
            r[f"dnbr_p{q}"] = float(np.percentile(d, q))
        r.update(_meta(bundle))

        if per is not None:                                # pre-fire lifeform, where staged
            g4 = per.to_crs(4326).union_all()
            hit = tiles[tiles.contains(g4.centroid)]
            if len(hit):
                tile = hit.iloc[0]
                with rasterio.open(tile["path"]) as ds:
                    g = per.to_crs(ds.crs).union_all()
                    win = rasterio.windows.from_bounds(*g.bounds, ds.transform)
                    evt = ds.read(1, window=win)
                    tr = ds.window_transform(win)
                if evt.size:
                    mask = rasterio.features.geometry_mask([g], evt.shape, tr, invert=True)
                    vat = gpd.read_file(str(tile["path"]) + ".vat.dbf")
                    lut = dict(zip(vat.Value, vat.EVT_LF))
                    lf = pd.Series([lut.get(int(v), "NA") for v in evt[mask]]).value_counts(normalize=True)
                    nmt = dict(zip(vat.Value, vat.EVT_NAME))
                    top = pd.Series([nmt.get(int(v), "NA") for v in evt[mask]]).value_counts(normalize=True)
                    r.update({"lf_tree": float(lf.get("Tree", 0)), "lf_shrub": float(lf.get("Shrub", 0)),
                              "lf_herb": float(lf.get("Herb", 0)), "evt_top1": top.index[0],
                              "evt_top1_frac": float(top.iloc[0])})
        rows.append(r)
        print(f"{f['incid_name'][:22]:22s} {f['state']} implied={implied:6.1f} ours={ours:6.1f} "
              f"p50={r['dnbr_p50']:6.1f} rank(implied)={r['pct_rank_implied']:5.1f}% "
              f"rank(ours)={r['pct_rank_ours']:5.1f}%  {r['assess_type']}", flush=True)

    df = pd.DataFrame(rows).merge(ig, on="event_id", how="left")
    for c in ("ig_date", "pre_date", "post_date"):
        df[c] = pd.to_datetime(df[c])
    df["days_post"] = (df.post_date - df.ig_date).dt.days      # fire -> post-fire image
    df["pre_lead"] = (df.ig_date - df.pre_date).dt.days        # staleness of the baseline
    df["image_span_d"] = (df.post_date - df.pre_date).dt.days  # the sum; NOT a post-fire delay
    df.to_csv(CAL / "baer_outlier_diagnostics.csv", index=False)

    u = df.drop_duplicates("event_id")
    from scipy import stats
    r2 = stats.pearsonr(u.dnbr_p50, u.implied).statistic ** 2
    print(f"\n{len(u)} distinct fires -> {CAL/'baer_outlier_diagnostics.csv'}")
    print(f"  implied break ~ fire's median dNBR: r2 = {r2:.2f}")
    print(f"  BAER's line sits at the {u.pct_rank_implied.median():.0f}th percentile of the "
          f"fire's own dNBR (range {u.pct_rank_implied.min():.0f}-{u.pct_rank_implied.max():.0f})")
    print(f"  ours lands at the {u.pct_rank_ours.median():.0f}th "
          f"(range {u.pct_rank_ours.min():.0f}-{u.pct_rank_ours.max():.0f})")
    g = u.dropna(subset=["modhi_gap_ours_minus_baer"])
    print(f"  moderate+ area gap (ours - BAER): median {g.modhi_gap_ours_minus_baer.median():+.3f}, "
          f"{(g.modhi_gap_ours_minus_baer > .2).sum()} fires >20 pp too hot, "
          f"{(g.modhi_gap_ours_minus_baer < -.2).sum()} >20 pp too cool")
    grp = pd.cut(g.days_post, [-1, 60, 1000], labels=["imaged <=60 d after", "imaged >60 d after"])
    print(g.groupby(grp, observed=True).agg(
        n=("event_id", "size"), days_post=("days_post", "median"),
        dnbr_p50=("dnbr_p50", "median"), implied=("implied", "median"),
        gap=("modhi_gap_ours_minus_baer", "median"),
        hot20=("modhi_gap_ours_minus_baer", lambda s: int((s > .2).sum())),
        cool20=("modhi_gap_ours_minus_baer", lambda s: int((s < -.2).sum()))).round(3).to_string())
    a = g[g.days_post <= 60].modhi_gap_ours_minus_baer
    b = g[g.days_post > 60].modhi_gap_ours_minus_baer
    print(f"  fresh-vs-delayed Mann-Whitney p={stats.mannwhitneyu(a, b).pvalue:.4f}; "
          f"days_post vs gap rho={stats.spearmanr(g.days_post, g.modhi_gap_ours_minus_baer).statistic:+.3f} "
          f"(p={stats.spearmanr(g.days_post, g.modhi_gap_ours_minus_baer).pvalue:.3f})")
    print(f"  for contrast, where our break lands in the fire's own dNBR: "
          f"rho={stats.spearmanr(g.pct_rank_ours, g.modhi_gap_ours_minus_baer).statistic:+.3f}")
    if "lf_tree" in u:
        v = u.dropna(subset=["lf_tree"])
        print(f"\n  pre-fire tree fraction ({len(v)} fires): vs implied break "
              f"r={stats.pearsonr(v.lf_tree, v.implied).statistic:+.2f}, "
              f"vs dNBR p90 r={stats.pearsonr(v.lf_tree, v.dnbr_p90).statistic:+.2f}")


if __name__ == "__main__":
    main()
