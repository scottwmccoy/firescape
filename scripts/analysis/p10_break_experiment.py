"""Does an assessment-type-aware BARC break beat the single regional one?

Raised by ``p9``: fires whose post-fire image was taken while the scar was
still fresh run a median 10 points of burned area hotter than BAER, so the
obvious candidate is to shift the low/moderate break when the severity map
comes from an initial assessment rather than an extended one.

Stage 1 (``--cache``) pulls each fire's BAER SBS once, locked to the matched
tile, and stores the JOINT HISTOGRAM of (integer dNBR, BAER class) over the
jointly-valid pixels.  That histogram is a sufficient statistic for scoring
ANY break: no pixel arrays, no re-fetching, and the whole experiment below
then runs in seconds.

Stage 2 evaluates offset schemes LEAVE-ONE-FIRE-OUT -- the offset applied to
a fire is fitted only on the other fires -- against the current break and
against two controls that matter:

* a single GLOBAL offset.  If this does as well, "assessment-type-aware" is
  not earning anything; it is just a shifted break.
* a PER-REGION offset, i.e. simply re-tuning the regional numbers on BAER.
  Not a deployable rule (it is fitted on the ground truth), but it bounds
  how much any break-shifting scheme can win.

    python scripts/analysis/p10_break_experiment.py --cache   # slow, network
    python scripts/analysis/p10_break_experiment.py           # fast, offline
"""
import glob
import sys
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features

from firescape import baer, paths, severity

CAL = paths.products_dir("calibration")
CACHE = CAL / "baer_joint_hist.parquet"
DIAG = CAL / "baer_outlier_diagnostics.csv"
LOW, HIGH = 125.0, 500.0          # the other two BARC breaks, held at convention


# ----------------------------------------------------------------- stage 1 ---
def build_cache():
    diag = pd.read_csv(DIAG)
    ca_set = pd.read_csv(paths.__file__.replace("paths.py", "") +
                         "data/calibration/fire_sets/ca_baer_candidates.csv")
    oid = dict(zip(ca_set.event_id, ca_set.baer_oid))

    rows = []
    for eid, g in diag.groupby("event_id", sort=False):
        f = g.iloc[0]
        bundle = paths.raw_dir("mtbs", "fires", eid)
        tif = sorted(glob.glob(str(bundle / "*_dnbr.tif")))[0]
        with rasterio.open(tif) as ds:
            dnbr = ds.read(1).astype("float64")
            nodata, epsg, b, shape = ds.nodata, ds.crs.to_epsg(), ds.bounds, dnbr.shape
        o = oid.get(eid)
        if o is None or o != o:
            per = gpd.read_file(sorted(bundle.glob("*_burn_area.shp"))[0]).to_crs(4326)
            m = baer.match_perimeters(per, min_frac=0.90, fire_years=[int(f.ig_year)])
            o = int(m.iloc[0]["baer_oid"]) if len(m) else None
        cls = baer.fetch_aligned((b.left, b.bottom, b.right, b.top), shape, epsg,
                                 lock_raster_id=o)
        ok = severity.valid_dnbr(dnbr, nodata) & (cls >= 1) & (cls <= 4)
        if ok.sum() < 200:
            print(f"  {f.incid_name}: only {int(ok.sum())} jointly-valid px -- skipped", flush=True)
            continue
        d = np.rint(dnbr[ok]).astype(np.int32)
        c = cls[ok].astype(np.uint8)
        pair, n = np.unique(np.stack([d, c]), axis=1, return_counts=True)
        rows.append(pd.DataFrame({"event_id": eid, "dnbr": pair[0], "baer": pair[1], "n": n}))
        print(f"{f.incid_name[:24]:24s} {int(ok.sum()):9,d} px -> {len(rows[-1]):6,d} bins", flush=True)

    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(CACHE, index=False)
    print(f"\n{out.event_id.nunique()} fires -> {CACHE} ({CACHE.stat().st_size/1e6:.1f} MB)")


# ----------------------------------------------------------------- stage 2 ---
def score(h, brk):
    """Agreement metrics for one fire's joint histogram at a low/moderate break.

    ``h`` is that fire's (dnbr, baer, n) frame.  Everything is a weighted
    count over histogram bins, so this is exact, not sampled.
    """
    ours = np.where(h.dnbr.values < LOW, 1,
                    np.where(h.dnbr.values < brk, 2,
                             np.where(h.dnbr.values < HIGH, 3, 4)))
    b, n = h.baer.values, h.n.values.astype("float64")
    tot = n.sum()
    modhi_agree = n[(ours >= 3) == (b >= 3)].sum() / tot
    our_modhi = n[ours >= 3].sum() / tot
    baer_modhi = n[b >= 3].sum() / tot
    po = n[ours == b].sum() / tot
    pe = sum((n[ours == k].sum() / tot) * (n[b == k].sum() / tot) for k in (1, 2, 3, 4))
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
    return modhi_agree, our_modhi - baer_modhi, kappa


def best_offset(hists, fires, idx):
    """Offset (dNBR units) maximizing mean moderate+ agreement over ``idx``."""
    grid = np.arange(-160, 161, 5.0)
    best, bo = -1.0, 0.0
    for o in grid:
        v = np.mean([score(hists[e], fires.at[e, "our_break"] + o)[0] for e in idx])
        if v > best:
            best, bo = v, o
    return bo


def main():
    diag = pd.read_csv(DIAG).drop_duplicates("event_id").set_index("event_id")
    h = pd.read_parquet(CACHE)
    hists = {e: g for e, g in h.groupby("event_id", sort=False)}
    fires = diag.loc[[e for e in hists if e in diag.index]].copy()
    hists = {e: hists[e] for e in fires.index}

    # MTBS's own label where it exists; BAER/RAVG/provisional bundles are
    # initial products by construction, and the NV rows carry no label column.
    fires["atype"] = np.where(fires.assess_type.eq("Extended"), "extended", "initial")
    fires["atype"] = np.where(fires.assess_type.isna() & (fires.days_post > 60),
                              "extended", fires.atype)
    print(f"{len(fires)} fires: {(fires.atype=='initial').sum()} initial, "
          f"{(fires.atype=='extended').sum()} extended "
          f"(median days_post {fires.groupby('atype').days_post.median().to_dict()})\n")

    schemes = {
        "current regional break": lambda e, tr: 0.0,
        "+ global offset": lambda e, tr: best_offset(hists, fires, tr),
        "+ assessment-type offset": lambda e, tr: best_offset(
            hists, fires, [j for j in tr if fires.at[j, "atype"] == fires.at[e, "atype"]]),
        "+ per-region offset": lambda e, tr: best_offset(
            hists, fires, [j for j in tr if fires.at[j, "region_name"] == fires.at[e, "region_name"]]),
    }

    res = {}
    for name, fn in schemes.items():
        rows = []
        for e in fires.index:
            tr = [j for j in fires.index if j != e]          # leave-one-fire-out
            try:
                off = fn(e, tr)
            except ValueError:
                off = 0.0
            a, gap, k = score(hists[e], fires.at[e, "our_break"] + off)
            rows.append({"event_id": e, "offset": off, "modhi_agree": a,
                         "area_gap": gap, "kappa": k})
        res[name] = pd.DataFrame(rows).set_index("event_id")
        print(f"  scored: {name}", flush=True)

    base = res["current regional break"]
    print(f"\n{'scheme':28s} {'modhi':>7s} {'kappa':>7s} {'|gap|':>7s} {'>20pp':>6s} "
          f"{'better':>7s} {'worse':>6s} {'Wilcoxon p':>11s}  offsets")
    from scipy import stats
    for name, r in res.items():
        d = r.modhi_agree - base.modhi_agree
        w = stats.wilcoxon(d).pvalue if name != "current regional break" and d.abs().sum() else np.nan
        offs = r.offset.round(0).astype(int)
        tag = (f"{offs.iloc[0]:+d}" if offs.nunique() == 1 else
               " / ".join(f"{k}:{int(v):+d}" for k, v in
                          r.join(fires.atype).groupby("atype").offset.median().items())
               if name == "+ assessment-type offset" else
               f"{offs.min():+d}..{offs.max():+d}")
        print(f"  {name:26s} {r.modhi_agree.median():7.3f} {r.kappa.median():7.3f} "
              f"{r.area_gap.abs().median():7.3f} {int((r.area_gap.abs()>.2).sum()):6d} "
              f"{int((d>1e-6).sum()):7d} {int((d<-1e-6).sum()):6d} "
              f"{('' if w != w else f'{w:.4f}'):>11s}  {tag}")

    out = pd.concat({k: v for k, v in res.items()}, names=["scheme"]).reset_index()
    out.to_csv(CAL / "break_experiment_loo.csv", index=False)
    print(f"\nper-fire results -> {CAL/'break_experiment_loo.csv'}")


if __name__ == "__main__":
    (build_cache if "--cache" in sys.argv else main)()
