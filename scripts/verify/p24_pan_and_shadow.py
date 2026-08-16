"""Part A: does a native pan-difference channel lift narrow-channel (fluvial)
detection? Part B: Scott's shadow-alignment idea -- correlate DEM-modelled
shading against observed brightness as an independent DEM-to-imagery
registration measure.
"""
import sys, warnings, json
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pyogrio, rasterio
from rasterio.merge import merge
from rasterio.warp import Resampling, reproject, transform_bounds
from scipy.stats import mannwhitneyu
from scipy.ndimage import gaussian_filter, shift as ndshift
from shapely.geometry import box
from shapely import affinity

sys.path.insert(0, str(Path(__file__).parent))
from p20_snap_rescore import maxar_z, BOX4326, WV, INV, DOLAN
from firescape import change as ch, corridor as co, epochs, paths, sources

S = Path(sys.argv[1])
SHIFT_M = (6.0, 50.0)      # network alignment: y north+, x east+ (metres)

def auc(hi, lo):
    hi, lo = hi.dropna(), lo.dropna()
    return float(mannwhitneyu(hi, lo, alternative="greater").statistic
                 / (len(hi)*len(lo))) if len(hi) and len(lo) else float("nan")

def dem_native():
    b = transform_bounds("EPSG:32610", "EPSG:4326",
                         *box(*transform_bounds("EPSG:4326", "EPSG:32610",
                                                *BOX4326)).bounds)
    want = (b[0]-.03, b[1]-.03, b[2]+.03, b[3]+.03)
    srcs = [rasterio.open(t) for t in sorted(DOLAN.glob("USGS_13_*.tif"))]
    dem, tr = merge(srcs, bounds=want, nodata=np.nan)
    crs = srcs[0].crs
    [s.close() for s in srcs]
    return dem[0].astype(np.float32), tr, crs, want

def cosi_of(dem, tr, want, az, el):
    lat = (want[1]+want[3])/2
    gy, gx = np.gradient(dem, abs(tr.e)*111132.0,
                         abs(tr.a)*111320.0*np.cos(np.radians(lat)))
    sl, asp = np.arctan(np.hypot(gx, gy)), np.arctan2(-gx, gy)
    z, a = np.radians(90-el), np.radians(az)
    return (np.cos(z)*np.cos(sl)+np.sin(z)*np.sin(sl)*np.cos(a-asp)).astype(np.float32)

def togrid(arr, tr, crs, ref, resamp=Resampling.bilinear):
    dst = np.full((ref["height"], ref["width"]), np.nan, np.float32)
    reproject(arr, dst, src_transform=tr, src_crs=crs,
              dst_transform=ref["transform"], dst_crs=ref["crs"],
              resampling=resamp, src_nodata=np.nan, dst_nodata=np.nan)
    return dst

# =============================== PART B: shadow alignment ==================
print("=== PART B: shadow-model alignment ===", flush=True)
try:
    dem, tr_nat, crs_nat, want = dem_native()
    ref2 = epochs.grid(BOX4326, "EPSG:32610", 2.0)
    pre_src = sources.MaxarStrips(WV / "2020_11_29")
    az0, el0 = pre_src.sun()
    pre = pre_src.read_into(ref2)
    obs = ch.brightness(pre["blue"], pre["green"], pre["red"], pre["nir"])
    mod = togrid(cosi_of(dem, tr_nat, want, az0, el0), tr_nat, crs_nat, ref2)
    def hp(a, sig=25):
        f = np.ma.filled(np.ma.masked_invalid(np.ma.asarray(a)), np.nan)
        m = np.isfinite(f)
        f = np.where(m, f, np.nanmean(f))
        return np.where(m, f - gaussian_filter(f, sig), 0.0)
    O, M = hp(obs), hp(mod)
    valid = (~np.ma.getmaskarray(obs)) & np.isfinite(mod)
    ys, xs = np.nonzero(valid[::8, ::8])
    offs = []
    rng = np.random.default_rng(2)
    for k in rng.choice(len(ys), size=min(3, len(ys)), replace=False):
        cy, cx = ys[k]*8, xs[k]*8
        slw = (slice(max(cy-768, 0), cy+768), slice(max(cx-768, 0), cx+768))
        if valid[slw].mean() < 0.7:
            continue
        dy, dx = ch.scene_offset(M[slw], O[slw])
        offs.append((round(dy, 1), round(dx, 1)))
    print(f"Dolan DEM->imagery shading offsets (px@2m, windows): {offs}",
          flush=True)
    # Hidden Valley
    hvpre, _, _, _, hvref = epochs.build(
        paths.raw_dir("planet", "hidden_valley", "pre_20260619"),
        indices=("brightness",), rgb=True, keep_nir=False, verbose=False)
    tiles = sorted((paths.cache_root()/"3dep_tiles").glob("USGS_13_*.tif"))
    from firescape import fans
    demhv, slhv = fans.dem_and_slope(tiles, hvref)
    # june sun over HV approx az 125, el 65 (mean of epoch scenes ~ from planet metadata; use rough summer value)
    gy, gx = np.gradient(demhv, 3.0, 3.0)
    slp, asp = np.arctan(np.hypot(gx, gy)), np.arctan2(-gx, gy)
    z_, a_ = np.radians(90-65), np.radians(125)
    modhv = (np.cos(z_)*np.cos(slp)+np.sin(z_)*np.sin(slp)*np.cos(a_-asp)).astype(np.float32)
    obshv = hvpre["brightness"][0]
    Oh, Mh = hp(obshv), hp(modhv)
    vh = (~np.ma.getmaskarray(obshv)) & np.isfinite(modhv)
    cy, cx = np.array(Oh.shape)//2
    slw = (slice(cy-700, cy+700), slice(cx-700, cx+700))
    dy, dx = ch.scene_offset(Mh[slw], Oh[slw])
    print(f"HV DEM->imagery shading offset (px@3m): ({dy:.1f}, {dx:.1f})",
          flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"PART B failed: {e}", flush=True)

# =============================== PART A: pan channel =======================
print("=== PART A: pan-difference channel ===", flush=True)
try:
    ref1 = epochs.grid(BOX4326, "EPSG:32610", 1.0)
    dem, tr_nat, crs_nat, want = dem_native()
    demfile = S/"dolan_dem_native.tif"
    if not demfile.exists():
        prof = dict(driver="GTiff", height=dem.shape[0], width=dem.shape[1],
                    count=1, dtype="float32", crs=crs_nat, transform=tr_nat,
                    nodata=np.nan)
        with rasterio.open(demfile, "w", **prof) as d:
            d.write(dem, 1)
    shape1 = (ref1["height"], ref1["width"])

    def mosaic_pan(paths_, rpc=False):
        acc = np.full(shape1, np.nan, np.float32)
        for p in paths_:
            imd = p.with_suffix(".IMD")
            scale = 1.0
            if imd.exists():
                m = sources.parse_imd(imd)
                b0 = m["bands"][0]
                if np.isfinite(b0["abscal"]) and np.isfinite(b0["bandwidth"]):
                    scale = b0["abscal"]/b0["bandwidth"]
            with rasterio.open(p) as src:
                dst = np.full(shape1, np.nan, np.float32)
                kw = {}
                if rpc and src.rpcs:
                    kw = dict(rpcs=src.rpcs, RPC_DEM=str(demfile),
                              src_crs="EPSG:4326")
                else:
                    kw = dict(src_transform=src.transform, src_crs=src.crs)
                reproject(rasterio.band(src, 1), dst,
                          dst_transform=ref1["transform"], dst_crs=ref1["crs"],
                          resampling=Resampling.bilinear,
                          src_nodata=0, dst_nodata=np.nan, **kw)
                f = np.isnan(acc) & np.isfinite(dst)
                acc[f] = dst[f]*scale
        return np.ma.masked_invalid(acc)

    post_ntf = sorted((WV/"2021_05_08").glob("*/*P1BS*/*.NTF"))
    pre_p2 = sorted((WV/"2020_11_09").glob("*/*P?AS*/*.TIF")) \
        + sorted((WV/"2020_11_09").glob("*/*P?AS*/*.NTF")) \
        + sorted((WV/"2020_11_09").glob("*/*P?BS*/*.NTF"))
    print(f"pre pan files: {len(pre_p2)}, post P1BS: {len(post_ntf)}", flush=True)
    pre_pan = mosaic_pan(pre_p2)
    post_pan = mosaic_pan(post_ntf, rpc=True)
    both = ~(np.ma.getmaskarray(pre_pan) | np.ma.getmaskarray(post_pan))
    print(f"pan joint coverage: {both.mean():.2%}", flush=True)
    if both.mean() < 0.002:
        raise RuntimeError("pan joint coverage too small")
    d = post_pan/np.ma.median(post_pan[both]) - pre_pan/np.ma.median(pre_pan[both])
    # illumination regression on 1m grid
    az0 = np.mean([sources.parse_imd(p.with_suffix('.IMD'))['sun_az'] for p in pre_p2 if p.with_suffix('.IMD').exists()])
    el0 = np.mean([sources.parse_imd(p.with_suffix('.IMD'))['sun_el'] for p in pre_p2 if p.with_suffix('.IMD').exists()])
    az1 = np.mean([sources.parse_imd(p.with_suffix('.IMD'))['sun_az'] for p in post_ntf if p.with_suffix('.IMD').exists()])
    el1 = np.mean([sources.parse_imd(p.with_suffix('.IMD'))['sun_el'] for p in post_ntf if p.with_suffix('.IMD').exists()])
    ci0 = togrid(cosi_of(dem, tr_nat, want, az0, el0), tr_nat, crs_nat, ref1)
    ci1 = togrid(cosi_of(dem, tr_nat, want, az1, el1), tr_nat, crs_nat, ref1)
    dci = ci1-ci0
    sh = (ci0 < 0.08)|(ci1 < 0.08)
    m = both & np.isfinite(dci) & ~sh & ~np.ma.getmaskarray(d)
    x, y = dci[m], np.ma.filled(d, np.nan)[m]
    sub = np.random.default_rng(1).choice(len(x), min(1_500_000, len(x)), False)
    x, y = x[sub], y[sub]
    keep = np.abs(y-np.median(y)) < 8*np.std(y)
    for _ in range(3):
        b, a = np.polyfit(x[keep], y[keep], 1)
        keep = np.abs(y-(a+b*x)) < 2.5*np.std(y[keep]-(a+b*x[keep]))
    z1 = np.ma.masked_where(sh | ~np.isfinite(dci) | ~both, d-(a+b*dci))
    z1 = z1 - np.ma.median(z1)
    z1 = z1/(1.4826*np.ma.median(np.ma.abs(z1)))

    seg = pyogrio.read_dataframe(INV, columns=["UpArea_km2", "Confidence"]).to_crs(ref1["crs"])
    seg = seg[seg.intersects(box(*transform_bounds("EPSG:4326", ref1["crs"], *BOX4326)))].reset_index(drop=True)
    seg["geometry"] = seg.geometry.apply(
        lambda g: affinity.translate(g, xoff=SHIFT_M[1], yoff=SHIFT_M[0]))
    lab1 = co.corridor_raster(seg, ref1, area_col="UpArea_km2")
    t_all = seg["Confidence"].replace({2: 3})
    st = co.segment_stats(lab1, {"zp": z1}).reindex(range(len(seg)))
    ok = st["n_pixels"].fillna(0) >= 24
    v, t = st.loc[ok, "zp_mean"], t_all[ok]
    r_pan = dict(responded=round(auc(v[t > 0], v[t == 0]), 3),
                 fluvial_vs_none=round(auc(v[t == 1], v[t == 0]), 3),
                 df_vs_fluvial=round(auc(v[t == 3], v[t == 1]), 3),
                 n=int(ok.sum()))
    print(f"PAN 1m: {r_pan}", flush=True)

    # MS 2m baseline on the SAME segments
    ref2 = epochs.grid(BOX4326, "EPSG:32610", 2.0)
    z2 = maxar_z(WV/"2020_11_29", [WV/"2021_04_19", WV/"2021_05_08"], ref2)
    seg2 = pyogrio.read_dataframe(INV, columns=["UpArea_km2", "Confidence"]).to_crs(ref2["crs"])
    seg2 = seg2[seg2.intersects(box(*transform_bounds("EPSG:4326", ref2["crs"], *BOX4326)))].reset_index(drop=True)
    seg2["geometry"] = seg2.geometry.apply(
        lambda g: affinity.translate(g, xoff=SHIFT_M[1], yoff=SHIFT_M[0]))
    lab2 = co.corridor_raster(seg2, ref2, area_col="UpArea_km2")
    st2 = co.segment_stats(lab2, {"zm": z2}).reindex(range(len(seg2)))
    same = st2.loc[ok[ok].index]          # same segment ids as pan-scored
    v2, t2 = same["zm_mean"], t_all[ok]
    r_ms = dict(responded=round(auc(v2[t2 > 0], v2[t2 == 0]), 3),
                fluvial_vs_none=round(auc(v2[t2 == 1], v2[t2 == 0]), 3),
                df_vs_fluvial=round(auc(v2[t2 == 3], v2[t2 == 1]), 3))
    print(f"MS 2m same segments: {r_ms}", flush=True)
    (S/"dolan_pan_summary.json").write_text(json.dumps(
        {"pan_1m": r_pan, "ms_2m_same_segments": r_ms,
         "pre_pan_files": len(pre_p2), "post_pan_files": len(post_ntf)}, indent=2))
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"PART A failed: {e}", flush=True)
