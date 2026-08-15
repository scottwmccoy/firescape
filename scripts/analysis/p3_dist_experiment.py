"""Distributional severity experiment: det vs dispersed on 12 units x 2 volume models.

For each unit: delineate once, build per-class catchment weights W (all) and
Ws (steep), then evaluate both severity closures on the SAME network:
  deterministic: d_c at regional P; T = Ws @ (d >= brk); F = W @ d / 1000
  dispersed:     p_c = class_exceedance;  T = Ws @ p_c;  F = W @ E[dnbr] / 1000
Bmh = catchment area x (W @ indicator/p). Likelihood via M1; volumes via
Gartner-14 (needs Bmh) and RANGES (severity-free); Cannon class for every
combination. sigma = 0.91 (measured), regional pdsim/breaks (statewide_v1).
"""
import json
import os
import time
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize as rio_rasterize

from firescape import config, delineate, paths, severity, soils, statewide
from firescape import hazard as hz
from firescape.calibrate import STEEP_GRADIENT

UNIT_NAMES = [
    "City of Reno-Truckee River", "Marlette Lake-Frontal Lake Tahoe",
    "Middle West Walker River", "Kyle Canyon-Las Vegas Wash", "Bull Creek",
    "Upper Jarbidge River", "Thousand Creek", "Headwaters Duck Creek",
    "Steptoe Creek", "Rock Creek", "Upper Buffalo Valley",
    "Martins Creek-White River",
]
SIGMA = severity.SIGMA_Z_DEFAULT
TILE_DIR = paths.cache_root() / "3dep_tiles"
EVT_DIR = paths.raw_dir("landfire", "LF2025_EVT_nv")
OUT = paths.products_dir("prefire", "statewide_v1")

hu = gpd.read_file(paths.interim_dir("statewide") / "nv_hu10.geojson")
regions = pd.read_csv(paths.interim_dir("statewide") / "hu10_regions.csv",
                      dtype={"huc10": str})
region_of = dict(zip(regions["huc10"], regions["region"]))
legend = pd.read_csv(paths.interim_dir("statewide") / "evt_legend_union.csv")
mapping, _ = severity.remap_crosswalk(legend)
water = set(delineate.water_codes_from_legend(legend))
table = severity.load_cdf_table(table="nv_merged")
cal_pack = config.packaged_calibration("statewide_v1")


def slug(r):
    return r.lower().replace(" ", "_").replace("&", "and")


with rasterio.open(paths.interim_dir("statewide") / "pga25_g.tif") as pg:
    pga_arr, pga_tr = pg.read(1), pg.transform

z14 = np.load(paths.interim_dir("statewide") / "atlas14_i15.npz")
from rasterio.transform import Affine

i1g, tr14 = z14["i1"], Affine(*z14["transform"])

rows, unit_rows = [], []
sel = hu[hu["name"].isin(UNIT_NAMES)]
print(f"{len(sel)} experiment units", flush=True)
for _, r in sel.iterrows():
    huc, name = str(r["huc10"]), r["name"]
    reg = region_of[huc]
    cal = cal_pack.region(slug(reg))
    P, brk = cal.pdsim, cal.barc_breaks[1]
    t1 = time.time()
    g5070 = gpd.GeoSeries([r.geometry], crs=hu.crs).to_crs("EPSG:5070")
    b = tuple(g5070.total_bounds)
    dem = statewide.dem_for_unit(b, TILE_DIR)
    evt = statewide.evt_for_unit(b, EVT_DIR)
    from pfdf.raster import Raster

    dom_arr = rio_rasterize(((g, 1) for g in g5070.geometry),
                            out_shape=dem.shape,
                            transform=dem.transform.affine
                            if hasattr(dem.transform, "affine") else dem.transform,
                            fill=0, dtype="uint8").astype(bool)
    domain = Raster.from_array(dom_arr, spatial=dem, isbool=True)
    segments, terr = delineate.network(dem, domain)
    if segments is None:
        continue
    evt_g = delineate.match_grid(evt, dem, resampling="nearest")
    evt_vals = severity.apply_crosswalk(evt_g.values, mapping)
    evt_vals = np.where(np.isin(evt_g.values, list(water)), -1, evt_vals)
    classes = [int(c) for c in np.unique(evt_vals) if c > 0]
    steep = terr.slopes.values >= STEEP_GRADIENT
    W = np.zeros((segments.size, len(classes)))
    Ws = np.zeros_like(W)
    for j, c in enumerate(classes):
        in_c = evt_vals == c
        W[:, j] = np.asarray(segments.catchment_ratio(
            Raster.from_array(in_c, spatial=dem, isbool=True)), dtype=float)
        Ws[:, j] = np.asarray(segments.catchment_ratio(
            Raster.from_array(in_c & steep, spatial=dem, isbool=True)), dtype=float)

    fb = severity.BARREN_EVT_CODE
    lam = table["Weibull_Lambda_Scale"].reindex(classes).fillna(
        table.at[fb, "Weibull_Lambda_Scale"]).to_numpy(float)
    kap = table["Weibull_Kappa_Shape"].reindex(classes).fillna(
        table.at[fb, "Weibull_Kappa_Shape"]).to_numpy(float)
    fbtab = pd.DataFrame({"Weibull_Lambda_Scale": lam,
                          "Weibull_Kappa_Shape": kap}, index=classes)

    d_det = severity.weibull_dnbr(P, lam, kap)
    p_soft = severity.class_exceedance(classes, brk, P, fbtab, sigma=SIGMA)
    e_soft = severity.expected_dnbr(classes, P, fbtab, sigma=SIGMA)

    area = np.asarray(segments.area(units="kilometers"), dtype=float)
    relief = np.asarray(segments.relief(terr.relief), dtype=float)
    # KF for the S term (severity-independent)
    kf, kf_src = soils.kf_raster(dem, polygons=paths.raw_dir("ssurgo")
                                 / "nv_ssurgo_kf.gpkg", key=f"exp_{huc}")
    S = np.asarray(segments.catchment_summary("mean", kf), dtype=float)

    res = {}
    for mode, Tv, Fv, mh in (("det", Ws @ (d_det >= brk), W @ d_det / 1000.0,
                              W @ (d_det >= brk).astype(float)),
                             ("soft", Ws @ p_soft, W @ e_soft / 1000.0,
                              W @ p_soft)):
        p = hz.likelihood_m1(Tv, Fv, S)
        Bmh = area * mh
        Vg, _, _ = hz.volume_g14(Bmh, relief)
        res[mode] = dict(P=p, Bmh=Bmh, Vg=Vg,
                         Hg=hz.combined_c10(p, Vg))

    # RANGES (severity-free): zonal part if staged, else quick inline zonal
    part = paths.interim_dir("statewide") / "ranges_parts" / f"{huc}.parquet"
    if part.exists():
        pt = pd.read_parquet(part)
        ids = np.asarray(segments.ids)
        m = pt.set_index("Segment_ID").reindex(ids)
        slope_deg, frac_north = m["SlopeDeg"].to_numpy(), m["FracNorth"].to_numpy()
    else:
        z = np.asarray(dem.values, dtype="float64")
        gr, gc = np.gradient(z, 10.0)
        sdeg = np.degrees(np.arctan(np.hypot(gr, gc)))
        north = ((np.degrees(np.arctan2(-gc, gr)) % 360.0 >= 315)
                 | (np.degrees(np.arctan2(-gc, gr)) % 360.0 < 45))
        slope_deg = np.asarray(segments.catchment_summary(
            "mean", Raster.from_array(sdeg.astype("float32"), spatial=dem,
                                      nodata=np.float32(np.nan))), float)
        frac_north = np.asarray(segments.catchment_ratio(
            Raster.from_array(north, spatial=dem, isbool=True)), float)
    # sample PGA/I15_1yr at the unit centroid: the 25-km-smoothed PGA and
    # the Atlas 14 grids vary little within one HU10
    cx, cy = g5070.centroid.iloc[0].x, g5070.centroid.iloc[0].y
    rr, cc = rasterio.transform.rowcol(pga_tr, cx, cy)
    pga_u = float(pga_arr[rr, cc])
    lon, lat = gpd.GeoSeries([g5070.centroid.iloc[0]],
                             crs="EPSG:5070").to_crs("EPSG:4269").iloc[0].coords[0]
    r14, c14 = int((lat - tr14.f) / tr14.e), int((lon - tr14.c) / tr14.a)
    i1_u = float(i1g[r14, c14]) if (0 <= r14 < i1g.shape[0]
                                    and 0 <= c14 < i1g.shape[1]) else np.nan
    ratio = 24.0 / i1_u if np.isfinite(i1_u) and i1_u > 0 else np.nan
    Vr, _, _ = hz.volume_ranges(area, slope_deg, ratio, pga_u, frac_north)
    for mode in ("det", "soft"):
        res[mode]["Hr"] = hz.combined_c10(res[mode]["P"], Vr)

    n = segments.size
    urow = {"huc10": huc, "name": name, "region": reg, "pdsim": P, "brk": brk,
            "n_basins": n, "pga_g": round(pga_u, 3),
            "i15_ratio": round(ratio, 2) if np.isfinite(ratio) else np.nan}
    for mode in ("det", "soft"):
        m = res[mode]
        urow[f"{mode}_medP"] = float(np.nanmedian(m["P"]))
        urow[f"{mode}_zeroV_g14"] = float((m["Vg"] <= 0).mean())
        urow[f"{mode}_mod+_g14"] = int((m["Hg"] >= 2).sum())
        urow[f"{mode}_mod+_ranges"] = int((m["Hr"] >= 2).sum())
    urow["zeroV_ranges"] = float(np.mean(~np.isfinite(Vr) | (Vr <= 0)))
    urow["medV_ranges"] = float(np.nanmedian(Vr))
    urow["sec"] = round(time.time() - t1)
    unit_rows.append(urow)
    print(f"{name[:28]:<28} {reg[:12]:<12} n={n:5d} "
          f"medP {urow['det_medP']:.2f}->{urow['soft_medP']:.2f} "
          f"zeroVg14 {urow['det_zeroV_g14']:.0%}->{urow['soft_zeroV_g14']:.0%} "
          f"mod+ g14 {urow['det_mod+_g14']}->{urow['soft_mod+_g14']} "
          f"| ranges mod+ {urow['det_mod+_ranges']}->{urow['soft_mod+_ranges']} "
          f"({urow['sec']}s)", flush=True)

df = pd.DataFrame(unit_rows)
out = paths.products_dir("calibration") / "dist_experiment_units.csv"
df.to_csv(out, index=False)
print("\nwrote", out)
print(df[["name", "region", "det_medP", "soft_medP", "det_zeroV_g14",
          "soft_zeroV_g14", "det_mod+_g14", "soft_mod+_g14",
          "det_mod+_ranges", "soft_mod+_ranges", "medV_ranges"]]
      .to_string(index=False))
