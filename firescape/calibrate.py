"""Nevada P_dsim calibration — the Rossi et al. (2025) DFL method (M3).

Per calibration fire: delineate basins in the buffered perimeter; compute the
observed M1 terrain/fire variables (T_obs, F_obs) from MTBS severity; sweep
P_dsim over a grid, computing the simulated (T_sim(P), F_sim(P)) via an exact
class decomposition; pick each basin's best P; fire P_dsim = median of basin
values over the selection (>=75% of catchment in perimeter, area <= 8 km^2,
catchment median dNBR >= the fire's unburned-low threshold). Regional P_dsim
= median of fire values; regional low-moderate break = median of fire mod_t.

Two structural tricks:

1. **Class decomposition.** Simulated dNBR is constant per EVT class, so for
   catchment weights W[b,c] (fraction of catchment b in class c) and
   steep-and-in-class weights Ws[b,c]:
       F_sim(P) = W @ d(P) / 1000
       T_sim(P) = Ws[:, d(P) >= regional_break] summed over hot classes
   where d(P) is the per-class Weibull dNBR. One catchment_ratio per class
   replaces one full M1.variables call per P value (99x fewer accumulations).
   Validated against a direct pfdf M1.variables call at P=0.5 on every fire.

2. **Soil cancellation.** X_sim - X_obs = (Ct*(T_sim-T_obs) + Cf*(F_sim-F_obs))*R
   — the Cs*S*R term cancels, so the best-P solve needs no KF factor at all
   (logit-space solve; ~identical to Rossi's DFL-space solve away from
   logistic saturation). DFL-space diagnostics (NSE etc.) are computed later
   from the cached T/F curves once KF is available.

Simulated burning is masked to the fire perimeter during calibration (outside
contributes zero to T and F, approximating the ~zero observed dNBR there);
basins are required to have >=75% of their catchment inside, limiting the
effect.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from firescape import mtbs, paths, severity
from firescape.config import FilterDefaults
from firescape.delineate import match_grid, network

PDSIM_GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)
STEEP_GRADIENT = math.tan(math.radians(23.0))  # M1 terrain slope threshold
PERIMETER_BUFFER_M = 3000.0
OUTSIDE_CODE = -9999


@dataclass
class FireCalib:
    event_id: str
    pdsim: float | None
    n_delineated: int
    n_selected: int
    validation_dT: float
    validation_dF: float
    cache_dir: Path

    @property
    def ok(self) -> bool:
        return self.pdsim is not None and np.isfinite(self.pdsim)


def _cache_dir(event_id: str, tag: str = "") -> Path:
    """Per-fire cache. ``tag`` separates runs that differ in an input the
    cache cannot otherwise distinguish (e.g. the EVT vintage)."""
    name = event_id.upper() + (f"__{tag}" if tag else "")
    return paths.interim_dir("calib", name)


def load_cached(event_id: str, tag: str = "", table: str = "") -> FireCalib | None:
    d = _cache_dir(event_id, tag)
    f = d / "fire.json"
    if not f.exists():
        return None
    meta = json.loads(f.read_text())
    pdsim = meta.get("pdsim")
    if table:
        pdsim = meta.get("pdsim_by_table", {}).get(table)
        if pdsim is None:
            return None
    return FireCalib(event_id=event_id.upper(), pdsim=pdsim,
                     n_delineated=meta["n_delineated"], n_selected=meta["n_selected"],
                     validation_dT=meta["validation_dT"], validation_dF=meta["validation_dF"],
                     cache_dir=d)


def fire_calibration(event_id: str, thresholds: tuple[float, float],
                     regional_break: float, *, evt_path: Path,
                     crosswalk: dict[int, int], dem_path: Path | None = None,
                     i15_mmh: float = 24.0, cdf_tables: dict | None = None,
                     tag: str = "") -> FireCalib | dict:
    """Calibrate P_dsim for one fire. ``thresholds`` = (low_t, mod_t) analyst
    values; ``regional_break`` classifies the simulated dNBR (Rossi Fig 3).

    ``cdf_tables`` maps a name to a CDF parameter table. The expensive work
    (delineation, terrain, observed T/F, per-class catchment weights) does not
    depend on the table, so every table is evaluated in the same pass and the
    only difference between them is the per-class dNBR vector — which makes a
    table-vs-table comparison exact rather than merely reproducible. Returns a
    single FireCalib when ``cdf_tables`` is None, else a dict keyed by name.
    """
    import geopandas as gpd
    import rasterio
    from pfdf import severity as pfsev
    from pfdf.models import staley2017 as s17
    from pfdf.projection import BoundingBox
    from pfdf.raster import Raster
    from pfdf.utils import intensity

    multi = cdf_tables is not None
    cached = load_cached(event_id, tag)
    if cached is not None and not multi:
        return cached
    if cached is not None and multi:
        have = json.loads((_cache_dir(event_id, tag) / "fire.json").read_text())
        by_table = have.get("pdsim_by_table", {})
        if all(name in by_table for name in cdf_tables):
            return {name: load_cached(event_id, tag, name) for name in cdf_tables}

    low_t, mod_t = thresholds
    cache = _cache_dir(event_id, tag)
    bundle = mtbs.fire_bundle(event_id)
    dem_path = Path(dem_path) if dem_path else paths.interim_dir("pilot") / "pilot_dem.tif"
    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
    perim_gdf = gpd.read_file(bundle["burn_area"]).to_crs(dem_crs)
    w, s, e, n = perim_gdf.union_all().buffer(PERIMETER_BUFFER_M).bounds
    dem = Raster.from_file(dem_path, bounds=BoundingBox(w, s, e, n, crs=dem_crs))
    res = dem.resolution("meters")
    perim = match_grid(Raster.from_polygons(bundle["burn_area"], bounds=dem, resolution=res), dem)
    perim_arr = perim.values.astype(bool)

    # domain = buffered perimeter: rasterize the buffered geometry
    dom_path = cache / "_domain.geojson"
    cache.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(geometry=[perim_gdf.union_all().buffer(PERIMETER_BUFFER_M)],
                     crs=dem_crs).to_file(dom_path, driver="GeoJSON")
    domain = match_grid(Raster.from_polygons(dom_path, bounds=dem, resolution=res), dem)

    segments, terr = network(dem, domain)
    n0 = segments.size

    # ---- observed side ------------------------------------------------------
    dnbr = match_grid(Raster.from_file(bundle["dnbr"]), dem, resampling="bilinear")
    if "dnbr6" in bundle:
        dnbr6 = match_grid(Raster.from_file(bundle["dnbr6"]), dem, resampling="nearest")
        barc4_arr = mtbs.dnbr6_to_barc4(dnbr6.values)
    else:
        barc4_arr = pfsev.estimate(dnbr).values.astype(np.uint8)
    barc4 = Raster.from_array(barc4_arr, spatial=dem, nodata=0)
    modhigh = pfsev.mask(barc4, ["moderate", "high"])
    kf_dummy = Raster.from_array(np.full(dem.shape, 0.25, dtype="float32"),
                                 spatial=dem, nodata=np.float32(np.nan))
    T_obs, F_obs, _ = s17.M1.variables(segments, modhigh, terr.slopes, dnbr, kf_dummy,
                                       omitnan=True)

    # ---- selection ----------------------------------------------------------
    area = np.asarray(segments.area(units="kilometers"), dtype=float)
    ratio_in = np.asarray(segments.catchment_ratio(perim), dtype=float)
    try:
        med_dnbr = np.asarray(segments.catchment_summary("median", dnbr), dtype=float)
    except Exception:
        med_dnbr = np.asarray(segments.scaled_dnbr(dnbr), dtype=float) * 1000.0  # mean fallback
    sel = ((area <= FilterDefaults().max_area_km2) & (ratio_in >= 0.75)
           & (med_dnbr >= low_t) & np.isfinite(T_obs) & np.isfinite(F_obs))

    # ---- simulated side: class decomposition -------------------------------
    evt = match_grid(Raster.from_file(evt_path, bounds=dem.bounds), dem, resampling="nearest")
    evt_vals = severity.apply_crosswalk(evt.values, crosswalk)
    evt_vals = np.where(perim_arr, evt_vals, OUTSIDE_CODE)
    tables = dict(cdf_tables) if multi else {"default": severity.load_cdf_table()}
    cdf = next(iter(tables.values()))
    classes = [int(c) for c in np.unique(evt_vals) if c != OUTSIDE_CODE]

    def _params(table):
        lam = np.array([table["Weibull_Lambda_Scale"].get(c, np.nan) for c in classes])
        kap = np.array([table["Weibull_Kappa_Shape"].get(c, np.nan) for c in classes])
        fb_lam = table.at[severity.BARREN_EVT_CODE, "Weibull_Lambda_Scale"]
        fb_kap = table.at[severity.BARREN_EVT_CODE, "Weibull_Kappa_Shape"]
        return (np.where(np.isfinite(lam), lam, fb_lam),
                np.where(np.isfinite(kap), kap, fb_kap))

    lam, kap = _params(cdf)

    steep = terr.slopes.values >= STEEP_GRADIENT
    W = np.zeros((segments.size, len(classes)))
    Ws = np.zeros_like(W)
    for j, c in enumerate(classes):
        in_c = evt_vals == c
        W[:, j] = np.asarray(segments.catchment_ratio(
            Raster.from_array(in_c, spatial=dem, isbool=True)), dtype=float)
        Ws[:, j] = np.asarray(segments.catchment_ratio(
            Raster.from_array(in_c & steep, spatial=dem, isbool=True)), dtype=float)

    D = severity.weibull_dnbr(PDSIM_GRID[:, None], lam[None, :], kap[None, :])  # [nP, nc]
    F_sim = (W @ D.T) / 1000.0                         # [nb, nP]
    hot = D >= regional_break                          # [nP, nc]
    T_sim = Ws @ hot.T.astype(float)                   # [nb, nP]

    # ---- decomposition validation at P=0.5 ---------------------------------
    k = int(np.argmin(np.abs(PDSIM_GRID - 0.5)))
    sim_dnbr05, _src = severity.simulate_dnbr(evt_vals, 0.5, cdf, evt_nodata=OUTSIDE_CODE)
    sim_dnbr05 = np.where(perim_arr, sim_dnbr05, 0.0).astype(np.float32)
    barc05 = severity.classify_barc4(sim_dnbr05, (low_t if 0 < low_t < mod_t else 125.0,
                                                  regional_break, 5000.0))
    barc05[~perim_arr] = 1
    Tv, Fv, _ = s17.M1.variables(
        segments,
        pfsev.mask(Raster.from_array(barc05, spatial=dem, nodata=0), ["moderate", "high"]),
        terr.slopes,
        Raster.from_array(sim_dnbr05, spatial=dem, nodata=np.float32(np.nan)),
        kf_dummy, omitnan=True)
    dT = float(np.nanmax(np.abs(np.asarray(Tv) - T_sim[:, k])))
    dF = float(np.nanmax(np.abs(np.asarray(Fv) - F_sim[:, k])))

    # ---- best P per basin (logit space; S cancels) -------------------------
    B, Ct, Cf, Cs = s17.M1.parameters(durations=[15])
    Ct, Cf = float(np.squeeze(Ct)), float(np.squeeze(Cf))
    R = float(np.squeeze(intensity.to_accumulation(i15_mmh, durations=[15])))

    def _solve(T_s, F_s):
        dX = (Ct * (T_s - T_obs[:, None]) + Cf * (F_s - F_obs[:, None])) * R
        bp = PDSIM_GRID[np.argmin(np.abs(dX), axis=1)]
        return bp, (float(np.median(bp[sel])) if sel.sum() >= 3 else None)

    best_p, fire_pdsim = _solve(T_sim, F_sim)

    # Per-table solves reuse W/Ws — only the per-class dNBR vector changes, so
    # the tables differ by nothing but their parameters.
    per_table, curves_extra = {}, {}
    for name, table in tables.items():
        if name == "default":
            per_table[name], curves_extra[f"best_p_{name}"] = fire_pdsim, best_p
            continue
        lam_t, kap_t = _params(table)
        D_t = severity.weibull_dnbr(PDSIM_GRID[:, None], lam_t[None, :], kap_t[None, :])
        bp_t, pd_t = _solve(Ws @ (D_t >= regional_break).T.astype(float), (W @ D_t.T) / 1000.0)
        per_table[name] = pd_t
        curves_extra[f"best_p_{name}"] = bp_t

    # ---- cache --------------------------------------------------------------
    np.savez_compressed(
        cache / "curves.npz", ids=segments.ids, T_obs=T_obs, F_obs=F_obs,
        T_sim=T_sim.astype(np.float32), F_sim=F_sim.astype(np.float32),
        pdsim_grid=PDSIM_GRID, best_p=best_p, sel=sel, area=area,
        ratio_in=ratio_in, med_dnbr=med_dnbr, **curves_extra)
    pd.DataFrame({"Segment_ID": segments.ids, "area_km2": area, "ratio_in": ratio_in,
                  "med_dnbr": med_dnbr, "T_obs": T_obs, "F_obs": F_obs,
                  "best_pdsim": best_p, "selected": sel}).to_parquet(cache / "basins.parquet")
    meta = {"event_id": event_id.upper(), "pdsim": fire_pdsim,
            "pdsim_by_table": per_table,
            "evt_tag": tag,
            "n_delineated": int(n0), "n_selected": int(sel.sum()),
            "low_t": float(low_t), "mod_t": float(mod_t),
            "regional_break": float(regional_break),
            "validation_dT": dT, "validation_dF": dF,
            "severity_source": "dnbr6" if "dnbr6" in bundle else "estimate(dnbr)",
            "solve_space": "logit (S cancels; KF-independent)"}
    (cache / "fire.json").write_text(json.dumps(meta, indent=2))
    if multi:
        return {name: FireCalib(event_id=event_id.upper(), pdsim=p,
                                n_delineated=int(n0), n_selected=int(sel.sum()),
                                validation_dT=dT, validation_dF=dF, cache_dir=cache)
                for name, p in per_table.items()}
    return FireCalib(event_id=event_id.upper(), pdsim=fire_pdsim, n_delineated=int(n0),
                     n_selected=int(sel.sum()), validation_dT=dT, validation_dF=dF,
                     cache_dir=cache)


def regional_summary(fire_calibs: list[FireCalib], *, region: str,
                     fire_breaks: dict[str, float], toml_path: Path,
                     cdf_source: str = "doi:10.5066/P9TKYL5K") -> dict:
    """Median fire P_dsims and breaks into a calibration TOML."""
    ok = [fc for fc in fire_calibs if fc.ok]
    pdsims = np.array([fc.pdsim for fc in ok])
    breaks = np.array([fire_breaks[fc.event_id] for fc in ok])
    regional = {
        "pdsim": float(np.median(pdsims)),
        "break_lowmod": float(np.median(breaks)),
        "n_fires": len(ok),
        "pdsim_iqr": [float(np.percentile(pdsims, 25)), float(np.percentile(pdsims, 75))],
    }
    from datetime import date

    lines = [
        "[meta]",
        f'name = "{toml_path.stem}"',
        f"created = {date.today().isoformat()}",
        'method = "rossi2025-dfl-calibration (logit-space solve)"',
        f'cdf_source = "{cdf_source}"',
        f'fires_used = {len(ok)}',
        "",
        f"[regions.{region}]",
        f"pdsim = {regional['pdsim']:.2f}",
        f"barc_breaks = [125.0, {regional['break_lowmod']:.0f}, 500.0]",
        "fires = [" + ", ".join(f'"{fc.event_id}"' for fc in ok) + "]",
    ]
    toml_path.parent.mkdir(parents=True, exist_ok=True)
    toml_path.write_text("\n".join(lines) + "\n")
    return regional
