"""Pre-fire hazard surface: the M4 product (Rossi et al. 2025, Fig. 5).

``run_unit`` is the sole compute entry point and the seam that carries this
tool from the pilot to the whole state: one call = one hydrologic unit (or any
AOI). The pilot is a loop over its HU-10 watersheds; statewide is the same
loop over more of them.

Differences from the observed-severity assessment (assess.run_observed):

- Severity is *simulated* everywhere from LANDFIRE EVT at the calibrated
  P_dsim — there is no perimeter, the whole unit is treated as burned.
- Delineation is limited by the Rossi valley/sink/water masks instead of a
  burn perimeter, and filtering uses the basin-area criteria only
  (0.025-8 km^2), matching Rossi's described statewide method. The
  burn-ratio / slope / confinement filters are USGS *post*-fire assessment
  conventions and are left off by default here (``strict_filters=True``
  applies them).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from firescape import hazard as hz
from firescape import paths, severity, soils
from firescape.config import FilterDefaults, I15_REFERENCE_MMH
from firescape.delineate import (match_grid, network, rossi_masks, terrain,
                                 water_codes_from_legend)

#: Placeholder used only when STATSGO cannot be reached (flagged in run_meta).
KF_FALLBACK = 0.25

_KF_POLY_CACHE: dict = {}


def _load_kf_polygons(path):
    """Read (once per process) a soil-polygon layer carrying a ``kf`` column."""
    import geopandas as gpd

    key = str(path)
    if key not in _KF_POLY_CACHE:
        _KF_POLY_CACHE[key] = gpd.read_file(path)
    return _KF_POLY_CACHE[key]


@dataclass
class PreFireConfig:
    pdsim: float
    barc_breaks: tuple[float, float, float]
    evt_path: Path
    evt_legend_path: Path
    crosswalk: dict[int, int]
    dem_path: Path
    i15_mmh: float = I15_REFERENCE_MMH
    threshold_p: float = 0.5
    filters: FilterDefaults = field(default_factory=FilterDefaults)
    strict_filters: bool = False
    kf_cache_dir: Path | None = None
    region: str = "pilot"
    #: packaged CDF table name (see severity.PACKAGED_TABLES)
    cdf_table: str = "staley2018"
    #: pre-fetched soil polygons with a ``kf`` column; used instead of STATSGO
    kf_polygons: Path | None = None
    #: dispersed-quantile severity closure (None = deterministic). The
    #: measured value is severity.SIGMA_Z_DEFAULT = 0.91.
    severity_sigma: float | None = None
    #: emit per-basin SlopeDeg/FracNorth (RANGES volume predictors)
    emit_ranges_topo: bool = False


def _kf_raster(dem, cfg: PreFireConfig, unit_key: str):
    """KF raster for a unit (delegates to the shared soils.kf_raster chain:
    supplied soil polygons, else STATSGO cached, else a flagged constant)."""
    return soils.kf_raster(dem, polygons=cfg.kf_polygons,
                           cache_dir=cfg.kf_cache_dir, key=unit_key)


def run_unit(unit_geom, cfg: PreFireConfig, *, unit_key: str, crs=None,
             dem=None, evt=None) -> dict:
    """Compute the pre-fire hazard surface for one unit.

    ``unit_geom``: shapely geometry (in ``crs``, default the DEM's CRS).
    ``dem``/``evt``: optional pre-built pfdf Rasters (statewide tile-store
    reads); when omitted they load from cfg.dem_path / cfg.evt_path.
    Returns a dict of per-segment arrays, the Segments object, and metadata.
    """
    import geopandas as gpd
    from pfdf.projection import BoundingBox
    from pfdf.raster import Raster

    # --- clip DEM to the unit ------------------------------------------------
    if dem is None:
        import rasterio

        with rasterio.open(cfg.dem_path) as src:
            dem_crs = src.crs
        gseries = gpd.GeoSeries([unit_geom], crs=crs or dem_crs).to_crs(dem_crs)
        w, s, e, n = gseries.total_bounds
        dem = Raster.from_file(cfg.dem_path, bounds=BoundingBox(w, s, e, n, crs=dem_crs))
    else:
        dem_crs = dem.crs
        gseries = gpd.GeoSeries([unit_geom], crs=crs or dem_crs).to_crs(dem_crs)

    # Delineation is confined to the unit polygon so adjacent units cannot
    # produce duplicate basins. HU boundaries follow drainage divides, so a
    # catchment's upslope area stays inside its own unit.
    unit_path = paths.cache_root() / f"_unit_{unit_key}.geojson"
    gpd.GeoDataFrame(geometry=list(gseries), crs=dem_crs).to_file(unit_path, driver="GeoJSON")
    try:
        res = dem.resolution("meters")
    except TypeError:
        res = dem.resolution
    in_unit = match_grid(
        Raster.from_polygons(unit_path, bounds=dem, resolution=res), dem
    ).values.astype(bool)
    unit_path.unlink(missing_ok=True)

    # --- simulated severity --------------------------------------------------
    if evt is None:
        evt = Raster.from_file(cfg.evt_path, bounds=dem.bounds)
    evt = match_grid(evt, dem, resampling="nearest")
    evt_values = severity.apply_crosswalk(evt.values, cfg.crosswalk)
    cdf = severity.load_cdf_table(table=cfg.cdf_table)
    sim_dnbr, src_flag = severity.simulate_dnbr(evt_values, cfg.pdsim, cdf)
    sim_barc = severity.classify_barc4(sim_dnbr, cfg.barc_breaks)

    # --- masks + network -----------------------------------------------------
    terr = terrain(dem)
    legend = gpd.read_file(cfg.evt_legend_path)
    masks = rossi_masks(dem, flow=terr.flow, evt=evt,
                        evt_water_codes=water_codes_from_legend(legend))
    domain = Raster.from_array(~masks["exclude"] & in_unit, spatial=dem, isbool=True)
    segments, _terr2 = network(dem, domain, min_area_km2=cfg.filters.min_area_km2,
                               max_length_m=cfg.filters.max_length_m)
    n0 = 0 if segments is None else segments.size

    def _empty(note: str) -> dict:
        return {"unit_key": unit_key, "segments": segments, "n_segments": 0, "props": {},
                "meta": {"unit_key": unit_key, "note": note,
                         "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         "segments_delineated": int(n0), "segments_kept": 0,
                         "mask_fraction": {k: float(v.mean()) for k, v in masks.items()},
                         "dem_shape": list(dem.shape)}}

    # Units that are almost entirely lake, playa, or valley floor (the Rossi
    # masks can exclude >95% of such a unit) legitimately yield no segments.
    if n0 == 0:
        return _empty("no segments delineated (unit fully masked or below min area)")

    area = np.asarray(segments.area(units="kilometers"), dtype=float)
    keep = area <= cfg.filters.max_area_km2
    if cfg.strict_filters:
        keep &= (np.asarray(segments.slope(terr.slopes), dtype=float)
                 >= cfg.filters.min_slope)
        keep &= (np.asarray(segments.confinement(terr.conditioned, 4), dtype=float)
                 <= cfg.filters.max_confinement_deg)
    if not keep.any():
        return _empty("no segments passed filtering")
    segments.keep(np.asarray(segments.continuous(keep), dtype=bool))
    if segments.size == 0:
        return _empty("no segments after flow-continuity filtering")
    segments.locate_basins()

    # --- models --------------------------------------------------------------
    kf, kf_source = _kf_raster(dem, cfg, unit_key)
    barc_r = Raster.from_array(sim_barc, spatial=dem, nodata=0)
    dnbr_r = Raster.from_array(sim_dnbr.astype("float32"), spatial=dem,
                               nodata=np.float32(np.nan))
    T, F, S = hz.m1_inputs(segments, barc_r, terr.slopes, dnbr_r, kf, omitnan=True)
    p = hz.likelihood_m1(T, F, S, i15_mmh=cfg.i15_mmh)
    from pfdf import severity as pfsev

    modhigh = pfsev.mask(barc_r, ["moderate", "high"])
    bmh = np.asarray(segments.burned_area(modhigh, units="kilometers"), dtype=float)
    relief = np.asarray(segments.relief(terr.relief), dtype=float)
    V, Vmin, Vmax = hz.volume_g14(bmh, relief, i15_mmh=cfg.i15_mmh)
    hclass = hz.combined_c10(p, V)
    thresh = hz.threshold_i15(T, F, S, p=cfg.threshold_p)

    severity_mode = "deterministic"
    if cfg.severity_sigma:
        # Dispersed-quantile closure: per-class break-exceedance probability
        # and expected dNBR replace the 0/1 indicator and the constant class
        # value (severity.class_exceedance/expected_dnbr; sigma measured from
        # observed severity). S is severity-independent and keeps the
        # M1.variables value from above; T/F/Bmh become expectations via the
        # same class decomposition the calibration validates exactly.
        import pandas as pd

        from firescape.calibrate import STEEP_GRADIENT

        severity_mode = f"dispersed(sigma={cfg.severity_sigma})"
        vals = np.where(domain.values, evt_values, -1)
        classes = [int(c) for c in np.unique(vals) if c > 0]
        fbcode = severity.BARREN_EVT_CODE
        lam = cdf["Weibull_Lambda_Scale"].reindex(classes).fillna(
            cdf.at[fbcode, "Weibull_Lambda_Scale"]).to_numpy(float)
        kap = cdf["Weibull_Kappa_Shape"].reindex(classes).fillna(
            cdf.at[fbcode, "Weibull_Kappa_Shape"]).to_numpy(float)
        tab = pd.DataFrame({"Weibull_Lambda_Scale": lam,
                            "Weibull_Kappa_Shape": kap}, index=classes)
        p_c = severity.class_exceedance(classes, cfg.barc_breaks[1], cfg.pdsim,
                                        tab, sigma=cfg.severity_sigma)
        e_c = severity.expected_dnbr(classes, cfg.pdsim, tab,
                                     sigma=cfg.severity_sigma)
        steep = terr.slopes.values >= STEEP_GRADIENT
        W = np.zeros((segments.size, len(classes)))
        Ws = np.zeros_like(W)
        for j, c in enumerate(classes):
            in_c = vals == c
            W[:, j] = np.asarray(segments.catchment_ratio(
                Raster.from_array(in_c, spatial=dem, isbool=True)), float)
            Ws[:, j] = np.asarray(segments.catchment_ratio(
                Raster.from_array(in_c & steep, spatial=dem, isbool=True)),
                float)
        T = Ws @ p_c
        F = W @ e_c / 1000.0
        p = hz.likelihood_m1(T, F, S, i15_mmh=cfg.i15_mmh)
        bmh = (np.asarray(segments.area(units="kilometers"), dtype=float)
               * (W @ p_c))
        V, Vmin, Vmax = hz.volume_g14(bmh, relief, i15_mmh=cfg.i15_mmh)
        hclass = hz.combined_c10(p, V)
        thresh = hz.threshold_i15(T, F, S, p=cfg.threshold_p)

    ranges_topo = {}
    if cfg.emit_ranges_topo:
        # RANGES volume predictors, computed on the unit DEM at native res
        try:
            dx, dy = float(res[0]), float(res[1])
        except (TypeError, IndexError):
            dx = dy = float(res)
        z = np.asarray(dem.values, dtype="float64")
        gr, gc = np.gradient(z, dy, dx)
        sdeg = np.degrees(np.arctan(np.hypot(gr, gc))).astype("float32")
        aspect = np.degrees(np.arctan2(-gc, gr)) % 360.0
        north = (aspect >= 315.0) | (aspect < 45.0)
        ranges_topo["SlopeDeg"] = np.asarray(segments.catchment_summary(
            "mean", Raster.from_array(sdeg, spatial=dem,
                                      nodata=np.float32(np.nan))), float)
        ranges_topo["FracNorth"] = np.asarray(segments.catchment_ratio(
            Raster.from_array(north, spatial=dem, isbool=True)), float)

    tag = f"{int(round(cfg.i15_mmh))}mmh"
    props = {
        "Segment_ID": segments.ids.astype(float),
        "Area_km2": area[: segments.size] if area.size == segments.size else
                    np.asarray(segments.area(units="kilometers"), dtype=float),
        "Slope": np.asarray(segments.slope(terr.slopes), dtype=float),
        "Relief_m": relief,
        "Bmh_km2": bmh,
        "Terrain_M1": np.asarray(T, dtype=float),
        "Fire_M1": np.asarray(F, dtype=float),
        "Soil_M1": np.asarray(S, dtype=float),
        f"P_{tag}": np.asarray(p, dtype=float),
        f"V_{tag}": np.asarray(V, dtype=float),
        f"H_{tag}": np.asarray(hclass, dtype=float),
        f"I15_{int(cfg.threshold_p * 100)}": np.asarray(thresh, dtype=float),
    }
    for k, v in ranges_topo.items():
        props[k] = v
    meta = {
        "unit_key": unit_key,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "region": cfg.region,
        "pdsim": cfg.pdsim,
        "severity_mode": severity_mode,
        "cdf_table": cfg.cdf_table,
        "barc_breaks": list(cfg.barc_breaks),
        "i15_mmh": cfg.i15_mmh,
        "threshold_p": cfg.threshold_p,
        "kf_source": kf_source,
        "strict_filters": cfg.strict_filters,
        "segments_delineated": int(n0),
        "segments_kept": int(segments.size),
        "mask_fraction": {k: float(v.mean()) for k, v in masks.items()},
        "evt_fallback_fraction": float((src_flag == severity.SRC_FALLBACK).mean()),
        "dem_shape": list(dem.shape),
    }
    return {"unit_key": unit_key, "segments": segments, "n_segments": int(segments.size),
            "props": props, "meta": meta,
            "rasters": {"sim_dnbr": sim_dnbr, "sim_barc": sim_barc, "masks": masks,
                        "dem": dem}}


def save_unit(result: dict, out_dir: Path) -> dict:
    """Write one unit's basins/segments GPKG + run metadata."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    key = result["unit_key"]
    paths_out = {}
    if result["n_segments"]:
        segs = result["segments"]
        for kind, suffix in (("segments", "segments"), ("basins", "basins")):
            p = out_dir / f"{key}_{suffix}.gpkg"
            segs.save(p, type=kind, properties=result["props"], overwrite=True)
            paths_out[kind] = p
    (out_dir / f"{key}_meta.json").write_text(json.dumps(result["meta"], indent=2) + "\n")
    return paths_out
