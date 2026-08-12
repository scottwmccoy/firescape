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


def _kf_raster(dem, cfg: PreFireConfig, unit_key: str):
    """Real STATSGO KF if reachable (cached per unit), else flagged constant."""
    from pfdf.raster import Raster

    cache_dir = cfg.kf_cache_dir or paths.interim_dir("kf_cache")
    cache = Path(cache_dir) / f"{unit_key}_kf.tif"
    if cache.exists():
        return match_grid(Raster.from_file(cache), dem, resampling="nearest"), "statsgo-cached"
    try:
        kf = soils.kf_factor(dem)
        try:
            kf.save(cache, overwrite=True)
        except Exception:
            pass
        return match_grid(kf, dem, resampling="nearest"), "statsgo"
    except Exception as e:  # ScienceBase outage -> provisional constant
        arr = np.full(dem.shape, KF_FALLBACK, dtype="float32")
        return (Raster.from_array(arr, spatial=dem, nodata=np.float32(np.nan)),
                f"CONSTANT-{KF_FALLBACK}-PROVISIONAL ({type(e).__name__})")


def run_unit(unit_geom, cfg: PreFireConfig, *, unit_key: str, crs=None) -> dict:
    """Compute the pre-fire hazard surface for one unit.

    ``unit_geom``: shapely geometry (in ``crs``, default the DEM's CRS).
    Returns a dict of per-segment arrays, the Segments object, and metadata.
    """
    import geopandas as gpd
    from pfdf.projection import BoundingBox
    from pfdf.raster import Raster

    # --- clip DEM to the unit ------------------------------------------------
    import rasterio

    with rasterio.open(cfg.dem_path) as src:
        dem_crs = src.crs
    gseries = gpd.GeoSeries([unit_geom], crs=crs or dem_crs).to_crs(dem_crs)
    w, s, e, n = gseries.total_bounds
    dem = Raster.from_file(cfg.dem_path, bounds=BoundingBox(w, s, e, n, crs=dem_crs))

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
    evt = match_grid(Raster.from_file(cfg.evt_path, bounds=dem.bounds), dem,
                     resampling="nearest")
    evt_values = severity.apply_crosswalk(evt.values, cfg.crosswalk)
    cdf = severity.load_cdf_table()
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
    meta = {
        "unit_key": unit_key,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "region": cfg.region,
        "pdsim": cfg.pdsim,
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
