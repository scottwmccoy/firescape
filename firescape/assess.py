"""M1 walking skeleton: hazard assessment of one MTBS fire, observed severity.

Chain (USGS standard, pfdf-backed): DEM -> condition/flow/slopes/relief ->
Segments (accum >= 0.025 km^2 within buffered perimeter, max 500 m) ->
USGS physical filters -> severity from observed MTBS dnbr6 -> M1 likelihood
at the reference storm -> Gartner 2014 volume -> Cannon 2010 combined class ->
rainfall-threshold inversion -> wildcat-schema GPKG + map.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from firescape import mtbs, paths, soils
from firescape.config import FilterDefaults, I15_REFERENCE_MMH
from firescape.delineate import match_grid, network, usgs_filter
from firescape import hazard as hz

PERIMETER_BUFFER_M = 3000.0  # analysis domain = perimeter + buffer (wildcat-style)


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent.parent), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or None
    except Exception:
        return None


def run_observed(event_id: str, *, dem_path: Path | None = None, dem=None,
                 i15_mmh: float = I15_REFERENCE_MMH, out_dir: Path | None = None,
                 threshold_p: float = 0.5) -> dict:
    """Run the observed-severity hazard chain for one downloaded MTBS fire.

    ``dem`` injects a pre-built pfdf Raster covering the perimeter plus
    ``PERIMETER_BUFFER_M`` (the statewide tile-store seam, mirror of
    prefire.run_unit); ``dem_path`` remains the pilot route.

    Returns a dict with the Segments object, per-segment arrays, and paths of
    everything written.
    """
    import geopandas as gpd
    from pfdf import severity as pfdf_severity
    from pfdf.raster import Raster

    filters = FilterDefaults()
    bundle = mtbs.fire_bundle(event_id)
    out_dir = Path(out_dir) if out_dir else paths.products_dir("assess", event_id.upper())

    # --- domain: burn perimeter (+buffer) on the DEM grid -------------------
    perim_gdf = gpd.read_file(bundle["burn_area"])
    import rasterio

    if dem is None:
        dem_path = Path(dem_path) if dem_path else paths.interim_dir("pilot") / "pilot_dem.tif"
        with rasterio.open(dem_path) as src:
            dem_crs = src.crs
    else:
        dem_crs = dem.crs
    perim_gdf = perim_gdf.to_crs(dem_crs)
    domain_gdf = gpd.GeoDataFrame(geometry=[perim_gdf.union_all().buffer(PERIMETER_BUFFER_M)],
                                  crs=dem_crs)

    from pfdf.projection import BoundingBox

    w, s, e, n = domain_gdf.total_bounds
    if dem is None:
        dem = Raster.from_file(dem_path, bounds=BoundingBox(w, s, e, n, crs=dem_crs))
    try:
        res = dem.resolution(units="meters")
    except TypeError:
        res = dem.resolution
    perim = Raster.from_polygons(bundle["burn_area"], bounds=dem, resolution=res)
    perim = match_grid(perim, dem)
    domain_path = out_dir / "_domain.geojson"
    out_dir.mkdir(parents=True, exist_ok=True)
    domain_gdf.to_file(domain_path, driver="GeoJSON")
    domain = Raster.from_polygons(domain_path, bounds=dem, resolution=res)
    domain = match_grid(domain, dem)

    # --- network ------------------------------------------------------------
    segments, terr = network(dem, domain)
    n0 = segments.size

    # --- severity (observed dnbr6 -> BARC4) --------------------------------
    dnbr = match_grid(Raster.from_file(bundle["dnbr"]), dem, resampling="bilinear")
    if "dnbr6" in bundle:
        dnbr6 = match_grid(Raster.from_file(bundle["dnbr6"]), dem, resampling="nearest")
        barc4_arr = mtbs.dnbr6_to_barc4(dnbr6.values)
    else:  # BAER-delivered fires: estimate from dNBR with standard thresholds
        est = pfdf_severity.estimate(dnbr)
        barc4_arr = est.values.astype(np.uint8)
    barc4 = Raster.from_array(barc4_arr, spatial=dem, nodata=0)
    burned = pfdf_severity.mask(barc4, ["low", "moderate", "high"])
    moderate_high = pfdf_severity.mask(barc4, ["moderate", "high"])

    # --- filters ------------------------------------------------------------
    usgs_filter(segments, burned=burned, perimeter=perim, slopes=terr.slopes,
                dem_conditioned=terr.conditioned, filters=filters)
    segments.locate_basins()

    # --- soils (disk-cached: ScienceBase is intermittently unavailable) -----
    kf_cache = paths.interim_dir("assess_cache", event_id.upper()) / "kf.tif"
    if kf_cache.exists():
        kf = Raster.from_file(kf_cache)
    else:
        kf = soils.kf_factor(dem)
        try:
            kf.save(kf_cache, overwrite=True)
        except Exception:
            pass
    kf = match_grid(kf, dem, resampling="nearest")

    # --- models -------------------------------------------------------------
    T, F, S = hz.m1_inputs(segments, barc4, terr.slopes, dnbr, kf, omitnan=True)
    p = hz.likelihood_m1(T, F, S, i15_mmh=i15_mmh)
    bmh = segments.burned_area(moderate_high, units="kilometers")
    relief = segments.relief(terr.relief)
    V, Vmin, Vmax = hz.volume_g14(bmh, relief, i15_mmh=i15_mmh)
    hclass = hz.combined_c10(p, V)
    i15_thresh = hz.threshold_i15(T, F, S, p=threshold_p)

    # --- outputs (wildcat-style property schema) ---------------------------
    tag = f"{int(round(i15_mmh))}mmh"
    props = {
        "Segment_ID": segments.ids.astype(float),
        "Area_km2": np.asarray(segments.area(units="kilometers"), dtype=float),
        "BurnRatio": np.asarray(segments.burn_ratio(burned), dtype=float),
        "Slope": np.asarray(segments.slope(terr.slopes), dtype=float),
        "Relief_m": np.asarray(relief, dtype=float),
        "Bmh_km2": np.asarray(bmh, dtype=float),
        "Terrain_M1": np.asarray(T, dtype=float),
        "Fire_M1": np.asarray(F, dtype=float),
        "Soil_M1": np.asarray(S, dtype=float),
        f"P_{tag}": np.asarray(p, dtype=float),
        f"V_{tag}": np.asarray(V, dtype=float),
        f"Vmin_{tag}": np.asarray(Vmin, dtype=float),
        f"Vmax_{tag}": np.asarray(Vmax, dtype=float),
        f"H_{tag}": np.asarray(hclass, dtype=float),
        f"I15_{int(threshold_p * 100)}": np.asarray(i15_thresh, dtype=float),
    }
    seg_path = out_dir / f"{event_id.upper()}_segments.gpkg"
    basin_path = out_dir / f"{event_id.upper()}_basins.gpkg"
    segments.save(seg_path, type="segments", properties=props, overwrite=True)
    segments.save(basin_path, type="basins", properties=props, overwrite=True)

    meta = {
        "event_id": event_id.upper(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "firescape_git_sha": _git_sha(),
        "severity_source": "observed dnbr6" if "dnbr6" in bundle else "severity.estimate(observed dnbr)",
        "i15_mmh": i15_mmh,
        "threshold_p": threshold_p,
        "filters": vars(filters) | {"note": "no developed-area filter (M1 scope)"},
        "perimeter_buffer_m": PERIMETER_BUFFER_M,
        "segments_initial": int(n0),
        "segments_kept": int(segments.size),
        "inputs": {k: str(v) for k, v in bundle.items()},
        "dem": str(dem_path),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    return {"segments": segments, "props": props, "meta": meta,
            "paths": {"segments": seg_path, "basins": basin_path, "out_dir": out_dir},
            "rasters": {"dem": dem, "barc4": barc4, "dnbr": dnbr, "kf": kf,
                        "slopes": terr.slopes, "relief": terr.relief,
                        "perimeter": perim}}


def shootout(event_id: str, calibrations: dict[str, tuple[float, tuple[float, float, float]]],
             *, evt_layer: str, lfps_email: str | None = None,
             evt_raster=None, crosswalk: dict[int, int] | None = None,
             i15_mmh: float = I15_REFERENCE_MMH) -> dict:
    """M2 shootout: simulated vs observed severity on the SAME segment network.

    Runs the observed chain once (delineation + filters use observed severity,
    per Rossi), then for each named calibration (pdsim, barc_breaks) simulates
    dNBR from the EVT layer and recomputes M1 likelihood on the same segments.
    Returns per-basin arrays, fire-wide stats, and the EVT coverage report.
    """
    from pfdf.raster import Raster

    from firescape import landfire, severity
    from firescape.delineate import match_grid

    obs = run_observed(event_id, i15_mmh=i15_mmh)
    segments = obs["segments"]
    dem = obs["rasters"]["dem"]
    slopes = obs["rasters"]["slopes"]
    kf = obs["rasters"]["kf"]
    p_obs = obs["props"][f"P_{int(round(i15_mmh))}mmh"]

    if evt_raster is None:
        evt_raster = match_grid(
            landfire.evt(dem.bounds, evt_layer, email=lfps_email), dem, resampling="nearest"
        )
    cdf = severity.load_cdf_table()
    evt_values = evt_raster.values
    if crosswalk:
        evt_values = severity.apply_crosswalk(evt_values, crosswalk)
    coverage = severity.coverage_report(
        evt_values[obs["rasters"]["perimeter"].values.astype(bool)], cdf
    )

    runs = {}
    for name, (pdsim, breaks) in calibrations.items():
        sim_dnbr, src = severity.simulate_dnbr(evt_values, pdsim, cdf)
        sim_barc = severity.classify_barc4(sim_dnbr, breaks)
        dnbr_r = Raster.from_array(sim_dnbr, spatial=dem, nodata=np.nan)
        barc_r = Raster.from_array(sim_barc, spatial=dem, nodata=0)
        T, F, S = hz.m1_inputs(segments, barc_r, slopes, dnbr_r, kf, omitnan=True)
        p_sim = hz.likelihood_m1(T, F, S, i15_mmh=i15_mmh)
        ok = np.isfinite(p_obs) & np.isfinite(p_sim)
        resid = p_sim[ok] - p_obs[ok]
        denom = np.sum((p_obs[ok] - p_obs[ok].mean()) ** 2)
        runs[name] = {
            "pdsim": pdsim, "breaks": breaks, "p_sim": p_sim,
            "sim_dnbr": sim_dnbr, "src": src,
            "mean_obs": float(p_obs[ok].mean()), "mean_sim": float(p_sim[ok].mean()),
            "rmse": float(np.sqrt(np.mean(resid ** 2))),
            "nse": float(1 - np.sum(resid ** 2) / denom) if denom > 0 else float("nan"),
            "n": int(ok.sum()),
        }
    return {"observed": obs, "p_obs": p_obs, "runs": runs,
            "coverage": coverage, "evt_layer": evt_layer, "evt": evt_raster}
