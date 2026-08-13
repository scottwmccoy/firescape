"""Statewide scale-out: per-unit windowed reads from local tile stores.

The pilot ran from one pre-warped DEM file. Statewide, the DEM lives as 57
3DEP 1/3-arcsecond tiles and EVT as LFPS chunks; this module mosaics the
pieces intersecting one hydrologic unit in memory and hands pfdf a Raster on
the working grid (EPSG:5070, 10 m, bilinear for elevation / nearest for
classes) — the same single-warp discipline stormscape's dem module enforces
(never resample elevation twice, never resample classes bilinearly).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

WORK_CRS = "EPSG:5070"
DEM_RES_M = 10.0


def _mosaic_to_grid(sources: list[Path], bounds5070, *, resolution: float,
                    resampling: str, dtype=None, pad_m: float = 300.0):
    """Composite every source intersecting ``bounds5070`` onto the working
    grid, reprojecting each source independently (LFPS delivers each chunk in
    its own Albers parameterization, so a shared-CRS merge is impossible).
    Each source is warped exactly once, native grid -> working grid. Returns
    (array, transform) or None if nothing intersects."""
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject, transform_bounds
    from rasterio.windows import from_bounds as window_from_bounds

    w, s, e, n = bounds5070
    w, s, e, n = w - pad_m, s - pad_m, e + pad_m, n + pad_m
    # snap the destination grid to the resolution so unit grids align
    w = np.floor(w / resolution) * resolution
    s = np.floor(s / resolution) * resolution
    e = np.ceil(e / resolution) * resolution
    n = np.ceil(n / resolution) * resolution
    width = int(round((e - w) / resolution))
    height = int(round((n - s) / resolution))
    dst_transform = from_origin(w, n, resolution, resolution)
    fillv = np.nan if dtype is None else 0
    dst = np.full((height, width), fillv, dtype="float32" if dtype is None else dtype)

    hit = False
    for p in sources:
        with rasterio.open(p) as ds:
            try:
                tb = transform_bounds(ds.crs, WORK_CRS, *ds.bounds)
            except Exception:
                continue
            if not (tb[0] < e and tb[2] > w and tb[1] < n and tb[3] > s):
                continue
            # read only the window covering the unit (plus margin)
            sw_, ss_, se_, sn_ = transform_bounds(WORK_CRS, ds.crs, w, s, e, n)
            win = window_from_bounds(sw_, ss_, se_, sn_, ds.transform)
            win = win.round_offsets().round_lengths()
            try:  # raises WindowError when the overlap is empty
                win = win.intersection(rasterio.windows.Window(0, 0, ds.width, ds.height))
            except rasterio.errors.WindowError:
                continue
            if win.width <= 0 or win.height <= 0:
                continue
            arr = ds.read(1, window=win)
            piece = np.full_like(dst, fillv)
            reproject(arr, piece,
                      src_transform=ds.window_transform(win), src_crs=ds.crs,
                      src_nodata=ds.nodata,
                      dst_transform=dst_transform, dst_crs=WORK_CRS,
                      dst_nodata=fillv,
                      resampling=getattr(Resampling, resampling))
            empty = np.isnan(dst) if dtype is None else (dst == fillv)
            have = ~np.isnan(piece) if dtype is None else (piece != fillv)
            put = empty & have
            dst[put] = piece[put]
            hit = True
    if not hit:
        return None
    return dst, dst_transform


def dem_for_unit(bounds5070, tile_dir: Path):
    """3DEP DEM for a unit from local 1-degree tiles -> pfdf Raster (5070/10 m).

    Bilinear, warped exactly once from the native 1/3-arcsecond grid — the
    resampling discipline validated in stormscape (nearest halves plan
    curvature; double-resampling smears derivatives).
    """
    from pfdf.raster import Raster

    tiles = sorted(Path(tile_dir).glob("USGS_13_*.tif"))
    out = _mosaic_to_grid(tiles, bounds5070, resolution=DEM_RES_M,
                          resampling="bilinear")
    if out is None:
        raise FileNotFoundError("no DEM tiles intersect the unit")
    arr, transform = out
    return Raster.from_array(arr, nodata=np.float32(np.nan), crs=WORK_CRS,
                             transform=transform)


def evt_for_unit(bounds5070, chunk_dir: Path, *, resolution: float = 30.0):
    """LANDFIRE EVT for a unit from staged LFPS chunks -> pfdf Raster (nearest)."""
    from pfdf.raster import Raster

    chunks = sorted(Path(chunk_dir).glob("*/evt_*.tif"))
    out = _mosaic_to_grid(chunks, bounds5070, resolution=resolution,
                          resampling="nearest", dtype="int32")
    if out is None:
        raise FileNotFoundError("no EVT chunks intersect the unit")
    arr, transform = out
    return Raster.from_array(arr, nodata=0, crs=WORK_CRS, transform=transform)


def bounds4326(bounds5070):
    """(W, S, E, N) of a 5070 bounds tuple in lon/lat, for bbox file reads."""
    from rasterio.warp import transform_bounds

    return transform_bounds(WORK_CRS, "EPSG:4326", *bounds5070)
