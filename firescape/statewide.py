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
    """Mosaic the tiles intersecting ``bounds5070`` and warp once to the
    working grid. Returns (array, transform) or None if no tiles intersect."""
    import rasterio
    from rasterio.merge import merge as rio_merge
    from rasterio.warp import (Resampling, calculate_default_transform,
                               reproject, transform_bounds)

    w, s, e, n = bounds5070
    w, s, e, n = w - pad_m, s - pad_m, e + pad_m, n + pad_m

    datasets = []
    for p in sources:
        ds = rasterio.open(p)
        try:
            tb = transform_bounds(ds.crs, WORK_CRS, *ds.bounds)
        except Exception:
            ds.close()
            continue
        if tb[0] < e and tb[2] > w and tb[1] < n and tb[3] > s:
            datasets.append(ds)
        else:
            ds.close()
    if not datasets:
        return None
    try:
        src_crs = datasets[0].crs
        src_bounds = transform_bounds(WORK_CRS, src_crs, w, s, e, n)
        arr, src_transform = rio_merge(datasets, bounds=src_bounds)
        arr = arr[0]
        nodata = datasets[0].nodata
    finally:
        for ds in datasets:
            ds.close()

    dst_transform, width, height = calculate_default_transform(
        src_crs, WORK_CRS, arr.shape[1], arr.shape[0], *src_bounds,
        resolution=(resolution, resolution))
    dst = np.full((height, width), np.nan if dtype is None else 0,
                  dtype="float32" if dtype is None else dtype)
    reproject(arr, dst, src_transform=src_transform, src_crs=src_crs,
              src_nodata=nodata, dst_transform=dst_transform, dst_crs=WORK_CRS,
              dst_nodata=np.nan if dtype is None else 0,
              resampling=getattr(Resampling, resampling))
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
