"""Fan-deposit detection beyond the stream network (the design's fan zone).

The pfdf network deliberately stops at the valley mask, so the most
consequential part of a debris flow -- the lobe it leaves on the fan --
falls outside every corridor. This module builds the *fan zone* (low
height-above-channel, gentle slope, near the network) from the DEM and
finds compact change objects inside it.

v0 approximations, stated plainly:

* **HAND-lite**: height above the *euclidean*-nearest channel pixel, not
  the flow-path-nearest (true HAND needs flow routing). On range-front
  fans, where deposits sit within a few hundred metres of their feeder
  channel, the two agree; in complex valley bottoms they diverge.
* No down-gradient connectivity between fan objects and their feeder
  segment yet -- objects are filtered by geometry (zone, area) and
  evidence (z) only, so irrigation/grading inside the zone can still
  slip through.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: Fan-zone defaults (design: HAND < 2--5 m, slope 2--15 deg).
HAND_MAX_M = 5.0
SLOPE_MIN_DEG = 1.5
SLOPE_MAX_DEG = 15.0
CHANNEL_DIST_MAX_M = 500.0


def dem_and_slope(tiles, ref):
    """Elevation and slope (degrees) on the epoch grid, resampling once.

    Tiles are mosaicked on their native 1/3-arcsecond grid (lossless --
    same grid), slope is computed there with metric cell sizes, then
    elevation and slope are each warped ONCE, bilinear, to ``ref``. This
    honours the house resampling rules (never derive gradients after a
    second resample; never nearest for elevation).
    """
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import Resampling, reproject, transform_bounds

    tr = ref["transform"]
    bounds = rasterio.transform.array_bounds(ref["height"], ref["width"], tr)
    b4326 = transform_bounds(ref["crs"], "EPSG:4326", *bounds)
    pad = 0.02
    want = (b4326[0] - pad, b4326[1] - pad, b4326[2] + pad, b4326[3] + pad)

    srcs = []
    for t in tiles:
        with rasterio.open(t) as s:
            tb = s.bounds
        if not (tb.right < want[0] or tb.left > want[2] or
                tb.top < want[1] or tb.bottom > want[3]):
            srcs.append(rasterio.open(t))
    if not srcs:
        raise ValueError("no DEM tiles intersect the grid")
    dem_nat, tr_nat = merge(srcs, bounds=want, nodata=np.nan)
    crs_nat = srcs[0].crs
    for s in srcs:
        s.close()
    dem_nat = dem_nat[0].astype(np.float32)

    # metric cell sizes on the geographic native grid, at its mid-latitude
    lat = (want[1] + want[3]) / 2.0
    dy_m = abs(tr_nat.e) * 111_132.0
    dx_m = abs(tr_nat.a) * 111_320.0 * np.cos(np.radians(lat))
    gy, gx = np.gradient(dem_nat, dy_m, dx_m)
    slope_nat = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)

    out = []
    for arr in (dem_nat, slope_nat):
        dst = np.full((ref["height"], ref["width"]), np.nan, np.float32)
        reproject(arr, dst, src_transform=tr_nat, src_crs=crs_nat,
                  dst_transform=tr, dst_crs=ref["crs"],
                  resampling=Resampling.bilinear,
                  src_nodata=np.nan, dst_nodata=np.nan)
        out.append(dst)
    return out[0], out[1]


def channel_mask(segments, ref):
    """1-px rasterization of the network centrelines (bool)."""
    from rasterio import features

    shapes = [(g, 1) for g in segments.geometry
              if g is not None and not g.is_empty]
    return features.rasterize(
        shapes, out_shape=(ref["height"], ref["width"]),
        transform=ref["transform"], fill=0, all_touched=True,
        dtype="uint8").astype(bool)


def hand_lite(dem, channels, *, pixel_m: float):
    """Height above the euclidean-nearest channel pixel, plus that distance.

    Returns ``(hand_m, dist_m)``. See the module docstring for how this
    differs from true (flow-routed) HAND.
    """
    from scipy import ndimage

    if not channels.any():
        raise ValueError("channel mask is empty")
    dist_px, (iy, ix) = ndimage.distance_transform_edt(
        ~channels, return_indices=True)
    hand = dem - dem[iy, ix]
    return hand.astype(np.float32), (dist_px * pixel_m).astype(np.float32)


def fan_zone(hand, slope_deg, dist_m, *, hand_max=HAND_MAX_M,
             slope_min=SLOPE_MIN_DEG, slope_max=SLOPE_MAX_DEG,
             dist_max=CHANNEL_DIST_MAX_M):
    """Where a fan deposit is geometrically allowed to be."""
    return (np.nan_to_num(hand, nan=np.inf) < hand_max) \
        & (slope_deg >= slope_min) & (slope_deg <= slope_max) \
        & (dist_m <= dist_max)


def detect_fans(z, zone, ref, *, z_t: float = 2.5, min_area_px: int = 30):
    """Compact change objects inside the fan zone, as polygons.

    Thresholds the (background-centred) z field inside the zone, takes
    connected components, drops the small ones, polygonizes. Returns a
    GeoDataFrame (grid CRS) with area_m2, mean_z, max_z per object.
    """
    import geopandas as gpd
    import pandas as pd
    from rasterio import features
    from scipy import ndimage
    from shapely.geometry import shape as _shape

    zz = np.ma.masked_invalid(np.ma.asarray(z))
    hot = zone & ~np.ma.getmaskarray(zz) & (zz.filled(-np.inf) >= z_t)
    lab, n = ndimage.label(hot)
    if n == 0:
        return gpd.GeoDataFrame({"area_m2": [], "mean_z": [], "max_z": []},
                                geometry=[], crs=ref["crs"])
    sizes = np.bincount(lab.ravel())
    keep = np.flatnonzero(sizes >= min_area_px)
    keep = keep[keep > 0]
    px_area = abs(ref["transform"].a * ref["transform"].e)
    rows = []
    keepset = set(keep.tolist())
    remap = np.zeros(n + 1, np.int32)
    remap[list(keepset)] = list(keepset)
    lab = remap[lab]
    for geom, val in features.shapes(lab.astype(np.int32), mask=lab > 0,
                                     transform=ref["transform"]):
        val = int(val)
        m = lab == val
        rows.append({"label": val, "geometry": _shape(geom),
                     "area_m2": float(sizes[val] * px_area),
                     "mean_z": float(zz[m].mean()),
                     "max_z": float(zz[m].max())})
    g = gpd.GeoDataFrame(pd.DataFrame(rows), geometry="geometry",
                         crs=ref["crs"])
    # shapes() emits one polygon per contiguous ring; dissolve per label
    g = g.dissolve(by="label", aggfunc={"area_m2": "first",
                                        "mean_z": "first",
                                        "max_z": "first"}).reset_index(drop=True)
    return g.sort_values("area_m2", ascending=False).reset_index(drop=True)
