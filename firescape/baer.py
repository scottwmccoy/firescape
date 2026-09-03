"""BAER (Burned Area Emergency Response) Soil Burn Severity access.

Endpoint validated 2026-09-03: the USDA Forest Service's national SBS mosaic
has moved at least once (``apps.fs.usda.gov/fsgisx01/.../ImageServer`` now
502s with "migrated to IIPP") -- the live copy is

    https://imagery.geoplatform.gov/iipp/rest/services/Fire_Aviation/
    USFS_EDW_BAER_SoilBurnSeverityClassification/ImageServer

An ArcGIS mosaic ImageServer over ~820 per-fire (or per-complex) SBS rasters,
2017-present, native 20-30 m depending on source imagery (Sentinel-2 vs
Landsat). Like ScienceBase, it 403s a plain ``requests`` fetch -- use a
browser User-Agent (see ``UA`` below).

Two operations:

- ``footprints()`` -- query the mosaic's catalog table (``/query``) for which
  fires are covered, where, and when. ``category=1`` means a primary source
  raster; ``category=2`` are pre-built multi-fire overview mosaics (e.g.
  ``"BAER_SBS_2018"``) and should normally be excluded -- they inflate area
  and swallow the per-fire ``name`` field.
- ``fetch_aligned()`` -- pull the classified raster on an EXACT caller-given
  grid (``exportImage`` with matching bbox/size/CRS), so it lines up
  pixel-for-pixel with whatever it's being compared against. No resampling
  step needed downstream.

**Pixel values** (confirmed empirically against the legend on the WILDCAT
fire, NV4164611529420220713, 2026-09-03): 1=unburned-to-very-low, 2=low,
3=moderate, 4=high, 5=masked (developed). This happens to match
``firescape.severity.classify_barc4``'s own 1..4 numbering -- no remap
needed when comparing the two directly.

**Always lock the mosaic to the tile you matched.** The service composites
overlapping assessments with ``mosaicOperator: "First"``, so an
``exportImage`` over a fire's window returns whichever tile the default rule
picks -- not necessarily the one ``match_perimeters`` chose. Where BAER
assessments overlap (California reburns; the August Complex 2020 tile alone
covers 5,433 km2 of the Northern Coast Ranges) that silently substitutes a
DIFFERENT fire's severity map, and the result looks like a real but weak
comparison rather than a wrong one. On the BUCK 2017 fire the default mosaic
gave dNBR medians of 13/2/-7/5 across BAER's four classes -- no burn signal
at all -- while locking to ``buck_sbs`` gave 81/173/355/559 and moved
Youden's J from 0.05 to 0.47. Pass ``lock_raster_id``.

**Do NOT pass ``noData`` to exportImage.** The service's true background
code is NOT 0 -- it was 15 on the fire tested, and forcing ``noData=0``
silently merges real "unburned to very low" pixels (also coded close to the
low end) into the background instead of leaving them classified. Fetch raw,
then mask anything outside ``{1, 2, 3, 4, 5}``.
"""

from __future__ import annotations

import numpy as np
import rasterio
import requests

from firescape.paths import BROWSER_UA as UA

BASE = ("https://imagery.geoplatform.gov/iipp/rest/services/Fire_Aviation/"
        "USFS_EDW_BAER_SoilBurnSeverityClassification/ImageServer")

#: Valid classified codes. 5 (masked/developed) is real but should usually be
#: excluded from a severity comparison, same as firescape excludes developed
#: area from the hazard chain elsewhere.
CLASS_CODES = {1: "unburned", 2: "low", 3: "moderate", 4: "high", 5: "masked"}


def footprints(bounds4326, *, primary_only: bool = True, timeout: float = 60.0):
    """Catalog items (fire name, year, footprint geometry) intersecting bounds.

    ``bounds4326``: (w, s, e, n) lon/lat. Returns a GeoDataFrame in EPSG:4326
    with ``name, beginyear, endyear, category`` plus ``area_km2``. Geometry
    is the SBS analysis footprint, which is normally larger than the fire
    perimeter (buffer/watershed context) -- match against a fire perimeter by
    intersection fraction, not equality.
    """
    import geopandas as gpd
    from shapely.geometry import shape

    w, s, e, n = bounds4326
    where = "category=1" if primary_only else "1=1"
    r = requests.get(f"{BASE}/query", headers={"User-Agent": UA}, params={
        "where": where, "geometry": f"{w},{s},{e},{n}",
        "geometryType": "esriGeometryEnvelope", "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "objectid,name,beginyear,endyear,category",
        "returnGeometry": "true",
        "outSR": 4326, "f": "json"}, timeout=timeout)
    r.raise_for_status()
    feats = r.json().get("features", [])
    rows = []
    for f in feats:
        geo = f.get("geometry")
        if not geo or "rings" not in geo:
            continue
        rows.append({**f["attributes"],
                    "geometry": shape({"type": "Polygon", "coordinates": geo["rings"]})})
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    if len(gdf):
        gdf["area_km2"] = gdf.to_crs(5070).area / 1e6
    return gdf


def match_perimeters(perimeters, *, min_frac: float = 0.90, fire_years=None,
                     year_window: tuple = (0, 0), primary_only: bool = True):
    """Which of ``perimeters`` (GeoDataFrame, any CRS, one row per fire) has a
    clean BAER SBS match, and what raster covers it.

    **Pass ``fire_years``.** Spatial overlap alone is not a match: a fire
    perimeter frequently sits inside a DIFFERENT fire's BAER footprint --
    California burns over its own scars, and the big complexes cover ground
    that reburns within a few seasons. Matching on geometry only put 29 of
    140 CA fires against an assessment from another year (gaps of -7 to +7),
    and those 29 came out with median Youden J 0.05 and kappa 0.002 against
    0.52 and 0.20 for the rest: a BAER map of a different fire is noise, and
    it looked like a weak result rather than a wrong join. Nevada happened to
    be clean (all 19 gap 0), which is exactly why this went unnoticed there.

    ``fire_years`` is an ignition year per row, aligned to ``perimeters``
    (Series indexed like it, or any array-like in row order).
    ``year_window`` is the allowed ``beginyear - ig_year`` range. It defaults
    to ``(0, 0)`` -- same year only. A ``+1`` allowance was tried, on the
    reasoning that a late-season fire is walked the following spring, and in
    California it admitted nothing but reburns: all 10 of the +1 matches were
    to a LATER fire's tile (three 2019 fires against AugustComplexNorth 2020),
    median kappa -0.001. Even RANCH, which ignited in November, drew the next
    year's complex rather than its own assessment.

    A match is "clean" when >= ``min_frac`` of the fire's own perimeter area
    is covered by ONE BAER footprint -- computed relative to the fire, not
    ``min(fire_area, baer_area)``, so a giant multi-fire overview mosaic that
    merely brushes a small fire's corner does not count (that fraction would
    be tiny relative to the fire even though it might be ~1.0 relative to the
    mosaic). Requires an identifying column the caller can carry through
    (pass it already merged into ``perimeters``, e.g. as the index).

    Returns a DataFrame indexed like ``perimeters`` with ``baer_name,
    baer_year, baer_oid, coverage_frac`` (pass ``baer_oid`` to
    :func:`fetch_aligned` as ``lock_raster_id`` -- see the module docstring) for every row that cleared ``min_frac``; rows
    with no adequate match are simply absent, not zero-filled.
    """
    import pandas as pd

    p4326 = perimeters.to_crs("EPSG:4326")
    w, s, e, n = p4326.total_bounds
    pad = 0.3
    baer = footprints((w - pad, s - pad, e + pad, n + pad), primary_only=primary_only)
    if not len(baer):
        return pd.DataFrame(columns=["baer_name", "baer_year", "coverage_frac"])

    p5070, b5070 = p4326.to_crs(5070), baer.to_crs(5070)
    years = None
    if fire_years is not None:
        years = (fire_years if hasattr(fire_years, "reindex")
                 else pd.Series(list(fire_years), index=p5070.index))
    lo, hi = year_window
    rows = []
    for idx, prow in p5070.iterrows():
        parea = prow.geometry.area
        if parea <= 0:
            continue
        cand = b5070[b5070.intersects(prow.geometry)]
        if years is not None:
            iy = years.get(idx) if hasattr(years, "get") else None
            if iy is not None and pd.notna(iy):
                gap = cand["beginyear"] - float(iy)
                cand = cand[(gap >= lo) & (gap <= hi)]
        best_name, best_year, best_oid, best_frac = None, None, None, 0.0
        for _, brow in cand.iterrows():
            frac = prow.geometry.intersection(brow.geometry).area / parea
            if frac > best_frac:
                best_name, best_year, best_frac = brow["name"], brow["beginyear"], frac
                best_oid = brow.get("objectid")
        if best_frac >= min_frac:
            rows.append({"_idx": idx, "baer_name": best_name, "baer_year": best_year,
                        "baer_oid": best_oid, "coverage_frac": round(best_frac, 4)})
    return pd.DataFrame(rows).set_index("_idx") if rows else pd.DataFrame(
        columns=["baer_name", "baer_year", "baer_oid", "coverage_frac"])


def fetch_aligned(bounds, shape, crs_epsg: int, *, lock_raster_id=None,
                  timeout: float = 120.0) -> np.ndarray:
    """BAER SBS classified codes on the EXACT grid described by ``bounds``
    (left, bottom, right, top), ``shape`` (rows, cols), ``crs_epsg``.

    Pass the target raster's own ``rasterio`` bounds/shape/CRS (e.g. an MTBS
    dNBR tile) to get pixel-for-pixel alignment with no separate resample.
    Nearest-neighbor -- these are categorical codes, and any other
    interpolation invents class values that no source pixel held (same
    reasoning as ``plotting._severity_rgba``).

    ``lock_raster_id`` is a catalog ``objectid`` (from :func:`footprints` or
    :func:`match_perimeters`). Pass it: without it the service composites
    every overlapping assessment by its own default rule and can return a
    different fire's map. See the module docstring.
    """
    import json as _json

    left, bottom, right, top = bounds
    h, w = shape
    params = {
        "bbox": f"{left},{bottom},{right},{top}", "bboxSR": crs_epsg,
        "imageSR": crs_epsg, "size": f"{w},{h}", "format": "tiff",
        "pixelType": "U8", "interpolation": "RSP_NearestNeighbor", "f": "image"}
    if lock_raster_id is not None:
        params["mosaicRule"] = _json.dumps({
            "mosaicMethod": "esriMosaicLockRaster",
            "lockRasterIds": [int(lock_raster_id)]})
    r = requests.get(f"{BASE}/exportImage", headers={"User-Agent": UA},
                     params=params, timeout=timeout)
    r.raise_for_status()
    with rasterio.io.MemoryFile(r.content) as mf, mf.open() as ds:
        return ds.read(1)
