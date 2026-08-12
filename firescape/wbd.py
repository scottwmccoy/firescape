"""Watershed Boundary Dataset hydrologic units (tiling for large runs).

Hydrologic units are the right tiling unit for basin delineation: their
boundaries follow drainage divides, so catchments are not split the way a
regular grid would split them. pfdf's docs recommend HU-10 for large-scale
analyses; Rossi et al. (2025) used HU-8 statewide and HU-10 in their Basin
and Range region (fewer delineation artifacts) — we use HU-10 in Nevada.

Served from the USGS hydro.nationalmap.gov ArcGIS service (layer 5 = HU-10).
"""

from __future__ import annotations

from firescape import paths

WBD_URL = "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer"
LAYER_HU10 = 5
LAYER_HU8 = 4


def units(bbox, *, layer: int = LAYER_HU10, timeout: float = 120.0):
    """HU polygons intersecting a lon/lat bbox, as a GeoDataFrame (EPSG:4326)."""
    import json

    import geopandas as gpd
    import requests

    w, s, e, n = bbox
    params = {
        "where": "1=1",
        "geometry": json.dumps({"xmin": w, "ymin": s, "xmax": e, "ymax": n,
                                "spatialReference": {"wkid": 4326}}),
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "outSR": 4326,
        "f": "geojson",
    }
    r = requests.get(f"{WBD_URL}/{layer}/query", params=params, timeout=timeout,
                     headers={"User-Agent": paths.BROWSER_UA})
    r.raise_for_status()
    gdf = gpd.GeoDataFrame.from_features(r.json().get("features", []), crs="EPSG:4326")
    for col in ("huc10", "huc8", "name", "areasqkm"):
        if col in gdf.columns:
            continue
        alt = next((c for c in gdf.columns if c.lower() == col), None)
        if alt:
            gdf = gdf.rename(columns={alt: col})
    return gdf
