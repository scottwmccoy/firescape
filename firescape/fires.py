"""Active and archived fire perimeters from NIFC WFIGS ArcGIS services.

Endpoints validated 2026-08-12. The current-perimeters layer refreshes every
~5 minutes. Names are bare and inconsistently cased (``Bug``, ``Stallion``,
``MOUSE MEADOW``) so matching uses ``UPPER(attr_IncidentName) LIKE``; the
stable join key across WFIGS/MTBS/BAER is the IRWIN ID (``attr_IrwinID``) —
names get reused, and merged fires drop records (e.g. Fred Mountain absorbed
into Bug, Aug 2026). ``poly_GISAcres`` is the geometry-derived size;
``attr_IncidentSize`` is the ICS-209 reported size — they can differ by >15%.
"""

from __future__ import annotations

import json
from pathlib import Path

import requests

from firescape import paths

_HOST = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services"
CURRENT_URL = f"{_HOST}/WFIGS_Interagency_Perimeters_Current/FeatureServer/0/query"
ARCHIVE_URL = f"{_HOST}/WFIGS_Interagency_Perimeters/FeatureServer/0/query"  # all years

_PAGE = 1000


def _query(url: str, where: str, *, bbox4326=None, timeout: float = 120.0) -> list[dict]:
    """Paged ArcGIS REST query returning raw GeoJSON features."""
    features: list[dict] = []
    offset = 0
    while True:
        params = {
            "where": where,
            "outFields": "*",
            "outSR": 4326,
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": _PAGE,
        }
        if bbox4326 is not None:
            w, s, e, n = bbox4326
            params.update(
                geometry=json.dumps({"xmin": w, "ymin": s, "xmax": e, "ymax": n,
                                     "spatialReference": {"wkid": 4326}}),
                geometryType="esriGeometryEnvelope",
                inSR=4326,
                spatialRel="esriSpatialRelIntersects",
            )
        r = requests.get(url, params=params, timeout=timeout,
                         headers={"User-Agent": paths.BROWSER_UA})
        r.raise_for_status()
        page = r.json()
        got = page.get("features", [])
        features.extend(got)
        if len(got) < _PAGE:
            return features
        offset += len(got)


def perimeters(names=None, *, bbox=None, where=None, url: str = CURRENT_URL):
    """Fetch fire perimeters as a GeoDataFrame (EPSG:4326).

    ``names``: iterable of incident names, matched case-insensitively as
    prefixes. ``bbox``: (W, S, E, N) lon/lat. ``where``: raw SQL override.
    """
    import geopandas as gpd

    if where is None:
        if names:
            clauses = [f"UPPER(attr_IncidentName) LIKE '{n.upper()}%'" for n in names]
            where = "(" + " OR ".join(clauses) + ")"
        else:
            where = "1=1"
    feats = _query(url, where, bbox4326=bbox)
    if not feats:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
    keep_first = ["attr_IncidentName", "attr_IrwinID", "attr_UniqueFireIdentifier",
                  "poly_GISAcres", "attr_IncidentSize", "attr_PercentContained",
                  "attr_FireDiscoveryDateTime", "poly_DateCurrent"]
    cols = [c for c in keep_first if c in gdf.columns]
    cols += [c for c in gdf.columns if c not in cols and c != "geometry"]
    return gdf[cols + ["geometry"]]


def save_perimeters(gdf, dest: Path, *, source_url: str = CURRENT_URL, note: str | None = None) -> Path:
    """Write perimeters GeoJSON into raw/ with a provenance sidecar."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(dest, driver="GeoJSON")
    paths.write_provenance(dest, url=source_url, note=note,
                           n_features=len(gdf))
    return dest
