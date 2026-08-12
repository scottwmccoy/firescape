"""MTBS (Monitoring Trends in Burn Severity) access.

Endpoints validated 2026-08-12:

- Burned-area boundaries + fire-occurrence points: static daily-refreshed zips
  (safe to automate).
- Per-fire attribute records incl. the analyst dNBR thresholds: GeoServer WFS
  layer ``mtbs:burn_severity_fire_polygons``. GEOMETRY IS EPSG:3857 — a
  lon/lat ``BBOX()`` cql filter silently returns zero rows, so spatial
  filtering happens client-side here.
- Annual severity mosaics: GeoServer WCS, coverageId ``mtbs__mtbs_CONUS_<year>``
  (double underscore), bbox-subset GeoTIFF. 30 m, classes 0-6.
- Per-fire bundles (dnbr.tif etc.): NO static URL — ZipServlet POST or the
  viewer's email queue (max 500 fires). See ``bundle_note()``.

MTBS thematic severity (dnbr6) is SIX classes, not BARC4:
1 unburned-to-low, 2 low, 3 moderate, 4 high, 5 increased greenness,
6 non-mapping mask (plus 0 background). Classes 5 and 6 must never be treated
as burned.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import requests

from firescape import paths

_DL = "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/MTBS_Fire/data/composite_data"
PERIMETERS_ZIP_URL = f"{_DL}/burned_area_extent_shapefile/mtbs_perimeter_data.zip"
FOD_POINTS_ZIP_URL = f"{_DL}/fod_pt_shapefile/mtbs_fod_pts_data.zip"

GEOSERVER = "https://edcintl.cr.usgs.gov/geoserver"
DOI = "10.5066/P9NETC0T"

#: MTBS dnbr6 class semantics.
SEVERITY_CLASSES = {
    0: "background",
    1: "unburned_to_low",
    2: "low",
    3: "moderate",
    4: "high",
    5: "increased_greenness",   # exclude from burned-area logic
    6: "non_mapping",           # cloud/shadow/water mask -> nodata
}

#: dnbr6 -> BARC4 (pfdf numbering; 0 = nodata). 5 and 6 are NOT burned classes.
_DNBR6_TO_BARC4 = np.array([0, 1, 2, 3, 4, 0, 0], dtype=np.uint8)


def dnbr6_to_barc4(dnbr6: np.ndarray) -> np.ndarray:
    """Map an MTBS dnbr6 array onto BARC4 (0=nodata, 1..4)."""
    arr = np.asarray(dnbr6)
    out = np.zeros(arr.shape, dtype=np.uint8)
    valid = (arr >= 0) & (arr <= 6)
    out[valid] = _DNBR6_TO_BARC4[arr[valid].astype(np.intp)]
    return out


def fire_records(*, event_id_like: str | None = "NV%", after: str | None = None,
                 bbox4326=None, timeout: float = 180.0):
    """Per-fire MTBS records (attributes + perimeter) from the WFS layer.

    Attributes include ``event_id, irwinid, incid_name, ig_date, burnbndac,
    dnbr_offst, dnbr_stddv, nodata_t, incgreen_t, low_t, mod_t, high_t`` — the
    ``*_t`` fields are the fire-specific analyst dNBR thresholds needed for
    calibration (unburned-low = low_t, low-moderate = mod_t, mod-high = high_t).

    ``bbox4326`` filters client-side after reprojection (see module docstring).
    """
    import geopandas as gpd

    cql = []
    if event_id_like:
        cql.append(f"event_id LIKE '{event_id_like}'")
    if after:
        cql.append(f"ig_date AFTER {after}T00:00:00Z")
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": "mtbs:burn_severity_fire_polygons",
        "outputFormat": "application/json",
    }
    if cql:
        params["cql_filter"] = " AND ".join(cql)
    r = requests.get(f"{GEOSERVER}/mtbs/wfs", params=params, timeout=timeout,
                     headers={"User-Agent": paths.BROWSER_UA})
    r.raise_for_status()
    gdf = gpd.GeoDataFrame.from_features(r.json().get("features", []), crs="EPSG:3857")
    if len(gdf):
        gdf = gdf.to_crs("EPSG:4326")
        if "ig_date" in gdf.columns:  # served as e.g. '1985-06-19Z'
            import pandas as pd

            gdf["ig_date"] = pd.to_datetime(
                gdf["ig_date"].astype(str).str.rstrip("Z"), errors="coerce"
            )
        if bbox4326 is not None:
            from shapely.geometry import box

            gdf = gdf[gdf.intersects(box(*bbox4326))].reset_index(drop=True)
    return gdf


def dedupe_records(gdf):
    """One row per event_id, preferring rows with usable analyst thresholds.

    The WFS layer carries duplicate records for some fires (re-assessments,
    alternate mappings with zeroed thresholds); keep the row with a valid
    low-moderate threshold (mod_t > 0), then the larger mapped area.
    """
    df = gdf.copy()
    df["_has_t"] = (df.get("mod_t", 0) > 0).astype(int)
    df = df.sort_values(["_has_t", "burnbndac"], ascending=False)
    return df.drop_duplicates("event_id", keep="first").drop(columns="_has_t")


def severity_mosaic(year: int, bounds4326, dest: Path, *, timeout: float = 600.0) -> Path:
    """Download a bbox subset of the annual CONUS severity mosaic (GeoTIFF).

    Uses WCS 2.0.1 GetCoverage with Lat/Long subsetting axes (the mosaics are
    served in geographic coordinates). Verify axis labels against
    DescribeCoverage on first use per year — GeoServer axis naming can vary.
    """
    w, s, e, n = bounds4326
    params = {
        "service": "WCS",
        "version": "2.0.1",
        "request": "GetCoverage",
        "coverageId": f"mtbs__mtbs_CONUS_{year}",
        "format": "image/geotiff",
    }
    subsets = [("subset", f"Long({w},{e})"), ("subset", f"Lat({s},{n})")]
    query = list(params.items()) + subsets
    dest = Path(dest)
    with requests.get(f"{GEOSERVER}/mtbs/wcs", params=query, stream=True,
                      timeout=timeout, headers={"User-Agent": paths.BROWSER_UA}) as r:
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        if "xml" in ctype:  # WCS exceptions come back as XML with HTTP 200
            raise RuntimeError(f"WCS exception for {year}: {r.text[:500]}")
        import tempfile

        with tempfile.NamedTemporaryFile(dir=paths.cache_root(), delete=False,
                                         suffix=".part") as tmp:
            for chunk in r.iter_content(1 << 20):
                tmp.write(chunk)
            tmp_path = Path(tmp.name)
    paths.atomic_into(tmp_path, dest)
    paths.write_provenance(dest, url=f"{GEOSERVER}/mtbs/wcs", doi=DOI,
                           note=f"CONUS mosaic {year}, bbox {bounds4326}")
    return dest


def download_perimeters(dest_dir: Path | None = None) -> Path:
    """Fetch the full MTBS burned-area boundaries shapefile zip (~390 MB)."""
    dest_dir = Path(dest_dir) if dest_dir else paths.raw_dir("mtbs", "perimeters")
    dest = dest_dir / "mtbs_perimeter_data.zip"
    paths.download(PERIMETERS_ZIP_URL, dest)
    paths.write_provenance(dest, url=PERIMETERS_ZIP_URL, doi=DOI)
    return dest


def bundle_note() -> str:
    return (
        "Per-fire MTBS bundles (dnbr.tif, dnbr6.tif, burn_bndy.shp, metadata) "
        "have no static URL. Order via https://burnseverity.cr.usgs.gov/viewer/"
        "?product=MTBS (email queue, <=500 fires, ~1 h) into "
        "raw/mtbs/fires/<event_id>/, or POST file_paths="
        "'<map_prog>/<year>/<event_id>.zip' to "
        "https://edcintl.cr.usgs.gov/mtbs_remote_zip_servlet/ZipServlet."
    )
