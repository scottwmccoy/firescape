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
                 min_acres: float | None = None, bbox4326=None,
                 timeout: float = 180.0):
    """Per-fire MTBS records (attributes + perimeter) from the WFS layer.

    Attributes include ``event_id, irwinid, incid_name, ig_date, burnbndac,
    dnbr_offst, dnbr_stddv, nodata_t, incgreen_t, low_t, mod_t, high_t`` — the
    ``*_t`` fields are the fire-specific analyst dNBR thresholds needed for
    calibration (unburned-low = low_t, low-moderate = mod_t, mod-high = high_t).

    ``min_acres`` filters server-side (burnbndac is numeric). ``after``
    (YYYY-MM-DD) and ``bbox4326`` filter client-side: ``ig_date`` is served as
    a *string* property, so CQL temporal predicates 500 on this layer, and
    lon/lat BBOX silently matches nothing (see module docstring).
    """
    import geopandas as gpd

    cql = []
    if event_id_like:
        cql.append(f"event_id LIKE '{event_id_like}'")
    if min_acres is not None:
        cql.append(f"burnbndac >= {float(min_acres):.0f}")
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
        if after and "ig_date" in gdf.columns:
            import pandas as pd

            gdf = gdf[gdf["ig_date"] > pd.Timestamp(after)].reset_index(drop=True)
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


def fire_bundle(event_id: str, fires_dir: Path | None = None) -> dict[str, Path]:
    """Locate a downloaded per-fire bundle's files by role.

    Returns a dict with keys among {'dnbr', 'dnbr6', 'burn_area', 'metadata'}
    for raw/mtbs/fires/<EVENT_ID>/ as written by the addQueue ingest.
    """
    fires_dir = Path(fires_dir) if fires_dir else paths.raw_dir("mtbs", "fires")
    d = fires_dir / event_id.upper()
    if not d.is_dir():
        raise FileNotFoundError(f"no bundle directory for {event_id} at {d}")
    out: dict[str, Path] = {}
    for f in d.iterdir():
        n = f.name.lower()
        if n.endswith("_dnbr.tif"):
            out["dnbr"] = f
        elif n.endswith("_dnbr6.tif"):
            out["dnbr6"] = f
        elif n.endswith("_burn_area.shp"):
            out["burn_area"] = f
        elif n.endswith("_metadata.xml") and "iso" not in n:
            out["metadata"] = f
    missing = {"dnbr", "burn_area"} - set(out)
    if missing:
        raise FileNotFoundError(f"bundle {event_id} missing {sorted(missing)} in {d}")
    return out


ADDQUEUE_URL = "https://burnseverity.cr.usgs.gov/downloads/addQueue.php"

#: mapping_products entries are the viewer checkbox labels VERBATIM.
BUNDLE_PRODUCTS_DEFAULT = (
    "Metadata",
    "Continuous severity (i.e dnbr)",
    "6 - Class thematic severity",
    "Burned area boundary",
)


def order_bundles(event_ids, email: str, *, products=BUNDLE_PRODUCTS_DEFAULT,
                  projection: str = "Albers", timeout: float = 180.0) -> dict:
    """Order per-fire MTBS bundles through the burn-severity email queue.

    Replicates the viewer's cDownloadBtn flow (reverse-engineered from
    burnseverity.cr.usgs.gov/viewer main bundle, validated 2026-08-12):
    a WFS lookup resolves each event to (map_id, nonstandard); standard fires
    go in ``mapping_ids``, nonstandard ones as ``[path, map_id]`` bundles with
    path ``<map_prog_lower>/<ig_year>/<event_id>.zip``. The queue emails
    download links to ``email``, usually within ~1 h; max 500 fires/request.
    ``projection`` is "Albers" or "UTM". Returns the parsed server response
    (expect ``{"success": true}``).

    NOTE: the legacy ZipServlet at edcintl returns 503 — this queue is the
    only working programmatic route for bundles.
    """
    import json as _json

    import requests

    event_ids = list(event_ids)
    if len(event_ids) > 500:
        raise ValueError(f"queue accepts <=500 fires per request, got {len(event_ids)}")
    ids = ",".join(f"'{e}'" for e in event_ids)
    r = requests.get(f"{GEOSERVER}/mtbs/wfs", params={
        "service": "WFS", "version": "1.1.0", "request": "GetFeature",
        "typeNames": "mtbs:burn_severity_fire_polygons",
        "outputFormat": "application/json",
        "cql_filter": f"event_id IN ({ids})",
    }, timeout=timeout, headers={"User-Agent": paths.BROWSER_UA})
    r.raise_for_status()
    seen: dict[str, dict] = {}
    for f in r.json()["features"]:
        p = f["properties"]
        prev = seen.get(p["event_id"])
        if prev is None or (prev.get("nonstandard") and not p.get("nonstandard")):
            seen[p["event_id"]] = p
    missing = set(event_ids) - set(seen)
    if missing:
        raise ValueError(f"event_ids not found in MTBS WFS: {sorted(missing)}")

    mapping_ids, mapping_bundles = [], []
    for p in seen.values():
        if p.get("nonstandard"):
            path = f"{str(p['map_prog']).lower()}/{str(p['ig_date'])[:4]}/{p['event_id']}.zip"
            mapping_bundles.append([path, p["map_id"]])
        else:
            mapping_ids.append(p["map_id"])

    payload = {
        "download_type": "mapping_products",
        "mapping_bundles": mapping_bundles,
        "mapping_ids": mapping_ids,
        "mapping_products": list(products),
        "projection": projection,
        "mosaics": [],
    }
    resp = requests.post(ADDQUEUE_URL, data={
        "products": _json.dumps(payload),
        "email": email,
        "request_origin": "'viewer'",   # literal quotes, matching the viewer JS
    }, timeout=timeout, headers={
        "User-Agent": paths.BROWSER_UA,
        "Referer": "https://burnseverity.cr.usgs.gov/viewer/",
        "Origin": "https://burnseverity.cr.usgs.gov",
    })
    resp.raise_for_status()
    return resp.json()
