"""Soil erodibility (KF factor) for the M1 soil term.

Two independent sources:

- ``kf_factor`` — STATSGO KFFACT via ``pfdf.data.usgs.statsgo``. This is what
  Rossi et al. (2025) and the USGS assessments use, so it stays the default.
  Hosted on ScienceBase.
- ``kf_factor_ssurgo`` — SSURGO via the NRCS Soil Data Access services: WFS
  ``mapunitpoly`` for map-unit geometry plus an SDA tabular query for the
  component-weighted surface-horizon ``kffact``. A completely independent
  host, so the tool keeps working through ScienceBase outages, and SSURGO is
  finer-grained than STATSGO. Using it is a documented deviation from the
  USGS method — always record which source produced a product.

KF < 0 is treated as nodata (Rossi et al. excluded negative values).
"""

from __future__ import annotations

import numpy as np

WFS_URL = "https://sdmdataaccess.nrcs.usda.gov/Spatial/SDMWGS84Geographic.wfs"
SDA_URL = "https://sdmdataaccess.nrcs.usda.gov/Tabular/post.rest"


def kf_factor(bounds):
    """STATSGO KFFACT raster for ``bounds`` (pfdf BoundsInput). KF<0 -> NaN."""
    from pfdf.data.usgs import statsgo
    from pfdf.raster import Raster

    kf = statsgo.read("KFFACT", bounds=bounds)
    values = kf.values.astype("float64", copy=True)
    if kf.nodata is not None:
        values[values == kf.nodata] = np.nan
    values[values < 0] = np.nan
    return Raster.from_array(values, nodata=np.nan, spatial=kf)


def _wfs_tile(bbox, *, timeout: float, scratch):
    """One WFS mapunitpoly request -> GeoDataFrame (EPSG:4326)."""
    import geopandas as gpd
    import requests

    from firescape import paths

    w, s, e, n = bbox
    r = requests.get(WFS_URL, params={
        "service": "WFS", "version": "1.1.0", "request": "GetFeature",
        "typename": "mapunitpoly", "bbox": f"{w},{s},{e},{n}",
        "srsname": "EPSG:4326", "outputformat": "GML2",
    }, timeout=timeout, headers={"User-Agent": paths.BROWSER_UA})
    r.raise_for_status()
    tmp = scratch / f"soil_{w:.3f}_{s:.3f}.gml"
    tmp.write_bytes(r.content)
    try:
        gdf = gpd.read_file(tmp)
    finally:
        tmp.unlink(missing_ok=True)
    if len(gdf):
        gdf = _fix_axis_order(gdf)
    return gdf


def _fix_axis_order(gdf):
    """Put GML coordinates in (lon, lat) and stamp EPSG:4326.

    WFS 1.1.0 honours the EPSG:4326 authority axis order, so this service
    returns (lat, lon) — geopandas reads the geometries CRS-naive, and without
    this the polygons land in the wrong place silently. Detected rather than
    assumed: in CONUS longitude is < -90 and latitude is < 90, so if the first
    ordinate is the small one, the axes are swapped.
    """
    from shapely.ops import transform as shapely_transform

    xmin, ymin, xmax, ymax = gdf.total_bounds
    if abs(xmin) <= 90 < abs(ymin):
        gdf = gdf.set_geometry(
            gdf.geometry.apply(
                lambda g: shapely_transform(lambda x, y, z=None: (y, x), g)))
    return gdf.set_crs("EPSG:4326", allow_override=True)


def _sda_kffact(mukeys, *, timeout: float, chunk: int = 400) -> dict:
    """Component-weighted surface-horizon kffact for a list of mukeys.

    Weighted by component percentage over horizons starting at the surface
    (hzdept_r = 0), which is the erodibility of the fine fraction that the
    Staley 2017 models expect.
    """
    import requests

    from firescape import paths

    out: dict[int, float] = {}
    mukeys = [str(int(m)) for m in mukeys]
    for i in range(0, len(mukeys), chunk):
        part = mukeys[i:i + chunk]
        sql = (
            "SELECT c.mukey, SUM(CAST(ch.kffact AS FLOAT) * c.comppct_r) "
            "/ SUM(c.comppct_r) AS kf "
            "FROM component c INNER JOIN chorizon ch ON ch.cokey = c.cokey "
            f"WHERE c.mukey IN ({','.join(part)}) AND ch.hzdept_r = 0 "
            "AND ch.kffact IS NOT NULL AND c.comppct_r IS NOT NULL "
            "GROUP BY c.mukey"
        )
        r = requests.post(SDA_URL, json={"query": sql, "format": "JSON"},
                          timeout=timeout, headers={"User-Agent": paths.BROWSER_UA})
        r.raise_for_status()
        for mukey, kf in r.json().get("Table", []):
            try:
                out[int(mukey)] = float(kf)
            except (TypeError, ValueError):
                continue
    return out


def kf_factor_ssurgo(bounds4326, *, tile_deg: float = 0.25, timeout: float = 240.0,
                     scratch=None, verbose: bool = True):
    """SSURGO KF-factor map units for a lon/lat bbox, as a GeoDataFrame.

    Returns columns ``mukey``, ``kf`` (KF<0 or unmatched -> NaN) and geometry
    in EPSG:4326. Tiled because the WFS returns whole polygons per request.
    """
    import geopandas as gpd
    import pandas as pd

    from firescape import paths

    scratch = scratch or paths.cache_root()
    w, s, e, n = bounds4326
    xs = np.arange(w, e, tile_deg)
    ys = np.arange(s, n, tile_deg)
    frames = []
    for x in xs:
        for y in ys:
            tile = (x, y, min(x + tile_deg, e), min(y + tile_deg, n))
            try:
                g = _wfs_tile(tile, timeout=timeout, scratch=scratch)
            except Exception as exc:  # keep going; report the gap
                if verbose:
                    print(f"  tile {tile} failed: {type(exc).__name__}")
                continue
            if len(g):
                frames.append(g)
            if verbose:
                print(f"  tile ({x:.2f},{y:.2f}) -> {len(g)} polygons", flush=True)
    if not frames:
        raise RuntimeError("no SSURGO polygons returned for the requested bounds")
    soils = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True))
    soils = soils.set_crs("EPSG:4326", allow_override=True)  # set per tile above
    mukey_col = next(c for c in soils.columns if c.lower() == "mukey")
    soils["mukey"] = soils[mukey_col].astype("int64")
    # Dedupe on the per-POLYGON key, never on mukey: one map unit legitimately
    # occurs as many separate polygons, and collapsing on mukey silently throws
    # away ~90% of the coverage.
    # NB: gml_id is NOT per-polygon on this service - it repeats for every
    # polygon of the same map unit - so mupolygonkey must win regardless of
    # column order.
    lower = {c.lower(): c for c in soils.columns}
    poly_key = next((lower[k] for k in ("mupolygonkey",) if k in lower), None)
    if poly_key:
        soils = soils.drop_duplicates(subset=[poly_key])

    kf_map = _sda_kffact(sorted(soils["mukey"].unique()), timeout=timeout)
    soils["kf"] = soils["mukey"].map(kf_map)
    soils.loc[soils["kf"] < 0, "kf"] = np.nan
    if verbose:
        got = soils["kf"].notna().mean()
        print(f"  {len(soils)} map units, kffact resolved for {got:.1%}")
    return soils[["mukey", "kf", "geometry"]]


def zonal_kf(basins, soils, *, kf_col: str = "kf"):
    """Area-weighted mean KF within each basin polygon.

    The saved basin polygon *is* the catchment, so an area-weighted overlay
    reproduces pfdf's catchment-mean KF exactly, without re-delineating.
    """
    import geopandas as gpd
    import numpy as np

    soils = soils.to_crs(basins.crs)
    basins = basins.copy()
    basins["_bid"] = np.arange(len(basins))
    inter = gpd.overlay(basins[["_bid", "geometry"]], soils[[kf_col, "geometry"]],
                        how="intersection", keep_geom_type=True)
    inter = inter[inter[kf_col].notna()].copy()
    inter["_w"] = inter.geometry.area
    inter["_wk"] = inter["_w"] * inter[kf_col]
    agg = inter.groupby("_bid").agg(_wk=("_wk", "sum"), _w=("_w", "sum"))
    kf = (agg["_wk"] / agg["_w"]).reindex(basins["_bid"]).to_numpy()
    return kf
