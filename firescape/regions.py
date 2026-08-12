"""Prefire modeling regions.

The pilot ships a single box region ("pilot"). Statewide Nevada regions
(vegetation zones x geomorphic provinces, snapped to HU10 boundaries per Rossi
et al. 2025) drop in later: everything downstream keys on the ``region`` name
shared between the region GeoJSON and the calibration TOML sections.
"""

from __future__ import annotations

from importlib import resources

#: Pilot AOI north of Reno: Peavine, Petersen, Dogskin, Virginia Mtns, west
#: Pyramid Lake. Confirmed (2026-08-12) to contain the Bug and Stallion fire
#: origins; Bug's west flank crosses into California near Hallelujah Junction.
PILOT_BBOX = (-120.30, 39.50, -119.30, 40.50)


def load_regions(name: str = "pilot"):
    """Load a packaged region polygon set as a GeoDataFrame (EPSG:4326)."""
    import geopandas as gpd

    ref = resources.files("firescape") / "data" / "regions" / f"{name}.geojson"
    with resources.as_file(ref) as p:
        gdf = gpd.read_file(p)
    if "region" not in gdf.columns:
        raise ValueError(f"region file {name!r} lacks a 'region' column")
    return gdf


def region_for_aoi(aoi_geometry, regions=None) -> str:
    """Name of the region with the largest overlap with the AOI geometry."""
    if regions is None:
        regions = load_regions()
    inter = regions.geometry.intersection(aoi_geometry).area
    if not (inter > 0).any():
        raise ValueError("AOI does not intersect any prefire modeling region")
    return str(regions.loc[inter.idxmax(), "region"])
