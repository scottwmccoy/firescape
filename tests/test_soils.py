"""SSURGO ingest guards. Both bugs these cover were silent-corruption class:
swapped axes put the soils in the wrong hemisphere, and deduping on the wrong
key discarded ~90% of the polygons while still "succeeding"."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon, box

from firescape import soils


def _gdf(coords_are_latlon: bool):
    """A square near Reno, written in either axis order, CRS-naive like GML."""
    lon0, lat0 = -120.0, 39.7
    pts = [(lon0, lat0), (lon0 + 0.1, lat0), (lon0 + 0.1, lat0 + 0.1), (lon0, lat0 + 0.1)]
    if coords_are_latlon:
        pts = [(y, x) for x, y in pts]
    return gpd.GeoDataFrame({"mukey": [1]}, geometry=[Polygon(pts)])


class TestAxisOrder:
    def test_swapped_axes_are_corrected(self):
        out = soils._fix_axis_order(_gdf(coords_are_latlon=True))
        assert out.crs.to_epsg() == 4326
        xmin, ymin, xmax, ymax = out.total_bounds
        assert xmin == pytest.approx(-120.0) and ymin == pytest.approx(39.7)

    def test_correct_axes_are_left_alone(self):
        out = soils._fix_axis_order(_gdf(coords_are_latlon=False))
        xmin, ymin, _, _ = out.total_bounds
        assert xmin == pytest.approx(-120.0) and ymin == pytest.approx(39.7)

    def test_result_lands_in_nevada(self):
        for swapped in (True, False):
            out = soils._fix_axis_order(_gdf(swapped))
            assert out.intersects(box(-120.3, 39.5, -119.3, 40.5)).all()


class TestZonalKf:
    def test_area_weighted_mean(self):
        # basin split 75/25 between two soils -> weighted mean, not simple mean
        basin = gpd.GeoDataFrame({"id": [0]}, geometry=[box(0, 0, 4, 1)], crs="EPSG:5070")
        soil = gpd.GeoDataFrame(
            {"kf": [0.20, 0.40]},
            geometry=[box(0, 0, 3, 1), box(3, 0, 4, 1)], crs="EPSG:5070")
        kf = soils.zonal_kf(basin, soil)
        assert kf[0] == pytest.approx(0.75 * 0.20 + 0.25 * 0.40)

    def test_missing_soil_gives_nan(self):
        basin = gpd.GeoDataFrame({"id": [0, 1]},
                                 geometry=[box(0, 0, 1, 1), box(10, 10, 11, 11)],
                                 crs="EPSG:5070")
        soil = gpd.GeoDataFrame({"kf": [0.3]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:5070")
        kf = soils.zonal_kf(basin, soil)
        assert kf[0] == pytest.approx(0.3)
        assert np.isnan(kf[1])

    def test_nan_kf_polygons_are_ignored(self):
        basin = gpd.GeoDataFrame({"id": [0]}, geometry=[box(0, 0, 2, 1)], crs="EPSG:5070")
        soil = gpd.GeoDataFrame({"kf": [0.30, np.nan]},
                                geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
                                crs="EPSG:5070")
        assert soils.zonal_kf(basin, soil)[0] == pytest.approx(0.30)
