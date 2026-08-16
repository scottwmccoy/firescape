"""Known-answer tests for corridor conditioning (offline, synthetic grid)."""
import numpy as np
import pandas as pd
import pytest

import geopandas as gpd
from shapely.geometry import LineString

from firescape import corridor as co
from rasterio.transform import from_origin


@pytest.fixture
def grid():
    # 100 x 100 px at 3 m, origin (0, 300): x right, y down
    return {"transform": from_origin(0, 300, 3, 3),
            "crs": "EPSG:32611", "width": 100, "height": 100}


@pytest.fixture
def segments():
    # one horizontal reach through the middle, one vertical on the right
    return gpd.GeoDataFrame(
        {"Area_km2": [0.25, 4.0]},
        geometry=[LineString([(30, 150), (150, 150)]),
                  LineString([(240, 30), (240, 270)])],
        crs="EPSG:32611")


def test_corridor_width_scales_and_caps():
    w = co.corridor_width_m(np.array([0.0, 1.0, 100.0]))
    assert w[0] == 9.0                       # floor
    assert w[1] == pytest.approx(21.0)       # 9 + 12*1
    assert w[2] == 60.0                      # cap
    assert co.corridor_width_m(np.nan) == 9.0


def test_corridor_raster_labels(grid, segments):
    lab = co.corridor_raster(segments, grid)
    assert lab.shape == (100, 100)
    assert set(np.unique(lab)) == {0, 1, 2}
    # horizontal reach: row y=150 -> px row 50, cols 10..50
    assert lab[50, 20] == 1
    # vertical reach at x=240 -> col 80
    assert lab[40, 80] == 2
    assert lab[5, 5] == 0
    # bigger area -> wider corridor
    assert (lab == 2).sum() > (lab == 1).sum()


def test_segment_stats_and_classify(grid, segments):
    lab = co.corridor_raster(segments, grid)
    z = np.zeros((100, 100), np.float32)
    z[lab == 1] = 3.5                        # hot reach
    z[lab == 2] = 0.2                        # quiet reach
    z[0, 0] = 99.0                           # background noise, ignored
    stats = co.segment_stats(lab, {"z_brightness": z})
    assert list(stats.index) == [0, 1]
    assert stats.loc[0, "z_brightness_mean"] == pytest.approx(3.5)
    assert stats.loc[1, "z_brightness_mean"] == pytest.approx(0.2)
    assert stats.loc[0, "z_brightness_frac"] == 1.0

    cls = co.classify(stats)
    assert cls.loc[0] == 3 and cls.loc[1] == 0

    # masked pixels drop out; a fully-masked segment stays class 0
    zm = np.ma.masked_array(z, mask=(lab == 2))
    stats2 = co.segment_stats(lab, {"z_brightness": zm})
    assert 1 not in stats2.index or np.isnan(stats2.loc[1, "z_brightness_mean"])
    cls2 = co.classify(stats2.reindex([0, 1]))
    assert cls2.loc[1] == 0


def test_classify_min_pixels_guard():
    stats = pd.DataFrame({"z_brightness_mean": [5.0], "n_pixels": [3]})
    assert co.classify(stats).iloc[0] == 0   # too few pixels to call
