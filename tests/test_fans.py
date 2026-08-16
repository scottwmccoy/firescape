"""Known-answer tests for the fan stage (offline, synthetic terrain)."""
import numpy as np
import pytest

import geopandas as gpd
from rasterio.transform import from_origin
from shapely.geometry import LineString

from firescape import fans


@pytest.fixture
def ref():
    return {"transform": from_origin(0, 300, 3, 3),
            "crs": "EPSG:32611", "width": 100, "height": 100}


def test_hand_lite_tilted_plane(ref):
    # plane rising 1 m per row away from a channel along row 50
    dem = np.abs(np.arange(100) - 50)[:, None] * np.ones((1, 100), np.float32)
    channels = np.zeros((100, 100), bool)
    channels[50, :] = True
    hand, dist = fans.hand_lite(dem, channels, pixel_m=3.0)
    assert hand[50, 10] == 0.0
    assert hand[53, 10] == pytest.approx(3.0)      # 3 rows up = 3 m
    assert dist[53, 10] == pytest.approx(9.0)      # 3 px * 3 m
    with pytest.raises(ValueError, match="empty"):
        fans.hand_lite(dem, np.zeros_like(channels), pixel_m=3.0)


def test_fan_zone_logic():
    hand = np.array([[0.5, 10.0], [1.0, np.nan]], np.float32)
    slope = np.array([[5.0, 5.0], [20.0, 5.0]], np.float32)
    dist = np.array([[50.0, 50.0], [50.0, 600.0]], np.float32)
    zone = fans.fan_zone(hand, slope, dist)
    assert zone[0, 0]                  # low hand, gentle, near
    assert not zone[0, 1]              # too high above channel
    assert not zone[1, 0]              # too steep
    assert not zone[1, 1]              # nan hand + too far


def test_detect_fans_finds_the_blob(ref):
    z = np.zeros((100, 100), np.float32)
    z[40:50, 20:35] = 4.0              # 150 px deposit
    z[10:12, 80:82] = 4.0              # 4 px speck -> dropped
    zone = np.ones((100, 100), bool)
    got = fans.detect_fans(z, zone, ref, z_t=2.5, min_area_px=30)
    assert len(got) == 1
    assert got.loc[0, "area_m2"] == pytest.approx(150 * 9.0)
    assert got.loc[0, "mean_z"] == pytest.approx(4.0)
    # zone gating works: same z, empty zone -> nothing
    assert len(fans.detect_fans(z, np.zeros_like(zone), ref)) == 0


def test_channel_mask(ref):
    seg = gpd.GeoDataFrame(geometry=[LineString([(30, 150), (150, 150)])],
                           crs="EPSG:32611")
    m = fans.channel_mask(seg, ref)
    assert m[50, 20] and not m[10, 10]
