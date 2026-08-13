"""Regression tests for statewide tile-store mosaics.

The int-path collar bug (2026-08-13): rasterio 1.5 initializes the uncovered
destination with SRC nodata when src/dst nodata differ on integer grids, so a
chunk's out-of-coverage area came back as -9999 *data* and, under
first-valid-wins, masked every later chunk. These tests pin the fixed
behavior: uncovered destination stays at the fill value, explicit source
nodata never survives as data, and later sources fill what earlier ones
did not cover.
"""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from firescape import statewide


def _write(path, arr, *, west, north, res, crs="EPSG:5070", nodata=None):
    h, w = arr.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=1,
        dtype=arr.dtype.name, crs=crs, nodata=nodata,
        transform=from_origin(west, north, res, res),
    ) as ds:
        ds.write(arr, 1)
    return path


@pytest.fixture()
def halves(tmp_path):
    """Two int16 sources, each covering half of (0,0,200,100), nodata=-9999.

    Source A (left) carries a column of explicit nodata inside its footprint.
    """
    a = np.full((10, 10), 7, dtype=np.int16)
    a[:, -1] = -9999
    b = np.full((10, 10), 9, dtype=np.int16)
    return [
        _write(tmp_path / "a.tif", a, west=0, north=100, res=10, nodata=-9999),
        _write(tmp_path / "b.tif", b, west=100, north=100, res=10, nodata=-9999),
    ]


def test_uncovered_destination_stays_fill_int(halves):
    arr, _ = statewide._mosaic_to_grid(halves[:1], (0, 0, 200, 100),
                                       resolution=10, resampling="nearest",
                                       dtype="int32", pad_m=0)
    right = arr[:, arr.shape[1] // 2:]
    assert not (arr == -9999).any(), "source nodata leaked into the mosaic"
    assert (right == 0).all(), "uncovered area must stay at the fill value"
    assert (arr[:, :9] == 7).all()


def test_later_sources_fill_uncovered_area(halves):
    arr, _ = statewide._mosaic_to_grid(halves, (0, 0, 200, 100),
                                       resolution=10, resampling="nearest",
                                       dtype="int32", pad_m=0)
    assert not (arr == -9999).any()
    assert (arr[:, :9] == 7).all()
    assert (arr[:, 10:] == 9).all(), "second chunk must fill past the first"
    # the explicit-nodata column inside A is filled by B's coverage there? no —
    # A's nodata column is at x=90-100, outside B; it must be fill, not -9999
    assert (arr[:, 9] == 0).all()


def test_float_path_uncovered_is_nan(tmp_path):
    dem = np.full((10, 10), 1234.5, dtype=np.float32)
    dem[0, 0] = -999999.0
    src = _write(tmp_path / "dem.tif", dem, west=0, north=100, res=10,
                 nodata=-999999.0)
    arr, _ = statewide._mosaic_to_grid([src], (0, 0, 200, 100),
                                       resolution=10, resampling="bilinear",
                                       pad_m=0)
    right = arr[:, arr.shape[1] // 2:]
    assert np.isnan(right).all()
    assert not (arr <= -999998).any(), "float nodata leaked as data"
    assert np.isnan(arr[0, 0])
