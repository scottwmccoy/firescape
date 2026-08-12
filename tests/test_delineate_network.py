"""Network construction on synthetic terrain (uses pfdf/pysheds locally, no network I/O)."""

import numpy as np
import pytest

pytest.importorskip("pfdf")

from firescape.delineate import network  # noqa: E402


def _synthetic_dem(n=60):
    """A simple tilted, dissected surface with a real drainage pattern."""
    from pfdf.raster import Raster

    y, x = np.mgrid[0:n, 0:n].astype("float64")
    z = 2000 - 4 * y + 12 * np.abs(np.sin(x / 4.0))
    return Raster.from_array(z, nodata=-9999.0, crs=5070,
                             transform=(10.0, 0.0, -2_000_000.0, 0.0, -10.0, 2_100_000.0))


def test_fully_masked_unit_returns_no_network():
    """Playa/lake units mask out entirely; pfdf raises on an empty mask, so
    network() must report None rather than propagate EmptyArrayError."""
    from pfdf.raster import Raster

    dem = _synthetic_dem()
    empty_domain = Raster.from_array(np.zeros(dem.shape, dtype=bool), spatial=dem,
                                     isbool=True)
    segments, terr = network(dem, empty_domain)
    assert segments is None
    assert terr.flow is not None  # terrain still computed


def test_open_domain_produces_segments():
    from pfdf.raster import Raster

    dem = _synthetic_dem()
    domain = Raster.from_array(np.ones(dem.shape, dtype=bool), spatial=dem, isbool=True)
    segments, _terr = network(dem, domain, min_area_km2=0.0005)
    assert segments is not None and segments.size > 0
