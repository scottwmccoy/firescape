"""Offline test fixtures. No network, no Box: data/cache roots -> tmp."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _isolated_roots(tmp_path, monkeypatch):
    """Never touch Box or the real cache from tests."""
    monkeypatch.setenv("FIRESCAPE_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("FIRESCAPE_CACHE", str(tmp_path / "cache"))


@pytest.fixture
def mini_cdf() -> pd.DataFrame:
    """Tiny CDF table with hand-checkable parameters."""
    return pd.DataFrame(
        {
            "EVT_Code": [3019, 3001, 9999],
            "Weibull_Lambda_Scale": [0.668816, 0.632514, 0.5],
            "Weibull_Kappa_Shape": [9.481709, 8.408742, 10.0],
            "CLASSNAME": [
                "Great Basin Pinyon-Juniper Woodland",
                "Inter-Mountain Basins Sparsely Vegetated Systems",
                "Synthetic",
            ],
        }
    ).set_index("EVT_Code")


@pytest.fixture
def evt_grid() -> np.ndarray:
    """3x3 EVT grid: known class, unknown class, and a repeat."""
    return np.array(
        [
            [3019, 3019, 9999],
            [1234, 3019, 9999],   # 1234 has no parameters
            [3019, 1234, 3019],
        ],
        dtype=np.int32,
    )
