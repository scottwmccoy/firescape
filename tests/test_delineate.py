"""Rossi mask primitives (pure array logic, no pfdf/network needed)."""

import numpy as np
import pandas as pd
import pytest

from firescape import delineate


class TestFocalStd:
    def test_flat_surface_has_zero_std(self):
        assert delineate._focal_std(np.full((20, 20), 1500.0), 3) == pytest.approx(0, abs=1e-9)

    def test_matches_numpy_std_in_window(self):
        rng = np.random.default_rng(0)
        a = rng.normal(1500, 50, (40, 40))
        got = delineate._focal_std(a, 2)
        # interior cell: compare against an explicit 5x5 window std
        assert got[10, 10] == pytest.approx(a[8:13, 8:13].std(), rel=1e-9)

    def test_rough_terrain_exceeds_valley_threshold(self):
        y, x = np.mgrid[0:60, 0:60]
        ridges = 1500 + 40 * np.sin(x / 2.0)          # steep, high relief
        flat = np.full((60, 60), 1500.0)
        assert delineate._focal_std(ridges, 3).mean() > 5.0
        assert delineate._focal_std(flat, 3).mean() <= 5.0


class TestDropSmall:
    def test_removes_clusters_below_area(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[:20, :20] = True     # 400 px
        mask[40, 40] = True       # 1 px speck
        px_km2 = 0.01             # 100 m pixels -> big block = 4 km2
        out = delineate._drop_small(mask, min_km2=1.0, pixel_area_km2=px_km2)
        assert out[:20, :20].all()
        assert not out[40, 40]

    def test_empty_mask_survives(self):
        out = delineate._drop_small(np.zeros((10, 10), dtype=bool), 1.0, 0.01)
        assert not out.any()


class TestWaterCodes:
    def test_matches_water_snow_quarry_by_name(self):
        legend = pd.DataFrame({
            "Value": [7292, 7297, 7295, 7019, 9308],
            "EVT_NAME": ["Open Water", "Perennial Ice/Snow", "Quarries-Strip Mines",
                          "Great Basin Pinyon-Juniper Woodland",
                          "Great Basin & Intermountain Introduced Annual Grassland"],
        })
        codes = delineate.water_codes_from_legend(legend)
        assert set(codes) == {7292, 7297, 7295}
