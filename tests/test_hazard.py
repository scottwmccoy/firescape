"""Pure-numpy reference tests (no pfdf needed). The pfdf-backed wrappers get
exercised for real in milestone M1's walking skeleton."""

import numpy as np
import pytest

from firescape import hazard


def test_m1_reference_zero_inputs():
    # T=F=S=0 -> p = logistic(-3.63)
    p = hazard.logistic_m1_reference(0.0, 0.0, 0.0, R_mm=6.0)
    assert p == pytest.approx(1 / (1 + np.exp(3.63)), rel=1e-9)


def test_m1_reference_typical_basin():
    # T=0.6, F=0.35, S=0.25 at R15=6 mm (i15=24 mm/h):
    # X = -3.63 + (0.41*0.6 + 0.67*0.35 + 0.70*0.25)*6
    #   = -3.63 + (0.246 + 0.2345 + 0.175)*6 = -3.63 + 3.933 = 0.303
    p = hazard.logistic_m1_reference(0.6, 0.35, 0.25, R_mm=6.0)
    assert p == pytest.approx(1 / (1 + np.exp(-0.303)), rel=1e-9)
    assert 0.5 < p < 0.6


def test_m1_reference_monotone_in_rain():
    T, F, S = 0.5, 0.3, 0.25
    probs = [hazard.logistic_m1_reference(T, F, S, R_mm=r) for r in (2, 4, 6, 8, 10)]
    assert np.all(np.diff(probs) > 0)


def test_volume_ranges_known_answer():
    import numpy as np

    from firescape import hazard as hz

    # hand-computed: ln V = 4.79 + 1.052 ln 2 + 1.376 ln 20 + 0.610 ln 1.5
    #                + 0.476 ln 0.3 - 1.388*0.4
    lnv = (4.79 + 1.052 * np.log(2.0) + 1.376 * np.log(20.0)
           + 0.610 * np.log(1.5) + 0.476 * np.log(0.3) - 1.388 * 0.4)
    V, Vmin, Vmax = hz.volume_ranges(2.0, 20.0, 1.5, 0.3, 0.4)
    assert np.isclose(float(V), np.exp(lnv), rtol=1e-12)
    assert np.isclose(float(Vmax) / float(V), np.exp(hz.RANGES_RMSE_LN))
    V2, _, _ = hz.volume_ranges([2.0, -1.0], [20.0, 20.0], [1.5, 1.5],
                                [0.3, 0.3], [0.4, 0.4])
    assert np.isfinite(V2[0]) and np.isnan(V2[1])
