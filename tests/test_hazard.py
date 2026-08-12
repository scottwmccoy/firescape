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
