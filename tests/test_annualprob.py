"""Rossi et al. (2025) Table 3 equations — analytic known answers."""

import numpy as np
import pytest

from firescape import annualprob as ap


class TestLogLinearFit:
    def test_anchors_return_their_own_recurrence(self):
        i1, i50 = np.array([20.0, 15.0]), np.array([70.0, 60.0])
        m, b = ap.fit_log_linear(i1, i50)
        # by construction RI(i1) == 1 yr and RI(i50) == 50 yr
        assert ap.recurrence_interval(i1, m, b) == pytest.approx([1.0, 1.0])
        assert ap.recurrence_interval(i50, m, b) == pytest.approx([50.0, 50.0])

    def test_midpoint_is_geometric_mean_of_recurrence(self):
        m, b = ap.fit_log_linear(20.0, 70.0)
        ri = ap.recurrence_interval(45.0, m, b)  # halfway in intensity
        assert ri == pytest.approx(np.sqrt(50.0))  # => halfway in log10(RI)

    def test_degenerate_anchors_give_nan(self):
        m, b = ap.fit_log_linear(np.array([30.0, 30.0]), np.array([30.0, 20.0]))
        assert np.isnan(m).all()


class TestAnnualProbability:
    def test_known_values(self):
        assert ap.annual_probability(1.0) == pytest.approx(1 - np.exp(-1))    # 0.6321
        assert ap.annual_probability(50.0) == pytest.approx(1 - np.exp(-0.02))  # 0.0198

    def test_monotone_decreasing_in_recurrence(self):
        p = ap.annual_probability(np.array([1.0, 2.0, 10.0, 100.0]))
        assert np.all(np.diff(p) < 0)


class TestPExceedThreshold:
    def test_threshold_at_one_year_intensity(self):
        # a basin whose debris-flow threshold equals its 1-yr storm should
        # have an annual exceedance probability of 1 - e^-1
        p, _m, _b = ap.p_exceed_threshold(20.0, 20.0, 70.0)
        assert p == pytest.approx(1 - np.exp(-1))

    def test_higher_threshold_is_less_likely(self):
        p_low, _, _ = ap.p_exceed_threshold(25.0, 20.0, 70.0)
        p_high, _, _ = ap.p_exceed_threshold(60.0, 20.0, 70.0)
        assert p_low > p_high


def test_combined_is_product_and_passthrough():
    p_rt = np.array([0.4, 0.2])
    assert ap.combined(p_rt) is p_rt
    assert ap.combined(p_rt, np.array([0.05, 0.10])) == pytest.approx([0.02, 0.02])
