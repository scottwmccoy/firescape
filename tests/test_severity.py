"""Known-answer and behavior tests for the severity simulation core."""

import numpy as np
import pytest

from firescape import severity


class TestWeibullKnownAnswers:
    """Hand-computed from SimdNBR = lam*(-ln(1-P))^(1/kap) * 2000 - 1000,
    using the Great Basin Pinyon-Juniper class (EVT 3019):
    lam = 0.668816, kap = 9.481709 (Staley 2018 release)."""

    def test_median(self):
        assert severity.weibull_dnbr(0.5, 0.668816, 9.481709) == pytest.approx(286.91, abs=0.05)

    def test_p90(self):
        assert severity.weibull_dnbr(0.9, 0.668816, 9.481709) == pytest.approx(460.63, abs=0.05)

    def test_monotone_in_pdsim(self):
        vals = [severity.weibull_dnbr(p, 0.668816, 9.481709) for p in (0.1, 0.3, 0.5, 0.7, 0.9)]
        assert np.all(np.diff(vals) > 0)

    def test_pdsim_domain(self):
        for bad in (0.0, 1.0, -0.1, 1.1):
            with pytest.raises(ValueError):
                severity.weibull_dnbr(bad, 0.6, 9.0)


class TestSimulateDnbr:
    def test_direct_and_fallback_sources(self, evt_grid, mini_cdf):
        dnbr, src = severity.simulate_dnbr(evt_grid, 0.5, mini_cdf, fallback_code=3001)
        assert dnbr.shape == evt_grid.shape == src.shape
        # class with parameters: exact known answer
        assert dnbr[0, 0] == pytest.approx(286.91, abs=0.05)
        assert src[0, 0] == severity.SRC_DIRECT
        # class without parameters: barren fallback value, flagged
        fb = severity.weibull_dnbr(0.5, 0.632514, 8.408742)
        assert dnbr[1, 0] == pytest.approx(fb, abs=0.05)
        assert src[1, 0] == severity.SRC_FALLBACK

    def test_no_fallback_gives_nan(self, evt_grid, mini_cdf):
        dnbr, src = severity.simulate_dnbr(evt_grid, 0.5, mini_cdf, fallback_code=None)
        assert np.isnan(dnbr[1, 0])
        assert src[1, 0] == severity.SRC_NODATA

    def test_evt_nodata_masked(self, evt_grid, mini_cdf):
        dnbr, src = severity.simulate_dnbr(
            evt_grid, 0.5, mini_cdf, fallback_code=3001, evt_nodata=9999
        )
        assert np.isnan(dnbr[0, 2])
        assert src[0, 2] == severity.SRC_NODATA

    def test_packaged_table_loads_and_covers_fallback(self):
        table = severity.load_cdf_table()
        assert len(table) > 250
        # our fallback class must have parameters. NOTE: Rossi et al.'s 7294
        # (Remap-era barren) is NOT in the Staley 2018 table — hence 3001.
        assert 7294 not in table.index
        assert severity.BARREN_EVT_CODE in table.index
        # spot-check the packaged Great Basin PJ row used in known-answer tests
        assert table.at[3019, "Weibull_Lambda_Scale"] == pytest.approx(0.668816)
        assert table.at[3019, "Weibull_Kappa_Shape"] == pytest.approx(9.481709)


class TestClassifyBarc4:
    def test_class_edges(self):
        dnbr = np.array([-50.0, 124.9, 125.0, 249.9, 250.0, 499.9, 500.0, np.nan])
        barc = severity.classify_barc4(dnbr, (125.0, 250.0, 500.0))
        assert barc.tolist() == [1, 1, 2, 2, 3, 3, 4, 0]

    def test_breaks_must_increase(self):
        with pytest.raises(ValueError):
            severity.classify_barc4(np.zeros(3), (250.0, 125.0, 500.0))


class TestRemapCrosswalk:
    def _legend(self):
        import pandas as pd

        return pd.DataFrame({
            "Value": [3019, 7019, 8888, 9555, 7126],
            "EVT_NAME": [
                "Great Basin Pinyon-Juniper Woodland",          # identity
                "Great Basin Pinyon-Juniper Woodland",          # minus4000 (7019-4000=3019)
                "Inter-Mountain Basins Sparsely Vegetated Systems",  # name match -> 3001
                "Some Introduced Grassland",                    # semantic via extra
                "Totally Novel Class",                          # unmapped
            ],
        })

    def test_rules(self, mini_cdf):
        from firescape.severity import remap_crosswalk

        mapping, audit = remap_crosswalk(self._legend(), mini_cdf, extra={9555: 3019})
        rules = dict(zip(audit["code"], audit["rule"]))
        assert mapping[3019] == 3019 and rules[3019] == "identity"
        assert mapping[7019] == 3019 and rules[7019] == "minus4000"
        assert mapping[8888] == 3001 and rules[8888] == "name"
        assert mapping[9555] == 3019 and rules[9555] == "semantic"
        assert 7126 not in mapping and rules[7126] == "UNMAPPED"

    def test_minus4000_guard_rejects_name_mismatch(self, mini_cdf):
        import pandas as pd

        from firescape.severity import remap_crosswalk

        legend = pd.DataFrame({"Value": [7019], "EVT_NAME": ["Completely Different Thing"]})
        mapping, audit = remap_crosswalk(legend, mini_cdf, extra={})
        assert 7019 not in mapping
        assert audit.iloc[0]["rule"] == "UNMAPPED"

    def test_apply_crosswalk_passthrough(self):
        import numpy as np

        from firescape.severity import apply_crosswalk

        evt = np.array([[7019, 1234], [7019, 9999]])
        out = apply_crosswalk(evt, {7019: 3019})
        assert out.tolist() == [[3019, 1234], [3019, 9999]]


class TestCoverageReport:
    def test_fractions_and_flags(self, evt_grid, mini_cdf):
        rep = severity.coverage_report(evt_grid, mini_cdf)
        assert rep["fraction"].sum() == pytest.approx(1.0)
        row = rep.set_index("EVT_Code")
        assert bool(row.at[3019, "has_params"]) is True
        assert bool(row.at[1234, "has_params"]) is False
        missing_frac = rep.loc[~rep["has_params"], "fraction"].sum()
        assert missing_frac == pytest.approx(2 / 9)


class TestDistributionalSeverity:
    def test_sigma_zero_reduces_to_deterministic(self, mini_cdf):
        import numpy as np

        from firescape import severity

        codes = np.array(mini_cdf.index[:2])
        det = severity.weibull_dnbr(
            0.5, mini_cdf.loc[codes, "Weibull_Lambda_Scale"].to_numpy(),
            mini_cdf.loc[codes, "Weibull_Kappa_Shape"].to_numpy())
        soft = severity.expected_dnbr(codes, 0.5, mini_cdf, sigma=0.0)
        assert np.allclose(det, soft)
        exc = severity.class_exceedance(codes, float(det[0]) - 1.0, 0.5,
                                        mini_cdf, sigma=0.0)
        assert exc[0] == 1.0

    def test_exceedance_matches_monte_carlo(self, mini_cdf):
        import numpy as np
        from scipy.stats import norm

        from firescape import severity

        code = int(mini_cdf.index[0])
        lam = float(mini_cdf.at[code, "Weibull_Lambda_Scale"])
        kap = float(mini_cdf.at[code, "Weibull_Kappa_Shape"])
        rng = np.random.default_rng(3)
        q = norm.cdf(norm.ppf(0.5) + 0.9 * rng.standard_normal(200_000))
        d = (lam * (-np.log(1 - q)) ** (1 / kap)) * 2000 - 1000
        thr = float(np.median(d))
        mc = float((d >= thr).mean())
        an = float(severity.class_exceedance([code], thr, 0.5, mini_cdf,
                                             sigma=0.9)[0])
        assert abs(mc - an) < 0.01
        mc_mean = float(d.mean())
        an_mean = float(severity.expected_dnbr([code], 0.5, mini_cdf,
                                               sigma=0.9)[0])
        assert abs(mc_mean - an_mean) < 5.0

    def test_field_reduces_and_disperses(self, mini_cdf):
        import numpy as np

        from firescape import severity

        code = int(mini_cdf.index[0])
        evt = np.full((60, 60), code)
        d0, s0 = severity.simulate_dnbr_field(evt, 0.5, mini_cdf, sigma=0.0,
                                              rng=1)
        det, _ = severity.simulate_dnbr(evt, 0.5, mini_cdf)
        assert np.allclose(d0, det, atol=0.5)
        d1, _ = severity.simulate_dnbr_field(evt, 0.5, mini_cdf, sigma=0.9,
                                             corr_px=3, rng=1)
        assert d1.std() > 50.0
        assert (s0 == severity.SRC_DIRECT).all()
