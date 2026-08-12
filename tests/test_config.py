import pytest

from firescape import config


def test_packaged_bootstrap_calibration():
    cal = config.packaged_calibration("default_rossi_ca")
    pilot = cal.region("pilot")
    assert pilot.pdsim == pytest.approx(0.38)
    assert pilot.barc_breaks == (125.0, 272.0, 500.0)
    assert "CA-derived" in cal.meta.get("warning", "")


def test_unknown_region_raises():
    cal = config.packaged_calibration("default_rossi_ca")
    with pytest.raises(KeyError, match="pilot"):
        cal.region("mojave")


def test_bad_breaks_rejected(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("[regions.x]\npdsim = 0.5\nbarc_breaks = [125.0, 250.0]\n")
    with pytest.raises(ValueError, match="exactly 3"):
        config.load_calibration(bad)


def test_usgs_defaults_frozen():
    d = config.FilterDefaults()
    assert d.min_area_km2 == 0.025
    assert d.max_area_km2 == 8.0
    assert d.min_slope == 0.12
    assert config.I15_REFERENCE_MMH == 24.0
