"""Offline tests for imagery-source adapters."""
import pytest

from firescape import sources


IMD = '''
BEGIN_GROUP = BAND_B
	absCalFactor = 1.783568e-02;
	effectiveBandwidth = 5.430000e-02;
END_GROUP = BAND_B
BEGIN_GROUP = BAND_N
	absCalFactor = 1.826500e-02;
	effectiveBandwidth = 9.890000e-02;
END_GROUP = BAND_N
	satId = "WV02";
	meanSunAz = 168.8;
	meanSunEl = 31.6;
	cloudCover = 0.001;
	meanProductGSD = 2.000;
'''


def test_parse_imd(tmp_path):
    f = tmp_path / "x.IMD"
    f.write_text(IMD)
    got = sources.parse_imd(f)
    assert got["sat"] == "WV02"
    assert got["sun_el"] == pytest.approx(31.6)
    assert got["gsd"] == pytest.approx(2.0)
    assert len(got["bands"]) == 2
    assert got["bands"][0]["abscal"] == pytest.approx(1.783568e-02)
    assert got["bands"][1]["bandwidth"] == pytest.approx(9.89e-02)


def test_band_maps_cover_pipeline_contract():
    for bm in (sources.MAXAR_8B, sources.MAXAR_4B):
        assert set(bm) == {"blue", "green", "red", "nir"}


def test_maxar_strips_requires_products(tmp_path):
    with pytest.raises(FileNotFoundError):
        sources.MaxarStrips(tmp_path)
