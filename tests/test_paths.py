import json

from firescape import paths


def test_roots_respect_env(tmp_path):
    # _isolated_roots fixture points FIRESCAPE_DATA/CACHE at tmp
    assert str(paths.data_root()).startswith(str(tmp_path))
    assert str(paths.cache_root()).startswith(str(tmp_path))


def test_dir_helpers_create(tmp_path):
    d = paths.raw_dir("mtbs", "fires")
    assert d.is_dir()
    assert d == paths.data_root() / "raw" / "mtbs" / "fires"


def test_provenance_sidecar(tmp_path):
    f = paths.raw_dir("x") / "thing.txt"
    f.write_text("hello")
    side = paths.write_provenance(f, url="https://example.org/thing", note="test")
    meta = json.loads(side.read_text())
    assert side.name == "thing.txt.provenance.json"
    assert meta["url"] == "https://example.org/thing"
    assert meta["size_bytes"] == 5
    assert meta["sha256"] and len(meta["sha256"]) == 64
    assert meta["retrieved_utc"].endswith("+00:00")


def test_package_data_resolves_inside_the_package():
    from firescape import paths

    f = paths.package_data("calibration", "statewide_v1_4.toml")
    assert f.is_file() and f.parent.parent.name == "data"
    assert "firescape" in f.parts


def test_research_root_env_and_default(monkeypatch, tmp_path):
    from firescape import paths

    monkeypatch.setenv("FIRESCAPE_DATA", str(tmp_path / "PreFireAssessment"))
    monkeypatch.delenv("FIRESCAPE_RESEARCH", raising=False)
    assert paths.research_root() == tmp_path
    monkeypatch.setenv("FIRESCAPE_RESEARCH", str(tmp_path / "elsewhere"))
    assert paths.research_root() == tmp_path / "elsewhere"


def test_data_root_refuses_to_invent_a_box_folder(monkeypatch, tmp_path):
    import pytest
    from firescape import paths

    monkeypatch.delenv("FIRESCAPE_DATA", raising=False)
    monkeypatch.setattr(paths, "_BOX_DEFAULT", tmp_path / "no-such-box")
    with pytest.raises(FileNotFoundError, match="FIRESCAPE_DATA"):
        paths.data_root()
    monkeypatch.setattr(paths, "_BOX_DEFAULT", tmp_path)
    assert paths.data_root() == tmp_path


def test_python_executable_env_override(monkeypatch):
    import sys
    from firescape import paths

    monkeypatch.delenv("FIRESCAPE_PYTHON", raising=False)
    assert paths.python_executable() == sys.executable
    monkeypatch.setenv("FIRESCAPE_PYTHON", "/some/env/bin/python")
    assert paths.python_executable() == "/some/env/bin/python"
