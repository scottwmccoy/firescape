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
