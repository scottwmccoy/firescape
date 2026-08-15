"""Offline tests for the PlanetScope access layer.

Everything network-shaped is monkeypatched; the live API was probed once
(2026-08-15) and the response shapes here mirror what it actually returned.
"""
import json

import pytest

from firescape import planetscope as ps


# --------------------------------------------------------------------------- #
# key resolution
# --------------------------------------------------------------------------- #
def test_key_prefers_arg_then_env_then_file(tmp_path, monkeypatch):
    f = tmp_path / "planet_api_key"
    f.write_text("PLAKfilekey\n")
    monkeypatch.setattr(ps, "KEY_FILE", f)
    monkeypatch.setenv("PL_API_KEY", "PLAKenvkey")
    assert ps._key("PLAKargkey") == "PLAKargkey"
    assert ps._key() == "PLAKenvkey"
    monkeypatch.delenv("PL_API_KEY")
    assert ps._key() == "PLAKfilekey"          # stripped of the newline


def test_key_missing_is_a_helpful_error(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "KEY_FILE", tmp_path / "absent")
    monkeypatch.delenv("PL_API_KEY", raising=False)
    with pytest.raises(ps.PlanetError, match="PL_API_KEY"):
        ps._key()


# --------------------------------------------------------------------------- #
# filter composition
# --------------------------------------------------------------------------- #
def test_search_filter_layout():
    geom = {"type": "Point", "coordinates": [-119.7, 39.5]}
    f = ps._search_filter(geom, "2026-07-01", "2026-08-15", min_clear=90)
    kinds = [c["type"] for c in f["config"]]
    assert f["type"] == "AndFilter"
    assert kinds == ["GeometryFilter", "DateRangeFilter", "RangeFilter"]
    date = f["config"][1]["config"]
    assert date == {"gte": "2026-07-01T00:00:00Z", "lte": "2026-08-15T00:00:00Z"}
    assert f["config"][2]["field_name"] == "clear_percent"
    assert f["config"][2]["config"] == {"gte": 90.0}


def test_rfc3339_accepts_datetimes_and_strings():
    from datetime import datetime, timezone
    assert ps._rfc3339("2026-07-01") == "2026-07-01T00:00:00Z"
    t = datetime(2026, 7, 1, 12, 30, tzinfo=timezone.utc)
    assert ps._rfc3339(t) == "2026-07-01T12:30:00Z"
    assert ps._rfc3339(datetime(2026, 7, 1, 12, 30)) == "2026-07-01T12:30:00Z"


def test_aoi_registry_is_sane():
    for name, (w, s, e, n) in ps.AOIS.items():
        assert w < e and s < n, name
        assert -125 < w < -113 and 34 < s < 43, name   # Nevada-ish


# --------------------------------------------------------------------------- #
# search (paginated, mocked)
# --------------------------------------------------------------------------- #
def _feature(fid, acquired, clear):
    return {"id": fid, "geometry": {
                "type": "Polygon",
                "coordinates": [[[-119.8, 39.4], [-119.6, 39.4],
                                 [-119.6, 39.6], [-119.8, 39.6],
                                 [-119.8, 39.4]]]},
            "properties": {"acquired": acquired, "instrument": "PSB.SD",
                           "clear_percent": clear, "cloud_cover": 0.01,
                           "quality_category": "standard",
                           "publishing_stage": "finalized",
                           "ground_control": True,
                           "satellite_id": "24de",
                           "sun_elevation": 60.0, "view_angle": 3.0}}


class _Resp:
    ok = True
    url = "https://api.planet.com/test"

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_search_paginates_and_sorts(monkeypatch):
    page1 = {"features": [_feature("b", "2026-08-02T18:00:00Z", 95),
                          _feature("c", "2026-08-03T18:00:00Z", 91)],
             "_links": {"_next": "https://api.planet.com/page2"}}
    page2 = {"features": [_feature("a", "2026-08-01T18:00:00Z", 99)],
             "_links": {}}
    posted = {}

    def fake_post(url, params=None, auth=None, json=None, timeout=None):
        posted["payload"] = json
        return _Resp(page1)

    def fake_get(url, auth=None, timeout=None):
        assert url == "https://api.planet.com/page2"
        return _Resp(page2)

    monkeypatch.setattr(ps.requests, "post", fake_post)
    monkeypatch.setattr(ps.requests, "get", fake_get)
    monkeypatch.setenv("PL_API_KEY", "PLAKtest")

    got = ps.search((-119.72, 39.45, -119.63, 39.53),
                    "2026-08-01", "2026-08-04", min_clear=90)
    assert list(got["id"]) == ["a", "b", "c"]          # sorted by acquired
    assert got.crs.to_epsg() == 4326
    assert {"acquired", "clear_percent", "instrument"} <= set(got.columns)
    assert posted["payload"]["item_types"] == ["PSScene"]
    kinds = [c["type"] for c in posted["payload"]["filter"]["config"]]
    assert "GeometryFilter" in kinds and "RangeFilter" in kinds


def test_search_empty_returns_empty_gdf(monkeypatch):
    monkeypatch.setattr(ps.requests, "post",
                        lambda *a, **k: _Resp({"features": [], "_links": {}}))
    monkeypatch.setenv("PL_API_KEY", "PLAKtest")
    got = ps.search("hidden_valley", "2026-01-01", "2026-01-02")
    assert len(got) == 0


# --------------------------------------------------------------------------- #
# order payloads
# --------------------------------------------------------------------------- #
def test_order_request_payload():
    req = ps.order_request("hv_pre", ["s1", "s2"], "hidden_valley")
    assert req["name"] == "hv_pre"
    (prod,) = req["products"]
    assert prod["item_ids"] == ["s1", "s2"]
    assert prod["product_bundle"] == "analytic_sr_udm2"
    clip, harm = req["tools"]
    ring = clip["clip"]["aoi"]["coordinates"][0]
    assert ring[0] == ring[-1]                          # closed ring
    assert harm == {"harmonize": {"target_sensor": "Sentinel-2"}}
    assert json.dumps(req)                              # JSON-serialisable


def test_order_request_variants():
    req = ps.order_request("x", ["s1"], "hidden_valley",
                           bands="8b", harmonize=False)
    assert req["products"][0]["product_bundle"] == "analytic_8b_sr_udm2"
    assert [list(t) for t in req["tools"]] == [["clip"]]
    with pytest.raises(ps.PlanetError, match="bands"):
        ps.order_request("x", ["s1"], "hidden_valley", bands="12b")
    with pytest.raises(ps.PlanetError, match="item id"):
        ps.order_request("x", [], "hidden_valley")
