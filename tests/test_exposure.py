"""Offline known-answer tests for the asset-exposure joins."""
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from firescape import annualprob, exposure

CRS = "EPSG:5070"


@pytest.fixture
def segments():
    # A: trunk at x=0 (4 km^2 -> 33 m width); B: feeder at x=2000 (10.2 m)
    return gpd.GeoDataFrame(
        {"Area_km2": [4.0, 0.01], "P_24mmh": [0.8, 0.2],
         "V_24mmh": [5000.0, 100.0], "H_24mmh": [3.0, 1.0],
         "I15_50": [12.0, 40.0]},
        geometry=[LineString([(0, 0), (0, 2000)]),
                  LineString([(2000, 0), (2000, 2000)])], crs=CRS)


@pytest.fixture
def assets():
    # straddles A's corridor / 300 m off B / far from everything
    return gpd.GeoDataFrame(
        {"name": ["on_trunk", "near_feeder", "remote"]},
        geometry=[box(-10, 500, 30, 560), Point(2300, 1000),
                  Point(9000, 1000)], crs=CRS)


def test_corridor_buffers_width_law(segments):
    buf = exposure.corridor_buffers(segments, pad_m=0.0)
    w = 9.0 + 12.0 * np.sqrt(4.0)  # 33 m
    want = w * 2000 + np.pi * (w / 2) ** 2  # rectangle + round caps
    assert buf.iloc[0].area == pytest.approx(want, rel=1e-3)


def test_hazard_at_assets(assets, segments):
    hz = exposure.hazard_at_assets(assets, segments)
    assert hz["exposed"].tolist() == [True, False, False]
    assert hz["dist_m"].iloc[0] == 0.0
    assert hz["dist_m"].iloc[1] == pytest.approx(300.0, abs=1.0)
    assert np.isnan(hz["dist_m"].iloc[2])
    assert hz["P_24mmh"].tolist()[:2] == [0.8, 0.2]
    assert np.isnan(hz["P_24mmh"].iloc[2])
    assert hz["n_deliver"].tolist() == [1, 0, 0]
    assert hz["seg_id"].iloc[0] == 0 and hz["seg_id"].iloc[1] == 1


def test_hazard_at_assets_empty_segments(assets, segments):
    hz = exposure.hazard_at_assets(assets, segments.iloc[0:0])
    assert not hz["exposed"].any()
    assert hz["dist_m"].isna().all()


def test_geographic_crs_refused(assets, segments):
    with pytest.raises(ValueError, match="projected"):
        exposure.hazard_at_assets(assets.to_crs(4326), segments.to_crs(4326))


def test_combine_best(assets, segments):
    # unit 1 sees only the feeder; unit 2 only the trunk — combined keeps the
    # trunk hit for asset 0 and the feeder near-miss for asset 1
    f1 = exposure.hazard_at_assets(assets, segments.iloc[[1]])
    f2 = exposure.hazard_at_assets(assets, segments.iloc[[0]])
    best = exposure.combine_best([f1, f2])
    assert bool(best.loc[0, "exposed"]) and best.loc[0, "P_24mmh"] == 0.8
    assert not bool(best.loc[1, "exposed"])
    assert best.loc[1, "dist_m"] == pytest.approx(300.0, abs=1.0)
    assert best.loc[1, "P_24mmh"] == 0.2
    assert best.loc[0, "n_deliver"] == 1


def test_site_annual(assets, segments):
    hz = exposure.hazard_at_assets(assets, segments)
    basins = gpd.GeoDataFrame(
        {"I15_1yr": [10.0], "I15_50yr": [40.0], "P_F": [0.02]},
        geometry=[box(-500, -500, 3000, 3000)], crs=CRS)
    out = exposure.site_annual(hz, assets, basins)
    p_want, _, _ = annualprob.p_exceed_threshold(12.0, 10.0, 40.0)
    assert out["P_RgtT_site"].iloc[0] == pytest.approx(float(p_want))
    assert out["P_annual_site"].iloc[0] == pytest.approx(float(p_want) * 0.02)
    # asset 1 needs 40 mm/h vs asset 0's 12 -> rarer trigger, lower rate
    assert out["P_RgtT_site"].iloc[1] < out["P_RgtT_site"].iloc[0]
    assert out["basin_dist_m"].iloc[0] == 0.0
    # remote site is >5 km from the basin: no climatology, no annual rate
    assert np.isnan(out["P_annual_site"].iloc[2])


def test_nearest_receptor_names_and_radius(assets):
    waters = gpd.GeoDataFrame(
        {"name": ["Sixmile Creek", "Far Lake"], "kind": ["perennial", "lake"]},
        geometry=[box(-600, 400, -400, 600), box(20000, 0, 20100, 100)],
        crs=CRS)
    rec = exposure.nearest_receptor(assets, waters, cols=("name", "kind"),
                                    max_m=5000.0)
    assert rec.loc[0, "water_dist_m"] == pytest.approx(390.0, abs=1.0)
    assert rec.loc[0, "name"] == "Sixmile Creek"
    assert rec.loc[0, "kind"] == "perennial"
    # the remote site's nearest water is beyond the question radius
    assert np.isnan(rec.loc[2, "water_dist_m"]) and pd.isna(rec.loc[2, "name"])


def test_within_any(assets):
    blm = gpd.GeoDataFrame(
        geometry=[box(-1000, 0, 1000, 2000)], crs=CRS)
    flag = exposure.within_any(assets, blm)
    assert flag.tolist() == [True, False, False]


def test_water_distance_and_rank(assets, segments):
    waters = gpd.GeoDataFrame(
        geometry=[box(-600, 400, -400, 600)], crs=CRS)
    d = exposure.water_distance(assets, waters)
    assert d.iloc[0] == pytest.approx(390.0, abs=1.0)  # box edge to x=-10
    basins = gpd.GeoDataFrame(
        {"I15_1yr": [10.0], "I15_50yr": [40.0], "P_F": [0.02]},
        geometry=[box(-500, -500, 3000, 3000)], crs=CRS)
    hz = exposure.site_annual(
        exposure.hazard_at_assets(assets, segments), assets, basins)
    ranked = exposure.rank(hz)
    assert ranked.index.tolist()[0] == 0  # the exposed site leads
    assert ranked["rank"].tolist() == [1, 2, 3]
