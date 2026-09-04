import numpy as np

from firescape import mtbs


def test_dnbr6_to_barc4_semantics():
    dnbr6 = np.array([[0, 1, 2, 3], [4, 5, 6, 3]])
    barc = mtbs.dnbr6_to_barc4(dnbr6)
    # 1..4 pass through; background(0), greening(5), non-mapping(6) -> nodata 0
    assert barc.tolist() == [[0, 1, 2, 3], [4, 0, 0, 3]]
    assert barc.dtype == np.uint8


def test_greening_and_mask_never_burned():
    barc = mtbs.dnbr6_to_barc4(np.array([5, 6]))
    assert (barc == 0).all()


def test_barc256_to_dnbr_matches_key_benson_defaults():
    """GTAC's BARC256 default breaks 76/110/187 are dNBR 105/275/660."""
    from firescape import mtbs

    assert list(mtbs.barc256_to_dnbr([76, 110, 187])) == [105.0, 275.0, 660.0]
    assert mtbs.barc256_to_dnbr(158) == 515.0            # Davis, = its SBS-recovered break


def test_normalize_thresholds_converts_only_barc256_baer_rows():
    import pandas as pd
    from firescape import mtbs

    df = pd.DataFrame({
        "event_id": ["davis", "line", "mtbs_fire", "sentinel", "ravg"],
        "map_prog": ["BAER", "BAER", "MTBS", "MTBS", "RAVG"],
        "low_t": [75, 40, 60, 0, 80], "mod_t": [158, 400, 310, 9999, 290],
        "high_t": [217, 675, 600, 9999, 9999],
    })
    out = mtbs.normalize_thresholds(df)
    assert out.loc[0, ["low_t", "mod_t", "high_t"]].tolist() == [100.0, 515.0, 810.0]
    assert out.loc[0, "threshold_scale"] == "barc256->dnbr"
    # BAER but already dNBR-scale, MTBS, RAVG and sentinels: untouched
    for i in (1, 2, 3, 4):
        assert out.loc[i, ["low_t", "mod_t", "high_t"]].tolist() == df.loc[i, ["low_t", "mod_t", "high_t"]].tolist()
        assert out.loc[i, "threshold_scale"] == "dnbr"
    # a mask works without a map_prog column
    out2 = mtbs.normalize_thresholds(df.drop(columns="map_prog"), is_barc=[True, True, False, False, False])
    assert out2.loc[0, "mod_t"] == 515.0 and out2.loc[1, "mod_t"] == 400
