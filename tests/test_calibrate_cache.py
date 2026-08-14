"""Decomposition cache: instant re-solves without raster work.

A synthetic decomposition (three identical single-class basins whose observed
T/F sit exactly at the class's P=0.37 values) must yield pdsim=0.37 from the
fast path — including from a sibling cache with a different break/sigma tag,
which is the whole point (recalibration without re-delineation).
"""

import json

import numpy as np

from firescape import calibrate, paths, severity

EID = "NV0000000000000000001"
CODE = 3019          # Great Basin Pinyon-Juniper, present in the Staley table
P_TRUE = 0.37
BRK = 281.0


def _write_decomp(tag):
    table = severity.load_cdf_table()
    d_true = float(severity.weibull_dnbr(
        P_TRUE, table.at[CODE, "Weibull_Lambda_Scale"],
        table.at[CODE, "Weibull_Kappa_Shape"]))
    n = 3
    d = paths.interim_dir("calib", f"{EID}__{tag}")
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        d / "decomp.npz",
        W=np.ones((n, 1), dtype=np.float32),
        Ws=np.ones((n, 1), dtype=np.float32),
        classes=np.array([CODE], dtype=np.int32),
        ids=np.arange(1, n + 1),
        T_obs=np.full(n, 1.0 if d_true >= BRK else 0.0),
        F_obs=np.full(n, d_true / 1000.0),
        sel=np.ones(n, dtype=bool),
        area=np.full(n, 1.0), ratio_in=np.ones(n), med_dnbr=np.full(n, 400.0),
        n0=n, validation_dT=0.0, validation_dF=0.0)
    return d


def test_fast_path_recovers_known_pdsim():
    _write_decomp("lf2016_b281")
    fc = calibrate.fire_calibration(EID, (125.0, 281.0), BRK,
                                    crosswalk={}, tag="lf2016_b281")
    assert fc.ok and abs(fc.pdsim - P_TRUE) < 1e-9
    meta = json.loads((fc.cache_dir / "fire.json").read_text())
    assert meta["solve_source"] == "decomp-cache"


def test_cross_tag_reuse_with_dispersion():
    src = _write_decomp("lf2016_b281")
    new_tag = "lf2016_b325_d91"
    res = calibrate.fire_calibration(
        EID, (125.0, 325.0), 325.0, crosswalk={},
        cdf_tables={"staley2018": severity.load_cdf_table()},
        dispersed_sigma=0.91, tag=new_tag)
    fc = res["staley2018"]
    assert fc.ok and 0.01 <= fc.pdsim <= 0.99
    new_dir = paths.interim_dir("calib", f"{EID}__{new_tag}")
    assert (new_dir / "decomp.npz").exists() and new_dir != src
    meta = json.loads((new_dir / "fire.json").read_text())
    assert "staley2018_disp" in meta["pdsim_by_table"]
    assert meta["solve_source"] == "decomp-cache"
