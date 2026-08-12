"""Burn-severity simulation: Staley et al. (2018) EVT-dNBR Weibull CDFs.

Implements Eqn 1 of Rossi et al. (2025):

    SimdNBR = [ lambda * (-ln(1 - P_dsim))**(1/kappa) ] * 2000 - 1000

where (lambda, kappa) are the per-EVT-class Weibull CDF parameters from the
Staley et al. (2018) data release (doi:10.5066/P9TKYL5K; packaged at
``firescape/data/staley2018/CDFParameters.txt``) and P_dsim is the regionally
calibrated percentile at which severity is simulated.

dNBR here follows the MTBS convention (raw index x1000), matching both the CDF
release and what ``pfdf.severity.estimate`` / ``pfdf`` M1 expect.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

#: Transform from Weibull-normalized [0, 1] space back to dNBR(x1000).
_DNBR_SCALE = 2000.0
_DNBR_OFFSET = -1000.0

#: Fallback class for EVT codes lacking CDF parameters. Rossi et al. used
#: 7294 ("barren"), but that is a LANDFIRE-Remap-era code ABSENT from the
#: Staley 2018 table (verified: fits use the older code series). We fall back
#: to 3001 "Inter-Mountain Basins Sparsely Vegetated Systems" — the Great
#: Basin sparse/barren analog that IS in the table. Real EVT crosswalking /
#: back-sampling belongs to the landfire.py adapter (milestone M2).
BARREN_EVT_CODE = 3001

#: Source-flag values written alongside simulated dNBR.
SRC_NODATA, SRC_DIRECT, SRC_FALLBACK = 0, 1, 2


def load_cdf_table(path: str | Path | None = None) -> pd.DataFrame:
    """Load the Weibull CDF parameter table, indexed by EVT code.

    Columns: N, Weibull_Lambda_Scale, Weibull_Kappa_Shape, Weibull_R2,
    Weibull_RMSE, CLASSNAME.
    """
    if path is None:
        ref = resources.files("firescape") / "data" / "staley2018" / "CDFParameters.txt"
        with resources.as_file(ref) as p:
            table = pd.read_csv(p)
    else:
        table = pd.read_csv(path)
    required = {"EVT_Code", "Weibull_Lambda_Scale", "Weibull_Kappa_Shape"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"CDF table missing columns: {sorted(missing)}")
    return table.set_index("EVT_Code").sort_index()


def weibull_dnbr(pdsim: float, lam: np.ndarray | float, kappa: np.ndarray | float):
    """Invert the Weibull CDF at percentile ``pdsim`` -> dNBR(x1000)."""
    if not 0.0 < pdsim < 1.0:
        raise ValueError(f"pdsim must be in (0, 1), got {pdsim}")
    z = lam * (-np.log1p(-pdsim)) ** (1.0 / np.asarray(kappa, dtype=float))
    return z * _DNBR_SCALE + _DNBR_OFFSET


def simulate_dnbr(
    evt: np.ndarray,
    pdsim: float,
    cdf_table: pd.DataFrame | None = None,
    *,
    fallback_code: int | None = BARREN_EVT_CODE,
    evt_nodata: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a dNBR(x1000) array from an EVT-code array at percentile pdsim.

    Returns ``(dnbr, src)`` where ``dnbr`` is float32 with NaN where no
    parameters exist (and no fallback applies), and ``src`` is uint8:
    0 = nodata, 1 = class had CDF parameters, 2 = fallback class used.
    """
    if cdf_table is None:
        cdf_table = load_cdf_table()

    evt = np.asarray(evt)
    codes, inverse = np.unique(evt, return_inverse=True)

    lam = cdf_table["Weibull_Lambda_Scale"].reindex(codes).to_numpy(dtype=float)
    kap = cdf_table["Weibull_Kappa_Shape"].reindex(codes).to_numpy(dtype=float)
    src_codes = np.where(np.isfinite(lam), SRC_DIRECT, SRC_NODATA).astype(np.uint8)

    if fallback_code is not None:
        if fallback_code not in cdf_table.index:
            raise ValueError(f"fallback EVT code {fallback_code} not in CDF table")
        fb_lam = float(cdf_table.at[fallback_code, "Weibull_Lambda_Scale"])
        fb_kap = float(cdf_table.at[fallback_code, "Weibull_Kappa_Shape"])
        missing = ~np.isfinite(lam)
        lam = np.where(missing, fb_lam, lam)
        kap = np.where(missing, fb_kap, kap)
        src_codes = np.where(missing, SRC_FALLBACK, src_codes).astype(np.uint8)

    if evt_nodata is not None:
        is_nd = codes == evt_nodata
        lam[is_nd] = np.nan
        src_codes[is_nd] = SRC_NODATA

    per_code = np.where(
        np.isfinite(lam), weibull_dnbr(pdsim, np.nan_to_num(lam, nan=1.0), np.nan_to_num(kap, nan=1.0)), np.nan
    ).astype(np.float32)

    dnbr = per_code[inverse].reshape(evt.shape)
    src = src_codes[inverse].reshape(evt.shape)
    return dnbr, src


def classify_barc4(dnbr: np.ndarray, breaks: tuple[float, float, float]) -> np.ndarray:
    """Classify dNBR(x1000) into BARC4 classes (1=unburned/very low ... 4=high).

    ``breaks`` are the (unburned-low, low-moderate, moderate-high) thresholds.
    NaN maps to 0 (nodata). Matches pfdf.severity.classification numbering.
    """
    t1, t2, t3 = breaks
    if not (t1 < t2 < t3):
        raise ValueError(f"breaks must increase, got {breaks}")
    dnbr = np.asarray(dnbr, dtype=float)
    barc = np.ones(dnbr.shape, dtype=np.uint8)
    barc[dnbr >= t1] = 2
    barc[dnbr >= t2] = 3
    barc[dnbr >= t3] = 4
    barc[~np.isfinite(dnbr)] = 0
    return barc


def coverage_report(evt: np.ndarray, cdf_table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tabulate EVT codes in an array vs. CDF-parameter availability.

    Answers "what fraction of this AOI lacks Staley 2018 parameters?" —
    the key Nevada-coverage diagnostic (plan milestone M2).
    """
    if cdf_table is None:
        cdf_table = load_cdf_table()
    codes, counts = np.unique(np.asarray(evt), return_counts=True)
    rep = pd.DataFrame({"EVT_Code": codes, "n_pixels": counts})
    rep["fraction"] = rep["n_pixels"] / rep["n_pixels"].sum()
    rep["has_params"] = rep["EVT_Code"].isin(cdf_table.index)
    rep = rep.merge(
        cdf_table[["CLASSNAME"]], left_on="EVT_Code", right_index=True, how="left"
    )
    return rep.sort_values("n_pixels", ascending=False, ignore_index=True)
