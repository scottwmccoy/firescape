"""USGS hazard models via pfdf: likelihood (M1), volume (G14), class (C10).

EVERY unit conversion between our world and pfdf's lives in this module (and
delineate.py). The traps (see CLAUDE.md):

- dNBR passed around firescape is ALWAYS the x1000 MTBS-style integer scale.
- Staley-2017 rainfall enters as ACCUMULATION (mm per duration):
  I15 = 24 mm/h  ->  R15 = 6 mm  (pfdf.utils.intensity.to_accumulation).
- Slopes are gradients (rise/run), never degrees.
- Thresholds returned by ``threshold_i15`` are converted back to mm/h.

pfdf is imported lazily so `import firescape` works without it.
"""

from __future__ import annotations

import numpy as np

from firescape.config import I15_REFERENCE_MMH

#: Staley et al. (2017) M1 15-minute coefficients — for reference and tests
#: only; runtime coefficients come from pfdf's M1.parameters().
M1_15MIN = {"B": -3.63, "Ct": 0.41, "Cf": 0.67, "Cs": 0.70}


def _s17():
    from pfdf.models import staley2017

    return staley2017


def m1_inputs(segments, barc4_raster, slopes_raster, dnbr_raster, kf_raster,
              *, omitnan: bool = True):
    """Compute the M1 (T, F, S) segment vectors from catchment rasters.

    ``barc4_raster``: BARC4 severity (1..4); moderate+high mask derived here.
    ``dnbr_raster``: dNBR x1000. ``slopes_raster``: gradients.
    Returns (T, F, S) numpy vectors aligned with ``segments``.
    """
    from pfdf import severity as pfdf_severity

    s17 = _s17()
    moderate_high = pfdf_severity.mask(barc4_raster, ["moderate", "high"])
    return s17.M1.variables(
        segments, moderate_high, slopes_raster, dnbr_raster, kf_raster,
        omitnan=omitnan,
    )


def likelihood_m1(T, F, S, *, i15_mmh: float = I15_REFERENCE_MMH,
                  durations=(15,)) -> np.ndarray:
    """Debris-flow likelihood for a design storm given in mm/h."""
    from pfdf.utils import intensity

    s17 = _s17()
    B, Ct, Cf, Cs = s17.M1.parameters(durations=list(durations))
    R = intensity.to_accumulation(i15_mmh, durations=list(durations))
    return np.squeeze(s17.likelihood(R, B, Ct, T, Cf, F, Cs, S))


def threshold_i15(T, F, S, *, p: float = 0.5, durations=(15,)) -> np.ndarray:
    """Rainfall intensity (mm/h) that yields likelihood ``p`` (inversion)."""
    from pfdf.utils import intensity

    s17 = _s17()
    B, Ct, Cf, Cs = s17.M1.parameters(durations=list(durations))
    R = s17.accumulation(p, B, Ct, T, Cf, F, Cs, S, screen=True)
    return np.squeeze(intensity.from_accumulation(R, durations=list(durations)))


def volume_g14(bmh_km2, relief_m, *, i15_mmh: float = I15_REFERENCE_MMH,
               CI: float = 0.95):
    """Gartner et al. (2014) emergency volume model. i15 in mm/h directly.

    Returns (V, Vmin, Vmax) in m^3.
    """
    from pfdf.models import gartner2014 as g14

    return g14.emergency(i15_mmh, bmh_km2, relief_m, CI=CI)


def combined_c10(likelihoods, volumes):
    """Cannon et al. (2010) combined relative-hazard classification (1-3)."""
    from pfdf.models import cannon2010 as c10

    return c10.hazard(likelihoods, volumes)


def logistic_m1_reference(T, F, S, R_mm):
    """Pure-numpy M1 (15-min) reference implementation for known-answer tests."""
    c = M1_15MIN
    X = c["B"] + (c["Ct"] * T + c["Cf"] * F + c["Cs"] * S) * R_mm
    return 1.0 / (1.0 + np.exp(-X))
