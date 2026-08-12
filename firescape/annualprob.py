"""Annual exceedance probability of the debris-flow rainfall threshold (M5).

Follows Rossi et al. (2025) Table 3. NOAA Atlas 14 gives the 15-minute
intensity for a set of recurrence intervals; the intensity-RI relationship is
log-linear, so two anchors (1-year and 50-year) define it per basin:

    RI = 10^(m * I15 + b)                                            (Eqn 6)
    m  = (log10(50) - log10(1)) / (I15_50yr - I15_1yr)               (Eqn 7)
    b  = -m * I15_1yr                                                (Eqn 8)
    P  = 1 - exp(-1 / RI)                                            (Eqn 9)

Evaluating Eqn 6 at the modeled rainfall threshold T (the intensity that
yields a 50% debris-flow likelihood) gives that basin's threshold recurrence
interval, and Eqn 9 converts it to an annual exceedance probability.

The full pre-fire annual probability of a postfire debris flow is
P(F) x P(R>T), where P(F) is annual burn probability (FSim / Wildfire Risk to
Communities). ``combined`` holds that seam; the P(F) fetcher is deferred.

Atlas 14 grids come from stormscape (validated unit conversion, region
auto-selection). Nevada is Atlas 14 region ``sw``; pass it explicitly rather
than relying on auto-detection for statewide runs, whose bounds straddle the
``sw``/``inw`` seam.
"""

from __future__ import annotations

import numpy as np

#: Atlas 14 anchors used for the log-linear fit (years).
ARI_LOW, ARI_HIGH = 1, 50


def climatology_i15(aoi, *, ari: int, region: str = "sw", cache_dir=None,
                    stat: str = "mean", series: str = "pds"):
    """15-minute intensity grid (mm/h) for one recurrence interval."""
    from stormscape import atlas14

    from firescape import paths

    cache_dir = str(cache_dir or (paths.cache_root() / "atlas14"))
    result = atlas14.climatology_field(aoi, durations=(15,), ari=ari, region=region,
                                       stat=stat, series=series, cache_dir=cache_dir)
    fields = result["fields"]
    key = next(iter(fields)) if 15 not in fields else 15
    return fields[key], result


def fit_log_linear(i15_1yr, i15_50yr):
    """Per-basin slope/intercept of log10(RI) vs I15 (Eqns 7-8)."""
    i1 = np.asarray(i15_1yr, dtype=float)
    i50 = np.asarray(i15_50yr, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = (np.log10(ARI_HIGH) - np.log10(ARI_LOW)) / (i50 - i1)
    m = np.where(np.isfinite(m) & (i50 > i1), m, np.nan)
    return m, -m * i1


def recurrence_interval(i15, m, b):
    """RI (years) of a 15-minute intensity, from the fitted line (Eqn 6)."""
    return np.power(10.0, np.asarray(m) * np.asarray(i15, dtype=float) + np.asarray(b))


def annual_probability(ri):
    """Annual exceedance probability from a recurrence interval (Eqn 9)."""
    ri = np.asarray(ri, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = 1.0 - np.exp(-1.0 / ri)
    return np.where(np.isfinite(p), p, np.nan)


def p_exceed_threshold(threshold_i15_mmh, i15_1yr, i15_50yr):
    """P(R>T): annual probability of exceeding each basin's modeled threshold."""
    m, b = fit_log_linear(i15_1yr, i15_50yr)
    return annual_probability(recurrence_interval(threshold_i15_mmh, m, b)), m, b


def combined(p_rt, p_f=None):
    """Annual PFDF probability = P(F) x P(R>T); returns P(R>T) if p_f is None."""
    if p_f is None:
        return p_rt
    return np.asarray(p_rt, dtype=float) * np.asarray(p_f, dtype=float)
