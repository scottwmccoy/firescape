"""Annual exceedance probability of the debris-flow rainfall threshold (M5).

P(R>T): per-basin log-linear recurrence fit through the NOAA Atlas 14 1-yr and
50-yr 15-min intensities (stormscape.atlas14.climatology_field, region 'sw'
passed explicitly for the pilot): RI = 10^(m*I15 + b); P = 1 - exp(-1/RI).
Later multiplied by annual burn probability P(F) (FSim / Wildfire Risk to
Communities) via ``combined`` - the P(F) fetcher is a deferred seam.
"""

from __future__ import annotations


def p_exceed_threshold(threshold_i15_mmh, aoi, *, region: str = "sw"):
    """P(R>T) per basin from Atlas 14 1-yr/50-yr I15 grids."""
    raise NotImplementedError("annualprob.p_exceed_threshold lands in milestone M5")


def combined(p_rt, p_f=None):
    """Annual PFDF probability = P(F) x P(R>T); returns P(R>T) if p_f is None."""
    if p_f is None:
        return p_rt
    return p_rt * p_f
