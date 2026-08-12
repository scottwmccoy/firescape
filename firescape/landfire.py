"""LANDFIRE Existing Vegetation Type via pfdf.data.landfire (LFPS).

LFPS is a job-queue API (submit -> poll -> download) and requires an email
address as a usage-statistics field. We never hardcode one: pass ``email`` or
set ``$FIRESCAPE_LFPS_EMAIL``.

Version strategy (milestone M2): the Staley 2018 CDF table indexes the
pre-Remap LANDFIRE class codes (fits built from 2001-2014 burns), so recent
EVT layers may carry codes without CDF parameters. ``evt()`` takes an explicit
LFPS layer name (e.g. "240EVT" for the LF 2023 release, "140EVT" for LF 1.4.0
whose codes match the CDF vintage); the coverage_report in severity.py
quantifies the tradeoff. Full Rossi-style back-sampling across versions is a
deferred seam.
"""

from __future__ import annotations

import os


def _email(email: str | None) -> str:
    email = email or os.environ.get("FIRESCAPE_LFPS_EMAIL")
    if not email:
        raise ValueError(
            "LFPS requires an email (usage tracking). Pass email= or set "
            "$FIRESCAPE_LFPS_EMAIL."
        )
    return email


def evt(bounds, layer: str, *, email: str | None = None,
        max_job_time: float | None = 600.0):
    """Fetch a LANDFIRE EVT raster for ``bounds`` (BoundsInput with CRS).

    ``layer`` is the LFPS layer name (https://lfps.usgs.gov/products), e.g.
    "240EVT". Returns a pfdf Raster.
    """
    from pfdf.data import landfire

    return landfire.read(layer, bounds, _email(email), max_job_time=max_job_time)


def available_products(search: str = "EVT") -> list[str]:
    """List LFPS product layer names containing ``search`` (best effort)."""
    from pfdf.data import landfire

    try:
        prods = landfire.products()
    except TypeError:
        prods = landfire.products
    names = []
    try:
        for p in prods:
            name = p if isinstance(p, str) else p.get("layerName") or p.get("name") or str(p)
            if search.lower() in str(name).lower():
                names.append(str(name))
    except TypeError:
        return []
    return sorted(set(names))
