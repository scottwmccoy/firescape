"""LANDFIRE Existing Vegetation Type adapter (milestone M2).

Thin, version-pinned wrapper over ``pfdf.data.landfire`` (which is exempt from
pfdf's backwards-compatibility policy - hence the adapter). Will own: EVT
raster fetch for an AOI, the back-sampling seam for classes lacking Staley
2018 CDF parameters (deferred), and the open-water class extraction used by
the Rossi water mask.
"""

from __future__ import annotations


def evt(aoi, *, version: str = "latest"):
    """Fetch the LANDFIRE EVT raster for an AOI."""
    raise NotImplementedError("landfire.evt lands in milestone M2")
