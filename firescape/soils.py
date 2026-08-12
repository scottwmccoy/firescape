"""STATSGO soil erodibility (KF-factor) adapter (milestone M1).

Wraps ``pfdf.data.usgs.statsgo`` (KFFACT cloud-optimized GeoTIFF). KF < 0 is
treated as nodata (Rossi et al. 2025 excluded negative values).
"""

from __future__ import annotations


def kf_factor(aoi):
    """Fetch the STATSGO KFFACT raster for an AOI (KF<0 -> nodata)."""
    raise NotImplementedError("soils.kf_factor lands in milestone M1")
