"""STATSGO soil erodibility (KF-factor) via pfdf.data.usgs.statsgo.

KF < 0 is treated as nodata (Rossi et al. 2025 excluded negative values).
The pfdf.data modules are exempt from pfdf's backwards-compatibility policy,
hence this adapter.
"""

from __future__ import annotations

import numpy as np


def kf_factor(bounds):
    """Fetch the STATSGO KFFACT raster for ``bounds`` (a pfdf BoundsInput,
    e.g. a Raster or BoundingBox). Returns a pfdf Raster with KF<0 -> NaN."""
    from pfdf.data.usgs import statsgo
    from pfdf.raster import Raster

    kf = statsgo.read("KFFACT", bounds=bounds)
    values = kf.values.astype("float64", copy=True)
    if kf.nodata is not None:
        values[values == kf.nodata] = np.nan
    values[values < 0] = np.nan
    return Raster.from_array(values, nodata=np.nan, spatial=kf)
