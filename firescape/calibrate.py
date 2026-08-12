"""Nevada P_dsim calibration - the Rossi et al. (2025) DFL method (milestone M3).

Per calibration fire: delineate basins from observed MTBS severity; drop
basins <75% inside the perimeter or with median dNBR below the fire's
unburned-low threshold; compute "observed" debris-flow likelihood at
I15 = 24 mm/h; sweep P_dsim over 0.01..0.99 to find each basin's best match;
fire P_dsim = median(basins); regional P_dsim = median(fires); regional BARC
break = median of fire low-moderate thresholds. Per-fire intermediates are
cached under interim/calib/<event_id>/ keyed by a criteria hash so swapping
fire sets (CSV) re-medians without recomputation.
"""

from __future__ import annotations


def fire_calibration(event_id: str, cfg):
    """Calibrate P_dsim for one MTBS fire; returns a cacheable FireCalib."""
    raise NotImplementedError("calibrate.fire_calibration lands in milestone M3")


def regional_summary(fire_calibs, regions):
    """Median fire calibrations into per-region values; write calibration TOML."""
    raise NotImplementedError("calibrate.regional_summary lands in milestone M3")
