"""Configuration: calibration TOMLs and USGS/wildcat model defaults.

Calibration values (regional P_dsim, BARC breaks, calibration fire sets) are
TOML *files*, not Python constants: ``firescape calibrate`` writes one into the
run directory, and adopting a calibration means copying it into
``firescape/data/calibration/`` and committing it (git history = provenance).
Python constants are reserved for immutable science (model reference values).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

#: USGS-standard reference storm (~1-yr RI 15-min intensity; Staley et al. 2020).
I15_REFERENCE_MMH = 24.0

#: Standard I15 sweep used by USGS assessments (wildcat default), mm/h.
I15_SWEEP_MMH = (16.0, 20.0, 24.0, 40.0)


@dataclass(frozen=True)
class FilterDefaults:
    """USGS stream-segment delineation/filter defaults (wildcat / tutorial 05)."""

    min_area_km2: float = 0.025
    min_burned_area_km2: float = 0.01
    max_length_m: float = 500.0
    max_area_km2: float = 8.0
    min_burn_ratio: float = 0.25
    min_slope: float = 0.12          # gradient, not degrees
    max_developed_area_km2: float = 0.025
    max_confinement_deg: float = 174.0


@dataclass(frozen=True)
class RegionCalibration:
    """Calibrated severity-simulation parameters for one prefire region.

    barc_breaks are dNBR(x1000) thresholds (unburned-low, low-moderate,
    moderate-high). Only the low-moderate break matters for the hazard models
    (moderate+high are pooled); the unburned-low break is used to exclude
    unburned basins during calibration.
    """

    pdsim: float
    barc_breaks: tuple[float, float, float]
    fires: tuple[str, ...] = ()      # MTBS event_ids used to calibrate


@dataclass(frozen=True)
class Calibration:
    meta: dict = field(default_factory=dict)
    regions: dict[str, RegionCalibration] = field(default_factory=dict)

    def region(self, name: str) -> RegionCalibration:
        try:
            return self.regions[name]
        except KeyError:
            raise KeyError(
                f"region {name!r} not in calibration "
                f"(has: {sorted(self.regions)})"
            ) from None


def _parse(doc: dict) -> Calibration:
    regions = {}
    for name, sect in doc.get("regions", {}).items():
        breaks = sect["barc_breaks"]
        if len(breaks) != 3:
            raise ValueError(f"region {name!r}: barc_breaks needs exactly 3 values, got {breaks}")
        regions[name] = RegionCalibration(
            pdsim=float(sect["pdsim"]),
            barc_breaks=tuple(float(b) for b in breaks),
            fires=tuple(sect.get("fires", ())),
        )
    return Calibration(meta=dict(doc.get("meta", {})), regions=regions)


def load_calibration(path: str | Path) -> Calibration:
    with open(path, "rb") as f:
        return _parse(tomllib.load(f))


def packaged_calibration(name: str = "default_rossi_ca") -> Calibration:
    """Load a calibration TOML shipped inside the package."""
    ref = resources.files("firescape") / "data" / "calibration" / f"{name}.toml"
    return _parse(tomllib.loads(ref.read_text()))
