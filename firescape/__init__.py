"""firescape: pre-fire postfire-debris-flow hazard assessment for Nevada.

Simulated burn severity (Rossi et al. 2025 method; Staley et al. 2018 EVT-dNBR
Weibull CDFs) driving the USGS pfdf hazard models. Storm-side machinery
(DEM, Atlas 14, streams, cartography) comes from stormscape.
"""

__version__ = "0.1.0"

from firescape import paths  # noqa: F401  (light, no heavy deps)
from firescape.config import (  # noqa: F401
    Calibration,
    RegionCalibration,
    FilterDefaults,
    I15_REFERENCE_MMH,
    load_calibration,
    packaged_calibration,
)
from firescape.severity import (  # noqa: F401
    load_cdf_table,
    simulate_dnbr,
    classify_barc4,
    coverage_report,
)

# Heavy/geospatial modules (fires, mtbs, delineate, hazard, ...) are imported
# explicitly by callers so that `import firescape` stays cheap and works in
# environments without the full geo stack.
