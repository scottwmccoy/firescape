"""Stream-segment network delineation via pfdf (milestone M1).

Will own: the xarray <-> pfdf.Raster bridge (with EXPLICIT nodata — pysheds
treats missing nodata as 0), watershed.condition/flow/slopes/relief/
accumulation, Segments construction (delineation mask = accumulation >=
min_area AND in-perimeter/AOI), USGS-default filtering with continuous(),
locate_basins(), and the Rossi valley/sink/water masks (M4).
"""

from __future__ import annotations


def network(dem, *, perimeter=None, filters=None):
    """Build and filter the segment network for a DEM (+ optional perimeter)."""
    raise NotImplementedError("delineate.network lands in milestone M1")


def rossi_masks(dem, evt=None):
    """Valley (focal std <=5 m @200 m, >=1 km2), sink, and water masks (M4)."""
    raise NotImplementedError("delineate.rossi_masks lands in milestone M4")
