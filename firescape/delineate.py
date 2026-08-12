"""Stream-segment network delineation via pfdf (milestone M1).

Owns the pfdf watershed chain and Segments construction, plus the raster
grid-matching helper. All rasters entering a Segments computation must share
the DEM grid; ``match_grid`` is the single place that enforces it.

pysheds trap: rasters without explicit nodata get NoData=0 — every Raster we
build here sets nodata explicitly.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from firescape.config import FilterDefaults


def match_grid(raster, template, *, resampling: str = "nearest"):
    """Return ``raster`` on ``template``'s exact grid (CRS, transform, shape).

    pfdf's ``reproject(template=...)`` matches CRS/resolution/registration but
    keeps the source footprint, so a clip to the template bounds is required
    for shapes to align.
    """
    same = (
        raster.crs == template.crs
        and raster.shape == template.shape
        and raster.transform == template.transform
    )
    if same:
        return raster
    out = raster.copy()
    r = out.reproject(template=template, resampling=resampling)
    out = r if r is not None else out
    r = out.clip(bounds=template)
    out = r if r is not None else out
    if out.shape != template.shape:
        raise ValueError(
            f"grid match failed: {out.shape} != template {template.shape}"
        )
    return out


def pixel_km2(raster) -> float:
    """Pixel area of a projected raster in km^2."""
    try:
        return float(raster.pixel_area(units="kilometers"))
    except (AttributeError, TypeError):
        aff = raster.transform.affine
        return abs(aff.a * aff.e) / 1e6


def terrain(dem) -> SimpleNamespace:
    """DEM conditioning + D8 flow, slopes (gradients), vertical relief."""
    from pfdf import watershed

    conditioned = watershed.condition(dem)
    flow = watershed.flow(conditioned)
    slopes = watershed.slopes(conditioned, flow)
    relief = watershed.relief(conditioned, flow)
    return SimpleNamespace(conditioned=conditioned, flow=flow, slopes=slopes, relief=relief)


def network(dem, domain_mask, *, min_area_km2: float | None = None,
            max_length_m: float | None = None):
    """Build the segment network: mask = (accum area >= min) & domain.

    ``domain_mask``: boolean Raster (any grid) limiting delineation — for a
    fire assessment this is the (buffered) burn perimeter. Returns
    (segments, terrain_namespace).
    """
    from pfdf.raster import Raster
    from pfdf.segments import Segments
    from pfdf import watershed

    f = FilterDefaults()
    min_area = f.min_area_km2 if min_area_km2 is None else min_area_km2
    max_len = f.max_length_m if max_length_m is None else max_length_m

    terr = terrain(dem)
    domain = match_grid(domain_mask, dem)
    npix = watershed.accumulation(terr.flow)
    area_km2 = npix.values.astype("float64") * pixel_km2(dem)
    mask_arr = (area_km2 >= min_area) & (domain.values.astype(bool))
    mask = Raster.from_array(mask_arr, spatial=dem, isbool=True)
    segments = Segments(terr.flow, mask, max_length=max_len, units="meters")
    return segments, terr


def usgs_filter(segments, *, burned, perimeter, slopes, dem_conditioned,
                filters: FilterDefaults | None = None,
                neighborhood: int = 4) -> np.ndarray:
    """Apply the standard USGS/wildcat physical filters, preserving flow
    continuity (``continuous``). Returns the boolean keep-vector applied.

    burned: boolean Raster (BARC 2-4); perimeter: boolean Raster (unbuffered);
    slopes: flow-slope raster (gradients); dem_conditioned for confinement.
    Deviation from wildcat: no developed-area filter yet (needs a development
    raster; the pilot area is largely undeveloped) — noted in run metadata.
    """
    f = filters or FilterDefaults()
    area = segments.area(units="kilometers")
    burn_ratio = segments.burn_ratio(burned)
    slope = segments.slope(slopes)
    confinement = segments.confinement(dem_conditioned, neighborhood)
    in_perim = segments.in_perimeter(perimeter)
    keep = (
        (area <= f.max_area_km2)
        & (burn_ratio >= f.min_burn_ratio)
        & (slope >= f.min_slope)
        & (confinement <= f.max_confinement_deg)
        & np.asarray(in_perim, dtype=bool)
    )
    adjusted = segments.continuous(keep)
    segments.keep(adjusted)
    return np.asarray(adjusted, dtype=bool)


def rossi_masks(dem, evt=None):
    """Valley (focal std <=5 m @200 m, >=1 km2), sink, and water masks (M4)."""
    raise NotImplementedError("delineate.rossi_masks lands in milestone M4")
