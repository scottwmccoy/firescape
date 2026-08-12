"""Stream-segment network delineation via pfdf (milestone M1).

Owns the pfdf watershed chain and Segments construction, plus the raster
grid-matching helper. All rasters entering a Segments computation must share
the DEM grid; ``match_grid`` is the single place that enforces it.

pysheds trap: rasters without explicit nodata get NoData=0 — every Raster we
build here sets nodata explicitly.
"""

from __future__ import annotations

import math
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
    if not mask_arr.any():
        # Fully masked unit (playa, lake, valley floor): pfdf raises on an
        # empty mask, so report it as "no network" instead.
        return None, terr
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


#: LANDFIRE EVT class-name fragments treated as non-burnable / non-source
#: (water bodies, permanent snow and ice, quarries/mines). Matched
#: case-insensitively against the EVT legend, so this survives code changes.
WATER_NAME_PATTERNS = ("open water", "snow", "ice", "quarr", "aquaculture")


def _focal_std(values: np.ndarray, radius_px: int) -> np.ndarray:
    """Standard deviation of a raster in a square window (Rossi's focal std).

    Uses E[x^2] - E[x]^2 via uniform filters — O(n) regardless of radius.
    """
    from scipy.ndimage import uniform_filter

    a = np.asarray(values, dtype="float64")
    size = 2 * radius_px + 1
    mean = uniform_filter(a, size=size, mode="nearest")
    mean_sq = uniform_filter(a * a, size=size, mode="nearest")
    return np.sqrt(np.clip(mean_sq - mean * mean, 0, None))


def _drop_small(mask: np.ndarray, min_km2: float, pixel_area_km2: float) -> np.ndarray:
    """Remove connected clusters smaller than ``min_km2`` (Rossi's polygon-area
    filter, done in raster space)."""
    from scipy.ndimage import label

    lab, n = label(mask)
    if n == 0:
        return mask
    counts = np.bincount(lab.ravel())
    min_px = max(1, int(round(min_km2 / pixel_area_km2)))
    keep = np.zeros(counts.size, dtype=bool)
    keep[1:] = counts[1:] >= min_px
    return keep[lab]


def rossi_masks(dem, flow=None, evt=None, evt_water_codes=None, *,
                valley_radius_m: float = 200.0, valley_std_m: float = 5.0,
                min_cluster_km2: float = 1.0) -> dict:
    """Exclusion masks for pre-fire delineation, after Rossi et al. (2025).

    - **valley**: focal std of elevation <= 5 m within a 200 m radius,
      clusters >= 1 km^2 (flat valley floors where basins are artifacts).
    - **sink**: cells with no flow direction (pfdf/pysheds nulls after
      conditioning), clusters >= 1 km^2, restricted to cells that also fall in
      the valley mask — exactly Rossi's "must intersect the valley mask" rule.
    - **water**: EVT open-water / snow-ice / quarry classes.

    Returns a dict of boolean arrays plus their union under ``"exclude"``.
    """
    px_km2 = pixel_km2(dem)
    px_m = math.sqrt(px_km2 * 1e6)
    radius_px = max(1, int(round(valley_radius_m / px_m)))

    valley = _focal_std(dem.values, radius_px) <= valley_std_m
    valley = _drop_small(valley, min_cluster_km2, px_km2)

    if flow is not None:
        nulls = ~np.isfinite(flow.values.astype("float64")) if flow.nodata is None else (
            flow.values == flow.nodata)
        sink = _drop_small(np.asarray(nulls, dtype=bool), min_cluster_km2, px_km2) & valley
    else:
        sink = np.zeros(dem.shape, dtype=bool)

    if evt is not None and evt_water_codes:
        water = np.isin(evt.values, list(evt_water_codes))
    else:
        water = np.zeros(dem.shape, dtype=bool)

    return {"valley": valley, "sink": sink, "water": water,
            "exclude": valley | sink | water}


def water_codes_from_legend(legend, *, value_col: str = "Value",
                            name_col: str = "EVT_NAME") -> list[int]:
    """EVT codes whose class names indicate water / snow-ice / quarries."""
    names = legend[name_col].astype(str).str.lower()
    hit = np.zeros(len(legend), dtype=bool)
    for pat in WATER_NAME_PATTERNS:
        hit |= names.str.contains(pat, regex=False).to_numpy()
    return [int(v) for v in legend.loc[hit, value_col]]
