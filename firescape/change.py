"""Epoch compositing and change statistics for the Planet verification module.

The v0 slice of the "stack--z--segment" design
(``docs/planet_verification_design.md``): per-scene indices -> epoch median +
MAD -> robust z. Pure array functions; file I/O, gridding and corridor
conditioning live in the drivers (and later modules). Everything here works
on masked arrays and returns masked arrays.

Why per-scene indices, then the median of the *index* stack (not indices of
band medians): ratio indices cancel per-scene gain, so residual cross-Dove
calibration scatter mostly divides out before the median ever sees it --
brightness is the exception (not a ratio), which is one reason it needs the
z-normalization more than the greenness channels do.

Band conventions (PSB.SD 8-band SR, 1-indexed as delivered):
1 coastal blue, 2 blue, 3 green-I, 4 green, 5 yellow, 6 red, 7 red-edge,
8 NIR. The 4-band-equivalent set used here is 2/4/6/8 (blue/green/red/NIR).
Reflectance scaling: SR rasters are uint16 x10000; callers pass reflectance
(already /1e4) -- these functions do not rescale.
"""

from __future__ import annotations

import numpy as np

#: Consistency constant: MAD * 1.4826 estimates sigma for a normal population.
MAD_TO_SIGMA = 1.4826

#: Denominator floor for rdNDVI (HazMapper Eq. 2 has none and goes unstable
#: as pre+post -> 0 -- playa, rock, salt crust; RdNBR-lineage fix).
RDNDVI_FLOOR = 0.05


# --------------------------------------------------------------------------- #
# per-scene indices
# --------------------------------------------------------------------------- #
def ndvi(nir, red):
    return (nir - red) / np.ma.masked_values(nir + red, 0.0)


def msavi2(nir, red):
    """Modified SAVI (Qi et al. 1994) -- the greenness channel for sparse
    rangeland, where plain NDVI is soil-dominated at 0.1-0.25 baselines."""
    s = 2.0 * nir + 1.0
    disc = np.ma.clip(s * s - 8.0 * (nir - red), 0.0, None)
    return (s - np.ma.sqrt(disc)) / 2.0


def brightness(blue, green, red, nir):
    """Flat VIS-NIR mean -- the lead signal for fresh mineral sediment
    (deposits brighten; SLIP/Tasseled-Cap-Brightness lineage)."""
    return (blue + green + red + nir) / 4.0


def redness(red, green):
    """Red/green ratio -- oxidized-sediment / ash-to-mineral discriminator.
    High commission on its own; use only under corridor conditioning."""
    return red / np.ma.masked_values(green, 0.0)


def rdndvi(ndvi_pre, ndvi_post, *, floor: float = RDNDVI_FLOOR):
    """Relative dNDVI (Scheip & Wegmann 2021 Eq. 2), denominator floored.

    x100, so values are percent-like. Computed on epoch MEDIANS, never on
    single scenes -- the normalization amplifies noise ~1.3-2.2x at Nevada
    baselines and only earns its keep after compositing.
    """
    denom = np.ma.sqrt(np.ma.clip(ndvi_pre + ndvi_post, floor, None))
    return 100.0 * (ndvi_post - ndvi_pre) / denom


# --------------------------------------------------------------------------- #
# epoch statistics
# --------------------------------------------------------------------------- #
def epoch_stats(stack):
    """Median, MAD and per-pixel usable count of a (scene, y, x) masked stack.

    The median is the epoch composite (never greenest-pixel: a per-pixel max
    selects positive noise excursions and the hottest-calibrated Doves). The
    MAD is the per-pixel temporal noise scale that the z-score divides by --
    the statistic that formalizes "average multiple scenes per epoch".
    """
    stack = np.ma.asarray(stack)
    if stack.ndim != 3:
        raise ValueError("epoch_stats wants a (scene, y, x) stack")
    med = np.ma.median(stack, axis=0)
    mad = np.ma.median(np.ma.abs(stack - med), axis=0)
    count = (~np.ma.getmaskarray(stack)).sum(axis=0)
    return med, mad, count


def robust_z(pre_med, post_med, pre_mad, *, floor_q: float = 0.6,
             min_floor: float = 1e-4):
    """(post - pre) in units of pre-epoch temporal sigma.

    ``pre_mad`` is scaled by 1.4826 to sigma; the divisor is floored at the
    ``floor_q`` quantile of the valid MADs (plus ``min_floor``) so pixels
    that happened to be dead-stable in the pre stack cannot manufacture
    infinite z. A stratum-aware floor (slope/aspect/severity) replaces the
    global quantile in the full engine.
    """
    sigma = MAD_TO_SIGMA * np.ma.asarray(pre_mad)
    valid = sigma.compressed()
    if valid.size == 0:
        raise ValueError("robust_z: pre-epoch MAD is fully masked")
    floor = max(float(np.quantile(valid, floor_q)), min_floor)
    return (post_med - pre_med) / np.ma.clip(sigma, floor, None)


def scene_offset(ref, img, *, upsample: int = 10):
    """Sub-pixel (dy, dx) displacement of ``img`` content relative to ``ref``
    by phase correlation -- the cheap registration *check* (AROSICS is the
    fix). Shifting ``img`` by the NEGATED return registers it onto ``ref``.

    Both inputs 2-D masked arrays on the same grid; masked/NaN pixels are
    filled with the mean before the FFT. Returns pixels.
    """
    from skimage.registration import phase_cross_correlation

    def _fill(a):
        a = np.ma.masked_invalid(np.ma.asarray(a))
        m = a.mean()
        fill = float(m) if np.isfinite(m) else 0.0
        return np.nan_to_num(a.filled(fill), nan=fill)

    # skimage returns the correction to apply to img; displacement is its
    # negation, which is what a mis-registration report should quote.
    shift, _, _ = phase_cross_correlation(_fill(ref), _fill(img),
                                          upsample_factor=upsample,
                                          normalization="phase")
    return float(-shift[0]), float(-shift[1])
