"""Corridor conditioning: evaluate change evidence along the stream network.

The step that turns z maps into a per-segment response inventory (the
"segment" of stack--z--segment). A 1--3-pixel-wide track is marginal per
pixel; integrated over the few hundred corridor pixels of one stream
segment it is a robust along-reach statistic -- matched filtering along
the network. Decision units are network segments (pfdf's, or any
LineString layer such as the Dolan inventory's), which is also the schema
of the USGS reach inventories (Cavagnaro et al. 2025) these products are
scored against.

Class codes follow the Dolan inventory so confusion matrices read
directly: 0 = no erosion, 1 = fluvial erosion, 3 = debris flow (2, the
Dolan "remotely mapped DF" class, is never produced by :func:`classify`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Dolan-inventory class semantics (``Confidence`` column).
CLASSES = {0: "no erosion", 1: "fluvial erosion",
           2: "debris flow (remote)", 3: "debris flow"}


def corridor_width_m(area_km2, *, w_min: float = 9.0, w_max: float = 60.0,
                     k: float = 12.0):
    """Corridor full width from contributing area: w_min + k*sqrt(A), capped.

    3 px minimum keeps a 1-px track inside its corridor despite ~sub-px
    registration wobble; the cap stops trunk rivers from swallowing their
    floodplains.
    """
    a = np.clip(np.nan_to_num(np.asarray(area_km2, dtype=float)), 0.0, None)
    return np.clip(w_min + k * np.sqrt(a), w_min, w_max)


def corridor_raster(segments, ref, *, area_col: str = "Area_km2",
                    w_min: float = 9.0, w_max: float = 60.0, k: float = 12.0):
    """Rasterize buffered segments onto the epoch grid as a label image.

    ``segments`` must already be in ``ref['crs']`` (a metric CRS -- buffers
    are in meters). Returns int32 labels: 0 background, row position + 1
    otherwise. Later segments win ties, which is acceptable at 3 m: overlap
    happens only at confluences.
    """
    from rasterio import features

    widths = corridor_width_m(segments.get(area_col, 0.0),
                              w_min=w_min, w_max=w_max, k=k)
    shapes = [(geom.buffer(w / 2.0), i + 1)
              for i, (geom, w) in enumerate(zip(segments.geometry, widths))
              if geom is not None and not geom.is_empty]
    labels = features.rasterize(
        shapes, out_shape=(ref["height"], ref["width"]),
        transform=ref["transform"], fill=0, all_touched=True, dtype="int32")
    return labels


def segment_stats(labels, fields: dict, *, frac_threshold: float = 2.0):
    """Per-segment corridor statistics of each field.

    ``fields`` maps name -> 2-D (masked) array on the label grid. Returns a
    DataFrame indexed by segment row position (label - 1) with, per field,
    the corridor mean, p90 and the fraction of pixels above
    ``frac_threshold``, plus ``n_pixels`` actually sampled.
    """
    lab = np.asarray(labels).ravel()
    out = None
    for name, arr in fields.items():
        m = np.ma.masked_invalid(np.ma.asarray(arr)).ravel()
        ok = (lab > 0) & ~np.ma.getmaskarray(m)
        df = pd.DataFrame({"lab": lab[ok], "v": np.asarray(m[ok])})
        g = df.groupby("lab")["v"]
        stat = pd.DataFrame({
            f"{name}_mean": g.mean(),
            f"{name}_p90": g.quantile(0.9),
            f"{name}_frac": g.apply(lambda s: float((s > frac_threshold).mean())),
        })
        out = stat if out is None else out.join(stat, how="outer")
        out[f"n_pixels"] = df.groupby("lab")["v"].size()
    out.index = out.index - 1                      # back to segment row order
    out.index.name = "segment"
    return out


def classify(stats, *, mean_col: str = "z_brightness_mean",
             df_t: float = 2.5, fluvial_t: float = 1.0,
             min_pixels: int = 8) -> pd.Series:
    """Three-way response class from corridor-integrated z.

    Monotone thresholds on the corridor-mean z: >= ``df_t`` -> debris flow
    (3), >= ``fluvial_t`` -> fluvial erosion (1), else no erosion (0).
    Segments with fewer than ``min_pixels`` usable pixels stay 0 -- absence
    of evidence, flagged by the pixel count, not evidence of absence.
    Thresholds are provisional until calibrated against the Dolan
    inventory; pass what the calibration finds.
    """
    cls = pd.Series(0, index=stats.index, dtype=int, name="response")
    ok = stats["n_pixels"].fillna(0) >= min_pixels
    v = stats[mean_col]
    cls[ok & (v >= fluvial_t)] = 1
    cls[ok & (v >= df_t)] = 3
    return cls
