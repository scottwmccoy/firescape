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


def neighbors_by_node(segments, *, tol: float = 3.0) -> dict[int, set]:
    """Adjacency from shared endpoints (no to/from ids in the GPKGs).

    Endpoint coordinates are snapped to ``tol`` metres; segments sharing a
    snapped node are neighbors. MultiLineStrings are line-merged first.
    """
    from collections import defaultdict

    from shapely import ops

    nodes = defaultdict(set)
    for i, geom in enumerate(segments.geometry):
        if geom is None or geom.is_empty:
            continue
        g = ops.linemerge(geom) if geom.geom_type == "MultiLineString" else geom
        parts = list(g.geoms) if g.geom_type == "MultiLineString" else [g]
        ends = [parts[0].coords[0], parts[-1].coords[-1]]
        for x, y in ends:
            nodes[(round(x / tol), round(y / tol))].add(i)
    adj: dict[int, set] = defaultdict(set)
    for members in nodes.values():
        for i in members:
            adj[i] |= members - {i}
    return dict(adj)


def continuity_demote(segments, cls: pd.Series, *, tol: float = 3.0,
                      adj: dict | None = None) -> pd.Series:
    """Demote isolated hot segments by one class.

    A real flow occupies a connected chain of reaches; grading, road dust
    and field edges produce isolated hot segments. Any segment classified
    >= fluvial with NO network neighbor >= fluvial drops one level
    (3 -> 1, 1 -> 0). Chains are untouched -- every member has a hot
    neighbor. One pass, simultaneous (decisions use the input classes).
    """
    if adj is None:
        adj = neighbors_by_node(segments, tol=tol)
    out = cls.copy()
    hot = set(cls.index[cls >= 1])
    for i in cls.index[cls >= 1]:
        if not (set(adj.get(i, ())) & hot):
            out.loc[i] = 1 if cls.loc[i] == 3 else 0
    return out


def link_fans(fan_gdf, segments, cls: pd.Series, *, max_dist: float = 150.0):
    """Attach feeder evidence to fan objects.

    Adds ``feeder_class`` (the strongest response class among segments
    within ``max_dist`` of the polygon) and ``fed`` (feeder_class >= 1).
    Deposits with no responding feeder are the road/irrigation imposters.
    """
    if not len(fan_gdf):
        fan_gdf = fan_gdf.copy()
        fan_gdf["feeder_class"] = []
        fan_gdf["fed"] = []
        return fan_gdf
    out = fan_gdf.copy()
    hot = segments[cls.reindex(segments.index).fillna(0) >= 1]
    out["feeder_class"] = 0
    if len(hot):
        import geopandas as gpd

        joined = gpd.sjoin_nearest(out[["geometry"]], hot[["geometry"]],
                                   max_distance=max_dist, how="left",
                                   distance_col="_d")
        near = joined.dropna(subset=["index_right"])
        best = near.groupby(level=0)["index_right"].apply(
            lambda s: int(cls.loc[s.astype(int)].max()))
        out.loc[best.index, "feeder_class"] = best
    out["fed"] = out["feeder_class"] >= 1
    return out


def local_offsets(labels, z, *, block: int = 750, max_off: int = 8,
                  min_corridor_px: int = 2000, min_snr: float = 8.0,
                  smooth: bool = True, with_info: bool = False):
    """Per-block network-to-imagery offset, estimated WITHOUT truth labels.

    DEM-delineated flowlines ride a different georeference than the
    orthoimagery, by a spatially varying ~5-20 m (measured on Dolan, p19).
    Per block, the corridor mask is cross-correlated (FFT) against |z| --
    the offset that concentrates absolute change energy under the corridors
    is taken as the local misalignment. Every segment in a block gets the
    same shift, and class labels are never consulted, so the alignment
    cannot cherry-pick per-segment evidence. Blocks with thin corridor
    coverage inherit the median of estimated neighbours.

    Returns ``(dy, dx)`` int arrays of shape (n_blocky, n_blockx): the roll
    to apply to the LABEL raster to land corridors on the imagery.
    """
    from scipy.signal import fftconvolve

    zabs = np.abs(np.ma.filled(np.ma.masked_invalid(np.ma.asarray(z)), 0.0))
    mask = (np.asarray(labels) > 0).astype(np.float32)
    H, W = mask.shape
    nby, nbx = int(np.ceil(H / block)), int(np.ceil(W / block))
    dy = np.full((nby, nbx), np.iinfo(np.int32).min, np.int32)
    dx = np.zeros((nby, nbx), np.int32)
    snr_grid = np.full((nby, nbx), np.nan, np.float32)
    raw_dy = np.zeros((nby, nbx), np.int32)
    raw_dx = np.zeros((nby, nbx), np.int32)
    for by in range(nby):
        for bx in range(nbx):
            sl = (slice(by * block, min((by + 1) * block, H)),
                  slice(bx * block, min((bx + 1) * block, W)))
            m = mask[sl]
            if m.sum() < min_corridor_px:
                continue
            za = zabs[sl]
            za = np.where(za > 0, za - za[za > 0].mean(), 0.0)
            c = fftconvolve(za, m[::-1, ::-1], mode="same")
            cy, cx = np.array(c.shape) // 2
            w = c[cy - max_off:cy + max_off + 1, cx - max_off:cx + max_off + 1]
            p = np.unravel_index(np.argmax(w), w.shape)
            # prominence gate: mean of the 3x3 peak core vs the window
            # median, in MAD units. A real alignment peak (even the ridge a
            # linear channel produces) scores >>10; a noise argmax is a
            # single-cell spike whose core mean collapses (~3). Boundary
            # peaks are rejected outright -- the measured Dolan pathology
            # was noise blocks pinning at the search bounds.
            core = w[max(p[0] - 1, 0):p[0] + 2,
                     max(p[1] - 1, 0):p[1] + 2].mean()
            mad = np.median(np.abs(w - np.median(w))) + 1e-12
            snr = (core - np.median(w)) / (1.4826 * mad)
            snr_grid[by, bx] = snr
            raw_dy[by, bx] = p[0] - max_off
            raw_dx[by, bx] = p[1] - max_off
            on_edge = (p[0] in (0, w.shape[0] - 1)
                       or p[1] in (0, w.shape[1] - 1))
            if on_edge or snr < min_snr:
                continue
            dy[by, bx] = p[0] - max_off
            dx[by, bx] = p[1] - max_off
    have = dy != np.iinfo(np.int32).min
    info = {"snr": snr_grid, "accepted": have.copy(),
            "raw_dy": raw_dy, "raw_dx": raw_dx}
    if not have.any():
        out = (np.zeros_like(dy), np.zeros_like(dx))
        return (*out, info) if with_info else out
    med = (int(np.median(dy[have])), int(np.median(dx[have])))
    dy[~have], dx[~have] = med
    if smooth and min(dy.shape) >= 2:
        from scipy.ndimage import median_filter
        dy = median_filter(dy, size=3, mode="nearest")
        dx = median_filter(dx, size=3, mode="nearest")
    return (dy, dx, info) if with_info else (dy, dx)


def apply_offsets(labels, dy, dx, *, block: int = 750):
    """Shift the label raster block-wise by the ``local_offsets`` field."""
    from scipy.ndimage import shift as ndshift

    labels = np.asarray(labels)
    out = np.zeros_like(labels)
    H, W = labels.shape
    for by in range(dy.shape[0]):
        for bx in range(dy.shape[1]):
            sl = (slice(by * block, min((by + 1) * block, H)),
                  slice(bx * block, min((bx + 1) * block, W)))
            out[sl] = ndshift(labels[sl], (dy[by, bx], dx[by, bx]),
                              order=0, mode="constant", cval=0)
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
