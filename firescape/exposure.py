"""Hazard exposure for assets — what would a debris flow hit?

The verification side of firescape asks "did a flow happen?"; this module asks
the forward question: given the pre-fire hazard surface, which mapped assets
could a post-fire debris flow reach? Assets are any GeoDataFrame of points or
polygons. The motivating case is USMIN mine-waste footprints (dumps,
tailings) — loose, sometimes contaminated material sitting in Great Basin
canyons, whose remobilisation into a drainage is a water-quality event as much
as a geomorphic one.

Exposure model (deliberately transparent, v1):

- A segment can *deliver* to an asset when the asset touches the segment's
  corridor, using the same contributing-area width law the verification
  corridors use (:func:`tracescape.corridor.corridor_width_m`) plus a
  registration pad. This is a proximity proxy, not a runout model: it will
  call exposed an asset that stands beside a channel on a 3 m terrace, and it
  will miss an asset on distal fan surfaces beyond the corridor. Both errors
  are visible in ``dist_m``, so the ranking degrades gracefully rather than
  silently.
- Assets that touch no corridor get the nearest segment within ``near_max_m``
  and its distance, so "almost exposed" stays distinguishable from "clear".
- Hazard columns come from the delivering segments (the max-likelihood
  deliverer wins); site annual rates push that segment's triggering threshold
  through the containing basin's rain climatology
  (:func:`firescape.annualprob.p_exceed_threshold`) times the basin's P(F).

Every step returns plain columns (which segment, how far, which basin), so a
site's score can always be audited back to the model products.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from firescape import annualprob
# The contributing-area width law is tracescape's: it was derived for the
# verification corridors that read imagery along the network, and exposure
# deliberately reuses it so "delivers to this asset" and "we looked for change
# here" mean the same strip of ground. One shared definition, not two that
# drift.
from tracescape.corridor import corridor_width_m

#: hazard columns carried from the delivering segment, statewide-product names
SEGMENT_COLS = ("P_24mmh", "V_24mmh", "H_24mmh", "I15_50")


def _check_metric(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame) -> None:
    if a.crs is None or b.crs is None or a.crs != b.crs:
        raise ValueError(f"CRS mismatch: {a.crs} vs {b.crs}")
    if a.crs.is_geographic:
        raise ValueError("exposure joins need a projected (metre) CRS")


def corridor_buffers(segments: gpd.GeoDataFrame, *, area_col: str = "Area_km2",
                     pad_m: float = 0.0, w_min: float = 9.0,
                     w_max: float = 60.0, k: float = 12.0) -> gpd.GeoSeries:
    """Segment corridors as polygons: half the width law each side, plus pad."""
    w = corridor_width_m(segments[area_col].to_numpy(), w_min=w_min,
                         w_max=w_max, k=k)
    return segments.geometry.buffer(w / 2.0 + pad_m)


def hazard_at_assets(assets: gpd.GeoDataFrame, segments: gpd.GeoDataFrame, *,
                     cols=SEGMENT_COLS, rank_col: str = "P_24mmh",
                     area_col: str = "Area_km2", pad_m: float = 25.0,
                     near_max_m: float = 1000.0) -> pd.DataFrame:
    """Per-asset delivery: corridor hits, else nearest segment within reach.

    Returns a DataFrame indexed like ``assets`` with ``exposed`` (touches a
    corridor), ``dist_m`` (0 when exposed, corridor-edge distance otherwise,
    NaN beyond ``near_max_m``), ``n_deliver`` (corridor hits), ``seg_id`` (the
    max-``rank_col`` deliverer, or the nearest segment), and ``cols`` from
    that segment.
    """
    _check_metric(assets, segments)
    cols = [c for c in cols if c != rank_col]
    cols = [rank_col] + cols
    out = pd.DataFrame(index=assets.index)
    out["exposed"] = False
    out["dist_m"] = np.nan
    out["n_deliver"] = 0
    out["seg_id"] = pd.Series(pd.NA, index=assets.index, dtype="object")
    for c in cols:
        out[c] = np.nan
    if not len(segments) or not len(assets):
        return out

    bare = assets[["geometry"]]
    corr = gpd.GeoDataFrame(segments[cols].reset_index(drop=True),
                            geometry=corridor_buffers(
                                segments, area_col=area_col, pad_m=pad_m
                            ).reset_index(drop=True), crs=segments.crs)
    corr["seg_id"] = segments.index.to_numpy()

    hit = gpd.sjoin(bare, corr, predicate="intersects", how="inner")
    if len(hit):
        # whole-row winner: groupby().first() blends rows when columns hold
        # NaN, taking each column's first non-null independently
        srt = hit.sort_values(rank_col, ascending=False)
        best = srt[~srt.index.duplicated(keep="first")]
        n = hit.groupby(level=0).size()
        out.loc[best.index, "exposed"] = True
        out.loc[best.index, "dist_m"] = 0.0
        out.loc[n.index, "n_deliver"] = n
        out.loc[best.index, "seg_id"] = best["seg_id"]
        for c in cols:
            out.loc[best.index, c] = best[c]

    miss = bare.loc[~out["exposed"]]
    if len(miss):
        seg = gpd.GeoDataFrame(segments[cols], geometry=segments.geometry,
                               crs=segments.crs)
        seg["seg_id"] = segments.index.to_numpy()
        near = gpd.sjoin_nearest(miss, seg, how="inner",
                                 max_distance=near_max_m,
                                 distance_col="dist_m")
        if len(near):
            near = near.sort_values("dist_m")
            near = near[~near.index.duplicated(keep="first")]
            out.loc[near.index, "dist_m"] = near["dist_m"]
            out.loc[near.index, "seg_id"] = near["seg_id"]
            for c in cols:
                out.loc[near.index, c] = near[c]
    return out


def combine_best(frames: list[pd.DataFrame], *,
                 rank_col: str = "P_24mmh") -> pd.DataFrame:
    """Merge per-tile :func:`hazard_at_assets` results, keeping each asset's
    best row: exposed beats near, higher ``rank_col`` beats lower, then
    nearer beats farther. ``n_deliver`` sums across tiles (a corridor can
    cross a tile seam)."""
    allrows = pd.concat([f for f in frames if len(f)])
    srt = allrows.sort_values(["exposed", rank_col, "dist_m"],
                              ascending=[False, False, True],
                              na_position="last")
    best = srt[~srt.index.duplicated(keep="first")].copy()
    best["n_deliver"] = allrows.groupby(level=0)["n_deliver"].sum()
    return best


def site_annual(hz: pd.DataFrame, assets: gpd.GeoDataFrame,
                basins: gpd.GeoDataFrame, *, thresh_col: str = "I15_50",
                i1_col: str = "I15_1yr", i50_col: str = "I15_50yr",
                pf_col: str = "P_F",
                max_basin_m: float = 5000.0) -> pd.DataFrame:
    """Annual rates at each site: the delivering segment's threshold pushed
    through the containing (else nearest, within ``max_basin_m``) basin's rain
    climatology, times that basin's fire probability.

    Adds ``I15_1yr``, ``I15_50yr``, ``P_F``, ``basin_dist_m``,
    ``P_RgtT_site`` and ``P_annual_site`` to a copy of ``hz``.
    """
    _check_metric(assets, basins)
    pts = gpd.GeoDataFrame(
        geometry=assets.geometry.representative_point(), crs=assets.crs)
    bcols = [i1_col, i50_col, pf_col]
    b = basins[bcols + ["geometry"]]
    within = gpd.sjoin(pts, b, predicate="within", how="inner")
    within = within[~within.index.duplicated(keep="first")]
    out = hz.copy()
    for c in bcols:
        out[c] = np.nan
    out["basin_dist_m"] = np.nan
    out.loc[within.index, bcols] = within[bcols].to_numpy()
    out.loc[within.index, "basin_dist_m"] = 0.0

    rest = pts.loc[out[i1_col].isna()]
    if len(rest):
        near = gpd.sjoin_nearest(rest, b, how="inner",
                                 max_distance=max_basin_m,
                                 distance_col="basin_dist_m")
        near = near.sort_values("basin_dist_m")
        near = near[~near.index.duplicated(keep="first")]
        if len(near):
            out.loc[near.index, bcols] = near[bcols].to_numpy()
            out.loc[near.index, "basin_dist_m"] = near["basin_dist_m"]

    p_rt, _, _ = annualprob.p_exceed_threshold(
        out[thresh_col].to_numpy(), out[i1_col].to_numpy(),
        out[i50_col].to_numpy())
    out["P_RgtT_site"] = p_rt
    out["P_annual_site"] = annualprob.combined(p_rt, out[pf_col].to_numpy())
    return out


def nearest_receptor(assets: gpd.GeoDataFrame, waters: gpd.GeoDataFrame, *,
                     cols=(), max_m: float | None = None) -> pd.DataFrame:
    """Nearest water feature per asset: ``water_dist_m`` plus carried ``cols``.

    Receptor context for the pollution pathway at whatever fidelity
    ``waters`` carries — straight-line proximity, not routed connectivity.
    With ``cols=("name", "kind")`` on an NHD receptor layer each site learns
    *which* water it threatens; beyond ``max_m`` everything is NaN, which
    reads as "no mapped receptor within the question radius".
    """
    _check_metric(assets, waters)
    keep = [c for c in cols if c in waters.columns]
    near = gpd.sjoin_nearest(assets[["geometry"]],
                             waters[keep + ["geometry"]], how="left",
                             max_distance=max_m, distance_col="water_dist_m")
    near = near.sort_values("water_dist_m")
    near = near[~near.index.duplicated(keep="first")]
    return near[["water_dist_m"] + keep].reindex(assets.index)


def water_distance(assets: gpd.GeoDataFrame,
                   waters: gpd.GeoDataFrame) -> pd.Series:
    """Distance (m) from each asset to the nearest mapped water feature."""
    return nearest_receptor(assets, waters)["water_dist_m"]


def within_any(assets: gpd.GeoDataFrame,
               polygons: gpd.GeoDataFrame) -> pd.Series:
    """Boolean per asset: representative point inside any of ``polygons``.

    The land-manager flag (e.g. staged BLM SMA holdings). Point-in-polygon on
    the representative point, so a footprint straddling a parcel line is
    attributed to the parcel holding its interior point.
    """
    _check_metric(assets, polygons)
    pts = gpd.GeoDataFrame(geometry=assets.geometry.representative_point(),
                           crs=assets.crs)
    hit = gpd.sjoin(pts, polygons[["geometry"]], predicate="within",
                    how="inner")
    out = pd.Series(False, index=assets.index)
    out.loc[hit.index.unique()] = True
    return out


def rank(df: pd.DataFrame) -> pd.DataFrame:
    """Sort sites most-exposed-first and add a 1-based ``rank`` column.

    Order: exposed before near-miss, then annual site probability, then
    conditional likelihood, then distance. NaNs sink.
    """
    out = df.sort_values(
        ["exposed", "P_annual_site", "P_24mmh", "dist_m"],
        ascending=[False, False, False, True], na_position="last").copy()
    out["rank"] = np.arange(1, len(out) + 1)
    return out
