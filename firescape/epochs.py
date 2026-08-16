"""Epoch stacks: staged Planet scenes -> common grid -> index composites.

The I/O half of the stack engine (``docs/planet_verification_design.md``);
the pure statistics live in :mod:`firescape.change`. An "epoch" is a
directory of clipped SR + udm2 rasters staged by
:func:`firescape.planetscope.stage_order`; :func:`build` turns one into
median/MAD/count composites of the change indices, plus a median RGB for
display, all on a single reference grid.

Scenes are warped onto the grid of the FIRST scene of the first epoch built
(bilinear for reflectance, nearest for the mask) -- pass the returned ``ref``
to every later :func:`build` so both epochs share one grid exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from firescape import change as ch

#: PSB.SD 8-band SR band indices (1-based, as delivered).
BANDS_8B = {"blue": 2, "green": 4, "red": 6, "nir": 8}


def grid(bounds4326, crs, res: float = 3.0) -> dict:
    """A reference grid covering an AOI, independent of any scene.

    ALWAYS build the epoch grid from the ordered AOI, never by adopting the
    first scene's clip: when deliveries are strip tiles (frames covering
    fractions of the AOI -- the Dolan case), a scene-adopted grid silently
    confines every later scene to the first frame's corner and most of the
    window scores as no-data.
    """
    import numpy as np
    from rasterio.transform import from_origin
    from rasterio.warp import transform_bounds

    b = transform_bounds("EPSG:4326", crs, *bounds4326)
    return {"transform": from_origin(b[0], b[3], res, res), "crs": crs,
            "width": int(np.ceil((b[2] - b[0]) / res)),
            "height": int(np.ceil((b[3] - b[1]) / res))}


def scene_pairs(epoch_dir: Path) -> list[tuple[str, Path, Path]]:
    """(scene_id, sr_path, udm2_path) for every staged scene, sorted."""
    epoch_dir = Path(epoch_dir)
    out = []
    for sr in sorted(epoch_dir.glob("*_SR_*clip*.tif")):
        sid = sr.name.split("_3B_")[0]
        udm = sorted(epoch_dir.glob(f"{sid}*udm2*.tif"))
        if udm:
            out.append((sid, sr, udm[0]))
    if not out:
        sys.exit(f"no staged SR/udm2 scene pairs under {epoch_dir}")
    return out


def read_scene(sr: Path, udm: Path, ref: dict | None = None,
               bands: dict = BANDS_8B):
    """Reflectance bands + clear mask on the reference grid.

    ``ref=None`` adopts this scene's own grid and returns it for reuse.
    """
    with rasterio.open(sr) as s:
        if ref is None:
            ref = {"transform": s.transform, "crs": s.crs,
                   "width": s.width, "height": s.height}
        shape = (ref["height"], ref["width"])
        out = {}
        for name, idx in bands.items():
            dst = np.zeros(shape, np.float32)
            reproject(rasterio.band(s, idx), dst,
                      dst_transform=ref["transform"], dst_crs=ref["crs"],
                      resampling=Resampling.bilinear, dst_nodata=0.0)
            out[name] = dst / 1e4
    with rasterio.open(udm) as u:
        clear = np.zeros(shape, np.uint8)
        reproject(rasterio.band(u, 1), clear,
                  dst_transform=ref["transform"], dst_crs=ref["crs"],
                  resampling=Resampling.nearest, dst_nodata=0)
    bad = (clear != 1) | (out["nir"] <= 0)
    return {k: np.ma.masked_array(v, bad) for k, v in out.items()}, ref


#: Index name -> the band-dict function that computes it.
_INDEX_FNS = {
    "brightness": lambda b: ch.brightness(b["blue"], b["green"], b["red"],
                                          b["nir"]),
    "msavi2": lambda b: ch.msavi2(b["nir"], b["red"]),
    "ndvi": lambda b: ch.ndvi(b["nir"], b["red"]),
    "redness": lambda b: ch.redness(b["red"], b["green"]),
}


def build(epoch_dir: Path, ref: dict | None = None, *,
          indices: tuple = ("brightness", "msavi2", "ndvi", "redness"),
          rgb: bool = True, keep_nir: bool = True, dtype=np.float32,
          verbose: bool = True):
    """One epoch -> composites.

    Returns ``(stats, rgb, nirs, ids, ref)`` where ``stats`` maps index name
    -> ``(median, mad, count)`` from :func:`firescape.change.epoch_stats`,
    ``rgb`` is the median-reflectance display composite (``None`` when
    ``rgb=False``), and ``nirs`` keeps the per-scene NIR planes for
    registration checks (empty when ``keep_nir=False``).

    Memory scales as scenes x pixels x (len(indices) + 3*rgb + keep_nir) x
    itemsize. Big windows (the Dolan grid is ~36M px x 30 frames) should
    pass ``indices=("brightness","msavi2","ndvi"), rgb=False,
    keep_nir=False, dtype=np.float16`` -- reflectance indices live in
    [-1, 1.5], where float16 resolution (~5e-4) is far below the cross-scene
    noise the median is there to beat.
    """
    idx_stacks = {k: [] for k in indices}
    rgb_stacks = {k: [] for k in ("red", "green", "blue")} if rgb else None
    nirs, ids = [], []
    for sid, sr, udm in scene_pairs(epoch_dir):
        b, ref = read_scene(sr, udm, ref)
        for k in indices:
            idx_stacks[k].append(_INDEX_FNS[k](b).astype(dtype))
        if rgb:
            for k in rgb_stacks:
                rgb_stacks[k].append(b[k].astype(dtype))
        if keep_nir:
            nirs.append(b["nir"])
        ids.append(sid)
        if verbose:
            print(f"  {Path(epoch_dir).name}: {sid} loaded", flush=True)
    stats = {k: ch.epoch_stats(np.ma.stack(v)) for k, v in idx_stacks.items()}
    rgb_img = None
    if rgb:
        rgb_img = np.dstack([
            np.ma.median(np.ma.stack(rgb_stacks[k]), axis=0)
            .astype(np.float32).filled(0)
            for k in ("red", "green", "blue")])
    return stats, rgb_img, nirs, ids, ref
