"""Put the RANGES volume on the per-unit segment files, as the merge did for basins.

``sw_repair_ranges`` fixed the merged *basin* product, which left the two
halves of the surface disagreeing: basins carried RANGES while the 589 segment
files still carried Gartner-14 in ``V_24mmh``. Everything that reads segments
-- ``p12_urban_zoom_forecast``, ``firescape.exposure`` (whose ``SEGMENT_COLS``
pulls ``V_24mmh`` and ``H_24mmh`` straight onto every AML asset) -- was
therefore on a different volume model than the basin sheets beside it.

**RANGES applies per segment, not by analogy.** Its predictors are catchment
quantities, and ``prefire`` emits exactly those per segment: ``Area_km2`` is
``segments.area()``, and ``SlopeDeg``/``FracNorth`` come from pfdf's
``catchment_summary``/``catchment_ratio``. Each segment already carries the
statistics of the basin draining to *its* outlet, which is the same footing on
which Gartner's ``V_24mmh`` is computed there. The two remaining predictors are
smooth regional fields sampled at the segment centroid: NSHM-2023 PGA (already
a 25-km-radius average) and the Atlas 14 1-yr I15 behind the anomaly ratio.

Idempotent, and the guard matters more here than for basins because this walks
589 files: a unit that already carries ``Vg14_24mmh`` is recomputed from that
column, never from ``V_24mmh``, so a second pass cannot promote RANGES volumes
into the Gartner slot. Each file is staged in the local cache and moved in.

    python scripts/surface/sw_ranges_segments.py [statewide_v1_2] [--dry-run]
"""
import json
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import Affine

from firescape import hazard as hz
from firescape import paths

VERSION = next((a for a in sys.argv[1:] if not a.startswith("-")),
               "statewide_v1_2")
DRY = "--dry-run" in sys.argv
OUT = paths.products_dir("prefire", VERSION)
files = sorted(p for p in OUT.glob("*_segments.gpkg") if p.name[0].isdigit())
if not files:
    sys.exit(f"no per-unit segment files under {OUT}")
print(f"{len(files)} segment files in {VERSION}"
      f"{' (dry run)' if DRY else ''}", flush=True)

with rasterio.open(paths.interim_dir("statewide") / "pga25_g.tif") as pg:
    PGA, PGA_TR, PGA_CRS = pg.read(1), pg.transform, pg.crs
_z = np.load(paths.interim_dir("statewide") / "atlas14_i15.npz")
I1G, A14_TR = _z["i1"], Affine(*_z["transform"])
print(f"PGA grid {PGA.shape} {PGA_CRS}; Atlas 14 1-yr grid {I1G.shape}",
      flush=True)


def sample(grid, tr, xs, ys):
    """Nearest-cell sample; NaN outside. Both grids are far coarser than a
    segment, so nearest is the honest choice -- interpolating would imply a
    precision the source does not have."""
    cols = np.floor((xs - tr.c) / tr.a).astype(int)
    rows = np.floor((ys - tr.f) / tr.e).astype(int)
    ok = ((rows >= 0) & (rows < grid.shape[0])
          & (cols >= 0) & (cols < grid.shape[1]))
    out = np.full(len(xs), np.nan)
    out[ok] = grid[rows[ok], cols[ok]]
    return out


t0 = time.time()
tot = dict(segments=0, files=0, already=0, nan_v=0, no_pga=0, no_a14=0)
med_r, med_g = [], []
for i, f in enumerate(files):
    g = gpd.read_file(f)
    if not len(g):
        continue
    repaired = "Vg14_24mmh" in g.columns
    tot["already"] += int(repaired)
    g14 = g["Vg14_24mmh" if repaired else "V_24mmh"].to_numpy().copy()

    # A point ON the reach, not the centroid of its bounding geometry: a
    # sinuous segment's centroid can fall off the line, and on a canyon reach
    # that puts the sample on the far side of a ridge.
    cent = g.geometry.representative_point()
    cp = cent if cent.crs == PGA_CRS else cent.to_crs(PGA_CRS)
    pga = sample(PGA, PGA_TR, cp.x.to_numpy(), cp.y.to_numpy())
    ll = cent.to_crs("EPSG:4269")
    i1 = sample(I1G, A14_TR, ll.x.to_numpy(), ll.y.to_numpy())

    ratio = 24.0 / i1
    V, Vmin, Vmax = hz.volume_ranges(g["Area_km2"].to_numpy(),
                                     g["SlopeDeg"].to_numpy(), ratio, pga,
                                     g["FracNorth"].to_numpy())
    tot["segments"] += len(g)
    tot["nan_v"] += int((~np.isfinite(V)).sum())
    tot["no_pga"] += int((~np.isfinite(pga)).sum())
    tot["no_a14"] += int((~np.isfinite(i1)).sum())
    med_r.append(np.nanmedian(V)); med_g.append(np.nanmedian(g14))

    if not DRY:
        g["PGA25_g"], g["i15_ratio"], g["I15_1yr"] = pga, ratio, i1
        g["Vg14_24mmh"] = g14
        g["Hg14_24mmh"] = hz.combined_c10(g["P_24mmh"].to_numpy(), g14)
        g["V_24mmh"], g["Vmin_24mmh"], g["Vmax_24mmh"] = V, Vmin, Vmax
        g["H_24mmh"] = hz.combined_c10(g["P_24mmh"].to_numpy(), V)
        tmp = paths.cache_root() / f"{f.stem}.ranges.gpkg"
        tmp.unlink(missing_ok=True)
        g.to_file(tmp, driver="GPKG")
        paths.atomic_into(tmp, f)
    tot["files"] += 1
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(files)}  ({time.time()-t0:.0f}s)", flush=True)

report = {
    "version": VERSION,
    "files": tot["files"],
    "segments": tot["segments"],
    "files_already_repaired": tot["already"],
    "segments_without_ranges_volume": tot["nan_v"],
    "segments_off_pga_grid": tot["no_pga"],
    "segments_off_atlas14_grid": tot["no_a14"],
    "median_of_unit_medians_ranges_m3": round(float(np.nanmedian(med_r)), 1),
    "median_of_unit_medians_gartner_m3": round(float(np.nanmedian(med_g)), 1),
    "elapsed_s": round(time.time() - t0, 1),
}
print(json.dumps(report, indent=2), flush=True)
if not DRY:
    (OUT / f"{VERSION}_segments_ranges.json").write_text(
        json.dumps(report, indent=2))
    print(f"wrote {VERSION}_segments_ranges.json", flush=True)
