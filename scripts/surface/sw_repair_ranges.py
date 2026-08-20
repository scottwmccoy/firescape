"""Repair a merged statewide product whose RANGES volume columns never landed.

``sw_merge_v11``/``sw_merge_v12`` wrote the merged basin GPKG three assignments
too early: the RANGES block ran afterwards, overwrote ``V_24mmh``/``H_24mmh``
in memory, and built the summary from those -- but nothing wrote the frame
again. So the product on disk carried Gartner-14 volumes and a Gartner hazard
class while its own summary said ``"volume_model": "RANGES (headline)"``, and
``Vg14_24mmh``, ``Vmin_24mmh``, ``Vmax_24mmh``, ``PGA25_g`` and ``i15_ratio``
were never persisted at all. Measured on statewide_v1_2: the file's hazard
class matched ``hazard_class_g14`` exactly (368,090/44,410/1) and not the
headline ``hazard_class`` (373,057/32,384/0) it was published under.

The merge scripts now write once, last. This repairs the products already on
disk without re-reading 591 unit files, applying exactly the block those
scripts apply. Idempotent: a file that already carries ``Vg14_24mmh`` is
recomputed from ``Vg14_24mmh`` rather than from ``V_24mmh``, so running it
twice cannot promote RANGES volumes into the Gartner column.

    python scripts/surface/sw_repair_ranges.py statewide_v1_2 [--dry-run]
"""
import json
import sys
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import rasterio

from firescape import hazard as hz
from firescape import paths

VERSION = sys.argv[1] if len(sys.argv) > 1 else "statewide_v1_2"
DRY = "--dry-run" in sys.argv
OUT = paths.products_dir("prefire", VERSION)
GP = OUT / f"{VERSION}_basins.gpkg"
if not GP.exists():
    sys.exit(f"no merged basins for {VERSION}: {GP}")

basins = gpd.read_file(GP)
print(f"{len(basins):,} basins from {GP.name}", flush=True)

# Idempotence: on a repaired file V_24mmh is already RANGES, so the Gartner
# source of truth is Vg14_24mmh. On a broken one it is V_24mmh itself.
repaired_before = "Vg14_24mmh" in basins.columns
g14 = basins["Vg14_24mmh" if repaired_before else "V_24mmh"].to_numpy().copy()
print("already repaired once; recomputing from Vg14_24mmh" if repaired_before
      else "first repair; V_24mmh currently holds Gartner-14", flush=True)

with rasterio.open(paths.interim_dir("statewide") / "pga25_g.tif") as pg:
    arr, tr = pg.read(1), pg.transform
cent = basins.geometry.centroid
cols = ((cent.x - tr.c) / tr.a).astype(int).to_numpy()
rows = ((cent.y - tr.f) / tr.e).astype(int).to_numpy()
ok = (rows >= 0) & (rows < arr.shape[0]) & (cols >= 0) & (cols < arr.shape[1])
pga = np.full(len(basins), np.nan)
pga[ok] = arr[rows[ok], cols[ok]]

ratio = 24.0 / basins["I15_1yr"].to_numpy()
V, Vmin, Vmax = hz.volume_ranges(basins["Area_km2"].to_numpy(),
                                 basins["SlopeDeg"].to_numpy(), ratio, pga,
                                 basins["FracNorth"].to_numpy())
Hr = hz.combined_c10(basins["P_24mmh"].to_numpy(), V)
Hg = hz.combined_c10(basins["P_24mmh"].to_numpy(), g14)

report = {
    "version": VERSION,
    "volume_model": "RANGES (headline); Gartner-14 soft-Bmh retained as *_g14",
    "basins": int(len(basins)),
    "pga_missing": int((~ok).sum()),
    "ranges_nan": int((~np.isfinite(V)).sum()),
    "median_V_ranges_m3": round(float(np.nanmedian(V)), 1),
    "median_V_g14_m3": round(float(np.nanmedian(g14)), 1),
    "hazard_class": {int(c): int((Hr == c).sum()) for c in (1, 2, 3)},
    "hazard_class_g14": {int(c): int((Hg == c).sum()) for c in (1, 2, 3)},
}
print(json.dumps(report, indent=2), flush=True)
if DRY:
    sys.exit("dry run: nothing written")

basins["PGA25_g"] = pga
basins["i15_ratio"] = ratio
basins["Vg14_24mmh"] = g14
basins["Hg14_24mmh"] = Hg
basins["V_24mmh"], basins["Vmin_24mmh"], basins["Vmax_24mmh"] = V, Vmin, Vmax
basins["H_24mmh"] = Hr

# Staged locally, then moved in: this is a 412k-feature product on Box, and a
# write that dies partway through an in-place GPKG rewrite leaves no file to
# go back to. Same rule the downloads follow.
tmp = paths.cache_root() / f"{VERSION}_basins.repair.gpkg"
tmp.unlink(missing_ok=True)
basins.to_file(tmp, driver="GPKG")
paths.atomic_into(tmp, GP)
print(f"rewrote {GP.name} with RANGES as the headline volume", flush=True)

# Bring the summary's own numbers back into agreement with the file.
sp = OUT / f"{VERSION}_summary.json"
if sp.exists():
    s = json.loads(sp.read_text())
    s["hazard_class"] = {str(k): v for k, v in report["hazard_class"].items()}
    s["hazard_class_g14"] = {str(k): v
                             for k, v in report["hazard_class_g14"].items()}
    s["zero_volume_frac"] = float((~np.isfinite(V) | (V <= 0)).mean())
    s["ranges_repair"] = (
        "RANGES columns written by sw_repair_ranges.py; the original merge "
        "wrote the GPKG before the RANGES block, so the published file "
        "carried Gartner-14 under the RANGES summary")
    sp.write_text(json.dumps(s, indent=2))
    print(f"updated {sp.name}", flush=True)
