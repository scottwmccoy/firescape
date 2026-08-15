"""Stage statewide KF: STATSGO if ScienceBase answers, else SSURGO tiles."""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd

from firescape import paths, soils

T0 = time.time()
BUDGET = float(os.environ.get("FIRESCAPE_STAGE_BUDGET", 480))
PARTS = paths.interim_dir("statewide", "ssurgo_parts")
FINAL = paths.raw_dir("ssurgo") / "nv_ssurgo_kf.gpkg"

if FINAL.exists():
    print("statewide soils already staged")
    sys.exit(0)

# one cheap STATSGO probe per run - if ScienceBase is back we note it (the
# fleet's per-unit fallback order already prefers supplied polygons)
try:
    from pfdf.data.usgs import statsgo
    from pfdf.projection import BoundingBox

    statsgo.read("KFFACT", bounds=BoundingBox(-116.5, 39.0, -116.45, 39.05, crs=4326),
                 timeout=25)
    print("NOTE: ScienceBase/STATSGO is UP again", flush=True)
except Exception:
    print("STATSGO still unreachable; SSURGO it is", flush=True)

hu = gpd.read_file(paths.interim_dir("statewide") / "nv_hu10.geojson")
w0, s0, e0, n0 = hu.total_bounds
xs = np.arange(np.floor(w0 * 2) / 2, e0, 0.5)
ys = np.arange(np.floor(s0 * 2) / 2, n0, 0.5)
from shapely.geometry import box

tiles = [(x, y) for x in xs for y in ys if hu.intersects(box(x, y, x + 0.5, y + 0.5)).any()]
print(f"{len(tiles)} SSURGO tiles over the statewide units", flush=True)

done, pending = 0, 0
for x, y in tiles:
    part = PARTS / f"kf_{int(-x*10):04d}w_{int(y*10):04d}n.parquet"
    if part.exists():
        done += 1
        continue
    if time.time() - T0 > BUDGET:
        pending += 1
        continue
    try:
        g = soils._wfs_tile((x, y, x + 0.5, y + 0.5), timeout=240, scratch=paths.cache_root())
        if len(g):
            lower = {c.lower(): c for c in g.columns}
            g["mukey"] = g[lower["mukey"]].astype("int64")
            if "mupolygonkey" in lower:
                g = g.drop_duplicates(subset=[lower["mupolygonkey"]])
            g[["mukey", "geometry"]].to_parquet(part)
        else:
            part.write_bytes(b"")     # empty marker: no soils mapped here
        done += 1
        print(f"  ({x:.1f},{y:.1f}) {len(g)} polys", flush=True)
    except Exception as e:
        print(f"  ({x:.1f},{y:.1f}) {type(e).__name__}", flush=True)

if pending == 0:
    frames = [gpd.read_parquet(p) for p in sorted(PARTS.glob("*.parquet"))
              if p.stat().st_size > 0]
    allsoil = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    print(f"assembled {len(allsoil)} polygons; querying kffact for "
          f"{allsoil['mukey'].nunique()} map units", flush=True)
    kf_map = soils._sda_kffact(sorted(allsoil["mukey"].unique()), timeout=240)
    allsoil["kf"] = allsoil["mukey"].map(kf_map)
    allsoil.loc[allsoil["kf"] < 0, "kf"] = np.nan
    allsoil.to_file(FINAL, driver="GPKG")
    paths.write_provenance(FINAL, url=soils.WFS_URL,
                           note="statewide SSURGO mapunitpoly + SDA surface-horizon "
                                "kffact (component-weighted); STATSGO unavailable")
    print(f"wrote {FINAL}: {len(allsoil)} polygons, "
          f"kf resolved {allsoil['kf'].notna().mean():.1%}", flush=True)

print(f"\nsoils: {done} tiles done, {pending} pending, {time.time()-T0:.0f}s", flush=True)
sys.exit(42 if pending else 0)
