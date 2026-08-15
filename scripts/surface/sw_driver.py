"""Statewide v0 fleet: 591 HU10 units, pilot_v2 calibration applied statewide.

PROVISIONAL by construction: one calibration region (northwest-Nevada values),
labeled as such in every unit's metadata. Resumable; exit 42 = budget hit.
Set FIRESCAPE_LANE=0/1 (with FIRESCAPE_NLANES=2) to split units across two
concurrent drivers.
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd

from firescape import config, paths, prefire, severity, statewide

T0 = time.time()
BUDGET = float(os.environ.get("FIRESCAPE_M4_BUDGET", 480))
LANE = int(os.environ.get("FIRESCAPE_LANE", 0))
NLANES = int(os.environ.get("FIRESCAPE_NLANES", 1))
ONLY = os.environ.get("FIRESCAPE_ONLY")
OUT = paths.products_dir("prefire", "statewide_v0")
TILE_DIR = paths.cache_root() / "3dep_tiles"
EVT_DIR = paths.raw_dir("landfire", "LF2025_EVT_nv")

cal = config.packaged_calibration("pilot_v2").region("pilot")
vat_paths = sorted(EVT_DIR.glob("*/evt_*.tif.vat.dbf"))
import pandas as pd
legend = pd.concat([gpd.read_file(v) for v in vat_paths],
                   ignore_index=True).drop_duplicates(subset="Value")
legend_csv = paths.interim_dir("statewide") / "evt_legend_union.csv"
legend.drop(columns=[c for c in legend.columns if c == "geometry"],
            errors="ignore").to_csv(legend_csv, index=False)
vat_path = legend_csv
mapping, _ = severity.remap_crosswalk(legend)

cfg = prefire.PreFireConfig(
    pdsim=cal.pdsim,
    barc_breaks=cal.barc_breaks,
    evt_path=vat_path,                # unused (evt injected); kept for meta
    evt_legend_path=vat_path,
    crosswalk=mapping,
    dem_path=paths.interim_dir("pilot") / "pilot_dem.tif",   # unused (dem injected)
    region="statewide_v0 (pilot calibration applied statewide - PROVISIONAL)",
    cdf_table="nv_merged",
    kf_polygons=paths.raw_dir("ssurgo") / "nv_ssurgo_kf.gpkg",
)

hu = gpd.read_file(paths.interim_dir("statewide") / "nv_hu10.geojson")
hu = hu.sort_values("areasqkm").reset_index(drop=True)
hu = hu[hu.index % NLANES == LANE]
if ONLY:
    hu = hu[hu["huc10"] == ONLY]

done, pending, failed = [], [], []
for _, row in hu.iterrows():
    key = str(row["huc10"])
    if (OUT / f"{key}_meta.json").exists():
        done.append(key)
        continue
    if time.time() - T0 > BUDGET:
        pending.append(key)
        continue
    t1 = time.time()
    try:
        g5070 = gpd.GeoSeries([row.geometry], crs=hu.crs).to_crs("EPSG:5070")
        b5070 = tuple(g5070.total_bounds)
        dem = statewide.dem_for_unit(b5070, TILE_DIR)
        evt = statewide.evt_for_unit(b5070, EVT_DIR)
        result = prefire.run_unit(row.geometry, cfg, unit_key=key, crs=hu.crs,
                                  dem=dem, evt=evt)
        prefire.save_unit(result, OUT)
        print(f"{key} {str(row.get('name'))[:24]:<24} {row['areasqkm']:6.0f} km2 -> "
              f"{result['n_segments']:5d} segs "
              f"({result['meta'].get('kf_source', '?')[:14]}, {time.time()-t1:.0f}s)",
              flush=True)
        done.append(key)
    except FileNotFoundError as e:
        print(f"{key} WAITING: {e}", flush=True)   # tiles not staged yet
        pending.append(key)
    except Exception as e:
        print(f"{key} FAILED: {type(e).__name__}: {str(e)[:140]}", flush=True)
        failed.append(key)

print(f"\nlane {LANE}/{NLANES}: {len(done)} done, {len(pending)} pending, "
      f"{len(failed)} failed, {time.time()-T0:.0f}s", flush=True)
sys.exit(42 if pending else 0)
