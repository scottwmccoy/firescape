"""Statewide v1 surface driver: per-region calibration (statewide_v1.toml).

Clone of the v0 driver with one change: each HU10 unit runs under its EPA-L3
region's calibrated pdsim + breaks (hu10_regions.csv lookup). Resumable
(exit 42), lane-parallel via FIRESCAPE_LANE/NLANES.
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import pandas as pd

from firescape import config, paths, prefire, severity, statewide

T0 = time.time()
BUDGET = float(os.environ.get("FIRESCAPE_M4_BUDGET", 520))
LANE = int(os.environ.get("FIRESCAPE_LANE", 0))
NLANES = int(os.environ.get("FIRESCAPE_NLANES", 1))
OUT = paths.products_dir("prefire", "statewide_v1")
TILE_DIR = paths.cache_root() / "3dep_tiles"
EVT_DIR = paths.raw_dir("landfire", "LF2025_EVT_nv")

legend = pd.read_csv(paths.interim_dir("statewide") / "evt_legend_union.csv")
mapping, _ = severity.remap_crosswalk(legend)
legend_csv = paths.interim_dir("statewide") / "evt_legend_union.csv"

regions = pd.read_csv(paths.interim_dir("statewide") / "hu10_regions.csv",
                      dtype={"huc10": str})
region_of = dict(zip(regions["huc10"], regions["region"]))


def slug(r):
    return r.lower().replace(" ", "_").replace("&", "and")


cal_pack = config.packaged_calibration("statewide_v1")
cfgs = {}
for name in sorted(set(region_of.values())):
    cal = cal_pack.region(slug(name))
    cfgs[name] = prefire.PreFireConfig(
        pdsim=cal.pdsim,
        barc_breaks=cal.barc_breaks,
        evt_path=legend_csv,              # unused (evt injected); kept for meta
        evt_legend_path=legend_csv,
        crosswalk=mapping,
        dem_path=paths.interim_dir("pilot") / "pilot_dem.tif",  # unused (dem injected)
        region=f"statewide_v1:{slug(name)}",
        cdf_table="nv_merged",
        kf_polygons=paths.raw_dir("ssurgo") / "nv_ssurgo_kf.gpkg",
    )
    print(f"{name}: pdsim {cal.pdsim}, breaks {cal.barc_breaks}", flush=True)

hu = gpd.read_file(paths.interim_dir("statewide") / "nv_hu10.geojson")
hu = hu.sort_values("areasqkm").reset_index(drop=True)
hu = hu[hu.index % NLANES == LANE]

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
        cfg = cfgs[region_of[key]]
        g5070 = gpd.GeoSeries([row.geometry], crs=hu.crs).to_crs("EPSG:5070")
        b5070 = tuple(g5070.total_bounds)
        dem = statewide.dem_for_unit(b5070, TILE_DIR)
        evt = statewide.evt_for_unit(b5070, EVT_DIR)
        result = prefire.run_unit(row.geometry, cfg, unit_key=key, crs=hu.crs,
                                  dem=dem, evt=evt)
        prefire.save_unit(result, OUT)
        print(f"{key} {str(row.get('name'))[:22]:<22} "
              f"{region_of[key][:14]:<14} {row['areasqkm']:6.0f} km2 -> "
              f"{result['n_segments']:5d} segs ({time.time()-t1:.0f}s)",
              flush=True)
        done.append(key)
    except FileNotFoundError as e:
        print(f"{key} WAITING: {e}", flush=True)
        pending.append(key)
    except Exception as e:
        print(f"{key} FAILED: {type(e).__name__}: {str(e)[:140]}", flush=True)
        failed.append(key)

print(f"\nlane {LANE}/{NLANES}: {len(done)} done, {len(pending)} pending, "
      f"{len(failed)} failed, {time.time()-T0:.0f}s", flush=True)
sys.exit(42 if pending else 0)
