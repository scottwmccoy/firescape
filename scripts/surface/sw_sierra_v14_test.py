"""What does statewide_v1_4 do to the Sierra Nevada pre-fire surface?

The v1_4 recalibration changed only Sierra Nevada: P_dsim 0.47 -> 0.48 and
the low/moderate break 312 -> 350. Nine of 589 HU10 units. Before deciding
whether the statewide product needs regenerating, run those nine under the
new pair -- and under a break-only control (0.47, 350) so the change splits
into its two parts -- into a scratch directory, and diff them against the
shipped statewide_v1_2 units (whose Sierra pair is the v1_3 pair).

Not a product. Output: products/prefire/_sierra_v1_4_test/<variant>/.

    python scripts/surface/sw_sierra_v14_test.py
"""
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import pandas as pd

from firescape import config, paths, prefire, severity, statewide

T0 = time.time()
TILE_DIR = paths.cache_root() / "3dep_tiles"
EVT_DIR = paths.raw_dir("landfire", "LF2025_EVT_nv")
legend_csv = paths.interim_dir("statewide") / "evt_legend_union.csv"
mapping, _ = severity.remap_crosswalk(pd.read_csv(legend_csv))
regions = pd.read_csv(paths.interim_dir("statewide") / "hu10_regions.csv", dtype={"huc10": str})
sierra = set(regions.loc[regions.region == "Sierra Nevada", "huc10"])

cal = config.packaged_calibration("statewide_v1_4").region("sierra_nevada")
VARIANTS = {"v1_4": (cal.pdsim, cal.barc_breaks),
            "break_only": (0.47, cal.barc_breaks)}          # v1_3 P_dsim, v1_4 break
print("variants:", VARIANTS, flush=True)

hu = gpd.read_file(paths.interim_dir("statewide") / "nv_hu10.geojson")
hu = hu[hu["huc10"].astype(str).isin(sierra)].sort_values("areasqkm").reset_index(drop=True)
print(f"{len(hu)} Sierra Nevada units", flush=True)

for variant, (pdsim, breaks) in VARIANTS.items():
    out = paths.products_dir("prefire", "_sierra_v1_4_test", variant)
    out.mkdir(parents=True, exist_ok=True)
    cfg = prefire.PreFireConfig(
        pdsim=pdsim, barc_breaks=tuple(breaks), evt_path=legend_csv, evt_legend_path=legend_csv,
        crosswalk=mapping, dem_path=paths.interim_dir("pilot") / "pilot_dem.tif",
        region=f"_sierra_v1_4_test:{variant}", severity_sigma=0.91, emit_ranges_topo=True,
        cdf_table="nv_statewide", kf_polygons=paths.raw_dir("ssurgo") / "nv_ssurgo_kf.gpkg")
    for _, row in hu.iterrows():
        key = str(row["huc10"])
        if (out / f"{key}_meta.json").exists():
            continue
        t1 = time.time()
        try:
            g5070 = gpd.GeoSeries([row.geometry], crs=hu.crs).to_crs("EPSG:5070")
            b5070 = tuple(g5070.total_bounds)
            dem = statewide.dem_for_unit(b5070, TILE_DIR)
            evt = statewide.evt_for_unit(b5070, EVT_DIR)
            res = prefire.run_unit(row.geometry, cfg, unit_key=f"{variant}_{key}", crs=hu.crs, dem=dem, evt=evt)
            prefire.save_unit(res, out)
            print(f"{variant:10s} {key} {str(row.get('name'))[:22]:<22} {row['areasqkm']:6.0f} km2 -> "
                  f"{res['n_segments']:5d} segs ({time.time()-t1:.0f}s)", flush=True)
        except Exception as e:
            print(f"{variant:10s} {key} FAILED: {type(e).__name__}: {str(e)[:140]}", flush=True)
print(f"\ndone in {time.time()-T0:.0f}s", flush=True)
