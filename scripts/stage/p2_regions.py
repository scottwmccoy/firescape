"""Nevada prefire calibration regions from EPA Level III ecoregions.

Rossi et al. 2025 calibrated P_dsim per CGS 'prefire region'; the Nevada
analog is the Omernik/EPA Level III ecoregions clipped to the statewide HU10
domain. Regions with too few calibration fires merge into their dominant
neighbor (documented in the output). Assignments are snapped to whole HU10
units (centroid rule) so every statewide unit gets exactly one region.
"""
import json
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd

from firescape import paths

MIN_FIRES = 5
MERGE = {  # semantic fallbacks for sliver/undersampled L3 regions in the domain
    "Snake River Plain": "Northern Basin and Range",
    "Eastern Cascades Slopes and Foothills": "Northern Basin and Range",
    "Blue Mountains": "Northern Basin and Range",
    "Colorado Plateaus": "Central Basin and Range",
    "Wasatch and Uinta Mountains": "Central Basin and Range",
    "Arizona/New Mexico Plateau": "Mojave Basin and Range",
    "Sonoran Basin and Range": "Mojave Basin and Range",
}

eco = gpd.read_file("zip://" + str(paths.raw_dir("boundaries") / "us_eco_l3.zip"))
eco = eco.to_crs("EPSG:5070")
hu = gpd.read_file(paths.interim_dir("statewide") / "nv_hu10.geojson").to_crs("EPSG:5070")
fires = gpd.read_file(paths.interim_dir("statewide") / "calib_fires_v1.geojson").to_crs("EPSG:5070")
fire_set = pd.read_csv(paths.package_data("calibration", "fire_sets", "statewide_v1.csv"))

# ecoregions present in the domain
dom = hu.union_all().convex_hull
eco = eco[eco.intersects(dom)].copy()
eco["region"] = eco["US_L3NAME"].replace(MERGE)
eco = eco.dissolve(by="region", as_index=False)[["region", "geometry"]]
print("L3 regions in domain (after semantic merges):")
print(" ", ", ".join(sorted(eco["region"])))


def assign(gdf):
    cent = gpd.GeoDataFrame(geometry=gdf.geometry.centroid, crs=gdf.crs)
    j = gpd.sjoin(cent, eco, how="left", predicate="within")
    out = j["region"]
    # centroids in slivers outside any polygon -> nearest region
    miss = out.isna()
    if miss.any():
        near = gpd.sjoin_nearest(cent[miss], eco, how="left")
        out.loc[miss] = near["region"].values
    return out.values


fires["region"] = assign(fires)
hu["region"] = assign(hu)

# fire counts per region (era-matched, usable thresholds)
ok = fires.merge(fire_set[["event_id", "thresholds_ok"]], on="event_id")
counts = ok[ok["thresholds_ok"] & ok["include"]].groupby("region").size()
print("\ncalibration fires per region:")
print(counts.to_string())

# fold regions still under MIN_FIRES into the neighbor with the longest shared
# boundary inside the domain
folds = {}
for r in counts[counts < MIN_FIRES].index.tolist() + \
        [r for r in eco["region"] if r not in counts.index]:
    geom = eco.loc[eco["region"] == r, "geometry"].iloc[0]
    best, blen = None, -1.0
    for _, o in eco[eco["region"] != r].iterrows():
        if counts.get(o["region"], 0) < MIN_FIRES:
            continue
        L = geom.boundary.intersection(o.geometry.buffer(500)).length
        if L > blen:
            best, blen = o["region"], L
    if best:
        folds[r] = best
if folds:
    print("\nfolded (too few fires):", folds)
    fires["region"] = fires["region"].replace(folds)
    hu["region"] = hu["region"].replace(folds)
    eco["region"] = eco["region"].replace(folds)
    eco = eco.dissolve(by="region", as_index=False)

counts = (fires.merge(fire_set[["event_id", "thresholds_ok"]], on="event_id")
          .query("thresholds_ok & include").groupby("region").size())
print("\nfinal regions:")
for r in sorted(eco["region"]):
    nu = int((hu["region"] == r).sum())
    print(f"  {r:<28} fires={counts.get(r, 0):>3}  HU10 units={nu:>3}")

# outputs
eco44 = eco.to_crs("EPSG:4326")
repo = (paths.package_data("regions", "nv_prefire_regions.geojson"))
eco44.to_file(repo, driver="GeoJSON")
hu_out = hu[["huc10", "region"]].copy()
hu_out["huc10"] = hu_out["huc10"].astype(str)
hu_out.to_csv(paths.interim_dir("statewide") / "hu10_regions.csv", index=False)
fs = fire_set.merge(
    fires[["event_id", "region"]].drop_duplicates("event_id"), on="event_id",
    how="left")
fs.to_csv(paths.package_data("calibration", "fire_sets", "statewide_v1.csv"), index=False)
print(f"\nwrote {repo}")
print("wrote hu10_regions.csv; statewide_v1.csv now carries a region column")
