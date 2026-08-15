"""Close the Lake Tahoe / Truckee River basin across the California line.

The statewide inventory was built as "every HU10 unit that intersects Nevada",
which is right for a Nevada product but cuts a cross-border river system in
half. Coverage stops at lon -120.183 -- Tahoe's west shore -- so the west-shore
drainages (Blackwood, Ward, McKinney creeks) and the Truckee River reach from
the lake outlet through Truckee to the state line are missing, even though
they feed the same river that runs through Reno.

Every input those units need is already staged: the 3DEP tiles n39w120/121 and
n40w120/121 are in the cache, and the LF2025 EVT chunks reach to lon -121.03.
Only the HU10 polygons were never fetched, because they never touched Nevada.

This adds the missing units of HU8 16050101 (Lake Tahoe) and 16050102 (Truckee
River) to the unit list, leaving the file otherwise untouched so the existing
591 products stay valid and only the new units run.
"""
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import pandas as pd

from firescape import paths, wbd

HU_PATH = paths.interim_dir("statewide") / "nv_hu10.geojson"
WANT_HU8 = ("16050101", "16050102")          # Lake Tahoe, Truckee River

hu = gpd.read_file(HU_PATH)
hu["huc10"] = hu["huc10"].astype(str)
have = set(hu["huc10"])
print(f"{len(hu)} units currently staged; "
      f"{sum(c[:8] in WANT_HU8 for c in have)} in the Tahoe/Truckee HU8s",
      flush=True)

# the whole basin sits inside this window; WBD returns any HU10 intersecting it
fetched = wbd.units((-120.60, 38.85, -119.85, 39.75))
fetched["huc10"] = fetched["huc10"].astype(str)
missing = fetched[fetched.huc10.str[:8].isin(WANT_HU8) & ~fetched.huc10.isin(have)]
missing = missing.drop_duplicates(subset="huc10")

if missing.empty:
    print("nothing missing -- the basin is already complete")
    raise SystemExit(0)

print(f"\n{len(missing)} unit(s) to add:", flush=True)
for _, r in missing.iterrows():
    print(f"   {r.huc10}  {str(r.get('name'))[:46]:<48} {r.get('areasqkm', float('nan')):7.0f} km2",
          flush=True)

# align columns to the existing file, then append
cols = [c for c in hu.columns if c in missing.columns]
missing = missing[cols].to_crs(hu.crs)
out = pd.concat([hu, missing], ignore_index=True)
out = gpd.GeoDataFrame(out, geometry="geometry", crs=hu.crs)
out = out.drop_duplicates(subset="huc10").reset_index(drop=True)

backup = HU_PATH.with_suffix(".geojson.bak")
if not backup.exists():
    HU_PATH.replace(backup)
    print(f"\noriginal kept at {backup.name}", flush=True)
out.to_file(HU_PATH, driver="GeoJSON")
print(f"unit list now {len(out)} HU10s -> {HU_PATH}", flush=True)
print("\nrun the v1.2 fleet again; it skips finished units and picks up only "
      "the new ones, then re-merge.", flush=True)
