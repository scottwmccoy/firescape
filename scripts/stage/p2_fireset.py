"""Statewide era-matched calibration fire set (phase 2).

Selection follows Rossi et al. 2025 calibration criteria plus our era-matching
rule: fires >=10 km2 igniting 2017+ (so LF2016 is the oldest defensible
pre-fire vegetation), intersecting Nevada + 15 km, with usable analyst
thresholds. Era pairs: vintage_year <= ig_year - 1 ->
2017-2020 = LF2016_EVT, 2021-2022 = LF2020_EVT, 2023+ = LF2022_EVT.
"""
import warnings

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd

from firescape import mtbs, paths

REPO_SETS = (paths.package_data("calibration", "fire_sets"))
MIN_KM2 = 10.0
AC_TO_KM2 = 0.00404686

nv = gpd.read_file(paths.raw_dir("boundaries") / "nv_state.geojson")
buf = gpd.GeoSeries(nv.to_crs("EPSG:5070").buffer(15_000).union_all(),
                    crs="EPSG:5070").to_crs("EPSG:4326").iloc[0]

frames = []
for st in ("NV", "CA", "OR", "ID", "UT", "AZ"):
    gdf = mtbs.fire_records(event_id_like=f"{st}%", after="2016-12-31",
                            min_acres=2471)
    gdf = gdf[gdf.intersects(buf)]
    print(f"{st}: {len(gdf)} fires 2017+ touching NV+15km", flush=True)
    frames.append(gdf)
fires = pd.concat(frames, ignore_index=True)
fires = mtbs.dedupe_records(gpd.GeoDataFrame(fires, crs="EPSG:4326"))

fires["ig_year"] = fires["ig_date"].dt.year
fires["km2"] = fires["burnbndac"] * AC_TO_KM2
fires["incid_type"] = fires["incid_type"].astype(str)

print("\nby incident type (all sizes):")
print(fires["incid_type"].value_counts().to_string())

big = fires[(fires["km2"] >= MIN_KM2)
            & fires["incid_type"].isin(["Wildfire", "Unknown"])].copy()
# mod_t=9999 is the MTBS "no analyst threshold" sentinel. Such fires still
# calibrate P_dsim (their observed classes come from dnbr6, not mod_t) but
# must stay OUT of the regional break median -- Rossi's rule is a median
# of real analyst thresholds. thresholds_ok gates the break median only.
# BAER-programme records can carry GTAC's 0-255 BARC256 thresholds instead
# of dNBR (three 2024 fires did: Davis 158, Bear 120, Broom Canyon 111, and
# they sat inside the v1..v1_3 Sierra Nevada break median, 312 vs 350).
# Convert BEFORE anything reads the numbers.
big = mtbs.normalize_thresholds(big)
big["thresholds_ok"] = (big["mod_t"].fillna(0) > 0) & (big["mod_t"].fillna(9999) < 2000)
# newest LFPS vintage strictly predating ignition; the catalog holds only
# LF2016/LF2022/LF2023/LF2024/LF2025 (no LF2020), so 2017-2022 pair with
# LF2016 (up to 6-yr veg lag for 2021-22 fires - documented caveat)
big["evt_vintage"] = np.select(
    [big["ig_year"] <= 2022, big["ig_year"] <= 2023],
    ["LF2016_EVT", "LF2022_EVT"], default="LF2023_EVT")
have = {p.name for p in (paths.raw_dir("mtbs", "fires")).iterdir() if p.is_dir()}
big["have_bundle"] = big["event_id"].isin(have)
big["include"] = big["mod_t"].fillna(0) > 0   # as-run membership (see above)

cent = big.geometry.centroid
big["lon"], big["lat"] = cent.x.round(4), cent.y.round(4)

print(f"\n=== statewide era-matched set: {len(big)} fires >= {MIN_KM2:.0f} km2 ===")
print("by ignition year:")
print(big.groupby("ig_year").agg(n=("event_id", "size"),
                                 km2=("km2", "sum")).round(0).to_string())
print("\nby EVT vintage pair:")
print(big.groupby("evt_vintage")["event_id"].count().to_string())
print(f"\nthresholds usable: {int(big['thresholds_ok'].sum())}/{len(big)}"
      f"   bundles already on disk: {int(big['have_bundle'].sum())}")
print(f"largest: ")
print(big.nlargest(8, "km2")[["event_id", "incid_name", "ig_year", "km2"]]
      .round(0).to_string(index=False))

cols = ["event_id", "incid_name", "ig_date", "ig_year", "incid_type", "km2",
        "map_prog", "asmnt_type", "low_t", "mod_t", "high_t", "threshold_scale",
        "dnbr_offst", "evt_vintage", "have_bundle", "thresholds_ok", "include",
        "lon", "lat"]
out_csv = f"{REPO_SETS}/statewide_v1.csv"
big.sort_values(["ig_year", "km2"], ascending=[True, False])[cols].to_csv(
    out_csv, index=False)
print(f"\nwrote {out_csv}")

slim = big[["event_id", "incid_name", "ig_year", "km2", "evt_vintage",
            "include", "geometry"]]
gj = paths.interim_dir("statewide") / "calib_fires_v1.geojson"
slim.to_file(gj, driver="GeoJSON")
print(f"wrote {gj}")

need = sorted(big.loc[big["include"] & ~big["have_bundle"], "event_id"])
print(f"\nbundles to order: {len(need)}")
