"""Delineate the Bear Fire (2024) debris-flow channel network for tracescape.

Why this script lives **here** and not in tracescape: it imports ``pfdf``,
which is GPL-3. tracescape is MIT and must never import it. The boundary is a
file format -- tracescape's ``corridor.py`` accepts "pfdf's, or any LineString
layer" -- so the network crosses as a **GeoPackage**, not as a call.

What makes this different from ``assess.run_observed``: that applies the USGS
filter chain, which includes ``in_perimeter`` and a burn-ratio floor, and so
returns only channels inside the fire. The Bear change-detection experiment
needs the opposite -- the same steep channels **outside** the burn, as the
negative control that says whether the change maps light up only where the
fire was. So only the geometric filters are applied (area, slope,
confinement); burn state is recorded as an *attribute*, never as a filter.

The DEM is fetched over the AOI plus a further ``DEM_PAD_M`` so that flow
accumulation for edge segments is not truncated by the fetch footprint;
the segment mask is still confined to the AOI.

Run: /opt/anaconda3/envs/FireMan/bin/python scripts/stage/b1_bear_network.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import rasterio

from firescape.delineate import match_grid, network

BOX = Path("/Users/scottmccoy/Library/CloudStorage/Box-Box/SWMresearch"
           "/PostFireDebrisFlows/2024_BearFire")
OUT = BOX / "TraceScape_outputa"
PERIM = BOX / "GIS/shp/Bear_Fire_2024_Perimeter.shp"
BARC = BOX / ("GIS/CGS_BARC/bear_barc4_regbarc_90312570_s2a_20240902"
              "_20240905_cliptoperim.tif")

#: The fire spans -120.234..-120.157 -- just west of the zone 10/11 seam, so
#: zone 10 is native even though the CGS BARC arrived on a zone-11 S2 grid.
EPSG = 32610
#: >= 1 km was the ask. 3 km is the largest buffer that stays inside the
#: StormScape R* grid (-120.29..-120.10, 39.54..39.70), which is what gates
#: the dose-response, and it roughly decuples the AOI relative to the 13.5
#: km^2 burn -- plenty of unburned channel for the control arm.
BUFFER_M = 3000.0
DEM_PAD_M = 4000.0
RES_M = 10.0

#: Geometric filters only (firescape FilterDefaults minus burn/perimeter).
MIN_AREA_KM2 = 0.025
MAX_AREA_KM2 = 8.0
MAX_LENGTH_M = 500.0
MIN_SLOPE = 0.12            # gradient, ~6.8 deg
MAX_CONFINEMENT_DEG = 174.0


def main() -> None:
    from pfdf import severity as pfdf_severity
    from pfdf.raster import Raster

    OUT.mkdir(parents=True, exist_ok=True)
    interim = OUT / "interim"
    interim.mkdir(exist_ok=True)

    perim = gpd.read_file(PERIM).to_crs(EPSG)
    aoi_geom = perim.union_all().buffer(BUFFER_M)
    aoi = gpd.GeoDataFrame(geometry=[aoi_geom], crs=EPSG)
    b = aoi.total_bounds
    print(f"perimeter {perim.area.sum()/1e6:.1f} km2 -> AOI "
          f"{aoi_geom.area/1e6:.1f} km2 "
          f"({(b[2]-b[0])/1e3:.1f} x {(b[3]-b[1])/1e3:.1f} km)", flush=True)
    aoi4326 = aoi.to_crs(4326)
    print("AOI bounds4326:", np.round(aoi4326.total_bounds, 4), flush=True)

    # ---- DEM ------------------------------------------------------------
    dem_path = interim / "bear_dem_10m.tif"
    if not dem_path.exists():
        from stormscape.dem import get_dem
        fetch = gpd.GeoDataFrame(geometry=[aoi_geom.buffer(DEM_PAD_M)],
                                 crs=EPSG).to_crs(4326)
        da = get_dem(tuple(fetch.total_bounds), resolution=int(RES_M),
                     dst_crs=f"EPSG:{EPSG}", pad_deg=0.0)
        da.rio.to_raster(dem_path)
        print(f"DEM -> {dem_path} {da.shape}", flush=True)
    dem = Raster.from_file(dem_path)
    print(f"DEM {dem.shape} @ {dem.resolution(units='meters')} m "
          f"{dem.crs}", flush=True)

    # ---- domain mask: the whole AOI, burned or not ----------------------
    domain_path = interim / "_domain.geojson"
    aoi.to_file(domain_path, driver="GeoJSON")
    domain = match_grid(
        Raster.from_polygons(domain_path, bounds=dem, resolution=RES_M), dem)

    segments, terr = network(dem, domain, min_area_km2=MIN_AREA_KM2,
                             max_length_m=MAX_LENGTH_M)
    if segments is None:
        raise SystemExit("empty network")
    n0 = segments.size
    print(f"raw network: {n0} segments", flush=True)

    # ---- severity: BARC4 outside the fire is nodata == unburned ---------
    barc_src = Raster.from_file(BARC)
    barc = match_grid(barc_src, dem, resampling="nearest")
    arr = barc.values.astype("float64").copy()
    # 15 is the BARC nodata and the clip leaves everything outside the
    # perimeter as nodata -- which IS the unburned control, so it must
    # become class 1 rather than propagate as no-data through burn_ratio.
    arr[~np.isfinite(arr)] = 1
    arr[(arr < 1) | (arr > 4)] = 1
    barc4 = Raster.from_array(arr.astype("uint8"), spatial=dem, nodata=0)
    burned = pfdf_severity.mask(barc4, ["low", "moderate", "high"])
    modhigh = pfdf_severity.mask(barc4, ["moderate", "high"])

    perim_path = interim / "_perimeter.geojson"
    perim.to_file(perim_path, driver="GeoJSON")
    perim_r = match_grid(
        Raster.from_polygons(perim_path, bounds=dem, resolution=RES_M), dem)

    # ---- geometric filters ONLY -----------------------------------------
    area = segments.area(units="kilometers")
    slope = segments.slope(terr.slopes)
    conf = segments.confinement(terr.conditioned, 4)
    keep = ((area <= MAX_AREA_KM2) & (slope >= MIN_SLOPE)
            & (conf <= MAX_CONFINEMENT_DEG))
    segments.keep(segments.continuous(keep))
    print(f"after geometric filters: {segments.size} segments "
          f"(from {n0})", flush=True)

    # ---- attributes ------------------------------------------------------
    fc = segments.geojson(type="segments")   # a geojson.FeatureCollection
    geoms = gpd.GeoDataFrame.from_features(fc["features"], crs=segments.crs)

    df = gpd.GeoDataFrame(
        {
            "seg_uid": [f"BEAR-{i:05d}" for i in segments.ids],
            "Segment_ID": np.asarray(segments.ids),
            "Area_km2": segments.area(units="kilometers"),
            "Slope": segments.slope(terr.slopes),
            "Relief_m": segments.relief(terr.relief),
            "ConfAngle": segments.confinement(terr.conditioned, 4),
            "BurnRatio": segments.burn_ratio(burned),
            "BmhRatio": segments.catchment_ratio(modhigh),
            "Bmh_km2": segments.burned_area(modhigh, units="kilometers"),
            "InPerim": np.asarray(segments.in_perimeter(perim_r), dtype=float),
            "length_m": segments.length(units="meters"),
        },
        geometry=geoms.geometry.values, crs=segments.crs)
    df["SlopeDeg"] = np.degrees(np.arctan(df["Slope"]))
    #: The control split. A segment counts as burned when a quarter of its
    #: upslope area burned -- the same 0.25 the USGS filter uses to decide a
    #: channel is fire-affected, reused here as a label rather than a filter.
    df["burned"] = df["BurnRatio"] >= 0.25

    df.to_file(OUT / "bear_regional_segments.gpkg", layer="segments",
               driver="GPKG")
    aoi4326.to_file(OUT / "bear_aoi.gpkg", layer="aoi", driver="GPKG")

    print(f"\n{len(df)} segments, {df.length_m.sum()/1e3:.1f} km total",
          flush=True)
    print(f"burned (BurnRatio>=0.25): {int(df.burned.sum())} "
          f"({df.burned.mean():.0%});  unburned control: "
          f"{int((~df.burned).sum())}", flush=True)
    print(f"slope deg  median {df.SlopeDeg.median():.1f} "
          f"[{df.SlopeDeg.min():.1f}-{df.SlopeDeg.max():.1f}]", flush=True)
    print(f"area km2   median {df.Area_km2.median():.3f} "
          f"[{df.Area_km2.min():.3f}-{df.Area_km2.max():.2f}]", flush=True)
    print(f"-> {OUT / 'bear_regional_segments.gpkg'}", flush=True)


if __name__ == "__main__":
    main()
