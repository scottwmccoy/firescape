"""Product writing: rasters, hazard polygon layers, run metadata (M1/M4).

Will own: dtype/nodata-aware raster writes named <key>_<field>.tif (mirroring
stormscape's save_fields contract), the wildcat-schema segment/basin GPKG
(Segment_ID, Area_km2, BurnRatio, Slope, Terrain_M1, Fire_M1, Soil_M1,
Bmh_km2, Relief_m, P_24mmh, V_24mmh, H_24mmh, I15_50, ...), run_meta.json
(git sha, calibration TOML hash, input versions), and map/export glue via
stormscape.plot.drape_i15 / stormscape.export. MyHAZARDS export schema is a
deferred seam (products.to_myhazards).
"""

from __future__ import annotations


def run_meta(out_dir, **info):
    """Write run_meta.json capturing code+calibration+input provenance."""
    raise NotImplementedError("products.run_meta lands in milestone M1")
