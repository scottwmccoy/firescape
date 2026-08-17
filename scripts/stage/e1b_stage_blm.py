"""Stage BLM surface-management polygons for Nevada -> raw/blm/nv_blm_sma.gpkg.

One agency from the BLM SMA fabric (the layer the regional zoom sheets
already draw), fetched tile-by-tile with a recursive split wherever the
cached service hits its transfer limit, geometry simplified server-side to
~30 m. Used for the per-site "on BLM-managed land" flag and the statewide
map wash — fine for management screening, not for parcel-line work
(checkerboarded patented claims are ubiquitous in the districts).
"""
import json
import sys
import tempfile
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import geopandas as gpd

from firescape import paths

SMA = ("https://gis.blm.gov/arcgis/rest/services/lands/"
       "BLM_Natl_SMA_Cached_without_PriUnk/MapServer/1/query")
DEST = paths.raw_dir("blm") / "nv_blm_sma.gpkg"

if DEST.exists():
    print(f"already staged (raw/ is immutable): {DEST}")
    sys.exit(0)

nv = gpd.read_file(paths.raw_dir("boundaries") / "nv_state.geojson")
W, S, E, N = nv.total_bounds


def fetch(bounds, depth=0):
    q = urllib.parse.urlencode({
        "geometry": ",".join(f"{b:.6f}" for b in bounds),
        "geometryType": "esriGeometryEnvelope", "inSR": 4326, "outSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "where": "ADMIN_AGENCY_CODE='BLM'",
        "outFields": "OBJECTID,ADMIN_AGENCY_CODE",
        "maxAllowableOffset": 0.0003, "returnGeometry": "true", "f": "geojson"})
    with urllib.request.urlopen(f"{SMA}?{q}", timeout=180) as r:
        payload = json.load(r)
    if payload.get("properties", {}).get("exceededTransferLimit") or \
            payload.get("exceededTransferLimit"):
        if depth > 6:
            raise RuntimeError(f"transfer limit at depth {depth}: {bounds}")
        w, s, e, n = bounds
        mx, my = (w + e) / 2, (s + n) / 2
        feats = []
        for bb in ((w, s, mx, my), (mx, s, e, my),
                   (w, my, mx, n), (mx, my, e, n)):
            feats += fetch(bb, depth + 1)
        return feats
    return payload.get("features", [])


tiles = [(x, y, min(x + 1, E + .05), min(y + 1, N + .05))
         for x in [W - .05 + i for i in range(int(E - W) + 2)]
         for y in [S - .05 + j for j in range(int(N - S) + 2)]]
feats, seen = [], set()
for i, t in enumerate(tiles):
    for f in fetch(t):
        oid = f["properties"].get("OBJECTID")
        if oid not in seen:
            seen.add(oid)
            feats.append(f)
    print(f"  tile {i + 1}/{len(tiles)}: {len(feats)} polygons", flush=True)

gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
print(f"{len(gdf)} BLM polygons")
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td) / DEST.name
    gdf.to_file(tmp, layer="blm", driver="GPKG")
    paths.atomic_into(tmp, DEST)
paths.write_provenance(
    DEST, url=SMA.rsplit("/query", 1)[0],
    note="BLM Natl SMA (without Private/Unknown), ADMIN_AGENCY_CODE='BLM', "
         "NV bbox, ~30 m server-side simplification. Management screening "
         "fidelity, not parcel-line fidelity.",
    n_features=int(len(gdf)))
print(f"staged {DEST}")
