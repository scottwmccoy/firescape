"""PlanetScope imagery access, for mapping what a storm actually did.

The rest of firescape predicts debris flows; this module is the start of the
machinery that checks the predictions -- finding the tracks and fan deposits a
flow leaves behind in 3 m PlanetScope imagery over a burn (or rangeland) AOI.
Detection/change code will land separately once the method study settles; this
layer owns credentials, scene search, and order construction.

Endpoints validated 2026-08-15 (see also ``quota()``):

- ``GET  /data/v1/``            -- auth check (200 with a valid key).
- ``POST /data/v1/quick-search``-- paginated scene search. Free: searching
  costs no quota, only downloading assets does.
- ``GET  /auth/v1/experimental/public/my/subscriptions`` -- plan + quota.
  This account rides the NASA CSDA IDIQ pool (~5e9 km^2), so quota is not a
  practical constraint -- but orders still take processing time, so search
  first, order deliberately.

Cadence reality check (Hidden Valley window, 6 weeks of summer 2026):
101 PSScene items, 70 with ``clear_percent >= 90`` -- roughly one usable
scene per clear day. Multi-scene stacking per epoch is therefore cheap, which
matters because single-scene PlanetScope radiometry is noisy across the Dove
fleet; every epoch should be a composite, never one scene.

API key resolution (in order): ``key=`` argument, ``$PL_API_KEY``, then
``~/.config/firescape/planet_api_key`` (single line, chmod 600). The key is
never written into this repo, Box, or figure metadata.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from firescape import paths

API = "https://api.planet.com"
QUICK_SEARCH = f"{API}/data/v1/quick-search"
ORDERS = f"{API}/compute/ops/orders/v2"

#: On-disk fallback for the API key (0600; created 2026-08-15).
KEY_FILE = Path.home() / ".config" / "firescape" / "planet_api_key"

#: Verification AOIs, lon/lat (W, S, E, N). ``hidden_valley`` is the
#: Virginia Range front above the Hidden Valley neighbourhood of east Reno --
#: known (unburned) debris-flow tracks close to home, our detector test bed.
#: Bounds are provisional until walked over imagery.
AOIS = {
    "hidden_valley": (-119.72, 39.45, -119.63, 39.53),
}

#: Orders-API product bundles that pair surface reflectance with the udm2
#: usable-data mask. 8-band needs PSB.SD (SuperDove, ~2021+) scenes.
BUNDLES = {"4b": "analytic_sr_udm2", "8b": "analytic_8b_sr_udm2"}


class PlanetError(Exception):
    """Planet API returned an error (bad key, bad filter, order failure...)."""


# --------------------------------------------------------------------------- #
# auth
# --------------------------------------------------------------------------- #
def _key(key: str | None = None) -> str:
    tok = key or os.environ.get("PL_API_KEY")
    if not tok and KEY_FILE.exists():
        tok = KEY_FILE.read_text().strip()
    if not tok:
        raise PlanetError(
            "No Planet API key. Pass key=..., set $PL_API_KEY, or store the "
            f"key (one line, chmod 600) at {KEY_FILE}.")
    return tok


def _auth(key: str | None = None) -> tuple[str, str]:
    """HTTP basic auth pair (Planet convention: key as user, empty password)."""
    return (_key(key), "")


def _raise_for_planet(r: requests.Response) -> None:
    """Surface Planet HTTP errors with their human-readable message."""
    if r.ok:
        return
    detail = ""
    try:
        j = r.json()
        parts = []
        for v in (j.get("general") or []):
            parts.append(v.get("message", ""))
        for field, msgs in (j.get("field") or {}).items():
            for m in msgs:
                parts.append(f"{field}: {m.get('message', m)}")
        detail = "; ".join(p for p in parts if p) or j.get("message", "")
    except Exception:
        detail = r.text[:300]
    raise PlanetError(f"HTTP {r.status_code} from {r.url}: {detail}")


def quota(key: str | None = None) -> list[dict]:
    """Plan/quota summaries for the account (name, state, used, total km^2).

    Uses an endpoint Planet labels experimental; if it disappears, this
    returns an empty list rather than failing anything that matters.
    """
    r = requests.get(f"{API}/auth/v1/experimental/public/my/subscriptions",
                     auth=_auth(key), timeout=30)
    if not r.ok:
        return []
    return [{"plan": s.get("plan", {}).get("name"), "state": s.get("state"),
             "quota_used_sqkm": s.get("quota_used"),
             "quota_sqkm": s.get("quota_sqkm")} for s in r.json()]


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #
def _rfc3339(t) -> str:
    """Coerce a datetime or ISO string ('2026-07-01' ok) to RFC-3339 UTC."""
    if isinstance(t, str):
        t = datetime.fromisoformat(t)
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _aoi_geom(spec) -> dict:
    """AOI spec -> GeoJSON geometry dict (lon/lat).

    Accepts an ``AOIS`` key, plus everything ``stormscape.aoi.load_aoi``
    takes: a (W, S, E, N) tuple, a vector file path, or a shapely geometry.
    """
    from shapely.geometry import box, mapping

    if isinstance(spec, str) and spec in AOIS:
        spec = AOIS[spec]
    from stormscape import aoi as _aoi
    bounds, geom = _aoi.load_aoi(spec)
    return mapping(geom if geom is not None else box(*bounds))


def _search_filter(geom: dict, start, end, *, max_cloud: float | None = None,
                   min_clear: float | None = None) -> dict:
    """Compose the Data-API filter tree for one AOI and date window.

    ``min_clear`` filters on udm2-derived ``clear_percent`` (per-pixel
    usability; the better field); ``max_cloud`` on the older scene-level
    ``cloud_cover`` fraction. Use one, not both, unless you mean it.
    """
    config = [
        {"type": "GeometryFilter", "field_name": "geometry", "config": geom},
        {"type": "DateRangeFilter", "field_name": "acquired",
         "config": {"gte": _rfc3339(start), "lte": _rfc3339(end)}},
    ]
    if max_cloud is not None:
        config.append({"type": "RangeFilter", "field_name": "cloud_cover",
                       "config": {"lte": float(max_cloud)}})
    if min_clear is not None:
        config.append({"type": "RangeFilter", "field_name": "clear_percent",
                       "config": {"gte": float(min_clear)}})
    return {"type": "AndFilter", "config": config}


#: Scene properties worth carrying into the search result frame.
_KEEP = ["acquired", "instrument", "satellite_id", "clear_percent",
         "cloud_cover", "quality_category", "publishing_stage",
         "ground_control", "sun_elevation", "view_angle"]


def search(aoi, start, end, *, item_type: str = "PSScene",
           max_cloud: float | None = None, min_clear: float | None = None,
           extra_filters: list[dict] | None = None, key: str | None = None):
    """Scenes intersecting an AOI in a date window, newest last.

    Returns a GeoDataFrame (EPSG:4326) of footprints with ``id`` plus the
    ``_KEEP`` properties. Searching is free -- be liberal here and strict at
    order time. ``extra_filters`` are appended verbatim to the AndFilter
    (e.g. a PermissionFilter for downloadability, StringInFilter on
    quality_category).
    """
    import geopandas as gpd
    from shapely.geometry import shape

    geom = _aoi_geom(aoi)
    flt = _search_filter(geom, start, end, max_cloud=max_cloud,
                         min_clear=min_clear)
    if extra_filters:
        flt["config"].extend(extra_filters)
    payload = {"item_types": [item_type], "filter": flt}
    auth = _auth(key)
    rows = []
    r = requests.post(QUICK_SEARCH, params={"_page_size": 250}, auth=auth,
                      json=payload, timeout=120)
    _raise_for_planet(r)
    while True:
        j = r.json()
        for f in j.get("features", []):
            p = f.get("properties", {})
            rows.append({"id": f["id"], "item_type": item_type,
                         **{k: p.get(k) for k in _KEEP},
                         "geometry": shape(f["geometry"])})
        nxt = (j.get("_links") or {}).get("_next")
        if not nxt:
            break
        r = requests.get(nxt, auth=auth, timeout=120)
        _raise_for_planet(r)
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326") \
        if rows else gpd.GeoDataFrame({"id": []}, geometry=[], crs="EPSG:4326")
    if len(gdf):
        gdf = gdf.sort_values("acquired").reset_index(drop=True)
    return gdf


# --------------------------------------------------------------------------- #
# screening (free -- scene clear_percent is SCENE-wide, not AOI-wide)
# --------------------------------------------------------------------------- #
def aoi_coverage(item_id: str, aoi, *, mode: str = "estimate",
                 item_type: str = "PSScene", key: str | None = None) -> dict:
    """AOI-specific usable-coverage estimate for one scene. Costs no quota.

    ``mode="estimate"`` is synchronous (browse-image based); ``mode="udm2"``
    is the accurate one but activates the udm2 server-side (poll: HTTP 202
    means ask again in ~10 s). The request body is the bare geometry.
    """
    url = f"{API}/data/v1/item-types/{item_type}/items/{item_id}/coverage"
    r = requests.post(url, params={"mode": mode}, auth=_auth(key),
                      json={"geometry": _aoi_geom(aoi)}, timeout=60)
    _raise_for_planet(r)
    return r.json()


# --------------------------------------------------------------------------- #
# orders
# --------------------------------------------------------------------------- #
def order_request(name: str, item_ids: list[str], aoi, *,
                  bands: str = "4b", harmonize: bool = True,
                  item_type: str = "PSScene") -> dict:
    """Orders-API payload: SR + udm2 bundle, clipped, harmonized, COG.

    ``harmonize=True`` adds Planet's harmonization tool (target Sentinel-2),
    which shrinks the cross-Dove radiometric scatter that forced us into
    multi-scene averaging in the first place -- leave it on for change
    detection (PS2.SD/PSB.SD scenes only; do not mix PS2-era items in).
    Clipping keeps deliveries small and quota honest (clip AOI: <=1500
    vertices, no holes). ``file_format: COG`` makes Planet deliver tiled
    LZW Cloud-Optimized GeoTIFFs with overviews -- viewer-ready, no local
    conversion. Tool order here is cosmetic; the server applies its fixed
    sequence (harmonize -> clip -> ... -> file_format) regardless.
    """
    if bands not in BUNDLES:
        raise PlanetError(f"bands must be one of {sorted(BUNDLES)}")
    if not item_ids:
        raise PlanetError("order_request needs at least one item id")
    tools: list[dict] = [{"clip": {"aoi": _aoi_geom(aoi)}}]
    if harmonize:
        tools.append({"harmonize": {"target_sensor": "Sentinel-2"}})
    tools.append({"file_format": {"format": "COG"}})
    return {
        "name": name,
        "products": [{"item_ids": list(item_ids), "item_type": item_type,
                      "product_bundle": BUNDLES[bands]}],
        "tools": tools,
    }


def submit_order(request: dict, *, key: str | None = None) -> dict:
    """POST an order; returns the order JSON (``id``, ``state``...).

    This is the one call in the module that consumes quota. Search and
    screen first; orders are charged per intersecting scene.
    """
    r = requests.post(ORDERS, auth=_auth(key), json=request, timeout=120)
    _raise_for_planet(r)
    return r.json()


def order_state(order_id: str, *, key: str | None = None) -> dict:
    r = requests.get(f"{ORDERS}/{order_id}", auth=_auth(key), timeout=60)
    _raise_for_planet(r)
    return r.json()


def wait_order(order_id: str, *, poll: float = 60.0, timeout: float = 5400.0,
               key: str | None = None) -> dict:
    """Poll until the order reaches a terminal state.

    Returns the final order JSON on ``success`` or ``partial`` (caller decides
    whether partial is acceptable); raises on ``failed``/``cancelled`` or on
    timeout. Poll the ORDER, never a log file -- lesson learned.
    """
    t0 = time.monotonic()
    while True:
        o = order_state(order_id, key=key)
        state = o.get("state")
        if state in ("success", "partial"):
            return o
        if state in ("failed", "cancelled"):
            raise PlanetError(f"order {order_id} {state}: "
                              f"{o.get('last_message', '')}")
        if time.monotonic() - t0 > timeout:
            raise PlanetError(f"order {order_id} still '{state}' after "
                              f"{timeout:.0f}s")
        time.sleep(poll)


def download_order(order: dict | str, dest_dir: Path, *,
                   key: str | None = None) -> list[Path]:
    """Download every delivered file and verify it against ``manifest.json``.

    File names in the delivery are prefixed ``<order-id>/``; that prefix is
    stripped so paths on disk match the manifest's relative ``path`` entries.
    Expired result URLs (orders sign them with ``expires_at``) get one
    refresh via a re-fetch of the order. Raises on any checksum mismatch or
    manifest entry with no file.
    """
    import hashlib
    import json as _json

    if isinstance(order, str):
        order = order_state(order, key=key)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    auth = _auth(key)

    def _fetch(results):
        out = []
        for res in results:
            rel = res["name"].split("/", 1)[1] if "/" in res["name"] \
                else res["name"]
            target = dest_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            r = requests.get(res["location"], auth=auth, stream=True,
                             timeout=600)
            if r.status_code in (401, 403):
                return None                     # URLs expired -- refresh
            _raise_for_planet(r)
            part = target.with_suffix(target.suffix + ".part")
            with open(part, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
            part.rename(target)
            out.append(target)
        return out

    got = _fetch(order["_links"]["results"])
    if got is None:
        order = order_state(order["id"], key=key)
        got = _fetch(order["_links"]["results"])
        if got is None:
            raise PlanetError(f"order {order['id']}: result URLs expired "
                              "twice; giving up")

    manifest = next((p for p in got if p.name == "manifest.json"), None)
    if manifest is None:
        raise PlanetError("delivery contained no manifest.json")
    listed = _json.loads(manifest.read_text())["files"]
    for entry in listed:
        f = dest_dir / entry["path"]
        if not f.exists():
            raise PlanetError(f"manifest lists {entry['path']} but it was "
                              "not delivered")
        want = entry["digests"]["sha256"]
        have = hashlib.sha256(f.read_bytes()).hexdigest()
        if have != want:
            raise PlanetError(f"sha256 mismatch on {entry['path']}")
    return got


def stage_order(files: list[Path], aoi_name: str, epoch: str, *,
                order_id: str, request: dict | None = None) -> Path:
    """Atomic-move a verified delivery into Box ``raw/planet/<aoi>/<epoch>/``.

    Every raster gets the house ``.provenance.json`` sidecar. Orders vanish
    from Planet's listings after ~3 months, so the sidecar (order id, bundle,
    tool chain, item ids) is the durable record of what was ordered.
    """
    dest = paths.raw_dir("planet", aoi_name, epoch)
    prods = (request or {}).get("products", [{}])
    meta = {
        "order_id": order_id,
        "bundle": prods[0].get("product_bundle"),
        "item_ids": prods[0].get("item_ids"),
        "tools": [list(t)[0] for t in (request or {}).get("tools", [])],
    }
    moved = []
    for f in files:
        out = paths.atomic_into(f, dest / f.name)
        moved.append(out)
        if out.suffix.lower() in (".tif", ".tiff"):
            paths.write_provenance(
                out, url=f"{ORDERS}/{order_id}",
                note=f"Planet Orders API delivery for AOI '{aoi_name}', "
                     f"epoch '{epoch}'", **meta)
    return dest
