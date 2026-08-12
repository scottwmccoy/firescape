"""firescape command-line interface.

Implemented now: ``dem``, ``fires``. The rest are registered stubs that name
their milestone (see the approved plan) so the CLI surface is stable from M0.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

from firescape import paths
from firescape.regions import PILOT_BBOX


def _add_aoi(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("area of interest")
    g.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"),
                   help=f"lon/lat bounds (default: pilot {PILOT_BBOX})")
    g.add_argument("--aoi", type=Path, help="vector file (geojson/shp/gpkg/kmz)")
    g.add_argument("--pad-deg", type=float, default=0.05,
                   help="padding applied to the AOI bounds (deg, default 0.05)")
    g.add_argument("--key", default="pilot",
                   help="output name prefix, <key>_<field>.tif (default: pilot)")


def _resolve_aoi(args):
    from stormscape import aoi as saoi

    spec = args.aoi if args.aoi else tuple(args.bbox) if args.bbox else PILOT_BBOX
    return saoi.load_aoi(spec, pad_deg=args.pad_deg)


def _redirect_hyriver_cache() -> None:
    # py3dep/HyRiver writes an aiohttp sqlite cache into CWD by default;
    # keep it local and out of Box/the repo (stormscape's grew to ~5 GB).
    cache = paths.cache_root() / "hyriver"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HYRIVER_CACHE_NAME", str(cache / "aiohttp_cache.sqlite"))


def cmd_dem(args) -> int:
    _redirect_hyriver_cache()
    from stormscape import dem as sdem

    bounds, geom = _resolve_aoi(args)
    out = paths.interim_dir(args.key)
    dem_path = out / f"{args.key}_dem.tif"
    hs_path = out / f"{args.key}_hillshade.tif"
    sdem.fetch_dem_and_hillshade(
        args.aoi if args.aoi else (tuple(args.bbox) if args.bbox else PILOT_BBOX),
        resolution=args.resolution,
        dem_path=str(dem_path),
        hillshade_path=str(hs_path),
    )
    aoi_path = out / f"{args.key}_aoi.geojson"
    _save_aoi(bounds, geom, aoi_path)
    print(f"wrote {dem_path}\nwrote {hs_path}\nwrote {aoi_path}")
    return 0


def _save_aoi(bounds, geom, dest: Path) -> None:
    import geopandas as gpd
    from shapely.geometry import box

    g = geom if geom is not None else box(*bounds)
    gpd.GeoDataFrame({"kind": ["aoi"]}, geometry=[g], crs="EPSG:4326").to_file(
        dest, driver="GeoJSON"
    )


def cmd_fires(args) -> int:
    from firescape import fires

    gdf = fires.perimeters(
        args.names or None,
        bbox=tuple(args.bbox) if args.bbox else (None if args.names else PILOT_BBOX),
    )
    if not len(gdf):
        print("no perimeters matched", file=sys.stderr)
        return 1
    cols = ["attr_IncidentName", "attr_IrwinID", "poly_GISAcres",
            "attr_PercentContained", "poly_DateCurrent"]
    have = [c for c in cols if c in gdf.columns]
    print(gdf[have].to_string(index=False))
    if not args.no_save:
        dest = paths.raw_dir("perimeters") / f"wfigs_current_{date.today():%Y%m%d}.geojson"
        fires.save_perimeters(gdf, dest, note=" ".join(args.names or ["bbox query"]))
        print(f"wrote {dest}")
    return 0


def _stub(milestone: str):
    def run(_args) -> int:
        print(f"not implemented yet - lands in milestone {milestone} of the plan",
              file=sys.stderr)
        return 2

    return run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="firescape",
                                 description="Pre-fire PFDF hazard assessment for Nevada")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("dem", help="fetch 10 m 3DEP DEM + hillshade for the AOI")
    _add_aoi(p)
    p.add_argument("--resolution", type=int, default=10)
    p.set_defaults(func=cmd_dem)

    p = sub.add_parser("fires", help="fetch current WFIGS fire perimeters")
    _add_aoi(p)
    p.add_argument("--names", nargs="*", help="incident names (prefix match)")
    p.add_argument("--no-save", action="store_true")
    p.set_defaults(func=cmd_fires)

    for name, hlp, ms in [
        ("assess", "hazard chain on one fire with observed severity", "M1"),
        ("severity", "simulated dNBR/BARC surfaces for the AOI", "M2"),
        ("calibrate", "Nevada P_dsim calibration on MTBS fires", "M3"),
        ("prefire", "pre-fire hazard surface for the AOI", "M4"),
        ("annualprob", "P(R>T) annual probability layer", "M5"),
        ("map", "production maps from existing rasters", "M4"),
    ]:
        p = sub.add_parser(name, help=f"[{ms}] {hlp}")
        _add_aoi(p)
        p.set_defaults(func=_stub(ms))

    args = ap.parse_args(argv)
    return args.func(args)
