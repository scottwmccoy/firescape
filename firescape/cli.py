"""firescape command-line interface.

``dem`` and ``fires`` are implemented here. Everything else is a thin launcher
for the as-run pipeline in ``scripts/`` beside the package -- each subcommand
runs one script with the arguments that script expects (fire name,
calibration), so the README's worked example is four commands rather than
four paths. ``--dry-run`` prints the command instead of running it.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

from firescape import paths
from firescape.config import CURRENT_CALIBRATION
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


def scripts_dir() -> Path:
    """The as-run pipeline: ``scripts/`` beside the package in a checkout.

    Deliberately not part of the installed distribution (it is provenance,
    not library code), so a non-editable install has nothing to launch.
    """
    d = Path(__file__).resolve().parents[1] / "scripts"
    if not d.is_dir():
        raise SystemExit("firescape's pipeline scripts are not installed: run from an "
                         "editable checkout (pip install -e path/to/firescape).")
    return d


def _launch(rel_and_args: list[str], dry_run: bool = False) -> int:
    """Run ``scripts/<rel>`` with the interpreter from ``paths.python_executable``."""
    import subprocess

    script = scripts_dir() / rel_and_args[0]
    if not script.is_file():
        raise SystemExit(f"no such pipeline script: {script}")
    cmd = [paths.python_executable(), str(script), *rel_and_args[1:]]
    print("$ " + " ".join(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.call(cmd)


def _cmd_map(args) -> int:
    sheets = {"postfire": ["figures/p15_fire_postfire_hazard.py", args.fire, args.calibration],
              "prefire": ["figures/p15_fire_postfire_hazard.py", args.fire, args.calibration, "--prefire"],
              "storm": ["figures/p8_fire_storm_map.py", args.fire, args.calibration],
              "forecast": ["figures/p8_fire_forecast_map.py", args.fire, args.calibration]}
    order = ["postfire", "prefire", "storm"] if args.sheet == "all" else [args.sheet]
    rc = 0
    for k in order:
        rc = _launch(sheets[k], args.dry_run) or rc
    return rc


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

    # ---- the as-run pipeline, launched by name -----------------------------
    def fire_parser(name, help_, script_note):
        p = sub.add_parser(name, help=help_,
                           description=f"{help_}. Runs scripts/{script_note}.")
        p.add_argument("fire", help="incident name as WFIGS spells it, e.g. Bug")
        p.add_argument("calibration", nargs="?", default=CURRENT_CALIBRATION,
                       help=f"calibration TOML name (default {CURRENT_CALIBRATION})")
        p.add_argument("--dry-run", action="store_true", help="print the command, do not run it")
        return p

    fire_parser("prefire", "pre-fire forecast for one fire on simulated severity",
                "products/p8_fire_forecast.py").set_defaults(
        func=lambda a: _launch(["products/p8_fire_forecast.py", a.fire, a.calibration], a.dry_run))
    fire_parser("storm", "score a fire's forecast network against the storm that fell",
                "products/p8_fire_storm_response.py").set_defaults(
        func=lambda a: _launch(["products/p8_fire_storm_response.py", a.fire, a.calibration], a.dry_run))
    fire_parser("assess", "hazard chain on one fire with OBSERVED severity (CIMSS BRISK)",
                "products/p9_fire_observed.py").set_defaults(
        func=lambda a: _launch(["products/p9_fire_observed.py", a.fire, a.calibration], a.dry_run))

    p = fire_parser("map", "the fire sheets from existing products", "figures/p15_*, p8_*")
    p.add_argument("--sheet", choices=["postfire", "prefire", "storm", "forecast", "all"],
                   default="postfire",
                   help="postfire = observed-severity hazard sheet (default); prefire = the same "
                        "sheet on simulated severity; storm = storm-response map; forecast = the "
                        "older forecast sheet; all = postfire + prefire + storm")
    p.set_defaults(func=_cmd_map)

    p = sub.add_parser("hindcast", help="hazard hindcast on MTBS severity for a historic fire",
                       description="Runs scripts/products/p16_mtbs_fire_hindcast.py.")
    p.add_argument("event_id", nargs="?", help="MTBS event id, e.g. NV3930511982820240907")
    p.add_argument("calibration", nargs="?", default=CURRENT_CALIBRATION)
    p.add_argument("--all", action="store_true", help="every inventory fire with a flow after it")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=lambda a: _launch(
        ["products/p16_mtbs_fire_hindcast.py"] + ([a.event_id] if a.event_id else []) + [a.calibration]
        + (["--all"] if a.all else []), a.dry_run))

    p = sub.add_parser("severity", help="statewide simulated-severity intermediates, mapped",
                       description="Runs scripts/analysis/p2_severity_mosaic.py.")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=lambda a: _launch(["analysis/p2_severity_mosaic.py"], a.dry_run))

    p = sub.add_parser("calibrate", help="per-region P_dsim + break calibration on the MTBS fire set",
                       description="Runs scripts/calibrate/p8_calib_driver.py (resumable; exit 42 = "
                                   "budget hit, run again; FIRESCAPE_CAL_BUDGET seconds); "
                                   "--summary assembles the TOML from the per-fire caches.")
    p.add_argument("--summary", action="store_true", help="assemble the TOML from the per-fire caches")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=lambda a: _launch(
        ["calibrate/p8_calib_summary.py" if a.summary else "calibrate/p8_calib_driver.py"], a.dry_run))

    p = sub.add_parser("annualprob", help="P(F) x P(R>T): annual debris-flow probability for a surface",
                       description="Runs scripts/surface/p7_annual_probability.py.")
    p.add_argument("version", nargs="?", default="statewide_v1_2",
                   help="merged surface version under products/prefire/ (default statewide_v1_2)")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=lambda a: _launch(["surface/p7_annual_probability.py", a.version], a.dry_run))

    args = ap.parse_args(argv)
    return args.func(args)
