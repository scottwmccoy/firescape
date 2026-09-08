# firescape

Pre-fire assessment of postfire debris-flow (PFDF) hazard for Nevada.

It recreates the California Geological Survey statewide pre-fire workflow
(Rossi et al. 2025, *Int. J. Wildland Fire* 34, WF24225) with Nevada-specific
calibration, and runs the same hazard chain on real fires as they burn:

1. **Simulated burn severity** — dNBR simulated per LANDFIRE Existing Vegetation
   Type class from the Staley et al. (2018) Weibull CDFs (doi:10.5066/P9TKYL5K),
   at a regionally calibrated percentile *P_dsim*, with the measured within-class
   dispersion.
2. **Regional calibration** — *P_dsim* and the BARC breaks are set so simulated
   debris-flow likelihood matches the likelihood computed from observed MTBS dNBR
   on 123 historical Nevada fires ("DFL calibration"), and validated against
   field-verified BAER soil burn severity.
3. **USGS hazard models** — likelihood (Staley et al. 2017 M1), volume (Gartner
   et al. 2014; RANGES), combined hazard (Cannon et al. 2010) and the triggering
   rainfall, through the USGS [pfdf](https://ghsc.code-pages.usgs.gov/lhp/pfdf)
   library.
4. **Annual probability** — P(R>T) from NOAA Atlas 14 rainfall climatology,
   combined with annual burn probability P(F).

Storm-side machinery (DEM, Atlas 14 grids, NHD, radar rainfall, cartography)
comes from [stormscape](https://github.com/scottwmccoy/stormscape); imagery
change detection that *validates* what firescape predicts lives in the sibling
`tracescape`. firescape owns the fire and hazard side.

Funded by the USGS Landslide Hazards Program (NBMG / UNR). Products publish to
the NBMG MyHAZARDS portal.

## Getting started

### 1. Install

A conda-forge Python 3.12 environment (the project's is called `FireMan`;
`CLAUDE.md` has the environment rules). With it activated, from checkouts of the
three sibling repos:

```bash
python -m pip install -e path/to/stormscape        # MIT — rainfall, DEM, cartography
python -m pip install -e path/to/tracescape        # MIT — used by firescape.exposure
python -m pip install -e path/to/firescape
# pfdf comes from the USGS registry, NOT PyPI (the PyPI "pfdf" is unrelated):
python -m pip install pfdf -i https://code.usgs.gov/api/v4/groups/859/-/packages/pypi/simple
```

`firescape` must be an *editable* install: the pipeline it launches lives in
`scripts/` beside the package, on purpose (it is provenance, not library code).

### 2. Configure

Nothing in the code depends on where your checkout or home directory is.
Everything machine-specific is an environment variable, and all have sensible
defaults except the email:

| variable | meaning | default |
|---|---|---|
| `FIRESCAPE_DATA` | the shared project data folder (`raw/`, `interim/`, `products/`, `figures/`) | the Box folder at Box Drive's standard location, `~/Library/CloudStorage/Box-Box/SWMresearch/PostFireDebrisFlows/PreFireAssessment` |
| `FIRESCAPE_RESEARCH` | the folder above it, holding sibling per-fire / per-storm project folders a few scripts read | parent of `FIRESCAPE_DATA` |
| `FIRESCAPE_CACHE` | local caches and download staging (never on Box) | `~/.cache/firescape` |
| `FIRESCAPE_PYTHON` | interpreter the CLI, shell wrappers and batch drivers use | the running interpreter / `python3` |
| `FIRESCAPE_EMAIL` | address LANDFIRE (LFPS) and MTBS notify when a job or bundle order is ready | **none — required** for staging and ordering |

The data folder is the group's Box share: sync it, or point `FIRESCAPE_DATA` at
wherever it lives. Files shipped inside the package (calibration TOMLs, fire
sets, region GeoJSONs, the Staley 2018 tables) are found through
`firescape.paths.package_data()` and need no configuration.

### 3. Check it

```bash
python -m pytest tests/        # offline, synthetic fixtures, ~5 s
firescape --help
```

## The worked example: a fire that is burning now

Every stage takes an incident name as WFIGS spells it and, optionally, a
calibration name (default: `statewide_v1_4`, `firescape.config.CURRENT_CALIBRATION`).
This is how the 2026 Stallion, Bug and Hawk fires were run.

```bash
firescape prefire Bug                # 1. forecast on SIMULATED severity, before any imagery
firescape storm   Bug                # 2. score that network against the storm that fell
firescape assess  Bug                # 3. re-run on OBSERVED severity (CIMSS BRISK dNBR, near-real-time)
firescape map     Bug --sheet all    # 4. the sheets: post-fire hazard, pre-fire hazard, storm response
```

What each step does and where it lands (paths relative to `FIRESCAPE_DATA`):

| step | what it needs | what it produces |
|---|---|---|
| `prefire` | the fire's current WFIGS perimeter (fetched), 3DEP tiles, LANDFIRE EVT, soils and Atlas 14 from the staged data tree | `products/forecast/<fire>_<calibration>/` — segment network with likelihood, volume, hazard class and triggering I15 at the 24 mm/h design storm |
| `storm` | a stormscape storm composite (`STORM` in `scripts/products/p8_fire_storm_response.py`, currently the 12–14 Aug 2026 event) | per-segment response under the rainfall that actually fell, written beside the forecast |
| `assess` | nothing extra — pulls BRISK dNBR through `stormscape.burn` and injects it, so no MTBS bundle and no waiting | `products/forecast/<fire>_observed/` — the same chain on measured severity |
| `map` | the products above | `figures/<fire>_postfire_hazard.pdf` (the emergency-assessment sheet in the form the USGS publishes), `<fire>_prefire_hazard.pdf`, `<fire>_storm_response.pdf` |

Add `--dry-run` to any command to see the script it runs. Two things the
example glosses over: `prefire` for a fire outside the staged statewide tiles
needs the DEM/EVT staging in `scripts/stage/` first, and the BRISK severity a
fresh burn returns is a *snapshot* that matures for weeks — re-run `assess`
after it settles.

Historic fires are hindcast the same way from MTBS severity, against the
Nevada debris-flow inventory:

```bash
firescape hindcast NV3930511982820240907     # one MTBS event id
firescape hindcast --all                     # every inventory fire with a flow after it
```

## The statewide product

The pre-fire surface for all of Nevada is the same chain run over 591 HU10
units on simulated severity. It is produced by the resumable drivers in
`scripts/` rather than by the CLI, because a full pass is hours of compute in
parallel lanes:

```
stage/     3DEP tiles, LANDFIRE EVT, soils, Atlas 14, the MTBS calibration fire set
calibrate/ per-region P_dsim + breaks   →  firescape/data/calibration/statewide_v1_4.toml
surface/   sw_fleet_v12.sh → sw_driver_v12.py (lanes) → sw_merge_v12.py → sw_maps_v12.py
           p7_annual_probability.py     →  P(F) × P(R>T)
```

The shipped surface is `products/prefire/statewide_v1_2/` (412,501 basins).
[`scripts/README.md`](scripts/README.md) documents every driver, the regional
zoom sheets, the exposure products and the conventions (time budgets, exit 42,
caches).

### Calibration versions

Calibration values are TOML **files** in `firescape/data/calibration/`, never
constants in code; each file's header says what changed and what it
supersedes. The current one is `statewide_v1_4`.

| TOML | what it is |
|---|---|
| `statewide_v1` | first statewide per-region calibration, 123 era-matched MTBS fires (2026-08-13) |
| `statewide_v1_1` | dispersed severity (measured within-class dispersion σ_z = 0.91) |
| `statewide_v1_2` | the Nevada statewide EVT–dNBR CDF refit (`nv_statewide` table) |
| `statewide_v1_3` | MTBS `mod_t = 9999` sentinel kept out of the regional break medians |
| `statewide_v1_4` | BARC-programme thresholds converted from GTAC's BARC256 scale; fires without a `dnbr6` raster observed at the analyst's own thresholds (Sierra Nevada 312 → 350) |

Regional values: `firescape.config.packaged_calibration("statewide_v1_4").region("sierra_nevada")`.

## What is where

`firescape/` is the library; `scripts/` is the as-run pipeline that produced
every product; `tests/` is offline; data is on Box.

| module | owns |
|---|---|
| `severity` | Staley 2018 EVT–dNBR Weibull CDFs, simulation at *P_dsim*, dispersion, BARC classification, the Nevada refit |
| `calibrate` | the Rossi DFL calibration: per-fire class decomposition (cached), *P_dsim* solve, regional medians, TOML writer |
| `hazard` | the USGS models through pfdf — M1 likelihood, Gartner 14 / RANGES volume, Cannon 10 class, threshold inversion; all unit conversions |
| `delineate` | stream-segment network delineation (pfdf/pysheds), Rossi's valley / sink / water masks |
| `prefire` | the pre-fire surface for one unit (`run_unit`), the seam the statewide fleet tiles over |
| `assess` | the hazard chain on one fire with observed severity (MTBS bundle or injected dNBR) |
| `annualprob` | P(R>T) from Atlas 14 and the P(F) combination |
| `exposure` | which mapped assets a debris flow could reach (AML sites, receptors, BLM flag) |
| `mtbs`, `baer`, `landfire`, `soils`, `fires`, `wbd` | data access: MTBS records and bundles, BAER soil burn severity, LANDFIRE EVT, SSURGO/STATSGO KF, WFIGS perimeters, hydrologic units |
| `statewide` | per-unit windowed reads from the local tile stores |
| `regions`, `config`, `paths`, `products`, `plotting` | modeling regions, calibration TOMLs and model defaults, path policy, product writing with `run_meta.json`, the map house style |

## License

GPL-3.0-only (required: firescape links against the GPL-3 licensed USGS pfdf
library). Data products (rasters, GPKGs, maps, calibration TOMLs) are data, not
code, and carry their own public-domain / CC attributions — see NOTICE.md.
