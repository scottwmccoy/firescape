# firescape

Pre-fire assessment of postfire debris-flow (PFDF) hazard for Nevada.

Recreates the California Geological Survey statewide pre-fire workflow (Rossi et
al. 2025, *Int. J. Wildland Fire* 34, WF24225) with Nevada-specific calibration:

1. **Simulated burn severity** — dNBR simulated per LANDFIRE Existing Vegetation
   Type class from the Staley et al. (2018) Weibull CDFs
   (doi:10.5066/P9TKYL5K), at a regionally calibrated percentile (P_dsim).
2. **Regional calibration** — P_dsim and BARC breaks calibrated so simulated
   debris-flow likelihood matches likelihood computed from observed MTBS dNBR
   on historical Nevada fires ("DFL calibration").
3. **USGS hazard models** — likelihood (Staley et al. 2017 M1), volume (Gartner
   et al. 2014), combined hazard (Cannon et al. 2010), and rainfall thresholds,
   via the USGS [pfdf](https://ghsc.code-pages.usgs.gov/lhp/pfdf) library.
4. **Annual probability** — P(R>T) from NOAA Atlas 14 rainfall climatology,
   later combined with annual burn probability P(F).

Storm-side machinery (DEM, Atlas 14 grids, NHD streams, cartography) comes from
[stormscape](../stormscape) — firescape owns only the fire/hazard side.

Funded by the USGS Landslide Hazards Program (NBMG/UNR). Products publish to the
NBMG MyHAZARDS portal.

## Install (development)

Use a conda-forge Python 3.12 environment (the project's is called
`FireMan`; see `CLAUDE.md` for the environment rules). With it activated:

```bash
python -m pip install -e path/to/stormscape        # sibling checkout, MIT
python -m pip install -e path/to/firescape
# pfdf comes from the USGS registry, NOT PyPI:
python -m pip install pfdf -i https://code.usgs.gov/api/v4/groups/859/-/packages/pypi/simple
```

## Configuration

Nothing in the code depends on where your checkout or home directory is.
Everything machine-specific is an environment variable, and all have
sensible defaults except the email:

| variable | meaning | default |
|---|---|---|
| `FIRESCAPE_DATA` | the shared project data folder (`raw/`, `interim/`, `products/`, `figures/`) | the Box folder at Box Drive's standard location, `~/Library/CloudStorage/Box-Box/SWMresearch/PostFireDebrisFlows/PreFireAssessment` |
| `FIRESCAPE_RESEARCH` | the folder above it, holding sibling per-fire / per-storm project folders a few scripts read | parent of `FIRESCAPE_DATA` |
| `FIRESCAPE_CACHE` | local caches and download staging (never on Box) | `~/.cache/firescape` |
| `FIRESCAPE_PYTHON` | interpreter the shell wrappers and batch drivers use | `python3` on `PATH` / the running interpreter |
| `FIRESCAPE_EMAIL` | address LANDFIRE (LFPS) and MTBS notify when a job or bundle order is ready | **none — required** for staging and ordering |

Files shipped inside the package (calibration TOMLs, fire sets, region
GeoJSONs, the Staley 2018 tables) are located through
`firescape.paths.package_data()`, so they work from any checkout or install.

## Quickstart (pilot AOI north of Reno)

```bash
firescape dem                      # 10 m 3DEP DEM + hillshade for the pilot AOI
firescape fires --names Bug Stallion   # current WFIGS perimeters
firescape assess --fire <event_id>     # hazard chain on one historic fire (observed severity)
firescape prefire                      # pre-fire hazard surface (simulated severity)
```

## License

GPL-3.0-only (required: firescape links against the GPL-3 licensed USGS pfdf
library). Data products (rasters, GPKGs, maps) are not code and carry their own
public-domain/CC attributions — see NOTICE.md.
