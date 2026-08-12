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

Use the `FireMan` conda env — see `CLAUDE.md` for the exact environment rules.

```bash
/opt/anaconda3/envs/FireMan/bin/python -m pip install -e ~/git/code/stormscape
/opt/anaconda3/envs/FireMan/bin/python -m pip install -e ~/git/code/firescape
# pfdf comes from the USGS registry, NOT PyPI:
/opt/anaconda3/envs/FireMan/bin/python -m pip install pfdf -i https://code.usgs.gov/api/v4/groups/859/-/packages/pypi/simple
```

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
