# Third-party components and data

## Code dependencies

- **pfdf** (USGS, GPL-3.0-only) — postfire debris-flow hazard models and
  watershed tools. firescape imports pfdf, which is why firescape is licensed
  GPL-3.0-only. https://code.usgs.gov/ghsc/lhp/pfdf
- **stormscape** (MIT) — storm rainfall / DEM / cartography toolkit, imported
  as a library. stormscape itself never imports pfdf and remains MIT.

## Data (packaged in firescape/data/)

- **Staley et al. (2018) Weibull CDF parameters** (`data/staley2018/`) — USGS
  data release, public domain, doi:10.5066/P9TKYL5K. See PROVENANCE.md there.

## Data (fetched at runtime, not redistributed)

MTBS (doi:10.5066/P9NETC0T), LANDFIRE EVT, STATSGO (KFFACT), USGS 3DEP,
NOAA Atlas 14, NIFC WFIGS perimeters — all US-government public data; product
rasters/tables derived from them carry the upstream attributions listed in
each product's run_meta.json.

Note: firescape's *outputs* (rasters, GPKGs, maps, calibration TOMLs) are data,
not GPL-covered code.
