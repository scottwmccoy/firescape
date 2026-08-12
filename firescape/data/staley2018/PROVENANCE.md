# Staley et al. (2018) EVT-dNBR Weibull CDF parameters

- **Source**: Staley, D.M., 2018, *Data used to characterize the historical
  distribution of wildfire severity in the western United States in support of
  pre-fire assessment of debris-flow hazards*. U.S. Geological Survey data
  release. **doi:10.5066/P9TKYL5K**
- ScienceBase item: `5b2d0704e4b040769c10b72c`
  (https://www.sciencebase.gov/catalog/item/5b2d0704e4b040769c10b72c)
- Retrieved: 2026-08-12, via HTTPS with a browser User-Agent (ScienceBase
  returns 403 to non-browser agents). File URLs came from the item's
  `?format=json` metadata.
- License: USGS data release, public domain (US Government work).
- Companion paper: Staley et al. (2018), *Estimating post-fire debris-flow
  hazards prior to wildfire using a statistical analysis of historical
  distributions of fire severity from remote sensing data*, IJWF 27(9),
  595-608, doi:10.1071/WF17122. Fits are from 3,163 western-US burn areas,
  2001-2014.

## Files

| File | sha256 | Notes |
|---|---|---|
| `CDFParameters.txt` | `7fb77bed55d080bc670294d0b572e5ac0ca3ee33b2bc4c824bfdc8c286bfbf0c` | 282 EVT classes: EVT_Code, N, Weibull_Lambda_Scale, Weibull_Kappa_Shape, Weibull_R2, Weibull_RMSE, CLASSNAME. dNBR normalized as (dNBR+1000)/2000 |
| `BARCThresholds.txt` | `eb5e4beb724c6c56ac757a0ecd69adb822b5f4ef29f6323d17b17e920e024318` | Historical BARC threshold data |
| `DataDescription.txt` | `bb2fb7b51f18478e3ccde6376dc39285b20472f0394e7034fd2e0bdf6a5716e2` | 16-byte placeholder as served (upstream file is effectively empty) |

The 3.6 GB `EVT_SampleData.zip` (per-fire samples underlying the fits) is NOT
packaged - only needed if/when we refit Nevada-specific CDFs (deferred; see
plan). Download it to Box `raw/staley2018/` at that point.
