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

## Nevada refit (added 2026-08-13)

`CDFParameters_NV_refit.csv` (35 classes) and `CDFParameters_NV_merged.csv`
(288 classes = Staley 2018 with the Nevada refits overriding matched classes).

**Method.** Weibull CDFs refit from observed MTBS dNBR pooled over pilot-area
fires, paired with **era-matched** vegetation: LANDFIRE **LF2016** EVT with
fires that ignited **2017 or later**. Era matching is essential — EVT mapped
after a fire describes POST-fire vegetation, and using it leaks burn severity
into what is supposed to be a pre-fire prediction. Pixels are restricted to
the MTBS burn perimeter, with dnbr6 classes 5 (increased greenness) and 6
(non-mapping) excluded. Classes with fewer than 2,000 pixels are not refit.

**Fit.** `severity.fit_weibull_cdf` — Weibull probability-plot linearization
(`ln(-ln(1-F)) = kappa*ln(z) - kappa*ln(lambda)`), the same construction whose
R^2/RMSE the Staley release tabulates. Median R^2 = 0.965 (min 0.858).

**Result.** Refit from 1.95M pixels (16 fires). Staley 2018 systematically
compresses the Nevada range: it *under*-predicts cheatgrass (Introduced Annual
Grassland: 185 simulated vs 272 observed median dNBR) and montane conifer
(363 vs 661), and *over*-predicts chaparral (516 vs 387). Weighted mean
absolute error against observed class medians falls from **61.5 to 9.0 dNBR**.

**Validation (leave-one-fire-out).** Refit from 15 fires with Loyalton (2020)
held out entirely, then used to predict it: NSE **0.238 -> 0.497** at
P_dsim 0.48 and **0.275 -> 0.519** at each table's own optimum; RMSE
0.183 -> 0.149. See `products/calibration/cdf_holdout_eval.json`.

**Caveats.** Fit from pilot-area fires only, so several classes (notably the
Sierra conifer types) rest on Sierra-front data; statewide adoption needs a
statewide fire set. The regional P_dsim (0.48) was calibrated against the
Staley table — P_dsim and the CDF table are coupled, so adopting the merged
table warrants recalibration (its held-out optimum sits near 0.54).
