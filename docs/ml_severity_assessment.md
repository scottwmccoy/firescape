# Can ML beat the per-class quantile severity simulator? An assessment

*firescape working document — 2026-08-13. Companion to the statewide v1
calibration. Data behind the probe: `interim/calib/refit_samples_cov.parquet`
(2.04M era-matched pixels, 16 pilot-area fires); result table:
`products/calibration/ml_probe_lofo.csv`.*

## 1. The question

firescape currently simulates pre-fire burn severity the Staley/Rossi way:
each LANDFIRE EVT class draws its dNBR at a single regional quantile
`P_dsim` from a per-class Weibull CDF, and `P_dsim` is calibrated so
simulated debris-flow likelihood (M1) matches likelihood computed from
observed MTBS severity over calibration fires (Rossi et al. 2025). The
question: would a modern ML model of severity — conditioned on more than
vegetation class — measurably improve the statewide product, and is it worth
running in parallel with the per-region Weibull calibration?

## 2. What the current method is, statistically

The Staley 2018 table is a set of *marginal* conditional distributions:
`F(dNBR | EVT class)`, one covariate, categorical. `P_dsim` is a single
latent scenario variable shared by every pixel — it plays the role of
"how bad a fire year/weather is" without naming it. Structural limits:

1. **No within-class conditioning.** Two sagebrush pixels — one on a cool
   north slope with heavy fuel, one on thin rocky south-facing ground —
   simulate identically.
2. **One dial for all classes.** The same quantile is applied to cheatgrass
   and to subalpine conifer, though their sensitivity to fire weather
   differs.
3. **No spatial structure.** Simulated severity is spatially white within a
   class; observed severity has patch structure that matters for the M1
   terrain term (contiguity of moderate+high on steep slopes).
4. **Stationarity.** The CDFs pool 1984–2015-era fires; severity is trending
   (Parks & Abatzoglou 2020). (Our Nevada era-matched refit already
   addresses the worst of this with 2017+ fires.)

## 3. What the literature says (survey, 2026-08-13)

**Pre-fire severity prediction with ML is established, forest-centric, and
USGS-precedented.**

- Wells et al. 2023 (IJWF, doi:10.1071/WF22200) and Wells et al. 2024 (IJWF,
  doi:10.1071/WF23199) — both USGS-authored — predict dNBR with random
  forests from pre-fire fuels/topography/weather covariates and push the
  result through the same Staley M1 + volume machinery we use. Honest
  cross-fire skill: R² ≈ 0.25 (Dollar Ridge model → East Fork fire);
  within-region mean validation R² 0.54.
- Klimas et al. 2025 (Fire Ecology, doi:10.1186/s42408-024-00346-z) is the
  statewide template: Utah, 203 fires, RF on 23 covariates (LANDFIRE canopy
  and fuels, topography, one gridMET ERC scalar per fire as a scenario
  dial). OOB R² 0.67, **leave-fire-out R² 0.41**, 3-class accuracy 0.85;
  accuracy improves at basin aggregation (0.63 pixel → 0.75 sub-catchment).
- Parks et al. 2018 (ERL, doi:10.1088/1748-9326/aab791): probability of
  stand-replacing fire across western US forests, BRT, AUC ≈ 0.72; live
  fuel carried 53% of influence, fire weather 23%, topography only 10% —
  and weather was handled by *marginalizing over observed burning weather*,
  the other accepted way to make a pre-fire model.
- **Gaps:** no published conditional-*quantile* severity model (the direct
  ML generalization of Staley 2018), and no Great Basin shrubland severity
  prediction at all. Both are open contributions.
- **Caution:** on the debris-flow likelihood side, ML wins on heterogeneous
  multi-region data (Nikolopoulos 2018: RF AUC 0.94 vs 0.80 logistic;
  Roten 2022: 0.93 vs 0.79) but *not* on a clean regional dataset with good
  features (Fernandez Sirgo 2026, doi:10.1029/2025JF008627: logistic
  regression matched or beat RF/XGBoost). ML gains over well-specified
  simple models are not automatic.
- Deep learning: all published CNN/U-Net severity work maps severity from
  post-fire imagery; none predicts a hypothetical future fire's dNBR from
  pre-fire covariates alone. No precedent to lean on, and ~150 fires cannot
  train one from scratch.

## 4. The Nevada probe (2026-08-13)

Leave-one-fire-out over the six largest era-matched pilot fires (train on
the other 15; 2.04M pixels total): per-class-median predictor (the marginal
analog) vs gradient boosting (`HistGradientBoostingRegressor`) on class +
elevation + slope + aspect.

| held-out fire | class-median R² | +terrain HGB R² | MAE (dNBR) |
|---|---|---|---|
| Sugar 2021 | −0.06 | −0.02 | 181 → 180 |
| Long Valley 2017 | −0.34 | −0.24 | 96 → 94 |
| Tohakum 2 2017 | +0.02 | +0.02 | 87 → 88 |
| Perry 2018 | −0.22 | −0.17 | 113 → 110 |
| Loyalton 2020 | +0.37 | +0.39 | 121 → 117 |
| Earthstone 2017 | +0.10 | +0.02 | 97 → 101 |
| **mean** | **−0.02** | **+0.00** | **116 → 115** |

**Terrain conditioning adds nothing out-of-fire in this rangeland sample.**
The spread between fires (−0.34 to +0.39) dwarfs the spread between models
(≤0.1): the binding unknown is the *fire-level* state — weather, drought,
and (for cheatgrass) the fuel year — precisely the latent variable that
`P_dsim` already represents. This is consistent with Parks 2018's importance
ranking (topography last), with Wells' cross-fire R² 0.25, and with the
Fernandez Sirgo caution. Probe limits: 16 fires from one corner of the
state; no fuels, cover, or climate covariates; pixel-level scoring (the
operational metric is basin-level M1 agreement, which is kinder to all
models).

## 5. Where ML can actually help here (ranked)

1. **Conditional-quantile severity (the serious challenger).** Replace
   `F(dNBR | class)` with `F(dNBR | class, fuels, cover, climate normals)`
   via quantile gradient boosting or quantile regression forests, keep the
   quantile dial, and calibrate the dial exactly as Rossi does. Downstream
   pipeline unchanged. The probe says the covariates that might matter in
   Nevada are **fuels and cover, not terrain**: RAP/RCMAP herbaceous +
   shrub + annual-grass fractional cover (rangelands), LANDFIRE canopy
   variables (conifer belts), PRISM/gridMET climate normals. Apparently
   unpublished; defensible as "a covariate-conditioned generalization of
   Staley et al. 2018, calibrated identically to Rossi et al. 2025."
2. **Class-group-specific dials.** Cheaper than full ML: let cheatgrass-like
   classes and conifer classes sit at different calibrated quantiles
   (2-3 parameters instead of 1). Captures differing weather sensitivity
   with almost no methodological risk.
3. **An annual severity scenario dial.** For cheatgrass systems the fuel
   year (prior-winter precipitation) modulates severity; an ERC/fuel-year
   percentile input (Klimas-style) would let the statewide map be issued as
   "given a bad fire year" vs median — a product NV users would feel.
4. **Spatially correlated severity fields.** If the M1 terrain term proves
   sensitive to severity patch structure, simulate quantile *fields* with a
   variogram-matched Gaussian copula rather than a learned generator.
5. **Not recommended:** replacing M1 itself (mixed evidence, breaks the
   USGS contract), or deep learning (no precedent, insufficient fires).

Separate but adjacent: for volume, evaluate the Gorr et al. 2026 WEST model
(doi:10.5194/nhess-26-2111-2026) against Gartner 2014 — Gartner overpredicts
outside its SoCal training domain by orders of magnitude, Nevada is outside
that domain, and WEST's rainfall-ratio normalization matches our Atlas 14
layer. Likewise Cavagnaro et al. 2025 rainfall-anomaly normalization for
threshold transferability.

## 6. Evaluation harness (already built)

The decision metric is not pixel R². It is: **leave-one-fire-out agreement
of simulated vs observed M1 likelihood at the basin scale** — the same
NSE machinery used for the CDF refit (`cdf_holdout` flow; NSE 0.238 → 0.497
when the NV refit replaced Staley 2018 on Loyalton). Any challenger slots
into `calibrate.fire_calibration` as a severity engine: per-class dNBR
vector → per-basin T/F via the exact class decomposition, or per-pixel
fields → direct M1.variables. The benchmark ladder, run identically:

| rung | severity engine | status |
|---|---|---|
| B0 | Staley 2018 @ calibrated P | done (pilot NSE 0.238) |
| B1 | NV refit (nv_merged) @ calibrated P | done (pilot NSE 0.497) |
| B2 | per-region statewide refit @ per-region P | in progress (123 fires) |
| B3 | class-group dials | cheap, after B2 |
| B4 | QRF/GBM conditional quantiles + fuels/cover covariates | the ML experiment |

Promotion rule: a rung ships only if it beats the rung below on held-out
DFL NSE with the same calibration protocol.

## 7. Recommendation

Proceed with the per-region Weibull calibration (B2) as the operational v1
— it is the Rossi method, statewide, era-matched, and nothing in the
literature or the probe suggests ML displaces it this cycle. Stand up B4 as
a *parallel research track* once the 123-fire statewide sample is on disk:
QRF with RAP/RCMAP + LANDFIRE fuels + climate normals, leave-fire-out,
scored on the DFL harness. B3 and the annual dial are low-cost add-ons with
user-visible value. Publishing angle regardless of outcome: first severity
prediction benchmark for Great Basin shrublands, and the conditional-
quantile formulation of pre-fire severity simulation.

## 8. Key references

Staley et al. 2018, IJWF, doi:10.1071/WF17122 · Rossi et al. 2025, IJWF,
doi:10.1071/WF24225 · Wells et al. 2023, IJWF, doi:10.1071/WF22200 ·
Wells et al. 2024, IJWF, doi:10.1071/WF23199 · Klimas et al. 2025, Fire
Ecology, doi:10.1186/s42408-024-00346-z · Parks et al. 2018, ERL,
doi:10.1088/1748-9326/aab791 · Dillon et al. 2011, Ecosphere,
doi:10.1890/ES11-00271.1 · Keyser & Westerling 2017, ERL,
doi:10.1088/1748-9326/aa6b10 · Parks & Abatzoglou 2020, GRL,
doi:10.1029/2020GL089858 · Nikolopoulos et al. 2018, NHESS,
doi:10.5194/nhess-18-2331-2018 · Kern et al. 2017, Math Geosci,
doi:10.1007/s11004-017-9681-2 · Roten et al. 2022, IEEE Big Data,
doi:10.1109/BigData55660.2022.10020574 · Fernandez Sirgo et al. 2026,
JGR-ES, doi:10.1029/2025JF008627 · Cavagnaro et al. 2025, JGR-ES,
doi:10.1029/2024JF007781 and GRL, doi:10.1029/2025GL114791 · Gorr et al.
2024, JGR-ES, doi:10.1029/2024JF007825 · Gorr et al. 2026, NHESS,
doi:10.5194/nhess-26-2111-2026 · Kean & Staley 2021, Earth's Future,
doi:10.1029/2020EF001735 · Tillery et al. 2014, USGS SIR 2014-5161 ·
Meinshausen 2006, JMLR 7:983–999 (quantile regression forests) · Simafranca
et al. 2024, Env Ecol Stat, doi:10.1007/s10651-024-00601-1 · Pascolini-
Campbell et al. 2022, GEB, doi:10.1111/geb.13526 · Wildfire Risk to
Communities (FSim conditional flame length): fs.usda.gov RDS-2016-0034-3.
