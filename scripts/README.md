# scripts — the as-run pipeline

These are the drivers that actually produced the pilot and statewide products,
kept for reproducibility and provenance. They are **not** library code: they
carry hard-coded run parameters, they assume the Box data tree is present
(`FIRESCAPE_DATA`), and several are long-running. Reusable logic belongs in the
`firescape` package; if something here starts getting imported, promote it.

Run them with the project environment:

```bash
/opt/anaconda3/envs/FireMan/bin/python scripts/<group>/<script>.py
```

Filename prefixes are historical run phases: `m3`–`m5` = pilot milestones,
`p2`–`p5` = statewide phases 2–5, `sw` = statewide fleet, `cdf` = severity
CDF refit.

## Conventions worth knowing

- **Resumable drivers** (`*_driver.py`) are time-budgeted and exit **42** when
  the budget is hit, meaning "more work remains, run me again". The `*.sh`
  wrappers loop on that until a pass exits 0.
- **Per-fire and per-unit caches** live in `interim/`; drivers skip finished
  work on restart, so re-running is cheap and safe.
- **Calibration outputs are TOML files** committed to
  `firescape/data/calibration/`, never numbers pasted into code.

## stage/ — acquisition and staging

| script | what it does |
|---|---|
| `sw_inventory.py` | NV boundary + the 591 HU10 units, the statewide work list |
| `sw_stage_dem.py` | 3DEP 1/3-arcsecond tiles from USGS S3 (resumable) |
| `sw_stage_evt.py` | LANDFIRE EVT over all units via LFPS 2-degree chunks |
| `sw_stage_soils.py` | statewide KF: STATSGO if ScienceBase answers, else SSURGO |
| `sw_atlas14.py`, `sw_atlas14_fill.py` | Atlas 14 grids; mosaics the `sw`/`inw`/`ca` regions across the seam |
| `p2_fireset.py` | era-matched statewide calibration fire set (EVT vintage ≤ ignition year − 1) |
| `p2_order.py` | places the MTBS bundle order for fires not already downloaded |
| `p2_stage.py` | era-matched EVT under each calibration fire + DEM gap fill |
| `p2_regions.py` | the four NV prefire regions, folded from EPA Level III ecoregions |
| `p3_ranges_stage.py` | NSHM-2023 PGA predictor for the RANGES volume model |
| `sw_stage_chain.sh`, `sw_stage_pass.sh` | chain/relaunch wrappers for the above |

## calibrate/ — severity CDFs and P_dsim

| script | what it does |
|---|---|
| `cdf_refit.py` | Nevada EVT–dNBR Weibull refit, era-matched → `CDFParameters_NV_{refit,merged}` |
| `cdf_eval.py` | does the refit beat Staley 2018? (per-class medians, weighted MAE) |
| `cdf_holdout.py` | leave-one-fire-out: refit without Loyalton, then predict it |
| `m3_driver.py` / `m3_summary.py` | pilot P_dsim calibration → `pilot_v1.toml` |
| `m3b_driver.py` / `m3b_summary.py` | recalibration against `nv_merged` → `pilot_v2.toml` |
| `p2_calib_driver.py` / `p2_calib_summary.py` | statewide per-region calibration → `statewide_v1.toml` |
| `p4_calib_driver.py` / `p4_calib_summary.py` | recalibration under dispersed severity (σ_z = 0.91) → `statewide_v1_1.toml` |

## surface/ — hazard surfaces and merges

| script | what it does |
|---|---|
| `m4_driver.py`, `m4v2_driver.py`, `m5_merge.py`, `m5v2_merge.py`, `m4_chain.sh` | pilot AOI surface, HU10 at a time, plus P(R>T) |
| `sw_driver.py`, `sw_merge.py`, `sw_fleet_pass.sh` | statewide **v0** (pilot calibration applied statewide) |
| `sw_driver_v1.py`, `sw_merge_v1.py`, `sw_fleet_v1.sh` | statewide **v1** (per-region calibration) |
| `sw_driver_v11.py`, `sw_merge_v11.py`, `sw_fleet_v11.sh` | statewide **v1.1** (dispersed severity + RANGES volume) — the current product |
| `p3_ranges_apply.py`, `p3_ranges_final.py` | per-basin RANGES predictors (mean slope, north-facing fraction) and the volume columns |

## analysis/ — the studies behind the model choices

| script | what it does |
|---|---|
| `p3_sigma.py` | measures within-fire severity quantile dispersion → **σ_z = 0.91**, now `severity.SIGMA_Z_DEFAULT` |
| `p3_dist_experiment.py` | deterministic vs dispersed severity × Gartner vs RANGES volume, 12 units |
| `p2_ml_probe.py` | leave-one-fire-out probe: does conditioning beyond EVT class help? (feeds `docs/ml_severity_assessment.md`) |
| `p2_severity_mosaic.py` | statewide simulated-severity intermediates — the layer that exposed the EVT seam bug |

## figures/

`sw_maps*.py` (statewide hazard, one per version), `p2_severity_maps.py`,
`p2_class_plots.py` (severity by vegetation class), `p2_fires_overlay.py`
(historic MTBS perimeters over the hazard map), `p3_experiment_fig.py`,
`m4_maps.py` / `m4v2_maps.py` / `v1v2_maps.py` (pilot).

All of them draw through `firescape.plotting` (house style) and
`firescape.relief` (hillshade); see those modules before changing how a figure
looks, and the repo CLAUDE.md for the DEM-resampling traps.

## products/

| script | what it does |
|---|---|
|  `p5_inventory_pack.py` | builds the manual-survey package: 4.24 M-segment GPKG, regionated KMZ tiles by HU8, hot-segment layer, priority sheet |
| `p2_hindcast_batch.py`, `p2_hindcast_fig.py` | observed-severity hindcasts for historic fires (M6 validation). The batch script invokes the figure script **by path** — keep them in the same directory |

### Active-fire pipeline (`p8_*`, `p9_*`)

Every stage takes a **fire name** and a calibration version, so any WFIGS fire
runs through unchanged — Stallion and Bug were both done this way:

```bash
python scripts/products/p8_fire_forecast.py       Bug statewide_v1_1  # pre-fire, simulated severity
python scripts/products/p8_fire_storm_response.py Bug statewide_v1_1  # score vs observed MRMS rainfall
python scripts/products/p9_fire_observed.py       Bug statewide_v1_1  # re-assess on BRISK measured severity
python scripts/figures/p8_fire_forecast_map.py    Bug statewide_v1_1
python scripts/figures/p8_fire_storm_map.py       Bug statewide_v1_1
python scripts/figures/p9_fire_observed_map.py    Bug statewide_v1_1
```

`p9_fire_observed.py` needs no MTBS bundle: it pulls near-real-time dNBR through
`stormscape.burn` (CIMSS BRISK) and passes it to `assess.run_observed` via its
`perimeter=` / `dnbr=` injection, so a fire can be assessed while still burning.

## Deliberately not kept

Superseded one-off diagnostics: the hillshade method comparison (now
`tests/test_relief.py`), the EVT seam blast-radius probe (now
`tests/test_statewide.py`), and two provisional KF fill scripts (superseded by
`soils.kf_raster`).
