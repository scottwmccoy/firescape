# scripts — the as-run pipeline

These are the drivers that actually produced the pilot and statewide products,
kept for reproducibility and provenance. They are **not** library code: they
carry hard-coded run parameters, they assume the Box data tree is present
(`FIRESCAPE_DATA`, see the top-level README "Configuration"), and several are
long-running. Reusable logic belongs in the `firescape` package; if something
here starts getting imported, promote it.

Run them with the project environment active:

```bash
python scripts/<group>/<script>.py
```

The `firescape` CLI is a thin launcher for the most-used ones (`prefire`,
`storm`, `assess`, `map`, `hindcast`, `calibrate`, `annualprob`, `severity`);
`--dry-run` prints the script it would run.

Filename prefixes are historical run phases: `m3`–`m5` = pilot milestones,
`p2`–`p16` = statewide phases and studies in order, `sw` = statewide fleet,
`cdf` = severity CDF refit, `e` = exposure, `b` = per-fire staging.

## Conventions worth knowing

- **Resumable drivers** (`*_driver.py`) are time-budgeted and exit **42** when
  the budget is hit, meaning "more work remains, run me again". The `*.sh`
  wrappers loop on that until a pass exits 0. Budgets are env vars
  (`FIRESCAPE_CAL_BUDGET`, `FIRESCAPE_M4_BUDGET`, `FIRESCAPE_STAGE_BUDGET`).
- **Per-fire and per-unit caches** live in `interim/`; drivers skip finished
  work on restart, so re-running is cheap and safe. A calibration fire's class
  decomposition (`decomp.npz`) makes any later re-solve at a new break or CDF
  table a two-second job.
- **Calibration outputs are TOML files** committed to
  `firescape/data/calibration/`, never numbers pasted into code. The current
  one is `statewide_v1_4`; the shipped statewide surface is
  `products/prefire/statewide_v1_2` (a nine-unit test showed v1_4 does not
  visibly change it — `surface/sw_sierra_v14_*`).
- **Versioned scripts are kept, not overwritten**: `sw_driver_v12.py` is the
  current fleet driver and `sw_driver.py` / `_v1` / `_v11` are the runs that
  produced the earlier products.

## stage/ — acquisition and staging

| script | what it does |
|---|---|
| `sw_inventory.py` | NV boundary + the HU10 units, the statewide work list; `p11_add_tahoe_units.py` closes the Tahoe / Truckee basin across the state line |
| `sw_stage_dem.py` | 3DEP 1/3-arcsecond tiles from USGS S3 (resumable) |
| `sw_stage_evt.py` | LANDFIRE EVT over all units via LFPS 2-degree chunks (needs `FIRESCAPE_EMAIL`) |
| `sw_stage_soils.py` | statewide KF: STATSGO if ScienceBase answers, else SSURGO |
| `sw_atlas14.py`, `sw_atlas14_fill.py` | Atlas 14 grids; mosaics the `sw`/`inw`/`ca` regions across the seam |
| `p2_fireset.py` | era-matched statewide calibration fire set (EVT vintage ≤ ignition year − 1); converts BARC256-scale thresholds |
| `p2b_fireset_barc_scale.py` | one-off patch of the staged fire set for the BARC256 threshold scale (adds `map_prog`, `asmnt_type`, `threshold_scale`) |
| `p2_order.py` | places the MTBS bundle order for fires not already downloaded (needs `FIRESCAPE_EMAIL`) |
| `p2_stage.py` | era-matched EVT under each calibration fire + DEM gap fill |
| `p2_regions.py` | the four NV prefire regions, folded from EPA Level III ecoregions |
| `p3_ranges_stage.py` | NSHM-2023 PGA predictor for the RANGES volume model |
| `p7_stage_burnprob.py` | annual burn probability P(F), Wildfire Risk to Communities 2nd ed. |
| `e1_stage_usmin.py`, `e1b_stage_blm.py`, `e1c_stage_nhd_receptors.py` | USMIN mine features (waste + openings), BLM surface-management polygons, NHD receptor waters — the exposure inputs |
| `b1_bear_network.py` | Bear Fire (2024) channel network for tracescape |
| `sw_stage_chain.sh`, `sw_stage_pass.sh` | chain / relaunch wrappers for the staging lanes |

## calibrate/ — severity CDFs and P_dsim

| script | what it does |
|---|---|
| `cdf_refit.py`, `cdf_eval.py`, `cdf_holdout.py` | Nevada EVT–dNBR Weibull refit, era-matched; does it beat Staley 2018; leave-Loyalton-out test |
| `p6_cdf_refit_statewide.py`, `p6_cdf_eval_statewide.py` | the statewide refit (`nv_statewide` table) and its held-out evaluation |
| `m3_driver.py` / `m3_summary.py`, `m3b_*` | pilot P_dsim calibration → `pilot_v1.toml`, recalibration on `nv_merged` → `pilot_v2.toml` |
| `p2_calib_driver.py` / `p2_calib_summary.py` | statewide per-region calibration → `statewide_v1.toml` |
| `p4_calib_driver.py` / `p4_calib_summary.py` | dispersed severity (σ_z = 0.91) → `statewide_v1_1.toml` |
| `p7_calib_driver.py` / `p7_calib_summary.py` | on the statewide CDF refit → `statewide_v1_2.toml`; v1_3 was re-solved from the same caches with the 9999 sentinel guarded |
| `p8_calib_driver.py` / `p8_calib_summary.py` | **current**: BARC256-corrected thresholds, no-`dnbr6` fires observed at analyst thresholds → `statewide_v1_4.toml` |

## surface/ — hazard surfaces and merges

| script | what it does |
|---|---|
| `m4_driver.py`, `m4v2_driver.py`, `m5_merge.py`, `m5v2_merge.py`, `m4_chain.sh` | pilot AOI surface, HU10 at a time, plus P(R>T) |
| `sw_driver.py`, `sw_merge.py`, `sw_fleet_pass.sh` | statewide **v0** (pilot calibration applied statewide) |
| `sw_driver_v1.py`, `sw_merge_v1.py`, `sw_fleet_v1.sh` | statewide **v1** (per-region calibration) |
| `sw_driver_v11.py`, `sw_merge_v11.py`, `sw_fleet_v11.sh` | statewide **v1.1** (dispersed severity + RANGES volume) |
| `sw_driver_v12.py`, `sw_merge_v12.py`, `sw_fleet_v12.sh`, `run_v12_overnight.sh` | statewide **v1.2** — the shipped product; the overnight script chains calibration → fleet → merge → annual probability → maps |
| `p3_ranges_apply.py`, `p3_ranges_final.py`, `sw_ranges_segments.py`, `sw_repair_ranges.py` | per-basin RANGES predictors and volume columns, on basins and on the per-unit segment files |
| `p7_annual_probability.py` | P(F) × P(R>T) for a merged surface (`firescape annualprob <version>`) |
| `sw_sierra_v14_test.py`, `sw_sierra_v14_compare.py` | re-run the nine Sierra units under v1_4 and diff them against the shipped surface |

## analysis/ — the studies behind the model choices

| script | what it does |
|---|---|
| `p3_sigma.py`, `p3_dist_experiment.py` | within-fire severity quantile dispersion → **σ_z = 0.91**; deterministic vs dispersed severity × Gartner vs RANGES on 12 units |
| `p2_ml_probe.py` | leave-one-fire-out probe: does conditioning beyond EVT class help? (`docs/ml_severity_assessment.md`) |
| `p2_severity_mosaic.py` | statewide simulated-severity intermediates, mapped (`firescape severity`) |
| `p6_prt_decomposition.py` | why P(R>T) rises toward southeast Nevada |
| `p7_baer_comparison.py`, `p8_baer_comparison_ca.py` | the calibrated break against field-verified BAER soil burn severity — 19 Nevada fires, 112 California fires on Rossi's breaks |
| `p9_baer_outlier_diagnostics.py` | why the BAER-implied break scatters: dNBR level, assessment timing, imagery cadence |
| `p10_break_experiment.py` | an assessment-type-aware break, tested leave-one-fire-out on cached joint histograms |
| `p11_baer_patchiness.py`, `p11_baer_fireweather.py`, `p11_baer_sign_model.py` | can patchiness, perimeter shape or gridMET fire weather predict which way a fire misses BAER (no) |

## products/

| script | what it does |
|---|---|
| `p8_fire_forecast.py` | pre-fire forecast for a named WFIGS fire on simulated severity (`firescape prefire`) |
| `p8_fire_storm_response.py` | which segments a measured storm should have triggered (`firescape storm`) |
| `p9_fire_observed.py` | the same fire on OBSERVED severity from CIMSS BRISK, injected — no MTBS bundle, assessable while burning (`firescape assess`) |
| `p16_mtbs_fire_hindcast.py` | post-fire hazard hindcast on MTBS severity, against the observed debris-flow inventory (`firescape hindcast`) |
| `p2_hindcast_batch.py`, `p2_hindcast_fig.py` | observed-severity hindcasts + figures for a list of MTBS fires (the batch invokes the figure script **by path** — keep them together) |
| `p13_caltopo_recon.py` | field reconnaissance package for CalTopo (Bug + Stallion) |
| `p5_inventory_pack.py` | manual-survey package: the segment network as regionated KMZ tiles by HU8, hot-segment layer, priority sheet |
| `e2_aml_exposure.py`, `e5_perry_canyon.py`, `e7_openings_exposure.py` | AML waste-site exposure against the statewide network (`firescape.exposure`), the Perry Canyon pilot, and the statewide openings variant |

### The active-fire pipeline

Every stage takes a **fire name** and a calibration version, so any WFIGS fire
runs through unchanged — Stallion, Bug and Hawk were all done this way. The
CLI wraps it:

```bash
firescape prefire Bug                 # = python scripts/products/p8_fire_forecast.py       Bug statewide_v1_4
firescape storm   Bug                 # = python scripts/products/p8_fire_storm_response.py Bug statewide_v1_4
firescape assess  Bug                 # = python scripts/products/p9_fire_observed.py       Bug statewide_v1_4
firescape map     Bug --sheet all     # = figures/p15_fire_postfire_hazard.py (+ --prefire), figures/p8_fire_storm_map.py
```

## figures/

| script | what it does |
|---|---|
| `p15_fire_postfire_hazard.py` | **the fire sheet**: observed severity, likelihood at the design storm, triggering intensity — the emergency-assessment form the USGS publishes; `--prefire` draws it on simulated severity |
| `p8_fire_forecast_map.py`, `p8_fire_storm_map.py`, `p9_fire_observed_map.py` | the earlier fire sheets: forecast, storm response, observed severity under the measured storm (superseded by p15 for hazard; kept for the storm map) |
| `p16_debrisflow_inventory_statewide.py` | the statewide surface with the observed debris-flow inventory on top |
| `sw_maps*.py`, `p2_severity_maps.py`, `p2_class_plots.py`, `p2_fires_overlay.py`, `p3_experiment_fig.py`, `p7_annual_probability_map.py` | statewide hazard (one per version), simulated severity, severity by vegetation class, historic perimeters over the hazard map, the dispersion experiment, annual probability |
| `p9_baer_case_panels.py`, `p9_baer_timing_plot.py`, `p10_break_response.py`, `p10_imagery_cadence.py` | the BAER comparison figures: side-by-side classified maps, timing, the break-response curves, MTBS image cadence |
| `p14_zoom_volume.py` | zoom volume sheets |
| `e3_aml_map.py`, `e4_aml_zooms.py`, `e6_perry_canyon.py`, `e8_aml_compilation.py` | the exposure maps |
| `m4_maps.py`, `m4v2_maps.py`, `v1v2_maps.py` | pilot |

### Regional zooms

Two sheets over each window, in two formats:

| script | format | asks |
|---|---|---|
| `p10_urban_zooms.py` | triptych, outlet-basin polygons | P(F), P(R>T) and their product — **how often** should this happen |
| `p12_urban_zoom_forecast.py` | pair, stream-segment lines | triggering $I_{15}$ and hazard class — **what happens if it burns**, the form the single-fire sheets take |

```bash
python scripts/figures/p10_urban_zooms.py          reno_carson statewide_v1_2
python scripts/figures/p12_urban_zoom_forecast.py  ely_white_pine statewide_v1_2
```

With no window argument, either script renders all of them. The windows:

| key | what it covers | note |
|---|---|---|
| `reno_carson` | Pyramid Lake → Reno–Sparks → Carson Valley, Sierra crest | crosses into CA |
| `las_vegas` | Clark County, Spring Mountains, Colorado River | 17.6% KF gap |
| `elko_corridor` | Carlin–Elko–Wells, Ruby Mountains | highest moderate share (26%) |
| `winnemucca_battle_mtn` | western I-80, Sonoma/Osgood/Shoshone ranges | |
| `nnss` | Nevada National Security Site, Beatty, Amargosa | **52% KF gap** |
| `ely_white_pine` | Egan/Schell Creek ranges, Snake Range, Great Basin NP | highest conditional hazard near people |
| `lincoln_caliente` | Meadow Valley Wash, Rainbow Canyon, US-93 | UP mainline exposure |
| `north_elko` | Jarbidge, Mountain City, Owyhee, Jackpot | shortest annual return intervals |

The last four were chosen from a scan of hazard within ~20 km of every Nevada
town outside the first four; `Fallon` and `Tonopah` were checked and rejected
(0.4% and 1.1% moderate).

Both import `_corridors.py`, which owns the window bounds, the place and
watercourse whitelists, the label styling and the figure sizing — so the two
sheets stay comparable and a new region is one dict entry. The forecast sheet
reads the **per-unit** `*_segments.gpkg` files (307k segments in Reno, 588k in
Las Vegas) rather than the merged basin layer, and repeats the merge's KF gap
fill so the two formats agree.

Two costs worth knowing before adding a window:

* The NHD named-stream query runs **75 s to 9 min** and sits at 0% CPU the
  whole time, blocked on the network — it looks exactly like a hang. It caches
  per window, **unfiltered**, so revising the `rivers` whitelist afterwards is
  free.
* Where SSURGO never mapped the soil, K is filled at the statewide median.
  That is 0.0% over Elko County but 52% over the Test Site, so above 5% the
  sheet says so in its own title rather than only in the summary JSON.

All of them draw through `firescape.plotting` (house style) and
`stormscape.relief` (hillshade); see those modules before changing how a
figure looks, and the repo CLAUDE.md for the DEM-resampling traps.
`plotting.save` writes **PDF only** — pass `formats=("png", "pdf")` if
something downstream needs a raster.

Three rules a zoomed sheet has to keep, each one learned by breaking it:

* **Take the hillshade from `plotting.hillshade`, never `relief.shaded_relief`
  directly.** It sizes the shading resolution to the window — relief rule 3
  is *shade finer than the display grid*, and a panel shaded at the statewide
  50 m under a 15 m grid arrives pre-blurred — and it keys its cache on the
  grid origin, so two windows of one shape cannot swap terrain.
* **Pass `extent=` to `draw_context`, and label after the layout is final.**
  Names are placed through `stormscape.plot.Labeller`, which moves a label
  that would land on another one or leave the frame; placement is computed in
  display coordinates, so the axes limits have to be set first.
* **`rasterized=True` on any collection with more than a few thousand
  members.** The district sheet was 57 MB as vectors and 4 MB rasterized, for
  no visible difference — each segment is a couple of pixels at print scale.
  Text, markers and leaders stay vector.

## Not here any more

The Planet-imagery event verification stack (`planetscope`, `epochs`,
`change`, `corridor`, `fans` and the `verify/` drivers) was carved out to the
sibling **tracescape** repo on 2026-08-19; firescape imports one function from
it (`tracescape.corridor.corridor_width_m`, for `exposure`). Superseded one-off
diagnostics were folded into tests (`test_relief` in stormscape,
`test_statewide` here) or replaced by library code (`soils.kf_raster`).
