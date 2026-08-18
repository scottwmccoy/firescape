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
| `e1_stage_usmin.py` | USMIN mine features for NV (waste + openings) → `raw/usmin/nv_mines.gpkg`; the waste polygons are the AML exposure assets |
| `e1b_stage_blm.py` | BLM SMA holdings for NV (14 dissolved polys, 66% of the state, ~30 m simplification) → `raw/blm/nv_blm_sma.gpkg` |
| `e1c_stage_nhd_receptors.py` | NHD receptor waters near the AML sites (perennial or GNIS-named, server-side filter; per-cell cache, threaded) → `interim/exposure/nhd_receptors.gpkg` |
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
`firescape.relief` (hillshade); see those modules before changing how a figure
looks, and the repo CLAUDE.md for the DEM-resampling traps. `plotting.save`
writes **PDF only** — pass `formats=("png", "pdf")` if something downstream
needs a raster.

Three rules a zoomed sheet has to keep, each one learned by breaking it:

* **Take the hillshade from `plotting.hillshade`, never `relief.shaded_relief`
  directly.** It sizes the shading resolution to the window — relief.py rule 3
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

## verify/ — Planet-imagery event verification (design: docs/planet_verification_design.md)

The stack–z–segment chain: epoch medians of PlanetScope SR → robust z →
corridor-integrated per-segment response classes (Dolan-inventory coding:
0 none / 1 fluvial / 3 debris flow), plus fan-deposit objects beyond the
network. Library half lives in `firescape/{planetscope,epochs,change,
corridor,fans}.py`; these drivers are the as-run event analyses.

| script | what it does |
|---|---|
| `p13_hidden_valley_event.py` | Hidden Valley 2026-06-19: epoch z maps + chips at the two reported fan sites (first real detection) |
| `p14_hv_corridor.py` | per-segment response map over the pfdf network (1,412 segments: 927/324/161 at provisional thresholds) |
| `p15_dolan_comparison.py` | score corridor z against the Cavagnaro/McCoy Dolan inventory (AUC separability, prevalence-matched thresholds, confusion, side-by-side maps) |
| `p16_hv_fans.py` | fan-stage v0: HAND-lite fan zone + compact change objects (both reported HV fans recovered at 0.3 m / 8.3 m) |

Costs worth knowing: Planet orders take ~15–60 min to process (poll the
ORDER, never a log); the ~30-day CSDA embargo delays post-storm imagery a
month; epoch builds on big windows need the lite options
(`epochs.build(..., indices=..., rgb=False, keep_nir=False,
dtype=np.float16)`) — the Dolan grid is ~36M px × 30 frames.

## products/

| script | what it does |
|---|---|
|  `p5_inventory_pack.py` | builds the manual-survey package: 4.24 M-segment GPKG, regionated KMZ tiles by HU8, hot-segment layer, priority sheet |
| `p2_hindcast_batch.py`, `p2_hindcast_fig.py` | observed-severity hindcasts for historic fires (M6 validation). The batch script invokes the figure script **by path** — keep them in the same directory |
| `e5_perry_canyon.py` | Perry Canyon pilot: adits + shafts as portal-dump proxies where USMIN maps no waste extent (dedupes overlapping quad records) → `products/exposure/perry_canyon_v1/`; sheet = `figures/e6_perry_canyon.py` |
| `e2_aml_exposure.py` | AML waste sites vs the v1.2 network (`firescape.exposure`: near-channel test → site annual rates → named NHD receptor → BLM flag) → ranked `products/exposure/aml_v2/`. Maps: `figures/e3_aml_map.py` (statewide) then `figures/e4_aml_zooms.py` (six district zooms, windows clustered from the ranking — same statewide-then-zoom pattern as `sw_maps_v12` → `p10`/`p12`) |

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
