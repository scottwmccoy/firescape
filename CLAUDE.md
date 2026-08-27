# firescape — development notes

Pre-fire PFDF hazard tool for Nevada (USGS LHP award, Task 2/Deliverable 2, due
2027-07-31). Approved build plan: `~/.claude/plans/cozy-growing-bachman.md`
(design companion: `cozy-growing-bachman-agent-a9036a316d8dd9f31.md`).

## Environment — IMPORTANT

- Run all Python with the **FireMan** conda env:
  `/opt/anaconda3/envs/FireMan/bin/python`
- FireMan is a conda-forge base (python 3.12) whose science stack is
  **pip-managed** (pfdf pulled it in). Extend it with FireMan's own pip —
  do NOT `conda install` scientific packages into it (double-manages numpy).
- `pfdf` is NOT on PyPI. Install/upgrade only from the USGS registry:
  `pip install pfdf -i https://code.usgs.gov/api/v4/groups/859/-/packages/pypi/simple`
  (PyPI packages named `pfdf`/`wildcat` are unrelated third-party projects.)
- `stormscape` is an editable local install from `~/git/code/stormscape`.
- `tracescape` is an editable local install from `~/git/code/tracescape`
  (`exposure.py` borrows its corridor width law).

## Data layout

- All heavy data on Box:
  `~/Library/CloudStorage/Box-Box/SWMresearch/PostFireDebrisFlows/PreFireAssessment/`
  with `raw/` (immutable + `<name>.provenance.json`), `interim/` (regeneratable),
  `products/` (versioned run dirs + `run_meta.json`), `figures/`.
- Caches are LOCAL, never Box: `~/.cache/firescape/` (`firescape.paths`).
  Downloads stage locally and move atomically into Box.
- This repo holds code, tests, calibration TOMLs, region GeoJSONs, and the two
  small Staley 2018 text files. No rasters, no zips (gitignored).

## pfdf unit traps (all conversions live in hazard.py / delineate.py)

| Trap | Rule |
|---|---|
| dNBR conventions | `severity.estimate()` wants dNBR×1000 (MTBS-style ints); `M1.variables(...dnbr...)` wants the SAME ×1000 values (divides by 1000 internally to make F) |
| Rainfall | Staley-2017 `R` = **accumulation in mm per duration**: I15 24 mm/h → R15 = 6 mm (`pfdf.utils.intensity.to_accumulation`). Gartner-2014 takes i15 in mm/h directly |
| Slopes | gradients (rise/run), NOT degrees — the ≥23°/≥30° cuts are applied inside pfdf |
| pysheds NoData | pysheds silently treats missing NoData as 0 — always set explicit nodata on Rasters |
| MTBS classes | dnbr6 is SIX classes: 1=unburned-low 2=low 3=moderate 4=high **5=increased greenness (exclude) 6=non-mapping (nodata)** — not BARC4 |

## Sibling packages — what lives where

| Package | Licence | Owns |
|---|---|---|
| **stormscape** | MIT | Rainfall (MRMS/NEXRAD/gauges/Atlas 14), DEMs, shaded relief, CalTopo export |
| **tracescape** | MIT | Imagery change detection: epochs, robust z, corridor/fan conditioning. *Validates* what firescape predicts |
| **firescape** | GPL-3 (pfdf) | Simulated severity, the hazard chain, calibration, statewide surfaces, exposure |

The imagery-validation workflow was carved out to `~/git/code/tracescape` on
2026-08-19 (it was `planetscope/sources/epochs/change/corridor/fans` +
`scripts/verify/`). firescape imports **one** function from it,
`corridor.corridor_width_m`, so `exposure.py`'s "delivers to this asset" and
tracescape's "we looked for change here" mean the same strip of ground.
**tracescape must never import firescape** — that arrow points one way only.

`relief.py` moved to **stormscape** at the same time: a terrain backdrop is not
specific to a science question, and both downstream packages draw on it.

## DEM resampling + shaded relief (stormscape/relief.py)

Hillshading differentiates, so it is the loudest display of a resampling
mistake. Inherited from stormscape's 2026-07-31 finding (a nearest warp inside
`py3dep.get_dem` biased slope ~1.8° and halved plan curvature); re-learned on
the statewide figure 2026-08-14.

| Trap | Rule |
|---|---|
| Double resampling | Warp each tile **exactly once**, native grid → target, **bilinear**. A 1/4 `average` decimating pre-pass cost **2.2 m elevation RMS**. Never nearest for elevation; cubic overshoots cliffs into bright rims |
| Tile seams | Mosaic elevations into one **continuous** surface *before* shading. Shading tiles separately puts a hard line at every boundary — the gradient kernel at a tile edge has no data past it. Read source windows with a ≥2-cell margin (stormscape's "+20 cells") |
| Mushy terrain | Shade **finer** than the display grid, then average the *shaded* array down (as stormscape `plot._prepare_hillshade` does). Shading at display resolution halved texture: Laplacian roughness 0.045 vs 0.088 |
| `LightSource.hillshade` | **Do not use.** Its contrast stretch rescales by the min/max of whatever array it is handed, so band-wise shading of a statewide DEM gives every band its own contrast — printing the very seams above. Use `relief.shade` (fixed map; flat ground = sin(altitude)) |
| Coverage gaps | Set uncovered ground to the flat-ground value, dilated past the kernel's reach — a NaN fill plateau otherwise shades as a cliff, and an arbitrary "neutral" constant prints as a grey block |

`statewide.dem_for_unit` (science grids, 10 m) and `relief.shaded_relief`
(display grids) both follow this. Statewide at 50 m: ~4.6 min, ~12 GB peak.

## Validated endpoints (probed 2026-08-12; see firescape/mtbs.py, fires.py)

- WFIGS current perimeters (5-min refresh):
  `https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters_Current/FeatureServer/0/query`
  — match names with `UPPER(attr_IncidentName) LIKE ...`; join on `attr_IrwinID`
  (names get reused; merged fires drop records). `poly_GISAcres` for geometry
  truth, `attr_IncidentSize` for reporting.
- MTBS: static `mtbs_perimeter_data.zip` (all perimeters, ~390 MB, daily) at
  `edcintl.cr.usgs.gov/downloads/sciweb1/shared/MTBS_Fire/data/composite_data/burned_area_extent_shapefile/`;
  per-fire dNBR thresholds via GeoServer WFS layer `mtbs:burn_severity_fire_polygons`
  (geometry is EPSG:3857 — a lon/lat BBOX() cql filter silently returns 0 rows);
  annual severity mosaics via WCS `edcintl.cr.usgs.gov/geoserver/mtbs/wcs`,
  coverageId `mtbs__mtbs_CONUS_<year>` (double underscore). Per-fire bundles
  (dnbr.tif): the legacy edcintl ZipServlet returns 503 — the WORKING route is
  the email queue, `mtbs.order_bundles()` (POST to
  burnseverity.cr.usgs.gov/downloads/addQueue.php with download_type
  "mapping_products", map_ids from a WFS lookup, product labels VERBATIM like
  "Continuous severity (i.e dnbr)", request_origin literally `'viewer'` with
  quotes; links arrive by email ~1 h, <=500 fires/request).
- NOAA Atlas 14 vol 1 grids: `hdsc.nws.noaa.gov/pub/hdsc/data/sw/sw{ARI}yr{DUR}ma[_ams].zip`
  (1000ths of inch; 1-yr exists as PDS only). NV = `sw`; CA sliver of pilot needs `ca`.
  Atlas 15: nothing covers NV yet (CONUS prelim ~Sept 2026, 1-h+ durations only).
- Planet Data API (planetscope.py, probed 2026-08-15): basic auth, key as user.
  Key resolution: `key=` arg → `$PL_API_KEY` → `~/.config/firescape/planet_api_key`
  (0600; NEVER in repo/Box/memory). Account is on the NASA CSDA IDIQ pool —
  quota effectively unbounded, but search first (free), order deliberately.
  Traps, all live-verified: **~30-day download embargo** (search/thumbnails
  immediate, `assets:download` appears ~day 30 — post-storm verification waits
  a month); `clear_percent` is SCENE-wide — rank scenes by
  `aoi_coverage()` (a 100%-clear scene covered only part of the Hidden Valley
  AOI; `mode="estimate"` is coarse triage — it said 8% where the delivered
  clip held 36.7% — use `mode="udm2"` for final selection);
  a `clear_percent` filter silently drops pre-Aug-2018 (pre-udm2) scenes —
  use `cloud_cover` for old baselines; udm2 shadow band is unreliable
  (published F1≈0.6) — do terrain shadows ourselves from the DEM. Orders
  vanish from listings after ~3 months; the `.provenance.json` sidecar
  (`stage_order`) is the durable record. Harmonize tool = PS2.SD/PSB.SD only.
- ScienceBase 403s plain fetches — use curl/requests with a browser User-Agent.
- BAER/SBS: NV fires are mostly BLM (ESR program, not USFS BAER) — query the
  burn-severity portal ImageServer by IRWIN ID; fallback = derive dNBR from
  Sentinel-2/Landsat with MTBS WFS threshold attributes.

## Conventions

- Mirror stormscape: AOI spec via `stormscape.aoi.load_aoi` (bbox/vector/shapely);
  result-dict contract (`fields/transform/crs/profile/meta`); outputs named
  `<key>_<field>.tif` + `<key>_aoi.geojson`; DEMs EPSG:5070; figures in auto-UTM;
  lazy-import heavy deps; offline tests only (synthetic fixtures).
- `firescape/` is library code; `scripts/` is the **as-run pipeline** that
  produced the products (see scripts/README.md). Drivers there are resumable
  and exit **42** to mean "budget hit, run me again"; the `.sh` wrappers loop
  on that. If a script starts getting imported, promote it into the package.
- Figures go through `firescape.plotting` (house style: geographic axes, data
  alpha ≤ 0.6, no graticule, colourblind-safe ramps, PDF only since 2026-08-14) and
  `firescape.relief` (hillshade). Never restyle a figure in a one-off script.
- **Captions, not titles** (since 2026-08-27): no `fig.suptitle` on a deliverable.
  Panel titles stay to one short line; the run, the numbers, the provenance and
  the caveats go in `plotting.caption(fig, text, label="Stallion fire.")`, which
  wraps to the panel span, sets wide sheets in columns, and reserves its own
  space. Scratch/diagnostic plots may still be titled.
- Calibration values are TOML **files** (firescape/data/calibration/), never
  Python constants; adopted calibrations get committed. Region polygons are
  GeoJSON keyed by `region` name; TOML sections use the same names.
- USGS standard: I15 = 24 mm/h reference storm; basins 0.025–8 km²; filters per
  wildcat defaults (config.py). Rossi masks: valley (focal σ(elev) ≤ 5 m in
  200 m radius, polygons ≥ 1 km²), sink (flow-dir nulls), water (EVT open water
  + NHD waterbodies).
