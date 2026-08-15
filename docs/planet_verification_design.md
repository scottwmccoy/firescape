# Planet verification module — design

*2026-08-15. Synthesis of a four-thread research pass (HazMapper deep-read,
NASA SALaD audit, methods survey, Planet platform study); citations inline.
Acquisition layer already implemented in `firescape/planetscope.py`.*

## The question the module answers

The rest of firescape predicts where postfire debris flows should happen.
This module reads what a storm actually did — debris-flow **tracks** (1–3
PlanetScope pixels wide, sinuous, strictly drainage-following) and **fan
deposits** (lobate, bright, tens of m wide, below the confinement break) — so
predictions can be scored per basin, Dolan-inventory style.

**State of the art, bluntly:** nobody has published an automated pixel-level
PFDF track/deposit detector at 3 m. Mature practice is (i) landslide-*scar*
detection (SALaD-CD: PA 83 %/UA 66 %, Amatya et al. 2022; U-Nets: F1 ≈ 0.7,
Meena et al. 2023) and (ii) expert reach-scale classification of daily Planet
imagery — the Cavagnaro/McCoy Dolan inventory (USGS data release, 2025) being
the closest analog and our output-schema template. Every ingredient below is
published; their fusion with a flow-routing network is the module's (low-risk)
novelty, and our statewide DEM + pfdf segment network is the unfair advantage
no published system had.

## Primary pipeline — "stack–z–segment"

**1. Epochs, not scenes** (HazMapper/ALDI/Notti precedent; matches our own
finding that single-scene dNDVI is unusable). Pre-storm: median of ≥5 usable
scenes, 2–6 weeks. Post-storm: median of the first ≥3 usable scenes (≤2
weeks; daily cadence makes this easy). A second post-epoch at +4–8 weeks
separates persistent deposits from ephemeral wetness/ash. Per-pixel
usable-scene-count rasters ride along; <3 obs ⇒ low confidence.

**2. Radiometry & geometry** — the make-or-break layer at 3 m:
- Harmonized SR (Sentinel-2 target; already the default in `order_request`);
  probe **ARPS** (CESTEM cross-calibration + BRDF + topographic correction,
  4-band) as the stronger operational base if CSDA carries it.
- udm2 `clear` filter, then per-pixel temporal MAD outlier rejection within
  the epoch stack (udm2 is a filter, not a mask — shadow band F1 ≈ 0.6).
- **AROSICS sub-pixel co-registration** of every scene to one epoch reference
  (Scheffler et al. 2017; pip-installable). A 1-px misregistration writes
  false change along every contrast edge — fatal for 1–3-px tracks.
- Per-scene gain/offset normalization against the epoch median over a
  stable-terrain mask that **excludes the corridors** (so the target signal
  cannot be normalized away).
- DEM cast-shadow raster per scene sun geometry; shadowed pixels dropped
  before the median (we have `relief` + 10 m DEM; udm2 can't do this).
- Pre-2019 baselines (e.g., Montecito benchmark): Huang & Roy (2021)
  PS0/PS1 generation transforms; harmonize tool covers PS2.SD/PSB.SD only.

**3. Indices** (no SWIR on PlanetScope — dNBR family is off the table):
- **ΔBrightness leads** (TC-like VIS-NIR sum): fresh mineral sediment
  brightens in both regimes (SLIP, Fayne et al. 2019; Coluzzi et al. 2025).
- ΔMSAVI as the greenness channel in rangeland (NDVI is soil-dominated at
  0.1–0.25 baselines); ΔNDVI demoted to context inside burn scars.
- ΔRedness (red/green) as the third channel (ash→mineral discriminator;
  desert-proven with high commission — usable only under conditioning).
- 8-band extras (yellow, red-edge) logged as experimental covariates, never
  gating: they are SuperDove's noisiest bands (Tu et al. 2022).

**4. Statistics, not thresholds:** per-pixel robust z = Δindex / temporal MAD
of the pre-stack, floored per slope/aspect/severity stratum; ALDI-style
pre-vs-post significance test as confirmation (Milledge et al. 2022 — the
only formulation with demonstrated cross-event parameter transfer);
TRP-style persistence — a flagged object must reappear in ≥2 independent
post-storm scenes.

**5. Geomorphic conditioning — the contribution:** evaluate z only inside
DEM-derived corridors and decide per **pfdf segment**:
- Track corridor: HAND < ~10 m, buffer 1–3 px widening with flow
  accumulation. Fan zone: HAND < 2–5 m ∧ slope 2–15° ∧ below the confinement
  break, optionally weighted by a Laharz/Gorr-style modeled inundation prior.
- **Strict down-gradient connectivity** through the flow-routing graph:
  changed pixels count only in chains contiguous downstream. Grading, OHV,
  grazing and ag almost never satisfy this over hundreds of meters.
- Segment decision: corridor-mean z, fraction-of-length changed, downstream
  continuity votes. Integrating a marginal 1–3-px signal over a segment's
  few hundred corridor pixels is matched filtering along the network — the
  single highest-leverage design choice. Fan polygons grow from seed
  segments (min ~10 px, smooth-texture check).
- Side effect: ALDI's amalgamation problem dissolves — objects are indexed
  by segment, so drainages can't merge.

**6. Outputs** (`products/` conventions, versioned, `run_meta.json`):
per-segment response table (class ∈ {none, indeterminate, response–flood-like,
response–debris-flow-like}, z stats, scene counts, confidence); fan/track
polygons (GPKG) with evidence attributes; z + epoch-median rasters for audit.
Flood vs debris flow is a **geometry call** (fan deposit above confinement
break, width ≫ channel), not a spectral one — nobody separates them
spectrally at 3 m; final inventory keeps a human confirmation step.

## The two regimes

| | (a) unburned rangeland (Hidden Valley) | (b) inside fresh burn scars (the real target) |
|---|---|---|
| greenness | ΔMSAVI useful | suppressed & confounded — context only |
| lead signal | ΔBrightness + ΔMSAVI | **ΔBrightness + significance**, ΔRedness for ash→soil |
| baseline | 2–6 wk + same-calendar-window prior year (cheatgrass phenology) | tight 1–3 wk post-fire pre-storm (ash surface drifts on week scales, Lewis et al. 2021) |
| attribution | temporal only | **spatial control too**: z against burned non-corridor pixels of the same severity/aspect stratum — fire-wide ash dispersal and green-up cancel |
| validation | hand-mapped Hidden Valley tracks | segment-level precision/recall vs Dolan-style inventory |

## Fallback + viewer (same machinery, human decision)

If the automated detector underdelivers, the pipeline through step 4 still
runs; the decision moves to a person: rank pfdf segments by corridor-
integrated |z|, present top-N pre/post/Δ chips for rapid yes/no per reach —
a Dolan-style inventory in hours, identical output schema, so the variants
are interchangeable downstream. This *is* the manual-identification viewer:
local leafmap/COG stack (Planet delivers COGs; public-GEE distribution is
structurally unavailable for PlanetScope), HazMapper's UI grammar — linked
pre/post/Δ layers with opacity, click-to-inspect, re-window-and-recompute,
draw-to-GeoJSON — and a serialized JSON analysis state per run. HazMapper
itself is reimplemented from the CC-BY paper (rdNDVI floored at ε ≈ 0.05–0.1),
never ported: the GEE code is NCSU research-only. Planet XYZ tiles only for
free recon; never bake the key into shared HTML.

## Validation plan

Segment-level confusion matrix is the operational score (pixel F1 ≈ 0.7 is
the published ceiling — and human inter-mapper TPR spans 0.08–0.8, Milledge
et al. 2022, so never tune to one hand map). Benchmarks: Hidden Valley
hand-map (regime a, tuning); Montecito 2018 field polygons (Kean et al.
2019, external fan-inundation benchmark, PS0-era transforms required);
first NV burn+storm pair for regime (b).

## Build order

1. ~~Acquisition~~ (done: search/screen/order/download/verify/stage).
2. Stack engine: epoch composites — udm2+MAD masking, AROSICS, per-scene
   normalization, cast shadows, per-pixel counts.
3. Change engine: Δ-index rasters, robust z, significance, persistence.
4. Corridor conditioning: HAND/corridor rasters from unit DEMs, pfdf-segment
   integration, connectivity, per-segment table + fan polygons.
5. Triage viewer (fallback UI = manual-ID tool).
6. Validation: Hidden Valley event pair; then a burned pair; A/B the
   automated detector against the triage fallback.

Known constraints: ~30-day CSDA download embargo (verification runs ~a month
behind the storm; search/thumbnails immediate); scene selection by
`aoi_coverage()`, never scene-wide `clear_percent`; `clear_percent` filters
drop the pre-Aug-2018 archive (use `cloud_cover` there); SAR (Handwerger
2022, Sentinel-1) is a weather-independent adjunct hook, not a dependency;
U-Nets become viable later using the labels this pipeline generates.

## Load-bearing references

Amatya et al. 2021 (Eng. Geol. 282:106000) · Amatya et al. 2022 (Geosci.
Data J. 9:315, SALaD-CD) · Milledge et al. 2022 (NHESS 22:481, ALDI) ·
Scheip & Wegmann 2021 (NHESS 21:1495, HazMapper) · Fayne et al. 2019 (Earth
Interact. 23, SLIP/DRIP) · Notti et al. 2023 (NHESS 23:2625) · Meena et al.
2023 (ESSD 15:3283, HR-GLDD) · Kean et al. 2019 (Geosphere 15:1140,
Montecito) · Cavagnaro et al. 2025 (USGS data release + GRL 52 + JGR-ES 130,
Dolan) · Scheffler et al. 2017 (Remote Sens. 9:676, AROSICS) · Huang & Roy
2021 (Sci. Remote Sens. 3:100014) · Tu et al. 2022 (Int. J. Appl. Earth
Obs. 114) · Frazier & Hemingway 2021 (Remote Sens. 13:3930) · Lewis et al.
2021 (Fire 4:68, ash) · Rennó et al. 2008 / Nobre et al. 2011 (HAND) ·
Gallant & Dowling 2003 (MrVBF) · Iverson et al. 1998 (Laharz) · Horton et
al. 2013 (Flow-R) · Francini et al. 2020 (TRP persistence) · Staley et al.
2017 (M1) · Kean & Staley 2021 (Earth's Future 9). Note: Handwerger's
PlanetScope work is pixel-tracking of slow landslides (2025), not spectral
change — his rapid-detection line is Sentinel-1 SAR (NHESS 22:753).
