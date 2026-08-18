"""Bug + Stallion field package for CalTopo -> products/caltopo/bugstallion_recon_v1.

The reconnaissance trip checks a prediction that already has a date on it: the
12-14 Aug 2026 storm fell on both burn scars, and ``p8_fire_storm_response``
ran M1 with the severity that was observed and the rainfall that actually fell.
This packages that prediction for CalTopo on a phone.

Four vector layers per fire, each in its own **CalTopo folder** so it is one
checkbox instead of one per segment -- the failure mode of a plain GeoJSON
export, where 1,420 line segments arrive as 1,420 independently-toggled map
objects:

1. **Predicted DF network**, coloured by likelihood at the observed storm
   (``P_observed``), in the USGS likelihood classes so the colours mean the
   same thing they do on a published assessment.
2. **Basins** -- the modelled catchments. They extend past the perimeter where
   a catchment drains from unburned ground above it.
3. **Fire perimeter.**
4. **Recon skeleton** -- the same channels with the survey schema and
   ``df_observed = 0``, i.e. nothing confirmed yet. It carries ``Segment_ID``
   so field observations rejoin the model network and the statewide inventory
   (``inventory/segment_survey_v1``) by key rather than by geometry.

Rasters go the other way: CalTopo wants **EPSG:3857** and caps an upload at
**40 MB**, so the storm's peak I15 and its Atlas-14 anomaly are reprojected to
Web Mercator as both raw float (for analysis) and colorized RGBA with the dry
tail transparent (so it draws over the CalTopo basemap without the viewer
having to style a float band).

What the two fires say before anyone leaves: Stallion's network is mostly above
threshold at the observed rainfall and Bug's is mostly well below it. That is a
prediction, and the point of the trip is that it can be wrong.

The severity behind it is the weak link and the README says so: both fires run
on a CIMSS **BRISK** near-real-time composite from a single 14 Aug 2026 scene,
which is vegetation change rather than soil burn severity and postdates most of
the rain. Rerun when the BARC/MTBS product releases.
"""
import json
import shutil
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd
from stormscape import caltopo, export

from firescape import paths

STORM = Path("/Users/scottmccoy/Library/CloudStorage/Box-Box/SWMresearch/"
             "PostFireDebrisFlows/2026_Bug_Stalion/storms/"
             "composite_20260812-20260814")
STORM_KEY = "BSE3day"
STORM_WINDOW = "12-14 Aug 2026"
OUT = paths.products_dir("caltopo", "bugstallion_recon_v1")
FIRES = ["bug", "stallion"]

# USGS likelihood classes -- the breaks a published PFDF assessment uses, so a
# colour here means what it means on one of those maps.
P_BREAKS = (0.2, 0.4, 0.6, 0.8)
P_NAMES = ("< 20%", "20-40%", "40-60%", "60-80%", "≥ 80%")

# Attributes that ride into the field in each segment's info panel. CalTopo has
# no attribute table, so the description is the only channel.
SEG_FIELDS = ("Segment_ID", "P_observed", "i15_obs", "I15_50", "exceed_ratio",
              "P_24mmh", "V_24mmh", "H_24mmh", "Area_km2")
BASIN_FIELDS = ("Segment_ID", "Area_km2", "BurnRatio", "P_24mmh", "I15_50")
# Matches inventory/segment_survey_v1's survey_master schema, so a filled-in
# recon merges into the statewide inventory without a column rename.
SURVEY_FIELDS = ("df_observed", "df_type", "observer", "survey_date",
                 "imagery_src", "notes")

SIMPLIFY_M = 5.0        # well under a 10 m DEM cell; trims the file, not the line


def load(fire):
    """Segments (observed severity + observed storm), basins, perimeter, severity meta."""
    src = paths.products_dir("forecast", f"{fire}_observed")
    seg = gpd.read_file(src / f"{fire}_storm_response_observed.gpkg")
    basins = gpd.read_file(src / f"{fire.upper()}_basins.gpkg")
    perim = gpd.read_file(src / "_perimeter.geojson")
    sev = json.loads((src / "observed_severity_summary.json").read_text())
    # Segment_ID rides into the field as a label and in the description; as a
    # float it reads "6637.0", which is not a key anyone wants to type.
    for g in (seg, basins):
        if "Segment_ID" in g.columns:
            g["Segment_ID"] = g["Segment_ID"].astype("Int64")
    return seg, basins, perim, sev


def basins_in_perimeter(basins, perim):
    """Modelled catchments touching the burn.

    Kept whole rather than clipped: a basin cut at the perimeter is no longer a
    basin, and the area column would stop matching the model that used it.
    """
    p = perim.to_crs(basins.crs)
    idx = basins.sindex.query(p.geometry.union_all(), predicate="intersects")
    return basins.iloc[sorted(idx)].copy()


def skeleton(seg):
    """The recording template: same channels, survey schema, nothing observed."""
    sk = seg[["Segment_ID", "Area_km2", "Slope", "P_24mmh", "I15_50",
              "geometry"]].copy()
    sk["df_observed"] = 0          # 0 = not yet surveyed (1 = DF, -1 = absent)
    for c in ("df_type", "observer", "survey_date", "imagery_src", "notes"):
        sk[c] = ""
    return sk


def layers_for(fire, seg, basins, perim):
    """The four CalTopo folders for one fire, drawing order bottom-up."""
    title = fire.capitalize()
    colors = caltopo.classify(seg["P_observed"], P_BREAKS)
    net = seg.copy()
    net["class"] = [P_NAMES[min(int(np.searchsorted(P_BREAKS, v, "right")),
                                len(P_NAMES) - 1)]
                    if np.isfinite(v) else "no rainfall"
                    for v in seg["P_observed"].to_numpy(float)]
    return [
        caltopo.Layer(f"{title} — fire perimeter", perim, color="#000000",
                      width=3, opacity=0.9, pattern="solid",
                      description=f"{title} Fire perimeter (WFIGS)"),
        caltopo.Layer(f"{title} — modelled basins", basins, color="#6B4E9B",
                      width=1.5, opacity=0.7, fields=BASIN_FIELDS,
                      label="Segment_ID", simplify_m=SIMPLIFY_M,
                      description="Modelled catchment. Extends past the "
                                  "perimeter where it drains unburned ground."),
        caltopo.Layer(f"{title} — recon skeleton (df_observed = 0)",
                      skeleton(seg), color="#4D4D4D", width=2, opacity=0.55,
                      label="Segment_ID", simplify_m=SIMPLIFY_M,
                      fields=("Segment_ID",) + SURVEY_FIELDS, visible=False,
                      description="Recording template — nothing confirmed yet. "
                                  "Record against Segment_ID."),
        caltopo.Layer(f"{title} — predicted DF network ({STORM_WINDOW} storm)",
                      net, color=colors, width=4, opacity=0.95,
                      label="class", fields=SEG_FIELDS, simplify_m=SIMPLIFY_M,
                      description=f"M1 likelihood at the observed storm "
                                  f"({STORM_WINDOW}), observed severity."),
    ]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report, all_layers = {}, []

    for fire in FIRES:
        seg, basins, perim, sev = load(fire)
        bas = basins_in_perimeter(basins, perim)
        p = seg["P_observed"].to_numpy(float)
        print(f"{fire}: {len(seg):,} segments, {len(bas):,} of "
              f"{len(basins):,} basins touch the perimeter; P_observed median "
              f"{np.nanmedian(p):.3f}, {int(np.nansum(p >= 0.5)):,} at or "
              f"above 0.5", flush=True)

        lay = layers_for(fire, seg, bas, perim)
        all_layers += lay
        path = caltopo.write(lay, str(OUT / f"{fire}_recon_caltopo.geojson"))
        s = caltopo.summary(path)
        print(f"  -> {Path(path).name}  {s['bytes'] / 1e6:.1f} MB  "
              f"{len(s['folders'])} folders", flush=True)
        report[fire] = {"segments": int(len(seg)),
                        "basins": int(len(bas)),
                        "basins_modelled": int(len(basins)),
                        "severity_scenes": sev["scene_dates"],
                        "composite_age_days": sev["composite_age_days"],
                        "P_observed_median": round(float(np.nanmedian(p)), 4),
                        "segments_P_ge_0.5": int(np.nansum(p >= 0.5)),
                        "geojson_mb": round(s["bytes"] / 1e6, 2),
                        "folders": s["folders"]}

        # GPKG alongside: the CalTopo GeoJSON reopens as 3-D geometry (its
        # 4-element positions), which is fine for CalTopo and noise in QGIS.
        gpkg = OUT / f"{fire}_recon.gpkg"
        seg.to_file(gpkg, layer="predicted_network", driver="GPKG")
        bas.to_file(gpkg, layer="basins", driver="GPKG")
        perim.to_file(gpkg, layer="perimeter", driver="GPKG")
        skeleton(seg).to_file(gpkg, layer="recon_skeleton", driver="GPKG")

    both = caltopo.write(all_layers, str(OUT / "bug_stallion_recon_caltopo.geojson"))
    sb = caltopo.summary(both)
    print(f"\ncombined -> {Path(both).name}  {sb['bytes'] / 1e6:.1f} MB  "
          f"{len(sb['folders'])} folders", flush=True)

    # Rasters: EPSG:3857 (CalTopo's projection) raw float + colorized RGBA.
    print("\nreprojecting storm rasters to EPSG:3857...", flush=True)
    tifs = export.export_geotiffs(str(STORM), STORM_KEY, str(OUT),
                                  fields=("i15max", "anom_i15"),
                                  out_key="storm_20260812-14")
    raster_report = {}
    for t in sorted(tifs):
        mb = Path(t).stat().st_size / 1e6
        flag = "" if mb <= 40 else "  ** OVER CalTopo's 40 MB upload cap **"
        rel = str(Path(t).relative_to(OUT))       # export_geotiffs sorts into rasters/
        print(f"  {rel}  {mb:.2f} MB{flag}", flush=True)
        raster_report[rel] = round(mb, 3)
    over = [k for k, v in raster_report.items() if v > 40]

    sha = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]),
                          "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "firescape_git_sha": sha,
        "purpose": "Bug + Stallion reconnaissance field package for CalTopo",
        "storm_window": STORM_WINDOW,
        "severity": "observed (BRISK dNBR)",
        "rainfall": "observed (MRMS peak I15, 1 km)",
        "likelihood_breaks": list(P_BREAKS),
        "caltopo": {"vector_crs": "EPSG:4326", "raster_crs": "EPSG:3857",
                    "raster_size_cap_mb": 40,
                    "grouping": "one CalTopo Folder per layer (class=Folder + "
                                "folderId), so each layer is a single toggle"},
        "fires": report, "rasters_mb": raster_report,
        "rasters_over_cap": over,
        "combined_geojson_mb": round(sb["bytes"] / 1e6, 2),
    }
    (OUT / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    write_readme(report, raster_report)
    print(f"\nwrote {OUT}")
    if over:
        print(f"WARNING: {len(over)} raster(s) exceed CalTopo's 40 MB cap: {over}")


def write_readme(report, rasters):
    ages = {r["composite_age_days"] for r in report.values()}
    scenes = ", ".join(sorted({s for r in report.values()
                               for s in r["severity_scenes"]}))
    age = min(ages) if len(ages) == 1 else f"{min(ages)}-{max(ages)}"
    lines = [
        "# Bug + Stallion reconnaissance package (CalTopo)",
        "",
        f"Storm window **{STORM_WINDOW}**; observed burn severity (CIMSS BRISK "
        f"dNBR, scene **{scenes}**) and observed rainfall (MRMS peak "
        "15-minute intensity, 1 km).",
        "",
        "## Import into CalTopo",
        "",
        "**Vectors.** Import `bug_stallion_recon_caltopo.geojson` (both fires) "
        "or the per-fire file. Map objects sidebar → Import. Each layer arrives "
        "as **one folder with one checkbox** — not one checkbox per segment. "
        "Uncheck a folder to hide the whole layer.",
        "",
        "**Rasters.** EPSG:3857 GeoTIFFs, uploaded as a *map sheet* / custom "
        "layer (needs a Pro, Desktop or Team account). The **40 MB** cap and "
        "the 3857 projection are CalTopo's requirements, not ours — a file in "
        "any other projection asks to be aligned by hand.",
        "",
        "Use the `_rgb.tif` files on the map: they are the styled image with "
        "dry cells transparent. The plain `_3857.tif` files are raw float, for "
        "analysis rather than display.",
        "",
        "## Layers",
        "",
        "| Folder | What it is |",
        "|---|---|",
        "| *fire* — predicted DF network | Modelled channels coloured by "
        "likelihood at the observed storm. Classes "
        f"{', '.join(P_NAMES)}, blue → red. |",
        "| *fire* — modelled basins | Catchments feeding those channels. Every "
        "modelled basin touches the burn, so this is all of them; they extend "
        "past the perimeter where a catchment drains unburned ground above the "
        "fire. |",
        "| *fire* — fire perimeter | WFIGS perimeter. |",
        "| *fire* — recon skeleton | The same channels with `df_observed = 0` "
        "and the survey schema, **off by default**. The recording template. |",
        "",
        "Tap a segment for its attributes — likelihood, observed I15, the "
        "triggering threshold `I15_50`, the exceedance ratio and modelled "
        "volume all travel in the description.",
        "",
        "## Recording observations",
        "",
        "The skeleton carries `Segment_ID`, the join key back to the model "
        "network and to `inventory/segment_survey_v1/survey_master.gpkg`. "
        "Record by **Segment_ID**, not by geometry — CalTopo edits do not "
        "come back as attribute changes. Set `df_observed` to 1 (debris flow "
        "confirmed) or -1 (confirmed absent); 0 means not yet surveyed.",
        "",
        "## What the model predicts, before you look",
        "",
    ]
    for fire, r in report.items():
        lines.append(
            f"- **{fire.capitalize()}** — {r['segments']:,} segments, median "
            f"likelihood {r['P_observed_median']:.3f} at the observed storm, "
            f"**{r['segments_P_ge_0.5']:,} at or above 0.5**.")
    lines += [
        "",
        "That contrast is the testable part of the trip: the same models and "
        "the same storm split the two scars, so the fires check the prediction "
        "in opposite directions — Stallion for false positives, Bug for false "
        "negatives.",
        "",
        "## Caveats to carry",
        "",
        f"- **The severity is BRISK, not BARC.** Scene {scenes}, "
        f"**{age} day(s) old**. It measures vegetation change, not soil burn "
        "severity, which is what the USGS models are calibrated on. Supersede "
        "it when the BAER/MTBS product releases.",
        "- **The composite is still maturing, and it moves fast.** BRISK "
        "sharpens as Landsat/Sentinel-2 overpasses accumulate on top of the "
        "immediate GOES look; under about 14 days the pattern is reliable but "
        "the magnitude under-reads. Between the 14 Aug and 17 Aug composites "
        "the unburned fraction fell from ~0.67 to ~0.44 on both fires, the "
        "networks grew by a third to a half, and Stallion's count of segments "
        "at or above 0.5 went 615 → 893. **Re-pull before you leave** — "
        "`p9_fire_observed.py` then `p9_fire_observed_map.py` then this "
        "script — and expect the target list to have grown again.",
        "- **MRMS is 1 km.** Convective cores are often smaller than a grid "
        "cell, so peak intensity in a small basin can be higher than shown.",
        "- Likelihood is per segment at the observed storm; it is not a runout "
        "model and says nothing about where material stops or what it hits.",
        "",
        "## Files",
        "",
    ]
    lines.append("- `bug_stallion_recon_caltopo.geojson` — both fires, "
                 "8 folders")
    for fire, r in report.items():
        lines.append(f"- `{fire}_recon_caltopo.geojson` "
                     f"({r['geojson_mb']:.1f} MB) — CalTopo import, 4 folders")
        lines.append(f"- `{fire}_recon.gpkg` — the same layers for QGIS")
    for name, mb in sorted(rasters.items()):
        lines.append(f"- `{name}` ({mb:.2f} MB)")
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
