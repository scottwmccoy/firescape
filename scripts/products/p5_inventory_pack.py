"""Systematic-inventory survey package: the PFDF segment network, tiled.

Outputs (products/inventory/segment_survey_v1/):
  survey_master.gpkg          every statewide segment; seg_uid; df_observed=0
                              + observer fields; P/Hr/Vranges/I15 attributes
  kmz/index.kml               regionated NetworkLink index (open THIS in GE)
  kmz/<HU8 name>/<huc10>_<name>.kmz    one tile per HU10
  kmz/hot_segments.kmz        statewide highest-likelihood layer (first pass)
  kmz/fires_context.kmz       805 historic MTBS perimeters by era
  survey_units.csv            per-unit stats + priority rank
  README.md                   survey workflow

Env: FIRESCAPE_INV_SRC (default statewide_v1_1), FIRESCAPE_INV_UNITS
(comma list => pilot mode, skips master/hot/fires).
"""
import os
import re
import warnings
import zipfile
from pathlib import Path

warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine

from firescape import paths
from firescape import hazard as hz

SRC = paths.products_dir("prefire",
                         os.environ.get("FIRESCAPE_INV_SRC", "statewide_v1_1"))
ONLY = [u for u in os.environ.get("FIRESCAPE_INV_UNITS", "").split(",") if u]
OUT = paths.products_dir("inventory", "segment_survey_v1")
KMZ = OUT / "kmz"
KMZ.mkdir(parents=True, exist_ok=True)

BINS = [0.0, 0.15, 0.25, 0.35, 0.50, 0.65, 1.001]
HEX = ["000004", "3b0f70", "8c2981", "de4968", "fe9f6d", "fcfdbf"]
WIDTH = [1.4, 1.6, 1.8, 2.2, 2.6, 3.0]


def kmlcolor(hexrgb, alpha="e6"):
    r, g, b = hexrgb[0:2], hexrgb[2:4], hexrgb[4:6]
    return alpha + b + g + r


STYLES = "\n".join(
    f'<Style id="p{i}"><LineStyle><color>{kmlcolor(h)}</color>'
    f"<width>{w}</width></LineStyle></Style>"
    for i, (h, w) in enumerate(zip(HEX, WIDTH)))


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def seg_kml(gdf, title):
    ply = []
    for _, r in gdf.iterrows():
        b = int(np.clip(np.digitize(r["P_24mmh"], BINS) - 1, 0, 5)) \
            if np.isfinite(r["P_24mmh"]) else 0
        coords = " ".join(f"{x:.5f},{y:.5f},0" for x, y in r.geometry.coords)
        ext = "".join(
            f'<Data name="{k}"><value>{esc(v)}</value></Data>'
            for k, v in (("seg_uid", r["seg_uid"]),
                         ("P_likelihood", f"{r['P_24mmh']:.3f}"),
                         ("hazard", int(r["Hr"]) if np.isfinite(r["Hr"]) else "-"),
                         ("V_ranges_m3", int(r["Vr"]) if np.isfinite(r["Vr"]) else "-"),
                         ("I15_thresh_mmh", f"{r['I15_50']:.1f}"),
                         ("df_observed", "false — record hits in survey_master.gpkg")))
        ply.append(
            f'<Placemark><name>{r["seg_uid"]}</name><styleUrl>#p{b}</styleUrl>'
            f"<ExtendedData>{ext}</ExtendedData>"
            f"<LineString><tessellate>1</tessellate>"
            f"<coordinates>{coords}</coordinates></LineString></Placemark>")
    return ('<?xml version="1.0" encoding="UTF-8"?>'
            '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
            f"<name>{esc(title)}</name>{STYLES}"
            + "\n".join(ply) + "</Document></kml>")


def write_kmz(path, kml):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", kml)


# ---- lookups ----------------------------------------------------------------
hu10 = gpd.read_file(paths.interim_dir("statewide") / "nv_hu10.geojson")
hu10["huc10"] = hu10["huc10"].astype(str)
unit_name = dict(zip(hu10["huc10"], hu10["name"].astype(str)))
regions = pd.read_csv(paths.interim_dir("statewide") / "hu10_regions.csv",
                      dtype={"huc10": str})
region_of = dict(zip(regions["huc10"], regions["region"]))
hu8 = gpd.read_file(paths.interim_dir("statewide") / "nv_hu8.geojson")
hu8_name = dict(zip(hu8["huc8"].astype(str), hu8["name"].astype(str)))

with rasterio.open(paths.interim_dir("statewide") / "pga25_g.tif") as pg:
    PGA, PTR = pg.read(1), pg.transform
z14 = np.load(paths.interim_dir("statewide") / "atlas14_i15.npz")
I1G, TR14 = z14["i1"], Affine(*z14["transform"])


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:40]


files = sorted(SRC.glob("*_segments.gpkg"))
if ONLY:
    files = [f for f in files if f.name.split("_")[0] in ONLY]
print(f"{len(files)} unit segment files from {SRC.name}")

master_parts, tiles, stats = [], [], []
for f in files:
    huc = f.name.split("_")[0]
    g = gpd.read_file(f)
    if not len(g):
        continue
    g5070 = g.to_crs("EPSG:5070")
    cent = g5070.geometry.centroid
    # RANGES per segment (severity-independent): PGA + Atlas-14 anomaly at
    # segment centroids; SlopeDeg/FracNorth emitted by the v1.1 fleet
    cc = ((cent.x - PTR.c) / PTR.a).astype(int).to_numpy()
    rr = ((cent.y - PTR.f) / PTR.e).astype(int).to_numpy()
    ok = (rr >= 0) & (rr < PGA.shape[0]) & (cc >= 0) & (cc < PGA.shape[1])
    pga = np.full(len(g), np.nan)
    pga[ok] = PGA[rr[ok], cc[ok]]
    c4326 = gpd.GeoSeries(cent, crs="EPSG:5070").to_crs("EPSG:4269")
    c14 = ((c4326.x - TR14.c) / TR14.a).astype(int).to_numpy()
    r14 = ((c4326.y - TR14.f) / TR14.e).astype(int).to_numpy()
    ok14 = (r14 >= 0) & (r14 < I1G.shape[0]) & (c14 >= 0) & (c14 < I1G.shape[1])
    i1 = np.full(len(g), np.nan)
    i1[ok14] = I1G[r14[ok14], c14[ok14]]
    if "SlopeDeg" in g.columns:
        Vr, _, _ = hz.volume_ranges(g["Area_km2"].to_numpy(),
                                    g["SlopeDeg"].to_numpy(), 24.0 / i1, pga,
                                    g["FracNorth"].to_numpy())
    else:                       # pilot mode against v1 files
        Vr = np.full(len(g), np.nan)
    g["Vr"] = Vr
    g["Hr"] = hz.combined_c10(g["P_24mmh"].to_numpy(), Vr)
    g["huc10"] = huc
    g["unit_name"] = unit_name.get(huc, "")
    g["region"] = region_of.get(huc, "")
    g["seg_uid"] = huc + "-" + g["Segment_ID"].astype(int).astype(str)
    g["df_observed"] = np.int16(0)
    for c in ("df_type", "observer", "survey_date", "imagery_src", "notes"):
        g[c] = ""

    h8 = huc[:8]
    folder = f"{h8}_{slug(hu8_name.get(h8, 'basin'))}"
    d = KMZ / folder
    d.mkdir(parents=True, exist_ok=True)
    disp = g.copy()
    disp["geometry"] = g5070.geometry.simplify(20.0)
    disp = disp.set_geometry("geometry", crs="EPSG:5070").to_crs("EPSG:4326")
    tile = d / f"{huc}_{slug(unit_name.get(huc, ''))}.kmz"
    write_kmz(tile, seg_kml(disp, f"{huc} {unit_name.get(huc, '')}"))
    w, s, e, n = disp.total_bounds
    nhot = int((g["P_24mmh"] >= 0.5).sum())
    tiles.append(dict(huc10=huc, name=unit_name.get(huc, ""), folder=folder,
                      file=f"{folder}/{tile.name}", w=w, s=s, e=e, n=n,
                      n_seg=len(g), maxP=float(g["P_24mmh"].max()),
                      n_hot=nhot))
    stats.append(dict(huc10=huc, unit_name=unit_name.get(huc, ""),
                      hu8=h8, hu8_name=hu8_name.get(h8, ""),
                      region=region_of.get(huc, ""), n_seg=len(g),
                      mean_P=float(g["P_24mmh"].mean()),
                      max_P=float(g["P_24mmh"].max()), n_P50=nhot,
                      n_haz2=int((g["Hr"] >= 2).sum()),
                      kmz=f"{folder}/{tile.name}"))
    master_parts.append(g.to_crs("EPSG:4326"))
    if len(master_parts) % 50 == 0:
        print(f"  {len(master_parts)}/{len(files)}", flush=True)

# ---- index.kml --------------------------------------------------------------
tl = pd.DataFrame(tiles)
folders = []
for folder, sub in tl.groupby("folder"):
    links = "\n".join(
        f"<NetworkLink><name>{esc(t.huc10)} {esc(t.name)} "
        f"(n={t.n_seg:,}, maxP={t.maxP:.2f})</name>"
        f"<Region><LatLonAltBox><north>{t.n:.4f}</north><south>{t.s:.4f}</south>"
        f"<east>{t.e:.4f}</east><west>{t.w:.4f}</west></LatLonAltBox>"
        f"<Lod><minLodPixels>160</minLodPixels><maxLodPixels>-1</maxLodPixels></Lod></Region>"
        f"<Link><href>{t.file}</href></Link></NetworkLink>"
        for t in sub.itertuples())
    folders.append(f"<Folder><name>{esc(folder[9:].replace('_', ' '))} "
                   f"({folder[:8]})</name>{links}</Folder>")
index = ('<?xml version="1.0" encoding="UTF-8"?>'
         '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
         "<name>firescape segment survey — Nevada</name>"
         "<description>Open this file; tiles stream in as you zoom. Color = "
         "debris-flow likelihood (v1.1 dispersed calibration). Record "
         "observations in survey_master.gpkg by seg_uid.</description>"
         + "\n".join(folders) + "</Document></kml>")
(KMZ / "index.kml").write_text(index)
print(f"index.kml over {len(tl)} tiles in {tl['folder'].nunique()} HU8 folders")

pd.DataFrame(stats).sort_values(["n_P50", "max_P"], ascending=False) \
    .assign(priority=lambda d: np.arange(1, len(d) + 1)) \
    .to_csv(OUT / "survey_units.csv", index=False)

if not ONLY:
    master = gpd.GeoDataFrame(pd.concat(master_parts, ignore_index=True),
                              geometry="geometry", crs="EPSG:4326")
    print(f"master: {len(master):,} segments")
    keep = ["seg_uid", "huc10", "unit_name", "region", "Segment_ID",
            "Area_km2", "Slope", "P_24mmh", "Hr", "Vr", "I15_50",
            "SlopeDeg", "FracNorth", "df_observed", "df_type", "observer",
            "survey_date", "imagery_src", "notes", "geometry"]
    master = master[[c for c in keep if c in master.columns]]
    master.to_file(OUT / "survey_master.gpkg", driver="GPKG",
                   layer="segments")
    hot = master[master["P_24mmh"] >= 0.5]
    thr = 0.5
    while len(hot) > 40000:
        thr += 0.05
        hot = master[master["P_24mmh"] >= thr]
    hot5070 = hot.to_crs("EPSG:5070")
    hd = hot.copy()
    hd["geometry"] = hot5070.geometry.simplify(20.0)
    hd = hd.set_geometry("geometry", crs="EPSG:5070").to_crs("EPSG:4326")
    write_kmz(KMZ / "hot_segments.kmz",
              seg_kml(hd, f"hot segments (P >= {thr:.2f}, n={len(hd):,})"))
    print(f"hot layer: {len(hd):,} segments at P >= {thr:.2f}")

    fires = gpd.read_file(paths.interim_dir("statewide")
                          / "mtbs_fires_all.geojson").to_crs("EPSG:5070")
    fires["geometry"] = fires.geometry.simplify(100.0)
    fires = fires.to_crs("EPSG:4326")
    era_style = {0: ("99777777", 1.2), 1: ("cc7777ff", 1.6), 2: ("ccffffff", 2.0)}
    pl = []
    for _, r in fires.iterrows():
        era = 0 if r["ig_year"] < 2000 else (1 if r["ig_year"] < 2017 else 2)
        col, wd = era_style[era]
        try:
            geoms = r.geometry.geoms
        except AttributeError:
            geoms = [r.geometry]
        rings = []
        for gm in geoms:
            coords = " ".join(f"{x:.4f},{y:.4f},0" for x, y in gm.exterior.coords)
            rings.append(f"<Polygon><outerBoundaryIs><LinearRing>"
                         f"<coordinates>{coords}</coordinates>"
                         f"</LinearRing></outerBoundaryIs></Polygon>")
        pl.append(f"<Placemark><name>{esc(r['incid_name'])} {int(r['ig_year'])}"
                  f"</name><Style><LineStyle><color>{col}</color>"
                  f"<width>{wd}</width></LineStyle><PolyStyle><fill>0</fill>"
                  f"</PolyStyle></Style><MultiGeometry>{''.join(rings)}"
                  f"</MultiGeometry></Placemark>")
    write_kmz(KMZ / "fires_context.kmz",
              '<?xml version="1.0" encoding="UTF-8"?>'
              '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
              "<name>MTBS fires 1984-2024 (gray pre-2000, red 2000-2016, "
              "white 2017+)</name>" + "\n".join(pl) + "</Document></kml>")
    print("fires_context.kmz written")

    (OUT / "README.md").write_text(f"""# firescape segment survey package (v1.1)

**Start with `SURVEY_GUIDE.md`** — the full workflow for the mapping team.

Systematic debris-flow inventory over the PFDF stream network
({len(master):,} segments, 586 HU10 units; playas/valley floors excluded by
the Rossi masks).

## Workflow
1. **Find** (Google Earth): open `kmz/index.kml` — tiles stream in as you
   zoom; color = simulated debris-flow likelihood (dispersed v1.1
   calibration). Start with `kmz/hot_segments.kmz` and the top rows of
   `survey_units.csv`; `kmz/fires_context.kmz` shows where fires have been.
2. **Record** (QGIS): edit `survey_master.gpkg` (layer `segments`):
   set `df_observed` = 1/-1 (confirmed / confirmed-absent), `df_type`
   (df, flood, uncertain), `observer`, `survey_date`, `imagery_src`
   (GE date / Planet scene), `notes`. Join key = `seg_uid`
   (= huc10-SegmentID), shown on every KML feature.
3. Never edit geometry — the network must stay aligned with the model
   products. Attribute edits only.

Columns: P_24mmh (M1 likelihood, I15=24 mm/h), Hr (Cannon class with RANGES
volume), Vr (RANGES volume m3), I15_50 (triggering threshold mm/h).
""")
print("README written")
print("\nDONE ->", OUT)
