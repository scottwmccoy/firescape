"""Fire-scale hindcast figure: observed-severity hazard for one MTBS fire.

Usage: python p2_hindcast_fig.py <EVENT_ID> "<Fire name, year>"
Panel A: observed BARC severity + stream segments colored by M1 likelihood.
Panel B: triggering threshold I15 per segment (low = dangerous), high-hazard
basin outlets flagged. WGS84 axes, roads for access, PNG 300 dpi + PDF.
"""
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import rasterio
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, LightSource, ListedColormap, Normalize
from matplotlib.lines import Line2D
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds

from firescape import mtbs, paths, statewide

EID = sys.argv[1] if len(sys.argv) > 1 else "NV3888711544420240730"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "Broom Canyon, 2024"
PAD = 0.04
RES = 0.0002                                   # ~20 m display grid

d = paths.products_dir("assess", EID)
segs = gpd.read_file(d / f"{EID}_segments.gpkg").to_crs("EPSG:4326")
basins = gpd.read_file(d / f"{EID}_basins.gpkg").to_crs("EPSG:4326")
bundle = mtbs.fire_bundle(EID)
perim = gpd.read_file(bundle["burn_area"]).to_crs("EPSG:4326")

w, s, e, n = perim.total_bounds
w, s, e, n = w - PAD, s - PAD, e + PAD, n + PAD
W, H = int((e - w) / RES), int((n - s) / RES)
tr = from_origin(w, n, RES, RES)
extent = (w, e, s, n)

# hillshade: metric mosaic -> shade -> warp to the display grid
w5, s5, e5, n5 = transform_bounds("EPSG:4326", "EPSG:5070", w, s, e, n)
out = statewide._mosaic_to_grid(
    sorted((paths.cache_root() / "3dep_tiles").glob("USGS_13_*.tif")),
    (w5, s5, e5, n5), resolution=20.0, resampling="bilinear")
dem5, tr5 = out
hs5 = LightSource(azdeg=315, altdeg=45).hillshade(
    np.nan_to_num(dem5, nan=float(np.nanmedian(dem5))), vert_exag=1.3,
    dx=20.0, dy=20.0).astype("float32")
hs = np.full((H, W), 0.5, dtype="float32")
reproject(hs5, hs, src_transform=tr5, src_crs="EPSG:5070", dst_transform=tr,
          dst_crs="EPSG:4326", resampling=Resampling.bilinear)

# observed BARC on the display grid (dnbr6 when present, else estimated from
# dNBR with the standard 125/250/500 breaks — the same run_observed fallback)
if "dnbr6" in bundle:
    with rasterio.open(bundle["dnbr6"]) as ds:
        barc_src = mtbs.dnbr6_to_barc4(ds.read(1)).astype("float32")
        src_tr, src_crs, src_nod = ds.transform, ds.crs, 0
else:
    from firescape import severity as fsev

    with rasterio.open(bundle["dnbr"]) as ds:
        dn = ds.read(1).astype("float64")
        dn[dn == ds.nodata] = np.nan
        barc_src = fsev.classify_barc4(dn, (125.0, 250.0, 500.0)).astype("float32")
        src_tr, src_crs, src_nod = ds.transform, ds.crs, 0
barc = np.zeros((H, W), dtype="float32")
reproject(barc_src, barc, src_transform=src_tr, src_crs=src_crs,
          src_nodata=src_nod, dst_transform=tr, dst_crs="EPSG:4326",
          dst_nodata=0, resampling=Resampling.nearest)
barc[barc == 0] = np.nan
# severity is only meaningful inside the mapped perimeter
from rasterio.features import rasterize as rio_rasterize

inperim = rio_rasterize(((g, 1) for g in perim.geometry), out_shape=(H, W),
                        transform=tr, fill=0, dtype="uint8").astype(bool)
barc[~inperim] = np.nan

# stormscape reference layers (local roads + GNIS places), cached per fire;
# streams are skipped — the model segments ARE the drainage network here
from stormscape import refdata
from stormscape import plot as ssplot

roads_f, places_f = d / "_context_roads.geojson", d / "_context_places.geojson"
if not roads_f.exists():
    refdata.roads((w, s, e, n), local=True).to_file(roads_f, driver="GeoJSON")
if not places_f.exists():
    p = refdata.places((w, s, e, n))
    (p if len(p) else p.assign(name=None)).to_file(places_f, driver="GeoJSON")
roads = gpd.read_file(roads_f)
places = gpd.read_file(places_f)

BARC_CMAP = ListedColormap(["#009E73", "#F0E442", "#E69F00", "#D55E00"])
BARC_NORM = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], BARC_CMAP.N)

fig, axes = plt.subplots(1, 2, figsize=(15.5, 8.2), dpi=140)
ref_handles = []
for ax in axes:
    ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=extent)
    ref_handles = ssplot.add_reference(ax, "EPSG:4326", roads=roads,
                                       places=places, label=True)
    perim.boundary.plot(ax=ax, color="black", linewidth=1.4, zorder=7)

ax = axes[0]
ax.imshow(np.ma.masked_invalid(barc), cmap=BARC_CMAP, norm=BARC_NORM,
          extent=extent, alpha=0.45, interpolation="nearest")
pnorm = Normalize(0, 1)
segs.plot(ax=ax, column="P_24mmh", cmap="magma", norm=pnorm, linewidth=1.7,
          zorder=5)
fig.colorbar(ScalarMappable(norm=pnorm, cmap="magma"), ax=ax, shrink=0.65,
             label="debris-flow likelihood (I15 = 24 mm/h)")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.55, label=l)
                   for c, l in zip(BARC_CMAP.colors,
                                   ["unburned/low", "low", "moderate", "high"])],
          title="observed BARC (MTBS)", loc="lower left", fontsize=7,
          title_fontsize=7.5)
ax.set_title("Observed burn severity + M1 likelihood per stream segment",
             fontsize=10.5)

ax = axes[1]
tnorm = Normalize(10, 45)
segs.plot(ax=ax, column="I15_50", cmap="viridis_r", norm=tnorm, linewidth=1.7,
          zorder=5)
fig.colorbar(ScalarMappable(norm=tnorm, cmap="viridis_r"), ax=ax, shrink=0.65,
             label="15-min triggering intensity at P=0.5 (mm/h)")
hi = basins[basins["H_24mmh"] == 3]
handles2 = list(ref_handles)
if len(hi):
    pts = hi.geometry.representative_point()
    ax.scatter(pts.x, pts.y, s=70, marker="o", facecolor="none",
               edgecolor="#C1272D", linewidth=2.0, zorder=8)
    handles2.append(Line2D([], [], marker="o", color="none", mec="#C1272D",
                           mew=2.0, markersize=8,
                           label=f"high-hazard basin (n={len(hi)})"))
if handles2:
    ax.legend(handles=handles2, loc="lower left", fontsize=7)
ax.set_title("Triggering rainfall threshold (yellow = most easily triggered)",
             fontsize=10.5)

for ax in axes:
    ax.set_xlim(w, e)
    ax.set_ylim(s, n)
    ax.set_aspect(1.0 / np.cos(np.radians((s + n) / 2)))
    xt = np.arange(np.ceil(w / 0.05) * 0.05, e, 0.05)
    yt = np.arange(np.ceil(s / 0.05) * 0.05, n, 0.05)
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.set_xticklabels([f"{abs(x):.2f}°W" for x in xt], fontsize=7)
    ax.set_yticklabels([f"{y:.2f}°N" for y in yt], fontsize=7)
    ax.tick_params(length=2.5)

fig.suptitle(
    f"firescape hindcast — {LABEL} ({EID})\n"
    "hazard chain driven by OBSERVED MTBS severity: where debris flows "
    "should have initiated, given how the fire actually burned",
    y=1.0, fontsize=12)
fig.tight_layout()
stem = ("hindcast_"
        + LABEL.replace(",", "").strip().lower().replace(" ", "_")
        .replace("(", "").replace(")", ""))
out = paths.figures_dir() / f"{stem}.pdf"     # PDF only -- see plotting.save
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
