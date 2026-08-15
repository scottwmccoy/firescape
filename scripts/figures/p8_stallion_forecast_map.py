"""Stallion fire: segment-level triggering rainfall intensity and likelihood."""
import json
import math
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

import geopandas as gpd
from firescape import paths, plotting as mc, relief

FIRE = sys.argv[1] if len(sys.argv) > 1 else "Stallion"
CAL = sys.argv[2] if len(sys.argv) > 2 else "statewide_v1_1"
SRC = paths.products_dir("forecast", f"{FIRE.lower()}_{CAL}")
meta = json.loads((SRC / "forecast_summary.json").read_text())

seg = gpd.read_file(SRC / f"{FIRE.lower()}_segments.gpkg").to_crs("EPSG:4326")
thr_col = "I15_50" if "I15_50" in seg.columns else "thresh_i15"
per = gpd.read_file(sorted(paths.raw_dir("perimeters").glob("wfigs_current_*.geojson"))[-1])
name_col = next(c for c in per.columns if c.endswith("IncidentName"))
per = per[per[name_col].str.fullmatch(FIRE, case=False, na=False)].to_crs("EPSG:4326")

w, s, e, n = per.total_bounds
pad = 0.035
bounds = (w - pad, s - pad, e + pad, n + pad)
tr, shape, extent = mc.grid(bounds, res=0.0002)          # ~20 m display grid
print(f"display grid {shape[1]}x{shape[0]}", flush=True)
hs = relief.shaded_relief(
    sorted((paths.cache_root() / "3dep_tiles").glob("USGS_13_*.tif")),
    tr, shape, crs="EPSG:4326", shade_res_m=20.0)
im_extent = (extent[0], extent[1], extent[2], extent[3])

try:
    from stormscape import plot as ssplot
    from stormscape import refdata
    roads = refdata.roads(bounds)
    places = refdata.places(bounds)
except Exception as exc:                                   # context is a bonus
    print(f"context unavailable: {type(exc).__name__}: {exc}", flush=True)
    roads = places = None

thr = seg[thr_col].to_numpy()
lo, hi = np.percentile(thr[np.isfinite(thr)], [2, 98])

HAZ = ListedColormap(["#4B9B6E", "#E8A33D", "#C1272D"])
NORM = BoundaryNorm([0.5, 1.5, 2.5, 3.5], HAZ.N)

fig, axes = plt.subplots(1, 2, figsize=(19, 10.5), dpi=140)
for ax, mode in zip(axes, ("threshold", "hazard")):
    ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=im_extent, zorder=0)
    if roads is not None and len(roads):
        roads.to_crs("EPSG:4326").plot(ax=ax, color="0.25", linewidth=0.5,
                                       alpha=0.7, zorder=2)
    per.boundary.plot(ax=ax, color="#56B4E9", linewidth=1.8, zorder=6,
                      path_effects=mc.fire_style("current")["path_effects"])
    if mode == "threshold":
        im = seg.plot(ax=ax, column=thr_col, cmap="plasma_r", linewidth=1.5,
                      vmin=lo, vmax=hi, zorder=5, legend=True,
                      legend_kwds={"shrink": 0.5, "pad": 0.02,
                                   "label": "triggering $I_{15}$ (mm/h) "
                                            "at 50% likelihood"})
        ax.set_title("Rainfall intensity that triggers a debris flow\n"
                     "lower = responds to a smaller storm", fontsize=10.5)
    else:
        seg.plot(ax=ax, column="H_24mmh", cmap=HAZ, norm=NORM, linewidth=1.5,
                 zorder=5)
        ax.legend(handles=[Line2D([0], [0], color=c, lw=3, label=l)
                           for c, l in zip(HAZ.colors,
                                           ["low", "moderate", "high"])],
                  title="Hazard class", loc="lower left", fontsize=8,
                  title_fontsize=8)
        ax.set_title("Combined hazard class at the 24 mm/h reference storm\n"
                     "(≈1-year, 15-minute intensity)", fontsize=10.5)
    if places is not None and len(places):
        from shapely.geometry import box as _box
        p = places.to_crs("EPSG:4326").copy()
        # GNIS returns MultiPoint for some records -- collapse to one point
        p["geometry"] = p.geometry.representative_point()
        p = p[p.geometry.within(_box(*bounds))]
        # GNIS carries many records per settlement; one label each, or they
        # stack into an unreadable smear
        if "name" in p.columns:
            p = p.drop_duplicates("name")
        for _, r in p.head(6).iterrows():
            ax.plot(r.geometry.x, r.geometry.y, "o", ms=4, color="black",
                    mec="white", zorder=9)
            ax.annotate(str(r.get("name", "")), (r.geometry.x, r.geometry.y),
                        xytext=(3, 3), textcoords="offset points", fontsize=7,
                        zorder=9.1,
                        path_effects=mc.fire_style("calibration")["path_effects"])
    mc.style_axes(ax, extent, step=0.1)

q = meta["triggering_i15_mmh"]
fig.suptitle(
    f"{FIRE} fire — pre-fire debris-flow forecast · {meta['acres']:,} acres, "
    f"perimeter {meta['perimeter_file'][14:22]} · {meta['segments']:,} segments\n"
    f"median triggering $I_{{15}}$ {q['median']} mm/h "
    f"(5–95%: {q['p05']}–{q['p95']}) · simulated severity, "
    f"{meta['region']} calibration (P$_{{dsim}}$={meta['pdsim']}) — "
    "no BARC product exists yet for an active fire",
    y=0.98, fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))
mc.save(fig, f"{FIRE.lower()}_forecast")
