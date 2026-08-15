"""Historic MTBS fire perimeters over the statewide v0 hazard maps, plus a
ranked list of validation targets (fires containing today's moderate+ basins).
WGS84 axes, stormscape context, PNG+PDF.
"""
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

from datetime import date

import geopandas as gpd
import numpy as np
import pandas as pd
from pyogrio import read_dataframe

from firescape import plotting as mc
from firescape import mtbs, paths

OUT = paths.products_dir("prefire", "statewide_v0")
ERA_MIN = 2017

# ---- all historic MTBS fires >=10 km2 in the domain (cached) ---------------
cache = paths.interim_dir("statewide") / "mtbs_fires_all.geojson"
if cache.exists():
    hist = gpd.read_file(cache)
else:
    nv0 = gpd.read_file(paths.raw_dir("boundaries") / "nv_state.geojson")
    buf = gpd.GeoSeries(nv0.to_crs("EPSG:5070").buffer(15_000).union_all(),
                        crs="EPSG:5070").to_crs("EPSG:4326").iloc[0]
    frames = []
    for st in ("NV", "CA", "OR", "ID", "UT", "AZ"):
        g = mtbs.fire_records(event_id_like=f"{st}%", min_acres=2471)
        g = g[g.intersects(buf)]
        frames.append(g)
    hist = mtbs.dedupe_records(
        gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326"))
    hist["ig_year"] = hist["ig_date"].dt.year
    hist = hist[["event_id", "incid_name", "ig_year", "burnbndac", "geometry"]]
    hist.to_file(cache, driver="GeoJSON")
print(f"{len(hist)} historic fires >=10 km2 in domain")

era = hist[hist["ig_year"] >= ERA_MIN]
old = hist[hist["ig_year"] < ERA_MIN]

# ---- validation targets ----------------------------------------------------
tgt_csv = OUT / "validation_targets.csv"
if not tgt_csv.exists():
    basins = read_dataframe(OUT / "statewide_v0_basins.gpkg",
                            columns=["H_24mmh", "P_24mmh", "V_24mmh", "I15_50"])
    basins = basins.to_crs("EPSG:4326")
    modplus = basins[basins["H_24mmh"] >= 2].copy()
    modplus["geometry"] = modplus.geometry.centroid
    join = gpd.sjoin(modplus,
                     hist[["event_id", "incid_name", "ig_year", "geometry"]],
                     how="inner", predicate="within")
    tgt = (join.groupby("event_id")
           .agg(incid_name=("incid_name", "first"),
                ig_year=("ig_year", "first"), n_modplus=("H_24mmh", "size"),
                n_high=("H_24mmh", lambda s: int((s == 3).sum())),
                max_P=("P_24mmh", "max"), min_thresh_mmh=("I15_50", "min"))
           .reset_index()
           .merge(hist[["event_id", "burnbndac"]], on="event_id"))
    tgt["km2"] = (tgt["burnbndac"] * 0.00404686).round(1)
    tgt = tgt.drop(columns="burnbndac").sort_values(
        ["n_high", "n_modplus"], ascending=False)
    tgt.to_csv(tgt_csv, index=False)
    print(f"{len(tgt)} fires contain moderate+ basins")

# ---- map --------------------------------------------------------------------
nv = gpd.read_file(paths.raw_dir("boundaries") / "nv_state.geojson").to_crs("EPSG:4326")
cur = gpd.read_file(
    paths.raw_dir("perimeters") / f"wfigs_current_{date(2026, 8, 12):%Y%m%d}.geojson"
).to_crs("EPSG:4326")
full = read_dataframe(OUT / "statewide_v0_basins.gpkg",
                      columns=["H_24mmh", "P_24mmh", "P_RgtT"]).to_crs("EPSG:4326")

transform, shape, extent = mc.grid(mc.domain_bounds4326())
hs = mc.hillshade(transform, shape)
ctx = mc.fetch_context()

HAZ = ListedColormap(["#4B9B6E", "#E8A33D", "#C1272D"])
NORM = BoundaryNorm([0.5, 1.5, 2.5, 3.5], HAZ.N)
im_extent = (extent[0], extent[1], extent[2], extent[3])
panels = [("H_24mmh", None, None, "Combined hazard class"),
          ("P_24mmh", "magma", (0, float(np.nanpercentile(full["P_24mmh"], 99))),
           "Debris-flow likelihood (I15 = 24 mm/h)"),
          ("P_RgtT", "viridis", (0, float(np.nanpercentile(full["P_RgtT"], 99))),
           "P(R>T): annual chance of threshold rain")]

fig, axes = plt.subplots(1, 3, figsize=(20, 10), dpi=140)
for ax, (col, cmap, clim, title) in zip(axes, panels):
    ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=im_extent)
    arr = mc.burn(full, col, transform, shape)
    if col == "H_24mmh":
        ax.imshow(np.ma.masked_invalid(arr), cmap=HAZ, norm=NORM,
                  extent=im_extent, alpha=0.6, interpolation="nearest")
    else:
        im = ax.imshow(np.ma.masked_invalid(arr), cmap=cmap, vmin=clim[0],
                       vmax=clim[1], extent=im_extent, alpha=0.6,
                       interpolation="nearest")
        fig.colorbar(im, ax=ax, shrink=0.5, label=title)
    mc.draw_context(ax, ctx, label_cities=(col == "H_24mmh"),
                    label_rivers=(col == "H_24mmh"))
    old.boundary.plot(ax=ax, zorder=6.5, **mc.fire_style("historic"))
    era.boundary.plot(ax=ax, zorder=7, **mc.fire_style("calibration"))
    nv.boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=8)
    cur.boundary.plot(ax=ax, zorder=8.5, **mc.fire_style("current"))
    mc.style_axes(ax, extent)
    ax.set_title(title, fontsize=11)

import matplotlib.patheffects as pe

axes[0].legend(handles=[
    Line2D([0], [0], marker="s", color="none", markersize=10,
           markerfacecolor=c, label=l)
    for c, l in zip(HAZ.colors, ["low", "moderate", "high"])] + [
    Line2D([0], [0], color="#333333", lw=0.9, label=f"MTBS fire pre-{ERA_MIN}"),
    Line2D([0], [0], color="white", lw=1.6,
           path_effects=[pe.withStroke(linewidth=2.6, foreground="black")],
           label=f"MTBS fire {ERA_MIN}+ (calibration set)"),
    Line2D([0], [0], color="#56B4E9", lw=1.6,
           path_effects=[pe.withStroke(linewidth=2.6, foreground="black")],
           label="burning now (Bug, Stallion)"),
    Line2D([0], [0], color="0.35", lw=0.7, label="county"),
    Line2D([0], [0], color="0.1", lw=1.0, label="primary road"),
    Line2D([0], [0], color="#2c7fb8", lw=1.0, label="major river")],
    loc="lower left", fontsize=7, title="Hazard / fire history / context",
    title_fontsize=7.5)

fig.suptitle(
    "firescape statewide v0 + MTBS fire history — where past fires meet "
    "today's pre-fire hazard\n"
    f"{len(full):,} basins · historic fires >=10 km2 ({len(old)} pre-{ERA_MIN}, "
    f"{len(era)} {ERA_MIN}+) · ranked search list: validation_targets.csv",
    y=0.99, fontsize=13)
fig.tight_layout()
mc.save(fig, "statewide_v0_hazard_fires")
