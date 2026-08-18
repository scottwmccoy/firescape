"""District zooms for the AML exposure ranking -> figures/aml_zoom_<key>.pdf.

The companion to ``e3_aml_map`` in the same relationship ``p10_urban_zooms``
has to the statewide hazard sheet: the statewide map answers *where in Nevada*,
which is the wrong scale for anyone deciding what to visit. One district per
sheet, at a scale where the individual dump and the channel beside it can be
pointed at.

The windows are **derived from the ranking, not hand-drawn**. The top-ranked
sites are single-linkage clustered at 6 km, each cluster scored by its best
rank, and the six best rendered. That way the sheet set follows the data: if a
recalibration reorders the ranking, the districts that get drawn change with
it, and nothing here has to be edited.

Reading the sheet: named NHD water goes down first and the modelled
debris-flow channels over it, so the network is seen arriving at the water it
drains into. A site counts as near-channel when it lies within 30-55 m of a
modelled channel -- half a corridor 9-60 m wide that widens with contributing
area, plus 25 m for registration. That is proximity to the network, not a
runout model.

    python scripts/figures/e4_aml_zooms.py            # all six
    python scripts/figures/e4_aml_zooms.py storey_01  # just one
"""
import json
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterio.warp import transform_bounds
from scipy.cluster.hierarchy import fcluster, linkage
from stormscape.plot import Labeller, interior_point, shorten

from firescape import paths
from firescape import plotting as mc

plt.rcParams["image.cmap"] = "viridis"
V12 = paths.products_dir("prefire", "statewide_v1_2")
EXPO = paths.products_dir("exposure", "aml_v2")

NTOP, CLUSTER_KM, NSHEETS = 40, 6.0, 6
MIN_SPAN_KM, PAD_KM = 9.0, 1.5
P_FLOOR, P_CEIL = 0.05, 0.60      # network colour range at district scale
NPX, MIN_M_PER_PX = 1900, 8.0     # 3DEP is 10 m; do not pretend to finer
CROWD_PX = 26.0

sites = gpd.read_file(EXPO / "aml_exposure.gpkg")          # EPSG:5070
sites4326 = sites.to_crs(4326)
sites4326["rep"] = sites4326.geometry.representative_point()
receptors = gpd.read_file(paths.interim_dir("exposure") / "nhd_receptors.gpkg")
blm = gpd.read_file(paths.raw_dir("blm") / "nv_blm_sma.gpkg").to_crs(4326)
ctx = mc.fetch_context()
ubounds = json.loads(
    (paths.interim_dir("exposure") / "unit_bounds_v1_2.json").read_text())

#: Waters carrying a perennial reach anywhere in the state. Resolved once off
#: the whole receptor set: NHD splits a river into reaches with their own
#: FCodes, so a name classed from one window's subset would be drawn perennial
#: on one sheet and intermittent on the next.
PER_NAMES = set(receptors.loc[(receptors["src"] == "fl")
                              & (receptors["kind"] == "perennial stream"),
                              "name"].dropna())
vmax = float(np.nanpercentile(
    sites.loc[sites["exposed"].astype(bool), "P_annual_site"], 99))


def site_name(row):
    for v in (row.get("name"), row.get("ftr_type")):
        if isinstance(v, str) and v.strip():
            return v.strip()
    return "unnamed working"


def clusters():
    """The districts worth a sheet, best-ranked first."""
    top = sites.nsmallest(NTOP, "rank").copy()
    xy = np.c_[top.geometry.centroid.x, top.geometry.centroid.y]
    top["grp"] = (fcluster(linkage(xy, method="single"),
                           t=CLUSTER_KM * 1000, criterion="distance")
                  if len(top) > 1 else np.ones(len(top), int))
    out = []
    for _, grp in top.groupby("grp"):
        best = grp.loc[grp["rank"].idxmin()]
        county = best["county"] if isinstance(best["county"], str) else "Nevada"
        out.append({"best": best, "members": grp, "county": county,
                    "rank": int(best["rank"]),
                    "key": f"{county.split()[0].lower()}_{int(best['rank']):02d}"})
    out.sort(key=lambda c: c["rank"])
    return out[:NSHEETS]


def window_for(members):
    """Square-on-screen window around a cluster, floored at MIN_SPAN_KM."""
    b = members.to_crs(4326).total_bounds
    cy = (b[1] + b[3]) / 2.0
    pad = PAD_KM * 1000 / 111_320.0
    w, s, e, n = b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad
    min_deg = MIN_SPAN_KM * 1000 / (111_320.0 * np.cos(np.radians(cy)))
    if (e - w) < min_deg:                       # grow about the centre
        cx = (w + e) / 2.0
        w, e = cx - min_deg / 2, cx + min_deg / 2
    return mc.square_window((w, s, e, n))


def network_in(win):
    win5070 = transform_bounds("EPSG:4326", "EPSG:5070", *win)
    segs = []
    for huc, b in ubounds.items():
        if not (b[2] < win5070[0] or b[0] > win5070[2]
                or b[3] < win5070[1] or b[1] > win5070[3]):
            g = pyogrio.read_dataframe(V12 / f"{huc}_segments.gpkg",
                                       columns=["P_24mmh", "Area_km2"],
                                       bbox=tuple(win5070))
            if len(g):
                segs.append(g)
    if not segs:
        return None
    s = pd.concat(segs).to_crs(4326).explode(index_parts=False)
    s = s[s.geometry.type == "LineString"]
    return s[s["P_24mmh"] >= P_FLOOR].sort_values("P_24mmh")


def sheet(c):
    win = window_for(c["members"])
    span_m = (win[2] - win[0]) * 111_320 * np.cos(np.radians((win[1]+win[3])/2))
    npx = int(min(NPX, max(600, span_m / MIN_M_PER_PX)))
    tr, shape, extent = mc.grid(win, res=(win[2] - win[0]) / npx)

    PANEL, LEFT, RIGHT, TOPM, BOTM = 7.0, 0.55, 0.30, 0.86, 1.60
    FW, FH = LEFT + PANEL + RIGHT, TOPM + PANEL + BOTM
    fig = plt.figure(figsize=(FW, FH), dpi=150)
    ax = fig.add_axes([LEFT / FW, BOTM / FH, PANEL / FW, PANEL / FH])

    ax.imshow(mc.hillshade(tr, shape), cmap="gray", vmin=0, vmax=1,
              extent=extent, zorder=0)
    print(f"  {shape[1]}x{shape[0]} px, {mc.grid_res_m(tr, shape):.0f} m/px, "
          f"shaded at {mc.shade_resolution(tr, shape):.0f} m", flush=True)
    for layer, kw in ((blm.clip(tuple(win)),
                       dict(facecolor="#D9C98C", edgecolor="none", alpha=0.28,
                            zorder=1)),
                      (ctx["counties"].clip(tuple(win)),
                       dict(color="0.35", linewidth=0.35, alpha=0.7, zorder=1.5))):
        if len(layer):
            (layer.boundary if kw.get("color") else layer).plot(
                ax=ax, rasterized=True, **kw)

    rec = receptors.cx[win[0]:win[2], win[1]:win[3]]
    wbod = rec[rec["src"] == "wb"]
    fl = rec[rec["src"] == "fl"]
    is_per = (fl["kind"] == "perennial stream") | fl["name"].isin(PER_NAMES)
    per, oth = fl[is_per], fl[~is_per]
    if len(wbod):
        wbod.plot(ax=ax, facecolor="#9ECAE1", edgecolor="#0072B2",
                  linewidth=0.5, alpha=0.75, zorder=2.0, rasterized=True)
    if len(oth):
        oth.plot(ax=ax, color="#56B4E9", linewidth=0.9,
                 linestyle=(0, (4, 2)), zorder=2.2, rasterized=True)
    if len(per):
        per.plot(ax=ax, color="#0072B2", linewidth=1.8, zorder=2.4,
                 rasterized=True)

    segs = network_in(win)
    net = None
    if segs is not None and len(segs):
        net = ax.add_collection(LineCollection(
            [np.asarray(g.coords) for g in segs.geometry],
            array=segs["P_24mmh"].to_numpy(), cmap="magma_r",
            clim=(P_FLOOR, P_CEIL),
            linewidths=np.clip(0.5 + 0.9 * np.sqrt(segs["Area_km2"]), 0.5, 2.4),
            alpha=0.9, zorder=3.5, rasterized=True))

    box = sites4326.cx[win[0]:win[2], win[1]:win[3]]
    clear = box[box["dist_m"].isna()]
    near = box[(~box["exposed"].astype(bool)) & box["dist_m"].notna()]
    ex = box[box["exposed"].astype(bool)]
    # footprints, not just anchors: at this scale a tailings pile has extent
    if len(ex):
        ex[ex.geometry.type.isin(["Polygon", "MultiPolygon"])].plot(
            ax=ax, facecolor="none", edgecolor="#222222", linewidth=0.7,
            zorder=9.5)
    ax.scatter([p.x for p in clear["rep"]], [p.y for p in clear["rep"]],
               s=14, marker=".", color="#777777", zorder=9, linewidths=0)
    ax.scatter([p.x for p in near["rep"]], [p.y for p in near["rep"]], s=34,
               marker="o", facecolors="none", edgecolors="#5D5D5D",
               linewidths=0.9, zorder=9)
    sitesc = ax.scatter([p.x for p in ex["rep"]], [p.y for p in ex["rep"]],
                        s=80, c=ex["P_annual_site"], vmin=0, vmax=vmax,
                        marker="o", edgecolors="white", linewidths=0.8,
                        zorder=10)
    places = ctx["places"].clip(tuple(win))
    if len(places):
        ax.scatter(places.geometry.x, places.geometry.y, s=16, color="black",
                   edgecolor="white", linewidth=0.6, zorder=9)

    mc.style_axes(ax, extent, step=mc.tick_step(win[2] - win[0]))
    fig.canvas.draw()

    lab = Labeller(ax)
    lab.block_many([p.x for p in box["rep"]], [p.y for p in box["rep"]],
                   radius_px=6.0)
    for _, r in places.iterrows():
        lab.label(r.geometry.x, r.geometry.y, shorten(r["name"]),
                  fontsize=8, color="black", weight="bold", zorder=11)
    named = rec[rec["name"].notna()].copy()
    if len(named):
        named["L"] = named.geometry.length
        for nm, grp in sorted(named.groupby("name"),
                              key=lambda kv: -kv[1]["L"].sum())[:8]:
            p = interior_point(grp.loc[grp["L"].idxmax()].geometry, win)
            if p:
                lab.label(p[0], p[1], shorten(nm), fontsize=7.5,
                          color="#0072B2", style="italic", zorder=11)
    for _, r in ex.sort_values("rank").iterrows():
        if r["rank"] <= NTOP:
            lab.label(r["rep"].x, r["rep"].y, f"{int(r['rank'])}", fontsize=9.5,
                      color="#C1272D", weight="bold", zorder=12,
                      force_leader=lab.crowded(r["rep"].x, r["rep"].y,
                                               radius_px=CROWD_PX))

    cby = (BOTM - 0.40) / FH
    if net is not None:
        cb = fig.colorbar(net, cax=fig.add_axes([0.10, cby, 0.34, 0.013]),
                          orientation="horizontal")
        cb.set_label("channel debris-flow likelihood (I15 = 24 mm/h)",
                     fontsize=8, labelpad=2)
        cb.ax.tick_params(labelsize=7, pad=1.5)
    cb2 = fig.colorbar(sitesc, cax=fig.add_axes([0.56, cby, 0.34, 0.013]),
                       orientation="horizontal")
    cb2.set_label("site annual hit probability  P(F) × P(R>T) × P(DF)",
                  fontsize=8, labelpad=2)
    cb2.ax.tick_params(labelsize=7, pad=1.5)

    fig.legend(handles=[
        Line2D([0], [0], color="#0072B2", lw=1.8,
               label="perennial stream (NHD)"),
        Line2D([0], [0], color="#56B4E9", lw=0.9, linestyle=(0, (4, 2)),
               label="named intermittent / canal"),
        Patch(facecolor="#9ECAE1", edgecolor="#0072B2",
              label="named waterbody"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#3B528B",
               markeredgecolor="white", markersize=8,
               label="waste site within 30–55 m of a channel"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor="#5D5D5D", markersize=7,
               label="within 1 km of a channel"),
        Patch(facecolor="#D9C98C", alpha=0.5, label="BLM-managed land"),
    ], loc="lower center", bbox_to_anchor=(0.5, 0.048), ncol=3, fontsize=8,
        frameon=False)

    best = c["best"]
    ri = 1.0 / float(best["P_annual_site"])
    water = (best["water_name"] if isinstance(best["water_name"], str)
             else f"nearest {best['water_kind']}"
             if isinstance(best.get("water_kind"), str) else "no mapped water")
    blm_n = int(ex["on_blm"].sum()) if len(ex) else 0
    fig.suptitle(
        f"{c['county']} County — abandoned-mine waste on the debris-flow "
        f"network", fontsize=13, y=0.985)
    fig.text(0.5, 0.945,
             f"#{c['rank']} statewide: {site_name(best)}, 1 in {ri:,.0f} yr · "
             f"drains to {water} at {best['water_dist_m']/1000:.1f} km · "
             f"{len(ex)} of {len(box)} sites in this window lie within "
             f"30–55 m of a channel, {blm_n} on BLM land",
             ha="center", fontsize=9.5, color="#333333")
    mc.footnote(fig,
        "Numbers are the statewide exposure rank. Proximity to the modelled "
        "network, not a runout model: it counts a dump on a terrace beside "
        "the channel and misses one on a distal fan. Assets are USMIN "
        "topo-sheet symbols — historical, not the state's operational AML "
        "inventory.")
    mc.save(fig, f"aml_zoom_{c['key']}")
    plt.close(fig)
    return {"key": c["key"], "county": c["county"], "best_rank": c["rank"],
            "best_site": site_name(best), "return_interval_yr": round(ri),
            "receptor": water, "sites_in_window": int(len(box)),
            "near_channel": int(len(ex)), "on_blm": blm_n,
            "bounds": [round(v, 4) for v in win]}


want = sys.argv[1] if len(sys.argv) > 1 else None
found = clusters()
print(f"{len(found)} districts: " + ", ".join(c["key"] for c in found),
      flush=True)
summary = []
for c in found:
    if want and c["key"] != want:
        continue
    print(f"\n=== {c['key']}: {c['county']} County, best rank "
          f"#{c['rank']}, {len(c['members'])} top-{NTOP} sites ===", flush=True)
    summary.append(sheet(c))
if not want:
    (EXPO / "zoom_districts.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {EXPO / 'zoom_districts.json'}")
