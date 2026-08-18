"""District zooms for the AML exposure ranking -> figures/aml_zoom_<key>.pdf.

The companion to ``e3_aml_map``/``e8_aml_compilation`` in the same relationship
``p10_urban_zooms`` has to the statewide hazard sheet: the statewide map
answers *where in Nevada*, which is the wrong scale for anyone deciding what to
visit. One district per sheet, at a scale where the individual working and the
channel beside it can be pointed at.

**Both asset classes on every sheet.** Mapped waste (circles) and underground
workings (triangles) are the same problem seen through two halves of one
record, and a sheet showing only the half that happened to be drawn as an
extent would mislead about what is in the district.

**Districts come from both rankings, because they disagree completely.** The
top waste districts are the Comstock and the Humboldt-Pershing belt, 1 in 78
to 1 in 112 years. The top opening districts are all in Elko, 1 in 29 to 1 in
76 — and the nearest waste district to any of them is 128-161 km away. So the
sheet set is the six best waste districts *plus* the five best opening
districts; ranking on either half alone would leave the other half's worst
ground undrawn.

Windows are derived from the rankings, not hand-drawn: the top-ranked assets
of a class are single-linkage clustered at 6 km and the best districts
rendered, so a recalibration changes which districts get drawn without
anything here being edited.

    python scripts/figures/e4_aml_zooms.py                # all eleven
    python scripts/figures/e4_aml_zooms.py storey_01      # one waste district
    python scripts/figures/e4_aml_zooms.py op_elko_01     # one opening district
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
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter
from rasterio.warp import transform_bounds
from scipy.cluster.hierarchy import fcluster, linkage
from stormscape.plot import Labeller, interior_point, shorten

from firescape import paths
from firescape import plotting as mc

plt.rcParams["image.cmap"] = "viridis"
V12 = paths.products_dir("prefire", "statewide_v1_2")
EXPO = paths.products_dir("exposure")

NTOP, CLUSTER_KM = 40, 6.0
NSHEETS = {"waste": 6, "openings": 5}
MIN_SPAN_KM, PAD_KM = 9.0, 1.5
P_FLOOR, P_CEIL = 0.05, 0.60      # network colour range at district scale
NPX, MIN_M_PER_PX = 1900, 8.0     # 3DEP is 10 m; do not pretend to finer
CROWD_PX, NLABEL = 26.0, 12
STYLE = {"waste": dict(marker="o", size=70, edge="#111111", lw=0.6),
         "openings": dict(marker="^", size=60, edge="white", lw=0.5)}

ASSETS = {
    "waste": gpd.read_file(EXPO / "aml_v2" / "aml_exposure.gpkg"),
    "openings": gpd.read_file(EXPO / "openings_v1" / "openings_exposure.gpkg"),
}
for _cls, _g in ASSETS.items():
    _g["cls"] = _cls
    _g["exposed"] = _g["exposed"].astype(bool)
ll = {c: g.to_crs(4326) for c, g in ASSETS.items()}
for _g in ll.values():
    _g["rep"] = _g.geometry.representative_point()

receptors = gpd.read_file(paths.interim_dir("exposure") / "nhd_receptors.gpkg")
blm = gpd.read_file(paths.raw_dir("blm") / "nv_blm_sma.gpkg").to_crs(4326)
ctx = mc.fetch_context()
ubounds = json.loads(
    (paths.interim_dir("exposure") / "unit_bounds_v1_2.json").read_text())
PER_NAMES = set(receptors.loc[(receptors["src"] == "fl")
                              & (receptors["kind"] == "perennial stream"),
                              "name"].dropna())

#: One colour scale for every sheet and both classes — log, because the top
#: Elko workings run an order of magnitude above the median district, and a
#: linear ramp printed every other district the same shade of purple. Bounds
#: are fixed to the return intervals the bar is labelled with rather than to
#: percentiles: a 2nd-percentile floor put three decades on the bar and
#: crushed every tick into its top third, and fixed bounds also make one
#: sheet's colour mean the same as the next one's.
NORM = LogNorm(vmin=1 / 3000, vmax=1 / 25)


def site_name(row):
    for v in (row.get("name"), row.get("ftr_type")):
        if isinstance(v, str) and v.strip():
            return v.strip()
    return "unnamed working"


def districts():
    """Sheets worth drawing, from each class's own ranking."""
    out = []
    for cls, n in NSHEETS.items():
        g = ASSETS[cls]
        top = g.nsmallest(NTOP, "rank").copy()
        xy = np.c_[top.geometry.centroid.x, top.geometry.centroid.y]
        top["grp"] = (fcluster(linkage(xy, method="single"),
                               t=CLUSTER_KM * 1000, criterion="distance")
                      if len(top) > 1 else np.ones(len(top), int))
        found = []
        for _, grp in top.groupby("grp"):
            best = grp.loc[grp["rank"].idxmin()]
            county = (best["county"] if isinstance(best["county"], str)
                      else "Nevada")
            pre = "" if cls == "waste" else "op_"
            found.append({"driver": cls, "best": best, "members": grp,
                          "county": county, "rank": int(best["rank"]),
                          "key": f"{pre}{county.split()[0].lower()}_"
                                 f"{int(best['rank']):02d}"})
        found.sort(key=lambda c: c["rank"])
        out.extend(found[:n])
    return out


def window_for(members):
    b = members.to_crs(4326).total_bounds
    cy = (b[1] + b[3]) / 2.0
    pad = PAD_KM * 1000 / 111_320.0
    w, s, e, n = b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad
    min_deg = MIN_SPAN_KM * 1000 / (111_320.0 * np.cos(np.radians(cy)))
    if (e - w) < min_deg:
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

    PANEL, LEFT, RIGHT, TOPM, BOTM = 7.0, 0.55, 0.42, 0.92, 1.60
    FW, FH = LEFT + PANEL + RIGHT, TOPM + PANEL + BOTM
    fig = plt.figure(figsize=(FW, FH), dpi=150)
    ax = fig.add_axes([LEFT / FW, BOTM / FH, PANEL / FW, PANEL / FH])

    ax.imshow(mc.hillshade(tr, shape), cmap="gray", vmin=0, vmax=1,
              extent=extent, zorder=0)
    print(f"  {shape[1]}x{shape[0]} px, {mc.grid_res_m(tr, shape):.0f} m/px, "
          f"shaded at {mc.shade_resolution(tr, shape):.0f} m", flush=True)
    bl = blm.clip(tuple(win))
    if len(bl):
        bl.plot(ax=ax, facecolor="#D9C98C", edgecolor="none", alpha=0.28,
                zorder=1, rasterized=True)
    cc = ctx["counties"].clip(tuple(win))
    if len(cc):
        cc.boundary.plot(ax=ax, color="0.35", linewidth=0.35, alpha=0.7,
                         zorder=1.5, rasterized=True)

    # named water down first, the modelled channels over it
    rec = receptors.cx[win[0]:win[2], win[1]:win[3]]
    wbod = rec[rec["src"] == "wb"]
    fl = rec[rec["src"] == "fl"]
    is_per = (fl["kind"] == "perennial stream") | fl["name"].isin(PER_NAMES)
    if len(wbod):
        wbod.plot(ax=ax, facecolor="#9ECAE1", edgecolor="#0072B2",
                  linewidth=0.5, alpha=0.75, zorder=2.0, rasterized=True)
    if (~is_per).any():
        fl[~is_per].plot(ax=ax, color="#56B4E9", linewidth=0.9,
                         linestyle=(0, (4, 2)), zorder=2.2, rasterized=True)
    if is_per.any():
        fl[is_per].plot(ax=ax, color="#0072B2", linewidth=1.8, zorder=2.4,
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

    inwin, sc, counts = {}, None, {}
    for cls in ("openings", "waste"):        # waste drawn last: it is rarer
        box = ll[cls].cx[win[0]:win[2], win[1]:win[3]]
        inwin[cls] = box
        st = STYLE[cls]
        far, near = box[~box["exposed"]], box[box["exposed"]]
        counts[cls] = (len(near), len(box),
                       int(near["on_blm"].sum()) if len(near) else 0)
        if len(far):
            ax.scatter([p.x for p in far["rep"]], [p.y for p in far["rep"]],
                       s=st["size"] * 0.25, marker=st["marker"],
                       facecolors="none", edgecolors="#6E6E6E",
                       linewidths=0.6, zorder=9)
        if len(near):
            sc = ax.scatter([p.x for p in near["rep"]],
                            [p.y for p in near["rep"]], s=st["size"],
                            c=near["P_annual_site"], norm=NORM,
                            marker=st["marker"], edgecolors=st["edge"],
                            linewidths=st["lw"], zorder=10)
    places = ctx["places"].clip(tuple(win))
    if len(places):
        ax.scatter(places.geometry.x, places.geometry.y, s=16, color="black",
                   edgecolor="white", linewidth=0.6, zorder=9)

    mc.style_axes(ax, extent, step=mc.tick_step(win[2] - win[0]))
    fig.canvas.draw()

    lab = Labeller(ax)
    for cls in ("waste", "openings"):
        b = inwin[cls]
        if len(b):
            lab.block_many([p.x for p in b["rep"]], [p.y for p in b["rep"]],
                           radius_px=6.0)
    for _, r in places.iterrows():
        lab.label(r.geometry.x, r.geometry.y, shorten(r["name"]), fontsize=8,
                  color="black", weight="bold", zorder=11)
    named = rec[rec["name"].notna()].copy()
    if len(named):
        named["L"] = named.geometry.length
        for nm, grp in sorted(named.groupby("name"),
                              key=lambda kv: -kv[1]["L"].sum())[:8]:
            p = interior_point(grp.loc[grp["L"].idxmax()].geometry, win)
            if p:
                lab.label(p[0], p[1], shorten(nm), fontsize=7.5,
                          color="#0072B2", style="italic", zorder=11)
    # number the driving class only: the two classes carry independent rank
    # scales, and numbering both would invite them to be read as one list
    drv = inwin[c["driver"]]
    drv = drv[drv["exposed"]].sort_values("rank").head(NLABEL)
    for _, r in drv.iterrows():
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
    if sc is not None:
        cb2 = fig.colorbar(sc, cax=fig.add_axes([0.56, cby, 0.34, 0.013]),
                           orientation="horizontal", extend="both")
        cb2.set_label("annual chance a debris flow reaches the site, "
                      "as a return interval (years)", fontsize=8, labelpad=2)
        ticks = [t for t in (1/30, 1/100, 1/300, 1/1000, 1/3000)
                 if NORM.vmin <= t <= NORM.vmax]
        cb2.ax.xaxis.set_major_locator(FixedLocator(ticks))
        cb2.ax.xaxis.set_minor_locator(FixedLocator([]))
        cb2.ax.xaxis.set_major_formatter(FuncFormatter(
            lambda x, _: f"{round(1/x, -1):,.0f}" if x > 0 else ""))
        cb2.ax.tick_params(labelsize=7, pad=1.5)

    nw, tw, bw = counts["waste"]
    no, to, bo = counts["openings"]
    fig.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#3B528B",
               markeredgecolor="#111111", markersize=8,
               label=f"mapped waste, near a channel ({nw} of {tw})"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#3B528B",
               markeredgecolor="white", markersize=8,
               label=f"adit / shaft, near a channel ({no} of {to})"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor="#6E6E6E", markersize=7,
               label="further than 30–55 m"),
        Line2D([0], [0], color="#0072B2", lw=1.8,
               label="perennial stream (NHD)"),
        Line2D([0], [0], color="#56B4E9", lw=0.9, linestyle=(0, (4, 2)),
               label="named intermittent / canal"),
        Patch(facecolor="#D9C98C", alpha=0.5, label="BLM-managed land"),
    ], loc="lower center", bbox_to_anchor=(0.5, 0.048), ncol=3, fontsize=8,
        frameon=False)

    best = c["best"]
    ri = 1.0 / float(best["P_annual_site"])
    water = (best["water_name"] if isinstance(best["water_name"], str)
             else f"nearest {best['water_kind']}"
             if isinstance(best.get("water_kind"), str) else "no mapped water")
    what = "mine waste" if c["driver"] == "waste" else "underground workings"
    fig.suptitle(f"{c['county']} County — abandoned-mine material on the "
                 f"debris-flow network", fontsize=13, y=0.985)
    fig.text(0.5, 0.952,
             f"ranked on {what}: #{c['rank']} statewide is "
             f"{site_name(best)}, 1 in {ri:,.0f} yr, draining to {water}",
             ha="center", fontsize=9.5, color="#333333")
    wtxt = (f"{nw} of {tw} mapped waste sites" if tw else
            "no mapped waste at all — USMIN drew none here")
    fig.text(0.5, 0.929,
             f"in this window: {wtxt} and {no} of {to} workings lie within "
             f"30–55 m of a channel · {bw + bo} of those on BLM land",
             ha="center", fontsize=9.5, color="#333333")
    mc.footnote(fig,
        "Numbers are the statewide exposure rank within the class this "
        "district was chosen on; the two classes are ranked separately and "
        "their scales are not comparable. Proximity to the modelled network, "
        "not a runout model. Assets are USMIN topo-sheet symbols — a dump is "
        "mapped as an extent only where a topographer drew one, which is why "
        "the workings are shown alongside. This orders exposure, not source "
        "size: nothing in USMIN records how much rock a working produced.")
    mc.save(fig, f"aml_zoom_{c['key']}")
    plt.close(fig)
    return {"key": c["key"], "driver": c["driver"], "county": c["county"],
            "best_rank": c["rank"], "best_site": site_name(best),
            "return_interval_yr": round(ri), "receptor": water,
            "waste_near": nw, "waste_total": tw,
            "openings_near": no, "openings_total": to,
            "on_blm": bw + bo, "bounds": [round(v, 4) for v in win]}


want = sys.argv[1] if len(sys.argv) > 1 else None
found = districts()
print(f"{len(found)} districts: " + ", ".join(c["key"] for c in found),
      flush=True)
summary = []
for c in found:
    if want and c["key"] != want:
        continue
    print(f"\n=== {c['key']}: {c['county']} County, ranked on {c['driver']}, "
          f"best #{c['rank']} ===", flush=True)
    summary.append(sheet(c))
if not want:
    (EXPO / "zoom_districts.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {EXPO / 'zoom_districts.json'}")
