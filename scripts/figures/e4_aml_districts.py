"""District sheets for the AML exposure ranking -> figures/aml_districts_v2.pdf.

Two data-driven windows around the top-ranked clusters (split at 40 N), each
showing the mechanism the statewide dots compress: the hazard network colored
by likelihood, the corridors delivering into named NHD receptor waters, BLM
holdings shaded, and the ranked waste sites numbered.

Three sheet-scale rules this file has to keep, none of which the statewide
map needs:

1. **Shade finer than the display grid.** ``plotting.hillshade`` shades at
   ``relief.SHADE_RES_M`` (50 m), which is *coarser* than a district panel's
   pixel (23-75 m here) and so breaks relief.py's own rule 3 — the terrain
   arrives pre-blurred. This module sizes ``shade_res_m`` to the window
   instead, floored at the 10 m 3DEP native grid. It also keeps its own
   bounds-keyed cache: ``plotting.hillshade``'s cache key carries resolution
   but not extent, which is safe for one statewide domain and not for two
   windows.

2. **Rasterize the dense layers.** The network is ~200k vector line segments
   per panel. Left as vectors the sheet was 57 MB and slow to open; rasterized
   at the save dpi it is a few MB, and the linework a reader zooms into (text,
   markers, leaders) stays vector.

3. **Place labels against a collision list.** Dense districts stack numbers on
   top of each other, so labels get pushed onto a leader line and every label
   is tested against the panel frame before it is drawn — which is also what
   keeps a long river name from running off the edge.
"""
import hashlib
import json
import math
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
from matplotlib.patheffects import withStroke
from rasterio.warp import transform_bounds

from firescape import paths, relief
from firescape import plotting as mc

plt.rcParams["image.cmap"] = "viridis"
V12 = paths.products_dir("prefire", "statewide_v1_2")
TOPN, PADDEG, NPX = 20, 0.05, 1900
P_FLOOR, P_CEIL = 0.10, 0.60      # network color range; below the floor is quiet
MAXNAME = 26


def shorten(name, limit=MAXNAME):
    """Trim a long feature name at a word boundary, not mid-word."""
    name = str(name)
    if len(name) <= limit:
        return name
    cut = name[:limit - 1].rsplit(" ", 1)[0]
    return (cut if cut else name[:limit - 1]) + "…"


def square_window(members, pad=PADDEG):
    """Window around ``members``, square *on screen*.

    ``style_axes`` gives the axes a 1/cos(lat) aspect, so a window square in
    degrees renders ~1.3x taller than wide and letterboxes inside its box.
    Squaring in screen units instead is what removes that white space.
    """
    w, s, e, n = members.total_bounds
    w, s, e, n = w - pad, s - pad, e + pad, n + pad
    cx, cy = (w + e) / 2.0, (s + n) / 2.0
    latf = 1.0 / math.cos(math.radians(cy))
    span = max(e - w, (n - s) * latf)
    return (cx - span / 2, cy - span / (2 * latf),
            cx + span / 2, cy + span / (2 * latf))


def hillshade(win, tr, shape):
    """Shaded relief for one window, shaded finer than the display grid."""
    lat = (win[1] + win[3]) / 2.0
    res_m = (win[2] - win[0]) * 111_320 * math.cos(math.radians(lat)) / shape[1]
    shade_m = max(10.0, res_m / 3.0)
    key = hashlib.sha1(
        f"{win}|{shape}|{shade_m:.2f}".encode()).hexdigest()[:12]
    cache = paths.interim_dir("exposure") / f"hs_{key}.npz"
    if cache.exists():
        return np.load(cache)["hs"], res_m, shade_m
    hs = relief.shaded_relief(
        sorted((paths.cache_root() / "3dep_tiles").glob("USGS_13_*.tif")),
        tr, shape, crs="EPSG:4326", shade_res_m=shade_m)
    np.savez_compressed(cache, hs=hs)
    return hs, res_m, shade_m


def tick_step(span):
    for s in (0.05, 0.1, 0.2, 0.25, 0.5, 1.0):
        if span / s <= 6:
            return s
    return 1.0


def interior_point(geom, win):
    """Vertex of ``geom`` furthest from any window edge (in-window only).

    Anchoring a river name at its representative point drops it wherever the
    reach happens to be centred, often against the frame. Picking the most
    interior vertex is the anchor move that keeps long names on the sheet.
    """
    w, s, e, n = win
    coords = []
    for part in getattr(geom, "geoms", [geom]):
        coords.extend(list(part.coords))
    if not coords:
        return None
    step = max(1, len(coords) // 60)
    best, score = None, -1.0
    for x, y in coords[::step]:
        if not (w < x < e and s < y < n):
            continue
        d = min((x - w) / (e - w), (e - x) / (e - w),
                (y - s) / (n - s), (n - y) / (n - s))
        if d > score:
            best, score = (x, y), d
    return best


#: Where a label may sit, in display pixels from its point: touching first,
#: then progressively further out.
_RINGS = [(7.0, 5.0)] + [
    (r * math.cos(math.radians(a)), r * math.sin(math.radians(a)))
    for r in (20, 32, 44, 58, 74)
    for a in (30, 150, 210, 330, 90, 270, 0, 180, 60, 120, 240, 300)]

#: A leader is only worth drawing once the offset clears the label's own box —
#: a two-digit number at 8.5 pt is ~22 px wide, so a 20 px offset leaves about
#: two pixels of visible line. Below this the label simply sits by its dot.
LEADER_FROM = 28.0


def leader(ax, px, py, cx, cy, w, h, color):
    """Line from a marker to the edge of its displaced label box.

    Drawn explicitly rather than through ``arrowprops`` so the near end stops
    on the box boundary and the line is actually visible.
    """
    dx, dy = px - cx, py - cy
    d = math.hypot(dx, dy)
    if not d:
        return
    ux, uy = dx / d, dy / d
    t = min((w / 2 + 1.0) / abs(ux) if ux else 1e9,
            (h / 2 + 1.0) / abs(uy) if uy else 1e9)
    inv = ax.transData.inverted()
    x0, y0 = inv.transform((cx + ux * t, cy + uy * t))       # at the label
    x1, y1 = inv.transform((px - ux * 2.5, py - uy * 2.5))   # short of the dot
    ax.add_line(Line2D([x0, x1], [y0, y1], lw=0.55, color=color, zorder=11.5,
                       solid_capstyle="round",
                       path_effects=[withStroke(linewidth=1.7,
                                                foreground="white")]))


def place_label(ax, taken, x, y, text, *, fontsize, color, style="normal",
                weight="normal", zorder=12, force_leader=False):
    """Draw ``text`` near (x, y), off a leader line if that is what it takes.

    Candidate offsets spiral outward; each is rejected if it would overlap a
    label already placed, a marker, or the panel frame. Returns False when
    every candidate fails, so the caller can drop the label rather than print
    it on top of another one.
    """
    fig = ax.figure
    px, py = ax.transData.transform((x, y))
    dpi = fig.dpi
    w = 0.62 * fontsize * len(text) * dpi / 72.0
    h = 1.10 * fontsize * dpi / 72.0
    ab = ax.get_window_extent()
    # In a crowd, any position close enough to be ambiguous is worse than one
    # far enough to carry a leader — so only offer the far rings there.
    for dx, dy in ([c for c in _RINGS if math.hypot(*c) >= LEADER_FROM]
                   if force_leader else _RINGS):
        r = (px + dx - w / 2, py + dy - h / 2, px + dx + w / 2, py + dy + h / 2)
        if (r[0] < ab.x0 + 3 or r[2] > ab.x1 - 3
                or r[1] < ab.y0 + 3 or r[3] > ab.y1 - 3):
            continue
        if any(not (r[2] < q[0] - 2 or r[0] > q[2] + 2
                    or r[3] < q[1] - 2 or r[1] > q[3] + 2) for q in taken):
            continue
        if force_leader or math.hypot(dx, dy) > LEADER_FROM:
            leader(ax, px, py, px + dx, py + dy, w, h, color)
        ax.annotate(text, (x, y), xytext=(dx * 72.0 / dpi, dy * 72.0 / dpi),
                    textcoords="offset points", ha="center", va="center",
                    fontsize=fontsize, color=color, style=style,
                    fontweight=weight, zorder=zorder,
                    path_effects=[withStroke(linewidth=2.0,
                                             foreground="white")])
        taken.append(r)
        return True
    return False


sites = gpd.read_file(
    paths.products_dir("exposure", "aml_v2") / "aml_exposure.gpkg").to_crs(4326)
sites["rep"] = sites.geometry.representative_point()
receptors = gpd.read_file(paths.interim_dir("exposure") / "nhd_receptors.gpkg")
blm = gpd.read_file(paths.raw_dir("blm") / "nv_blm_sma.gpkg").to_crs(4326)
ctx = mc.fetch_context()          # statewide cache; clipped per window below
ubounds = json.loads(
    (paths.interim_dir("exposure") / "unit_bounds_v1_2.json").read_text())

top = sites.nsmallest(TOPN, "rank")
vmax = float(np.nanpercentile(
    sites.loc[sites["exposed"].astype(bool), "P_annual_site"], 99))

panels = [(lab, square_window(top[sel]))
          for lab, sel in (("North-central Nevada districts", top["rep"].y >= 40),
                           ("Comstock–Virginia Range", top["rep"].y < 40))
          if len(top[sel])]

PANEL_IN, LEFT, GAP, RIGHT, TOPM, BOTM = 6.2, 0.42, 0.30, 0.42, 0.62, 1.25
FW = LEFT + len(panels) * PANEL_IN + (len(panels) - 1) * GAP + RIGHT
FH = TOPM + PANEL_IN + BOTM
fig = plt.figure(figsize=(FW, FH), dpi=150)
gs = fig.add_gridspec(1, len(panels), left=LEFT / FW, right=1 - RIGHT / FW,
                      top=1 - TOPM / FH, bottom=BOTM / FH,
                      wspace=GAP / PANEL_IN)

net = None
for i, (label, win) in enumerate(panels):
    ax = fig.add_subplot(gs[0, i])
    tr, shape, extent = mc.grid(win, res=(win[2] - win[0]) / NPX)
    hs, res_m, shade_m = hillshade(win, tr, shape)
    print(f"{label}: {shape[1]}x{shape[0]} px, {res_m:.0f} m/px, "
          f"shaded at {shade_m:.0f} m", flush=True)
    ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=extent, zorder=0)
    bl = blm.clip(tuple(win))
    if len(bl):
        bl.plot(ax=ax, facecolor="#D9C98C", edgecolor="none", alpha=0.28,
                zorder=1, rasterized=True)
    cc = ctx["counties"].clip(tuple(win))
    if len(cc):
        cc.boundary.plot(ax=ax, color="0.35", linewidth=0.35, alpha=0.7,
                         zorder=2, rasterized=True)

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
    segs = pd.concat(segs).to_crs(4326).explode(index_parts=False)
    segs = segs[segs.geometry.type == "LineString"]
    segs = segs[segs["P_24mmh"] >= P_FLOOR].sort_values("P_24mmh")
    print(f"  {len(segs)} network segments drawn (rasterized)", flush=True)
    net = ax.add_collection(LineCollection(
        [np.asarray(g.coords) for g in segs.geometry],
        array=segs["P_24mmh"].to_numpy(), cmap="magma_r",
        clim=(P_FLOOR, P_CEIL),
        linewidths=np.clip(0.2 + 0.3 * np.sqrt(segs["Area_km2"]), 0.2, 1.1),
        alpha=0.75, zorder=3, rasterized=True))

    # Receptors: at or below the weight of the widest hazard channel (1.1),
    # so the water reads as context and the hazard stays the subject.
    rec = receptors.cx[win[0]:win[2], win[1]:win[3]]
    wbod = rec[rec["src"] == "wb"]
    per = rec[(rec["src"] == "fl") & (rec["kind"] == "perennial stream")]
    oth = rec[(rec["src"] == "fl") & (rec["kind"] != "perennial stream")]
    if len(wbod):
        wbod.plot(ax=ax, facecolor="#9ECAE1", edgecolor="#0072B2",
                  linewidth=0.4, alpha=0.75, zorder=4, rasterized=True)
    if len(oth):
        oth.plot(ax=ax, color="#56B4E9", linewidth=0.55,
                 linestyle=(0, (4, 2)), zorder=5, rasterized=True)
    if len(per):
        per.plot(ax=ax, color="#0072B2", linewidth=1.0, zorder=5,
                 rasterized=True)

    box = sites.cx[win[0]:win[2], win[1]:win[3]]
    clear = box[box["dist_m"].isna()]
    near = box[(~box["exposed"].astype(bool)) & box["dist_m"].notna()]
    ex = box[box["exposed"].astype(bool)]
    ax.scatter([p.x for p in clear["rep"]], [p.y for p in clear["rep"]],
               s=6, marker=".", color="#777777", zorder=9, linewidths=0)
    ax.scatter([p.x for p in near["rep"]], [p.y for p in near["rep"]], s=16,
               marker="o", facecolors="none", edgecolors="#5D5D5D",
               linewidths=0.7, zorder=9)
    ax.scatter([p.x for p in ex["rep"]], [p.y for p in ex["rep"]], s=34,
               c=ex["P_annual_site"], vmin=0, vmax=vmax, marker="o",
               edgecolors="white", linewidths=0.5, zorder=10)
    places = ctx["places"].clip(tuple(win))
    if len(places):
        ax.scatter(places.geometry.x, places.geometry.y, s=11, color="black",
                   edgecolor="white", linewidth=0.5, zorder=9)

    mc.style_axes(ax, extent, step=tick_step(win[2] - win[0]))
    if i:
        # Degree labels on the outside edge, so the panels can sit close
        # together without the right panel's labels landing on the left
        # panel's frame.
        ax.yaxis.tick_right()
    ax.set_title(f"{label} — {int(ex['on_blm'].sum())} of {len(ex)} "
                 f"exposed sites on BLM land", fontsize=10)
    fig.canvas.draw()          # transforms must be final before labels land

    # Labels, most important last-resort-free first: towns, then the waters
    # they sit on, then the ranked sites, all sharing one collision list.
    taken, marks = [], []
    for pt in list(clear["rep"]) + list(near["rep"]) + list(ex["rep"]):
        px, py = ax.transData.transform((pt.x, pt.y))
        taken.append((px - 4, py - 4, px + 4, py + 4))
        marks.append((px, py))
    for _, row in places.iterrows():
        place_label(ax, taken, row.geometry.x, row.geometry.y,
                    shorten(row["name"]), fontsize=7.0, color="black",
                    weight="bold", zorder=11)
    named = rec[rec["name"].notna()].copy()
    if len(named):
        named["L"] = named.geometry.length
        ranked = sorted(named.groupby("name"), key=lambda kv: -kv[1]["L"].sum())
        for nm, grp in ranked[:8]:
            p = interior_point(grp.loc[grp["L"].idxmax()].geometry, win)
            if p:
                place_label(ax, taken, p[0], p[1], shorten(nm),
                            fontsize=6.5, color="#0072B2", style="italic",
                            zorder=11)
    dropped = 0
    for _, r in ex[ex["rank"] <= TOPN].sort_values("rank").iterrows():
        px, py = ax.transData.transform((r["rep"].x, r["rep"].y))
        crowd = sum((qx - px) ** 2 + (qy - py) ** 2 < 26 ** 2
                    for qx, qy in marks) - 1
        if not place_label(ax, taken, r["rep"].x, r["rep"].y,
                           f"{int(r['rank'])}", fontsize=8.5, color="#C1272D",
                           weight="bold", zorder=12, force_leader=crowd > 0):
            dropped += 1
    if dropped:
        print(f"  {dropped} site labels had nowhere to go", flush=True)

cax = fig.add_axes([0.5 - 0.13, (BOTM - 0.42) / FH, 0.26, 0.012])
cb = fig.colorbar(net, cax=cax, orientation="horizontal")
cb.set_label("segment debris-flow likelihood (I15 = 24 mm/h)", fontsize=8,
             labelpad=2)
cb.ax.tick_params(labelsize=7, pad=1.5)
fig.legend(handles=[
    Line2D([0], [0], color="#0072B2", lw=1.0, label="perennial stream (NHD)"),
    Line2D([0], [0], color="#56B4E9", lw=0.55, linestyle=(0, (4, 2)),
           label="named intermittent / canal"),
    Patch(facecolor="#9ECAE1", edgecolor="#0072B2", label="named waterbody"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#3B528B",
           markeredgecolor="white", markersize=7,
           label="waste site in a delivery corridor"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
           markeredgecolor="#5D5D5D", markersize=6,
           label="within 1 km of one"),
    Patch(facecolor="#D9C98C", alpha=0.5, label="BLM-managed land"),
], loc="lower center", bbox_to_anchor=(0.5, 0.022), ncol=6, fontsize=8,
    frameon=False)
fig.suptitle("Where the top-ranked AML sites meet the water: hazard network, "
             "delivery corridors, and named receptors", fontsize=12, y=0.975)
fig.text(0.5, 0.006,
         "Site color = annual hit probability; numbers are the statewide "
         "exposure rank. Delivery is a proximity proxy (corridor width "
         "9–60 m from contributing area), not a runout model.",
         ha="center", fontsize=6.5, color="#444444")
mc.save(fig, "aml_districts_v2")
print("saved figures/aml_districts_v2.pdf")
