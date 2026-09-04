"""Post-fire hazard hindcast on MTBS severity, against the observed flows.

The fire-scale companion to ``p16_debrisflow_inventory_statewide``: for a
fire the debris-flow inventory has points inside, run the SAME hazard chain
the operational sheets run (Stallion, Bug, Hawk) but on MTBS's dNBR for that
fire, then draw the observed points on top. Every panel is a prediction the
model would have made the day after the fire; the triangles are what
actually happened.

**Which severity classification.** ``assess.run_observed`` prefers MTBS's own
``dnbr6`` analyst classes whenever a full bundle is on disk, and silently
ignores ``barc_breaks`` when it does -- so this passes the bundle's own
perimeter and dNBR through the *injected* route instead, which forces
``severity.estimate(dnbr, barc_breaks)`` and puts the region's calibrated
break on the map. That is deliberate: these sheets exist to test the
product the project publishes, and the calibrated break is what the product
uses. The MTBS analyst classification of the same raster is reported
alongside in the summary (``class_fraction_mtbs_dnbr6``) so the two
classifications of one fire can be compared without a second run.

**On the dates.** A point's ``obs_date`` is the Google Earth imagery date it
was digitized against, not a surveyed event date, so the lag printed between
ignition and observation is an upper bound on how long after the fire the
flow occurred -- not a measurement of it.

    python scripts/products/p16_mtbs_fire_hindcast.py <event_id> [calibration]
    python scripts/products/p16_mtbs_fire_hindcast.py --all [calibration]
"""
import json
import pathlib
import re
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterio.warp import Resampling, reproject

import geopandas as gpd

from firescape import assess, config, mtbs, paths, plotting as mc, statewide
from stormscape import burn

CAL_DEFAULT = "statewide_v1_4"
DESIGN_I15 = 24.0
TILE_DIR = paths.cache_root() / "3dep_tiles"
INVENTORY = paths.raw_dir("debrisflow_inventory") / "nv_debrisflow_inventory.geojson"
#: Which fire each inventory point falls in, from the MTBS WFS join
#: (scripts/stage). Regeneratable, so interim/ rather than raw/.
POINT_FIRES = paths.interim_dir("debrisflow_inventory") / "points_to_fires.csv"
REGIONS = (pathlib.Path(paths.__file__).resolve().parent
           / "data" / "regions" / "nv_prefire_regions.geojson")

#: Same five USGS likelihood classes and colours the operational sheets and
#: the CalTopo layers use, so a hindcast reads like the live product.
P_BREAKS = (0.2, 0.4, 0.6, 0.8)
P_COLORS = ("#2C7BB6", "#ABD9E9", "#FFFFBF", "#FDAE61", "#D7191C")
P_LABELS = ("< 20%", "20–40%", "40–60%", "60–80%", "≥ 80%")
SEV_LEGEND = ("unburned (not drawn)", "low", "moderate", "high")
SEV_ALPHA = 0.60
#: Points are matched to the nearest modelled channel, but only out to here:
#: beyond it the "nearest" segment is in a different drainage and its
#: prediction says nothing about the point.
MATCH_MAX_M = 500.0
#: Deposition happens on the fan, which is often outside the burn perimeter,
#: so the sheet carries points a little beyond it -- flagged, not silently
#: pooled with the ones inside.
POINT_PAD_M = 2000.0


def fire_identity(perim_gdf, bundle):
    """Fire name, mapping programme and assessment type, from the bundle.

    The burn-area shapefile carries all three as attributes (``INCID_NAME``,
    ``MAP_PROG``, ``ASMNT_TYPE``), which beats inferring the programme from
    the filename: it is the product's own statement about itself. Worth
    carrying, because a fire from the last season or two is usually too
    recent for MTBS's 1-2 year lag and comes back as the BAER emergency
    assessment instead -- a different product, from a different pass, with
    its own conventions. Broom Canyon (2024) is BAER/Emergency; the 2004-2014
    fires here are MTBS/Extended.
    """
    def field(name, default):
        if name in perim_gdf.columns and perim_gdf[name].notna().any():
            return str(perim_gdf[name].dropna().iloc[0])
        return default

    stem = pathlib.Path(bundle["dnbr"]).name.lower()
    fallback = "BAER" if stem.startswith("baer_") else (
        "MTBS" if stem.startswith("mtbs_") else "unknown-source")
    return (field("INCID_NAME", "unnamed").title(),
            field("MAP_PROG", fallback),
            field("ASMNT_TYPE", "unknown"))


def name_slug(fire_name):
    """File-safe stem from a fire name -- what a person actually references."""
    return re.sub(r"[^a-z0-9]+", "_", fire_name.lower()).strip("_")


def region_breaks(perim_gdf, cal_name):
    reg = gpd.read_file(REGIONS).to_crs(4326)
    pt = perim_gdf.to_crs(4326).geometry.union_all().representative_point()
    hit = reg.loc[reg.contains(pt), "region"]
    region = str(hit.iloc[0]) if len(hit) else "Central Basin and Range"
    slug = region.lower().replace(" ", "_").replace("&", "and")
    return region, list(config.packaged_calibration(cal_name).region(slug).barc_breaks)


def severity_rgba(dnbr_path, breaks, shape, dst_transform, dst_crs):
    """MTBS dNBR on the display grid, in the four BAER class colours."""
    with rasterio.open(dnbr_path) as ds:
        arr = ds.read(1).astype("float64")
        if ds.nodata is not None:
            arr[arr == ds.nodata] = np.nan
        src_transform, src_crs = ds.transform, ds.crs
    dst = np.full(shape, np.nan, dtype="float32")
    reproject(arr, dst, src_transform=src_transform, src_crs=src_crs,
              dst_transform=dst_transform, dst_crs=dst_crs,
              resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan)
    rgba = np.zeros(shape + (4,), dtype="uint8")
    a = int(round(255 * SEV_ALPHA))
    idx = np.digitize(dst, list(breaks))          # MTBS dNBR is already x1000
    for i, (r, g, b) in enumerate(burn.BAER_CLASS_COLORS):
        m = np.isfinite(dst) & (idx == i)
        rgba[m] = (r, g, b, 0 if i == 0 else a)
    return rgba, dst


def run_one(event_id, cal_name):
    bundle = mtbs.fire_bundle(event_id)
    perim = gpd.read_file(bundle["burn_area"])
    region, breaks = region_breaks(perim, cal_name)
    print(f"\n=== {event_id} · {region} · breaks {breaks} ===", flush=True)

    fire_name, source, asmnt = fire_identity(perim, bundle)
    ig = pd.Timestamp(mtbs_ig_date(event_id))
    pts = gpd.read_file(INVENTORY).to_crs(perim.crs)
    pts["obs_date"] = pd.to_datetime(pts["obs_date"], errors="coerce")
    perim_u = perim.union_all()
    in_window = pts[pts.within(perim_u.buffer(POINT_PAD_M))]

    # A point seen BEFORE this fire is a flow from some earlier event that
    # happens to share the ground. It is not evidence about this burn, so it
    # is neither mapped nor counted -- only reported, so the drop is visible.
    predates = in_window[in_window["obs_date"] <= ig]
    usable = in_window.drop(index=predates.index)
    inside = usable[usable.within(perim_u)]
    near = usable.drop(index=inside.index)
    print(f"{fire_name} · {source} {asmnt} assessment · ignition {ig.date()}", flush=True)
    print(f"{len(inside)} inventory point(s) inside the perimeter, "
          f"{len(near)} within {POINT_PAD_M:.0f} m of it"
          + (f"; {len(predates)} dropped as pre-dating the fire" if len(predates) else ""),
          flush=True)
    if not len(inside) and not len(near):
        print("no post-fire inventory points here -- nothing to compare against", flush=True)
        return None

    OUT = paths.products_dir("hindcast", event_id.upper())
    b5070 = tuple(perim.to_crs("EPSG:5070").total_bounds)
    pad = 3500.0
    dem = statewide.dem_for_unit((b5070[0] - pad, b5070[1] - pad,
                                  b5070[2] + pad, b5070[3] + pad), TILE_DIR)
    # Injected route ON PURPOSE (see module docstring): passing the bundle's
    # own perimeter and dNBR is what makes barc_breaks take effect.
    result = assess.run_observed(
        event_id, dem=dem, out_dir=OUT,
        kf_polygons=paths.raw_dir("ssurgo") / "nv_ssurgo_kf.gpkg",
        perimeter=perim.to_crs(dem.crs), dnbr=str(bundle["dnbr"]),
        barc_breaks=breaks)
    seg = gpd.read_file(result["paths"]["segments"]).to_crs("EPSG:4326")
    print(f"{len(seg):,} segments modelled", flush=True)

    # --- what did the model predict where a flow actually happened? ---------
    allpts = pd.concat([inside.assign(where="inside perimeter"),
                        near.assign(where=f"within {POINT_PAD_M:.0f} m")])
    a5070, s5070 = allpts.to_crs("EPSG:5070"), seg.to_crs("EPSG:5070")
    hit = gpd.sjoin_nearest(a5070, s5070[["P_24mmh", "I15_50", "H_24mmh", "geometry"]],
                            how="left", max_distance=MATCH_MAX_M,
                            distance_col="dist_m")
    hit = hit[~hit.index.duplicated(keep="first")]
    matched = hit[hit["P_24mmh"].notna()]
    print(f"{len(matched)}/{len(allpts)} point(s) within {MATCH_MAX_M:.0f} m of a "
          f"modelled channel", flush=True)
    if len(matched):
        print(f"  predicted P at the design storm: median "
              f"{matched['P_24mmh'].median():.2f} "
              f"(range {matched['P_24mmh'].min():.2f}-{matched['P_24mmh'].max():.2f})",
              flush=True)

    # --- map ----------------------------------------------------------------
    w, s, e, n = perim.to_crs(4326).total_bounds
    padd = max(0.02, 0.12 * max(e - w, n - s))
    bounds = (w - padd, s - padd, e + padd, n + padd)
    tr, shape, extent = mc.grid(bounds, res=0.0002)
    hs = mc.hillshade(tr, shape)
    sev_rgba, sev_grid = severity_rgba(bundle["dnbr"], breaks, shape, tr, "EPSG:4326")

    P = seg["P_24mmh"].to_numpy()
    T = seg["I15_50"].to_numpy()
    lo, hi = np.percentile(T[np.isfinite(T)], [2, 98])
    PCMAP, PNORM = ListedColormap(P_COLORS), BoundaryNorm([0.0, *P_BREAKS, 1.0], 5)
    per4326 = perim.to_crs(4326)
    key = mc.clear_corner(tuple(extent), per4326.geometry, size=(0.34, 0.20))

    fig, axes = plt.subplots(1, 3, figsize=(22, 9.5), dpi=140)
    for ax, mode in zip(axes, ("severity", "likelihood", "threshold")):
        ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=tuple(extent), zorder=0)
        if mode == "severity":
            ax.imshow(sev_rgba, extent=tuple(extent), zorder=3, interpolation="nearest")
            handles = [Patch(facecolor=np.array(c) / 255.0, alpha=SEV_ALPHA,
                            edgecolor="0.3", label=l)
                      for c, l in zip(burn.BAER_CLASS_COLORS, SEV_LEGEND)]
            if len(inside):
                handles.append(Line2D([0], [0], marker="^", color="none",
                                      markerfacecolor="white", markeredgecolor="black",
                                      markersize=9, label="observed flow, in perimeter"))
            if len(near):
                handles.append(Line2D([0], [0], marker="^", color="none",
                                      markerfacecolor="none", markeredgecolor="black",
                                      markersize=9,
                                      label=f"observed, within {POINT_PAD_M/1000:g} km"))
            ax.legend(handles=handles,
                      title=f"BARC class (breaks {'/'.join(f'{b/1000:g}' for b in breaks)} dNBR)",
                      loc=key["loc"], fontsize=8, title_fontsize=8)
            ax.set_title(f"Burn severity — {source} dNBR at the calibrated break", fontsize=12)
        elif mode == "likelihood":
            seg.sort_values("P_24mmh").plot(ax=ax, column="P_24mmh", cmap=PCMAP,
                                            norm=PNORM, linewidth=1.5, zorder=5)
            ax.legend(handles=[Line2D([0], [0], color=c, lw=3, label=l)
                               for c, l in zip(P_COLORS, P_LABELS)],
                      title=f"P(debris flow) at {DESIGN_I15:g} mm/h",
                      loc=key["loc"], fontsize=8, title_fontsize=8)
            ax.set_title("Debris-flow likelihood at the design storm", fontsize=12)
        else:
            seg.sort_values("I15_50", ascending=False).plot(
                ax=ax, column="I15_50", cmap="plasma_r", linewidth=1.5,
                vmin=lo, vmax=hi, zorder=5, legend=False)
            sm = ScalarMappable(norm=Normalize(vmin=lo, vmax=hi), cmap="plasma_r")
            sm.set_array([])
            # Inset, not attached: an external colourbar steals width from
            # this panel alone and the three maps stop being the same size,
            # which is the one thing a reader compares them by.
            mc.inset_colorbar(ax, sm, "triggering $I_{15}$ (mm/h)",
                              corner=key["loc"])
            ax.set_title("Rainfall intensity that triggers a debris flow", fontsize=12)
        per4326.boundary.plot(ax=ax, color="#56B4E9", linewidth=1.8, zorder=6,
                              path_effects=mc.fire_style("current")["path_effects"])
        # White, for the same reason as the corridor sheets: every other hue
        # on these panels already means something.
        if len(inside):
            i4326 = inside.to_crs(4326)
            ax.scatter(i4326.geometry.x, i4326.geometry.y, marker="^", s=95,
                      facecolor="white", edgecolor="black", linewidth=1.1, zorder=9.6)
        if len(near):
            n4326 = near.to_crs(4326)
            ax.scatter(n4326.geometry.x, n4326.geometry.y, marker="^", s=95,
                      facecolor="none", edgecolor="black", linewidth=1.3, zorder=9.6)
        mc.style_axes(ax, extent, step=0.05)

    obs = allpts["obs_date"].dropna()
    obs_span = (f"{obs.min().date()} to {obs.max().date()}" if len(obs) else "undated")
    from rasterio.features import geometry_mask
    inside_mask = geometry_mask([per4326.union_all()], out_shape=shape,
                                transform=tr, invert=True)
    frac = class_fractions(sev_grid, breaks, inside_mask)
    modhigh = frac["moderate"] + frac["high"]

    fig.subplots_adjust(top=0.94, bottom=0.05, left=0.03, right=0.97)
    mc.caption(fig, (
        f"{source} {asmnt.lower()} assessment dNBR ({event_id}), classified at the "
        f"{region} calibrated break "
        f"({'/'.join(f'{b/1000:g}' for b in breaks)} dNBR), run through the same hazard chain "
        f"as the operational sheets: {len(seg):,} stream segments, {modhigh:.0%} of the burned "
        f"area moderate or high. Left: burn severity. Middle: modelled debris-flow likelihood at the "
        f"{DESIGN_I15:g} mm/h design storm (median {np.nanmedian(P):.2f}, "
        f"{int((P >= 0.5).sum()):,} segments at or above 50%). Right: the 15-minute intensity at "
        f"which that likelihood reaches 50% (median {np.nanmedian(T):.1f} mm/h). Filled triangles "
        f"are inventory debris flows inside the perimeter ({len(inside)}), open triangles within "
        f"{POINT_PAD_M/1000:g} km of it ({len(near)}); imagery dates {obs_span}, against an "
        f"ignition of {ig.date()}. Those dates are when a flow was VISIBLE in imagery, not when it "
        f"happened, so they bound the post-fire lag from above rather than measuring it. Every "
        f"panel is a prediction conditional on this burn; the triangles are the only observation."
    ), label=f"{fire_name} fire, {ig.year}.", fontsize=9.5)
    mc.save(fig, f"hindcast_{name_slug(fire_name)}_{cal_name}")
    plt.close(fig)

    summary = {
        "event_id": event_id, "fire_name": fire_name,
        "map_prog": source, "assessment_type": asmnt,
        "calibration": cal_name, "region": region,
        "barc_breaks_x1000": breaks, "segments": int(len(seg)),
        "ignition_date": str(ig.date()),
        "severity_source": f"{source} dnbr.tif classified at the calibrated regional break",
        "class_fraction_ours_within_perimeter": {k: round(v, 4) for k, v in frac.items()},
        "class_fraction_mtbs_dnbr6": mtbs_dnbr6_fractions(event_id),
        "P_design": {"median": round(float(np.nanmedian(P)), 3),
                     "n_ge_0.5": int((P >= 0.5).sum())},
        "I15_50_mmh": {"median": round(float(np.nanmedian(T)), 1)},
        "inventory": {
            "inside_perimeter": int(len(inside)),
            f"within_{int(POINT_PAD_M)}m": int(len(near)),
            "dropped_as_pre_dating_the_fire": int(len(predates)),
            "matched_to_a_channel": int(len(matched)),
            "match_max_m": MATCH_MAX_M,
            "predicted_P_at_points": ([round(float(v), 3)
                                       for v in matched["P_24mmh"]] if len(matched) else []),
            "predicted_P_median": (round(float(matched["P_24mmh"].median()), 3)
                                   if len(matched) else None),
        },
    }
    (OUT / f"hindcast_summary_{cal_name}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def class_fractions(grid, breaks, inside_mask):
    """Class shares of the BURNED AREA, not of the map window.

    The dNBR raster covers the fire plus a rectangular buffer, so taking the
    fraction over every finite pixel answers "how much of this rectangle
    burned hot", which is mostly a statement about how much unburned desert
    the window happens to include (83% "unburned" on Broom Canyon). Clipped
    to the perimeter it answers the question the number is read as.
    """
    v = grid[np.isfinite(grid) & inside_mask]
    if not v.size:
        return {lab: float("nan") for lab in ("unburned", "low", "moderate", "high")}
    idx = np.digitize(v, list(breaks))
    return {lab: float((idx == i).mean())
            for i, lab in enumerate(("unburned", "low", "moderate", "high"))}


def mtbs_dnbr6_fractions(event_id):
    """MTBS's own analyst classification of the same fire, for comparison."""
    bundle = mtbs.fire_bundle(event_id)
    if "dnbr6" not in bundle:
        return None
    with rasterio.open(bundle["dnbr6"]) as ds:
        barc = mtbs.dnbr6_to_barc4(ds.read(1))
    valid = barc > 0
    if not valid.any():
        return None
    return {lab: round(float((barc[valid] == i).mean()), 4)
            for i, lab in enumerate(("unburned", "low", "moderate", "high"), start=1)}


def mtbs_ig_date(event_id):
    """Ignition date straight out of the event_id (MTBS encodes it)."""
    return f"{event_id[-8:-4]}-{event_id[-4:-2]}-{event_id[-2:]}"


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cal = args[1] if len(args) > 1 else CAL_DEFAULT
    if "--all" in sys.argv:
        # Only fires a point was observed AFTER: a point that predates the
        # fire it sits in is a flow from some earlier event, and hindcasting
        # this fire says nothing about it.
        j = pd.read_csv(POINT_FIRES)
        ids = sorted(j.loc[j["fire_before_flow"] == True, "event_id"].dropna().unique())
    else:
        ids = [args[0]]
    for eid in ids:
        try:
            run_one(eid, cal)
        except FileNotFoundError as exc:
            print(f"{eid}: no MTBS bundle staged yet ({exc})", flush=True)
