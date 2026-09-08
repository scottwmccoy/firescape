"""Stallion on observed severity: what BRISK saw, and what the storm should have done.

Three panels: the measured burn severity, the debris-flow likelihood it implies
under the rain that actually fell, and the same likelihood from the pre-fire
simulated-severity run for comparison.
"""
import json
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.lines import Line2D
from rasterio.features import geometry_mask
from rasterio.warp import Resampling, reproject

import geopandas as gpd
from firescape import paths, plotting as mc
from stormscape import relief
from stormscape import burn

import sys
FIRE = sys.argv[1] if len(sys.argv) > 1 else "Stallion"
CAL = sys.argv[2] if len(sys.argv) > 2 else "statewide_v1_1"
STORM = (paths.research_root() / "2026_Bug_Stalion" / "storms" / "composite_20260812-20260814")
OBS = paths.products_dir("forecast", f"{FIRE.lower()}_observed")
SIM = paths.products_dir("forecast", f"{FIRE.lower()}_{CAL}")
DAYS = ["S12Aug", "S13Aug", "S14Aug"]
smeta = json.loads((OBS / "observed_severity_summary.json").read_text())

seg = gpd.read_file(OBS / f"{FIRE.upper()}_segments.gpkg").to_crs("EPSG:4326")
sim = gpd.read_file(SIM / f"{FIRE.lower()}_storm_response.gpkg").to_crs("EPSG:4326")
per = gpd.read_file(
    sorted(paths.raw_dir("perimeters").glob("wfigs_current_*.geojson"))[-1])
ncol = next(c for c in per.columns if c.endswith("IncidentName"))
per = per[per[ncol].str.fullmatch(FIRE, case=False, na=False)].to_crs("EPSG:4326")

w, s, e, n = per.total_bounds
pad = 0.035
tr, shape, extent = mc.grid((w - pad, s - pad, e + pad, n + pad), res=0.0002)
hs = relief.shaded_relief(
    sorted((paths.cache_root() / "3dep_tiles").glob("USGS_13_*.tif")),
    tr, shape, crs="EPSG:4326", shade_res_m=20.0)
im_extent = (extent[0], extent[1], extent[2], extent[3])


def onto_grid(path, resampling=Resampling.bilinear, band=1):
    with rasterio.open(path) as ds:
        out = np.full(shape, np.nan, dtype="float32")
        reproject(ds.read(band, masked=True).filled(np.nan).astype("float32"), out,
                  src_transform=ds.transform, src_crs=ds.crs, src_nodata=np.nan,
                  dst_transform=tr, dst_crs="EPSG:4326", dst_nodata=np.nan,
                  resampling=resampling)
    return out


# --- storm rainfall over the observed-severity segments ---------------------
def raster_means(path, geoms):
    with rasterio.open(path) as ds:
        arr = ds.read(1, masked=True).filled(np.nan).astype("float64")
        t, crs, shp = ds.transform, ds.crs, ds.shape
    g = geoms.to_crs(crs)
    out = np.full(len(g), np.nan)
    for i, geom in enumerate(g.geometry):
        try:
            m = geometry_mask([geom], out_shape=shp, transform=t, invert=True,
                              all_touched=True)
        except Exception:
            continue
        v = arr[m][np.isfinite(arr[m])]
        if v.size:
            out[i] = v.mean()
    return out


buf = gpd.GeoDataFrame(seg.to_crs("EPSG:5070").buffer(250.0).to_frame("geometry"),
                       crs="EPSG:5070")
i15 = np.nanmax(np.stack(
    [raster_means(f"{STORM}/per_storm/{d}_i15max.tif", buf) for d in DAYS]
    + [raster_means(f"{STORM}/rasters/BSE3day_i15max.tif", buf)]), axis=0)

from pfdf.models import s17

B, Ct, Cf, Cs = (float(np.ravel(x)[0]) for x in s17.M1.parameters(durations=[15]))
T, F, S = (seg[c].to_numpy(float) for c in ("Terrain_M1", "Fire_M1", "Soil_M1"))
p_obs = 1.0 / (1.0 + np.exp(-(B + (Ct * T + Cf * F + Cs * S) * i15 * 0.25)))
seg["i15_obs"], seg["P_observed"] = i15, p_obs
seg["exceed_ratio"] = i15 / seg["I15_50"].to_numpy(float)
fired = seg["exceed_ratio"] >= 1.0
seg.to_file(OBS / f"{FIRE.lower()}_storm_response_observed.gpkg", driver="GPKG")

sim_fired = sim["exceed_ratio"] >= 1.0
stats = {
    "observed_severity": {
        "segments": int(len(seg)),
        "median_threshold_mmh": round(float(np.nanmedian(seg["I15_50"])), 1),
        "median_i15_observed": round(float(np.nanmedian(i15)), 1),
        "exceeding": int(fired.sum()),
        "pct_exceeding": round(100 * float(fired.mean()), 1),
        "P_over_0.5": int((p_obs > 0.5).sum()),
        "P_over_0.9": int((p_obs > 0.9).sum())},
    "simulated_severity": {
        "segments": int(len(sim)),
        "median_threshold_mmh": round(float(np.nanmedian(sim["I15_50"])), 1),
        "exceeding": int(sim_fired.sum()),
        "pct_exceeding": round(100 * float(sim_fired.mean()), 1),
        "P_over_0.5": int((sim["P_observed"] > 0.5).sum()),
        "P_over_0.9": int((sim["P_observed"] > 0.9).sum())},
    "brisk": smeta,
}
print(json.dumps(stats, indent=2), flush=True)
(OBS / "observed_vs_simulated.json").write_text(json.dumps(stats, indent=2))

# --- figure -----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(22, 10.5), dpi=140)

ax = axes[0]
ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=im_extent, zorder=0)
# severity_colors' norm has its boundaries in dNBR units and does the classing
# itself, so it takes the RAW dNBR -- feeding it classify() output puts every
# burned class above the 0.66 break and paints the whole scar "high".
dn = onto_grid(OBS / f"{FIRE.lower()}_brisk_dnbr.tif", Resampling.nearest)
cmap, norm, ticks, labels = burn.severity_colors("usgs")
im = ax.imshow(np.ma.masked_invalid(dn), cmap=cmap, norm=norm, extent=im_extent,
               alpha=0.75, zorder=2, interpolation="nearest")
cb = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02, ticks=ticks)
cb.ax.set_yticklabels(labels, fontsize=8)
per.boundary.plot(ax=ax, color="black", linewidth=1.4, zorder=6)
frac = smeta["class_fraction"]
ax.set_title("Observed burn severity — CIMSS BRISK dNBR\n"
             f"composite {smeta['scene_dates'][0]} (1 day old) · "
             f"{100*(frac['moderate-high']+frac['high']):.1f}% moderate-high or high",
             fontsize=10)

for ax, g, col, ttl in ((axes[1], seg, "P_observed", "observed severity"),
                        (axes[2], sim, "P_observed", "simulated severity (pre-fire)")):
    ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=im_extent, zorder=0)
    hot = g[g["exceed_ratio"] >= 1.0]
    g[g["exceed_ratio"] < 1.0].plot(ax=ax, color="0.55", linewidth=0.7, zorder=4)
    if len(hot):
        im = hot.plot(ax=ax, column=col, cmap="inferno_r", linewidth=1.9,
                      vmin=0.5, vmax=1.0, zorder=5, legend=True,
                      legend_kwds={"shrink": 0.5, "pad": 0.02,
                                   "label": "likelihood at the observed storm"})
    per.boundary.plot(ax=ax, color="#D55E00", linewidth=1.6, zorder=6)
    ax.legend(handles=[Line2D([0], [0], color="0.55", lw=2, label="below threshold"),
                       Line2D([0], [0], color="#C1272D", lw=3,
                              label=f"exceeded ({len(hot):,} of {len(g):,})")],
              loc="lower left", fontsize=8)
    ax.set_title(f"Debris-flow likelihood, 12–14 Aug storm\n{ttl}", fontsize=10)

for ax in axes:
    mc.style_axes(ax, extent, step=0.1)

o, m = stats["observed_severity"], stats["simulated_severity"]
fig.suptitle(
    f"{FIRE} fire — measured severity vs assumed severity, against the same storm\n"
    f"observed: {o['exceeding']:,} of {o['segments']:,} segments exceed "
    f"({o['pct_exceeding']}%) · simulated: {m['exceeding']:,} of {m['segments']:,} "
    f"({m['pct_exceeding']}%) · BRISK is 1 day old and under-reads magnitude, "
    "so the observed panel is a floor", y=0.98, fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))
mc.save(fig, f"{FIRE.lower()}_observed_severity")
