"""Stallion: the storm that fell, and which segments it should have triggered."""
import json
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.lines import Line2D
from rasterio.warp import Resampling, reproject

import geopandas as gpd
from firescape import paths, plotting as mc
from stormscape import relief

FIRE = sys.argv[1] if len(sys.argv) > 1 else "Stallion"
CAL = sys.argv[2] if len(sys.argv) > 2 else "statewide_v1_1"
STORM = (paths.research_root() / "2026_Bug_Stalion" / "storms" / "composite_20260812-20260814")
SRC = paths.products_dir("forecast", f"{FIRE.lower()}_{CAL}")
meta = json.loads((SRC / "storm_response_summary.json").read_text())

seg = gpd.read_file(SRC / f"{FIRE.lower()}_storm_response.gpkg").to_crs("EPSG:4326")
per = gpd.read_file(sorted(paths.raw_dir("perimeters").glob("wfigs_current_*.geojson"))[-1])
name_col = next(c for c in per.columns if c.endswith("IncidentName"))
per = per[per[name_col].str.fullmatch(FIRE, case=False, na=False)].to_crs("EPSG:4326")

w, s, e, n = per.total_bounds
pad = 0.035
tr, shape, extent = mc.grid((w - pad, s - pad, e + pad, n + pad), res=0.0002)
hs = relief.shaded_relief(
    sorted((paths.cache_root() / "3dep_tiles").glob("USGS_13_*.tif")),
    tr, shape, crs="EPSG:4326", shade_res_m=20.0)
im_extent = (extent[0], extent[1], extent[2], extent[3])

# storm field onto the display grid (bilinear: MRMS is a smooth 1 km field)
with rasterio.open(f"{STORM}/rasters/BSE3day_i15max.tif") as ds:
    rain = np.full(shape, np.nan, dtype="float32")
    reproject(ds.read(1, masked=True).filled(np.nan).astype("float32"), rain,
              src_transform=ds.transform, src_crs=ds.crs, src_nodata=np.nan,
              dst_transform=tr, dst_crs="EPSG:4326", dst_nodata=np.nan,
              resampling=Resampling.bilinear)

fired = seg["exceed_ratio"] >= 1.0
fig, axes = plt.subplots(1, 2, figsize=(19, 10.5), dpi=140)

ax = axes[0]
ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=im_extent, zorder=0)
im = ax.imshow(np.ma.masked_invalid(rain), cmap="YlGnBu", extent=im_extent,
               alpha=mc.MAX_LAYER_ALPHA, zorder=2, interpolation="bilinear")
cb = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
cb.set_label("peak 15-min intensity (mm/h), 12–14 Aug", fontsize=9)
seg.plot(ax=ax, color="0.35", linewidth=0.5, zorder=4)
per.boundary.plot(ax=ax, color="#D55E00", linewidth=1.8, zorder=6)
ax.set_title("The storm: MRMS peak $I_{15}$ over the burn\n"
             f"median over segments {meta['observed_i15_mmh']['median']} mm/h, "
             f"max {meta['observed_i15_mmh']['max']} mm/h", fontsize=10.5)

ax = axes[1]
ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=im_extent, zorder=0)
seg[~fired].plot(ax=ax, color="0.55", linewidth=0.7, zorder=4)
hot = seg[fired]
im = hot.plot(ax=ax, column="P_observed", cmap="inferno_r", linewidth=1.9,
              vmin=0.5, vmax=1.0, zorder=5, legend=True,
              legend_kwds={"shrink": 0.5, "pad": 0.02,
                           "label": "modelled likelihood at the observed storm"})
per.boundary.plot(ax=ax, color="#D55E00", linewidth=1.8, zorder=6)
ax.legend(handles=[Line2D([0], [0], color="0.55", lw=2, label="below threshold"),
                   Line2D([0], [0], color="#C1272D", lw=3,
                          label=f"exceeded ({int(fired.sum()):,})")],
          loc="lower left", fontsize=8)
ax.set_title(f"The prediction: {int(fired.sum()):,} of {len(seg):,} segments "
             f"exceeded their triggering intensity\n"
             f"{meta['segments_P_observed_over_0.9']} above 0.9 likelihood — "
             "search these first", fontsize=10.5)

for ax in axes:
    mc.style_axes(ax, extent, step=0.1)

d = meta["peak_day_of_exceeding"]
fig.suptitle(
    f"{FIRE} fire — did it already happen? Segments scored against the storm "
    "that actually fell, 12–14 Aug 2026\n"
    f"peak day among exceeding segments: 12 Aug {d['S12Aug']}, "
    f"13 Aug {d['S13Aug']}, 14 Aug {d['S14Aug']} · "
    "simulated severity (fire still burning, no BARC yet) — treat as a search "
    "list, not a confirmed inventory", y=0.98, fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))
mc.save(fig, f"{FIRE.lower()}_storm_response")
