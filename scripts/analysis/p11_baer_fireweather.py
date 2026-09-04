"""gridMET fire weather at each fire's centroid over its first week.

The hypothesis under test: soil burn severity is set by residence time and
fuel consumption at the ground, which dNBR does not see, and both are
governed by how the fire burned -- wind-driven runs consume canopy fast and
heat soil little.  gridMET (4 km, daily) is the coarsest possible look at
that: wind speed, VPD, ERC and burning index on the ignition day and the six
days after it.  OPeNDAP, no key, one dataset open per (variable, year).
"""
import warnings; warnings.filterwarnings("ignore")
import glob
import geopandas as gpd, numpy as np, pandas as pd, xarray as xr
from shapely import make_valid
from firescape import paths

CAL = paths.products_dir("calibration")
d = pd.read_csv(CAL / "baer_outlier_diagnostics.csv").drop_duplicates("event_id")
d["ig_date"] = pd.to_datetime(d.ig_date)
cent = []
for eid in d.event_id:
    g = gpd.read_file(sorted(paths.raw_dir("mtbs", "fires", eid).glob("*_burn_area.shp"))[0])
    g["geometry"] = g.geometry.apply(make_valid)
    c = g.to_crs(4326).union_all().centroid
    cent.append({"event_id": eid, "lon": c.x, "lat": c.y})
d = d.merge(pd.DataFrame(cent), on="event_id")

BASE = "http://thredds.northwestknowledge.net:8080/thredds/dodsC/MET/{v}/{v}_{y}.nc"
VARS = {"vs": "wind", "vpd": "vpd", "erc": "erc", "bi": "bi"}
WIN = 7                                   # ignition day + 6

out = {e: {} for e in d.event_id}
for v, name in VARS.items():
    for y, g in d.groupby(d.ig_date.dt.year):
        try:
            ds = xr.open_dataset(BASE.format(v=v, y=int(y)), decode_times=True)
            var = [k for k in ds.data_vars if k != "crs"][0]
            for _, f in g.iterrows():
                t0 = f.ig_date
                sel = ds[var].sel(lat=f.lat, lon=f.lon, method="nearest").sel(
                    day=slice(t0, t0 + pd.Timedelta(days=WIN - 1)))
                vals = sel.values.astype(float)
                # a fire that spills into the next year gets the partial window; fine
                out[f.event_id][f"{name}_max"] = float(np.nanmax(vals)) if vals.size else np.nan
                out[f.event_id][f"{name}_mean"] = float(np.nanmean(vals)) if vals.size else np.nan
                out[f.event_id][f"{name}_d0"] = float(vals[0]) if vals.size else np.nan
            ds.close()
            print(f"  {v} {int(y)}: {len(g)} fires", flush=True)
        except Exception as ex:
            print(f"  !! {v} {int(y)}: {ex}", flush=True)

w = pd.DataFrame.from_dict(out, orient="index").rename_axis("event_id").reset_index()
w = d[["event_id", "incid_name", "state", "lon", "lat", "ig_date", "modhi_gap_ours_minus_baer",
       "pct_rank_ours", "days_post", "km2"]].merge(w, on="event_id")
w.to_csv(CAL / "baer_fireweather.csv", index=False)
print(f"\n{len(w)} fires -> {CAL/'baer_fireweather.csv'}")
