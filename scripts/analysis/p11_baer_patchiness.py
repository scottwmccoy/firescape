"""Does the SPATIAL texture of the dNBR map predict which way a fire misses BAER?

Hypothesis: a BAER team maps soil burn severity as hand-drawn polygons, so it
smooths.  If our moderate+ pixels are speckled through a low matrix, smoothing
erases them -> BAER cooler -> our map reads hot.  If our low pixels are
speckled through a hot matrix, smoothing fills them -> our map reads cool.
So the SIGN of the gap should track which class is the speckled minority.
All local: dNBR rasters only.
"""
import glob, warnings
warnings.filterwarnings("ignore")
import geopandas as gpd, numpy as np, pandas as pd, rasterio, rasterio.features
from scipy import ndimage, stats
from firescape import paths, severity

CAL = paths.products_dir("calibration")
d = pd.read_csv(CAL / "baer_outlier_diagnostics.csv").drop_duplicates("event_id")
K = np.ones((3, 3)); K[1, 1] = 0

rows = []
for _, f in d.iterrows():
    b = paths.raw_dir("mtbs", "fires", f.event_id)
    with rasterio.open(sorted(glob.glob(str(b / "*_dnbr.tif")))[0]) as ds:
        dn = ds.read(1).astype("float64")
        ok = severity.valid_dnbr(dn, ds.nodata)
        per = gpd.read_file(sorted(b.glob("*_burn_area.shp"))[0]).to_crs(ds.crs)
        ok &= rasterio.features.geometry_mask(per.geometry, dn.shape, ds.transform, invert=True)
    hot = ok & (dn >= f.our_break)
    cool = ok & (dn < f.our_break)
    nb_ok = ndimage.convolve(ok.astype(float), K, mode="constant")
    nb_hot = ndimage.convolve(hot.astype(float), K, mode="constant")
    frac_hot_nb = np.where(nb_ok > 0, nb_hot / np.maximum(nb_ok, 1), np.nan)
    # a pixel is "speckle" if fewer than 3 of its valid neighbours share its class
    hot_speckle = hot & (frac_hot_nb < 3 / 8)
    cool_speckle = cool & (frac_hot_nb > 5 / 8)
    # local roughness of dNBR itself, relative to the fire's spread
    sm = ndimage.uniform_filter(np.where(ok, dn, 0), 3) / np.maximum(ndimage.uniform_filter(ok.astype(float), 3), 1e-9)
    rough = np.nanstd((dn - sm)[ok]) / max(np.nanstd(dn[ok]), 1)
    rows.append({"event_id": f.event_id, "incid_name": f.incid_name, "state": f.state,
                 "gap": f.modhi_gap_ours_minus_baer * 100, "our_modhi": f.our_modhi,
                 "hot_speckle": hot_speckle.sum() / max(hot.sum(), 1),
                 "cool_speckle": cool_speckle.sum() / max(cool.sum(), 1),
                 "roughness": rough, "km2": f.km2, "dnbr_p50": f.dnbr_p50,
                 "spread": (f.dnbr_p90 - f.dnbr_p10) / max(abs(f.dnbr_p50), 1)})
p = pd.DataFrame(rows).dropna(subset=["gap"])
p["net_speckle"] = p.hot_speckle - p.cool_speckle
p.to_csv(CAL / "baer_patchiness.csv", index=False)

print(f"n={len(p)}")
print("prediction if BAER smooths: gap ~ +hot_speckle, ~ -cool_speckle, ~ +net_speckle\n")
for x in ["hot_speckle", "cool_speckle", "net_speckle", "roughness", "spread", "km2", "our_modhi"]:
    r = stats.spearmanr(p[x], p.gap)
    print(f"  gap vs {x:13s} rho={r.statistic:+.3f}  p={r.pvalue:.3f}")
print("\nspeckle fractions by sign of the gap:")
p["side"] = np.where(p.gap > 20, "our map >20 hot", np.where(p.gap < -20, "our map >20 cool", "within 20"))
print(p.groupby("side").agg(n=("gap", "size"), hot_speckle=("hot_speckle", "median"),
      cool_speckle=("cool_speckle", "median"), roughness=("roughness", "median"),
      km2=("km2", "median"), our_modhi=("our_modhi", "median")).round(3).to_string())
print("\ncase fires:")
print(p[p.incid_name.isin(["DAVIS", "SLATER", "TABOOSE", "MOUNTAIN VIEW", "CHERRY", "MAHOGANY", "HUGHES", "CORTA"])]
      [["incid_name", "gap", "our_modhi", "hot_speckle", "cool_speckle", "roughness"]].round(3).to_string(index=False))
