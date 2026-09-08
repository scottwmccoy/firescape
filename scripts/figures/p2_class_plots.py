"""Per-vegetation-class severity plots + cached era-matched pixel samples.

Panel A: what Nevada is made of — statewide class areas colored by the BARC
class each simulates at P_dsim=0.51.
Panel B: observed MTBS dNBR distributions per class (era-matched 2017+ fires,
LF2016 vegetation) against the Staley 2018 vs Nevada-refit simulated values.
Side product: interim/calib/refit_samples_pilot.parquet (reused by the ML probe).
"""
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray as rxr
from rasterio.enums import Resampling

from firescape import config, mtbs, paths, severity

OK = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00',
      '#CC79A7', '#000000']
BARC_COLORS = {1: "#009E73", 2: "#F0E442", 3: "#E69F00", 4: "#D55E00"}
BARC_LABELS = {1: "unburned/very low", 2: "low", 3: "moderate", 4: "high"}

cal = config.packaged_calibration("pilot_v2").region("pilot")
PD = cal.pdsim
staley = severity.load_cdf_table()
merged = severity.load_cdf_table(table="nv_merged")
OUT = paths.products_dir("prefire", "statewide_v0")
FIG = paths.figures_dir()

# ---- era-matched samples (cache) -------------------------------------------
cache = paths.interim_dir("calib") / "refit_samples_pilot.parquet"
if cache.exists():
    samp = pd.read_parquet(cache)
    print(f"loaded {len(samp):,} cached sample pixels")
else:
    FIRE_SET = (paths.package_data("calibration", "fire_sets", "pilot_v1.csv"))
    EVT_DIR = paths.raw_dir("landfire") / "LF2016_EVT_pilot"
    fires = pd.read_csv(FIRE_SET)
    use = fires[fires["include"] & (fires["ig_year"] >= 2017)]
    vat = gpd.read_file(EVT_DIR / "LF2016_EVT_pilot.tif.vat.dbf")
    mapping, _ = severity.remap_crosswalk(vat)
    evt_full = rxr.open_rasterio(EVT_DIR / "LF2016_EVT_pilot.tif",
                                 masked=False).squeeze()
    parts = []
    for _, row in use.iterrows():
        eid = row["event_id"]
        try:
            bundle = mtbs.fire_bundle(eid)
        except FileNotFoundError:
            continue
        dnbr = rxr.open_rasterio(bundle["dnbr"], masked=True).squeeze()
        evt_on = evt_full.rio.reproject_match(dnbr, resampling=Resampling.nearest)
        perim = gpd.read_file(bundle["burn_area"]).to_crs(dnbr.rio.crs)
        inside = dnbr.rio.clip(perim.geometry, drop=False)
        d = inside.values.astype("float64")
        e = evt_on.values
        ok = np.isfinite(d) & (d > -1000) & (d < 1000) & (e > 0)
        if "dnbr6" in bundle:
            d6 = rxr.open_rasterio(bundle["dnbr6"], masked=False).squeeze()
            ok &= np.isin(d6.rio.reproject_match(
                dnbr, resampling=Resampling.nearest).values, (1, 2, 3, 4))
        codes = severity.apply_crosswalk(e[ok], mapping)
        parts.append(pd.DataFrame({"event_id": eid,
                                   "code": codes.astype("int32"),
                                   "dnbr": d[ok].astype("float32")}))
        print(f"  {eid}: {int(ok.sum()):,} px", flush=True)
    samp = pd.concat(parts, ignore_index=True)
    samp["event_id"] = samp["event_id"].astype("category")
    samp.to_parquet(cache, index=False)
    print(f"cached {len(samp):,} pixels -> {cache}")

# ---- figure -----------------------------------------------------------------
tab = pd.read_csv(OUT / "severity_class_table.csv")
top = tab[tab["barc4_p51"] > 0].head(15).iloc[::-1]

fig, axes = plt.subplots(1, 2, figsize=(17.5, 8.5), dpi=140,
                         gridspec_kw={"width_ratios": [1, 1.25]})

ax = axes[0]
colors = [BARC_COLORS[c] for c in top["barc4_p51"]]
ax.barh(np.arange(len(top)), top["area_km2"] / 1000, color=colors,
        edgecolor="black", linewidth=0.4)
for i, (_, r) in enumerate(top.iterrows()):
    star = "*" if r["source"] == "nv_refit" else ""
    ax.text(r["area_km2"] / 1000 + 0.8, i,
            f"{r['simdnbr_p51']:.0f}{star}", va="center", fontsize=8)
ax.set_yticks(np.arange(len(top)))
ax.set_yticklabels([n[:42] for n in top["evt_name"]], fontsize=8)
ax.set_xlabel("statewide area (10$^3$ km$^2$)")
ax.set_title("Nevada's vegetation and the severity it simulates\n"
             f"labels: simulated dNBR at P$_{{dsim}}$={PD} "
             "(* = Nevada-refit CDF)", fontsize=11)
handles = [plt.Rectangle((0, 0), 1, 1, fc=BARC_COLORS[c], ec="black", lw=0.4,
                         label=BARC_LABELS[c]) for c in (1, 2, 3, 4)]
ax.legend(handles=handles, title="simulated BARC class", loc="lower right",
          fontsize=8, title_fontsize=8)

ax = axes[1]
counts = samp.groupby("code").size().sort_values(ascending=False)
codes = [c for c in counts.index if c in merged.index][:12]
names = dict(zip(tab["cdf_code"], tab["evt_name"]))
data, labels = [], []
for c in codes:
    v = samp.loc[samp["code"] == c, "dnbr"].to_numpy()
    if len(v) > 40000:
        v = np.random.default_rng(7).choice(v, 40000, replace=False)
    data.append(v)
    labels.append(f"{names.get(c, staley.at[c, 'CLASSNAME'] if c in staley.index else c)}"[:34]
                  + f"\n(n={counts[c]/1e3:,.0f}k)")
pos = np.arange(len(codes))
parts = ax.violinplot(data, positions=pos, widths=0.8, showextrema=False)
for b in parts["bodies"]:
    b.set_facecolor("#BBBBBB"); b.set_alpha(0.6)
obs_med = [float(np.median(d)) for d in data]
st_sim = [severity.weibull_dnbr(PD, staley.at[c, "Weibull_Lambda_Scale"],
                                staley.at[c, "Weibull_Kappa_Shape"])
          if c in staley.index else np.nan for c in codes]
nv_sim = [severity.weibull_dnbr(PD, merged.at[c, "Weibull_Lambda_Scale"],
                                merged.at[c, "Weibull_Kappa_Shape"])
          for c in codes]
ax.scatter(pos, obs_med, marker="_", s=600, color="black", linewidth=2.2,
           label="observed median (MTBS, era-matched)", zorder=5)
ax.scatter(pos, st_sim, marker="o", s=46, facecolor="none",
           edgecolor=OK[4], linewidth=1.8, label="Staley 2018 @ P$_{dsim}$", zorder=6)
ax.scatter(pos, nv_sim, marker="D", s=40, color=OK[5],
           label="Nevada refit @ P$_{dsim}$", zorder=6)
for y, lbl in zip(cal.barc_breaks, ("unburned-low", "low-mod", "mod-high")):
    ax.axhline(y, color="#888888", linestyle=":", linewidth=0.9)
    ax.text(len(codes) - 0.4, y + 8, lbl, fontsize=7, color="#555555",
            ha="right")
ax.set_xticks(pos)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
ax.set_ylabel("dNBR (x1000)")
ax.set_ylim(-150, 900)
ax.legend(loc="upper right", fontsize=8)
ax.set_title("Observed burn-severity distributions by vegetation class\n"
             f"{samp['event_id'].nunique()} era-matched fires, "
             f"{len(samp)/1e6:.1f}M pixels — why the refit matters", fontsize=11)

fig.suptitle("firescape intermediate product: simulated burn severity by "
             "vegetation class (statewide v0 config)", y=1.0, fontsize=13)
fig.tight_layout()
png = FIG / "severity_by_class.png"
fig.savefig(png, bbox_inches="tight")
print("wrote", png)
