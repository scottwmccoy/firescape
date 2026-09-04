"""Why the pre/post image pairs are spaced the way they are.

Landsat's revisit is 8-16 days and Sentinel-2's is 5, so a year between a
fire's two images is not a data-availability limit.  It is the ANNIVERSARY
rule: dNBR differences the two scenes, so anything that is not the fire --
sun angle, green-up, cure-off -- has to be matched between them.  MTBS
therefore pairs scenes at the same point in the ANNUAL cycle, and an extended
assessment's post-fire scene is the next season's green-up peak.  The span is
then forced to a whole number of years, and the number of years is set by
which anniversary windows were usable.
"""
import pathlib
import numpy as np
import pandas as pd

from firescape import paths

OUT = paths.figures_dir("baer_outliers"); OUT.mkdir(parents=True, exist_ok=True)
d = pd.read_csv(paths.products_dir("calibration") / "baer_outlier_diagnostics.csv").drop_duplicates("event_id")
for c in ("ig_date", "pre_date", "post_date"):
    d[c] = pd.to_datetime(d[c])
d["pre_doy"], d["post_doy"] = d.pre_date.dt.dayofyear, d.post_date.dt.dayofyear
d["doy_gap"] = np.minimum(abs(d.post_doy - d.pre_doy), 365 - abs(d.post_doy - d.pre_doy))
d["yrs"] = (d.image_span_d / 365.25).round().astype(int)

W, H, L, R, T, B = 880, 470, 62, 20, 26, 48
px = lambda v: L + v / 366 * (W - L - R)
py = lambda v: (H - B) - v / 366 * (H - T - B)
COL = {0: "var(--warn)", 1: "var(--baer)", 2: "var(--teal)", 3: "var(--accent)"}
LAB = {0: "same season", 1: "1 year apart", 2: "2 years", 3: "3 years"}

p = [f'    <svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Day of year of each '
     f'fire\'s pre-fire image against its post-fire image, showing the pairs land on the 1:1 line">']
p.append(f'  <line x1="{px(0):.1f}" y1="{py(0):.1f}" x2="{px(366):.1f}" y2="{py(366):.1f}" '
         f'stroke="var(--rule)" stroke-dasharray="4 4"/>')
p.append(f'  <text x="{px(58):.1f}" y="{py(70):.1f}" class="strip-axis" fill="var(--faint)" '
         f'transform="rotate(-45 {px(58):.1f} {py(70):.1f})">same day of year</text>')
for _, f in d.iterrows():
    p.append(f'  <circle cx="{px(f.pre_doy):.1f}" cy="{py(f.post_doy):.1f}" r="4" '
             f'fill="{COL.get(min(f.yrs, 3), "var(--faint)")}" fill-opacity=".75"/>')
MON = [(1, "Jan"), (60, "Mar"), (121, "May"), (182, "Jul"), (244, "Sep"), (305, "Nov")]
for v, lab in MON:
    p.append(f'  <line x1="{px(v):.1f}" y1="{py(0):.1f}" x2="{px(v):.1f}" y2="{py(0)+4:.1f}" stroke="var(--rule)"/>'
             f'<text x="{px(v):.1f}" y="{py(0)+17:.1f}" text-anchor="middle" class="strip-axis" fill="var(--faint)">{lab}</text>')
    p.append(f'  <text x="{L-8}" y="{py(v)+3.5:.1f}" text-anchor="end" class="strip-axis" fill="var(--faint)">{lab}</text>')
p.append(f'  <line x1="{L}" y1="{py(0):.1f}" x2="{W-R}" y2="{py(0):.1f}" stroke="var(--rule)"/>')
p.append(f'  <line x1="{L}" y1="{T}" x2="{L}" y2="{py(0):.1f}" stroke="var(--rule)"/>')
p.append(f'  <text x="{(L+W-R)/2:.1f}" y="{H-9}" text-anchor="middle" class="strip-axis" '
         f'fill="var(--muted)">day of year of the PRE-fire image</text>')
p.append(f'  <text transform="translate(14,{(T+py(0))/2:.1f}) rotate(-90)" text-anchor="middle" '
         f'class="strip-axis" fill="var(--muted)">day of year of the POST-fire image</text>')
for i, y in enumerate(sorted(set(d.yrs.clip(upper=3)))):
    n = int((d.yrs.clip(upper=3) == y).sum())
    yy = T + 14 + 17 * i
    p.append(f'  <circle cx="{px(12):.1f}" cy="{yy-3.5:.1f}" r="4" fill="{COL[y]}"/>'
             f'<text x="{px(24):.1f}" y="{yy:.1f}" class="strip-axis" fill="var(--muted)">'
             f'{LAB[y]} &#183; n={n}</text>')
e = d[d.assess_type.eq("Extended")]
p.append(f'  <text x="{W-R:.1f}" y="{T+14:.1f}" text-anchor="end" class="strip-axis" fill="var(--faint)">'
         f'{(e.doy_gap <= 21).mean()*100:.0f}% of extended assessments pair scenes within 3 weeks of the '
         f'same day of year</text>')
p.append(f'  <text x="{W-R:.1f}" y="{T+31:.1f}" text-anchor="end" class="strip-axis" fill="var(--faint)">'
         f'{(e.post_date.dt.month.isin([6, 7])).mean()*100:.0f}% take the post-fire scene in June or July '
         f'&#8212; one green-up window a year</text>')
p.append("    </svg>")
(OUT / "imagery_cadence.svg").write_text("\n".join(p))
print(f"n={len(d)}  extended within 3 wk of same DOY: {(e.doy_gap<=21).mean()*100:.0f}%")
print("  span in whole years:", d.yrs.value_counts().sort_index().to_dict())
print(f"-> {OUT/'imagery_cadence.svg'}")
