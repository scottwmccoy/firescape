"""Does the over/under-call against BAER depend on how long after the fire the
post-fire image was taken?

x is ``days_post`` -- ignition to post-fire image -- NOT the pre-to-post image
span, which is the sum of that and however stale the pre-fire baseline is, and
which reads as a much longer "delay" than any of these assessments actually
had (nothing here is imaged more than 462 days after ignition).
"""
import math, pathlib
import numpy as np, pandas as pd
from scipy import stats
from firescape import paths

OUT = paths.figures_dir("baer_outliers")
OUT.mkdir(parents=True, exist_ok=True)
u = (pd.read_csv(paths.products_dir("calibration") / "baer_outlier_diagnostics.csv")
     .drop_duplicates("event_id").dropna(subset=["modhi_gap_ours_minus_baer", "days_post"]))
u["gap"] = u.modhi_gap_ours_minus_baer * 100

W, H, L, R, T, B = 880, 468, 66, 20, 24, 48
XLO, XHI, YLIM = 1.6, 560.0, 45.0
px = lambda d: L + (math.log10(max(d, XLO)) - math.log10(XLO)) / (math.log10(XHI) - math.log10(XLO)) * (W - L - R)
py = lambda g: (H - B + T) / 2 - g / YLIM * ((H - T - B) / 2)

fresh, slow = u[u.days_post <= 60], u[u.days_post > 60]
rho = stats.spearmanr(u.days_post, u.gap)
mw = stats.mannwhitneyu(fresh.gap, slow.gap)
shift = np.median([a - b for a in fresh.gap for b in slow.gap])

p = [f'    <svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Scatter of the '
     f'moderate-or-high area gap between our map and BAER\'s against days from ignition to '
     f'the post-fire image, 124 fires">']
# group summaries behind the points
for g, lab, right in ((fresh, "imaged within 60 days", False),
                      (slow, "imaged 3-15 months later", True)):
    x0, x1 = px(g.days_post.min()), px(g.days_post.max())
    q1, q3, md = g.gap.quantile(.25), g.gap.quantile(.75), g.gap.median()
    p.append(f'  <rect x="{x0:.1f}" y="{py(q3):.1f}" width="{x1-x0:.1f}" height="{py(q1)-py(q3):.1f}" '
             f'fill="var(--scale-track)" fill-opacity=".75"/>')
    p.append(f'  <line x1="{x0:.1f}" y1="{py(md):.1f}" x2="{x1:.1f}" y2="{py(md):.1f}" '
             f'stroke="var(--ink)" stroke-width="2"/>')
    ax, anchor = (x1 - 5, "end") if right else (x0 + 5, "start")
    p.append(f'  <text x="{ax:.1f}" y="{py(q3)-7:.1f}" text-anchor="{anchor}" class="strip-axis" '
             f'fill="var(--ink)">{lab} &#183; n={len(g)} &#183; median {md:+.0f} pp</text>')
    p.append(f'  <text x="{ax:.1f}" y="{py(-24):.1f}" text-anchor="{anchor}" class="strip-axis" '
             f'fill="var(--teal)">{int((g.gap<-20).sum())} of {len(g)} more than 20 pp too cool</text>')
# reference lines
for v, dash, col in ((0, "", "var(--muted)"), (20, "3 4", "var(--rule)"), (-20, "3 4", "var(--rule)")):
    p.append(f'  <line x1="{L}" y1="{py(v):.1f}" x2="{W-R}" y2="{py(v):.1f}" stroke="{col}" '
             f'stroke-width="{1.4 if v==0 else 1}"{f" stroke-dasharray=\"{dash}\"" if dash else ""}/>')
for _, f in u.iterrows():
    p.append(f'  <circle cx="{px(f.days_post):.1f}" cy="{py(f.gap):.1f}" r="4" '
             f'fill="{"var(--accent)" if f.gap > 0 else "var(--teal)"}" fill-opacity=".72"/>')
# axes
for v in (2, 5, 10, 30, 100, 300):
    p.append(f'  <line x1="{px(v):.1f}" y1="{H-B:.1f}" x2="{px(v):.1f}" y2="{H-B+4:.1f}" stroke="var(--rule)"/>'
             f'<text x="{px(v):.1f}" y="{H-B+17:.1f}" text-anchor="middle" class="strip-axis" fill="var(--faint)">{v}</text>')
for v in (-40, -20, 0, 20, 40):
    p.append(f'  <text x="{L-9}" y="{py(v)+3.5:.1f}" text-anchor="end" class="strip-axis" '
             f'fill="var(--faint)">{v:+d}</text>')
p.append(f'  <line x1="{L}" y1="{H-B:.1f}" x2="{W-R}" y2="{H-B:.1f}" stroke="var(--rule)"/>')
p.append(f'  <line x1="{L}" y1="{T}" x2="{L}" y2="{H-B:.1f}" stroke="var(--rule)"/>')
p.append(f'  <text x="{(L+W-R)/2:.1f}" y="{H-9}" text-anchor="middle" class="strip-axis" fill="var(--muted)">'
         f'days from ignition to the post-fire image (log scale)</text>')
p.append(f'  <text transform="translate(16,{(H-B+T)/2:.1f}) rotate(-90)" text-anchor="middle" class="strip-axis" '
         f'fill="var(--muted)">moderate+ area, ours &#8722; BAER (percentage points)</text>')
p.append(f'  <text x="{W-R:.1f}" y="{py(44):.1f}" text-anchor="end" class="strip-axis" fill="var(--accent)">'
         f'our map hotter than BAER&#8217;s &#8593;</text>')
p.append(f'  <text x="{L+8}" y="{py(-42):.1f}" class="strip-axis" fill="var(--teal)">'
         f'our map cooler than BAER&#8217;s &#8595;</text>')
p.append(f'  <text x="{L+8}" y="{py(44):.1f}" class="strip-axis" fill="var(--faint)">'
         f'Spearman &#961; = {rho.statistic:+.2f} (p = {rho.pvalue:.2f}) &#183; fresh vs delayed: '
         f'{shift:+.0f} pp shift, p = {mw.pvalue:.3f} &#183; n = {len(u)}</text>')
p.append("    </svg>")
(OUT / "timing_plot.svg").write_text("\n".join(p))
print(f"rho={rho.statistic:+.3f} p={rho.pvalue:.3f} | MW p={mw.pvalue:.4f} shift={shift:+.1f} pp")
print(f"fresh n={len(fresh)} med {fresh.gap.median():+.1f} IQR {fresh.gap.quantile(.25):+.1f}..{fresh.gap.quantile(.75):+.1f} "
      f"hot20={int((fresh.gap>20).sum())} cool20={int((fresh.gap<-20).sum())} range {fresh.days_post.min():.0f}-{fresh.days_post.max():.0f} d")
print(f"slow  n={len(slow)} med {slow.gap.median():+.1f} IQR {slow.gap.quantile(.25):+.1f}..{slow.gap.quantile(.75):+.1f} "
      f"hot20={int((slow.gap>20).sum())} cool20={int((slow.gap<-20).sum())} range {slow.days_post.min():.0f}-{slow.days_post.max():.0f} d")
