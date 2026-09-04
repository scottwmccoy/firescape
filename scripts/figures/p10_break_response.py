"""How much does shifting the low/moderate break buy, and does the answer
depend on the assessment type?

One curve per group: the CHANGE in mean moderate-or-higher agreement with
BAER as the regional break is moved up or down.  Every curve is zero at
offset 0 by construction, so the peak height is the gain on offer and the
peak position is the break that group actually wants.  The dashed curve is
the control that matters -- initial assessments restricted to MTBS-programme
dNBR, i.e. the same sensor family and processing chain as the extended
fires, so the split cannot be explained by BAER's own dNBR product.
"""
import pathlib
import numpy as np

from firescape import paths

OUT = paths.figures_dir("baer_outliers"); OUT.mkdir(parents=True, exist_ok=True)
z = np.load(pathlib.Path(__file__).with_name("_break_curves.npz")
            if (pathlib.Path(__file__).with_name("_break_curves.npz")).exists()
            else paths.products_dir("calibration") / "break_response_curves.npz")
grid = z["grid"]

SERIES = [("initial assessments", "initial", "var(--accent)", 2.6, ""),
          ("initial, MTBS-programme dNBR only", "initial_mtbs", "var(--accent)", 1.5, "5 4"),
          ("extended assessments", "extended", "var(--baer)", 2.6, ""),
          ("all 124 fires", "all", "var(--faint)", 1.6, "")]
N = {"initial": 40, "initial_mtbs": 18, "extended": 84, "all": 124}

W, H, L, R, T, B = 880, 470, 70, 20, 26, 48
YLO, YHI = -0.045, 0.040
px = lambda v: L + (v - grid.min()) / (grid.max() - grid.min()) * (W - L - R)
py = lambda v: (H - B) - (v - YLO) / (YHI - YLO) * (H - T - B)

p = [f'    <svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Change in mean '
     f'moderate-or-higher agreement with BAER as the regional break is shifted, separately '
     f'for initial and extended assessments">']
p.append(f'  <line x1="{px(0):.1f}" y1="{T}" x2="{px(0):.1f}" y2="{py(YLO):.1f}" '
         f'stroke="var(--muted)" stroke-width="1.4"/>')
p.append(f'  <text x="{px(0)-7:.1f}" y="{py(YLO)-9:.1f}" text-anchor="end" class="strip-axis" '
         f'fill="var(--muted)">the break we use now</text>')
p.append(f'  <line x1="{L}" y1="{py(0):.1f}" x2="{W-R}" y2="{py(0):.1f}" stroke="var(--rule)"/>')
for lab, key, col, wdt, dash in SERIES:
    y = z[key] - z[key][grid == 0][0]
    pts = " ".join(f"{px(g):.1f},{py(v):.1f}" for g, v in zip(grid, y))
    p.append(f'  <polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{wdt}"'
             f'{f" stroke-dasharray=\"{dash}\"" if dash else ""} stroke-linejoin="round"/>')
    if not dash:
        b = grid[y.argmax()]
        p.append(f'  <circle cx="{px(b):.1f}" cy="{py(y.max()):.1f}" r="4.5" fill="{col}"/>')
        p.append(f'  <text x="{px(b):.1f}" y="{py(y.max())-10:.1f}" text-anchor="middle" '
                 f'class="strip-axis" fill="{col}">{b:+.0f}</text>')
for v in range(-120, 161, 40):
    p.append(f'  <line x1="{px(v):.1f}" y1="{py(YLO):.1f}" x2="{px(v):.1f}" y2="{py(YLO)+4:.1f}" stroke="var(--rule)"/>'
             f'<text x="{px(v):.1f}" y="{py(YLO)+17:.1f}" text-anchor="middle" class="strip-axis" '
             f'fill="var(--faint)">{v:+d}</text>')
for v in (-0.04, -0.02, 0.0, 0.02, 0.04):
    p.append(f'  <text x="{L-9}" y="{py(v)+3.5:.1f}" text-anchor="end" class="strip-axis" '
             f'fill="var(--faint)">{v*100:+.0f}</text>')
p.append(f'  <line x1="{L}" y1="{T}" x2="{L}" y2="{py(YLO):.1f}" stroke="var(--rule)"/>')
p.append(f'  <line x1="{L}" y1="{py(YLO):.1f}" x2="{W-R}" y2="{py(YLO):.1f}" stroke="var(--rule)"/>')
p.append(f'  <text x="{(L+W-R)/2:.1f}" y="{H-9}" text-anchor="middle" class="strip-axis" '
         f'fill="var(--muted)">dNBR added to the regional low/moderate break</text>')
p.append(f'  <text transform="translate(16,{(T+py(YLO))/2:.1f}) rotate(-90)" text-anchor="middle" '
         f'class="strip-axis" fill="var(--muted)">change in mean agreement with BAER '
         f'(percentage points)</text>')
for i, (lab, key, col, wdt, dash) in enumerate(SERIES):
    y = T + 14 + 17 * i
    p.append(f'  <line x1="{px(-118):.1f}" y1="{y-3.5:.1f}" x2="{px(-100):.1f}" y2="{y-3.5:.1f}" '
             f'stroke="{col}" stroke-width="{wdt}"{f" stroke-dasharray=\"{dash}\"" if dash else ""}/>'
             f'<text x="{px(-95):.1f}" y="{y:.1f}" class="strip-axis" fill="var(--muted)">'
             f'{lab} &#183; n={N[key]}</text>')
p.append("    </svg>")
(OUT / "break_response.svg").write_text("\n".join(p))
for lab, key, *_ in SERIES:
    y = z[key] - z[key][grid == 0][0]
    print(f"  {lab:36s} peak {y.max()*100:+.2f} pp at {grid[y.argmax()]:+.0f}")
print(f"-> {OUT/'break_response.svg'}")
