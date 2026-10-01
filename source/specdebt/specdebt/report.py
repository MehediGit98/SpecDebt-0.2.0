"""Reporting: provenance-stamped JSON and one self-contained HTML sheet."""

from __future__ import annotations

import json
from streetlux.jsonio import dumps as strict_dumps
import math
from pathlib import Path

_CSS = """
:root{--ink:#12100e;--mute:#6c6660;--rule:#ddd8d0;--bg:#fbfaf8;
      --warn:#a8341f;--ok:#2c6e49;--acc:#1f4e79}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:15px/1.55 "Iowan Old Style",Georgia,serif;padding:32px 22px 80px}
.wrap{max-width:960px;margin:0 auto}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:17px;margin:38px 0 10px;padding-bottom:6px;
   border-bottom:1px solid var(--rule);font-weight:600}
h3{font-size:14px;margin:22px 0 6px;font-weight:600}
.sub{color:var(--mute);margin:0 0 16px;font-size:13px}
table{border-collapse:collapse;width:100%;font-size:13px;
      font-family:ui-monospace,"SF Mono",Menlo,monospace}
th,td{padding:6px 8px;border-bottom:1px solid var(--rule);text-align:right}
th:first-child,td:first-child{text-align:left;font-family:inherit}
thead th{border-bottom:1.5px solid var(--ink);font-weight:600;
         font-family:inherit;font-size:12px;color:var(--mute)}
tbody tr:hover{background:#f2efe9}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
      gap:14px;margin:14px 0}
.card{border:1px solid var(--rule);background:#fff;padding:12px 14px}
.card .lab{font-size:11px;color:var(--mute);text-transform:uppercase;
           letter-spacing:.06em}
.card .val{font-size:22px;font-family:ui-monospace,Menlo,monospace;margin-top:3px}
.note{border-left:3px solid var(--acc);padding:8px 14px;background:#fff;
      margin:14px 0;font-size:13.5px}
.warn{border-left-color:var(--warn)}
.bar{height:9px;background:#e8e3da;position:relative}
.bar span{position:absolute;left:0;top:0;bottom:0;background:var(--acc)}
code{font-family:ui-monospace,Menlo,monospace;font-size:12px}
.prov{font-size:11.5px;color:var(--mute);font-family:ui-monospace,Menlo,monospace;
      white-space:pre-wrap;border-top:1px solid var(--rule);padding-top:10px;
      margin-top:26px}
.just{font-family:inherit;text-align:left;color:var(--mute);font-size:12px}
"""


def _f(v, d=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:,.{d}f}"


def _curve_svg(curve, w=880, h=170):
    pad_l, pad_b, pad_t = 52, 30, 10
    n = len(curve) - 1 or 1
    ymax = max(p["residual_flip_rate"] for p in curve) or 1.0

    def X(i):
        return pad_l + i / n * (w - pad_l - 14)

    def Y(v):
        return h - pad_b - (v / ymax) * (h - pad_b - pad_t)

    pts = " ".join(f"{X(p['n']):.1f},{Y(p['residual_flip_rate']):.1f}"
                   for p in curve)
    dots = "".join(f'<circle cx="{X(p["n"]):.1f}" '
                   f'cy="{Y(p["residual_flip_rate"]):.1f}" r="3" '
                   f'fill="#1f4e79"/>' for p in curve)
    labs = "".join(f'<text x="{X(p["n"]):.1f}" y="{h-10}" font-size="10" '
                   f'text-anchor="middle" fill="#6c6660">{p["n"]}</text>'
                   for p in curve)
    tol = Y(0.01)
    return (f'<svg viewBox="0 0 {w} {h}" width="100%">'
            f'<rect x="{pad_l}" y="{pad_t}" width="{w-pad_l-14}" '
            f'height="{h-pad_b-pad_t}" fill="#fff" stroke="#ddd8d0"/>'
            f'<line x1="{pad_l}" y1="{tol:.1f}" x2="{w-14}" y2="{tol:.1f}" '
            f'stroke="#a8341f" stroke-dasharray="4 3"/>'
            f'<text x="{pad_l-6}" y="{tol+3:.1f}" font-size="9" '
            f'text-anchor="end" fill="#a8341f">1%</text>'
            f'<polyline points="{pts}" fill="none" stroke="#1f4e79" '
            f'stroke-width="1.8"/>{dots}{labs}'
            f'<text x="10" y="{pad_t+10}" font-size="10" fill="#6c6660">'
            f'flip</text></svg>')


def _hist_svg(values, decisions, threshold=None, w=880, h=150):
    import numpy as np
    y = np.asarray(values, float)
    lo, hi = float(y.min()), float(y.max())
    bins = np.linspace(lo, hi, 41)
    idx = np.clip(np.digitize(y, bins) - 1, 0, 39)
    passed = np.asarray(decisions, bool)
    cnt_p = np.array([(passed & (idx == i)).sum() for i in range(40)])
    cnt_f = np.array([((~passed) & (idx == i)).sum() for i in range(40)])
    mx = max((cnt_p + cnt_f).max(), 1)
    pad_l, pad_b, pad_t = 52, 26, 8
    bw = (w - pad_l - 14) / 40
    bars = ""
    for i in range(40):
        x = pad_l + i * bw
        tot = cnt_p[i] + cnt_f[i]
        if not tot:
            continue
        hh = (h - pad_b - pad_t) * tot / mx
        hf = hh * cnt_f[i] / tot
        bars += (f'<rect x="{x:.1f}" y="{h-pad_b-hh:.1f}" width="{bw-1:.1f}" '
                 f'height="{hh-hf:.1f}" fill="#2c6e49" fill-opacity="0.75"/>'
                 f'<rect x="{x:.1f}" y="{h-pad_b-hf:.1f}" width="{bw-1:.1f}" '
                 f'height="{hf:.1f}" fill="#a8341f" fill-opacity="0.75"/>')
    thr = ""
    if threshold is not None and lo <= threshold <= hi:
        tx = pad_l + (threshold - lo) / (hi - lo) * (w - pad_l - 14)
        thr = (f'<line x1="{tx:.1f}" y1="{pad_t}" x2="{tx:.1f}" '
               f'y2="{h-pad_b}" stroke="#12100e" stroke-dasharray="4 3"/>')
    return (f'<svg viewBox="0 0 {w} {h}" width="100%">'
            f'<rect x="{pad_l}" y="{pad_t}" width="{w-pad_l-14}" '
            f'height="{h-pad_b-pad_t}" fill="#fff" stroke="#ddd8d0"/>'
            f'{bars}{thr}'
            f'<text x="{pad_l}" y="{h-8}" font-size="10" fill="#6c6660">'
            f'{lo:,.3g}</text>'
            f'<text x="{w-14}" y="{h-8}" font-size="10" text-anchor="end" '
            f'fill="#6c6660">{hi:,.3g}</text></svg>')


def _case_block(title, res, debt, threshold=None):
    fac = "".join(
        f'<tr><td>{f["name"]}</td><td>{f["n_levels"]}</td>'
        f'<td>{f["domain"]}</td>'
        f'<td>{_f(debt["attribution_value"].get(f["name"], 0), 3)}</td>'
        f'<td>{_f(debt["decision_flip_by_factor"].get(f["name"], 0), 3)}</td>'
        f'<td class="just">{f["justification"]}</td></tr>'
        for f in res.spec.describe()["factors"])
    env = debt["envelope"]
    mds = debt["minimum_declaration_set"]
    prof = debt["profile"]
    return f"""
<h2>{title}</h2>
<p class="sub">{debt['n_specifications']} specifications
({debt['mode'].replace('_',' ')}), {debt['n_undeclared_factors']} undeclared
free parameters. Decision: <code>{debt['decision']['label']}</code>.</p>
<div class="grid">
<div class="card"><div class="lab">Reported as</div>
<div class="val">{_f(env['median'], 2)}</div>
<div class="lab">median of {debt['n_specifications']} · {debt['units']}</div></div>
<div class="card"><div class="lab">Actually spans</div>
<div class="val">{_f(env['min'], 2)} – {_f(env['max'], 2)}</div>
<div class="lab">{('spans zero, spread ' + _f(env['spread'], 3)) if env['crosses_zero'] else _f(env['ratio_max_min'], 1) + '× envelope'}</div></div>
<div class="card"><div class="lab">Verdict flips in</div>
<div class="val">{_f(100*debt['flip_rate'], 1)}%</div>
<div class="lab">{_f(debt['decision_entropy_bits'], 2)} bits of decision entropy</div></div>
<div class="card"><div class="lab">Debt shape</div>
<div class="val">{prof['declarations_to_determinacy']}</div>
<div class="lab">declarations needed · {prof['shape']}</div></div>
</div>
{_hist_svg(res.values, res.decisions, threshold)}
<p class="sub">Green passes the decision, red fails it. The dashed line is the
threshold. Every bar is a defensible analysis.</p>
<h3>Where the debt sits</h3>
<table><thead><tr><th>factor</th><th>levels</th><th>domain</th>
<th>Sobol S₁</th><th>flip gap</th><th class="just">why these levels</th>
</tr></thead><tbody>{fac}</tbody></table>
<h3>Declaration sufficiency</h3>
{_curve_svg(debt['sufficiency_curve'])}
<div class="note">Minimum declaration set
(<code>{mds['search']}</code>): <strong>{', '.join(mds['set']) or 'none needed'}</strong>
— residual flip rate {_f(100*mds['residual_flip_rate'], 2)}% against a
baseline of {_f(100*mds['baseline_flip_rate'], 1)}%. Stating those values in
the method restores determinacy; nothing about the measurement or the model
has to change.</div>
"""


def write_json(obj, path) -> Path:
    p = Path(path)
    p.write_text(strict_dumps(obj), encoding="utf-8")
    return p


def write_html(cases, comparison, provenance, path,
               title="Specification debt") -> Path:
    blocks = "".join(_case_block(t, r, d, thr) for t, r, d, thr in cases)
    rows = "".join(
        f'<tr><td>{c["case"]}</td><td>{c["domain"]}</td>'
        f'<td>{_f(100*c["flip_rate"], 1)}</td>'
        f'<td>{_f(c["envelope_ratio"], 1)}</td>'
        f'<td>{_f(c["herfindahl"], 2)}</td><td>{c["shape"]}</td>'
        f'<td>{c["declarations"]}</td><td>{c["top_factor"]}</td></tr>'
        for c in comparison["rows"])
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>{_CSS}</style></head><body><div class="wrap">
<h1>{title}</h1>
<p class="sub">A standardised metric with an undeclared free parameter is
reported as determinate and is not. This measures how far from determinate,
which choice point is responsible, and the shortest set of declarations that
fixes it.</p>
<div class="note">The flip rate is the headline, not the envelope. A metric
can scatter widely and still give the same verdict every time, and that metric
has a stable binary decision on the grid. The minority share counts
uniformly weighted specifications that disagree with the modal decision.
Independent pairwise disagreement is a separate quantity, 2p(1-p).</div>
{blocks}
<h2>Across domains</h2>
<p class="sub">Same machinery, no special-casing. If the two are the same kind
of object, the analysis should transfer — and the differences in the profile
should be informative rather than artefactual.</p>
<table><thead><tr><th>case</th><th>domain</th><th>flip %</th>
<th>envelope ×</th><th>Herfindahl</th><th>shape</th><th>declarations</th>
<th>dominant factor</th></tr></thead><tbody>{rows}</tbody></table>
<div class="note">{comparison['reading']}</div>
<div class="prov">provenance: {json.dumps(provenance, default=str)}</div>
</div></body></html>"""
    p = Path(path)
    p.write_text(html, encoding="utf-8")
    return p
