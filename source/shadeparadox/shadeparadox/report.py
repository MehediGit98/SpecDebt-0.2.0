"""Reporting for the shading experiments: provenance-stamped JSON and one
self-contained HTML sheet per run."""

from __future__ import annotations

import json
from streetlux.jsonio import dumps as strict_dumps
import math
from pathlib import Path

import numpy as np

from .control import ENCODINGS

_CSS = """
:root{--ink:#12100e;--mute:#6c6660;--rule:#ddd8d0;--bg:#fbfaf8;
      --warn:#a8341f;--ok:#2c6e49;--acc:#1f4e79}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:15px/1.55 "Iowan Old Style",Georgia,serif;padding:32px 22px 80px}
.wrap{max-width:980px;margin:0 auto}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:17px;margin:38px 0 10px;padding-bottom:6px;
   border-bottom:1px solid var(--rule);font-weight:600}
.sub{color:var(--mute);margin:0 0 16px;font-size:13px}
table{border-collapse:collapse;width:100%;font-size:13px;
      font-family:ui-monospace,"SF Mono",Menlo,monospace}
th,td{padding:6px 8px;border-bottom:1px solid var(--rule);text-align:right}
th:first-child,td:first-child{text-align:left;font-family:inherit}
thead th{border-bottom:1.5px solid var(--ink);font-weight:600;
         font-family:inherit;font-size:12px;color:var(--mute)}
tbody tr:hover{background:#f2efe9}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
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
.ci{color:var(--mute);font-size:11px}
"""


def _f(v, d=1):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:,.{d}f}"


def _ci(c, d=2, pct=False):
    if c.get("mean") is None: return "undefined (zero denominator)"
    if c.get("lo") is None: return _f(c["mean"], d)
    k = 100.0 if pct else 1.0
    return (f'{_f(c["mean"]*k, d)} <span class="ci">[{_f(c["lo"]*k, d)}, '
            f'{_f(c["hi"]*k, d)}]</span>')


def _heatmap(cells, encodings, optimisers, key="outcome"):
    """Outcome distance of each cell from the grand mean, as a shaded table."""
    V = {(c["encoding"], c["optimiser"]): np.array(c["outcome"]) for c in cells}
    mean = np.mean(list(V.values()), axis=0)
    d = {k: float(np.linalg.norm(v - mean)) for k, v in V.items()}
    mx = max(d.values()) or 1.0
    head = "".join(f"<th>{o}</th>" for o in optimisers)
    body = ""
    for e in encodings:
        row = ""
        for o in optimisers:
            f = d[(e, o)] / mx
            col = f"rgba(31,78,121,{0.08 + 0.72*f:.2f})"
            row += (f'<td style="background:{col}">{d[(e,o)]:.3f}</td>')
        body += f"<tr><td>{e}</td>{row}</tr>"
    return (f"<table><thead><tr><th>encoding \\ optimiser</th>{head}</tr>"
            f"</thead><tbody>{body}</tbody></table>")


def _rose(headings, values, size=180):
    cx = cy = size / 2
    R = size / 2 - 22
    rmax = max(values) * 1.05 or 1.0
    pts = []
    for h, v in zip(list(headings) + [headings[0]], list(values) + [values[0]]):
        r = R * min(v / rmax, 1.0)
        a = math.radians(h - 90.0)
        pts.append(f"{cx + r*math.cos(a):.1f},{cy + r*math.sin(a):.1f}")
    rr = R * min(250.0 / rmax, 1.0)
    axes = ""
    for lbl, ang in (("N", 0), ("E", 90), ("S", 180), ("W", 270)):
        a = math.radians(ang - 90.0)
        axes += (f'<line x1="{cx}" y1="{cy}" x2="{cx+R*math.cos(a):.1f}" '
                 f'y2="{cy+R*math.sin(a):.1f}" stroke="#e2ddd4"/>'
                 f'<text x="{cx+(R+11)*math.cos(a):.1f}" '
                 f'y="{cy+(R+11)*math.sin(a)+4:.1f}" font-size="9" '
                 f'text-anchor="middle" fill="#6c6660">{lbl}</text>')
    return (f'<svg viewBox="0 0 {size} {size}" width="100%">'
            f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="#fff" stroke="#ddd8d0"/>'
            f'{axes}<circle cx="{cx}" cy="{cy}" r="{rr:.1f}" fill="none" '
            f'stroke="#a8341f" stroke-width="1" stroke-dasharray="3 3"/>'
            f'<polygon points="{" ".join(pts)}" fill="#1f4e79" '
            f'fill-opacity="0.22" stroke="#1f4e79" stroke-width="1.4"/></svg>')


def write_json(obj, path) -> Path:
    p = Path(path)
    p.write_text(strict_dumps(obj), encoding="utf-8")
    return p


def write_html(e1, e2, e3, provenance, path, title="Shading paradox testbed"):
    # E1
    ctrl_rows = ""
    base = e1["by_controller"]["always_open"]
    for ck, v in e1["by_controller"].items():
        ret = v["medi_retained"]
        bw = max(0.0, min(1.0, ret["mean"])) * 100
        ctrl_rows += (
            f'<tr><td>{v["label"]}</td>'
            f'<td>{_ci(ret, 2)}</td>'
            f'<td style="width:22%"><div class="bar">'
            f'<span style="width:{bw:.0f}%"></span></div></td>'
            f'<td>{_ci(v["compliance_delta_pp"], 1)}</td>'
            f'<td>{_ci(v["dgp_delta_pp"], 1)}</td>'
            f'<td>{_ci(v["energy_delta_pct"], 1)}</td></tr>')

    worst = min(e1["by_controller"].items(),
                key=lambda kv: kv[1]["medi_retained"]["mean"])
    cards = (
        f'<div class="card"><div class="lab">Largest circadian cost</div>'
        f'<div class="val">{100*(1-worst[1]["medi_retained"]["mean"]):.0f}%</div>'
        f'<div class="lab">mEDI lost · {worst[1]["label"]}</div></div>'
        f'<div class="card"><div class="lab">Encoding ÷ algorithm</div>'
        f'<div class="val">'
        f'{_f(e2["encoding_over_algorithm_ratio"]["mean"],1)}×</div>'
        f'<div class="lab">outcome dispersion ratio</div></div>'
        f'<div class="card"><div class="lab">View-direction spread</div>'
        f'<div class="val">'
        f'{max(r["medi_ratio"] for r in e3["rows"]):.2f}×</div>'
        f'<div class="lab">max ÷ min mEDI over heading</div></div>')

    # E2
    cell_rows = ""
    for c in sorted(e2["cells"], key=lambda c: (c["encoding"], c["optimiser"])):
        a = c["aggregates"]
        cell_rows += (f'<tr><td>{c["encoding"]} / {c["optimiser"]}</td>'
                      f'<td>{100*a["frac_dgp_above_035"]:.0f}</td>'
                      f'<td>{100*a["frac_medi_above_250"]:.0f}</td>'
                      f'<td>{a["electric_kwh"]:.2f}</td>'
                      f'<td>{a["mean_coverage"]:.2f}</td>'
                      f'<td>{c["params"]["max_coverage"]:.2f}</td>'
                      f'<td>{c["params"]["dgp_close"]:.3f}</td></tr>')

    enc_desc = "".join(
        f'<tr><td>{k}</td><td style="text-align:left;font-family:inherit">'
        f'{ENCODINGS[k]["description"]}</td></tr>' for k in e2["encodings"])

    # E3
    roses = "".join(
        f'<figure style="margin:0"><figcaption style="font-size:12px;'
        f'color:#6c6660;text-align:center">{r["label"]}<br>'
        f'{r["medi_ratio"]:.2f}×</figcaption>'
        f'{_rose([v["heading_deg"] for v in r["values"]], [v["medi_mean"] for v in r["values"]])}'
        f'</figure>' for r in e3["rows"])

    warn = ""
    if not provenance.get("official_action_spectra"):
        warn = ('<div class="note warn"><strong>Analytic action spectra.</strong> '
                f'K<sub>mel,v</sub><sup>D65</sup> = '
                f'{provenance["k_mel_v_d65_derived_W_per_lm"]*1000:.4f} mW/lm, '
                f'{provenance["k_mel_deviation_from_CIE_S026_pct"]:+.2f}% '
                'against CIE S 026. Adequate for an experimental testbed, not '
                'for a published absolute value.</div>')

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>{_CSS}</style></head><body><div class="wrap">
<h1>{title}</h1>
<p class="sub">Coupled thermal, visual and circadian shading control in a
tropical perimeter office. Melanopic values are <strong>lx melanopic EDI</strong>
(CIE S 026), not photopic lux. Intervals are 95% bootstrap.</p>
{warn}
<div class="grid">{cards}</div>

<h2>E1 · What shade operation costs</h2>
<p class="sub">Against blinds never operated, across
{len(e1['orientations'])} orientations × {len(e1['depths'])} desk depths.
"mEDI retained" is the ratio of trip-mean melanopic EDI to the never-operated
case: 1.00 means no circadian cost.</p>
<table><thead><tr><th>Shade operation</th><th>mEDI retained</th><th></th>
<th>Δ compliance pp</th><th>Δ DGP&gt;0.35 pp</th><th>Δ energy %</th>
</tr></thead><tbody>{ctrl_rows}</tbody></table>
<div class="note">Read the threshold-triggered rows against the stochastic
one. A blind that descends only as far as the glare source requires costs
little circadian exposure, because the sun and the bright sky sit in the upper
band of the aperture. The cost appears when the blind is operated the way
people actually operate blinds — closed readily, reopened rarely.</div>

<h2>E2 · Objective encoding versus optimisation algorithm</h2>
<p class="sub">Every cell is a control design found by one algorithm under one
encoding, on the same room and the same budget
({e2['budget']} evaluations, {e2['unique_evaluations']} unique simulations).
Distances are Euclidean in normalised outcome space
({', '.join(e2['outcome_dims'])}).</p>
<table><thead><tr><th>Source of disagreement</th><th>mean outcome distance</th>
</tr></thead><tbody>
<tr><td>Changing the optimisation algorithm</td>
<td>{_ci(e2['dispersion_from_algorithm'], 3)}</td></tr>
<tr><td>Changing the objective encoding</td>
<td>{_ci(e2['dispersion_from_encoding'], 3)}</td></tr>
<tr><td><strong>Ratio, encoding ÷ algorithm</strong></td>
<td><strong>{'&ge; ' if e2['encoding_over_algorithm_ratio']['is_lower_bound'] else ''}{_ci(e2['encoding_over_algorithm_ratio'], 1)}</strong></td></tr>
</tbody></table>
<div class="note{' warn' if e2['encoding_over_algorithm_ratio']['algorithm_dispersion_is_zero'] else ''}">
Distinct outcomes reached per encoding across all
{len(e2['optimisers'])} optimisers:
<code>{json.dumps(e2['distinct_outcomes_per_encoding'])}</code>.
Per optimiser across all {len(e2['encodings'])} encodings:
<code>{json.dumps(e2['distinct_outcomes_per_optimiser'])}</code>.
The evaluation budget was at most {e2['evaluation_budget_per_cell']} per cell;
actual counts are recorded in JSON. When both dispersions are zero, this
configuration cannot identify an encoding or optimiser effect.
</div>
{_heatmap(e2["cells"], e2["encodings"], e2["optimisers"])}
<p class="sub">Shading darkness is distance from the grand mean outcome. Rows
that differ from each other more than their own cells differ internally are
encodings that disagree about what good means.</p>
<table><thead><tr><th>encoding / optimiser</th><th>DGP&gt;.35 %</th>
<th>mEDI≥250 %</th><th>kWh</th><th>mean cov</th><th>max cov</th>
<th>dgp close</th></tr></thead><tbody>{cell_rows}</tbody></table>
<h3 style="font-size:14px;margin:22px 0 6px">Encodings</h3>
<table><thead><tr><th>key</th><th style="text-align:left">meaning</th></tr>
</thead><tbody>{enc_desc}</tbody></table>

<h2>E3 · Which way is the occupant facing?</h2>
<p class="sub">The same room, the same controller, swept over
{len(e3['headings'])} view directions. Dashed ring is 250 lx melanopic EDI.
The control decisions themselves move, because DGP depends on the view.</p>
<div class="grid">{roses}</div>
<div class="note">A shading-control result published without a stated view
direction is one draw from this spread, not a property of the design.</div>

<div class="prov">provenance: {json.dumps(provenance, default=float)}</div>
</div></body></html>"""
    p = Path(path)
    p.write_text(html, encoding="utf-8")
    return p
