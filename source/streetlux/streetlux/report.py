"""
Reporting.

Two artefacts, both offline and both regenerable from the same result object:
a provenance-stamped JSON, and a single self-contained HTML review sheet with
no external assets.

The HTML is deliberately austere.  It is a review document that will be
printed and argued over in a meeting, not a dashboard.
"""

from __future__ import annotations

import json
from streetlux.jsonio import dumps as strict_dumps
import math
from pathlib import Path

import numpy as np

from .exposure import THRESHOLDS

_CSS = """
:root{--ink:#12100e;--mute:#6c6660;--rule:#ddd8d0;--bg:#fbfaf8;
      --warn:#a8341f;--ok:#2c6e49;--acc:#1f4e79}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:15px/1.55 "Iowan Old Style",Georgia,serif;padding:32px 22px 80px}
.wrap{max-width:940px;margin:0 auto}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:17px;margin:38px 0 10px;padding-bottom:6px;
   border-bottom:1px solid var(--rule);font-weight:600}
h3{font-size:14px;margin:22px 0 6px;font-weight:600}
.sub{color:var(--mute);margin:0 0 18px;font-size:13px}
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
.roses{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}
.rose figcaption{font-size:12px;color:var(--mute);text-align:center;margin-top:2px}
code{font-family:ui-monospace,Menlo,monospace;font-size:12px}
.prov{font-size:11.5px;color:var(--mute);font-family:ui-monospace,Menlo,monospace;
      white-space:pre-wrap;border-top:1px solid var(--rule);padding-top:10px;
      margin-top:26px}
.bar{height:9px;background:#e8e3da;position:relative}
.bar span{position:absolute;left:0;top:0;bottom:0;background:var(--acc)}
"""


def _fmt(v, d=1):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:,.{d}f}"


def _rose_svg(headings, values, rmax=None, size=170):
    """Polar plot of trip-mean mEDI against heading, with the 250 lx ring."""
    cx = cy = size / 2
    R = size / 2 - 20
    rmax = rmax or max(max(values), THRESHOLDS["well_tier2_brown"]) * 1.05
    pts = []
    for h, v in zip(list(headings) + [headings[0]], list(values) + [values[0]]):
        r = R * min(v / rmax, 1.0)
        a = math.radians(h - 90.0)
        pts.append(f"{cx + r*math.cos(a):.1f},{cy + r*math.sin(a):.1f}")
    rings = ""
    for thr, col in ((THRESHOLDS["well_tier1"], "#c9a227"),
                     (THRESHOLDS["well_tier2_brown"], "#a8341f")):
        rr = R * min(thr / rmax, 1.0)
        rings += (f'<circle cx="{cx}" cy="{cy}" r="{rr:.1f}" fill="none" '
                  f'stroke="{col}" stroke-width="1" stroke-dasharray="3 3"/>')
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
            f'{axes}{rings}'
            f'<polygon points="{" ".join(pts)}" fill="#1f4e79" '
            f'fill-opacity="0.22" stroke="#1f4e79" stroke-width="1.4"/></svg>')


def _series_svg(rows, w=900, h=190):
    """mEDI against elapsed time for each compared case."""
    all_t, all_v = [], []
    for r in rows:
        all_t += [s["t"] for s in r["series"]]
        all_v += [s["medi"] for s in r["series"]]
    if not all_t:
        return ""
    t0, t1 = min(all_t), max(all_t)
    vmax = max(max(all_v), 300.0) * 1.08
    pad_l, pad_b, pad_t = 46, 26, 8

    def X(t):
        return pad_l + (t - t0) / max(t1 - t0, 1e-9) * (w - pad_l - 12)

    def Y(v):
        return h - pad_b - (v / vmax) * (h - pad_b - pad_t)

    cols = ["#1f4e79", "#2c6e49", "#a8341f", "#7d5ba6", "#b8860b",
            "#3d7ea6", "#8a5a44"]
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%">',
           f'<rect x="{pad_l}" y="{pad_t}" width="{w-pad_l-12}" '
           f'height="{h-pad_b-pad_t}" fill="#fff" stroke="#ddd8d0"/>']
    for thr, col, lab in ((136.0, "#c9a227", "136"), (250.0, "#a8341f", "250")):
        y = Y(thr)
        out.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w-12}" y2="{y:.1f}" '
                   f'stroke="{col}" stroke-dasharray="4 3" stroke-width="1"/>'
                   f'<text x="{pad_l-6}" y="{y+3:.1f}" font-size="9" '
                   f'text-anchor="end" fill="{col}">{lab}</text>')
    for i, r in enumerate(rows):
        pts = " ".join(f"{X(s['t']):.1f},{Y(s['medi']):.1f}" for s in r["series"])
        out.append(f'<polyline points="{pts}" fill="none" '
                   f'stroke="{cols[i%len(cols)]}" stroke-width="1.7"/>')
    out.append(f'<text x="{pad_l}" y="{h-8}" font-size="10" fill="#6c6660">'
               f'{t0:.2f} h local</text>'
               f'<text x="{w-12}" y="{h-8}" font-size="10" text-anchor="end" '
               f'fill="#6c6660">{t1:.2f} h</text>'
               f'<text x="10" y="{pad_t+10}" font-size="10" fill="#6c6660">'
               f'lx mEDI</text></svg>')
    return "".join(out)


def _legend(rows):
    cols = ["#1f4e79", "#2c6e49", "#a8341f", "#7d5ba6", "#b8860b",
            "#3d7ea6", "#8a5a44"]
    items = "".join(
        f'<span style="margin-right:14px;font-size:12px">'
        f'<span style="display:inline-block;width:14px;height:3px;'
        f'background:{cols[i%len(cols)]};vertical-align:middle"></span> '
        f'{r["label"]}</span>' for i, r in enumerate(rows))
    return f'<div style="margin:6px 0 2px">{items}</div>'


def write_json(result: dict, path) -> Path:
    p = Path(path)
    p.write_text(strict_dumps(result), encoding="utf-8")
    return p


def write_html(result: dict, path, title: str | None = None) -> Path:
    rows = result["rows"]
    kind = result["kind"]
    title = title or (f"Melanopic exposure review — {result['route']}")
    best, worst = rows[0], rows[-1]

    cards = ""
    b = best["summary"]
    cards += (f'<div class="card"><div class="lab">Best case</div>'
              f'<div class="val">{_fmt(b["mean_medi_travel_lx"],0)}</div>'
              f'<div class="lab">lx mEDI · {best["label"]}</div></div>')
    wsum = worst["summary"]
    cards += (f'<div class="card"><div class="lab">Worst case</div>'
              f'<div class="val">{_fmt(wsum["mean_medi_travel_lx"],0)}</div>'
              f'<div class="lab">lx mEDI · {worst["label"]}</div></div>')
    ratio = (b["mean_medi_travel_lx"] / max(wsum["mean_medi_travel_lx"], 1e-9))
    cards += (f'<div class="card"><div class="lab">Spread</div>'
              f'<div class="val">{_fmt(ratio,1)}×</div>'
              f'<div class="lab">best ÷ worst</div></div>')
    hs = b["heading_spread"]
    cards += (f'<div class="card"><div class="lab">Heading sensitivity</div>'
              f'<div class="val">{_fmt(hs["ratio_max_min"],2)}×</div>'
              f'<div class="lab">max ÷ min over view direction</div></div>')

    head = ("Traveller" if kind == "modes" else "Scenario")
    thead = (f"<tr><th>{head}</th><th>min</th><th>trip</th><th>trip</th>"
             f"<th>trip</th><th>% time</th><th>% time</th><th>mean</th>"
             f"<th>heading</th></tr>"
             f"<tr><th></th><th></th><th>mean mEDI</th><th>worst hdg</th>"
             f"<th>best hdg</th><th>≥250 travel</th><th>≥250 best</th>"
             f"<th>DER</th><th>max÷min</th></tr>")
    body = ""
    for r in rows:
        s = r["summary"]
        h = s["heading_spread"]
        flag = "" if s["frac_time_above_250_travel"] > 0.5 else ' style="color:#a8341f"'
        extra = ""
        if kind == "designs":
            extra = ""
        body += (f'<tr><td>{r["label"]}</td>'
                 f'<td>{_fmt(s["duration_min"],0)}</td>'
                 f'<td{flag}>{_fmt(s["mean_medi_travel_lx"],0)}</td>'
                 f'<td>{_fmt(s["mean_medi_worst_heading_lx"],0)}</td>'
                 f'<td>{_fmt(s["mean_medi_best_heading_lx"],0)}</td>'
                 f'<td>{_fmt(100*s["frac_time_above_250_travel"],0)}</td>'
                 f'<td>{_fmt(100*s["frac_time_above_250_best"],0)}</td>'
                 f'<td>{_fmt(s["mean_der"],2)}</td>'
                 f'<td>{_fmt(h["ratio_max_min"],2)}</td></tr>{extra}')

    rmax = max(max(r["rose"]) for r in rows) * 1.05
    roses = "".join(
        f'<figure class="rose">{_rose_svg(r["headings"], r["rose"], rmax)}'
        f'<figcaption>{r["label"]}</figcaption></figure>' for r in rows)

    shares = ""
    for r in rows:
        ss = r["summary"]["source_shares"]
        cells = "".join(f"<td>{_fmt(100*ss.get(k,0),0)}</td>" for k in
                        ("sky", "direct_sun", "facade_left", "facade_right", "ground"))
        shares += f'<tr><td>{r["label"]}</td>{cells}</tr>'

    delta = ""
    if kind == "designs":
        delta = "<h2>Change against the existing street</h2><table><thead>" \
                "<tr><th>Scenario</th><th>Δ dose</th><th></th></tr></thead><tbody>"
        for r in rows:
            d = r.get("delta_vs_baseline_pct", 0.0)
            wpx = min(abs(d), 100.0)
            col = "#2c6e49" if d >= 0 else "#a8341f"
            delta += (f'<tr><td>{r["label"]}</td><td>{d:+.1f}%</td>'
                      f'<td style="width:45%"><div class="bar">'
                      f'<span style="width:{wpx}%;background:{col}"></span>'
                      f'</div></td></tr>')
        delta += "</tbody></table>"

    decomp = ""
    if kind == "modes" and rows[0].get("decomposition"):
        ref = rows[0]["decomposition"]["reference_label"]
        decomp = ("<h2>Dose gap: duration or exposure rate?</h2>"
                  f'<p class="sub">Against <em>{ref}</em>, the highest-dose '
                  "traveller on this route. A shorter trip and a darker trip "
                  "produce the same headline shortfall and demand opposite "
                  "interventions, so the two are separated here rather than "
                  "reported as one number.</p>"
                  "<table><thead><tr><th>Traveller</th><th>trip min</th>"
                  "<th>rate</th><th>dose</th><th>gap</th><th>from duration</th>"
                  "<th>from rate</th><th>duration share</th></tr></thead><tbody>")
        for r in rows:
            d = r["decomposition"]
            decomp += (f'<tr><td>{r["label"]}</td>'
                       f'<td>{_fmt(d["duration_h"]*60,0)}</td>'
                       f'<td>{_fmt(d["exposure_rate_medi"],0)}</td>'
                       f'<td>{_fmt(d["dose_lxh"],0)}</td>'
                       f'<td>{_fmt(d["gap_lxh"],0)}</td>'
                       f'<td>{_fmt(d["gap_from_duration_lxh"],0)}</td>'
                       f'<td>{_fmt(d["gap_from_exposure_rate_lxh"],0)}</td>'
                       f'<td>{_fmt(100*d["duration_share_of_gap"],0)}%</td></tr>')
        decomp += ("</tbody></table><div class=\"note\">Rate is trip-mean "
                   "lx melanopic EDI at the travel heading; dose is lx melanopic "
                   "EDI · hours. The two gap columns sum exactly to the gap.</div>")

    obj = rows[0]["score"]
    prov = result["provenance"]
    counters = rows[0]["summary"]["counters"]

    warn = ""
    if not prov.get("official_action_spectra"):
        warn = ('<div class="note warn"><strong>Analytic action spectra.</strong> '
                'These numbers use the built-in Govardovskii/lens approximation, '
                f'giving K<sub>mel,v</sub><sup>D65</sup> = '
                f'{prov["k_mel_v_d65_derived_W_per_lm"]*1000:.4f} mW/lm, '
                f'{prov["k_mel_deviation_from_CIE_S026_pct"]:+.2f}% against '
                'CIE S 026. Adequate for design review, not for publication. '
                'Load the official tables before quoting any value.</div>')

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>{_CSS}</style></head><body><div class="wrap">
<h1>{title}</h1>
<p class="sub">Melanopic equivalent daylight illuminance on a vertical plane at
eye height, CIE S 026:2018. Values are <strong>lx melanopic EDI</strong>, not
photopic lux. Objective encoding: <code>{obj['objective']}</code>.</p>
{warn}
<div class="grid">{cards}</div>

<div class="note"><strong>Read the envelope, not the number.</strong> Each case
below reports the melanopic level at the traveller's actual heading, and the
best and worst headings at the same points. These are directional
exposure summaries. The exploratory 250 lx threshold does not establish
WELL certification or a physiological outcome.</div>

<h2>{'Travellers on this route' if kind=='modes' else 'Design scenarios'}</h2>
<table><thead>{thead}</thead><tbody>{body}</tbody></table>
{delta}

{decomp}

<h2>Melanopic level through the trip</h2>
{_legend(rows)}
{_series_svg(rows)}

<h2>Compliance rose — trip-mean mEDI by view direction</h2>
<p class="sub">Dashed rings mark 136 and 250 lx melanopic EDI. A circular rose
means the reading does not depend on where the traveller looks. A lobed one
means any single reported number is underdetermined.</p>
<div class="roses">{roses}</div>

<h2>Where the light comes from</h2>
<table><thead><tr><th></th><th>sky %</th><th>direct sun %</th>
<th>façade L %</th><th>façade R %</th><th>ground %</th></tr></thead>
<tbody>{shares}</tbody></table>

<h2>Objective encoding</h2>
<p class="sub">{obj['description']}</p>
<table><thead><tr><th>Term</th><th>Weight</th></tr></thead><tbody>
{''.join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k,v in obj['weights'].items())}
</tbody></table>
<div class="note">Changing this encoding reorders the table above more than
changing any other input. It is a design judgement, recorded here so that it
can be argued with rather than inherited silently.</div>

<div class="prov">branch activation: {json.dumps(counters)}
provenance: {json.dumps(prov, default=float)}</div>
</div></body></html>"""
    p = Path(path)
    p.write_text(html, encoding="utf-8")
    return p
