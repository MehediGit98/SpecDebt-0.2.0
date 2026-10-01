"""
Scenario comparison — the actual design-review product.

Two comparisons matter:

  * **compare_modes** — the same street, different travellers.  This is the
    equity question: who on this street receives circadian light and who does
    not.
  * **compare_designs** — the same traveller, different street.  This is the
    intervention question: what does a setback, a canopy, a façade finish or
    a glazing regulation actually change.

Both return explicit objective definitions and raw values. No dominance
claim is assumed when comparing objective encodings or algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from .exposure import THRESHOLDS, simulate
from .route import Route

#: Named objective encodings.  Changing the encoding changes the answer more
#: than changing anything else in the pipeline.  Name the one you used.
OBJECTIVES = {
    "brown_2022_daytime": {
        "description": "Fraction of trip at or above 250 lx melanopic EDI, "
                       "travel heading only.",
        "weights": {"frac_above_250_travel": 1.0},
    },
    "best_heading_250": {
        "description": "Exploratory 250 lx threshold in at least one "
                       "vertical direction.  Uses the best heading.",
        "weights": {"frac_above_250_best": 1.0},
    },
    "dose_weighted": {
        "description": "Total melanopic dose over the trip, travel heading.",
        "weights": {"dose_travel_lxh": 1.0},
    },
    "robust_heading": {
        "description": "Worst-case heading dose.  The reading that cannot be "
                       "gamed by choosing a favourable view direction.",
        "weights": {"dose_worst_lxh": 1.0},
    },
    "balanced": {
        "description": "Dose, heading robustness and spectral quality, "
                       "equally weighted.  An editorial choice, stated as one.",
        "weights": {"dose_travel_lxh": 0.34, "dose_worst_lxh": 0.33,
                    "mean_der": 0.33},
    },
}


def _objective_vector(summary: dict) -> dict:
    return {
        "frac_above_250_travel": summary["frac_time_above_250_travel"],
        "frac_above_250_best": summary["frac_time_above_250_best"],
        "dose_travel_lxh": summary["dose_travel_lxh"],
        "dose_worst_lxh": summary["dose_worst_lxh"],
        "mean_der": summary["mean_der"],
    }


def score(summary: dict, objective: str = "balanced") -> dict:
    if objective not in OBJECTIVES:
        raise KeyError(f"unknown objective {objective!r}; have {sorted(OBJECTIVES)}")
    spec = OBJECTIVES[objective]
    v = _objective_vector(summary)
    return {"objective": objective,
            "description": spec["description"],
            "weights": spec["weights"],
            "raw": v,
            "value": float(sum(w * v[k] for k, w in spec["weights"].items()))}


def pareto_front(rows: list[dict], keys: tuple[str, ...]) -> list[int]:
    """
    Indices of non-dominated rows on `keys` (all maximised).

    Returned alongside every scalarised ranking, because scalarisation hides
    exactly the trade-off the reviewer is being paid to make.
    """
    v = np.array([[r["objectives"][k] for k in keys] for r in rows])
    keep = []
    for i in range(len(v)):
        dominated = np.any(np.all(v >= v[i], axis=1) & np.any(v > v[i], axis=1))
        if not dominated:
            keep.append(i)
    return keep


def compare_modes(route: Route, modes: list[str] | None = None,
                  objective: str = "balanced", **kw) -> dict:
    """Same street, different travellers."""
    modes = modes or ["walk", "rickshaw", "cng", "bus", "car_clear",
                      "car_tinted", "car_film"]
    rows = []
    for m in modes:
        r = simulate(route, m, **kw)
        s = r.summary()
        rows.append({"key": m, "label": s["mode_label"], "summary": s,
                     "_result": r,
                     "objectives": _objective_vector(s),
                     "score": score(s, objective),
                     "rose": r.compliance_rose().tolist(),
                     "headings": r.headings_deg.tolist(),
                     "series": [{"t": st.t_hours, "medi": st.travel_medi,
                                 "section": st.section,
                                 "svf": st.sky_view_fraction}
                                for st in r.steps]})
    # Dose decomposition against the highest-dose traveller, so that a
    # duration difference is never read as a street-design difference.
    ref = max(rows, key=lambda x: x["objectives"]["dose_travel_lxh"])
    ref_rate = ref["summary"]["mean_medi_travel_lx"]
    ref_dur = ref["summary"]["duration_min"] / 60.0
    for x in rows:
        x["decomposition"] = x["_result"].decompose_dose(ref_rate, ref_dur)
        x["decomposition"]["reference_label"] = ref["label"]
    for x in rows:
        x.pop("_result", None)

    rows.sort(key=lambda x: -x["score"]["value"])
    front = pareto_front(rows, ("dose_travel_lxh", "dose_worst_lxh", "mean_der"))
    best, worst = rows[0], rows[-1]
    ratio = (best["objectives"]["dose_travel_lxh"]
             / max(worst["objectives"]["dose_travel_lxh"], 1e-9))
    return {"kind": "modes", "route": route.name, "objective": objective,
            "rows": rows, "pareto_indices": front,
            "spread_ratio_best_worst_dose": ratio,
            "provenance": rows[0]["summary"]["provenance"]}


@dataclass
class DesignScenario:
    """A design intervention expressed as section overrides."""
    key: str
    label: str
    overrides: dict = field(default_factory=dict)

    def apply(self, route: Route) -> Route:
        return route.variant(f"{route.name} / {self.label}", **self.overrides)


#: A starter set of interventions an urban design review actually argues about.
STANDARD_SCENARIOS = [
    DesignScenario("baseline", "As existing"),
    DesignScenario("canopy_40", "Street trees, 40% cover",
                   {"foliage_cover": 0.40}),
    DesignScenario("canopy_70", "Street trees, 70% cover",
                   {"foliage_cover": 0.70}),
    DesignScenario("widen_18", "Minimum street width 18 m",
                   {"min_width_m": 18.0}),
    DesignScenario("setback_low", "Height limit 6 storeys (18 m)",
                   {"max_height_m": 18.0}),
    DesignScenario("pale_facades", "Light-reflectance façade requirement",
                   {"left_material": "whitewash", "right_material": "whitewash"}),
    DesignScenario("pale_paving", "Pale paving instead of asphalt",
                   {"replace_ground": {"from": "asphalt", "to": "concrete_pavement"}}),
]


def compare_designs(route: Route, mode: str = "walk",
                    scenarios: list[DesignScenario] | None = None,
                    objective: str = "balanced", **kw) -> dict:
    """Same traveller, different street."""
    scenarios = scenarios or STANDARD_SCENARIOS
    rows = []
    for sc in scenarios:
        r = simulate(sc.apply(route), mode, **kw)
        s = r.summary()
        rows.append({"key": sc.key, "label": sc.label, "summary": s,
                     "objectives": _objective_vector(s),
                     "score": score(s, objective),
                     "rose": r.compliance_rose().tolist(),
                     "headings": r.headings_deg.tolist(),
                     "series": [{"t": st.t_hours, "medi": st.travel_medi,
                                 "section": st.section,
                                 "svf": st.sky_view_fraction}
                                for st in r.steps]})
    base = next(x for x in rows if x["key"] == "baseline") \
        if any(x["key"] == "baseline" for x in rows) else rows[0]
    for x in rows:
        b = base["objectives"]["dose_travel_lxh"]
        x["delta_vs_baseline_pct"] = (
            100.0 * (x["objectives"]["dose_travel_lxh"] - b) / max(b, 1e-9))
    ranked = sorted(rows, key=lambda x: -x["score"]["value"])
    front = pareto_front(ranked, ("dose_travel_lxh", "dose_worst_lxh", "mean_der"))
    return {"kind": "designs", "route": route.name, "mode": mode,
            "objective": objective, "rows": ranked, "pareto_indices": front,
            "scenarios": [asdict(s) for s in scenarios],
            "provenance": rows[0]["summary"]["provenance"]}
