"""
The analysis engine.

Given a metric and a specification space, this computes four things.  Only the
first is standard multiverse practice; the other three are what make the
result actionable rather than merely alarming.

1. **Envelope** — the range of the reported value over the undeclared
   subspace.  What the point estimate is one draw from.

2. **Flip rate** — the fraction of the undeclared subspace that disagrees with
   the modal decision.  Values scatter harmlessly all the time; what matters
   is whether the *decision* the number is used for changes.  A metric can
   have a wide envelope and a flip rate of zero, and that metric is fine.

3. **Attribution** — first-order Sobol indices over the categorical grid,
   computed exactly when the grid is full.  Which choice point is responsible.

4. **Minimum declaration set** — the smallest set of factors that, once
   stated, drives the residual flip rate below a tolerance.  This is the
   deliverable: not "your field has a problem" but "declare these three things
   and the problem goes away".

A flip rate of zero is always reported next to the number of specifications
evaluated, so that an untested space is never read as a stable one.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .spec import Specification


# --------------------------------------------------------------------------
# Decisions
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Threshold:
    """The decision a compliance metric is used for: pass or fail."""
    value: float
    kind: str = "ge"
    label: str = "threshold"

    def __call__(self, y: np.ndarray) -> np.ndarray:
        return y >= self.value if self.kind == "ge" else y <= self.value

    def describe(self):
        return {"type": "threshold", "value": self.value, "kind": self.kind,
                "label": self.label}


@dataclass(frozen=True)
class Comparison:
    """The decision a benchmark is used for: is A better than B."""
    label: str = "A beats B"

    def __call__(self, y: np.ndarray) -> np.ndarray:
        return y > 0.0

    def describe(self):
        return {"type": "comparison", "label": self.label}


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

def _binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * math.log2(p) - (1 - p) * math.log2(1 - p))


@dataclass
class Result:
    spec: Specification
    specs: list
    values: np.ndarray
    decisions: np.ndarray
    mode: str
    decision: object
    metric_name: str
    units: str = ""
    wall_clock_s: float = 0.0
    counters: dict = field(default_factory=dict)

    # -- envelope ----------------------------------------------------------
    def envelope(self) -> dict:
        y = self.values
        signed = bool(y.min() < 0 < y.max())
        return {"min": float(y.min()), "p05": float(np.quantile(y, 0.05)),
                "median": float(np.median(y)), "mean": float(y.mean()),
                "p95": float(np.quantile(y, 0.95)), "max": float(y.max()),
                "spread": float(y.max() - y.min()),
                # A max/min ratio is meaningless once the metric changes sign,
                # which it does for any difference score. Reported as None
                # rather than as a number that looks interpretable.
                "ratio_max_min": (None if signed or abs(y.min()) < 1e-12
                                  else float(y.max() / abs(y.min()))),
                "crosses_zero": signed,
                "spread_pct_of_median": float(
                    100.0 * (y.max() - y.min()) / max(abs(np.median(y)), 1e-12)),
                "n": int(len(y))}

    # -- decisions ---------------------------------------------------------
    def pass_rate(self) -> float:
        return float(self.decisions.mean())

    def flip_rate(self) -> float:
        """Fraction disagreeing with the modal decision."""
        p = self.pass_rate()
        return float(min(p, 1.0 - p))

    def decision_entropy(self) -> float:
        return _binary_entropy(self.pass_rate())

    def is_determinate(self, tol: float = 0.0) -> bool:
        return self.flip_rate() <= tol

    # -- attribution -------------------------------------------------------
    def _levels_array(self, name):
        return np.array([str(s[name]) for s in self.specs])

    def attribution(self, on: str = "value") -> dict:
        """
        First-order Sobol indices S_i = Var(E[Y | X_i]) / Var(Y).

        Exact on a full factorial; an estimate under sampling, which the mode
        field records.  Reported for the value and, separately, for the
        decision — a factor can move the number a lot and the decision not at
        all, and only the second matters.
        """
        y = self.values if on == "value" else self.decisions.astype(float)
        tot = float(np.var(y))
        out = {}
        for f in self.spec.free:
            lv = self._levels_array(f.name)
            means = np.array([y[lv == u].mean() for u in np.unique(lv)])
            weights = np.array([np.mean(lv == u) for u in np.unique(lv)])
            cond = float(np.sum(weights * (means - y.mean()) ** 2))
            out[f.name] = float(cond / tot) if tot > 1e-18 else 0.0
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def decision_flip_by_factor(self) -> dict:
        """
        For each factor, the largest gap in pass rate between its levels.

        More directly readable than a Sobol index when the question is "does
        this one choice decide compliance?": a value of 1.0 means the factor
        alone flips the verdict.
        """
        out = {}
        d = self.decisions.astype(float)
        for f in self.spec.free:
            lv = self._levels_array(f.name)
            rates = [d[lv == u].mean() for u in np.unique(lv)]
            out[f.name] = float(max(rates) - min(rates))
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    # -- declaration -------------------------------------------------------
    def residual_flip_rate(self, declared: tuple) -> float:
        """
        Flip rate remaining once `declared` factors are stated.

        Declaring a factor does not remove its effect; it makes the effect
        visible.  What remains is the disagreement *within* each cell of the
        declared factors, weighted by cell size.
        """
        if not declared:
            return self.flip_rate()
        keys = np.array(["|".join(str(s[n]) for n in declared)
                         for s in self.specs])
        tot, num = 0.0, 0.0
        for u in np.unique(keys):
            m = keys == u
            p = float(self.decisions[m].mean())
            num += m.sum() * min(p, 1.0 - p)
            tot += m.sum()
        return float(num / tot)

    def minimum_declaration_set(self, tol: float = 0.01,
                                max_exact: int | None = None) -> dict:
        """
        Smallest set of factors whose declaration brings the residual flip
        rate to `tol` or below.

        Exhaustive over ALL subset sizes by default (`max_exact=None`).
        This is exact, and for the factor counts this package's own case
        studies use (a handful), cheap: evaluating every subset of `n`
        factors costs at most 2**n calls to `residual_flip_rate`, so a
        7-factor specification is 127 calls, not a search that needs
        shortcuts. Pass a smaller `max_exact` to cap the exhaustive part
        for a specification with many undeclared factors; beyond that
        size the search falls back to greedy forward selection, which is
        not guaranteed to find the true minimum (see below) but is
        guaranteed to terminate, since declaring every undeclared factor
        always drives the residual to exactly 0.

        A decision that a single undeclared factor's levels never flip on
        their own -- it only flips through an INTERACTION between two or
        more factors -- defeats a greedy search that stops as soon as one
        more factor stops helping: every first step looks like zero
        improvement, and stopping there would wrongly report that no
        declaration helps at all, when jointly declaring the right pair
        might fully resolve it. The forward selection below does not
        stop on a zero-improvement step for exactly this reason.

        The whole point of the paper is that this set is short:
        specification debt is usually two or three sentences of method,
        not a research programme. Whether that holds is now checked
        exactly for every case this package ships, not asserted from a
        capped search that can silently return the wrong answer.
        """
        names = [f.name for f in self.spec.undeclared]
        base = self.flip_rate()
        if base <= tol:
            return {"set": [], "residual_flip_rate": base, "search": "none",
                    "baseline_flip_rate": base, "tolerance": tol}

        cap = len(names) if max_exact is None else min(max_exact, len(names))
        for k in range(1, cap + 1):
            for combo in itertools.combinations(names, k):
                r = self.residual_flip_rate(combo)
                if r <= tol:
                    return {"set": list(combo), "residual_flip_rate": r,
                            "search": "exhaustive", "baseline_flip_rate": base,
                            "tolerance": tol}
        if cap >= len(names):
            # Every subset was tried, including the full factor set, whose
            # residual is always exactly 0 (a single specification cannot
            # disagree with itself) and so is <= any tol >= 0. Not having
            # returned above means tol is negative.
            raise ValueError(f"tolerance {tol!r} cannot be satisfied by any "
                             "declaration; tol must be >= 0")

        chosen, cur = [], base
        pool = list(names)
        while pool and cur > tol:
            cand = min(pool, key=lambda n: self.residual_flip_rate(
                tuple(chosen + [n])))
            cur = self.residual_flip_rate(tuple(chosen + [cand]))
            chosen.append(cand)
            pool.remove(cand)
        return {"set": chosen, "residual_flip_rate": cur, "search": "greedy",
                "baseline_flip_rate": base, "tolerance": tol}

    def cell_residuals(self, declared: tuple) -> dict:
        groups = {}
        for sp, d in zip(self.specs, self.decisions):
            key = tuple(str(sp[n]) for n in declared)
            groups.setdefault(key, []).append(bool(d))
        residuals = [min(np.mean(v), 1-np.mean(v)) for v in groups.values()]
        return {"max_cell_residual": float(max(residuals)),
                "mixed_cells": sum(r > 0 for r in residuals),
                "n_cells": len(groups)}

    def sufficiency_curve(self, tol: float = 0.01) -> list:
        """Exact best subset at each cardinality; subsets need not be nested."""
        names = [f.name for f in self.spec.undeclared]
        curve = []
        for k in range(len(names)+1):
            candidates = itertools.combinations(names, k)
            best = min(candidates, key=lambda c: (round(self.residual_flip_rate(c),14), c))
            curve.append({"declared": list(best), "n": k,
                          "residual_flip_rate": self.residual_flip_rate(best),
                          "search": "exhaustive_best_subset", **self.cell_residuals(best)})
        return curve

    def profile(self, tol: float = 0.01) -> dict:
        """
        Shape of the debt, not just its size.

        Concentrated debt (one dominant factor) is cheap to repay: one
        sentence of method. Distributed debt needs a protocol. The two are
        different problems and the fix is different, so the distinction is
        computed rather than left to the reader.
        """
        a = self.attribution("value")
        shares = np.array(list(a.values()), float)
        shares = shares / max(shares.sum(), 1e-12)
        herfindahl = float(np.sum(shares ** 2))
        curve = self.sufficiency_curve(tol)
        need = next((p["n"] for p in curve
                     if p["residual_flip_rate"] <= tol), len(curve) - 1)
        return {
            "top_factor": next(iter(a)) if a else None,
            "top_factor_share": float(shares[0]) if len(shares) else 0.0,
            "herfindahl": herfindahl,
            "shape": ("concentrated" if herfindahl >= 0.45
                      else "mixed" if herfindahl >= 0.25 else "distributed"),
            "declarations_to_determinacy": int(need),
            "n_undeclared_factors": len(self.spec.undeclared),
        }

    # -- headline ----------------------------------------------------------
    def debt(self, tol: float = 0.01) -> dict:
        """
        The specification debt of this metric as reported.

        ``flip_rate`` is the minority share under uniform specification
        weights. Independent pairwise disagreement is 2*p*(1-p), not this rate.
        """
        env = self.envelope()
        mds = self.minimum_declaration_set(tol)
        mds.update(self.cell_residuals(tuple(mds["set"])))
        return {
            "metric": self.metric_name,
            "units": self.units,
            "decision": self.decision.describe(),
            "mode": self.mode,
            "n_specifications": int(len(self.values)),
            "n_undeclared_factors": len(self.spec.undeclared),
            "envelope": env,
            "pass_rate": self.pass_rate(),
            "pairwise_disagreement": 2*self.pass_rate()*(1-self.pass_rate()),
            "comparison_outcomes": ({"A_wins": int(np.sum(self.values > 0)),
                "B_wins": int(np.sum(self.values < 0)), "ties": int(np.sum(self.values == 0))}
                if isinstance(self.decision, Comparison) else None),
            "flip_rate": self.flip_rate(),
            "decision_entropy_bits": self.decision_entropy(),
            "determinate": self.is_determinate(tol),
            "attribution_value": self.attribution("value"),
            "attribution_decision": self.attribution("decision"),
            "decision_flip_by_factor": self.decision_flip_by_factor(),
            "minimum_declaration_set": mds,
            "profile": self.profile(tol),
            "sufficiency_curve": self.sufficiency_curve(tol),
            "counters": self.counters,
            "wall_clock_s": self.wall_clock_s,
        }


# --------------------------------------------------------------------------
# Study
# --------------------------------------------------------------------------

def run_study(metric_fn, spec: Specification, decision,
              metric_name: str, units: str = "",
              max_grid: int = 20000, n_sample: int = 4000,
              seed: int = 0) -> Result:
    """
    Evaluate `metric_fn` at every specification and assemble a Result.

    `metric_fn` takes a dict of factor values and returns a float.  For a
    comparison decision it should return the *difference* (A minus B), so that
    the sign is the verdict.
    """
    specs, mode = spec.enumerate(max_grid, n_sample, seed)
    t0 = time.time()
    vals = []
    for i, setting in enumerate(specs):
        try:
            value = float(metric_fn(setting))
        except Exception as exc:
            raise RuntimeError(f"Metric failed at specification {i}: {setting}") from exc
        if not math.isfinite(value):
            raise ValueError(f"Non-finite metric at specification {i}: {setting}")
        vals.append(value)
    y = np.asarray(vals, float)
    if not len(y):
        raise ValueError("The specification space is empty")
    failures = 0
    return Result(
        spec=spec, specs=specs, values=y, decisions=np.asarray(decision(y)),
        mode=mode, decision=decision, metric_name=metric_name, units=units,
        wall_clock_s=round(time.time() - t0, 2),
        counters={"specifications_evaluated": int(len(y)),
                  "evaluation_failures": int(failures),
                  "grid_size": spec.size,
                  "undeclared_grid_size": spec.undeclared_size})
