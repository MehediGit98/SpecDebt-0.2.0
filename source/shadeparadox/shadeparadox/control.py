"""
Objective encodings and optimisers.

Cross objective encodings and algorithms, with an explicit identifiability
check in experiment.py. No dominance claim is built into the comparison.

All objectives are written as **maximise**.  Penalties are negative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .behaviour import ControllerParams

# --------------------------------------------------------------------------
# Objective encodings
# --------------------------------------------------------------------------


def _norm_energy(kwh, scale=8.0):
    return -kwh / scale


ENCODINGS: dict[str, dict] = {
    "glare_only": {
        "description": "Minimise the fraction of occupied hours above "
                       "DGP 0.35.  Nothing else enters.",
        "fn": lambda a: -a["frac_dgp_above_035"],
    },
    "energy_only": {
        "description": "Minimise cooling plus lighting electricity.",
        "fn": lambda a: _norm_energy(a["electric_kwh"]),
    },
    "glare_energy": {
        "description": "Equal weight on glare exceedance and energy. The "
                       "encoding most shading papers actually use.",
        "fn": lambda a: -0.5 * a["frac_dgp_above_035"]
                        + 0.5 * _norm_energy(a["electric_kwh"]),
    },
    "circadian_only": {
        "description": "Maximise the fraction of occupied hours at or above "
                       "250 lx melanopic EDI.",
        "fn": lambda a: a["frac_medi_above_250"],
    },
    "coupled_linear": {
        "description": "Equal weight on glare, energy and circadian. An "
                       "editorial choice, stated as one.",
        "fn": lambda a: (-0.34 * a["frac_dgp_above_035"]
                         + 0.33 * _norm_energy(a["electric_kwh"])
                         + 0.33 * a["frac_medi_above_250"]),
    },
    "coupled_constraint": {
        "description": "Maximise threshold coverage with a soft glare penalty; "
                       "energy breaks ties.",
        "fn": lambda a: (a["frac_medi_above_250"]
                         - 10.0 * max(0.0, a["frac_dgp_above_035"] - 0.05)
                         + 0.05 * _norm_energy(a["electric_kwh"])),
    },
    "coupled_lexicographic": {
        "description": "Glare first, then circadian, then energy, each on a "
                       "separate order of magnitude.",
        "fn": lambda a: (-100.0 * a["frac_dgp_above_035"]
                         + 1.0 * a["frac_medi_above_250"]
                         + 0.01 * _norm_energy(a["electric_kwh"])),
    },
}


def score(agg: dict, encoding: str) -> float:
    if encoding not in ENCODINGS:
        raise KeyError(f"unknown encoding {encoding!r}; have {sorted(ENCODINGS)}")
    return float(ENCODINGS[encoding]["fn"](agg))


# --------------------------------------------------------------------------
# Search space
# --------------------------------------------------------------------------

# Effective occupied-action parameters for the melanopic-aware controller.
# Reopening is overwritten by per-step selection; medi_target only counts
# diagnostics. Neither is an optimisation variable.
SPACE = {
    "dgp_close": (0.30, 0.45),
    "solar_close_w_m2": (50.0, 200.0),
    "max_coverage": (0.25, 1.0),
}
KEYS = tuple(SPACE)


def _clip(v, k):
    lo, hi = SPACE[k]
    return float(np.clip(v, lo, hi))


def vec_to_params(v: np.ndarray, base: ControllerParams | None = None):
    p = base or ControllerParams()
    kw = {k: _clip(v[i], k) for i, k in enumerate(KEYS)}
    return replace(p, **kw)


def params_to_vec(p: ControllerParams) -> np.ndarray:
    return np.array([getattr(p, k) for k in KEYS], float)


def _unit(v):
    return np.array([(v[i] - SPACE[k][0]) / (SPACE[k][1] - SPACE[k][0])
                     for i, k in enumerate(KEYS)])


def _from_unit(u):
    return np.array([SPACE[k][0] + float(np.clip(u[i], 0, 1))
                     * (SPACE[k][1] - SPACE[k][0]) for i, k in enumerate(KEYS)])


# --------------------------------------------------------------------------
# Optimisers.  Deliberately a spread of families, all given the same budget.
# --------------------------------------------------------------------------

def random_search(objective, budget: int, seed: int):
    rng = np.random.default_rng(seed)
    best, best_v, trace = None, -math.inf, []
    for _ in range(budget):
        u = rng.random(len(KEYS))
        v = objective(_from_unit(u))
        trace.append(v)
        if v > best_v:
            best, best_v = u, v
    return _from_unit(best), best_v, trace


def latin_hypercube(objective, budget: int, seed: int):
    rng = np.random.default_rng(seed)
    n, d = budget, len(KEYS)
    grid = np.stack([rng.permutation(n) for _ in range(d)], axis=1)
    pts = (grid + rng.random((n, d))) / n
    best, best_v, trace = None, -math.inf, []
    for u in pts:
        v = objective(_from_unit(u))
        trace.append(v)
        if v > best_v:
            best, best_v = u, v
    return _from_unit(best), best_v, trace


def coordinate_descent(objective, budget: int, seed: int):
    rng = np.random.default_rng(seed)
    u = rng.random(len(KEYS))
    best_v = objective(_from_unit(u))
    trace = [best_v]
    used, step = 1, 0.35
    while used < budget:
        improved = False
        for i in range(len(KEYS)):
            for s in (+step, -step):
                if used >= budget:
                    break
                cand = u.copy()
                cand[i] = float(np.clip(cand[i] + s, 0, 1))
                v = objective(_from_unit(cand))
                used += 1
                trace.append(v)
                if v > best_v:
                    u, best_v, improved = cand, v, True
        if not improved:
            step *= 0.5
            if step < 0.02:
                break
    return _from_unit(u), best_v, trace


def annealing(objective, budget: int, seed: int):
    rng = np.random.default_rng(seed)
    u = rng.random(len(KEYS))
    cur = objective(_from_unit(u))
    best_u, best_v, trace = u.copy(), cur, [cur]
    for i in range(1, budget):
        T = max(1e-3, 0.35 * (1.0 - i / budget))
        cand = np.clip(u + rng.normal(0, 0.22, len(KEYS)), 0, 1)
        v = objective(_from_unit(cand))
        trace.append(v)
        if v > cur or rng.random() < math.exp((v - cur) / max(T, 1e-6)):
            u, cur = cand, v
        if v > best_v:
            best_u, best_v = cand, v
    return _from_unit(best_u), best_v, trace


OPTIMISERS = {
    "random_search": random_search,
    "latin_hypercube": latin_hypercube,
    "coordinate_descent": coordinate_descent,
    "annealing": annealing,
}
