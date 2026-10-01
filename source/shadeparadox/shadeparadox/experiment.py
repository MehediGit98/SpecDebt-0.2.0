"""
Experiments.

E1  Behaviour x orientation x desk depth.  Does shade operation cost
    melanopic exposure, and how much of the cost is physics versus habit?

E2  Objective encoding x optimisation algorithm.  Which of the two decides
    the control design?  Paper 3 says encoding.  This tests whether that
    carries into coupled thermal/visual/circadian shading control.

E3  View direction sweep.  The control decisions themselves depend on which
    way the occupant faces, because DGP does.  If the spread is large, every
    published shading-control result with an unstated view direction is
    reporting one draw from a distribution.

Every experiment returns raw per-cell records, not just summaries, so that
the bootstrap and the variance decomposition are reproducible from the JSON.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import asdict, replace

import numpy as np

from .behaviour import CONTROLLERS, ControllerParams, Occupancy, controller
from .control import (ENCODINGS, KEYS, OPTIMISERS, SPACE, score, vec_to_params)
from .metrics import EnergyModel
from .optics import SHADES, Room
from .sim import REPRESENTATIVE_DAYS, RunResult, Site, run

#: Outcome dimensions used for every dispersion and Pareto calculation.
OUTCOMES = ("frac_dgp_above_035", "frac_medi_above_250", "electric_kwh")
OUTCOME_SCALES = (1.0, 1.0, 10.0)


def _outcome_vec(a: dict) -> np.ndarray:
    return np.array([a[k] / s for k, s in zip(OUTCOMES, OUTCOME_SCALES)])


def bootstrap_ci(values, n: int = 2000, alpha: float = 0.05, seed: int = 0):
    v = np.asarray(values, float)
    if len(v) < 2:
        return {"mean": float(v.mean()) if len(v) else float("nan"),
                "lo": float("nan"), "hi": float("nan"), "n": int(len(v))}
    rng = np.random.default_rng(seed)
    draws = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return {"mean": float(v.mean()),
            "lo": float(np.quantile(draws, alpha / 2)),
            "hi": float(np.quantile(draws, 1 - alpha / 2)),
            "n": int(len(v))}


def _room(orientation, depth, **kw):
    base = dict(orientation_deg=orientation, desk_y_m=depth,
                window_head_m=2.7, window_sill_m=0.8, window_width_frac=0.85)
    base.update(kw)
    return Room(name=f"cell_{int(orientation)}deg_{depth:g}m", **base)


# --------------------------------------------------------------------------
# E1
# --------------------------------------------------------------------------

def e1_behaviour(orientations=(90, 180, 270, 0), depths=(2.5, 4.5, 6.5),
                 controllers=None, shade_key="roller_dark",
                 n_directions: int = 256, dt_h: float = 0.5,
                 seeds=(0, 1, 2, 3, 4)) -> dict:
    """
    Behaviour x orientation x depth, replicated over behavioural seeds.

    Every controller here is seed-dependent, not only ``haldi_stochastic``:
    the ``Controller.arrival()`` reopening rule that every controller
    inherits rolls the dice whenever the occupant arrives, or returns from
    lunch, with the blind already closed. A single fixed seed can therefore
    make an otherwise-deterministic controller look better or worse than it
    behaves on average, so every controller is run over the full ``seeds``
    tuple, not a single seed.
    """
    controllers = controllers or ["always_open", "glare_only", "thermal_only",
                                  "thermal_and_glare", "haldi_stochastic",
                                  "circadian_aware"]
    shade = SHADES[shade_key]
    t0 = time.time()
    rows = []
    for ori, depth, ck in itertools.product(orientations, depths, controllers):
        room = _room(ori, depth)
        reps = []
        for sd in seeds:
            r = run(room, lambda s, ck=ck: controller(ck, ControllerParams(), s),
                    shade, n_directions=n_directions, dt_h=dt_h, seed=sd)
            reps.append(r.aggregates())
        agg = {k: float(np.mean([x[k] for x in reps]))
               for k in reps[0] if k != "counters"}
        agg["counters"] = {k: float(np.mean([x["counters"][k] for x in reps]))
                           for k in reps[0]["counters"]}
        rows.append({"orientation_deg": ori, "depth_m": depth,
                     "controller": ck, "label": CONTROLLERS[ck].label,
                     "n_replicates": len(reps),
                     "medi_replicates": [x["medi_mean"] for x in reps],
                     "per_seed": [{"seed": sd, **x} for sd, x in zip(seeds, reps)],
                     "aggregates": agg})

    base = {(r["orientation_deg"], r["depth_m"]): r for r in rows
            if r["controller"] == "always_open"}
    for r in rows:
        b_row = base[(r["orientation_deg"], r["depth_m"])]
        b = b_row["aggregates"]
        a = r["aggregates"]
        r["medi_retained"] = a["medi_mean"] / max(b["medi_mean"], 1e-9)
        r["compliance_delta_pp"] = 100.0 * (a["frac_medi_above_250"]
                                            - b["frac_medi_above_250"])
        r["dgp_delta_pp"] = 100.0 * (a["frac_dgp_above_035"]
                                     - b["frac_dgp_above_035"])
        r["energy_delta_pct"] = 100.0 * (a["electric_kwh"] / max(
            b["electric_kwh"], 1e-9) - 1.0)
        # Paired by seed (this row's seed i against always_open's seed i,
        # same cell), kept alongside the seed-averaged quantities above so
        # the two bootstraps below measure two different kinds of spread
        # and neither is mistaken for the other.
        bs = b_row["per_seed"]
        r["medi_retained_by_seed"] = [
            x["medi_mean"] / max(y["medi_mean"], 1e-9)
            for x, y in zip(r["per_seed"], bs)]
        r["compliance_delta_pp_by_seed"] = [
            100.0 * (x["frac_medi_above_250"] - y["frac_medi_above_250"])
            for x, y in zip(r["per_seed"], bs)]
        r["dgp_delta_pp_by_seed"] = [
            100.0 * (x["frac_dgp_above_035"] - y["frac_dgp_above_035"])
            for x, y in zip(r["per_seed"], bs)]
        r["energy_delta_pct_by_seed"] = [
            100.0 * (x["electric_kwh"] / max(y["electric_kwh"], 1e-9) - 1.0)
            for x, y in zip(r["per_seed"], bs)]

    by_ctrl = {}
    for ck in controllers:
        sel = [r for r in rows if r["controller"] == ck]
        # Cell-level bootstrap: resamples the 12 orientation x depth cells,
        # each already averaged over seeds. This is spatial consistency --
        # does the finding hold across the tested geometries -- and is not
        # a substitute for seed-to-seed uncertainty.
        by_ctrl[ck] = {
            "label": CONTROLLERS[ck].label,
            "medi_retained": bootstrap_ci([r["medi_retained"] for r in sel]),
            "compliance_delta_pp": bootstrap_ci(
                [r["compliance_delta_pp"] for r in sel]),
            "dgp_delta_pp": bootstrap_ci([r["dgp_delta_pp"] for r in sel]),
            "energy_delta_pct": bootstrap_ci(
                [r["energy_delta_pct"] for r in sel]),
        }
        # Seed-level bootstrap: for each behavioural seed, average that
        # seed's cell-paired value across the 12 cells to get one replicate
        # per seed, then resample over THOSE. This is the run-to-run
        # uncertainty coming from the arrival/reopening RNG with geometry
        # held fixed -- the number that answers "how much would the
        # headline move on a rerun with different random draws".
        n_seeds = sel[0]["n_replicates"]
        for key, by_seed_key in (
                ("medi_retained_across_seeds", "medi_retained_by_seed"),
                ("compliance_delta_pp_across_seeds", "compliance_delta_pp_by_seed"),
                ("dgp_delta_pp_across_seeds", "dgp_delta_pp_by_seed"),
                ("energy_delta_pct_across_seeds", "energy_delta_pct_by_seed")):
            seed_level = [float(np.mean([r[by_seed_key][i] for r in sel]))
                         for i in range(n_seeds)]
            by_ctrl[ck][key] = bootstrap_ci(seed_level)
    return {"experiment": "E1", "shade": shade_key, "rows": rows,
            "by_controller": by_ctrl, "orientations": list(orientations),
            "depths": list(depths), "seeds": list(seeds),
            "wall_clock_s": round(time.time() - t0, 2)}


# --------------------------------------------------------------------------
# E2
# --------------------------------------------------------------------------

def _make_objective(room, shade, encoding, n_directions, dt_h, days, cache,
                    ctrl_key="haldi_stochastic"):
    def objective(vec):
        key = (round(float(x), 4) for x in vec)
        key = tuple(key)
        if key not in cache:
            p = vec_to_params(np.asarray(vec, float))
            # averaged over behavioural seeds: the designer is choosing
            # parameters for a stochastic occupant, not for one realisation
            aggs = [run(room, lambda s: controller(ctrl_key, p, s), shade,
                        n_directions=n_directions, dt_h=dt_h, days=days,
                        seed=sd).aggregates() for sd in (7, 8, 9)]
            cache[key] = {k: float(np.mean([a[k] for a in aggs]))
                          for k in aggs[0] if k != "counters"}
        return score(cache[key], encoding)
    return objective


def _param_sensitivity(room, shade, n_directions, dt_h, days, ctrl_key,
                       tol: float = 1e-6) -> dict:
    """
    Which of the :data:`control.SPACE` parameters actually move the
    outcome vector for ``ctrl_key``, each perturbed to its search-space
    extremes with the others held at their dataclass defaults.

    A parameter an optimiser searches over but that never changes the
    outcome is not neutral bookkeeping: it burns search budget and, because
    it is identical for every encoding and every algorithm, mechanically
    narrows the measured dispersion in both. This is checked here rather
    than assumed, so a future change of ``ctrl_key`` or of what a
    controller reads cannot silently reintroduce the gap this replaces.
    """
    def outcome_at(p):
        aggs = [run(room, lambda s: controller(ctrl_key, p, s), shade,
                    n_directions=n_directions, dt_h=dt_h, days=days,
                    seed=sd).aggregates() for sd in (7, 8, 9)]
        a = {k: float(np.mean([x[k] for x in aggs])) for k in aggs[0]
             if k != "counters"}
        return _outcome_vec(a)

    base_out = outcome_at(ControllerParams())
    shift, active = {}, {}
    for name, (lo, hi) in SPACE.items():
        d_lo = float(np.linalg.norm(
            outcome_at(replace(ControllerParams(), **{name: lo})) - base_out))
        d_hi = float(np.linalg.norm(
            outcome_at(replace(ControllerParams(), **{name: hi})) - base_out))
        shift[name] = max(d_lo, d_hi)
        active[name] = shift[name] > tol
    return {"active_parameters": [n for n in SPACE if active[n]],
            "inert_parameters": [n for n in SPACE if not active[n]],
            "outcome_shift_at_extremes": shift, "tolerance": tol}


def _degenerate_outcome_dims(cells: list, tol: float = 1e-9) -> dict:
    """
    Outcome dimensions with ~zero spread across every evaluated cell.

    An encoding that weights a dimension which never moves in this
    room/day configuration cannot be distinguished, by this experiment,
    from an encoding that ignores that dimension entirely -- the two look
    identical in outcome space regardless of how different their weights
    are.  Flagged rather than left for a reader to discover by re-deriving
    it from the raw cells.
    """
    v = np.array([c["outcome"] for c in cells])
    spread = v.max(axis=0) - v.min(axis=0)
    return {dim: bool(s <= tol) for dim, s in zip(OUTCOMES, spread)}


def e2_encoding_vs_algorithm(orientation=270, depth=6.5, shade_key="roller_dark",
                             encodings=None, optimisers=None, budget: int = 16,
                             n_directions: int = 192, dt_h: float = 1.0,
                             days=None, seed: int = 11,
                             ctrl_key: str = "circadian_aware") -> dict:
    """Cross encodings and optimisers using three effective control inputs.

    Reopening probability and the diagnostic mEDI target are held fixed
    because they do not change occupied-step decisions for this controller.
    """
    if ctrl_key != "circadian_aware":
        raise ValueError("E2 search space is defined for circadian_aware only")
    encodings = encodings or ["glare_only", "energy_only", "glare_energy",
                              "circadian_only", "coupled_linear",
                              "coupled_constraint"]
    optimisers = optimisers or list(OPTIMISERS)
    # This inherited deep cell is retained as an identifiability diagnostic.
    # Saturation is reported explicitly and is not evidence of equivalence.
    days = days or [REPRESENTATIVE_DAYS[1], REPRESENTATIVE_DAYS[2]]
    room = _room(orientation, depth, window_width_frac=0.55,
                 window_head_m=2.4)
    shade = SHADES[shade_key]
    t0 = time.time()
    cache: dict = {}

    cells = []
    for enc, opt in itertools.product(encodings, optimisers):
        obj = _make_objective(room, shade, enc, n_directions, dt_h, days,
                              cache, ctrl_key)
        vec, val, trace = OPTIMISERS[opt](obj, budget, seed)
        p = vec_to_params(np.asarray(vec, float))
        aggs = [run(room, lambda s: controller(ctrl_key, p, s), shade,
                    n_directions=n_directions, dt_h=dt_h, days=days,
                    seed=sd).aggregates() for sd in (7, 8, 9)]
        a = {k: float(np.mean([x[k] for x in aggs]))
             for k in aggs[0] if k != "counters"}
        cells.append({"encoding": enc, "optimiser": opt,
                      "params": p.as_dict(), "score": float(val),
                      "aggregates": {k: v for k, v in a.items()
                                     if k != "counters"},
                      "outcome": _outcome_vec(a).tolist(),
                      "evaluations": len(trace)})

    V = {(c["encoding"], c["optimiser"]): np.array(c["outcome"]) for c in cells}

    def pairwise(group_key, vary_key):
        out = []
        groups = sorted({c[group_key] for c in cells})
        varies = sorted({c[vary_key] for c in cells})
        for g in groups:
            vs = [V[(g, v)] if group_key == "encoding" else V[(v, g)]
                  for v in varies]
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    out.append(float(np.linalg.norm(vs[i] - vs[j])))
        return out

    within_enc = pairwise("encoding", "optimiser")   # algorithm effect
    within_opt = pairwise("optimiser", "encoding")   # encoding effect
    ci_alg = bootstrap_ci(within_enc, seed=seed)
    ci_enc = bootstrap_ci(within_opt, seed=seed + 1)

    # A zero denominator is undefined; flooring 0/0 invents evidence.
    denom = ci_alg["mean"]
    ratio_ci = {"mean": None, "lo": None, "hi": None,
                "is_lower_bound": False, "numerical_floor": None,
                "algorithm_dispersion_is_zero": bool(denom <= 1e-12),
                "status": "unidentifiable" if denom <= 1e-12 else "descriptive"}
    if denom > 1e-12:
        ratio_ci["mean"] = float(ci_enc["mean"] / denom)
    # Pairwise distances share cells; no independent-sample confidence
    # interval is claimed for their ratio.

    # distinct outcome clusters, which is what the ratio is really counting
    def clusters(vs, tol=1e-6):
        out = []
        for v in vs:
            if not any(np.linalg.norm(v - u) < tol for u in out):
                out.append(v)
        return len(out)
    per_encoding = {e: clusters([V[(e, o)] for o in optimisers])
                    for e in encodings}
    per_optimiser = {o: clusters([V[(e, o)] for e in encodings])
                     for o in optimisers}

    sensitivity = _param_sensitivity(room, shade, n_directions, dt_h, days,
                                     ctrl_key)
    degenerate = _degenerate_outcome_dims(cells)

    return {"experiment": "E2", "controller": ctrl_key,
            "orientation_deg": orientation,
            "depth_m": depth, "shade": shade_key, "budget": budget,
            "encodings": encodings, "optimisers": optimisers,
            "cells": cells,
            "dispersion_from_algorithm": ci_alg,
            "dispersion_from_encoding": ci_enc,
            "encoding_over_algorithm_ratio": ratio_ci,
            "distinct_outcomes_per_encoding": per_encoding,
            "distinct_outcomes_per_optimiser": per_optimiser,
            "parameter_sensitivity": sensitivity,
            "degenerate_outcome_dims": degenerate,
            "evaluation_budget_per_cell": budget,
            "evaluations_per_cell": {c["encoding"]+"/"+c["optimiser"]: c["evaluations"] for c in cells},
            "unique_evaluations": len(cache),
            "outcome_dims": list(OUTCOMES),
            "wall_clock_s": round(time.time() - t0, 2)}


# --------------------------------------------------------------------------
# E3
# --------------------------------------------------------------------------

def e3_view_direction(orientation=270, depth=2.5, shade_key="roller_dark",
                      controllers=("always_open", "thermal_and_glare",
                                   "haldi_stochastic", "circadian_aware"),
                      n_headings: int = 12, n_directions: int = 256,
                      dt_h: float = 0.5, seeds=(0, 1, 2, 3, 4)) -> dict:
    """
    Sweep the occupant's view direction and record the envelope.

    Every controller's arrival/reopening rule is seed-dependent (see
    e1_behaviour's docstring), so a heading-to-heading swing measured at a
    single fixed seed can be that seed's RNG draw rather than a genuine
    view-direction effect. Every heading is therefore run over the full
    ``seeds`` tuple. ``medi_ratio`` and ``coverage_range`` are reported
    from the seed-averaged values (the headline), and again computed
    independently within each seed (``*_seed_lo``/``*_seed_hi``), so a
    swing that only shows up because of which seed happened to be used is
    visible rather than silently baked into the headline number.
    """
    shade = SHADES[shade_key]
    headings = [orientation - 180.0 + i * (360.0 / n_headings)
                for i in range(n_headings)]
    t0 = time.time()
    rows = []
    for ck in controllers:
        vals = []
        for h in headings:
            room = _room(orientation, depth, view_heading_deg=h)
            reps = [run(room, lambda s: controller(ck, ControllerParams(), s),
                       shade, n_directions=n_directions, dt_h=dt_h,
                       seed=sd).aggregates() for sd in seeds]
            avg = {k: float(np.mean([x[k] for x in reps])) for k in reps[0]
                  if k != "counters"}
            vals.append({"heading_deg": h % 360.0,
                         "medi_mean": avg["medi_mean"],
                         "frac_medi_above_250": avg["frac_medi_above_250"],
                         "frac_dgp_above_035": avg["frac_dgp_above_035"],
                         "mean_coverage": avg["mean_coverage"],
                         "electric_kwh": avg["electric_kwh"],
                         "medi_mean_by_seed": [x["medi_mean"] for x in reps],
                         "mean_coverage_by_seed": [x["mean_coverage"]
                                                   for x in reps]})
        m = np.array([v["medi_mean"] for v in vals])
        cov = np.array([v["mean_coverage"] for v in vals])
        ratio_by_seed, range_by_seed = [], []
        for i in range(len(seeds)):
            m_i = np.array([v["medi_mean_by_seed"][i] for v in vals])
            c_i = np.array([v["mean_coverage_by_seed"][i] for v in vals])
            ratio_by_seed.append(float(m_i.max() / max(m_i.min(), 1e-9)))
            range_by_seed.append(float(c_i.max() - c_i.min()))
        rows.append({"controller": ck, "label": CONTROLLERS[ck].label,
                     "values": vals,
                     "medi_min": float(m.min()), "medi_max": float(m.max()),
                     "medi_ratio": float(m.max() / max(m.min(), 1e-9)),
                     "coverage_range": float(cov.max() - cov.min()),
                     "medi_ratio_seed_lo": float(min(ratio_by_seed)),
                     "medi_ratio_seed_hi": float(max(ratio_by_seed)),
                     "coverage_range_seed_lo": float(min(range_by_seed)),
                     "coverage_range_seed_hi": float(max(range_by_seed))})
    return {"experiment": "E3", "orientation_deg": orientation,
            "depth_m": depth, "shade": shade_key, "headings": headings,
            "seeds": list(seeds), "rows": rows,
            "wall_clock_s": round(time.time() - t0, 2)}
