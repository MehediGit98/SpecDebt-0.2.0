"""
Two case studies, one machinery.

The thesis of the package is that a melanopic compliance number with an
undeclared view direction and a benchmark score with an undeclared answer
extractor are the same object: a standardised metric with a free parameter,
reported as determinate.  If that is true, the identical analysis should apply
to both without special-casing, and the resulting debt profiles should be
structurally similar.  These two cases exist to test that claim, not to
illustrate it.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache

import numpy as np

from .spec import Specification
from .study import Comparison, Threshold

# ==========================================================================
# Case A — 250 lx melanopic threshold test in a perimeter office
# ==========================================================================

#: Legacy calibration ratio retained ONLY as an explicit sensitivity level.
#:
#: The physical baseline for publication runs is the official CIE S 026
#: photopic/melanopic tables plus ASTM E490, loaded by specdebt.__main__.  The
#: "analytic" level below reproduces the approximate +1.01% calibration shift
#: of the original analytic-action-spectrum implementation without switching
#: the shared spectral engine back to approximate data.
_LEGACY_ANALYTIC_K_MEL_W_PER_LM = 0.001339617521434617
_CIE_S026_K_MEL_W_PER_LM = 0.0013262
_K_RATIO = {
    "cie_s026": 1.0,
    "analytic": _LEGACY_ANALYTIC_K_MEL_W_PER_LM / _CIE_S026_K_MEL_W_PER_LM,
}


def melanopic_specification() -> Specification:
    s = Specification(name="250 lx melanopic threshold test")
    s.add("view_heading_deg", (0, 45, 90, 135, 180, 225, 270, 315),
          "This experimental grid varies the evaluated vertical direction "
          "using eight cardinal and intercardinal headings. These "
          "are uniformly weighted experimental alternatives.",
          domain="geometry")
    s.add("eye_height_m", (1.2, 1.5),
          "Two illustrative eye heights represent seated and elevated eyes; "
          "neither is claimed to be required by a certification protocol.",
          domain="geometry")
    s.add("hour", (9.0, 11.0, 13.0),
          "Three assumed daytime assessment hours span the "
          "morning and early afternoon.",
          domain="sampling")
    s.add("cloud_cover", (0.15, 0.50, 0.85),
          "Three assumed cloud fractions describe contrasting sky "
          "condition.", domain="sampling")
    s.add("month", (1, 7),
          "January and July scenarios in Dhaka. Month is varied as an "
          "experimental factor.", domain="sampling")
    s.add("blind_coverage", (0.0, 0.5),
          "Whether the occupant's blinds were up or half down at the moment "
          "of measurement. These are modelled operational alternatives.", domain="operation")
    s.add("action_spectrum", ("cie_s026", "analytic"),
          "Official CIE S 026 melanopic calibration versus the legacy "
          "analytic approximation used by the original testbed. The physical "
          "simulation itself remains on the official CIE/ASTM baseline.",
          domain="metrology")
    return s


@lru_cache(maxsize=4096)
def _solve_cached(month, hour, cloud, coverage, eye_h, heading, orientation,
                  depth, n_dir):
    import shadeparadox as sp
    from shadeparadox.optics import SHADES, Room, solve
    from shadeparadox.sim import Site, geometry_for, sky_for
    room = Room(name="Daylight threshold cell", orientation_deg=orientation, desk_y_m=depth,
                window_head_m=2.7, window_sill_m=0.8, window_width_frac=0.85,
                eye_height_m=eye_h, view_heading_deg=heading)
    geo = geometry_for(room, n_dir)
    sky, dome = sky_for(Site(), geo, 2026, month, 15, hour, cloud)
    return solve(room, geo, dome, sky, SHADES["roller_dark"], coverage, 0.0).medi


def melanopic_metric(orientation=270.0, depth=3.0, n_directions=192,
                     require_official_baseline: bool = True):
    """
    Return a metric_fn giving melanopic EDI at the eye, in lx melanopic EDI.

    Checks once, before returning the metric function, that the shared
    streetlux spectral engine has the official CIE S 026 / ASTM E490
    tables loaded. This case's own ``action_spectrum`` factor already
    covers calibration as one of the seven undeclared factors (see
    ``melanopic_specification``'s docstring) -- that factor is a stated
    +-1% sensitivity check layered on top of the physics, regardless of
    which baseline the physics itself is running on. This check is about
    that underlying baseline: without it, every number this case produces
    would silently come from the analytic approximation that streetlux's
    own spectra module documents as "NOT adequate for a manuscript
    number", no matter what ``action_spectrum`` said. Pass
    ``require_official_baseline=False`` to deliberately explore the fully
    analytic regime.
    """
    if require_official_baseline:
        from streetlux import spectra as _spectra
        state = _spectra.provenance()
        missing = [name for name, ok in (
            ("official CIE photopic/melanopic action spectra",
             state["official_action_spectra"]),
            ("ASTM E490 solar spectrum", state["official_solar_spectrum"]))
                  if not ok]
        if missing:
            raise RuntimeError(
                "melanopic_metric() called before the official baseline was "
                "loaded: " + ", ".join(missing) + " missing. Call "
                "streetlux.load_publication_references(reference_dir) "
                "first, or pass require_official_baseline=False to "
                "deliberately run on the analytic approximation.")

    def fn(s):
        v = _solve_cached(int(s["month"]), float(s["hour"]),
                          float(s["cloud_cover"]), float(s["blind_coverage"]),
                          float(s["eye_height_m"]), float(s["view_heading_deg"]),
                          float(orientation), float(depth), int(n_directions))
        return v * _K_RATIO[s["action_spectrum"]]
    return fn


MELANOPIC_DECISION = Threshold(250.0, "ge",
                               "Exploratory 250 lx melanopic EDI threshold")


# ==========================================================================
# Case B — a benchmark comparison between two models
# ==========================================================================

_FORMATS = {
    # each model has formatting habits; the scorer's extractor interacts
    # with them, which is the mechanism behind most reported rank reversals
    "bare": "{a}",
    "sentence": "The answer is {a}.",
    "boxed": "We compute step by step. \\\\boxed{{{a}}}",
    "worked": "First {d1}, then {d2}. So the result is {a}",
    "trailing": "{a} (approximately)",
    "units": "{a} metres",
}


def make_corpus(n_items: int = 400, seed: int = 0):
    """
    A fixture of model outputs.

    No model is called.  The specification debt studied here lives entirely in
    the scoring pipeline, which is the point: a benchmark can be irreproducible
    without any stochasticity in the model at all.
    """
    rng = np.random.default_rng(seed)
    gold = rng.integers(2, 999, size=n_items)

    def build(acc, style_mix, abstain_rate, tag):
        rows = []
        for i, g in enumerate(gold):
            if rng.random() < abstain_rate:
                rows.append({"gold": int(g), "text": "I cannot determine this.",
                             "abstain": True, "correct_value": None})
                continue
            correct = rng.random() < acc
            a = int(g) if correct else int(g) + int(rng.integers(1, 9))
            style = rng.choice(list(style_mix), p=list(style_mix.values()))
            d1, d2 = int(rng.integers(2, 50)), int(rng.integers(2, 50))
            text = _FORMATS[style].format(a=a, d1=d1, d2=d2)
            if style == "worked" and rng.random() < 0.5:
                text += "."
            rows.append({"gold": int(g), "text": text, "abstain": False,
                         "correct_value": a})
        return {"name": tag, "rows": rows}

    # A is marginally more accurate and tends toward worked prose and boxed
    # answers; B is marginally weaker and answers in the format most
    # extractors expect.  The gap is deliberately small and the format
    # distributions deliberately overlap, so that no single extractor is a
    # clean discriminator and ordinary resampling noise is in play too.
    a = build(0.710, {"worked": 0.34, "boxed": 0.16, "sentence": 0.16,
                      "bare": 0.10, "trailing": 0.14, "units": 0.10},
              0.055, "Model A")
    b = build(0.690, {"sentence": 0.38, "worked": 0.16, "boxed": 0.06,
                      "bare": 0.18, "trailing": 0.13, "units": 0.09},
              0.020, "Model B")
    return a, b


_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_AFTER = re.compile(r"(?:answer is|result is)\s*:?\s*([^\s.,;)]+)",
                    re.IGNORECASE)


def extract(text: str, rule: str):
    if rule == "boxed_else_last":
        m = _BOXED.search(text)
        if m:
            return m.group(1)
        nums = _NUM.findall(text)
        return nums[-1] if nums else None
    if rule == "after_answer_is":
        m = _AFTER.search(text)
        return m.group(1) if m else None
    nums = _NUM.findall(text)
    if not nums:
        return None
    return nums[0] if rule == "first_number" else nums[-1]


def normalise(s, rule: str):
    if s is None:
        return None
    s = s.strip()
    if rule == "exact":
        return s
    if rule == "strip_punct":
        return s.strip(" .,;:)($\\")
    if rule == "numeric_tolerant":
        m = _NUM.search(s)
        if not m:
            return None
        try:
            return f"{float(m.group(0)):.6g}"
        except ValueError:
            return None
    raise KeyError(rule)


def accuracy(corpus, spec: dict) -> float:
    rows = corpus["rows"]
    if spec["subsample_seed"] is not None:
        rng = np.random.default_rng(int(spec["subsample_seed"]))
        idx = rng.choice(len(rows), size=int(0.8 * len(rows)), replace=False)
        rows = [rows[i] for i in idx]
    lim = int(spec["max_chars"])
    hit, total = 0, 0
    for r in rows:
        if r["abstain"]:
            if spec["abstention"] == "excluded":
                continue
            total += 1
            continue
        total += 1
        pred = normalise(extract(r["text"][:lim], spec["extraction"]),
                         spec["normalisation"])
        gold = normalise(str(r["gold"]), spec["normalisation"])
        if pred is not None and pred == gold:
            hit += 1
    return hit / max(total, 1)


def benchmark_specification() -> Specification:
    s = Specification(name="Benchmark comparison, Model A vs Model B")
    s.add("extraction", ("last_number", "first_number", "after_answer_is",
                         "boxed_else_last"),
          "Four answer-extraction rules in common use across public "
          "evaluation harnesses. None is mandated by the benchmark.",
          domain="scoring")
    s.add("normalisation", ("exact", "strip_punct", "numeric_tolerant"),
          "String equality, punctuation-stripped equality, and numeric "
          "equality. Harnesses differ and rarely say which.",
          domain="scoring")
    s.add("abstention", ("wrong", "excluded"),
          "Whether a refusal counts against the model or is dropped from the "
          "denominator.", domain="scoring")
    s.add("max_chars", (200, 4000),
          "Response truncation before scoring. A real harness setting that "
          "interacts with chain-of-thought formats.", domain="scoring")
    s.add("subsample_seed", (None, 1, 2),
          "Whether the full set or an 80% resample is scored. Included to "
          "separate ordinary sampling noise from specification effects.",
          domain="sampling")
    return s


def benchmark_metric(n_items: int = 400, seed: int = 0):
    """Metric is accuracy(A) - accuracy(B): the sign is the published verdict."""
    a, b = make_corpus(n_items, seed)

    def fn(s):
        return accuracy(a, s) - accuracy(b, s)
    fn.corpus = (a, b)
    return fn


BENCHMARK_DECISION = Comparison("Model A scores above Model B")
