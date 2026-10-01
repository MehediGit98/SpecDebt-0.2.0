"""Run both specification-debt case studies and write the artefacts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import streetlux
from streetlux import photometry, spectra

from . import (BENCHMARK_DECISION, MELANOPIC_DECISION, benchmark_metric,
               benchmark_specification, melanopic_metric,
               melanopic_specification, run_study)
from .report import write_html, write_json

#: This package ships with reference_data/ as a sibling of streetlux/,
#: shadeparadox/ and specdebt/ (three .parent calls up from this file, at
#: specdebt/specdebt/__main__.py). Loading it by default here -- rather
#: than only when --reference-dir is typed out by hand -- is what makes
#: the plain `python -m specdebt` run on the official baseline instead of
#: silently falling back to the analytic approximation.
DEFAULT_REFERENCE_DIR = Path(__file__).resolve().parent.parent.parent / "reference_data"


def _load_references(reference_dir: str | None) -> dict | None:
    """
    Load the official CIE/ASTM tables.

    Defaults to DEFAULT_REFERENCE_DIR so the plain CLI invocation runs on
    the official baseline the same way reproduce.py does. An explicit
    --reference-dir that does not exist is an error (the user asked for a
    specific place); the default silently missing (e.g. this package
    copied out of the repository on its own) is a warning, not a crash,
    for anything short of --publication -- which is separately enforced
    by _require_publication_spectra below regardless of how we got here.
    """
    explicit = reference_dir is not None
    path = Path(reference_dir) if explicit else DEFAULT_REFERENCE_DIR
    if not path.is_dir():
        if explicit:
            raise FileNotFoundError(f"--reference-dir {path} does not exist")
        print(f"note: official reference data not found at the default "
              f"location ({path}); running on the analytic spectral "
              f"approximation, which spectra.py documents as adequate for "
              f"design review only. Pass --reference-dir PATH, or restore "
              f"this package's reference_data/ sibling directory, for "
              f"publication-grade numbers.", file=sys.stderr)
        return None
    from streetlux.reference_data import load_publication_references
    return load_publication_references(path)


def _require_publication_spectra() -> None:
    state = spectra.provenance()
    missing = []
    if not state["official_action_spectra"]:
        missing.append("official CIE photopic/melanopic action spectra")
    if not state["official_solar_spectrum"]:
        missing.append("ASTM E490 solar spectrum")
    if missing:
        raise RuntimeError(
            "Publication run blocked because these inputs are not loaded: "
            + ", ".join(missing)
            + ". Supply --reference-dir PATH."
        )


def _row(name, domain, debt):
    p = debt["profile"]
    return {"case": name, "domain": domain,
            "flip_rate": debt["flip_rate"],
            "envelope_ratio": debt["envelope"]["ratio_max_min"],
            "envelope_spread": debt["envelope"]["spread"],
            "crosses_zero": debt["envelope"]["crosses_zero"],
            "herfindahl": p["herfindahl"], "shape": p["shape"],
            "declarations": p["declarations_to_determinacy"],
            "top_factor": p["top_factor"],
            "entropy_bits": debt["decision_entropy_bits"]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="specdebt",
        description="Measure the specification debt of a reported metric.")
    p.add_argument("-o", "--out", default="out")
    p.add_argument("--tol", type=float, default=0.01)
    p.add_argument("--items", type=int, default=600)
    p.add_argument("--directions", type=int, default=192)
    p.add_argument("--orientation", type=float, default=270.0)
    p.add_argument("--depth", type=float, default=3.0)
    p.add_argument(
        "--reference-dir", default=None,
        help=("directory containing CIE_sle_photopic.csv, "
              "CIE_a-opic_action_spectra.csv and e490_00a_amo_nm.csv")
    )
    p.add_argument(
        "--publication", action="store_true",
        help=("refuse to run unless official CIE action spectra and ASTM E490 "
              "are loaded as the physical baseline")
    )
    a = p.parse_args(argv)

    # Load the publication-grade physical baseline BEFORE either case study.
    # Case A still contains a deliberate legacy-analytic sensitivity factor;
    # that factor is applied explicitly in cases.py and does not change the
    # underlying spectral engine back to the old approximation.
    ref_info = _load_references(a.reference_dir)
    if a.publication:
        _require_publication_spectra()

    t0 = time.time()

    if ref_info is not None:
        print("reference data:")
        print(json.dumps(ref_info, indent=2, default=float))

    mel = run_study(melanopic_metric(a.orientation, a.depth, a.directions,
                                     require_official_baseline=(
                                         ref_info is not None)),
                    melanopic_specification(), MELANOPIC_DECISION,
                    "melanopic EDI at the eye", "lx melanopic EDI")
    mel_debt = mel.debt(a.tol)

    ben = run_study(benchmark_metric(a.items), benchmark_specification(),
                    BENCHMARK_DECISION, "accuracy(A) − accuracy(B)",
                    "accuracy points")
    ben_debt = ben.debt(a.tol)

    for nm, d in (("250 lx melanopic threshold test", mel_debt),
                  ("Benchmark comparison A vs B", ben_debt)):
        e = d["envelope"]
        print(f"\n{nm}")
        print(f"  specifications      {d['n_specifications']} "
              f"({d['mode']}), {d['n_undeclared_factors']} undeclared")
        tail = (f"(spans zero, spread {e['spread']:.4g})"
                if e["crosses_zero"] else f"({e['ratio_max_min']:.1f}x)")
        print(f"  envelope            {e['min']:.4g} .. {e['max']:.4g} {tail}")
        print(f"  verdict flips in    {100*d['flip_rate']:.1f}% "
              f"({d['decision_entropy_bits']:.2f} bits)")
        print(f"  profile             {d['profile']['shape']}, "
              f"top factor {d['profile']['top_factor']} "
              f"(S1={d['profile']['top_factor_share']:.2f})")
        m = d["minimum_declaration_set"]
        print(f"  declare             {m['set']} -> "
              f"{100*m['residual_flip_rate']:.2f}% ({m['search']})")

    rows = [_row("250 lx melanopic threshold test", "architectural lighting",
                 mel_debt),
            _row("Benchmark comparison A vs B", "LLM evaluation", ben_debt)]
    hh = [r["herfindahl"] for r in rows]
    reading = ("These finite, uniformly weighted grids have different first-order "
               "concentration profiles and minimum declaration requirements. "
               "Minority share is conditional on the declared grid and decision.")
    comparison = {"rows": rows, "reading": reading}

    cal = photometry.calibration()
    spectral_state = spectra.provenance()
    prov = {
        "package": "specdebt",
        "tolerance": a.tol,
        "melanopic_cell": {"orientation_deg": a.orientation,
                           "desk_depth_m": a.depth,
                           "n_directions": a.directions},
        "benchmark_items": a.items,
        "optics_from": "shadeparadox + streetlux",
        "spectral_inputs": spectral_state,
        "official_action_spectra": spectral_state["official_action_spectra"],
        "official_solar_spectrum": spectral_state["official_solar_spectrum"],
        "official_daylight_components":
            spectral_state["official_daylight_components"],
        "k_mel_v_d65_derived_W_per_lm": cal["k_mel"],
        "k_mel_v_d65_official_W_per_lm":
            photometry.K_MEL_V_D65_OFFICIAL,
        "k_mel_deviation_from_CIE_S026_pct": cal["deviation_pct"],
        "known_biases": spectra.active_biases(),
        "reference_data": ref_info,
        "legacy_analytic_factor_note": (
            "The action_spectrum='analytic' level in Case A is an explicit "
            "legacy sensitivity factor applied to an otherwise official "
            "CIE/ASTM physical baseline; it is not the active spectral engine."
        ),
        "units_note": "melanopic values are lx melanopic EDI (CIE S 026), "
                      "not photopic lux",
        "wall_clock_s": round(time.time() - t0, 2),
    }

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bundle = {"melanopic": mel_debt, "benchmark": ben_debt,
              "melanopic_specification": mel.spec.describe(),
              "benchmark_specification": ben.spec.describe(),
              "comparison": comparison, "provenance": prov}
    print("\nwrote", write_json(bundle, out / "specdebt_results.json"))
    print("wrote", write_html(
        [("Case A · 250 lx melanopic threshold test", mel, mel_debt, 250.0),
         ("Case B · a benchmark comparison", ben, ben_debt, 0.0)],
        comparison, prov, out / "specdebt_report.html"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
