"""Run the three experiments and write the artefacts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import streetlux
from streetlux import photometry, spectra

from . import self_check
from .experiment import e1_behaviour, e2_encoding_vs_algorithm, e3_view_direction
from .report import write_html, write_json

#: reference_data/ ships as a sibling of streetlux/, shadeparadox/ and
#: specdebt/ (three .parent calls up from shadeparadox/shadeparadox/__main__.py).
#: Loading it here by default -- not only when --reference-dir is typed out
#: by hand -- is what makes the plain `python -m shadeparadox` run on the
#: official baseline instead of silently falling back to the analytic
#: approximation.
DEFAULT_REFERENCE_DIR = Path(__file__).resolve().parent.parent.parent / "reference_data"


def _load_references(reference_dir: str | None) -> dict | None:
    """
    Load the official CIE/ASTM tables, defaulting to DEFAULT_REFERENCE_DIR.

    An explicit --reference-dir that does not exist is an error; the
    default silently missing (this package copied out of the repository
    on its own) is a warning, not a crash, for anything short of
    --publication, which is separately enforced below regardless of how
    we got here.
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="shadeparadox",
        description="Coupled thermal/visual/circadian shading experiments.")
    p.add_argument("-o", "--out", default="out")
    p.add_argument("--quick", action="store_true",
                   help="smaller grids and budgets, for a smoke run")
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--shade", default="roller_dark")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument(
        "--reference-dir", default=None,
        help=("directory containing CIE_sle_photopic.csv, "
              "CIE_a-opic_action_spectra.csv and e490_00a_amo_nm.csv")
    )
    p.add_argument(
        "--publication", action="store_true",
        help="refuse to run unless official CIE action spectra and ASTM E490 are loaded"
    )
    a = p.parse_args(argv)

    ref_info = _load_references(a.reference_dir)
    if a.publication:
        _require_publication_spectra()

    quick = a.quick
    t0 = time.time()
    print(json.dumps(self_check(), indent=2, default=float))
    if ref_info is not None:
        print("\nreference data:")
        print(json.dumps(ref_info, indent=2, default=float))

    e1 = e1_behaviour(
        orientations=(90, 270) if quick else (0, 90, 180, 270),
        depths=(2.5, 6.5) if quick else (2.5, 4.5, 6.5),
        shade_key=a.shade,
        n_directions=192 if quick else 256,
        dt_h=1.0 if quick else 0.5,
        seeds=tuple(range(2 if quick else a.seeds)))
    print(f"\nE1 done in {e1['wall_clock_s']}s")
    for ck, v in e1["by_controller"].items():
        r = v["medi_retained"]
        print(f"  {v['label']:<42} mEDI retained {r['mean']:.2f} "
              f"[{r['lo']:.2f}, {r['hi']:.2f}]  "
              f"compliance {v['compliance_delta_pp']['mean']:+.1f} pp")

    e2 = e2_encoding_vs_algorithm(
        budget=a.budget or (8 if quick else 16),
        n_directions=192, dt_h=1.0 if quick else 0.5, shade_key=a.shade)
    print(f"\nE2 done in {e2['wall_clock_s']}s "
          f"({e2['unique_evaluations']} unique simulations)")
    r = e2["encoding_over_algorithm_ratio"]
    print(f"  algorithm dispersion {e2['dispersion_from_algorithm']['mean']:.4f}")
    print(f"  encoding  dispersion {e2['dispersion_from_encoding']['mean']:.4f}")
    print("  ratio encoding/algorithm", r["mean"], r["status"])

    e3 = e3_view_direction(
        depth=2.5, n_headings=6 if quick else 12,
        n_directions=192 if quick else 256,
        dt_h=1.0 if quick else 0.5, shade_key=a.shade)
    print(f"\nE3 done in {e3['wall_clock_s']}s")
    for row in e3["rows"]:
        print(f"  {row['label']:<42} mEDI {row['medi_min']:6.0f}-"
              f"{row['medi_max']:6.0f}  ratio {row['medi_ratio']:.2f}  "
              f"coverage range {row['coverage_range']:.2f}")

    cal = photometry.calibration()
    spectral_state = spectra.provenance()
    prov = {
        "package": "shadeparadox",
        "metrology_from": f"streetlux {streetlux.__version__}",
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
        "quick": quick, "shade": a.shade,
        "wall_clock_s": round(time.time() - t0, 2),
        "units": "melanopic values are lx melanopic EDI (CIE S 026), "
                 "not photopic lux",
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bundle = {"e1": e1, "e2": e2, "e3": e3, "provenance": prov}
    print("\nwrote", write_json(bundle, out / "shadeparadox_results.json"))
    print("wrote", write_html(e1, e2, e3, prov,
                              out / "shadeparadox_report.html"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
