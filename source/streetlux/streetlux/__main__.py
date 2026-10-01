"""Command line entry point:  python -m streetlux ..."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import self_check
from .report import write_html, write_json
from .route import load_route
from .scenarios import OBJECTIVES, compare_designs, compare_modes

#: reference_data/ ships as a sibling of streetlux/, shadeparadox/ and
#: specdebt/ (three .parent calls up from streetlux/streetlux/__main__.py).
#: Loading it here by default -- not only when --reference-dir is typed out
#: by hand -- is what makes the plain `python -m streetlux` run on the
#: official baseline instead of silently falling back to the analytic
#: approximation.
DEFAULT_REFERENCE_DIR = Path(__file__).resolve().parent.parent.parent / "reference_data"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="streetlux",
        description="Melanopic exposure review for urban routes "
                    "(lx melanopic EDI, CIE S 026).")
    p.add_argument("route", help="route JSON")
    p.add_argument("--review", choices=["modes", "designs"], default="modes")
    p.add_argument("--mode", default="walk",
                   help="traveller for a design review")
    p.add_argument("--modes", nargs="*", default=None)
    p.add_argument("--objective", default="balanced", choices=sorted(OBJECTIVES))
    p.add_argument("--headings", type=int, default=24)
    p.add_argument("--directions", type=int, default=768)
    p.add_argument("--step-seconds", type=float, default=60.0)
    p.add_argument("--hour", type=float, default=None,
                   help="override departure time, local decimal hours")
    p.add_argument("--cloud", type=float, default=None,
                   help="override cloud cover 0..1")
    p.add_argument("-o", "--out", default="out", help="output directory")
    p.add_argument("--self-check", action="store_true")
    p.add_argument(
        "--reference-dir", default=None,
        help=("directory containing CIE_sle_photopic.csv, "
              "CIE_a-opic_action_spectra.csv and e490_00a_amo_nm.csv"))
    p.add_argument(
        "--publication", action="store_true",
        help="refuse to run unless official CIE action spectra and ASTM E490 are loaded")
    a = p.parse_args(argv)

    ref_info = None
    explicit = a.reference_dir is not None
    ref_path = Path(a.reference_dir) if explicit else DEFAULT_REFERENCE_DIR
    if ref_path.is_dir():
        from .reference_data import load_publication_references
        ref_info = load_publication_references(ref_path)
        print("reference data:")
        print(json.dumps(ref_info, indent=2, default=float))
    elif explicit:
        raise FileNotFoundError(f"--reference-dir {ref_path} does not exist")
    else:
        print(f"note: official reference data not found at the default "
              f"location ({ref_path}); running on the analytic spectral "
              f"approximation, which spectra.py documents as adequate for "
              f"design review only. Pass --reference-dir PATH, or restore "
              f"this package's reference_data/ sibling directory, for "
              f"publication-grade numbers.", file=sys.stderr)
    if a.publication:
        from . import spectra as _spectra
        state = _spectra.provenance()
        missing = [name for name, ok in (
            ("official CIE photopic/melanopic action spectra",
             state["official_action_spectra"]),
            ("ASTM E490 solar spectrum", state["official_solar_spectrum"]))
                  if not ok]
        if missing:
            raise RuntimeError(
                "Publication run blocked because these inputs are not "
                "loaded: " + ", ".join(missing) + ". Supply --reference-dir PATH.")

    if a.self_check:
        print(json.dumps(self_check(), indent=2, default=float))

    route = load_route(a.route)
    if a.hour is not None:
        route.start_hour = a.hour
    if a.cloud is not None:
        from dataclasses import replace
        route.atmosphere = replace(route.atmosphere, cloud_cover=a.cloud)

    kw = dict(n_headings=a.headings, n_directions=a.directions,
              step_seconds=a.step_seconds)
    if a.review == "modes":
        res = compare_modes(route, a.modes, objective=a.objective, **kw)
    else:
        res = compare_designs(route, a.mode, objective=a.objective, **kw)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{a.review}_{route.name.replace(' ', '_').replace('/', '-')}"
    j = write_json(res, out / f"{stem}.json")
    h = write_html(res, out / f"{stem}.html")
    print(f"wrote {j}\nwrote {h}")
    for r in res["rows"]:
        s = r["summary"]
        print(f"  {r['label']:<34} {s['mean_medi_travel_lx']:8.0f} lx mEDI"
              f"   worst-hdg {s['mean_medi_worst_heading_lx']:7.0f}"
              f"   {100*s['frac_time_above_250_travel']:5.0f}% of trip >=250")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
