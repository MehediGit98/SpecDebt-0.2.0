#!/usr/bin/env python3
"""Convert the first two ASTM E490 CSV columns from micron units to nm units.

Input columns (from the spreadsheet export):
    Wavelength, microns
    E-490 W/m2/micron

Output columns:
    wavelength_nm
    e490_w_m2_nm

Conversions:
    wavelength_nm = wavelength_microns * 1000
    W/m2/nm       = W/m2/micron / 1000
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def convert(src: Path, dst: Path) -> int:
    rows = []
    # The saved spreadsheet may contain a micro symbol encoded in a legacy
    # code page elsewhere in the row.  Latin-1 preserves bytes safely; only
    # the first two numeric columns are used.
    with src.open("r", encoding="latin-1", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 2:
                continue
            try:
                wavelength_um = float(row[0])
                irradiance_um = float(row[1])
            except (TypeError, ValueError):
                continue
            rows.append((wavelength_um * 1000.0, irradiance_um / 1000.0))

    if not rows:
        raise ValueError("No ASTM E490 numeric rows found in the first two columns")
    if any(rows[i + 1][0] <= rows[i][0] for i in range(len(rows) - 1)):
        raise ValueError("Converted wavelength values are not strictly increasing")

    with dst.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["wavelength_nm", "e490_w_m2_nm"])
        for wl_nm, e_nm in rows:
            writer.writerow([f"{wl_nm:.12g}", f"{e_nm:.12g}"])
    return len(rows)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) not in (1, 2):
        print("usage: python convert_e490_to_nm.py INPUT.csv [OUTPUT.csv]")
        return 2
    src = Path(argv[0]).resolve()
    dst = Path(argv[1]).resolve() if len(argv) == 2 else src.with_name(
        "e490_00a_amo_nm.csv"
    )
    n = convert(src, dst)
    print(f"converted {n} ASTM E490 rows")
    print(f"source_sha256={sha256(src)}")
    print(f"output_sha256={sha256(dst)}")
    print(dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
