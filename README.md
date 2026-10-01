# SpecDebt 0.2.0 — code, results, tables and figures supplement

This archive is the **computational/reproducibility supplement** for the SpecDebt MethodsX article. It intentionally does **not** reproduce or rebuild the manuscript. It contains the complete local source needed to regenerate the reported computational results, all archived numerical result tables, standalone CSV versions of the article tables, verification records, and the article figures.

## Contents

- `source/` — local Python packages and reference data used by the experiments.
- `results/` — archived published-run outputs, complete per-specification grids, declaration curves, synthetic corpus, metrology checks and run metadata.
- `tables/` — standalone CSV versions of the two article tables, plus table documentation.
- `figures/` — all article figures and graphical abstract in PNG, SVG and PDF formats.
- `pipeline.py` — experiment settings and result export.
- `reproduce.py` — reproduces the archived study outputs and figures.
- `verify.py` — standard-library regression checks.
- `compare_results.py` — numerical comparison against archived results.
- `make_figures.py` — figure-generation script.
- `verification/` — archived regression/reproduction logs and supplement-only verification metadata.
- `requirements.txt` — computational dependencies.
- `MANIFEST_SHA256.json` — SHA-256 manifest for the release files.

## Reproduce the results

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe verify.py
.\.venv\Scripts\python.exe reproduce.py
```

### Linux/macOS

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python verify.py
.venv/bin/python reproduce.py
```

`reproduce.py` writes a fresh `reproduced/` directory, compares numerical fields in the principal JSON outputs against `results/`, and redraws the figures as 300 dpi PNG, vector SVG and PDF. The submitted `results/` directory is the fixed baseline. Do not add generated `reproduced/` files to the release archive.

The verified reference environment recorded in `results/run_metadata.json` is Python 3.12.14 with NumPy 2.3.5 on Linux. The experiments use only bundled source/reference data and do not require a trained model, API key, remote model or external account. Installing dependencies may require internet access; the experiments themselves do not.

## Results and tables

The `results/` directory contains the complete numerical record, including both per-specification grids, both exact declaration curves, the synthetic corpus, metrology checks and run metadata. The `tables/` directory provides the two article tables as standalone CSV files so they can be inspected without the manuscript.

## Figures

`figures/` contains Figures 1–4 and the graphical abstract, each supplied as PNG, SVG and PDF. These are deterministic plots generated from the bundled results/source.

## Reference data and scope

Reference-data provenance and attribution are documented in `source/reference_data/README.md`. The supplied metrology record includes the D65 calibration check and explicitly records which official reference tables were loaded. The experiments are finite-grid/synthetic demonstrations; the package does not claim matched Radiance or field validation for these scenes.

## Release hygiene

The archive intentionally excludes manuscript DOCX/PDF files, manuscript Markdown, manuscript-authoring sources, manuscript-only dependencies, merge/audit drafts, Python bytecode caches, and other manuscript-build material. No project licence is asserted by this archive; the author should select the appropriate licence before any public code release.
