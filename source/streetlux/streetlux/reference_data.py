"""Load publication-grade spectral reference data.

Expected reference files
------------------------
CIE_sle_photopic.csv
    Official CIE photopic luminous-efficiency function V(lambda).

CIE_a-opic_action_spectra.csv
    Official CIE S 026 alpha-opic action spectra.  Column 6 (zero-based
    index 5) is the melanopic action spectrum.

e490_00a_amo_nm.csv
    ASTM E490 extraterrestrial solar spectrum converted to wavelength in nm
    and spectral irradiance in W m-2 nm-1.

The CIE source tables are kept at their native 1 nm resolution when read and
are linearly resampled onto streetlux's 5 nm computational grid.  The ASTM
file is already unit-converted; no hidden micron-to-nm conversion occurs at
runtime.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from . import photometry, spectra

# Exact hashes of the official CIE CSV files supplied for this study.
CIE_PHOTOPIC_MD5 = "f389958555461a7d9a7562145e8ca9c0"
CIE_AOPIC_MD5 = "f1ddfef144176812c3cb9d8fba1f3141"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _require_monotonic(name: str, wl: np.ndarray) -> None:
    if wl.ndim != 1 or len(wl) < 2:
        raise ValueError(f"{name}: wavelength array is invalid")
    if not np.all(np.isfinite(wl)):
        raise ValueError(f"{name}: wavelength array contains non-finite values")
    if not np.all(np.diff(wl) > 0):
        raise ValueError(f"{name}: wavelengths must be strictly increasing")


def load_publication_references(reference_dir: str | Path,
                                verify_cie_checksums: bool = True) -> dict:
    """Load CIE action spectra and ASTM E490 for a publication run.

    Parameters
    ----------
    reference_dir:
        Directory containing the three expected CSV files.
    verify_cie_checksums:
        When True (default), reject CIE files whose MD5 hashes differ from
        the official source files used to prepare this study.

    Returns
    -------
    dict
        A compact validation/provenance record suitable for printing or
        embedding in a results JSON file.
    """
    root = Path(reference_dir).expanduser().resolve()
    photopic_file = root / "CIE_sle_photopic.csv"
    aopic_file = root / "CIE_a-opic_action_spectra.csv"
    solar_file = root / "e490_00a_amo_nm.csv"

    for path in (photopic_file, aopic_file, solar_file):
        if not path.is_file():
            raise FileNotFoundError(f"Required reference file not found: {path}")

    if verify_cie_checksums:
        got = _md5(photopic_file)
        if got != CIE_PHOTOPIC_MD5:
            raise ValueError(
                "CIE_sle_photopic.csv checksum mismatch: "
                f"expected {CIE_PHOTOPIC_MD5}, got {got}"
            )
        got = _md5(aopic_file)
        if got != CIE_AOPIC_MD5:
            raise ValueError(
                "CIE_a-opic_action_spectra.csv checksum mismatch: "
                f"expected {CIE_AOPIC_MD5}, got {got}"
            )

    # Official CIE photopic V(lambda): wavelength_nm, V(lambda)
    photopic = np.loadtxt(photopic_file, delimiter=",")
    if photopic.ndim != 2 or photopic.shape[1] < 2:
        raise ValueError("Unexpected CIE photopic CSV layout")
    v_wl = np.asarray(photopic[:, 0], float)
    v_lambda = np.asarray(photopic[:, 1], float)
    _require_monotonic("CIE photopic", v_wl)
    if not np.all(np.isfinite(v_lambda)) or np.any(v_lambda < 0):
        raise ValueError("CIE photopic values must be finite and non-negative")

    # CIE S 026 alpha-opic action spectra:
    # wavelength, S-cone-opic, M-cone-opic, L-cone-opic, rhodopic, melanopic
    aopic = np.loadtxt(aopic_file, delimiter=",")
    if aopic.ndim != 2 or aopic.shape[1] < 6:
        raise ValueError("Unexpected CIE alpha-opic CSV layout")
    a_wl = np.asarray(aopic[:, 0], float)
    s_mel = np.asarray(aopic[:, 5], float)
    _require_monotonic("CIE alpha-opic", a_wl)
    if not np.all(np.isfinite(s_mel)) or np.any(s_mel < 0):
        raise ValueError("CIE melanopic values must be finite and non-negative")

    # Align both CIE functions on the package's 5 nm computational grid.
    # CIE metadata specifies linear interpolation and zero extrapolation.
    wl = spectra.WL
    v_5nm = np.interp(wl, v_wl, v_lambda, left=0.0, right=0.0)
    mel_5nm = np.interp(wl, a_wl, s_mel, left=0.0, right=0.0)
    spectra.load_official_action_spectra(wl, v_5nm, mel_5nm)

    # Clean converted ASTM E490 file:
    # wavelength_nm,e490_w_m2_nm
    solar = np.genfromtxt(solar_file, delimiter=",", names=True, dtype=float)
    names = solar.dtype.names or ()
    required = {"wavelength_nm", "e490_w_m2_nm"}
    if not required.issubset(names):
        raise ValueError(
            "e490_00a_amo_nm.csv must contain headers "
            "wavelength_nm,e490_w_m2_nm"
        )
    solar_wl = np.atleast_1d(np.asarray(solar["wavelength_nm"], float))
    solar_e0 = np.atleast_1d(np.asarray(solar["e490_w_m2_nm"], float))
    good = np.isfinite(solar_wl) & np.isfinite(solar_e0)
    solar_wl, solar_e0 = solar_wl[good], solar_e0[good]
    _require_monotonic("ASTM E490", solar_wl)
    if np.any(solar_e0 < 0):
        raise ValueError("ASTM E490 spectral irradiance cannot be negative")
    if solar_wl.min() > spectra.WL.min() or solar_wl.max() < spectra.WL.max():
        raise ValueError("ASTM E490 file does not cover the 380-780 nm model grid")

    spectra.load_official_solar_spectrum(solar_wl, solar_e0)

    cal = photometry.calibration()
    state = spectra.provenance()
    result = {
        "reference_dir": str(root),
        "official_action_spectra": bool(state["official_action_spectra"]),
        "official_solar_spectrum": bool(state["official_solar_spectrum"]),
        "official_daylight_components": bool(
            state["official_daylight_components"]
        ),
        "cie_photopic_md5": _md5(photopic_file),
        "cie_aopic_md5": _md5(aopic_file),
        "astm_e490_nm_sha256": _sha256(solar_file),
        "model_grid_nm": {
            "start": float(wl[0]),
            "stop": float(wl[-1]),
            "step": float(wl[1] - wl[0]),
        },
        "photopic_peak_nm": float(wl[np.argmax(spectra.V_LAMBDA)]),
        "melanopic_peak_nm": float(wl[np.argmax(spectra.S_MEL)]),
        "k_mel_v_d65_derived_W_per_lm": float(cal["k_mel"]),
        "k_mel_v_d65_official_W_per_lm": float(
            photometry.K_MEL_V_D65_OFFICIAL
        ),
        "k_mel_deviation_pct": float(cal["deviation_pct"]),
        "der_d65_numerical": float(photometry.der(spectra.d65())),
    }

    if not result["official_action_spectra"]:
        raise RuntimeError("Official CIE action spectra were not activated")
    if not result["official_solar_spectrum"]:
        raise RuntimeError("Official ASTM E490 spectrum was not activated")

    return result
