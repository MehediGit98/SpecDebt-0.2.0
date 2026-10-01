"""Photometry: spectral irradiance -> photopic illuminance, melanopic EDI, DER.

Unit discipline
---------------
Melanopic EDI is reported in lux, but it is not photopic lux.  It is a
D65-referred equivalence scale that shares the unit.  Every public field name
in this package spells it ``medi_lx`` and every report label reads
"lx melanopic EDI".  Do not shorten it to "melanopic lux" anywhere.

Publication convention
----------------------
Melanopic EDI uses the CIE S 026 D65 melanopic efficacy constant
K_mel,v^D65 = 1.3262e-3 W/lm.  ``calibrate()`` independently derives the same
constant on the package's numerical wavelength grid and reports the residual
error; it does not redefine the standard constant.
"""

from __future__ import annotations

import numpy as np

from . import spectra

#: CIE S 026:2018 melanopic efficacy of luminous radiation for D65, W/lm.
K_MEL_V_D65_OFFICIAL = 1.3262e-3

#: Photopic maximum luminous efficacy, lm/W.
K_M = 683.0

_CAL = {"k_mel": None, "deviation_pct": None}


def calibrate() -> dict:
    """Numerically derive K_mel,v^D65 from the currently loaded spectra.

    This is a diagnostic check of spectral tables and wavelength resolution.
    Published mEDI values use :data:`K_MEL_V_D65_OFFICIAL`, not this derived
    value.  With the official CIE tables loaded, the residual should be very
    small; with the legacy analytic approximation it is expected to be larger.
    """
    sp = spectra.d65()
    e_mel = np.trapezoid(sp * spectra.S_MEL, spectra.WL)
    e_v = K_M * np.trapezoid(sp * spectra.V_LAMBDA, spectra.WL)
    k = e_mel / e_v
    _CAL["k_mel"] = float(k)
    _CAL["deviation_pct"] = float(
        100.0 * (k - K_MEL_V_D65_OFFICIAL) / K_MEL_V_D65_OFFICIAL
    )
    return dict(_CAL)


calibrate()


def k_mel() -> float:
    """Return the numerically derived diagnostic K_mel,v^D65."""
    return float(_CAL["k_mel"])


def calibration() -> dict:
    return dict(_CAL)


def illuminance(spd: np.ndarray) -> float:
    """Photopic illuminance (lx) from spectral irradiance (W m-2 nm-1)."""
    return float(K_M * np.trapezoid(spd * spectra.V_LAMBDA, spectra.WL))


def melanopic_irradiance(spd: np.ndarray) -> float:
    """Melanopic irradiance E_e,mel (W m-2)."""
    return float(np.trapezoid(spd * spectra.S_MEL, spectra.WL))


def medi(spd: np.ndarray) -> float:
    """Melanopic equivalent daylight illuminance (lx melanopic EDI)."""
    return melanopic_irradiance(spd) / K_MEL_V_D65_OFFICIAL


def der(spd: np.ndarray) -> float:
    """Melanopic daylight efficacy ratio, dimensionless."""
    ev = illuminance(spd)
    if ev <= 0.0:
        return float("nan")
    return medi(spd) / ev


def chromaticity(spd: np.ndarray) -> tuple[float, float]:
    """CIE 1931 (x, y) — used by the self-tests to validate the D-series."""
    X = np.trapezoid(spd * spectra.XBAR, spectra.WL)
    Y = np.trapezoid(spd * spectra.YBAR, spectra.WL)
    Z = np.trapezoid(spd * spectra.ZBAR, spectra.WL)
    s = X + Y + Z
    return float(X / s), float(Y / s)


def scale_to_illuminance(spd: np.ndarray, lux: float) -> np.ndarray:
    """Rescale a relative SPD so that it produces `lux` photopic lux."""
    cur = illuminance(spd)
    if cur <= 0:
        raise ValueError("SPD carries no photopic power")
    return spd * (lux / cur)
