"""
Spectral basis for streetlux.

Everything downstream is spectral. The module provides:

  * the wavelength grid (380-780 nm, 5 nm)
  * V(lambda) and the melanopic action spectrum s_mel(lambda)
  * the CIE daylight reconstruction S0/S1/S2 -> D-illuminant of any CCT
  * an extraterrestrial solar spectrum
  * spectral reflectances for tropical-urban surface materials
  * spectral transmittances for vehicle glazing and canopy foliage

APPROXIMATION POLICY
--------------------
Built-in action spectra are analytic (Wyman et al. multi-lobe Gaussians for
the CIE 1931 observer; a Govardovskii A1 template at 480 nm with a lens
transmittance for melanopsin).  They are adequate for design review and NOT
adequate for a manuscript number.  Call ``load_official_action_spectra`` with
the CIE S 026 tables before generating any value that will be published; the
provenance stamp on every result records which path was used.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Wavelength grid
# --------------------------------------------------------------------------

WL = np.arange(380.0, 781.0, 5.0)          # nm
DL = 5.0                                    # nm, uniform

_STATE = {"official_action_spectra": False,
          "official_daylight_components": False,
          "official_solar_spectrum": False}


#: Biases measured against reference data, declared rather than hidden.
#: Each entry names the input responsible and the fix.
KNOWN_BIASES = {
    "global_daylight_der_low": {
        "magnitude": "approx -10 to -15%",
        "detail": "DER of unobstructed global horizontal daylight comes out "
                  "near 0.85 where field measurement gives roughly 1.0. The "
                  "5778 K Planck stand-in for the extraterrestrial spectrum "
                  "is red-heavy in the visible band.",
        "fix": "load_official_solar_spectrum() with ASTM E490.",
        "affects": "absolute mEDI levels, not the ranking between modes or "
                   "scenarios, which share the same source spectrum.",
    },
    "analytic_action_spectra": {
        "magnitude": "K_mel,v^D65 about +1% against CIE S 026",
        "detail": "Govardovskii A1 template at 480 nm with an approximated "
                  "lens transmittance.",
        "fix": "load_official_action_spectra() with the CIE S 026 tables.",
        "affects": "all melanopic values uniformly.",
    },
    "one_bounce_canopy": {
        "magnitude": "unquantified",
        "detail": "Foliage is a slab with a Beer-Lambert interception "
                  "probability, not resolved geometry.",
        "fix": "none in this version; treat canopy scenarios as indicative.",
        "affects": "canopy scenarios only.",
    },
}


def provenance() -> dict:
    """Which data path each spectral input is currently using."""
    return dict(_STATE)


def active_biases() -> dict:
    """Return only approximation warnings that remain active.

    Loading official CIE action spectra resolves ``analytic_action_spectra``;
    loading ASTM E490 resolves ``global_daylight_der_low``.  The canopy
    simplification remains active because it is a model-form assumption rather
    than a spectral-data substitution.
    """
    out = dict(KNOWN_BIASES)
    if _STATE["official_action_spectra"]:
        out.pop("analytic_action_spectra", None)
    if _STATE["official_solar_spectrum"]:
        out.pop("global_daylight_der_low", None)
    return out


# --------------------------------------------------------------------------
# CIE 1931 colour matching functions (Wyman, Sloan & Shirley 2013)
# --------------------------------------------------------------------------

def _pg(x, mu, s1, s2):
    """Piecewise Gaussian."""
    s = np.where(x < mu, s1, s2)
    return np.exp(-0.5 * ((x - mu) / s) ** 2)


def _xbar(wl):
    return (1.056 * _pg(wl, 599.8, 37.9, 31.0)
            + 0.362 * _pg(wl, 442.0, 16.0, 26.7)
            - 0.065 * _pg(wl, 501.1, 20.4, 26.2))


def _ybar(wl):
    return (0.821 * _pg(wl, 568.8, 46.9, 40.5)
            + 0.286 * _pg(wl, 530.9, 16.3, 31.1))


def _zbar(wl):
    return (1.217 * _pg(wl, 437.0, 11.8, 36.0)
            + 0.681 * _pg(wl, 459.0, 26.0, 13.8))


XBAR = _xbar(WL)
YBAR = _ybar(WL)
ZBAR = _zbar(WL)

#: Photopic luminous efficiency function.  V(lambda) == ybar(lambda).
V_LAMBDA = YBAR.copy()


# --------------------------------------------------------------------------
# Melanopic action spectrum
# --------------------------------------------------------------------------

def _govardovskii(wl, lmax=480.0):
    """A1 visual-pigment nomogram (Govardovskii et al. 2000)."""
    x = lmax / wl
    a = 0.8795 + 0.0459 * np.exp(-((lmax - 344.0) ** 2) / 11940.0)
    A, B, C, D = 69.7, 28.0, -14.9, 0.674
    b, c = 0.922, 1.104
    alpha = 1.0 / (np.exp(A * (a - x)) + np.exp(B * (b - x))
                   + np.exp(C * (c - x)) + D)
    lmb = 189.0 + 0.315 * lmax
    d = -40.5 + 0.195 * lmax
    beta = 0.26 * np.exp(-(((wl - lmb) / d) ** 2))
    return alpha + beta


def _lens_transmittance(wl):
    """
    Crystalline-lens transmittance, standard ~32-year observer.

    Smooth approximation to the published age-32 lens transmission: ~0.55 at
    400 nm rising to ~0.97 beyond 600 nm.  Replaced wholesale when the
    official s_mel table is loaded.
    """
    return 0.97 / (1.0 + np.exp(-(wl - 424.0) / 26.0)) + 0.02


def _build_melanopic():
    s = _govardovskii(WL, 480.0) * _lens_transmittance(WL)
    return s / s.max()


#: Melanopic action spectrum, normalised to a maximum of 1 (CIE S 026 convention).
S_MEL = _build_melanopic()


def load_official_action_spectra(wl, v_lambda, s_mel):
    """
    Replace the analytic action spectra with official CIE tables.

    Parameters
    ----------
    wl : array of wavelengths (nm) for the supplied tables
    v_lambda : CIE V(lambda)
    s_mel : CIE S 026 melanopic action spectrum, peak-normalised

    Both are linearly resampled onto :data:`WL`.  Values outside the supplied
    range are zero, matching the CIE dataset interpolation/extrapolation rule.
    Call before any published number.
    """
    global V_LAMBDA, S_MEL, YBAR
    wl = np.asarray(wl, float)
    v_lambda = np.asarray(v_lambda, float)
    s_mel = np.asarray(s_mel, float)
    if wl.ndim != 1 or v_lambda.shape != wl.shape or s_mel.shape != wl.shape:
        raise ValueError("wl, v_lambda and s_mel must be equal-length 1-D arrays")
    if not np.all(np.isfinite(wl)) or not np.all(np.diff(wl) > 0):
        raise ValueError("wavelengths must be finite and strictly increasing")
    if not np.all(np.isfinite(v_lambda)) or not np.all(np.isfinite(s_mel)):
        raise ValueError("action spectra must be finite")
    if np.any(v_lambda < 0) or np.any(s_mel < 0):
        raise ValueError("action spectra cannot contain negative values")

    V_LAMBDA = np.interp(WL, wl, v_lambda, left=0.0, right=0.0)
    YBAR = V_LAMBDA.copy()
    s = np.interp(WL, wl, s_mel, left=0.0, right=0.0)
    peak = float(s.max())
    if peak <= 0.0:
        raise ValueError("melanopic action spectrum has no positive values")
    S_MEL = s / peak
    _STATE["official_action_spectra"] = True
    from . import photometry
    photometry.calibrate()


def using_official_tables() -> bool:
    return _STATE["official_action_spectra"]


# --------------------------------------------------------------------------
# CIE daylight components S0 / S1 / S2  (300-830 nm, 10 nm)
# --------------------------------------------------------------------------

_D_WL = np.arange(300.0, 831.0, 10.0)

_S0 = np.array([
    0.04, 6.00, 29.60, 55.30, 57.30, 61.80, 61.50, 68.80, 63.40, 65.80,
    94.80, 104.80, 105.90, 96.80, 113.90, 125.60, 125.50, 121.30, 121.30,
    113.50, 113.10, 110.80, 106.50, 108.80, 105.30, 104.40, 100.00, 96.00,
    95.10, 89.10, 90.50, 90.30, 88.40, 84.00, 85.10, 81.90, 82.60, 84.90,
    81.30, 71.90, 74.30, 76.40, 63.30, 71.70, 77.00, 65.20, 47.70, 68.60,
    65.00, 66.00, 61.00, 53.30, 58.90, 61.90])

_S1 = np.array([
    0.02, 4.50, 22.40, 42.00, 40.60, 41.60, 38.00, 42.40, 38.50, 35.00,
    43.40, 46.30, 43.90, 37.10, 36.70, 35.90, 32.60, 27.90, 24.30, 20.10,
    16.20, 13.20, 8.60, 6.10, 4.20, 1.90, 0.00, -1.60, -3.50, -3.50, -5.80,
    -7.20, -8.60, -9.50, -10.90, -10.70, -12.00, -14.00, -13.60, -12.00,
    -13.30, -12.90, -10.60, -11.60, -12.20, -10.20, -7.80, -11.20, -10.40,
    -10.60, -9.70, -8.30, -9.30, -9.80])

_S2 = np.array([
    0.00, 2.00, 4.00, 8.50, 7.80, 6.70, 5.30, 6.10, 3.00, 1.20, -1.10,
    -0.50, -0.70, -1.20, -2.60, -2.90, -2.80, -2.60, -2.60, -1.80, -1.50,
    -1.30, -1.20, -1.00, -0.50, -0.30, 0.00, 0.20, 0.50, 2.10, 3.20, 4.10,
    4.70, 5.10, 6.70, 7.30, 8.60, 9.80, 10.20, 8.30, 9.60, 8.50, 7.00, 7.60,
    8.00, 6.70, 5.20, 7.40, 6.80, 7.00, 6.40, 5.50, 6.10, 6.50])

S0 = np.interp(WL, _D_WL, _S0)
S1 = np.interp(WL, _D_WL, _S1)
S2 = np.interp(WL, _D_WL, _S2)


def load_official_daylight_components(wl, s0, s1, s2):
    global S0, S1, S2
    S0 = np.interp(WL, wl, s0)
    S1 = np.interp(WL, wl, s1)
    S2 = np.interp(WL, wl, s2)
    _STATE["official_daylight_components"] = True


def daylight_spd(cct: float) -> np.ndarray:
    """
    Relative SPD of a CIE D-illuminant of correlated colour temperature `cct`.

    Valid 4000-25000 K.  Returns a relative spectrum (S0 basis, ~100 at 560 nm).
    """
    T = float(cct)
    if T < 4000.0 or T > 25000.0:
        raise ValueError("CIE daylight reconstruction is defined for 4000-25000 K")
    if T <= 7000.0:
        xd = (-4.6070e9 / T**3 + 2.9678e6 / T**2 + 0.09911e3 / T + 0.244063)
    else:
        xd = (-2.0064e9 / T**3 + 1.9018e6 / T**2 + 0.24748e3 / T + 0.237040)
    yd = -3.000 * xd**2 + 2.870 * xd - 0.275
    den = 0.0241 + 0.2562 * xd - 0.7341 * yd
    m1 = (-1.3515 - 1.7703 * xd + 5.9114 * yd) / den
    m2 = (0.0300 - 31.4424 * xd + 30.0717 * yd) / den
    return S0 + m1 * S1 + m2 * S2


#: CCT of CIE standard illuminant D65 under the 1.4388 radiation constant.
D65_CCT = 6500.0 * 1.4388 / 1.4380


def d65() -> np.ndarray:
    """Relative SPD of D65."""
    return daylight_spd(D65_CCT)


# --------------------------------------------------------------------------
# Extraterrestrial solar spectrum
# --------------------------------------------------------------------------

def _planck(wl_nm, T):
    h, c, kb = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    lam = wl_nm * 1e-9
    return (2 * h * c**2 / lam**5) / (np.exp(h * c / (lam * kb * T)) - 1.0)


def _build_e0():
    """
    Extraterrestrial spectral irradiance, W m-2 nm-1.

    A 5778 K Planck curve scaled so that the 380-780 nm band carries
    535 W m-2 (the visible fraction of the 1361 W m-2 solar constant).
    Smooth: no Fraunhofer structure.  Load ASTM E490 for published work.
    """
    p = _planck(WL, 5778.0)
    p = p / np.trapezoid(p, WL)
    return p * 535.0


E0_SOLAR = _build_e0()


def load_official_solar_spectrum(wl, e0):
    """Replace the Planck stand-in with a referenced solar spectrum.

    ``wl`` is in nm and ``e0`` must be in W m-2 nm-1.
    """
    global E0_SOLAR
    wl = np.asarray(wl, float)
    e0 = np.asarray(e0, float)
    if wl.ndim != 1 or e0.shape != wl.shape:
        raise ValueError("wl and e0 must be equal-length 1-D arrays")
    if not np.all(np.isfinite(wl)) or not np.all(np.diff(wl) > 0):
        raise ValueError("wavelengths must be finite and strictly increasing")
    if not np.all(np.isfinite(e0)) or np.any(e0 < 0):
        raise ValueError("solar spectral irradiance must be finite and non-negative")
    if wl.min() > WL.min() or wl.max() < WL.max():
        raise ValueError("solar spectrum must cover the 380-780 nm model grid")
    E0_SOLAR = np.interp(WL, wl, e0)
    _STATE["official_solar_spectrum"] = True


# --------------------------------------------------------------------------
# Surface spectral reflectances
# --------------------------------------------------------------------------

def _from_points(points, clip=(0.0, 1.0)):
    """Smooth spectrum from (wavelength, value) control points."""
    wl = np.array([p[0] for p in points], float)
    v = np.array([p[1] for p in points], float)
    out = np.interp(WL, wl, v)
    # light smoothing so derived DERs are not sensitive to knot placement
    k = np.array([0.25, 0.5, 0.25])
    out = np.convolve(np.pad(out, 1, mode="edge"), k, mode="valid")
    return np.clip(out, *clip)


#: Spectral reflectance library for tropical-urban surfaces.
MATERIALS = {
    # Bangladeshi street materials.  Blue-end reflectance is what matters
    # melanopically, and it is exactly where warm masonry is weakest.
    "unrendered_brick":  _from_points([(380, .09), (450, .10), (500, .12),
                                       (550, .16), (600, .24), (650, .32),
                                       (700, .36), (780, .38)]),
    "whitewash":         _from_points([(380, .72), (450, .78), (550, .80),
                                       (650, .79), (780, .77)]),
    "grey_concrete":     _from_points([(380, .26), (450, .29), (550, .32),
                                       (650, .33), (780, .34)]),
    "cement_plaster":    _from_points([(380, .38), (450, .43), (550, .48),
                                       (650, .50), (780, .51)]),
    "corrugated_steel":  _from_points([(380, .42), (450, .46), (550, .50),
                                       (650, .50), (780, .49)]),
    "blue_painted":      _from_points([(380, .30), (450, .48), (500, .40),
                                       (550, .18), (620, .08), (780, .07)]),
    "ochre_painted":     _from_points([(380, .07), (450, .09), (500, .18),
                                       (550, .35), (600, .52), (700, .58),
                                       (780, .58)]),
    "asphalt":           _from_points([(380, .06), (550, .07), (780, .09)]),
    "concrete_pavement": _from_points([(380, .18), (550, .22), (780, .24)]),
    "brick_pavement":    _from_points([(380, .08), (550, .13), (650, .22),
                                       (780, .25)]),
    "water":             _from_points([(380, .06), (550, .05), (780, .04)]),
    "green_foliage":     _from_points([(380, .03), (450, .04), (500, .07),
                                       (550, .16), (600, .07), (660, .04),
                                       (700, .22), (780, .45)]),
    "glass_curtainwall": _from_points([(380, .12), (480, .15), (550, .14),
                                       (650, .12), (780, .11)]),
}


#: Spectral transmittance library for enclosures and canopies.
GLAZING = {
    "none":            np.ones_like(WL),
    "clear_float":     _from_points([(380, .82), (450, .88), (550, .89),
                                     (650, .88), (780, .86)]),
    "green_tinted":    _from_points([(380, .30), (450, .55), (520, .72),
                                     (580, .55), (650, .35), (780, .22)]),
    # Bronze glass plus the low-transmittance aftermarket film that is
    # ubiquitous on private cars in Dhaka.  Both cut the blue end hardest.
    "bronze_tinted":   _from_points([(380, .04), (450, .10), (500, .19),
                                     (550, .32), (620, .46), (700, .55),
                                     (780, .58)]),
    "aftermarket_film": _from_points([(380, .01), (450, .03), (500, .05),
                                      (550, .08), (650, .12), (780, .18)]),
    "laminated_ws":    _from_points([(380, .35), (450, .70), (550, .78),
                                     (650, .77), (780, .72)]),
    # Foliage transmittance: strong blue and red absorption, green leak.
    "canopy_leaf":     _from_points([(380, .02), (450, .02), (550, .10),
                                     (620, .03), (700, .18), (780, .40)]),
    "mesh_60":         np.full_like(WL, 0.60),
    "opaque":          np.zeros_like(WL),
}


def material(name: str) -> np.ndarray:
    if name not in MATERIALS:
        raise KeyError(f"unknown material {name!r}; have {sorted(MATERIALS)}")
    return MATERIALS[name]


def glazing(name: str) -> np.ndarray:
    if name not in GLAZING:
        raise KeyError(f"unknown glazing {name!r}; have {sorted(GLAZING)}")
    return GLAZING[name]
