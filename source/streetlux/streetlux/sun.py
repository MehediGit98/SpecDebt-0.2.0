"""
Solar geometry and clear-sky spectral irradiance.

The atmospheric model is a reduced Bird & Riordan SPCTRAL2: direct beam and
two diffuse components (Rayleigh, aerosol) with distinct spectra.  Keeping
the two diffuse components separate is not decoration — the Rayleigh
component is the blue, high-DER part of the sky and the aerosol component is
the sun-coloured, low-DER part concentrated near the solar disc.  Collapsing
them into one "diffuse" number is what makes conventional daylight tools
unable to answer melanopic questions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import spectra

DEG = math.pi / 180.0


# --------------------------------------------------------------------------
# Solar position (NOAA algorithm)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SunPos:
    altitude_deg: float
    azimuth_deg: float          # from north, clockwise
    zenith_deg: float
    air_mass: float
    up: bool


def solar_position(lat_deg: float, lon_deg: float, tz_hours: float,
                   year: int, month: int, day: int,
                   hour: float) -> SunPos:
    """Solar altitude and azimuth for a local clock time (decimal hours)."""
    if month <= 2:
        year -= 1
        month += 12
    a = year // 100
    b = 2 - a + a // 4
    jd = (math.floor(365.25 * (year + 4716))
          + math.floor(30.6001 * (month + 1))
          + day + b - 1524.5
          + (hour - tz_hours) / 24.0)
    t = (jd - 2451545.0) / 36525.0

    l0 = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360.0
    m = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    c = (math.sin(m * DEG) * (1.914602 - t * (0.004817 + 0.000014 * t))
         + math.sin(2 * m * DEG) * (0.019993 - 0.000101 * t)
         + math.sin(3 * m * DEG) * 0.000289)
    true_long = l0 + c
    omega = 125.04 - 1934.136 * t
    app_long = true_long - 0.00569 - 0.00478 * math.sin(omega * DEG)
    eps0 = (23.0 + (26.0 + ((21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))))
                    / 60.0) / 60.0)
    eps = eps0 + 0.00256 * math.cos(omega * DEG)
    decl = math.asin(math.sin(eps * DEG) * math.sin(app_long * DEG)) / DEG

    y = math.tan(eps * DEG / 2.0) ** 2
    eot = 4.0 / DEG * (y * math.sin(2 * l0 * DEG)
                       - 2 * e * math.sin(m * DEG)
                       + 4 * e * y * math.sin(m * DEG) * math.cos(2 * l0 * DEG)
                       - 0.5 * y * y * math.sin(4 * l0 * DEG)
                       - 1.25 * e * e * math.sin(2 * m * DEG))
    tst = (hour * 60.0 + eot + 4.0 * lon_deg - 60.0 * tz_hours) % 1440.0
    ha = tst / 4.0 - 180.0

    phi, dec = lat_deg * DEG, decl * DEG
    cz = (math.sin(phi) * math.sin(dec)
          + math.cos(phi) * math.cos(dec) * math.cos(ha * DEG))
    cz = max(-1.0, min(1.0, cz))
    zen = math.acos(cz) / DEG
    alt = 90.0 - zen

    den = math.cos(phi) * math.sin(zen * DEG)
    if abs(den) < 1e-9:
        az = 180.0
    else:
        ca = (math.sin(phi) * cz - math.sin(dec)) / den
        ca = max(-1.0, min(1.0, ca))
        az = math.acos(ca) / DEG
        if ha > 0:
            az = 360.0 - az
    # refraction-corrected apparent altitude, coarse
    if alt > -0.833:
        alt_r = alt + (0.0167 / math.tan((alt + 3.59 / (alt + 5.11)) * DEG)
                       if alt < 15 else 0.0)
    else:
        alt_r = alt
    if alt_r > 0:
        m_air = 1.0 / (math.sin(alt_r * DEG)
                       + 0.50572 * (alt_r + 6.07995) ** -1.6364)
    else:
        m_air = 40.0
    return SunPos(alt_r, az % 360.0, 90.0 - alt_r, m_air, alt_r > 0.0)


# --------------------------------------------------------------------------
# Spectral atmospheric transmittance
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Atmosphere:
    """Site atmospheric state."""
    turbidity_beta: float = 0.30     # Angstrom beta.  Dhaka dry season is high.
    turbidity_alpha: float = 1.20    # Angstrom alpha
    ozone_cm: float = 0.28           # atm-cm
    water_cm: float = 4.0            # precipitable water, cm
    pressure_mb: float = 1010.0
    cloud_cover: float = 0.0         # 0 clear .. 1 fully overcast
    ground_albedo: float = 0.15

    @property
    def name(self) -> str:
        if self.cloud_cover < 0.15:
            return "clear"
        if self.cloud_cover < 0.75:
            return "partly_cloudy"
        return "overcast"


_UM = spectra.WL / 1000.0


def _t_rayleigh(m, p_mb):
    mp = m * p_mb / 1013.25
    return np.exp(-mp / (_UM ** 4 * (115.6406 - 1.335 / _UM ** 2)))


def _t_aerosol(m, beta, alpha):
    return np.exp(-beta * _UM ** (-alpha) * m)


def _ozone_abs():
    """Chappuis band absorption coefficient, coarse Gaussian fit (cm-1)."""
    return 0.055 * np.exp(-(((spectra.WL - 602.0) / 92.0) ** 2))


def _t_ozone(m_o, u_o):
    return np.exp(-_ozone_abs() * u_o * m_o)


def _t_water(m, w):
    """Weak visible-band water absorption; the 720 nm band edge only."""
    a = 0.12 * np.exp(-(((spectra.WL - 725.0) / 12.0) ** 2))
    return np.exp(-0.2385 * a * w * m / (1.0 + 20.07 * a * w * m) ** 0.45)


@dataclass(frozen=True)
class SpectralSky:
    """
    Spectral irradiance components reaching the top of the street canyon.

    All arrays are W m-2 nm-1 on :data:`spectra.WL`.
    """
    beam_normal: np.ndarray          # direct normal
    diffuse_rayleigh: np.ndarray     # horizontal irradiance from Rayleigh sky
    diffuse_aerosol: np.ndarray      # horizontal irradiance from aerosol sky
    diffuse_cloud: np.ndarray        # horizontal irradiance from cloud
    sun: SunPos
    atmosphere: Atmosphere

    @property
    def diffuse_total(self) -> np.ndarray:
        return self.diffuse_rayleigh + self.diffuse_aerosol + self.diffuse_cloud

    @property
    def global_horizontal(self) -> np.ndarray:
        cz = max(0.0, math.cos(self.sun.zenith_deg * DEG))
        return self.beam_normal * cz + self.diffuse_total


def spectral_sky(sun: SunPos, atm: Atmosphere) -> SpectralSky:
    """Bird-style clear-sky spectral decomposition, degraded by cloud cover."""
    zero = np.zeros_like(spectra.WL)
    if not sun.up:
        return SpectralSky(zero, zero, zero, zero, sun, atm)

    m = sun.air_mass
    m_o = 1.003454 / math.sqrt(math.cos(sun.zenith_deg * DEG) ** 2 + 0.006908)
    tr = _t_rayleigh(m, atm.pressure_mb)
    ta = _t_aerosol(m, atm.turbidity_beta, atm.turbidity_alpha)
    to = _t_ozone(m_o, atm.ozone_cm)
    tw = _t_water(m, atm.water_cm)

    # single-scattering albedo split of aerosol extinction
    omega = 0.90
    ta_s = np.exp(-omega * atm.turbidity_beta * _UM ** (-atm.turbidity_alpha) * m)
    ta_a = ta / np.maximum(ta_s, 1e-12)

    beam = spectra.E0_SOLAR * tr * ta * to * tw
    cz = math.cos(sun.zenith_deg * DEG)

    # Bird diffuse components on a horizontal surface
    fs = 0.83                                   # forward-scatter fraction
    d_ray = spectra.E0_SOLAR * cz * to * tw * ta_a * (1.0 - tr ** 0.95) * 0.5
    d_aer = spectra.E0_SOLAR * cz * to * tw * (tr ** 1.5) * (1.0 - ta_s) * fs

    # Cloud: neutral scatterer.  Attenuate beam, convert the loss into a
    # diffuse component whose spectrum is the mixture that actually reaches
    # the cloud base.
    c = float(np.clip(atm.cloud_cover, 0.0, 1.0))
    beam_att = 1.0 - c ** 1.4
    lost = beam * cz * (1.0 - beam_att) + (d_ray + d_aer) * c * 0.15
    d_cloud = lost * 0.72                       # cloud albedo loss to space
    d_ray_o = d_ray * (1.0 - 0.55 * c)
    d_aer_o = d_aer * (1.0 - 0.85 * c)

    return SpectralSky(beam * beam_att, d_ray_o, d_aer_o, d_cloud, sun, atm)
