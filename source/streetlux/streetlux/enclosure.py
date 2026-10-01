"""
Travel-mode enclosures.

An enclosure is a directional spectral filter in the traveller's own frame:
forward is the direction of travel, and each sampled direction is classified
into an aperture (windscreen, side glazing, open, roof) or into opaque body.

This is where the equity result lives.  A pedestrian carries no filter.  A
private car carries bronze glass plus aftermarket film, whose transmittance
is lowest exactly where melanopsin is most sensitive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import spectra
from .sun import DEG


@dataclass(frozen=True)
class Aperture:
    """A window or opening, defined in the traveller's frame."""
    name: str
    glazing: str
    az_min: float          # relative azimuth, deg, 0 = ahead, +ve to the right
    az_max: float
    el_min: float          # elevation, deg
    el_max: float


@dataclass(frozen=True)
class Mode:
    """A way of getting to work."""
    key: str
    label: str
    speed_kmh: float
    eye_height_m: float
    apertures: tuple[Aperture, ...] = field(default_factory=tuple)
    open_air: bool = False
    body_transmittance: float = 0.0     # leakage through opaque body

    def tau(self, dirs: np.ndarray, heading_deg: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Per-direction spectral transmittance.

        Returns (tau_field, ) shaped (D, W).  Directions are world-frame; the
        traveller's frame is recovered from `heading_deg`.
        """
        nwl = len(spectra.WL)
        if self.open_air:
            return np.ones((len(dirs), nwl))

        t = heading_deg * DEG
        fwd = np.array([math.sin(t), math.cos(t), 0.0])
        right = np.array([math.cos(t), -math.sin(t), 0.0])

        f = dirs @ fwd
        r = dirs @ right
        z = dirs[:, 2]
        az = np.degrees(np.arctan2(r, f))                    # -180..180
        el = np.degrees(np.arcsin(np.clip(z, -1, 1)))

        tau = np.full((len(dirs), nwl), self.body_transmittance)
        for ap in self.apertures:
            g = spectra.glazing(ap.glazing)
            if ap.az_min <= ap.az_max:
                in_az = (az >= ap.az_min) & (az <= ap.az_max)
            else:                                            # wraps through 180
                in_az = (az >= ap.az_min) | (az <= ap.az_max)
            m = in_az & (el >= ap.el_min) & (el <= ap.el_max)
            tau[m] = np.maximum(tau[m], g[None, :])
        return tau

    def beam_tau(self, sun_dir: np.ndarray, heading_deg: float) -> np.ndarray:
        return self.tau(sun_dir[None, :], heading_deg)[0]


def _side_windows(glz, el_lo=-25.0, el_hi=35.0):
    return (Aperture("right_side", glz, 40.0, 140.0, el_lo, el_hi),
            Aperture("left_side", glz, -140.0, -40.0, el_lo, el_hi))


#: Travel modes as actually used in Dhaka.
MODES: dict[str, Mode] = {
    "walk": Mode("walk", "On foot", 4.5, 1.55, open_air=True),

    "rickshaw": Mode(
        "rickshaw", "Cycle rickshaw (hood down)", 9.0, 1.35,
        apertures=(Aperture("front", "none", -110.0, 110.0, -90.0, 60.0),
                   *_side_windows("none", -90.0, 60.0)),
        body_transmittance=0.0),

    "rickshaw_hood": Mode(
        "rickshaw_hood", "Cycle rickshaw (hood up)", 9.0, 1.35,
        apertures=(Aperture("front", "none", -80.0, 80.0, -90.0, 25.0),),
        body_transmittance=0.0),

    "cng": Mode(
        "cng", "CNG auto-rickshaw (caged)", 16.0, 1.25,
        apertures=(Aperture("front", "none", -55.0, 55.0, -30.0, 20.0),
                   *_side_windows("mesh_60", -30.0, 25.0)),
        body_transmittance=0.0),

    "bus": Mode(
        "bus", "City bus, window seat", 14.0, 1.60,
        apertures=(*_side_windows("green_tinted", -20.0, 30.0),),
        body_transmittance=0.0),

    "car_clear": Mode(
        "car_clear", "Private car, untinted", 20.0, 1.20,
        apertures=(Aperture("windscreen", "laminated_ws", -45.0, 45.0, -15.0, 35.0),
                   *_side_windows("clear_float"),
                   Aperture("rear", "clear_float", 150.0, -150.0, -10.0, 25.0)),
        body_transmittance=0.0),

    "car_tinted": Mode(
        "car_tinted", "Private car, bronze glazing", 20.0, 1.20,
        apertures=(Aperture("windscreen", "laminated_ws", -45.0, 45.0, -15.0, 35.0),
                   *_side_windows("bronze_tinted"),
                   Aperture("rear", "bronze_tinted", 150.0, -150.0, -10.0, 25.0)),
        body_transmittance=0.0),

    "car_film": Mode(
        "car_film", "Private car, aftermarket film", 20.0, 1.20,
        apertures=(Aperture("windscreen", "bronze_tinted", -45.0, 45.0, -15.0, 35.0),
                   *_side_windows("aftermarket_film"),
                   Aperture("rear", "aftermarket_film", 150.0, -150.0, -10.0, 25.0)),
        body_transmittance=0.0),
}


def mode(key: str) -> Mode:
    if key not in MODES:
        raise KeyError(f"unknown mode {key!r}; have {sorted(MODES)}")
    return MODES[key]
