"""
The four domains the shading decision actually spans.

Visual (DGP), thermal (solar gain on the occupant and on the cooling plant),
energy (cooling plus electric lighting) and circadian (melanopic EDI at the
eye).  Shade controllers in the literature optimise one or two of these.  The
hypothesis under test is that doing so is circadianly harmful in a tropical
climate, and that the harm is invisible unless the fourth domain is in the
objective.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .optics import Solution

#: Thresholds used throughout.  Each is a published convention, not a tuning knob.
THRESHOLDS = {
    "dgp_imperceptible": 0.35,     # Wienold & Christoffersen
    "dgp_disturbing": 0.40,
    "medi_brown_daytime": 250.0,   # Brown et al. 2022, vertical, at the eye
    "medi_well_tier1": 136.0,      # WELL v2 L03 Tier 1
    "lux_workplane_target": 500.0,
    "solar_on_occupant_w_m2": 100.0,   # radiant discomfort trigger
}


def position_index(dirs: np.ndarray, view: np.ndarray) -> np.ndarray:
    """
    Guth position index P for each sampled direction.

    Uses the standard formulation for sources at or above the line of sight
    and mirrors it below, which is the usual engineering simplification.
    """
    v = view / np.linalg.norm(view)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(v, up)
    n = np.linalg.norm(right)
    right = right / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    up = np.cross(right, v)

    cos_s = np.clip(dirs @ v, -1.0, 1.0)
    sigma = np.degrees(np.arccos(cos_s))
    y = dirs @ up
    x = dirs @ right
    alpha = np.degrees(np.arctan2(np.abs(x), np.abs(y) + 1e-12))

    a, s = alpha, sigma
    ln_p = ((35.2 - 0.31889 * a - 1.22 * np.exp(-2.0 * a / 9.0)) * 1e-3 * s
            + (21.0 + 0.26667 * a - 0.002963 * a * a) * 1e-5 * s * s)
    return np.clip(np.exp(ln_p), 1.0, 40.0)


def dgp(sol: Solution, dirs: np.ndarray, solid_angle: float,
        view: np.ndarray, source_factor: float = 5.0) -> tuple[float, int]:
    """
    Daylight Glare Probability (Wienold & Christoffersen 2006).

    Returns (DGP, number of glare source directions).  The source count is
    returned so that a DGP with no sources is never mistaken for a DGP whose
    source-detection branch failed to run.
    """
    ev = sol.lux_vertical
    if ev <= 1.0:
        return 0.0, 0
    lum = sol.eye_luminance
    in_fov = (dirs @ (view / np.linalg.norm(view))) > 0.0
    l_avg = float(np.mean(lum[in_fov])) if in_fov.any() else 0.0
    src = in_fov & (lum > max(source_factor * l_avg, 500.0))
    n_src = int(src.sum())
    if n_src == 0 and not sol.point_sources:
        return float(np.clip(5.87e-5 * ev + 0.16, 0.0, 1.0)), 0
    term = 0.0
    if n_src:
        p = position_index(dirs[src], view)
        term += float(np.sum(lum[src] ** 2 * solid_angle
                             / (ev ** 1.87 * p ** 2)))
    for L, omega, d in sol.point_sources:
        if float(d @ (view / np.linalg.norm(view))) <= 0.0:
            continue
        pi_ = float(position_index(np.asarray(d)[None, :], view)[0])
        term += L ** 2 * omega / (ev ** 1.87 * pi_ ** 2)
        n_src += 1
    val = 5.87e-5 * ev + 0.0918 * math.log10(1.0 + term) + 0.16
    return float(np.clip(val, 0.0, 1.0)), n_src


@dataclass
class EnergyModel:
    """Crude but explicit plant model.  A proxy, labelled as one."""
    cop_cooling: float = 3.0
    fan_fraction: float = 0.12
    envelope_load_w_m2: float = 18.0     # everything that is not the window
    internal_gain_w_m2: float = 12.0

    def cooling_electric_w(self, sol: Solution, floor_area: float) -> float:
        gain = (sol.transmitted_solar_w
                + sol.lighting_w
                + (self.envelope_load_w_m2 + self.internal_gain_w_m2) * floor_area)
        return gain / self.cop_cooling * (1.0 + self.fan_fraction)

    def total_electric_w(self, sol: Solution, floor_area: float) -> float:
        return self.cooling_electric_w(sol, floor_area) + sol.lighting_w


def solar_on_occupant(sol: Solution) -> float:
    """
    Solar irradiance actually reaching the occupant, W/m2.

    Direct beam only counts when the sun patch geometrically reaches the eye.
    That distinction matters: radiant discomfort — and therefore blind
    closure — is driven by being *in* the sun patch, not by the window's
    aggregate heat gain.
    """
    return sol.beam_solar_on_occupant_w_m2 + sol.diffuse_solar_on_occupant_w_m2


@dataclass
class StepMetrics:
    t_hours: float
    occupied: bool
    coverage: float
    electric_fraction: float
    medi: float
    lux_vertical: float
    lux_workplane: float
    der: float
    dgp: float
    n_glare_sources: int
    solar_on_occupant_w_m2: float
    electric_w: float
    window_luminance: float


def evaluate(sol: Solution, dirs, solid_angle, view, room, energy: EnergyModel,
             t_hours: float, occupied: bool) -> StepMetrics:
    g, n = dgp(sol, dirs, solid_angle, view)
    return StepMetrics(
        t_hours=t_hours, occupied=occupied, coverage=sol.coverage,
        electric_fraction=sol.lighting_w / max(
            room.lighting_power_density * room.floor_area, 1e-9),
        medi=sol.medi, lux_vertical=sol.lux_vertical,
        lux_workplane=sol.lux_workplane, der=sol.der, dgp=g,
        n_glare_sources=n,
        solar_on_occupant_w_m2=solar_on_occupant(sol),
        electric_w=energy.total_electric_w(sol, room.floor_area),
        window_luminance=sol.window_luminance_mean)
