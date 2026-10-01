"""
Angular distribution of sky radiance.

Three diffuse components are carried separately, each with its own angular
distribution and its own spectrum:

  Rayleigh   (1 + cos^2 gamma) single-scattering phase, blue, high DER
  aerosol    Henyey-Greenstein forward peak, sun-coloured, low DER
  cloud      Moon-Spencer (1 + 2 cos Z) / 3, near-neutral

Each is normalised so that its integral of L cos(Z) over the upper hemisphere
reproduces the horizontal irradiance the atmospheric model produced for it.
"""

from __future__ import annotations

import math

import numpy as np

from .sun import SpectralSky, DEG


def fibonacci_sphere(n: int) -> np.ndarray:
    """`n` roughly equal-area directions on the unit sphere, (n, 3) x=E y=N z=up."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.sin(phi) * np.cos(theta),
                     np.sin(phi) * np.sin(theta),
                     np.cos(phi)], axis=1)


def sun_vector(alt_deg: float, az_deg: float) -> np.ndarray:
    """Unit vector to the sun.  Azimuth measured from north, clockwise."""
    a, z = alt_deg * DEG, az_deg * DEG
    return np.array([math.cos(a) * math.sin(z),
                     math.cos(a) * math.cos(z),
                     math.sin(a)])


def _hg(cos_gamma, g):
    return (1.0 - g * g) / (1.0 + g * g - 2.0 * g * cos_gamma) ** 1.5


def _normalise(shape, dirs, w):
    """Scale `shape` so that sum(shape * cosZ * w) over the upper dome == 1."""
    cz = np.clip(dirs[:, 2], 0.0, None)
    s = float(np.sum(shape * cz * w))
    return shape / s if s > 0 else shape


class SkyDome:
    """
    Discretised spectral radiance of the open sky, W m-2 sr-1 nm-1.

    ``radiance[i]`` is the spectrum arriving from ``directions[i]``.  Only the
    upper hemisphere is populated; ground-directed entries are zero and are
    supplied by the canyon model instead.
    """

    def __init__(self, sky: SpectralSky, directions: np.ndarray):
        self.sky = sky
        self.directions = directions
        n = len(directions)
        w = 4.0 * np.pi / n                      # solid angle per sample
        self.solid_angle = w
        up = directions[:, 2] > 0.0

        if not sky.sun.up:
            self.radiance = np.zeros((n, len(sky.diffuse_rayleigh)))
            self.sun_dir = np.array([0.0, 0.0, -1.0])
            return

        sv = sun_vector(sky.sun.altitude_deg, sky.sun.azimuth_deg)
        self.sun_dir = sv
        cg = directions @ sv                      # cos of scattering angle

        cz = np.clip(directions[:, 2], 1e-3, None)
        path = (1.0 - np.exp(-0.32 / cz)) / (1.0 - np.exp(-0.32))

        s_ray = np.where(up, (1.0 + cg ** 2) * path, 0.0)
        s_aer = np.where(up, _hg(cg, 0.65) * path, 0.0)
        s_cld = np.where(up, (1.0 + 2.0 * np.clip(directions[:, 2], 0, 1)) / 3.0, 0.0)

        s_ray = _normalise(s_ray, directions, w)
        s_aer = _normalise(s_aer, directions, w)
        s_cld = _normalise(s_cld, directions, w)

        self.radiance = (s_ray[:, None] * sky.diffuse_rayleigh[None, :]
                         + s_aer[:, None] * sky.diffuse_aerosol[None, :]
                         + s_cld[:, None] * sky.diffuse_cloud[None, :])

    def horizontal_diffuse(self) -> np.ndarray:
        cz = np.clip(self.directions[:, 2], 0.0, None)
        return np.einsum("i,ij->j", cz * self.solid_angle, self.radiance)
