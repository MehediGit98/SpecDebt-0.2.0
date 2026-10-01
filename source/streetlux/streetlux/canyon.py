"""
Street canyon: geometry, spectral inter-reflection, and the eye integral.

The quantity this module exists to produce is the **spectral irradiance on a
vertical plane at eye height, in a stated heading**.  That is the plane the
melanopic recommendations are written on, and it is not obtainable from a
horizontal work-plane calculation.

Geometry is a laterally infinite canyon: two parallel façades of independent
height, a ground plane, and an optional canopy slab.  Coordinates are world
(x east, y north, z up).  Street azimuth is the direction of travel measured
from north, clockwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import spectra
from .sky import SkyDome, fibonacci_sphere
from .sun import DEG, SpectralSky

SKY, WALL_L, WALL_R, GROUND = 0, 1, 2, 3


@dataclass
class Canopy:
    """Street tree layer, modelled as a horizontal slab of foliage."""
    cover: float = 0.0               # 0..1 projected cover over the street
    base_m: float = 3.5
    top_m: float = 8.0
    leaf: str = "green_foliage"       # reflectance, from spectra.MATERIALS
    transmittance: str = "canopy_leaf"  # from spectra.GLAZING; "opaque" for a soffit

    @property
    def present(self) -> bool:
        return self.cover > 1e-3


@dataclass
class CanyonSection:
    """One homogeneous stretch of street."""
    name: str
    length_m: float
    azimuth_deg: float               # direction of travel, from north, clockwise
    width_m: float = 12.0
    left_height_m: float = 12.0
    right_height_m: float = 12.0
    left_material: str = "unrendered_brick"
    right_material: str = "unrendered_brick"
    ground_material: str = "asphalt"
    eye_offset_m: float = 0.0        # + toward the right of travel (footpath side)
    eye_height_m: float = 1.55
    canopy: Canopy = field(default_factory=Canopy)

    def validate(self):
        vals = [self.width_m, self.length_m, self.left_height_m,
                self.right_height_m, self.eye_offset_m, self.eye_height_m]
        if not np.all(np.isfinite(vals)):
            raise ValueError("Canyon dimensions must be finite")
        if self.width_m <= 0 or self.length_m <= 0 or self.eye_height_m <= 0:
            raise ValueError("Width, length and eye height must be positive")
        if min(self.left_height_m, self.right_height_m) < 0:
            raise ValueError("Building heights cannot be negative")
        if abs(self.eye_offset_m) >= self.width_m / 2:
            raise ValueError("Eye position must lie strictly inside the canyon")
        c = self.canopy
        if not (0 <= c.cover <= 1 and 0 <= c.base_m <= c.top_m):
            raise ValueError("Invalid canopy cover or height")

    @property
    def aspect_ratio(self) -> float:
        return 0.5 * (self.left_height_m + self.right_height_m) / self.width_m


# --------------------------------------------------------------------------
# Ray classification
# --------------------------------------------------------------------------

def _frame(azimuth_deg: float):
    t = azimuth_deg * DEG
    fwd = np.array([math.sin(t), math.cos(t), 0.0])
    right = np.array([math.cos(t), -math.sin(t), 0.0])
    return fwd, right


def _trace(sec: CanyonSection, origin_perp: float, origin_z: float,
           dp: np.ndarray, dz: np.ndarray):
    """
    Classify directions from a point inside the canyon.

    `dp` is the component along the street-normal (positive to the right),
    `dz` the vertical component.  Returns (code, hit_z).
    """
    d_r = sec.width_m / 2.0 - origin_perp
    d_l = sec.width_m / 2.0 + origin_perp
    inf = np.inf

    t_r = np.where(dp > 1e-9, d_r / np.where(dp > 1e-9, dp, 1.0), inf)
    t_l = np.where(dp < -1e-9, d_l / np.where(dp < -1e-9, -dp, 1.0), inf)
    t_wall = np.minimum(t_r, t_l)
    side = np.where(t_r <= t_l, WALL_R, WALL_L)
    h_wall = np.where(side == WALL_R, sec.right_height_m, sec.left_height_m)
    z_wall = origin_z + dz * np.where(np.isfinite(t_wall), t_wall, 0.0)

    t_g = np.where(dz < -1e-9, origin_z / np.where(dz < -1e-9, -dz, 1.0), inf)

    code = np.full(dp.shape, SKY, dtype=np.int8)
    hit_z = np.zeros(dp.shape)

    wall_hit = np.isfinite(t_wall) & (z_wall >= 0.0) & (z_wall <= h_wall) \
        & (t_wall <= t_g)
    code[wall_hit] = side[wall_hit]
    hit_z[wall_hit] = z_wall[wall_hit]

    ground_hit = np.isfinite(t_g) & (t_g < t_wall)
    code[ground_hit] = GROUND
    return code, hit_z


def _canopy_interception(sec: CanyonSection, dz: np.ndarray) -> np.ndarray:
    """Fraction of an upward ray intercepted by the canopy slab."""
    c = sec.canopy
    if not c.present or sec.eye_height_m >= c.top_m:
        return np.zeros(dz.shape)
    sin_alt = np.clip(dz, 0.05, 1.0)
    p = 1.0 - (1.0 - c.cover) ** (1.0 / sin_alt)
    return np.where(dz > 0.0, np.clip(p, 0.0, 1.0), 0.0)


# --------------------------------------------------------------------------
# Radiosity between the three canyon surfaces
# --------------------------------------------------------------------------

def _hemisphere(normal, dirs):
    c = dirs @ normal
    return c > 1e-6, c


def _view_factors(sec: CanyonSection, dirs, w):
    """View factors from each canyon patch to (sky, left, right, ground)."""
    _, right = _frame(sec.azimuth_deg)
    dp = dirs @ right
    dz = dirs[:, 2]

    patches = {
        GROUND: (0.0, 0.0, np.array([0.0, 0.0, 1.0])),
        WALL_L: (-sec.width_m / 2.0 + 1e-3, sec.left_height_m / 2.0, right),
        WALL_R: (sec.width_m / 2.0 - 1e-3, sec.right_height_m / 2.0, -right),
    }
    out = {}
    for pid, (perp, z, n) in patches.items():
        cn = np.where(pid == GROUND, dz, dp * (1.0 if pid == WALL_L else -1.0))
        mask = cn > 1e-6
        code, _ = _trace(sec, perp, z, dp, dz)
        f = {}
        for target in (SKY, WALL_L, WALL_R, GROUND):
            sel = mask & (code == target)
            f[target] = float(np.sum(cn[sel]) * w / np.pi)
        # normalise away discretisation error
        tot = sum(f.values())
        if tot > 0:
            for k in f:
                f[k] /= tot
        out[pid] = (f, perp, z, n, mask, cn)
    return out


def solve_radiosity(sec: CanyonSection, dome: SkyDome, sky: SpectralSky,
                    dirs: np.ndarray, w: float) -> dict:
    """
    Spectral radiosity B (W m-2 nm-1) of the left wall, right wall and ground.

    Two-bounce-exact: a 3x3 linear system per wavelength, so inter-reflection
    between opposing façades is resolved rather than approximated.
    """
    nwl = len(spectra.WL)
    vf = _view_factors(sec, dirs, w)
    rho = {
        WALL_L: spectra.material(sec.left_material),
        WALL_R: spectra.material(sec.right_material),
        GROUND: spectra.material(sec.ground_material),
    }
    order = [WALL_L, WALL_R, GROUND]

    # direct (sky + beam) irradiance on each patch
    e_direct = {}
    sv = dome.sun_dir
    for pid in order:
        f, perp, z, n, mask, cn = vf[pid]
        code, _ = _trace(sec, perp, z, dirs @ _frame(sec.azimuth_deg)[1], dirs[:, 2])
        sel = mask & (code == SKY)
        e = np.einsum("i,ij->j", cn[sel] * w, dome.radiance[sel])
        if sky.sun.up:
            cs = float(sv @ n)
            if cs > 0:
                dp_s = float(sv @ _frame(sec.azimuth_deg)[1])
                cd, _ = _trace(sec, perp, z,
                               np.array([dp_s]), np.array([float(sv[2])]))
                if cd[0] == SKY:
                    trans = 1.0
                    if sec.canopy.present:
                        p = _canopy_interception(sec, np.array([float(sv[2])]))[0]
                        trans = ((1.0 - p)
                                 + p * spectra.glazing(sec.canopy.transmittance))
                    e = e + sky.beam_normal * cs * trans
        e_direct[pid] = e

    # B_i = rho_i (E_i + sum_j F_ij B_j)
    A = np.zeros((3, 3, nwl))
    rhs = np.zeros((3, nwl))
    for i, pid in enumerate(order):
        f = vf[pid][0]
        r = rho[pid]
        for j, qid in enumerate(order):
            A[i, j] = -r * f[qid]
        A[i, i] += 1.0
        rhs[i] = r * e_direct[pid]
    B = np.linalg.solve(np.moveaxis(A, 2, 0), np.moveaxis(rhs, 1, 0)[:, :, None])
    B = np.moveaxis(B[:, :, 0], 0, 1)
    return {pid: B[i] for i, pid in enumerate(order)}


# --------------------------------------------------------------------------
# Eye radiance field
# --------------------------------------------------------------------------

class EyeField:
    """
    Spectral radiance arriving at the eye point from every sampled direction,
    before any enclosure is applied.
    """

    def __init__(self, sec: CanyonSection, dome: SkyDome, sky: SpectralSky,
                 dirs: np.ndarray, w: float):
        sec.validate()
        self.section = sec
        self.directions = dirs
        self.solid_angle = w
        self.sky = sky
        _, right = _frame(sec.azimuth_deg)
        dp = dirs @ right
        dz = dirs[:, 2]
        code, _ = _trace(sec, sec.eye_offset_m, sec.eye_height_m, dp, dz)
        self.code = code

        B = solve_radiosity(sec, dome, sky, dirs, w)
        L = np.zeros((len(dirs), len(spectra.WL)))
        L[code == SKY] = dome.radiance[code == SKY]

        if sec.canopy.present:
            p = _canopy_interception(sec, dz)[:, None]
            tau = spectra.glazing(sec.canopy.transmittance)[None, :]
            e_top = sky.global_horizontal[None, :]
            l_leaf = 0.5 * spectra.material(sec.canopy.leaf)[None, :] * e_top / np.pi
            m = (code == SKY)
            L[m] = (1.0 - p[m]) * L[m] + p[m] * (tau * L[m] + l_leaf)

        for pid in (WALL_L, WALL_R, GROUND):
            L[code == pid] = B[pid] / np.pi
        self.radiance = L

        # unoccluded direct beam
        self.beam_dir = dome.sun_dir
        self.beam = np.zeros(len(spectra.WL))
        if sky.sun.up:
            cd, _ = _trace(sec, sec.eye_offset_m, sec.eye_height_m,
                           np.array([float(dome.sun_dir @ right)]),
                           np.array([float(dome.sun_dir[2])]))
            if cd[0] == SKY:
                trans = np.ones(len(spectra.WL))
                if sec.canopy.present:
                    pp = _canopy_interception(
                        sec, np.array([float(dome.sun_dir[2])]))[0]
                    trans = ((1.0 - pp)
                             + pp * spectra.glazing(sec.canopy.transmittance))
                self.beam = sky.beam_normal * trans
        self.sky_view_fraction = float(np.mean(code == SKY))

    def vertical_irradiance(self, headings_deg: np.ndarray,
                            tau: np.ndarray | None = None,
                            beam_tau: np.ndarray | None = None) -> np.ndarray:
        """
        Spectral irradiance on a vertical plane at eye height for each heading.

        Parameters
        ----------
        headings_deg : (H,) azimuths the observer faces, from north, clockwise
        tau : (D, W) per-direction spectral transmittance of the enclosure
        beam_tau : (W,) enclosure transmittance along the sun direction

        Returns (H, W) array in W m-2 nm-1.
        """
        t = headings_deg * DEG
        n = np.stack([np.sin(t), np.cos(t), np.zeros_like(t)], axis=1)  # (H,3)
        proj = np.clip(self.directions @ n.T, 0.0, None) * self.solid_angle  # (D,H)
        L = self.radiance if tau is None else self.radiance * tau
        E = proj.T @ L                                                   # (H,W)
        if self.beam.any():
            cb = np.clip(n @ self.beam_dir, 0.0, None)                   # (H,)
            b = self.beam if beam_tau is None else self.beam * beam_tau
            E = E + cb[:, None] * b[None, :]
        return E
