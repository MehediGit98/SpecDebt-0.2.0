"""
Room optics for the shading paradox testbed.

Deliberately built **on top of** ``streetlux`` rather than beside it.  The
action spectra, the D-series, the melanopic calibration, the solar position
and the spectral sky all come from that package.  There is one melanopic
constant in this codebase and one place it is derived.  Forking those modules
here would reproduce exactly the failure mode that put a sign-contradicting
result into a four-paper suite.

Geometry is a shoebox with one glazed façade.  Room-local axes: x across the
façade, y into the room from the façade, z up.  The façade's outward normal
azimuth is the room's orientation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from streetlux import photometry, spectra
from streetlux.sky import SkyDome, fibonacci_sphere
from streetlux.sun import DEG, SpectralSky

# surface codes
WINDOW, FACADE, BACK, LEFT, RIGHT, CEILING, FLOOR = range(7)
INTERIOR = (FACADE, BACK, LEFT, RIGHT, CEILING, FLOOR)
NAMES = {WINDOW: "window", FACADE: "facade_wall", BACK: "back_wall",
         LEFT: "left_wall", RIGHT: "right_wall", CEILING: "ceiling",
         FLOOR: "floor"}


# --------------------------------------------------------------------------
# Shading devices
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ShadeDevice:
    """
    A shading device, described by what it does to light rather than by brand.

    ``openness`` is the fraction of the covered aperture that still passes a
    specular view of the exterior; the remainder is passed diffusely, which is
    what kills glare far faster than it kills flux.  That asymmetry is the
    whole mechanism behind the paradox this package tests.
    """
    key: str
    label: str
    fabric: str                     # key in streetlux.spectra.GLAZING
    openness: float = 0.03          # specular openness factor
    diffuse_transmittance: float = 0.18
    kind: str = "roller"            # roller | venetian
    slat_cutoff_deg: float = 35.0   # venetian: profile angle fully cut off

    def covered_band(self, room, coverage: float) -> float:
        """
        Height below which the aperture is still clear.

        A roller descends from the head, so a partly lowered blind blocks the
        upper band — which is where the sun and the bright sky are.  Treating
        coverage as a uniform transmittance loses exactly this, and with it
        the reason a half-lowered blind kills glare while costing little flux.
        """
        c = float(np.clip(coverage, 0.0, 1.0))
        return room.window_head_m - c * (room.window_head_m - room.window_sill_m)

    def transmission(self, coverage: float, profile_angle_deg: float):
        """
        Split the window into (specular fraction, diffuse fraction, spectrum).

        Returns (f_spec, f_diff, tau_lambda) where the two fractions apply to
        the covered part of the aperture and ``tau_lambda`` is the fabric
        spectrum.  The uncovered part is always fully specular.
        """
        c = float(np.clip(coverage, 0.0, 1.0))
        tau = spectra.glazing(self.fabric)
        if self.kind == "venetian":
            cut = float(np.clip(
                1.0 - abs(profile_angle_deg) / max(self.slat_cutoff_deg, 1e-6),
                0.0, 1.0))
            spec = self.openness * (1.0 - cut)
        else:
            spec = self.openness
        f_spec = (1.0 - c) + c * spec
        f_diff = c * self.diffuse_transmittance
        return f_spec, f_diff, tau


SHADES = {
    "roller_dark": ShadeDevice("roller_dark", "Dark roller blind",
                               "bronze_tinted", openness=0.02,
                               diffuse_transmittance=0.09),
    "roller_light": ShadeDevice("roller_light", "Light roller blind",
                                "mesh_60", openness=0.05,
                                diffuse_transmittance=0.28),
    "venetian": ShadeDevice("venetian", "Venetian blind",
                            "mesh_60", openness=0.10,
                            diffuse_transmittance=0.20, kind="venetian"),
    "none": ShadeDevice("none", "No shading device", "none",
                        openness=1.0, diffuse_transmittance=0.0),
}


# --------------------------------------------------------------------------
# Room
# --------------------------------------------------------------------------

@dataclass
class Room:
    """A single-sided tropical office cell."""
    name: str = "reference cell"
    width_m: float = 6.0
    depth_m: float = 8.0
    height_m: float = 3.0
    orientation_deg: float = 180.0       # azimuth of the façade outward normal
    window_sill_m: float = 0.9
    window_head_m: float = 2.5
    window_width_frac: float = 0.75
    glazing: str = "clear_float"
    glazing_solar_to_visible: float = 1.28   # tau_solar / tau_visible
    visible_band_solar_fraction: float = 0.45  # share of solar energy 380-780 nm
    wall_material: str = "whitewash"
    ceiling_material: str = "whitewash"
    floor_material: str = "grey_concrete"
    obstruction_deg: float = 22.0            # elevation of the opposing block
    obstruction_material: str = "grey_concrete"
    ground_material: str = "asphalt"
    desk_x_m: float = 3.0
    desk_y_m: float = 4.0
    eye_height_m: float = 1.20
    view_heading_deg: float | None = None    # None -> 45 deg off the façade normal
    lighting_power_density: float = 7.0      # W/m2 installed
    lighting_efficacy: float = 110.0         # lm/W
    lighting_cct: float = 4000.0

    @property
    def floor_area(self) -> float:
        return self.width_m * self.depth_m

    @property
    def window_area(self) -> float:
        return ((self.window_head_m - self.window_sill_m)
                * self.width_m * self.window_width_frac)

    @property
    def wwr(self) -> float:
        return self.window_area / (self.width_m * self.height_m)

    def rotation(self) -> np.ndarray:
        """Room-local -> world."""
        t = self.orientation_deg * DEG
        n_out = np.array([math.sin(t), math.cos(t), 0.0])
        ex = np.array([-math.cos(t), math.sin(t), 0.0])
        ey = -n_out
        ez = np.array([0.0, 0.0, 1.0])
        return np.stack([ex, ey, ez], axis=1)

    def outward_normal(self) -> np.ndarray:
        t = self.orientation_deg * DEG
        return np.array([math.sin(t), math.cos(t), 0.0])

    def view_normal_local(self) -> np.ndarray:
        """
        Unit vector the occupant faces, in room-local coordinates.

        The default is 45 degrees off the façade normal — the most common
        perimeter-desk arrangement, and a deliberate choice rather than a
        neutral one.  It matters more than most inputs in this model, which is
        why experiment E3 sweeps it instead of trusting it.
        """
        if self.view_heading_deg is None:
            return np.array([0.70710678, -0.70710678, 0.0])
        t = (self.view_heading_deg - self.orientation_deg) * DEG
        return np.array([math.sin(t), -math.cos(t), 0.0])


# --------------------------------------------------------------------------
# Geometry: direction classification inside the box
# --------------------------------------------------------------------------

def _box_hit(room: Room, origin, dirs):
    """Slab traversal against the six room planes.  Returns (code, point)."""
    W, D, H = room.width_m, room.depth_m, room.height_m
    o = np.asarray(origin, float)
    t_best = np.full(len(dirs), np.inf)
    code = np.full(len(dirs), FLOOR, dtype=np.int8)

    planes = [(0, 0.0, LEFT), (0, W, RIGHT), (1, 0.0, FACADE),
              (1, D, BACK), (2, 0.0, FLOOR), (2, H, CEILING)]
    for axis, val, cid in planes:
        d = dirs[:, axis]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (val - o[axis]) / d
        ok = np.isfinite(t) & (t > 1e-6) & (t < t_best)
        if not ok.any():
            continue
        p = o[None, :] + t[:, None] * dirs
        inside = np.ones(len(dirs), bool)
        for ax, lim in ((0, W), (1, D), (2, H)):
            if ax == axis:
                continue
            inside &= (p[:, ax] >= -1e-6) & (p[:, ax] <= lim + 1e-6)
        ok &= inside
        t_best[ok] = t[ok]
        code[ok] = cid
    pt = o[None, :] + np.where(np.isfinite(t_best), t_best, 0.0)[:, None] * dirs

    # carve the window out of the façade wall
    hw = room.width_m * room.window_width_frac / 2.0
    cx = room.width_m / 2.0
    on_facade = code == FACADE
    in_win = (on_facade
              & (np.abs(pt[:, 0] - cx) <= hw)
              & (pt[:, 2] >= room.window_sill_m)
              & (pt[:, 2] <= room.window_head_m))
    code[in_win] = WINDOW
    return code, pt


class RoomGeometry:
    """Direction classification and view factors — computed once per room."""

    def __init__(self, room: Room, n_directions: int = 512, seed: int = 0):
        self.room = room
        self.dirs = fibonacci_sphere(n_directions)
        self.w = 4.0 * math.pi / n_directions
        self.R = room.rotation()
        self.dirs_world = self.dirs @ self.R.T

        eye = np.array([room.desk_x_m, room.desk_y_m, room.eye_height_m])
        self.eye = eye
        self.eye_code, self.eye_hit = _box_hit(room, eye, self.dirs)

        self.patches = {
            FACADE: (np.array([room.width_m / 2, 1e-3, room.height_m / 2]),
                     np.array([0.0, 1.0, 0.0])),
            BACK: (np.array([room.width_m / 2, room.depth_m - 1e-3,
                             room.height_m / 2]), np.array([0.0, -1.0, 0.0])),
            LEFT: (np.array([1e-3, room.depth_m / 2, room.height_m / 2]),
                   np.array([1.0, 0.0, 0.0])),
            RIGHT: (np.array([room.width_m - 1e-3, room.depth_m / 2,
                              room.height_m / 2]), np.array([-1.0, 0.0, 0.0])),
            CEILING: (np.array([room.width_m / 2, room.depth_m / 2,
                                room.height_m - 1e-3]),
                      np.array([0.0, 0.0, -1.0])),
            FLOOR: (np.array([room.width_m / 2, room.depth_m / 2, 1e-3]),
                    np.array([0.0, 0.0, 1.0])),
        }
        self.vf = {}
        for pid, (o, n) in self.patches.items():
            cn = self.dirs @ n
            mask = cn > 1e-6
            code, _ = _box_hit(room, o, self.dirs)
            f = {}
            for tgt in (WINDOW, *INTERIOR):
                sel = mask & (code == tgt)
                f[tgt] = float(np.sum(cn[sel]) * self.w / math.pi)
            tot = sum(f.values()) or 1.0
            self.vf[pid] = {k: v / tot for k, v in f.items()}

        self.sky_view = float(np.mean(self.eye_code == WINDOW))
        self.materials = {
            FACADE: spectra.material(room.wall_material),
            BACK: spectra.material(room.wall_material),
            LEFT: spectra.material(room.wall_material),
            RIGHT: spectra.material(room.wall_material),
            CEILING: spectra.material(room.ceiling_material),
            FLOOR: spectra.material(room.floor_material),
        }


# --------------------------------------------------------------------------
# Exterior radiance seen through the window
# --------------------------------------------------------------------------

def exterior_field(room: Room, geo: RoomGeometry, dome: SkyDome,
                   sky: SpectralSky):
    """
    Spectral radiance arriving at the window plane from each direction, and
    the specular (sun-disc) term.

    Directions below the obstruction elevation see the opposing block; below
    the horizon they see the ground.
    """
    dw = geo.dirs_world
    el = np.degrees(np.arcsin(np.clip(dw[:, 2], -1, 1)))
    L = np.zeros((len(dw), len(spectra.WL)))

    ghi = sky.global_horizontal
    sky_sel = el > room.obstruction_deg
    L[sky_sel] = dome.radiance[sky_sel]
    obs_sel = (el <= room.obstruction_deg) & (el > 0.0)
    L[obs_sel] = (spectra.material(room.obstruction_material)[None, :]
                  * ghi[None, :] * 0.5 / math.pi)
    gnd_sel = el <= 0.0
    L[gnd_sel] = (spectra.material(room.ground_material)[None, :]
                  * ghi[None, :] / math.pi)

    # direct beam, if it clears the obstruction and strikes the façade
    beam = np.zeros(len(spectra.WL))
    n_out = room.outward_normal()
    cos_inc = 0.0
    profile = 0.0
    if sky.sun.up and sky.sun.altitude_deg > room.obstruction_deg:
        sv = dome.sun_dir
        cos_inc = float(sv @ n_out)
        if cos_inc > 0:
            beam = sky.beam_normal
            # profile angle of the sun in the plane normal to the façade
            horiz = sv - (sv @ n_out) * n_out
            horiz[2] = 0.0
            profile = math.degrees(math.atan2(
                sv[2], max(float(sv @ n_out), 1e-6)))
    return L, beam, cos_inc, profile


# --------------------------------------------------------------------------
# Interior solution
# --------------------------------------------------------------------------

#: Solid angle of the solar disc, sr.  Kept explicit because DGP depends on
#: L^2 * omega, which is resolution-dependent if the sun is folded into a
#: sampled direction instead of carried as a discrete source.
OMEGA_SUN = 6.8e-5


@dataclass
class Solution:
    eye_spectral: np.ndarray        # W m-2 nm-1 on the vertical view plane
    eye_luminance: np.ndarray       # cd m-2 per sampled direction
    point_sources: list             # (luminance, solid_angle, dir_local)
    medi: float
    lux_vertical: float
    lux_workplane: float
    der: float
    transmitted_solar_w: float      # through the glazing, whole-spectrum
    beam_solar_on_occupant_w_m2: float
    diffuse_solar_on_occupant_w_m2: float
    in_sun_patch: bool
    lighting_w: float
    coverage: float
    window_luminance_mean: float


def solve(room: Room, geo: RoomGeometry, dome: SkyDome, sky: SpectralSky,
          shade: ShadeDevice, coverage: float, electric_fraction: float = 0.0,
          heading_local: np.ndarray | None = None) -> Solution:
    """
    One timestep: exterior -> window -> interior radiosity -> eye.

    ``electric_fraction`` dims the installed electric lighting 0..1.
    """
    Lext, beam, cos_inc, profile = exterior_field(room, geo, dome, sky)
    f_spec, f_diff, tau_fabric = shade.transmission(coverage, profile)
    tau_glz = spectra.glazing(room.glazing)

    # radiance transmitted through the window, per direction
    diffuse_source = np.einsum(
        "i,ij->j", np.clip(geo.dirs_world @ room.outward_normal(), 0, None)
        * geo.w, Lext)
    if cos_inc > 0:
        diffuse_source = diffuse_source + beam * cos_inc
    L_diffused = (f_diff * tau_fabric * tau_glz * diffuse_source) / math.pi

    L_win = (f_spec * tau_glz[None, :] * Lext
             + L_diffused[None, :] * np.ones((len(geo.dirs), 1)))
    beam_through = beam * tau_glz * f_spec

    # irradiance entering the window plane
    e_win_in = np.einsum(
        "i,ij->j",
        np.clip(geo.dirs_world @ room.outward_normal(), 0, None) * geo.w,
        L_win)
    if cos_inc > 0:
        e_win_in = e_win_in + beam_through * cos_inc

    # electric lighting: recessed ceiling luminaires, represented as a
    # downward exitance on the ceiling patch rather than an ad-hoc addition
    lp = room.lighting_power_density * electric_fraction
    lumens = lp * room.floor_area * room.lighting_efficacy
    spd_lamp = spectra.daylight_spd(room.lighting_cct)
    spd_lamp = spd_lamp / max(photometry.illuminance(spd_lamp), 1e-9)
    lamp_exitance = spd_lamp * (lumens / max(room.floor_area, 1e-9))

    # radiosity over the six interior patches, window as a source
    nwl = len(spectra.WL)
    order = list(INTERIOR)
    A = np.zeros((6, 6, nwl))
    rhs = np.zeros((6, nwl))
    for i, pid in enumerate(order):
        f = geo.vf[pid]
        rho = geo.materials[pid]
        for j, qid in enumerate(order):
            A[i, j] = -rho * f[qid]
        A[i, i] += 1.0
        rhs[i] = rho * (f[WINDOW] * e_win_in)
        if pid == CEILING:
            rhs[i] = rhs[i] + lamp_exitance
    B = np.linalg.solve(np.moveaxis(A, 2, 0), np.moveaxis(rhs, 1, 0)[:, :, None])
    B = np.moveaxis(B[:, :, 0], 0, 1)
    Bm = {pid: B[i] for i, pid in enumerate(order)}

    # eye radiance field
    L_eye = np.zeros((len(geo.dirs), nwl))
    win = geo.eye_code == WINDOW
    L_eye[win] = L_win[win]
    for pid in INTERIOR:
        sel = geo.eye_code == pid
        L_eye[sel] = Bm[pid] / math.pi

    n_local = heading_local if heading_local is not None \
        else room.view_normal_local()
    proj = np.clip(geo.dirs @ n_local, 0.0, None) * geo.w
    e_eye = proj @ L_eye

    # The solar disc, carried as a discrete source with its true solid angle,
    # and blocked geometrically when the blind has descended past the height
    # at which the eye-to-sun ray crosses the glazing.
    point_sources = []
    sun_local = geo.R.T @ dome.sun_dir if sky.sun.up else np.zeros(3)
    disc_blocked = None
    if cos_inc > 0 and sun_local[1] < -1e-6:
        t_cross = -geo.eye[1] / sun_local[1]
        z_cross = geo.eye[2] + t_cross * sun_local[2]
        x_cross = geo.eye[0] + t_cross * sun_local[0]
        hw = room.width_m * room.window_width_frac / 2.0
        on_glass = (abs(x_cross - room.width_m / 2.0) <= hw
                    and room.window_sill_m <= z_cross <= room.window_head_m)
        clear_below = shade.covered_band(room, coverage)
        disc_blocked = bool(on_glass and z_cross > clear_below)
        if on_glass and not disc_blocked:
            cs = float(sun_local @ n_local)
            if cs > 0:
                e_eye = e_eye + beam * tau_glz * cs
            lux_disc = photometry.illuminance(beam * tau_glz)
            point_sources.append((lux_disc / OMEGA_SUN, OMEGA_SUN,
                                  sun_local.copy()))

    # window directions above the blind edge lose their specular view
    if coverage > 1e-9:
        clear_below = shade.covered_band(room, coverage)
        covered_dir = (geo.eye_code == WINDOW) & (geo.eye_hit[:, 2] > clear_below)
        if covered_dir.any():
            L_eye_cov = (shade.transmission(1.0, 0.0)[1] * tau_fabric * tau_glz
                         * diffuse_source) / math.pi
            L_eye[covered_dir] = L_eye_cov
            e_eye = proj @ L_eye

    lum = 683.0 * (L_eye @ spectra.V_LAMBDA) * float(np.diff(spectra.WL)[0])

    # horizontal work-plane illuminance, for comparison with conventional tools
    proj_h = np.clip(geo.dirs[:, 2], 0.0, None) * geo.w
    e_wp = proj_h @ L_eye
    if point_sources and sun_local[2] > 0:
        e_wp = e_wp + beam * tau_glz * sun_local[2]

    # Visible-band flux -> whole-spectrum solar.  Written out rather than
    # folded into one constant, because the two corrections pull in opposite
    # directions and quietly cancelling them is how a factor of two hides.
    vis_to_solar = (room.glazing_solar_to_visible
                    / max(room.visible_band_solar_fraction, 1e-6))
    e_win_vis = float(np.trapezoid(e_win_in, spectra.WL))
    solar = e_win_vis * vis_to_solar * room.window_area

    beam_occ = 0.0
    if point_sources:
        beam_vis = float(np.trapezoid(beam * tau_glz, spectra.WL))
        cs = float(sun_local @ n_local)
        beam_occ = beam_vis * vis_to_solar * max(cs, 0.0)
    diffuse_occ = (e_win_vis * vis_to_solar
                   * float(np.mean(geo.eye_code == WINDOW)))
    medi = photometry.medi(e_eye)
    lux_v = photometry.illuminance(e_eye)
    return Solution(
        eye_spectral=e_eye, eye_luminance=lum, point_sources=point_sources,
        medi=medi, lux_vertical=lux_v,
        lux_workplane=photometry.illuminance(e_wp),
        der=(medi / lux_v if lux_v > 0 else float("nan")),
        transmitted_solar_w=solar,
        beam_solar_on_occupant_w_m2=beam_occ,
        diffuse_solar_on_occupant_w_m2=diffuse_occ,
        in_sun_patch=bool(point_sources),
        lighting_w=lp * room.floor_area,
        coverage=coverage,
        window_luminance_mean=float(np.mean(lum[win])) if win.any() else 0.0)
