"""
Simulation driver.

One day is a sequence of timesteps.  At each, the controller is handed a
``probe`` that evaluates a candidate (coverage, electric dimming) pair and
returns full metrics, so that behaviour models and optimising controllers use
exactly the same physics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from streetlux import photometry, spectra
from streetlux.sky import SkyDome
from streetlux.sun import Atmosphere, solar_position, spectral_sky

from .behaviour import Controller, Occupancy
from .metrics import THRESHOLDS, EnergyModel, StepMetrics, evaluate
from .optics import Room, RoomGeometry, ShadeDevice, solve


_SKY_CACHE: dict = {}
_GEO_CACHE: dict = {}
CACHE_STATS = {"sky_hits": 0, "sky_misses": 0, "geo_hits": 0, "geo_misses": 0}


def clear_caches():
    _SKY_CACHE.clear()
    _GEO_CACHE.clear()
    for k in CACHE_STATS:
        CACHE_STATS[k] = 0


@dataclass(frozen=True)
class Site:
    name: str = "Dhaka"
    latitude: float = 23.8103
    longitude: float = 90.4125
    timezone_hours: float = 6.0


#: Representative days.  Chosen for solar geometry and monsoon cloud, weighted
#: to the number of days each stands for.
REPRESENTATIVE_DAYS = [
    {"key": "dry_winter", "month": 1, "day": 15, "cloud": 0.15, "weight": 90},
    {"key": "pre_monsoon", "month": 4, "day": 15, "cloud": 0.30, "weight": 90},
    {"key": "monsoon", "month": 7, "day": 15, "cloud": 0.80, "weight": 120},
    {"key": "post_monsoon", "month": 10, "day": 15, "cloud": 0.35, "weight": 65},
]


@dataclass
class DayResult:
    day_key: str
    steps: list[StepMetrics] = field(default_factory=list)
    counters: dict = field(default_factory=dict)
    weight: float = 1.0

    def _occ(self):
        return [s for s in self.steps if s.occupied]

    def aggregates(self, dt_h: float) -> dict:
        occ = self._occ()
        if not occ:
            return {}
        n = len(occ)
        medi = np.array([s.medi for s in occ])
        dgp = np.array([s.dgp for s in occ])
        cov = np.array([s.coverage for s in occ])
        pw = np.array([s.electric_w for s in occ])
        return {
            "occupied_hours": n * dt_h,
            "medi_mean": float(medi.mean()),
            "medi_dose_lxh": float(medi.sum() * dt_h),
            "frac_medi_above_250": float((medi >= 250.0).mean()),
            "frac_medi_above_136": float((medi >= 136.0).mean()),
            "dgp_mean": float(dgp.mean()),
            "frac_dgp_above_035": float((dgp >= 0.35).mean()),
            "frac_dgp_above_040": float((dgp >= 0.40).mean()),
            "mean_coverage": float(cov.mean()),
            "electric_kwh": float(pw.sum() * dt_h / 1000.0),
            "mean_workplane_lux": float(
                np.mean([s.lux_workplane for s in occ])),
            "mean_der": float(np.mean([s.der for s in occ])),
            "glare_source_steps": int(sum(1 for s in occ if s.n_glare_sources)),
        }


@dataclass
class RunResult:
    room: str
    controller: str
    controller_label: str
    shade: str
    days: list[DayResult] = field(default_factory=list)
    dt_h: float = 0.5
    provenance: dict = field(default_factory=dict)

    def aggregates(self) -> dict:
        rows = [(d.aggregates(self.dt_h), d.weight) for d in self.days]
        rows = [(a, w) for a, w in rows if a]
        if not rows:
            return {}
        wsum = sum(w for _, w in rows)
        out = {}
        for k in rows[0][0]:
            out[k] = float(sum(a[k] * w for a, w in rows) / wsum)
        out["annual_medi_dose_lxh"] = float(
            sum(a["medi_dose_lxh"] * w for a, w in rows))
        out["annual_electric_kwh"] = float(
            sum(a["electric_kwh"] * w for a, w in rows))
        counters = {}
        for d in self.days:
            for k, v in d.counters.items():
                counters[k] = counters.get(k, 0) + v
        out["counters"] = counters
        return out

    def decompose_medi_gap(self, ref_mean: float, ref_hours: float) -> dict:
        """
        Split the melanopic dose gap into an occupancy part and an exposure
        part, exactly as in the route work.  A design that loses dose because
        the occupant is at lunch needs a different fix from one that loses it
        because the blind is down.
        """
        a = self.aggregates()
        H, rate = a["occupied_hours"], a["medi_mean"]
        gap = rate * H - ref_mean * ref_hours
        occ_part = 0.5 * (rate + ref_mean) * (H - ref_hours)
        exp_part = 0.5 * (H + ref_hours) * (rate - ref_mean)
        denom = abs(occ_part) + abs(exp_part)
        return {"occupied_hours": H, "reference_hours": ref_hours,
                "exposure_rate_medi": rate, "reference_rate_medi": ref_mean,
                "gap_lxh": gap, "gap_from_occupancy_lxh": occ_part,
                "gap_from_exposure_lxh": exp_part,
                "occupancy_share_of_gap": (abs(occ_part) / denom
                                           if denom > 0 else float("nan"))}


def geometry_for(room: Room, n_directions: int) -> RoomGeometry:
    """Cached room geometry.  Direction classification and view factors depend
    only on the box, the window and the orientation."""
    key = (room.width_m, room.depth_m, room.height_m, room.orientation_deg,
           room.window_sill_m, room.window_head_m, room.window_width_frac,
           room.desk_x_m, room.desk_y_m, room.eye_height_m,
           room.wall_material, room.ceiling_material, room.floor_material,
           n_directions)
    if key in _GEO_CACHE:
        CACHE_STATS["geo_hits"] += 1
        return _GEO_CACHE[key]
    CACHE_STATS["geo_misses"] += 1
    g = RoomGeometry(room, n_directions)
    _GEO_CACHE[key] = g
    return g


def sky_for(site, geo, year, month, day, hour, cloud):
    """Cached (sun, sky, dome).  Reused across every optimiser evaluation."""
    key = (site.latitude, site.longitude, site.timezone_hours, year, month,
           day, round(hour, 4), round(cloud, 4), len(geo.dirs),
           round(float(geo.dirs_world[0, 0]), 9))
    hit = _SKY_CACHE.get(key)
    if hit is not None:
        CACHE_STATS["sky_hits"] += 1
        return hit
    CACHE_STATS["sky_misses"] += 1
    atm = Atmosphere(turbidity_beta=0.34, turbidity_alpha=1.2, ozone_cm=0.26,
                     water_cm=4.5, cloud_cover=cloud)
    sun = solar_position(site.latitude, site.longitude, site.timezone_hours,
                         year, month, day, hour)
    sky = spectral_sky(sun, atm)
    dome = SkyDome(sky, geo.dirs_world)
    _SKY_CACHE[key] = (sky, dome)
    return sky, dome


def simulate_day(room: Room, geo: RoomGeometry, ctrl: Controller,
                 shade: ShadeDevice, site: Site, year: int, month: int,
                 day: int, cloud: float, occupancy: Occupancy,
                 energy: EnergyModel, dt_h: float = 0.5,
                 start_h: float = 7.0, end_h: float = 19.0,
                 day_key: str = "day", weight: float = 1.0) -> DayResult:
    view = room.view_normal_local()
    res = DayResult(day_key=day_key, weight=weight)
    t = start_h
    while t < end_h - 1e-9:
        tm = t + dt_h / 2.0
        sky, dome = sky_for(site, geo, year, month, day, tm, cloud)
        occ = occupancy.occupied(tm)

        cache: dict = {}

        def probe(coverage: float, electric: float):
            key = (round(coverage, 4), round(electric, 4))
            if key not in cache:
                sol = solve(room, geo, dome, sky, shade, coverage, electric)
                cache[key] = evaluate(sol, geo.dirs, geo.w, view, room,
                                      energy, tm, occ)
            return cache[key]

        res.steps.append(ctrl.act(probe, tm, occ))
        t += dt_h
    res.counters = dict(ctrl.counters)
    return res


def run(room: Room, controller_factory, shade: ShadeDevice,
        site: Site | None = None, days=None, occupancy: Occupancy | None = None,
        energy: EnergyModel | None = None, n_directions: int = 384,
        dt_h: float = 0.5, year: int = 2026, seed: int = 0) -> RunResult:
    """Simulate one room / controller / shade combination across the year."""
    site = site or Site()
    occupancy = occupancy or Occupancy()
    energy = energy or EnergyModel()
    days = days or REPRESENTATIVE_DAYS
    geo = geometry_for(room, n_directions)
    t0 = time.time()

    # One controller for the whole run: a blind left down on Monday is still
    # down on Tuesday, and that persistence is the mechanism under test.
    ctrl = controller_factory(seed)
    out = RunResult(room=room.name, controller=ctrl.key,
                    controller_label=ctrl.label, shade=shade.key, dt_h=dt_h)
    for i, d in enumerate(days):
        out.days.append(simulate_day(
            room, geo, ctrl, shade, site, year, d["month"], d["day"],
            d["cloud"], occupancy, energy, dt_h=dt_h,
            day_key=d["key"], weight=d["weight"]))
    cal = photometry.calibration()
    spectral_state = spectra.provenance()
    out.provenance = {
        "package": "shadeparadox",
        "metrology_from": "streetlux",
        "spectral_inputs": spectral_state,
        "official_action_spectra": spectral_state["official_action_spectra"],
        "official_solar_spectrum": spectral_state["official_solar_spectrum"],
        "official_daylight_components": spectral_state["official_daylight_components"],
        "k_mel_v_d65_derived_W_per_lm": cal["k_mel"],
        "k_mel_v_d65_official_W_per_lm": photometry.K_MEL_V_D65_OFFICIAL,
        "k_mel_deviation_from_CIE_S026_pct": cal["deviation_pct"],
        "known_biases": spectra.active_biases(),
        "n_directions": n_directions, "dt_h": dt_h, "seed": seed,
        "representative_days": days,
        "wall_clock_s": round(time.time() - t0, 2),
        "units": "melanopic values are lx melanopic EDI (CIE S 026), "
                 "not photopic lux",
    }
    return out
