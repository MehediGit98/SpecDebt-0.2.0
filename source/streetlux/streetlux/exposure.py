"""
The exposure engine.

Design commitments, each of which is a finding from the deepgaze papers
turned into a hard requirement rather than an option:

1.  **No single-heading number is ever returned alone.**  Every result
    carries the full heading envelope.  The travel heading is reported as one
    member of that envelope, never as "the" value.
2.  **Dose is decomposed.**  Deficit attributable to trip duration is
    separated from deficit attributable to exposure rate, so a scheduling
    problem is never mistaken for a design problem.
3.  **Every branch is instrumented.**  A zero is only ever reported next to
    the count of times the code path that could have produced a non-zero was
    actually evaluated.
4.  **Provenance travels with the number.**  Every result records whether
    official CIE tables were loaded and what the derived melanopic constant
    was.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import photometry, spectra
from .canyon import GROUND, SKY, WALL_L, WALL_R, CanyonSection, EyeField
from .enclosure import Mode, mode as get_mode
from .route import Route
from .sky import SkyDome, fibonacci_sphere
from .sun import solar_position, spectral_sky

#: WELL v2 L03 melanopic EDI tiers and the Brown et al. (2022) daytime
#: recommendation, in lx melanopic EDI on a vertical plane at eye height.
THRESHOLDS = {"well_tier1": 136.0, "well_tier2_brown": 250.0}


@dataclass
class Counters:
    """Branch activation counters.  A zero with no evaluations is not a null."""
    steps: int = 0
    sun_up_steps: int = 0
    beam_visible_steps: int = 0
    canopy_sections: int = 0
    canopy_intercept_steps: int = 0
    enclosure_filtered_steps: int = 0

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class Step:
    section: str
    t_hours: float
    dt_s: float
    sun_altitude_deg: float
    sky_view_fraction: float
    heading_medi: np.ndarray      # (H,) lx melanopic EDI
    heading_lux: np.ndarray       # (H,) photopic lx
    travel_medi: float
    travel_lux: float
    travel_der: float
    components: dict              # travel-heading mEDI by source


def _component_split(field_: EyeField, heading, tau, beam_tau):
    """Travel-heading mEDI attributable to each source in the canyon."""
    out = {}
    base = field_.radiance
    for label, sel in (("sky", field_.code == SKY),
                       ("facade_left", field_.code == WALL_L),
                       ("facade_right", field_.code == WALL_R),
                       ("ground", field_.code == GROUND)):
        masked = np.zeros_like(base)
        masked[sel] = base[sel]
        saved, savedbeam = field_.radiance, field_.beam
        field_.radiance, field_.beam = masked, np.zeros_like(savedbeam)
        e = field_.vertical_irradiance(np.array([heading]), tau, beam_tau)[0]
        field_.radiance, field_.beam = saved, savedbeam
        out[label] = photometry.medi(e)
    saved = field_.radiance
    field_.radiance = np.zeros_like(saved)
    e = field_.vertical_irradiance(np.array([heading]), tau, beam_tau)[0]
    field_.radiance = saved
    out["direct_sun"] = photometry.medi(e)
    return out


@dataclass
class RouteResult:
    route: str
    mode: str
    mode_label: str
    headings_deg: np.ndarray
    steps: list = field(default_factory=list)
    counters: Counters = field(default_factory=Counters)
    provenance: dict = field(default_factory=dict)

    # ---------------- aggregates ----------------

    @property
    def duration_s(self) -> float:
        return sum(s.dt_s for s in self.steps)

    def _w(self):
        return np.array([s.dt_s for s in self.steps])

    def dose_lxh(self, which="travel") -> float:
        """Melanopic dose in lx melanopic EDI · hours."""
        w = self._w() / 3600.0
        if which == "travel":
            v = np.array([s.travel_medi for s in self.steps])
        elif which == "best":
            v = np.array([s.heading_medi.max() for s in self.steps])
        elif which == "worst":
            v = np.array([s.heading_medi.min() for s in self.steps])
        elif which == "mean_heading":
            v = np.array([s.heading_medi.mean() for s in self.steps])
        else:
            raise ValueError(which)
        return float(np.sum(v * w))

    def mean_medi(self, which="travel") -> float:
        d = self.duration_s / 3600.0
        return self.dose_lxh(which) / d if d > 0 else float("nan")

    def fraction_above(self, threshold: float, which="travel") -> float:
        w = self._w()
        if which == "travel":
            v = np.array([s.travel_medi for s in self.steps])
        elif which == "best":
            v = np.array([s.heading_medi.max() for s in self.steps])
        else:
            v = np.array([s.heading_medi.min() for s in self.steps])
        return float(np.sum(w[v >= threshold]) / np.sum(w)) if len(w) else 0.0

    def compliance_rose(self) -> np.ndarray:
        """Time-weighted mean mEDI for each heading over the whole trip."""
        w = self._w()
        M = np.stack([s.heading_medi for s in self.steps])
        return (w[:, None] * M).sum(0) / w.sum()

    def heading_spread(self) -> dict:
        """
        How much of the answer is decided by an unstated view direction.

        `ratio` is max/min over headings of the trip-mean mEDI.  If this is
        far from 1, any single reported number is underdetermined.
        """
        r = self.compliance_rose()
        return {"min": float(r.min()), "median": float(np.median(r)),
                "max": float(r.max()),
                "ratio_max_min": float(r.max() / max(r.min(), 1e-9)),
                "travel_heading_percentile": float(
                    100.0 * np.mean(r <= self.mean_medi("travel")))}

    def decompose_dose(self, ref_rate_medi: float,
                       ref_duration_h: float) -> dict:
        """
        Split the dose gap against a reference trip into a duration part and
        an exposure-rate part.

        Paper 1 found that 47% of an apparent melanopic deficit was occupant
        absence rather than reduced exposure.  The same confound governs
        commutes: a fast dim trip and a slow bright one are not comparable
        until this split is made, and a design review that skips it will
        blame the street for what is really a travel-time difference.

        The split is the exact two-factor Shapley decomposition: the
        average of the two sequential orderings (duration attributed
        first at the reference rate, then rate at this trip's own
        duration; and the reverse).  A single ordering is order-dependent
        -- swapping which factor goes first changes the reported share by
        tens of percentage points on real routes, even though neither
        ordering is privileged -- while the average is still exact (sums
        to the gap) and, unlike either ordering alone, is symmetric under
        swapping which trip is the reference:
            d_dose = 0.5*(T - T_ref)*(rate + rate_ref)
                   + 0.5*(T + T_ref)*(rate - rate_ref)
        """
        T = self.duration_s / 3600.0
        rate = self.mean_medi("travel")
        dose = rate * T
        ref_dose = ref_rate_medi * ref_duration_h
        gap = dose - ref_dose
        dur_part = 0.5 * (T - ref_duration_h) * (rate + ref_rate_medi)
        rate_part = 0.5 * (T + ref_duration_h) * (rate - ref_rate_medi)
        denom = abs(dur_part) + abs(rate_part)
        return {"duration_h": T, "reference_duration_h": ref_duration_h,
                "exposure_rate_medi": rate, "reference_rate_medi": ref_rate_medi,
                "dose_lxh": dose, "reference_dose_lxh": ref_dose,
                "gap_lxh": gap,
                "gap_from_duration_lxh": dur_part,
                "gap_from_exposure_rate_lxh": rate_part,
                "duration_share_of_gap": (abs(dur_part) / denom
                                          if denom > 0 else float("nan")),
                "dose_at_reference_duration_lxh": rate * ref_duration_h}

    def source_shares(self) -> dict:
        w = self._w()
        keys = self.steps[0].components.keys() if self.steps else []
        tot = {k: float(np.sum([s.components[k] * ww
                                for s, ww in zip(self.steps, w)])) for k in keys}
        s = sum(tot.values()) or 1.0
        return {k: v / s for k, v in tot.items()}

    def summary(self) -> dict:
        sp = self.heading_spread()
        return {
            "route": self.route,
            "mode": self.mode,
            "mode_label": self.mode_label,
            "duration_min": self.duration_s / 60.0,
            "mean_medi_travel_lx": self.mean_medi("travel"),
            "mean_medi_best_heading_lx": self.mean_medi("best"),
            "mean_medi_worst_heading_lx": self.mean_medi("worst"),
            "dose_travel_lxh": self.dose_lxh("travel"),
            "dose_best_lxh": self.dose_lxh("best"),
            "dose_worst_lxh": self.dose_lxh("worst"),
            "mean_photopic_lux_travel": float(
                np.average([s.travel_lux for s in self.steps],
                           weights=self._w())) if self.steps else float("nan"),
            "mean_der": float(
                np.average([s.travel_der for s in self.steps],
                           weights=self._w())) if self.steps else float("nan"),
            "frac_time_above_250_travel": self.fraction_above(250.0, "travel"),
            "frac_time_above_250_best": self.fraction_above(250.0, "best"),
            "frac_time_above_136_travel": self.fraction_above(136.0, "travel"),
            "heading_spread": sp,
            "source_shares": self.source_shares(),
            "counters": self.counters.as_dict(),
            "provenance": self.provenance,
        }


def simulate(route: Route, mode_key: str, n_headings: int = 24,
             n_directions: int = 768, step_seconds: float = 60.0,
             seed: int = 0) -> RouteResult:
    """Walk (or ride) the route and record heading-resolved melanopic exposure."""
    md: Mode = get_mode(mode_key)
    dirs = fibonacci_sphere(n_directions)
    w = 4.0 * np.pi / n_directions
    headings = np.arange(n_headings) * (360.0 / n_headings)

    res = RouteResult(route=route.name, mode=md.key, mode_label=md.label,
                      headings_deg=headings)
    t0 = time.time()
    clock = route.start_hour

    for sec in route.sections:
        sec_use = CanyonSection(**{**sec.__dict__,
                                   "eye_height_m": md.eye_height_m})
        travel_s = sec.length_m / (md.speed_kmh / 3.6)
        n_steps = max(1, int(round(travel_s / step_seconds)))
        dt = travel_s / n_steps
        if sec.canopy.present:
            res.counters.canopy_sections += 1

        for k in range(n_steps):
            t_mid = clock + (k + 0.5) * dt / 3600.0
            sun = solar_position(route.site.latitude, route.site.longitude,
                                 route.site.timezone_hours,
                                 route.year, route.month, route.day, t_mid)
            sky = spectral_sky(sun, route.atmosphere)
            dome = SkyDome(sky, dirs)
            fld = EyeField(sec_use, dome, sky, dirs, w)

            tau = md.tau(dirs, sec.azimuth_deg)
            beam_tau = md.beam_tau(dome.sun_dir, sec.azimuth_deg)
            if not md.open_air:
                res.counters.enclosure_filtered_steps += 1
            if sec.canopy.present:
                res.counters.canopy_intercept_steps += 1

            E = fld.vertical_irradiance(headings, tau, beam_tau)     # (H, W)
            medi = np.array([photometry.medi(e) for e in E])
            lux = np.array([photometry.illuminance(e) for e in E])

            Et = fld.vertical_irradiance(np.array([sec.azimuth_deg]),
                                         tau, beam_tau)[0]
            tm, tl = photometry.medi(Et), photometry.illuminance(Et)
            comp = _component_split(fld, sec.azimuth_deg, tau, beam_tau)

            res.counters.steps += 1
            if sun.up:
                res.counters.sun_up_steps += 1
            if fld.beam.any():
                res.counters.beam_visible_steps += 1

            res.steps.append(Step(
                section=sec.name, t_hours=t_mid, dt_s=dt,
                sun_altitude_deg=sun.altitude_deg,
                sky_view_fraction=fld.sky_view_fraction,
                heading_medi=medi, heading_lux=lux,
                travel_medi=tm, travel_lux=tl,
                travel_der=(tm / tl if tl > 0 else float("nan")),
                components=comp))

        clock += travel_s / 3600.0

    cal = photometry.calibration()
    res.provenance = {
        "package": "streetlux",
        "official_action_spectra": spectra.using_official_tables(),
        "spectral_inputs": spectra.provenance(),
        "known_biases": spectra.active_biases(),
        "k_mel_v_d65_derived_W_per_lm": cal["k_mel"],
        "k_mel_deviation_from_CIE_S026_pct": cal["deviation_pct"],
        "n_directions": n_directions,
        "n_headings": n_headings,
        "step_seconds": step_seconds,
        "wall_clock_s": round(time.time() - t0, 2),
        "units": "melanopic values are lx melanopic EDI (CIE S 026), not photopic lux",
    }
    return res
