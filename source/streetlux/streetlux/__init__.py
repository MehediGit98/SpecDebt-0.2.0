"""
streetlux — melanopic exposure along urban routes, for design review.

The question this package answers is not "how much light falls on the desk"
but "how much circadian-effective light reaches the eye of a person moving
through this street, facing the way they are actually facing".

    from streetlux import load_route, simulate, compare_modes

    route = load_route("examples/dhaka_mirpur_route.json")
    res = simulate(route, "walk")
    print(res.summary()["mean_medi_travel_lx"])

Melanopic values are always **lx melanopic EDI** (CIE S 026:2018).  They are
not photopic lux and must not be labelled as such.
"""

__version__ = "0.2.0"

from . import spectra, photometry            # noqa: F401  (import order matters)
from .canyon import Canopy, CanyonSection, EyeField     # noqa: F401
from .enclosure import MODES, Mode, mode                # noqa: F401
from .exposure import THRESHOLDS, RouteResult, simulate  # noqa: F401
from .reference_data import load_publication_references  # noqa: F401
from .route import Route, Site, load_route              # noqa: F401
from .scenarios import compare_modes, compare_designs, DesignScenario  # noqa: F401
from .sun import Atmosphere                             # noqa: F401


def self_check() -> dict:
    """Metrology self-check reported in generated artefacts.

    ``der_d65`` is allowed a tiny numerical residual because mEDI uses the
    fixed CIE S 026 D65 efficacy constant while the package integrates on a
    5 nm grid.  The independently derived constant and its percentage
    deviation make that discretisation error explicit.
    """
    x, y = photometry.chromaticity(spectra.d65())
    cal = photometry.calibration()
    state = spectra.provenance()
    der_d65 = photometry.der(spectra.d65())
    return {
        "d65_chromaticity": (round(x, 5), round(y, 5)),
        "d65_chromaticity_target": (0.31272, 0.32903),
        "d65_chromaticity_error": round(
            ((x - 0.31272) ** 2 + (y - 0.32903) ** 2) ** 0.5, 5),
        "der_d65": round(der_d65, 6),
        "der_d65_error_pct": round(100.0 * (der_d65 - 1.0), 4),
        "k_mel_v_d65_derived": cal["k_mel"],
        "k_mel_v_d65_official": photometry.K_MEL_V_D65_OFFICIAL,
        "k_mel_deviation_pct": round(cal["deviation_pct"], 4),
        "official_action_spectra_loaded": state["official_action_spectra"],
        "official_solar_spectrum_loaded": state["official_solar_spectrum"],
        "official_daylight_components_loaded":
            state["official_daylight_components"],
    }


SELF_CHECK = self_check()
