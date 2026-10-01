"""
shadeparadox — coupled thermal / visual / circadian shading experiments.

    from shadeparadox import e1_behaviour, e2_encoding_vs_algorithm

Melanopic metrology is imported from ``streetlux``.  There is one melanopic
constant in this codebase and one place it is derived.

Melanopic values are **lx melanopic EDI** (CIE S 026:2018), never photopic lux.
"""

__version__ = "0.2.0"

from streetlux import photometry, spectra  # noqa: F401

from .behaviour import (CONTROLLERS, ControllerParams, Occupancy,  # noqa: F401
                        controller)
from .control import ENCODINGS, OPTIMISERS, SPACE, score  # noqa: F401
from .experiment import (bootstrap_ci, e1_behaviour,  # noqa: F401
                         e2_encoding_vs_algorithm, e3_view_direction)
from .metrics import THRESHOLDS, EnergyModel, dgp  # noqa: F401
from .optics import SHADES, Room, RoomGeometry, ShadeDevice, solve  # noqa: F401
from .sim import REPRESENTATIVE_DAYS, Site, clear_caches, run  # noqa: F401


def self_check() -> dict:
    """Metrology self-check, inherited from streetlux and reported here too."""
    import streetlux
    c = streetlux.self_check()
    c["metrology_source"] = "streetlux " + streetlux.__version__
    return c


SELF_CHECK = self_check()
