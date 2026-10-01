"""Route definition and JSON round-tripping."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .canyon import Canopy, CanyonSection
from .sun import Atmosphere


@dataclass
class Site:
    name: str = "Dhaka"
    latitude: float = 23.8103
    longitude: float = 90.4125
    timezone_hours: float = 6.0


@dataclass
class Route:
    name: str
    site: Site = field(default_factory=Site)
    sections: list[CanyonSection] = field(default_factory=list)
    atmosphere: Atmosphere = field(default_factory=Atmosphere)
    year: int = 2026
    month: int = 3
    day: int = 15
    start_hour: float = 8.0

    @property
    def length_m(self) -> float:
        return sum(s.length_m for s in self.sections)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "site": asdict(self.site),
            "date": {"year": self.year, "month": self.month, "day": self.day,
                     "start_hour": self.start_hour},
            "atmosphere": asdict(self.atmosphere),
            "sections": [
                {**{k: v for k, v in asdict(s).items() if k != "canopy"},
                 "canopy": asdict(s.canopy)}
                for s in self.sections
            ],
        }

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    def variant(self, name: str, **section_overrides) -> "Route":
        """Copy the route; selective design operators preserve unrelated geometry."""
        import copy
        r = copy.deepcopy(self)
        r.name = name
        for s in r.sections:
            for k, v in section_overrides.items():
                if k == "min_width_m":
                    s.width_m = max(s.width_m, float(v))
                elif k == "max_height_m":
                    s.left_height_m = min(s.left_height_m, float(v))
                    s.right_height_m = min(s.right_height_m, float(v))
                elif k == "foliage_cover":
                    if s.canopy.transmittance != "opaque":
                        s.canopy.cover = float(v)
                elif k == "replace_ground":
                    if s.ground_material == v["from"]:
                        s.ground_material = v["to"]
                elif k == "canopy":
                    s.canopy = Canopy(**v) if isinstance(v, dict) else v
                else:
                    setattr(s, k, v)
        return r


def load_route(path) -> Route:
    d = json.loads(Path(path).read_text())
    site = Site(**d.get("site", {}))
    atm = Atmosphere(**d.get("atmosphere", {}))
    date = d.get("date", {})
    secs = []
    for s in d["sections"]:
        s = dict(s)
        can = Canopy(**s.pop("canopy", {}) or {})
        secs.append(CanyonSection(canopy=can, **s))
    return Route(name=d["name"], site=site, sections=secs, atmosphere=atm,
                 year=date.get("year", 2026), month=date.get("month", 3),
                 day=date.get("day", 15), start_hour=date.get("start_hour", 8.0))
