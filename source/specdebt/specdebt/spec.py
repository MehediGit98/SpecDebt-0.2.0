"""
The specification space.

A published metric is usually reported as a number.  It is almost never a
number: it is a function of the data *and* of a set of analyst choices, some
of which the standard or the benchmark fixes, and some of which it leaves
free.  When a free choice is left undeclared, the reported value is one draw
from a distribution that the reader cannot see and cannot reconstruct.

This module describes that distribution explicitly.  A :class:`Factor` is one
choice point.  A :class:`Specification` is the set of them, partitioned into
declared (fixed by the standard, or stated in the paper) and undeclared (free,
and silently chosen).

Vocabulary, used consistently throughout:

  free parameter      a factor the metric's definition does not pin down
  declared            a free parameter whose value the report states
  undeclared          a free parameter whose value the report does not state
  specification       one complete assignment of values to every factor
  envelope            the range of the metric over the undeclared subspace
  flip                a change in the reported decision, not just the value
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Factor:
    """
    One choice point in a metric's specification.

    ``levels`` are the plausible values a competent analyst might pick.  The
    honesty of everything downstream rests on this list: too narrow and the
    debt is understated, too wide and it is manufactured.  ``justification``
    is required rather than optional for that reason.
    """
    name: str
    levels: tuple
    justification: str
    declared: bool = False
    domain: str = "analysis"

    def __post_init__(self):
        if len(self.levels) < 1:
            raise ValueError(f"factor {self.name!r} has no levels")
        if not self.justification.strip():
            raise ValueError(
                f"factor {self.name!r} needs a justification for its levels; "
                "an undocumented level set is itself specification debt")

    @property
    def n(self) -> int:
        return len(self.levels)

    @property
    def is_free(self) -> bool:
        return self.n > 1


@dataclass
class Specification:
    """A set of factors, with a declared/undeclared partition."""
    factors: list = field(default_factory=list)
    name: str = "specification"

    def add(self, name, levels, justification, declared=False,
            domain="analysis") -> "Specification":
        self.factors.append(Factor(name, tuple(levels), justification,
                                   declared, domain))
        return self

    # -- views -------------------------------------------------------------
    def __len__(self):
        return len(self.factors)

    def __getitem__(self, name):
        for f in self.factors:
            if f.name == name:
                return f
        raise KeyError(name)

    @property
    def free(self) -> list:
        return [f for f in self.factors if f.is_free]

    @property
    def undeclared(self) -> list:
        return [f for f in self.free if not f.declared]

    @property
    def declared(self) -> list:
        return [f for f in self.free if f.declared]

    @property
    def size(self) -> int:
        """Number of distinct specifications in the full grid."""
        n = 1
        for f in self.factors:
            n *= f.n
        return n

    @property
    def undeclared_size(self) -> int:
        n = 1
        for f in self.undeclared:
            n *= f.n
        return n

    def declare(self, *names) -> "Specification":
        """Return a copy in which the named factors are declared."""
        out = Specification(name=self.name)
        for f in self.factors:
            out.factors.append(Factor(f.name, f.levels, f.justification,
                                      f.declared or f.name in names, f.domain))
        return out

    # -- enumeration -------------------------------------------------------
    def grid(self):
        """Every specification in the full factorial, as dicts."""
        names = [f.name for f in self.factors]
        for combo in itertools.product(*[f.levels for f in self.factors]):
            yield dict(zip(names, combo))

    def sample(self, n: int, seed: int = 0):
        """Random specifications, for spaces too large to enumerate."""
        rng = np.random.default_rng(seed)
        for _ in range(n):
            yield {f.name: f.levels[rng.integers(f.n)] for f in self.factors}

    def enumerate(self, max_grid: int = 20000, n_sample: int = 4000,
                  seed: int = 0):
        """
        Full factorial when it fits, random sample when it does not.

        Returns (list_of_specs, mode).  The mode travels into every result, so
        that a sampled estimate is never mistaken for an exhaustive one.
        """
        if self.size <= max_grid:
            return list(self.grid()), "full_factorial"
        return list(self.sample(n_sample, seed)), "random_sample"

    def describe(self) -> dict:
        return {
            "name": self.name,
            "n_factors": len(self.factors),
            "n_free": len(self.free),
            "n_undeclared": len(self.undeclared),
            "grid_size": self.size,
            "undeclared_grid_size": self.undeclared_size,
            "factors": [
                {"name": f.name, "levels": [str(x) for x in f.levels],
                 "n_levels": f.n, "declared": f.declared, "domain": f.domain,
                 "justification": f.justification}
                for f in self.factors],
        }
