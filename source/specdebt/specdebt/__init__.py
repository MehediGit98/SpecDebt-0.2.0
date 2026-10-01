"""
specdebt — measuring the specification debt of a reported metric.

A metric with an undeclared free parameter is reported as determinate and is
not.  This package makes the hidden distribution explicit, says which choice
point is responsible, and computes the shortest set of declarations that
restores determinacy.

    from specdebt import run_study, melanopic_specification, ...
"""

__version__ = "0.2.0"

from .cases import (BENCHMARK_DECISION, MELANOPIC_DECISION,  # noqa: F401
                    accuracy, benchmark_metric, benchmark_specification,
                    extract, make_corpus, melanopic_metric,
                    melanopic_specification, normalise)
from .spec import Factor, Specification  # noqa: F401
from .study import Comparison, Result, Threshold, run_study  # noqa: F401
