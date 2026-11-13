"""Sizing Bud-Decision-Runtime replicas for decision-only models (Bud spec 031 DM4, FRD §5.16.3, CONTRACTS C10.1).

The runtime serves encoder decision models (Laya, Julia, GLiNER) and Kev on CUDA and CPU. This package answers "what
does this checkpoint cost on this device": its precision there, its throughput, how many calls one replica holds within
a call-latency target, and its memory terms. It holds no Bud service logic.
"""

from .calibration import CALIBRATED, ESTIMATED, SAME_ARCHITECTURE, runtime_version
from .families import (
    FAMILIES,
    NOT_SIZED,
    DecisionFacts,
    FamilyError,
    detect,
    facts,
    facts_from_configs,
)
from .plan import DecisionPlan, admission_limits, call_work, cpu_threads, size
from .precision import FLOAT32, HALF16, MIXED16, choose

__all__ = [
    "CALIBRATED",
    "ESTIMATED",
    "FAMILIES",
    "FLOAT32",
    "HALF16",
    "MIXED16",
    "NOT_SIZED",
    "SAME_ARCHITECTURE",
    "DecisionFacts",
    "DecisionPlan",
    "FamilyError",
    "admission_limits",
    "call_work",
    "choose",
    "cpu_threads",
    "detect",
    "facts",
    "facts_from_configs",
    "runtime_version",
    "size",
]
