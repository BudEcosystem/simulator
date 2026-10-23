"""Runtime-spec modules (FRD-025 §6.3, FRD-027 §10). Importing this package registers every module."""

from . import (  # noqa: F401 -- importing a module registers it
    capacity,
    fallback,
    guard,
    kv,
    numerics,
    parsers,
    pod,
    testing,
)  # noqa: F401  (registration)
from .base import REGISTRY, Module, build
from .capacity import PlanError

#: The FRD-027 module set (v2 where it exists), the default for a tuned plan.
PLANNER = {
    "*": "v1",
    "weights.precision": "v2",
    "spec_decode": "v2",
    "state.dtype": "v2",
    "kv.dtype": "v2",
    "context.batching": "v2",
    "memory.gpu": "v2",
    "lora": "v2",
    "probes": "v2",
    "engine.guard": "v2",
    "engine.fallback": "v2",
    "plan.predicted": "v2",
}
#: Phase 1: FRD-027's capacity modules only (stage 1 off) -- the shadow diff C-GOLD-02 is defined on this set.
PLANNER_CAPACITY = {
    "*": "v1",
    "context.batching": "v2",
    "memory.gpu": "v2",
    "lora": "v2",
    "probes": "v2",
    "engine.guard": "v2",
    "engine.fallback": "v2",
    "plan.predicted": "v2",
}
#: Today's behaviour (FRD-025 M2: spec render == legacy render).
LEGACY = {"*": "v1"}

__all__ = ["LEGACY", "PLANNER", "PLANNER_CAPACITY", "REGISTRY", "Module", "PlanError", "build"]
