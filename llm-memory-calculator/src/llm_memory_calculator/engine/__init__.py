"""FRD-027 Part A, layer 2: the engine model (§8)."""

from typing import Any

from .base import (
    BootCheck,
    EngineModel,
    Estimate,
    Limits,
    Prediction,
    Refusal,
    UnsupportedSetting,
    Workload,
    draft_key,
)
from .record import Conflict, EngineRecord
from .vllm import CEILINGS, VLLMEngine


def engine_for(record: Any, store: Any = None, profiles: Any = None, corrections: Any = None) -> "VLLMEngine":
    """The ``EngineModel`` for a record. v1 implements vLLM; another engine gets the vLLM rules as a stand-in with
    every prediction at the prior level (so it is ungrounded and modules keep engine defaults, NFR-5). ``profiles``
    (a ``perf.store.ProfileStore``) predicts a profiled configuration from its measured steps, and ``corrections``
    (``perf.residual.Corrections``) grounds its outputs (FRD-027 §20).
    """
    if not isinstance(record, EngineRecord):
        record = EngineRecord.from_dict(record)
    return VLLMEngine(record, store, profiles, corrections)


__all__ = [
    "CEILINGS",
    "BootCheck",
    "Conflict",
    "EngineModel",
    "EngineRecord",
    "Estimate",
    "Limits",
    "Prediction",
    "Refusal",
    "UnsupportedSetting",
    "VLLMEngine",
    "Workload",
    "draft_key",
    "engine_for",
]
