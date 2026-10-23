"""The FRD-027 predictor's public API (IMPLEMENTATION_PLAN 0.5): predict · limits · validate · boot_time · versions.

A :class:`Predictor` wraps one engine record's ``EngineModel`` with the grounding store, memoises every call by
a digest of its inputs (NFR-2: a plan evaluates each candidate once) and counts evaluations so budsim can enforce
the plan's budget. The same inputs, records, device facts and store version always give the same answers
(NFR-3); :func:`versions` names what to pin in the runtime spec.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .engine import EngineRecord, Estimate, Limits, Prediction, Refusal, Workload, engine_for
from .grounding.store import GroundingStore, default_store

PREDICTOR_VERSION = "lmc-0.1.0+planner.1"


def versions(store: Optional[GroundingStore] = None) -> Dict[str, str]:
    store = store or default_store()
    return {"version": PREDICTOR_VERSION, "grounding": store.version}


def _digest(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _id(obj: Any) -> str:
    if obj is None:
        return "-"
    digest = getattr(obj, "digest", None)
    return digest() if callable(digest) else str(obj)


class EvaluationBudgetExceeded(RuntimeError):
    """The plan asked for more predictor evaluations than its budget allows (NFR-2)."""


class Predictor:
    def __init__(
        self,
        record: Any,
        store: Optional[GroundingStore] = None,
        *,
        max_evaluations: Optional[int] = None,
    ):
        if not isinstance(record, EngineRecord):
            record = EngineRecord.from_dict(record)
        self.record = record
        self.store = store or default_store()
        self.engine = engine_for(record, self.store)
        self.max_evaluations = max_evaluations
        self.evaluations = 0  # distinct predictions computed (memo misses)
        self.calls = 0
        self._memo: Dict[Tuple[str, str], Any] = {}

    # ------------------------------------------------------------------ identity

    def versions(self) -> Dict[str, str]:
        return versions(self.store)

    def _key(self, kind: str, *parts: Any) -> Tuple[str, str]:
        return kind, _digest(self.record.id, self.store.version, *parts)

    def _memoised(self, kind: str, parts: Tuple[Any, ...], compute):
        self.calls += 1
        key = self._key(kind, *parts)
        if key in self._memo:
            return self._memo[key]
        if kind == "predict":
            if self.max_evaluations is not None and self.evaluations >= self.max_evaluations:
                raise EvaluationBudgetExceeded(
                    f"{self.evaluations} predictor evaluations used of {self.max_evaluations}"
                )
            self.evaluations += 1
        value = compute()
        self._memo[key] = value
        return value

    # ------------------------------------------------------------------ the API

    def supports(self, setting: str) -> bool:
        return self.engine.supports(setting)

    def check_supported(self, settings: Mapping[str, Any]) -> None:
        self.engine.check_supported(settings)

    def validate(self, settings: Mapping[str, Any], model: Any = None) -> List[Refusal]:
        return self._memoised(
            "validate", (dict(settings), _id(model)), lambda: self.engine.validate(settings, model)
        )

    def limits(
        self, settings: Mapping[str, Any], model: Any, device: Any, *, draft: Any = None
    ) -> Limits:
        return self._memoised(
            "limits",
            (dict(settings), _id(model), _id(device), _id(draft)),
            lambda: self.engine.limits(settings, model, device, draft=draft),
        )

    def predict(
        self,
        settings: Mapping[str, Any],
        workload: Workload,
        model: Any,
        device: Any,
        *,
        draft: Any = None,
    ) -> Prediction:
        return self._memoised(
            "predict",
            (dict(settings), workload.digest_key(), _id(model), _id(device), _id(draft)),
            lambda: self.engine.predict(settings, workload, model, device, draft=draft),
        )

    def boot_time(self, settings: Mapping[str, Any], model: Any, device: Any) -> Estimate:
        return self._memoised(
            "boot_time",
            (dict(settings), _id(model), _id(device)),
            lambda: self.engine.boot_time(settings, model, device),
        )


def predict(
    settings: Mapping[str, Any],
    workload: Workload,
    model: Any,
    device: Any,
    record: Any,
    *,
    draft: Any = None,
) -> Prediction:
    return Predictor(record).predict(settings, workload, model, device, draft=draft)


def limits(
    settings: Mapping[str, Any], model: Any, device: Any, record: Any, *, draft: Any = None
) -> Limits:
    return Predictor(record).limits(settings, model, device, draft=draft)


def validate(settings: Mapping[str, Any], record: Any, model: Any = None) -> List[Refusal]:
    return Predictor(record).validate(settings, model)


def boot_time(settings: Mapping[str, Any], model: Any, device: Any, record: Any) -> Estimate:
    return Predictor(record).boot_time(settings, model, device)
