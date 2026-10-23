"""Validation records (FRD-027 §12): which numerics-changing settings passed the quality gate, per target.

A record is ``(target: family or model, engine record, setting) -> pass | fail`` with evidence (GSM8K zero-shot,
16-shot shared prefix, garbage count; 128K needle for KV dtype), harness version and date. A module offers a
setting only with a passing record; a failing one is shown with its reason. Tolerance: within 1 point of the
original-precision baseline on the same engine build, and no garbage. A family record covers ``base`` and
``quantized`` checkpoints of that lineage; ``finetune``/``merge`` need their own. A new build inherits a pass only
for settings whose kernels it declares unchanged (``kernels_unchanged``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

RECORDS_DIR = Path(__file__).resolve().parent / "validation_records"
TOLERANCE_POINTS = 1.0
FAMILY_RELATIONS = ("base", "quantized")


@dataclass
class ValidationRecord:
    target: str  # "family:Qwen3-8B" or "model:<id>"
    record: str  # engine record id
    setting: (
        str  # e.g. "weights.format=fp8", "spec.method=mtp", "state.dtype=bfloat16", "kv.dtype=fp8"
    )
    result: str  # pass | fail
    evidence: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    kernels_unchanged: Tuple[str, ...] = ()
    provenance: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidationRecord":
        return cls(
            target=str(data["target"]),
            record=str(data["record"]),
            setting=str(data["setting"]),
            result=str(data["result"]),
            evidence=dict(data.get("evidence") or {}),
            reason=str(data.get("reason") or ""),
            kernels_unchanged=tuple(data.get("kernels_unchanged") or ()),
            provenance=dict(data.get("provenance") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "record": self.record,
            "setting": self.setting,
            "result": self.result,
            "evidence": self.evidence,
            "reason": self.reason,
            "kernels_unchanged": list(self.kernels_unchanged),
            "provenance": self.provenance,
        }


def judge(baseline: float, candidate: float, garbage: int = 0) -> Tuple[str, str]:
    """The §12 tolerance on accuracy points (0-100)."""
    if garbage:
        return "fail", f"{garbage} garbage outputs"
    if candidate + TOLERANCE_POINTS < baseline:
        return (
            "fail",
            f"accuracy {candidate:.1f} vs baseline {baseline:.1f} (more than {TOLERANCE_POINTS} point below)",
        )
    return "pass", f"accuracy {candidate:.1f} vs baseline {baseline:.1f}"


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: str
    record: Optional[ValidationRecord] = None


class ValidationStore:
    def __init__(
        self,
        records: Iterable[ValidationRecord] = (),
        *,
        inherit: Optional[Mapping[str, str]] = None,
    ):
        self.records = list(records)
        self.inherit = dict(inherit or {})  # new record id -> predecessor record id

    @classmethod
    def load(cls, root: Optional[Path] = None) -> "ValidationStore":
        root = Path(root or RECORDS_DIR)
        records: List[ValidationRecord] = []
        inherit: Dict[str, str] = {}
        if root.exists():
            for path in sorted(root.glob("*.json")):
                data = json.loads(path.read_text())
                for item in data.get("records", []):
                    records.append(ValidationRecord.from_dict(item))
                inherit.update(data.get("inherit") or {})
        return cls(records, inherit=inherit)

    def gate(
        self,
        family: Optional[str],
        model_id: Optional[str],
        relation: str,
        record_id: str,
        setting: str,
    ) -> Verdict:
        """Whether ``setting`` may be offered for this checkpoint on this engine record."""
        candidates = []
        ids = [record_id]
        if record_id in self.inherit:
            ids.append(self.inherit[record_id])
        for rid in ids:
            for order, rec in enumerate(self.records):
                if rec.record != rid or rec.setting != setting:
                    continue
                if (
                    rid != record_id
                    and setting.split("=")[0] not in rec.kernels_unchanged
                    and setting not in rec.kernels_unchanged
                ):
                    continue  # inherited only for kernels the new build declares unchanged
                # most specific target first, then the build itself before its predecessor, then the newest
                # evidence (a re-run supersedes the earlier verdict; on the same date the record loaded last wins)
                newest = (str(rec.provenance.get("date") or ""), order)
                if model_id and rec.target == f"model:{model_id}":
                    candidates.append(((0, rid != record_id), newest, rec))
                elif family and rec.target == f"family:{family}":
                    if relation not in FAMILY_RELATIONS:
                        continue
                    candidates.append(((1, rid != record_id), newest, rec))
        if not candidates:
            why = f"no validation record for {setting} on {record_id}"
            if relation not in FAMILY_RELATIONS and family:
                why += f" (a {relation} checkpoint needs its own record: the family's evidence may not hold for it)"
            return Verdict(False, why)
        best = min(c[0] for c in candidates)
        rec = max((c for c in candidates if c[0] == best), key=lambda c: c[1])[2]
        return Verdict(rec.result == "pass", rec.reason or rec.result, rec)


_DEFAULT: Optional[ValidationStore] = None


def default_validation() -> ValidationStore:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ValidationStore.load()
    return _DEFAULT
