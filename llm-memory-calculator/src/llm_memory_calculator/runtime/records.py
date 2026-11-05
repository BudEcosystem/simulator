"""Engine runtime records: BudConnect's record merged with the vendored snapshot (FRD-025 §5, open question 1).

budsim reads BudConnect's compatible-engine rows (engine, image, version, upstream_vllm, kv_features, parsers,
supports_lora...). Until BudConnect carries the ``runtime`` block, the ``settings`` mapping, ``limits`` and
``conflicts`` (IMPLEMENTATION_PLAN 1.0b), the vendored record for the same engine and version supplies them.
Features are the union of what BudConnect lists and what the vendored record lists for that exact build --
both are "verified on that build" claims. A build with no vendored record gets the nearest record of the same
engine and upstream minor with its build-specific features dropped (the grounding engine ladder treats it as a
new build). A build of a family with records but none of its upstream minor gets the family's **baseline record**
(FRD-025 Revision 2, R2-2), so every such build is planned on a spec rather than left to budcluster's legacy sizing.
Every record says how it was resolved in ``record_basis``: ``budconnect``, ``exact``, ``inherited`` or ``baseline``.
"""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ..engine.record import KV_FEATURES, EngineRecord

RECORDS_DIR = Path(__file__).resolve().parent / "records"

#: Features only a verified build may claim; dropped when a record is inherited by another build.
BUILD_FEATURES = {
    "spec_mtp",
    "spec_eagle3",
    "spec_ngram",
    "kernel_warmup",
    "fused_gdn_mtp",
    "spec_prefix_cache_safe",
    "spec_prefix_cache_safe_full_attention",
}


@lru_cache(maxsize=1)
def vendored() -> Dict[str, Dict[str, Any]]:
    out = {}
    for path in sorted(RECORDS_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        out[data["id"]] = data
    return out


def _minor(version: Optional[str]) -> str:
    parts = str(version or "").split(".")
    return ".".join(parts[:2])


@lru_cache(maxsize=None)
def _baseline(engine: str) -> Optional[Dict[str, Any]]:
    """The family's baseline: what every recorded build of ``engine`` shares, and every conflict any of them has.

    Launch, argument syntax, ports, probes, limits, runner and parsers must be identical across the family's records
    to be carried (a field they disagree on is left out); settings are those every record maps, each as the oldest
    record maps it; features are those every record verified, minus build-specific ones. None when the family has no
    records.
    """
    family = sorted((r for r in vendored().values() if r.get("engine") == engine), key=lambda r: r["id"])
    if not family:
        return None
    base: Dict[str, Any] = {"engine": engine, "build": "baseline", "baseline_of": [r["id"] for r in family]}
    for key in ("runtime", "limits", "runner", "tool_parsers", "reasoning_parsers"):
        values = [json.dumps(r.get(key), sort_keys=True) for r in family]
        if len(set(values)) == 1 and family[0].get(key) is not None:
            base[key] = copy.deepcopy(family[0][key])
    shared = set.intersection(*(set(r.get("settings") or {}) for r in family))
    base["settings"] = {name: copy.deepcopy(family[0]["settings"][name]) for name in sorted(shared)}
    conflicts: Dict[str, Dict[str, Any]] = {}
    for r in family:
        for item in r.get("conflicts") or []:
            conflicts.setdefault(str(item.get("id")), copy.deepcopy(item))
    base["conflicts"] = [conflicts[k] for k in sorted(conflicts)]
    base["features"] = sorted(set.intersection(*(set(r.get("features") or ()) for r in family)) - BUILD_FEATURES)
    base["images"] = {}
    return base


def find_vendored(
    engine: str, version: Optional[str], upstream: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    records = list(vendored().values())
    for data in records:
        if data.get("engine") == engine and str(data.get("version")) == str(version):
            return {**copy.deepcopy(data), "record_basis": "exact"}
    for data in records:  # same upstream minor: inherit, without build-specific features
        if (
            data.get("engine") == engine
            and upstream
            and _minor(data.get("upstream_vllm")) == _minor(upstream)
        ):
            inherited = copy.deepcopy(data)
            inherited["features"] = [
                f for f in inherited.get("features", []) if f not in BUILD_FEATURES
            ]
            inherited["inherited_from"] = data["id"]
            inherited["record_basis"] = "inherited"
            return inherited
    baseline = _baseline(engine) if version not in (None, "") else None
    if baseline is None:  # no family records, or a row that names no build
        return None
    # FRD-025 R2-2: no record of this build or its upstream minor; plan it on what the family shares
    return {
        **copy.deepcopy(baseline),
        "id": f"{engine}@{version}~baseline",
        "version": str(version),
        "upstream_vllm": upstream,
        "record_basis": "baseline",
    }


def record_from_budconnect(row: Mapping[str, Any]) -> Optional[EngineRecord]:
    """An :class:`EngineRecord` for a budsim simulation row / node group (engine, engine_version, engine_image,
    upstream_vllm, kv_features, parsers). None for an engine the runtime spec does not cover yet."""
    engine = str(row.get("engine") or row.get("engine_type") or row.get("engine_name") or "")
    version = row.get("engine_version") or row.get("version")
    upstream = row.get("upstream_vllm")
    if row.get("runtime") and row.get("settings"):
        data = {**dict(row), "record_basis": "budconnect"}  # BudConnect already carries the FRD-025 block
    else:
        data = find_vendored(engine, version, upstream)
        if data is None:
            return None
    # FRD-023's KV features come from BudConnect only; the vendored record adds the rest of what the build verified
    features = (set(data.get("features") or ()) - KV_FEATURES) | set(row.get("kv_features") or ())
    if row.get("supports_lora") is False:
        features.discard("lora")
    data["features"] = sorted(features)
    image = row.get("engine_image") or row.get("image")
    if image:
        images = dict(data.get("images") or {})
        images["cuda"] = image
        data["images"] = images
    if row.get("tool_calling_parser_type"):
        data["tool_parsers"] = sorted(
            set(data.get("tool_parsers") or []) | {row["tool_calling_parser_type"]}
        )
    if row.get("reasoning_parser_type"):
        data["reasoning_parsers"] = sorted(
            set(data.get("reasoning_parsers") or []) | {row["reasoning_parser_type"]}
        )
    if version and data.get("inherited_from"):
        data["id"] = f"{engine}@{version}"
        data["version"] = str(version)
    return EngineRecord.from_dict(data)


def record(record_id: str) -> EngineRecord:
    return EngineRecord.from_dict(vendored()[record_id])


def all_records() -> List[EngineRecord]:
    return [EngineRecord.from_dict(d) for d in vendored().values()]
