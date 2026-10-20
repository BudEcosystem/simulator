"""The engine runtime record (FRD-025 §5, FRD-027 §11) as the planner reads it.

A record describes capability, not policy: features verified on the build, limits, the settings mapping (the
only place the engine's flag names appear) and the ``conflicts`` the build refuses or corrupts. budsim
fetches records from BudConnect; :mod:`llm_memory_calculator.runtime.records` vendors a snapshot for builds
whose BudConnect record has no ``runtime`` block yet (FRD-025 open question 1).
"""

from __future__ import annotations

import operator
import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Set, Tuple

from ..runtime.vocabulary import MAPPING_KEYS, VOCABULARY

_OPS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}
_COND = re.compile(r"^\s*(<=|>=|==|!=|<|>)\s*(.+?)\s*$")

#: FRD-023's KV features: BudConnect's ``kv_features`` is their only source (budcluster gates KV rendering on them).
KV_FEATURES = frozenset(
    {
        "kv_events",
        "cache_salt",
        "cpu_offload_shm",
        "cpu_offload_host_alloc",
        "cpu_offload_shm_per_copy",
        "kv_fp8",
        "tiering_fs",
    }
)

#: Settings whose value needs a record feature before an engine may use it.
FEATURE_GATES: Tuple[Tuple[str, Any, str], ...] = (
    ("spec.method", "mtp", "spec_mtp"),
    ("spec.method", "eagle3", "spec_eagle3"),
    ("spec.method", "ngram", "spec_ngram"),
    ("state.dtype", "bfloat16", "mamba_state_bf16"),
    ("kv.dtype", "fp8", "kv_fp8"),
    ("kv.dtype", "fp8_e4m3", "kv_fp8"),
    ("kv.dtype", "fp8_e5m2", "kv_fp8"),
    ("mm.text_only", True, "language_model_only"),
)


def _parse_condition(text: Any) -> Tuple[Any, Any]:
    if isinstance(text, (int, float, bool)) or text is None:
        return operator.eq, text
    match = _COND.match(str(text))
    if not match:
        return operator.eq, text
    op, raw = match.groups()
    try:
        value: Any = int(raw)
    except ValueError:
        try:
            value = float(raw)
        except ValueError:
            value = {"true": True, "false": False}.get(raw.lower(), raw)
    return _OPS[op], value


@dataclass(frozen=True)
class Conflict:
    id: str
    when: Mapping[str, Any]
    requires: Mapping[str, Any] = field(default_factory=dict)
    unless_feature: Optional[str] = None
    #: (feature, model trait) pairs: the build's feature waives the conflict for a model with that trait only
    unless_for: Tuple[Tuple[str, str], ...] = ()
    reason: str = ""

    def waived(self, features: Iterable[str], model: Any = None) -> bool:
        features = set(features)
        if self.unless_feature and self.unless_feature in features:
            return True
        traits = model_traits(model)
        return any(feature in features and trait in traits for feature, trait in self.unless_for)

    def matches(self, settings: Mapping[str, Any]) -> bool:
        for key, want in self.when.items():
            have = settings.get(key)
            if want == "*":
                if have in (None, False, "", 0):
                    return False
            elif have != want:
                return False
        return True

    def violated(self, settings: Mapping[str, Any], resolved: Mapping[str, Any]) -> Optional[str]:
        """The setting that breaks the requirement, or None. A conflict with no ``requires`` is violated by
        the ``when`` combination itself."""
        if not self.requires:
            return next(iter(self.when))
        for key, cond in self.requires.items():
            op, value = _parse_condition(cond)
            have = settings.get(key, resolved.get(key))
            if have is None:
                return key
            try:
                if not op(have, value):
                    return key
            except TypeError:
                return key
        return None


@dataclass(frozen=True)
class EngineRecord:
    id: str
    engine: str
    version: str
    upstream_vllm: Optional[str]
    build: str
    runner: str
    features: FrozenSet[str]
    limits: Mapping[str, Any]
    settings: Mapping[str, Mapping[str, Any]]
    conflicts: Tuple[Conflict, ...]
    images: Mapping[str, str] = field(default_factory=dict)
    runtime: Mapping[str, Any] = field(default_factory=dict)
    tool_parsers: Tuple[str, ...] = ()
    reasoning_parsers: Tuple[str, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EngineRecord":
        data = dict(data or {})
        settings = dict(data.get("settings") or {})
        for name, mapping in settings.items():
            if name not in VOCABULARY:
                raise ValueError(
                    f"record {data.get('id')!r}: settings.{name} is not in the settings vocabulary"
                )
            unknown = set(mapping or {}) - MAPPING_KEYS
            if unknown:
                raise ValueError(
                    f"record {data.get('id')!r}: settings.{name} has unknown keys {sorted(unknown)}"
                )
        conflicts = []
        for item in data.get("conflicts") or []:
            for scope in ("when", "requires"):
                for name in item.get(scope) or {}:
                    if name not in VOCABULARY:
                        raise ValueError(
                            f"record {data.get('id')!r}: conflict {item.get('id')!r} {scope}.{name} is not a setting"
                        )
            conflicts.append(
                Conflict(
                    id=str(item.get("id") or f"conflict-{len(conflicts)}"),
                    when=dict(item.get("when") or {}),
                    requires=dict(item.get("requires") or {}),
                    unless_feature=item.get("unless_feature"),
                    unless_for=tuple(
                        (str(w["feature"]), str(w["model"])) for w in item.get("unless_for") or ()
                    ),
                    reason=str(item.get("reason") or ""),
                )
            )
        version = str(data.get("version") or "0")
        engine = str(data.get("engine") or "vllm")
        return cls(
            id=str(data.get("id") or f"{engine}@{version}"),
            engine=engine,
            version=version,
            upstream_vllm=data.get("upstream_vllm"),
            build=str(data.get("build") or "unknown"),
            runner=str(data.get("runner") or "v2"),
            features=frozenset(
                str(f) for f in data.get("features") or data.get("kv_features") or ()
            ),
            limits=dict(data.get("limits") or {}),
            settings=settings,
            conflicts=tuple(conflicts),
            images=dict(data.get("images") or {}),
            runtime=dict(data.get("runtime") or {}),
            tool_parsers=tuple(data.get("tool_parsers") or ()),
            reasoning_parsers=tuple(data.get("reasoning_parsers") or ()),
            raw=data,
        )

    # ------------------------------------------------------------------ identity for the grounding ladder

    @property
    def minor_key(self) -> str:
        """``(engine, upstream minor, runner)``: the engine ladder's second level (FRD-027 §9.1)."""
        base = str(self.upstream_vllm or self.version)
        parts = base.split(".")
        minor = ".".join(parts[:2]) if len(parts) >= 2 else base
        return f"{self.engine}@{minor}:{self.runner}"

    # ------------------------------------------------------------------ settings and features

    @property
    def kv_features(self) -> FrozenSet[str]:
        return self.features & KV_FEATURES

    def maps(self, setting: str) -> bool:
        mapping = self.settings.get(setting)
        return bool(mapping) and not mapping.get("unsupported", False)

    def feature_for(self, setting: str, value: Any) -> Optional[str]:
        for name, want, feature in FEATURE_GATES:
            if name == setting and value == want:
                return feature
        if setting == "weights.online" and value:
            return "online_fp8"
        if setting == "lora.max_adapters" and value:
            return "lora"
        return None

    # ------------------------------------------------------------------ limits

    def _by_device(self, key: str, device: Any, default: Any) -> Any:
        table = self.limits.get(key)
        if isinstance(table, Mapping):
            sku = getattr(device, "sku", None)
            if sku and sku in table:
                return table[sku]
            klass = getattr(device, "device_class", "cuda")
            return table.get(klass, default)
        return table if table is not None else default

    def default_fraction(self, device: Any) -> float:
        return float(self._by_device("default_gpu_memory_utilization", device, 0.90))

    def max_fraction(self, device: Any) -> float:
        return float(self._by_device("max_gpu_memory_utilization", device, 0.95))

    def planning_max_running(self) -> int:
        """The record's ``default_max_num_seqs``: the floor ``context.batching`` uses (FRD-027 §10.2)."""
        return int(self.limits.get("default_max_num_seqs") or 256)

    def engine_default_max_running(self, device: Any) -> int:
        """What the engine itself uses when ``batch.max_running`` is unset (vLLM: 1024 on >= 70 GiB cards)."""
        table = self.limits.get("engine_default_max_num_seqs") or {}
        large = float(self.limits.get("engine_default_large_memory_gib") or 70)
        memory_gib = float(getattr(device, "usable_memory_bytes", 0)) / 1024**3
        if isinstance(table, Mapping):
            if memory_gib >= large:
                return int(table.get("cuda", 1024))
            return int(table.get("cuda_small", 256))
        return int(table or 256)

    def default_chunk(self, device: Any) -> int:
        return int(self._by_device("default_max_num_batched_tokens", device, 8192))

    def capture_max(self) -> int:
        return int(self.limits.get("cudagraph_capture_max") or 512)

    def max_spec_tokens(self, method: str) -> int:
        table = self.limits.get("max_spec_tokens") or {}
        return int(table.get(method, 3)) if isinstance(table, Mapping) else int(table or 3)

    def spec_prefix_cache_safe(self, model: Any = None) -> bool:
        """Whether speculative decoding may keep prefix caching for this model on this build (vLLM #53912): the
        build claims it for every model, or for full-attention models and this is one."""
        if "spec_prefix_cache_safe" in self.features:
            return True
        return "spec_prefix_cache_safe_full_attention" in self.features and "full_attention" in model_traits(
            model
        )

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.raw)


def model_traits(model: Any) -> Set[str]:
    """The model traits a conflict waiver can name (``unless_for``)."""
    traits = set()
    if model is not None and getattr(model, "full_attention", False):
        traits.add("full_attention")
    return traits


def conflicts_for(
    record: EngineRecord,
    settings: Mapping[str, Any],
    resolved: Mapping[str, Any],
    model: Any = None,
) -> List[Tuple[Conflict, str]]:
    out = []
    for conflict in record.conflicts:
        if conflict.waived(record.features, model):
            continue
        if not conflict.matches(settings):
            continue
        broken = conflict.violated(settings, resolved)
        if broken:
            out.append((conflict, broken))
    return out
