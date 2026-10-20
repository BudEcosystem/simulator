"""Plan inputs and module outputs (FRD-025 §6.1, amended by FRD-027 Appendix B).

``Intent`` is everything budapp sends about the deployment (FRD-025 §10, FRD-027 §10.1). ``PlanContext`` is what
a module reads: intent, model facts, device facts, the predictor, the engine record, the placement (stage 2 only),
node facts, the KV plan and the outputs of earlier modules. A module returns a ``Fragment``: engine-neutral
settings (resolved to args by the composer), direct args/env only for keys the vocabulary lacks, resources,
volumes, containers, pod fields, checks, alternatives, mutable bounds and decisions -- each with provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..device import DeviceFacts
from ..engine.record import EngineRecord
from ..model import ModelFacts
from ..predict import Predictor


@dataclass
class SpeculativeIntent:
    enabled: bool = False
    method: Optional[str] = None  # mtp | eagle3 | ngram
    num_speculative_tokens: Optional[int] = None
    draft_model_id: Optional[str] = None
    draft_path: Optional[str] = None


@dataclass
class PrecisionCandidate:
    id: str  # registry model id, or "online:<fmt>"
    format: str
    path: Optional[str] = None  # model path served (None for online conversion)
    relation: str = "quantized"  # base_model_relation of the variant
    config: Optional[Dict[str, Any]] = None


@dataclass
class Intent:
    model_id: Optional[str] = None
    model_path: Optional[str] = None
    family: Optional[str] = None
    base_model_relation: str = "base"
    input_tokens: float = 1024
    output_tokens: float = 256
    input_tokens_max: Optional[int] = None
    output_tokens_max: Optional[int] = None
    concurrency: int = 1
    target_ttft_ms: Optional[float] = None
    target_throughput_per_user: Optional[float] = None
    target_e2e_s: Optional[float] = None
    hardware_mode: str = "dedicated"
    enable_tool_calling: bool = False
    enable_reasoning: bool = False
    tool_calling_parser_type: Optional[str] = None
    reasoning_parser_type: Optional[str] = None
    chat_template: Optional[str] = None
    max_context_length: Optional[int] = None  # the user's request ceiling (FRD-027 §10.1)
    model_max_context_length: Optional[int] = None  # the model's own maximum (FRD-025 §10)
    max_loras: Optional[int] = None  # None: not asked (legacy behaviour: supports_lora decides)
    lora_rank: Optional[int] = None
    adapters: List[Dict[str, Any]] = field(default_factory=list)
    weight_precision: str = "auto"  # auto | original | <variant id>
    precision_candidates: List[PrecisionCandidate] = field(default_factory=list)
    enable_kv_fp8: Optional[bool] = None
    speculative: SpeculativeIntent = field(default_factory=SpeculativeIntent)
    sampling: str = "sampled"  # greedy | sampled | unknown
    text_only: Optional[bool] = None
    autoscaling: Optional[Dict[str, Any]] = None
    workload_profile: Optional[str] = None  # FRD-023 profile (agentic, chat, ...)

    @property
    def sampling_class(self) -> str:
        return "greedy" if self.sampling == "greedy" else "sampled"


@dataclass
class Placement:
    """The search result for one node group (stage 2)."""

    device_type: str  # cuda | cpu | hpu
    tp: int = 1
    pp: int = 1
    replicas: int = 1
    concurrency: int = 1  # per-replica concurrency c the search chose
    target_node: Optional[str] = None
    raw_name: Optional[str] = None
    device_name: Optional[str] = None
    device_model: Optional[str] = None
    hardware_mode: str = "dedicated"
    # physics terms budsim computed for the node group (decimal GB, as NodeGroupConfiguration ships them)
    group: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Value:
    value: Any
    module: str
    reason: str = ""


@dataclass
class Fragment:
    settings: Dict[str, Value] = field(default_factory=dict)
    args: Dict[str, Value] = field(
        default_factory=dict
    )  # engine container, keys the vocabulary lacks
    env: Dict[str, Value] = field(default_factory=dict)
    resources: Dict[str, Dict[str, Value]] = field(
        default_factory=dict
    )  # {"requests": {...}, "limits": {...}}
    volumes: List[Dict[str, Any]] = field(default_factory=list)
    mounts: List[Dict[str, Any]] = field(default_factory=list)
    ports: List[Dict[str, Any]] = field(default_factory=list)
    probes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    containers: List[Dict[str, Any]] = field(default_factory=list)  # sidecars
    pod: Dict[str, Value] = field(
        default_factory=dict
    )  # labels.X, annotations.X, node_selector.X, scheduler, ...
    engine: Dict[str, Any] = field(
        default_factory=dict
    )  # image, command, trailing args, pull policy
    checks: List[Dict[str, Any]] = field(default_factory=list)
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    mutable: Dict[str, Any] = field(default_factory=dict)
    predicted: Optional[Dict[str, Any]] = None
    decisions: List[str] = field(default_factory=list)
    overrides: Tuple[str, ...] = ()

    # builder helpers -------------------------------------------------------
    def set(self, name: str, value: Any, module: str, reason: str = "") -> "Fragment":
        self.settings[name] = Value(value, module, reason)
        return self

    def arg(self, key: str, value: Any, module: str, reason: str = "") -> "Fragment":
        self.args[key] = Value(value, module, reason)
        return self

    def envvar(self, key: str, value: Any, module: str, reason: str = "") -> "Fragment":
        self.env[key] = Value(str(value), module, reason)
        return self

    def resource(
        self, kind: str, name: str, value: Any, module: str, reason: str = ""
    ) -> "Fragment":
        self.resources.setdefault(kind, {})[name] = Value(value, module, reason)
        return self

    def podfield(self, path: str, value: Any, module: str, reason: str = "") -> "Fragment":
        self.pod[path] = Value(value, module, reason)
        return self


@dataclass
class Draft:
    """The spec as composed so far (read-only for modules)."""

    settings: Dict[str, Value] = field(default_factory=dict)
    args: Dict[str, Value] = field(default_factory=dict)
    env: Dict[str, Value] = field(default_factory=dict)
    pod: Dict[str, Value] = field(default_factory=dict)
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    checks: List[Dict[str, Any]] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    stage1: Dict[str, Any] = field(
        default_factory=dict
    )  # module -> structured output (profile, effects)

    def setting(self, name: str, default: Any = None) -> Any:
        v = self.settings.get(name)
        return v.value if v is not None else default


@dataclass
class PlanContext:
    stage: str  # numerics | capacity
    intent: Intent
    model: ModelFacts
    model_config: Dict[str, Any]
    record: EngineRecord
    device: DeviceFacts
    predictor: Predictor
    placement: Optional[Placement] = None
    node: Dict[str, Any] = field(default_factory=dict)
    kv_plan: Optional[Dict[str, Any]] = None
    draft: Draft = field(default_factory=Draft)
    module_versions: Dict[str, str] = field(default_factory=dict)
    validation: Any = None  # ValidationStore
    drafts: Dict[str, ModelFacts] = field(default_factory=dict)  # draft path -> facts
    namespace_binding: str = "$(BUD_SERVED_NAME)"
    cluster: Dict[str, Any] = field(
        default_factory=dict
    )  # renderer-known cluster facts budsim has (hami, registries)

    def version_of(self, module: str, default: str = "v1") -> str:
        return self.module_versions.get(module, self.module_versions.get("*", default))

    def workload(self, concurrency: Optional[int] = None):
        from ..engine import Workload

        i = self.intent
        return Workload(
            input_tokens=float(i.input_tokens or 1),
            output_tokens=float(i.output_tokens or 1),
            concurrency=int(
                concurrency
                or (self.placement.concurrency if self.placement else i.concurrency)
                or 1
            ),
            input_tokens_max=i.input_tokens_max,
            output_tokens_max=i.output_tokens_max,
            sampling=i.sampling_class,
            workload_class="real",
            arrival="steady",
        )


def lookup(mapping: Mapping[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = mapping
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur
