"""Runtime spec v1 (FRD-025 §7; FRD-027 Appendix B): the fully resolved, declarative pod one node group runs.

budsim emits it on ``NodeGroupConfiguration.runtime_spec``; budcluster renders it. Values are final (byte
counts, port numbers, argument values); unknown optional keys are ignored and unknown required things --
a topology, check kind or primitive not in ``requires`` support -- are refused by the renderer, never
half-applied. ``schema/v1.json`` is generated from these models (``python -m llm_memory_calculator.runtime.schema``).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

SPEC_VERSION = 1

#: Renderer primitives a spec may require (FRD-025 §7.2). budcluster publishes the set it supports.
PRIMITIVES = (
    "container.args",
    "container.env",
    "container.command",
    "resources.gpu",
    "resources.gpumem",
    "resources.memory",
    "resources.cpu",
    "volume.emptyDir",
    "volume.emptyDir.memory",
    "volume.pvc",
    "volume.model_store",
    "volume.configMap",
    "volume.secret",
    "volume.hostPath",
    "pod.scheduler",
    "pod.runtime_class",
    "pod.affinity",
    "sidecar",
    "check.node_memory",
    "check.pin_node",
    "check.hami_slice",
    "check.device_count",
    "check.storage_access_mode",
    "check.crd_present",
    "check.boot_failed",
    "kv.node_pvc",
)
CHECK_KINDS = (
    "node_memory",
    "pin_node",
    "hami_slice",
    "device_count",
    "storage_access_mode",
    "crd_present",
    "boot_failed",
)
TOPOLOGIES = ("single", "multi_node", "prefill_decode")


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class EngineRef(_Model):
    name: str
    version: str
    record: str
    image: str
    pull_policy: str = "IfNotPresent"
    arg_style: str = "--{key}={value}"
    bool_style: str = "flag"
    trailing_args: List[str] = Field(default_factory=list)
    # The record's runtime.boot_failure_patterns ({class: [regex, ...]}): the renderer tries them before its own.
    boot_failure_patterns: Optional[Dict[str, List[str]]] = None


class Topology(_Model):
    kind: Literal["single", "multi_node", "prefill_decode"] = "single"


class Port(_Model):
    name: str
    port: int
    protocol: str = "TCP"


class Probe(_Model):
    http: Optional[str] = None
    port: Optional[Union[str, int]] = None
    exec: Optional[List[str]] = None
    initial_delay_s: int = 0
    period_s: int = 10
    timeout_s: int = 1
    failure_threshold: int = 3
    success_threshold: int = 1


class Resources(_Model):
    requests: Dict[str, Union[int, str]] = Field(default_factory=dict)
    limits: Dict[str, Union[int, str]] = Field(default_factory=dict)


class Mount(_Model):
    volume: str
    path: str
    read_only: bool = False


class Volume(_Model):
    name: str
    kind: Literal["emptyDir", "pvc", "model_store", "configMap", "secret", "hostPath"]
    medium: Optional[str] = None
    size_mib: Optional[int] = None
    claim: Optional[str] = None
    path: Optional[str] = None
    config_map: Optional[str] = None
    default_mode: Optional[int] = None


class Container(_Model):
    role: Literal["engine", "sidecar"]
    name: str
    image: str
    command: List[str] = Field(default_factory=list)
    args: Dict[str, Any] = Field(
        default_factory=dict
    )  # keyed, ordered; flattened with engine.arg_style
    positional: List[str] = Field(default_factory=list)
    raw_args: List[str] = Field(default_factory=list)  # a sidecar's literal argv
    env: Dict[str, str] = Field(default_factory=dict)
    ports: List[Port] = Field(default_factory=list)
    resources: Resources = Field(default_factory=Resources)
    mounts: List[Mount] = Field(default_factory=list)
    probes: Dict[str, Probe] = Field(default_factory=dict)
    image_pull_policy: Optional[str] = None


class Pod(_Model):
    labels: Dict[str, str] = Field(default_factory=dict)
    annotations: Dict[str, str] = Field(default_factory=dict)
    node_selector: Dict[str, str] = Field(default_factory=dict)
    pin_to_node: Optional[str] = None
    scheduler: str = "default"
    runtime_class: Optional[str] = None
    volumes: List[Volume] = Field(default_factory=list)
    affinity: Optional[Dict[str, Any]] = None
    service_account: Optional[str] = None


class Check(_Model):
    model_config = ConfigDict(extra="allow")
    kind: Literal[
        "node_memory",
        "pin_node",
        "hami_slice",
        "device_count",
        "storage_access_mode",
        "crd_present",
        "boot_failed",
    ]


class Alternative(_Model):
    when: Dict[str, Any]
    patch: Dict[str, Any] = Field(default_factory=dict)
    settings: Dict[str, Any] = Field(
        default_factory=dict
    )  # the rung's settings (for the record / boot outcome)
    decision: str
    rung: Optional[int] = None
    stock_equivalent: bool = False
    # the smallest predicted memory margin over the engine's boot checks at this rung (bytes; < 0 predicts an OOM)
    predicted_margin_bytes: Optional[int] = None


class Mutable(_Model):
    adapters: Optional[Dict[str, Any]] = None
    replicas: Optional[Dict[str, int]] = None


class PredictorRef(_Model):
    version: str
    grounding: str


class Predicted(_Model):
    ttft_ms: List[float]
    tpot_ms: List[float]
    throughput_tok_s: List[float]
    boot_s: List[float]
    point: Dict[str, float] = Field(default_factory=dict)
    levels: Dict[str, str] = Field(default_factory=dict)
    ungrounded: bool = False
    concurrency: Optional[int] = None
    penalty_vs_dedicated: Optional[float] = None
    notes: List[str] = Field(default_factory=list)


class RuntimeSpec(_Model):
    version: Literal[1] = 1
    requires: List[str] = Field(default_factory=list)
    engine: EngineRef
    topology: Topology = Field(default_factory=Topology)
    replicas: int = 1
    settings: Dict[str, Any] = Field(default_factory=dict)
    predictor: Optional[PredictorRef] = None
    predicted: Optional[Predicted] = None
    pod: Pod = Field(default_factory=Pod)
    containers: List[Container]
    autoscaling: Dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
    checks: List[Check] = Field(default_factory=list)
    alternatives: List[Alternative] = Field(default_factory=list)
    mutable: Mutable = Field(default_factory=Mutable)
    decisions: List[str] = Field(default_factory=list)
    provenance: Dict[str, str] = Field(default_factory=dict)
    inputs_digest: str = ""

    def engine_container(self) -> Container:
        for c in self.containers:
            if c.role == "engine":
                return c
        raise KeyError("engine container")

    def to_wire(self) -> Dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    def canonical_json(self) -> str:
        return json.dumps(self.to_wire(), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        return "sha256:" + hashlib.sha256(self.canonical_json().encode()).hexdigest()[:16]


def flatten_args(
    args: Dict[str, Any],
    *,
    style: str = "--{key}={value}",
    bool_style: str = "flag",
    negatable: Optional[set] = None,
) -> List[str]:
    """The keyed map as argv, in order: ``True`` -> ``--key``; ``False`` -> ``--no-key`` when negatable, else
    omitted; JSON values as one compact token; ``None`` omitted."""
    out: List[str] = []
    negatable = negatable or set()
    for key, value in args.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                out.append(f"--{key}")
            elif key in negatable:
                out.append(f"--no-{key}")
            continue
        if isinstance(value, (dict, list)):
            value = json.dumps(value, separators=(",", ":"))
        if style == "--{key} {value}":
            out.extend([f"--{key}", str(value)])
        else:
            out.append(style.format(key=key, value=value))
    return out
