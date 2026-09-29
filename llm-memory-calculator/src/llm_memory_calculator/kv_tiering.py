"""KV cache tier planning (Bud FRD-023): where a deployment's KV cache should live.

``plan_kv_tiers()`` returns a :class:`KVTierPlan`. budsim ships it on each node group as
``kv_plan``, and budcluster renders it (the wire shape is FRD-023 §8.1). This release plans:

* **T0**, GPU memory: the dedicated-card fraction, a shared slice's KV growth toward the working
  set (FR-T0-3), and a CPU engine's KV space (FR-T0-4).
* **T1**, pod host memory through vLLM's OffloadingConnector, in the host-memory mode the node
  supports (FR-T1-*).

Later tiers (node storage, cluster pool, prefill/decode) come in later releases of this package,
never as configuration: what budsim can plan is fixed by the version it pins.

Every input is something the deployment already has (FR-PLAN-3), every choice is recorded in
``decisions`` (FR-PLAN-7), and the same inputs always give the same plan (NFR-6). An unknown
capability fact is never read as "available".

**Working set.** Estimated analytically from per-profile reuse assumptions (``PROFILE_REUSE``):
sessions alive per concurrency slot times the reusable part of each prompt. GenZ's ``RadixCache``
can't model it yet (it stops inserting when full instead of evicting), so a trace simulation is a
later refinement. The assumptions are defaults to be measured (``[MEASURE]``), and measured hit
rates don't feed back yet (FRD §12).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from .calculator import ModelMemoryCalculator
from .parameter_counter import UniversalParameterCounter

GIB = 1 << 30
PLAN_VERSION = 1

# ------------------------------------------------------------------------------------------ T0
# A dedicated card's KV pool gets what the plan doesn't need, up to this fraction (FR-T0-1; the
# same floor budcluster applies).
T0_GPU_UTIL = 0.92
GPU_UTIL_MAX = 0.94
# Non-torch memory vLLM counts inside its fraction (CUDA context, graph pool). [MEASURE] per image.
DEVICE_OVERHEAD_GIB = 1.0
# FR-T0-3: a shared slice's KV part grows by at most this share of the GPU's free memory at plan
# time. [MEASURE]
SLICE_KV_SHARE_MAX = 0.5
# FR-T0-4: a CPU engine's KV space grows by at most this share of the node's free host memory.
# [MEASURE]
CPU_KV_SHARE_MAX = 0.5

# ------------------------------------------------------------------------------------------ T1
# FR-T1-1: T1 = clamp(working set, 3x, 10x the GPU KV pool), within the node's free host memory.
T1_MIN_POOL_MULTIPLE = 3.0
T1_MAX_POOL_MULTIPLE = 10.0
# At most this share of the node's free host memory, shared by the replicas pinned to the node.
# [MEASURE]
T1_HOST_RAM_SHARE = 0.5
# Offload chunk, in tokens. Spike S2 (H100): moving from 16 to 256 took a 32K-token T1 hit from
# 5.5x to 7.9x faster than recompute.
T1_BLOCK_SIZE = 256
# A T1 load reaches this share of the node's pinned-copy bandwidth (S2: ~47 of 55 GB/s).
T1_LOAD_EFFICIENCY = 0.85
# Break-even: T1 must load a prefix in under this share of the time recomputing it takes.
T1_BREAK_EVEN_RATIO = 0.8
# Pinned-copy bandwidth (GB/s) by PCIe generation, when the node has no measurement.
PCIE_GBPS_BY_GEN = {3: 12.0, 4: 25.0, 5: 50.0, 6: 100.0}
DEFAULT_PINNED_GBPS = 12.0  # neither measured nor a known link
# Share of peak dense FLOPs a prefill reaches. [MEASURE]
PREFILL_MFU = 0.45
# The host-memory modes, and the engine feature each needs (FR-T1-2).
T1_MODE_FEATURES = {"register": "cpu_offload_shm", "alloc": "cpu_offload_host_alloc"}

# ------------------------------------------------------------------------------------ workload
# profile -> (sessions alive per concurrency slot within the reuse window, reusable share of
# each prompt). [MEASURE]
PROFILE_REUSE: Dict[str, Tuple[float, float]] = {
    "agentic": (4.0, 0.95),  # tool definitions and the whole history come back every turn
    "chat": (6.0, 0.8),  # multi-turn history
    "interactive": (6.0, 0.8),  # chat with a tight TTFT target
    "rag": (3.0, 0.3),  # a shared system prompt; retrieved documents vary
}
PREFIX_HEAVY_RATIO = 8.0  # input:output at or above this is prefix-heavy (ARCHITECTURE §10.2)
INTERACTIVE_TTFT_MS = 500.0

# --------------------------------------------------------------------------------------- dtype
# fp8 KV only where validated (FR-QUANT-2), as (model_type, GPU generation). Empty until budeval
# publishes its verdicts.
FP8_KV_VALIDATED: FrozenSet[Tuple[str, str]] = frozenset()

EVENTS_ENDPOINT = "tcp://*:5557"
EVENTS_REPLAY = "tcp://*:5558"


# ================================================================================== inputs


@dataclass(frozen=True)
class KVWorkload:
    """The reuse the planner assumes, inferred from the deployment's own fields (FR-PLAN-3)."""

    profile: str  # off | agentic | chat | interactive | rag
    input_tokens: int
    output_tokens: int
    concurrency: int
    sessions_per_slot: float = 0.0
    reusable_share: float = 0.0
    reason: str = ""

    @classmethod
    def from_deployment(
        cls,
        *,
        input_tokens: int,
        output_tokens: int,
        concurrency: int,
        is_embedding: bool = False,
        enable_tool_calling: bool = False,
        target_ttft_ms: Optional[float] = None,
    ) -> "KVWorkload":
        """Infer the reuse profile from the deployment's own fields."""
        input_tokens = max(int(input_tokens or 0), 0)
        output_tokens = max(int(output_tokens or 0), 0)
        concurrency = max(int(concurrency or 1), 1)
        if is_embedding:
            profile, reason = "off", "embedding or pooling model: no reusable KV cache"
        elif enable_tool_calling:
            profile, reason = (
                "agentic",
                "tool calling on: tool definitions and history are a shared prefix",
            )
        elif output_tokens and input_tokens / output_tokens >= PREFIX_HEAVY_RATIO:
            profile, reason = "rag", f"input:output {input_tokens}:{output_tokens} is prefix-heavy"
        elif target_ttft_ms is not None and target_ttft_ms <= INTERACTIVE_TTFT_MS:
            profile, reason = "interactive", f"TTFT target {target_ttft_ms:g} ms"
        else:
            profile, reason = "chat", "default profile"
        sessions, share = PROFILE_REUSE.get(profile, (0.0, 0.0))
        return cls(profile, input_tokens, output_tokens, concurrency, sessions, share, reason)

    @property
    def reusable_prefix_tokens(self) -> int:
        """Tokens of a typical prompt that another request can reuse (the p50 prefix)."""
        return int(self.input_tokens * self.reusable_share)

    @property
    def working_set_tokens(self) -> int:
        """Distinct reusable tokens one replica should keep to serve its reuse from cache."""
        return int(self.concurrency * self.sessions_per_slot * self.reusable_prefix_tokens)


@dataclass(frozen=True)
class NodeKVFacts:
    """A node's capability facts (FRD-023 §8.2). None is unknown, never available."""

    pinned_offload_ok: Optional[bool] = None
    host_pointer_for_registered_mem: Optional[bool] = None
    vgpu: Optional[bool] = None
    pinned_copy_gbps: Optional[float] = None
    pcie_gen: Optional[int] = None

    @classmethod
    def from_capabilities(cls, caps: Optional[Mapping[str, Any]]) -> "NodeKVFacts":
        """Read the facts from a node's ``kv_capabilities``; anything malformed is unknown."""
        caps = caps or {}

        def flag(key: str) -> Optional[bool]:
            value = caps.get(key)
            return value if isinstance(value, bool) else None

        def number(key: str) -> Optional[float]:
            value = caps.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                return None
            return float(value) if value > 0 else None

        gen = number("pcie_gen")
        return cls(
            pinned_offload_ok=flag("pinned_offload_ok"),
            host_pointer_for_registered_mem=flag("host_pointer_for_registered_mem"),
            vgpu=flag("vgpu"),
            pinned_copy_gbps=number("pinned_copy_gbps"),
            pcie_gen=int(gen) if gen else None,
        )

    def pinned_gbps(self) -> Tuple[float, str]:
        """Return the host-to-device bandwidth (GB/s) to plan with, and where it came from."""
        if self.pinned_copy_gbps:
            return self.pinned_copy_gbps, "measured"
        if self.pcie_gen in PCIE_GBPS_BY_GEN:
            return PCIE_GBPS_BY_GEN[self.pcie_gen], f"PCIe Gen{self.pcie_gen} default"
        return DEFAULT_PINNED_GBPS, "unknown link, conservative default"


# ================================================================================== output


@dataclass
class KVTierPlan:
    """A deployment's KV plan. :meth:`to_wire` is the FRD-023 §8.1 shape budcluster parses."""

    profile: str = "off"
    workload_profile: str = "off"
    kv_cache_dtype: str = "auto"
    gpu_memory_utilization: Optional[float] = None
    gpu_kv_gib: Optional[float] = None
    cpu_kv_gib: Optional[float] = None
    tiers: List[Dict[str, Any]] = field(default_factory=list)
    events_enabled: bool = False
    routing: Dict[str, Any] = field(default_factory=dict)
    predicted_hit_rate: Optional[float] = None
    predicted_ttft_ms: Optional[float] = None
    host_memory_gib: float = 0.0
    shm_gib: float = 0.0
    working_set_gib: float = 0.0
    gpu_pool_gib: float = 0.0
    decisions: List[str] = field(default_factory=list)

    def to_wire(self) -> Dict[str, Any]:
        """Return the plan as budsim ships it on ``NodeGroupConfiguration.kv_plan``."""
        return {
            "version": PLAN_VERSION,
            "namespace": None,
            "profile": self.profile,
            "kv_cache_dtype": self.kv_cache_dtype,
            "skip_layers_sliding_window": False,
            "hash_algo": "sha256",
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "gpu_kv_gib": self.gpu_kv_gib,
            "cpu_kv_gib": self.cpu_kv_gib,
            "tiers": [dict(t) for t in self.tiers],
            "events": {
                "enabled": self.events_enabled,
                "endpoint": EVENTS_ENDPOINT,
                "replay": EVENTS_REPLAY,
            },
            "routing": dict(self.routing),
            "pd": {"enabled": False, "role": None},
            "predicted": {"hit_rate": self.predicted_hit_rate, "ttft_ms": self.predicted_ttft_ms},
            "resources": {
                "host_memory_gib": self.host_memory_gib,
                "shm_gib": self.shm_gib,
                "ipc_lock": False,
                "rdma": False,
            },
            "decisions": list(self.decisions),
        }

    def as_dict(self) -> Dict[str, Any]:
        """Return every field, for logging."""
        return asdict(self)


# ================================================================================ planning


def kv_bytes_per_token_per_rank(
    model_config: Mapping[str, Any],
    *,
    seq_length: int,
    tensor_parallel: int = 1,
    precision: str = "bf16",
    engine_capabilities: Any = None,
    data_parallel_attention: bool = False,
) -> Tuple[float, float, str]:
    """Return ``(bytes per token per rank, bytes per token for the whole model, attention type)``.

    From :meth:`ModelMemoryCalculator.kv_cache_breakdown`, the one source of per-token KV
    (FR-PLAN-2). An MLA latent is not split by head-wise tensor parallelism -- every rank holds the
    whole latent -- unless attention is data-parallel.
    """
    breakdown = ModelMemoryCalculator().kv_cache_breakdown(
        dict(model_config),
        batch_size=1,
        seq_length=max(int(seq_length), 1),
        precision=precision,
        engine_capabilities=engine_capabilities,
    )
    total = float(breakdown.get("marginal_bytes_per_token") or 0.0)
    attention = str(breakdown.get("attention_type") or "")
    tp = max(int(tensor_parallel or 1), 1)
    replicated = attention == "mla" and not data_parallel_attention
    per_rank = total if replicated else total / tp
    return per_rank, total, attention


def prefill_seconds(
    model_config: Mapping[str, Any], tokens: int, device_tflops: Optional[float]
) -> Optional[float]:
    """Estimate one prefill of ``tokens`` on the whole model, or None without the device's FLOPs.

    Dense matmuls (2 x parameters per token) plus causal attention (2 x layers x tokens^2 x hidden).
    """
    if not device_tflops or device_tflops <= 0 or tokens <= 0:
        return None
    params = UniversalParameterCounter().count_parameters(dict(model_config))
    layers = int(model_config.get("num_hidden_layers") or model_config.get("n_layer") or 0)
    hidden = int(model_config.get("hidden_size") or model_config.get("n_embd") or 0)
    flops = 2.0 * params * tokens + 2.0 * layers * tokens * tokens * hidden
    return flops / (device_tflops * 1e12 * PREFILL_MFU)


def _gib(value: Optional[float]) -> float:
    return float(value) if value and value > 0 else 0.0


def _round_gib(value: float) -> float:
    return math.ceil(value * 100) / 100


def plan_kv_tiers(
    *,
    model_config: Mapping[str, Any],
    engine: str,
    engine_kv_features: Optional[Sequence[str]],
    device_type: str,
    hardware_mode: str = "dedicated",
    device_memory_gib: Optional[float] = None,
    weight_gib_per_rank: float = 0.0,
    kv_demand_gib_per_rank: float = 0.0,
    other_device_gib_per_rank: float = 0.0,
    tensor_parallel: int = 1,
    pipeline_parallel: int = 1,
    replicas: int = 1,
    workload: KVWorkload,
    node: Optional[Mapping[str, Any]] = None,
    host_free_gib: Optional[float] = None,
    device_free_gib: Optional[float] = None,
    gpu_kv_cap_gib: Optional[float] = None,
    device_tflops: Optional[float] = None,
    device_generation: Optional[str] = None,
    precision: str = "bf16",
    engine_capabilities: Any = None,
    data_parallel_attention: bool = False,
) -> KVTierPlan:
    """Plan a deployment's KV tiers (FRD-023 §5.1, ARCHITECTURE §10.3).

    Sizes are GiB. ``*_per_rank`` values are what budsim already sized for one rank (weights, the KV
    demand of the deployment's concurrency and context, and everything else on the device).
    ``host_free_gib`` is the target node's free host memory and ``device_free_gib`` the free memory
    of the GPU a shared slice lands on. ``gpu_kv_cap_gib`` is the most KV per rank a grown slice may
    hold, from the caller's placement check (budsim: every replica's slice still fits a card).
    ``node`` is that node's ``kv_capabilities``. A missing input never enables anything.
    """
    plan = KVTierPlan(workload_profile=workload.profile)
    decisions = plan.decisions
    decisions.append(f"profile: {workload.profile} ({workload.reason})")
    features = frozenset(engine_kv_features or ())
    tp = max(int(tensor_parallel or 1), 1)
    replicas = max(int(replicas or 1), 1)

    if workload.profile == "off":
        return plan
    if engine != "vllm":
        decisions.append(f"no KV plan: engine '{engine}' (vLLM only in this release)")
        return plan
    if not features:
        decisions.append("no KV plan: the engine record lists no KV features")
        return plan

    plan.profile = "native"
    plan.events_enabled = "kv_events" in features
    plan.routing = {
        "strategy": "prefix-cache",
        "affinity": "max" if workload.profile == "agentic" else "capped",
        "load_factor": 1.25,
    }

    # -- dtype (FR-QUANT-2): fp8 only where validated.
    model_type = str(model_config.get("model_type") or "")
    if (model_type, str(device_generation or "")) in FP8_KV_VALIDATED:
        plan.kv_cache_dtype = "fp8"
        decisions.append(f"dtype: fp8 (validated for {model_type} on {device_generation})")
    else:
        decisions.append(
            f"dtype: model default (no validated fp8 entry for {model_type or 'this model'})"
        )
    kv_precision = "fp8" if plan.kv_cache_dtype == "fp8" else precision

    per_rank_bytes, per_token_bytes, attention = kv_bytes_per_token_per_rank(
        model_config,
        seq_length=workload.input_tokens + workload.output_tokens,
        tensor_parallel=tp,
        precision=kv_precision,
        engine_capabilities=engine_capabilities,
        data_parallel_attention=data_parallel_attention,
    )
    if per_rank_bytes <= 0:
        decisions.append("no KV plan: the model has no per-token KV cache")
        plan.profile = "off"
        return plan
    if attention == "mla" and tp > 1 and not data_parallel_attention:
        decisions.append("KV: MLA latent counted on every tensor-parallel rank")

    working_set_gib = workload.working_set_tokens * per_rank_bytes / GIB
    plan.working_set_gib = _round_gib(working_set_gib)
    demand = _gib(kv_demand_gib_per_rank)
    decisions.append(
        f"working set: {workload.working_set_tokens} tokens ({plan.working_set_gib} GiB per rank)"
        f" from {workload.concurrency} x {workload.sessions_per_slot:g} sessions"
        f" x {workload.reusable_prefix_tokens} reusable tokens"
    )

    # -- T0
    pool_gib = _plan_t0(
        plan,
        device_type=device_type,
        hardware_mode=hardware_mode,
        device_memory_gib=_gib(device_memory_gib),
        weight_gib=_gib(weight_gib_per_rank),
        other_gib=_gib(other_device_gib_per_rank),
        demand_gib=demand,
        working_set_gib=working_set_gib,
        device_free_gib=device_free_gib,
        gpu_kv_cap_gib=gpu_kv_cap_gib,
        host_free_gib=host_free_gib,
        replicas=replicas,
        tp=tp,
    )
    plan.gpu_pool_gib = _round_gib(pool_gib)

    # -- T1
    t1_gib = 0.0
    if working_set_gib <= pool_gib:
        decisions.append("no offload tier: the working set fits the T0 KV pool (FR-PLAN-5)")
    else:
        t1_gib = _plan_t1(
            plan,
            model_config=model_config,
            device_type=device_type,
            features=features,
            facts=NodeKVFacts.from_capabilities(node),
            pool_gib_total=pool_gib * tp,
            working_set_gib_total=working_set_gib * tp,
            per_token_bytes=per_token_bytes,
            prefix_tokens=workload.reusable_prefix_tokens,
            host_free_gib=host_free_gib,
            device_tflops=device_tflops,
            replicas=replicas,
            pipeline_parallel=pipeline_parallel,
        )

    # -- prediction (FR-PLAN-8): informational; nothing sizes from it until spike S6.
    _predict(
        plan,
        workload,
        per_token_bytes,
        per_rank_bytes,
        pool_gib,
        t1_gib,
        tp,
        node,
        model_config,
        device_tflops,
    )
    return plan


def _plan_t0(
    plan: KVTierPlan,
    *,
    device_type: str,
    hardware_mode: str,
    device_memory_gib: float,
    weight_gib: float,
    other_gib: float,
    demand_gib: float,
    working_set_gib: float,
    device_free_gib: Optional[float],
    gpu_kv_cap_gib: Optional[float],
    host_free_gib: Optional[float],
    replicas: int,
    tp: int,
) -> float:
    """Decide the T0 KV pool per rank, in GiB, and record how."""
    decisions = plan.decisions
    if device_type in ("cpu", "cpu_high"):
        # FR-T0-4: a CPU engine's KV pool is host memory.
        cap = CPU_KV_SHARE_MAX * _gib(host_free_gib) / (replicas * tp) if host_free_gib else 0.0
        if working_set_gib > demand_gib and cap > 0:
            grown = demand_gib + min(working_set_gib - demand_gib, cap)
            plan.cpu_kv_gib = _round_gib(grown)
            decisions.append(
                f"T0 (CPU): KV space {demand_gib:.2f} -> {plan.cpu_kv_gib} GiB"
                " toward the working set"
            )
            return grown
        if working_set_gib > demand_gib:
            decisions.append("T0 (CPU): KV space stays at demand (free host memory unknown)")
        return demand_gib
    if device_type != "cuda":
        decisions.append(f"T0: no headroom on '{device_type}'")
        return demand_gib
    if hardware_mode == "shared":
        # FR-T0-3: grow the slice's KV part toward the working set, by at most a share of the GPU's
        # free memory.
        if working_set_gib <= demand_gib:
            decisions.append("T0 (slice): the working set fits the demand-sized slice")
            return demand_gib
        if not device_free_gib or device_free_gib <= 0:
            decisions.append("T0 (slice): no growth (the GPU's free memory is unknown)")
            return demand_gib
        # Replicas pinned to the node may land on the same card, so they share the cap.
        growth = min(working_set_gib - demand_gib, SLICE_KV_SHARE_MAX * device_free_gib / replicas)
        room = (
            GPU_UTIL_MAX * device_memory_gib
            - weight_gib
            - other_gib
            - DEVICE_OVERHEAD_GIB
            - demand_gib
        )
        growth = max(0.0, min(growth, room)) if device_memory_gib else growth
        if growth <= 0:
            decisions.append("T0 (slice): no room on the card to grow the slice")
            return demand_gib
        if gpu_kv_cap_gib is not None and demand_gib + growth > gpu_kv_cap_gib:
            # The cap comes in 0.01 GiB steps; rounding up past it would build a slice that doesn't fit.
            capped = math.floor(gpu_kv_cap_gib * 100 + 1e-6) / 100
            if capped <= demand_gib:
                decisions.append(
                    "T0 (slice): no growth (a grown slice per replica doesn't fit the free cards)"
                )
                return demand_gib
            plan.gpu_kv_gib = capped
            decisions.append(
                f"T0 (slice): KV {demand_gib:.2f} -> {capped} GiB, as much as still fits one slice"
                f" per replica ({replicas}) on the free cards"
            )
            return capped
        plan.gpu_kv_gib = _round_gib(demand_gib + growth)
        decisions.append(
            f"T0 (slice): KV {demand_gib:.2f} -> {plan.gpu_kv_gib} GiB,"
            f" at most {SLICE_KV_SHARE_MAX:.0%} of "
            f"the GPU's {device_free_gib:.1f} GiB free, shared by {replicas} replica(s)"
        )
        return demand_gib + growth
    # FR-T0-1: a dedicated card's leftover memory goes to the KV pool.
    plan.gpu_memory_utilization = T0_GPU_UTIL
    pool = demand_gib
    if device_memory_gib:
        pool = max(
            demand_gib,
            T0_GPU_UTIL * device_memory_gib - weight_gib - other_gib - DEVICE_OVERHEAD_GIB,
        )
    decisions.append(
        f"T0 (dedicated): --gpu-memory-utilization {T0_GPU_UTIL}, KV pool ~{pool:.1f} GiB per rank"
    )
    return pool


def _t1_mode(facts: NodeKVFacts, features: FrozenSet[str]) -> Tuple[Optional[str], str]:
    """Pick T1's host-memory mode for the node, or None with the reason."""
    if facts.host_pointer_for_registered_mem is True:
        mode, why = "register", "the GPU uses registered host memory at its host address"
    elif facts.host_pointer_for_registered_mem is False:
        mode, why = (
            "alloc",
            "the GPU can't use registered host memory at its host address (vGPU with UVM off)",
        )
    elif facts.vgpu is False:
        mode, why = "register", "bare-metal GPU (registered-memory fact not probed)"
    else:
        mode, why = "alloc", "registered-memory fact not probed on a vGPU or unknown node"
    feature = T1_MODE_FEATURES[mode]
    if feature not in features:
        return None, f"T1 dropped: {mode} mode ({why}) needs the engine feature '{feature}'"
    return mode, f"T1 mode: {mode} ({why})"


def _plan_t1(
    plan: KVTierPlan,
    *,
    model_config: Mapping[str, Any],
    device_type: str,
    features: FrozenSet[str],
    facts: NodeKVFacts,
    pool_gib_total: float,
    working_set_gib_total: float,
    per_token_bytes: float,
    prefix_tokens: int,
    host_free_gib: Optional[float],
    device_tflops: Optional[float],
    replicas: int,
    pipeline_parallel: int,
) -> float:
    """Plan T1 (FR-T1-*): its size in GiB, total across ranks, or 0 with the reason."""
    decisions = plan.decisions
    if device_type != "cuda":
        decisions.append(f"T1 dropped: vLLM's offloading connector isn't used on '{device_type}'")
        return 0.0
    if max(int(pipeline_parallel or 1), 1) > 1:
        decisions.append(
            "T1 dropped: a multi-node (Ray) deployment can't share one host-memory region"
        )
        return 0.0
    if facts.pinned_offload_ok is not True:
        decisions.append(
            f"T1 dropped: pinned host offload not verified on this node ({facts.pinned_offload_ok})"
        )
        return 0.0
    mode, why = _t1_mode(facts, features)
    decisions.append(why)
    if mode is None:
        return 0.0
    if not host_free_gib or host_free_gib <= 0:
        decisions.append("T1 dropped: the node's free host memory is unknown")
        return 0.0

    bandwidth, source = facts.pinned_gbps()
    load_s = prefix_tokens * per_token_bytes / (bandwidth * 1e9 * T1_LOAD_EFFICIENCY)
    recompute_s = prefill_seconds(model_config, prefix_tokens, device_tflops)
    if recompute_s is None:
        decisions.append("T1 break-even not checked: the device's FLOPs are unknown")
    elif load_s >= T1_BREAK_EVEN_RATIO * recompute_s:
        decisions.append(
            f"T1 dropped: loading a {prefix_tokens}-token prefix"
            f" ({load_s * 1e3:.0f} ms at {bandwidth:g} GB/s, "
            f"{source}) costs about as much as recomputing it ({recompute_s * 1e3:.0f} ms)"
        )
        return 0.0
    else:
        decisions.append(
            f"T1 break-even: {prefix_tokens}-token prefix loads in {load_s * 1e3:.0f} ms vs "
            f"{recompute_s * 1e3:.0f} ms to recompute ({bandwidth:g} GB/s, {source})"
        )

    floor_gib = T1_MIN_POOL_MULTIPLE * pool_gib_total
    ceiling_gib = T1_MAX_POOL_MULTIPLE * pool_gib_total
    wanted = min(max(working_set_gib_total, floor_gib), ceiling_gib)
    available = T1_HOST_RAM_SHARE * host_free_gib / replicas
    size = min(wanted, available)
    if available < wanted:
        limit = "the host memory"
    elif working_set_gib_total > ceiling_gib:
        limit = f"{T1_MAX_POOL_MULTIPLE:g}x the GPU pool"
    elif working_set_gib_total < floor_gib:
        limit = f"{T1_MIN_POOL_MULTIPLE:g}x the GPU pool"
    else:
        limit = "the working set"
    if size < pool_gib_total:
        decisions.append(
            f"T1 dropped: {available:.1f} GiB of host memory per replica is less than"
            " the GPU pool it would back"
        )
        return 0.0
    size = math.floor(size * 4) / 4  # quarter-GiB steps keep plans stable
    plan.tiers.append(
        {
            "tier": "T1",
            "backend": "cpu",
            "size_gib": size,
            "storage_class": None,
            "transport": None,
            "params": {"host_memory": mode, "block_size": T1_BLOCK_SIZE},
        }
    )
    plan.host_memory_gib = size
    plan.shm_gib = size if mode == "register" else 0.0
    decisions.append(f"T1: {size:g} GiB of pod host memory ({mode} mode), sized by {limit}")
    return size


def _predict(
    plan: KVTierPlan,
    workload: KVWorkload,
    per_token_bytes: float,
    per_rank_bytes: float,
    pool_gib: float,
    t1_gib: float,
    tp: int,
    node: Optional[Mapping[str, Any]],
    model_config: Mapping[str, Any],
    device_tflops: Optional[float],
) -> None:
    """Fill the predicted hit rate and hit-aware TTFT (FR-PLAN-8)."""
    working = workload.working_set_tokens
    if working <= 0:
        return
    pool_tokens = pool_gib * GIB / per_rank_bytes
    t1_tokens = t1_gib * GIB / per_token_bytes if t1_gib else 0.0
    share = workload.reusable_share
    gpu_hit = share * min(1.0, pool_tokens / working)
    total_hit = share * min(1.0, (pool_tokens + t1_tokens) / working)
    plan.predicted_hit_rate = round(total_hit, 3)
    prefill = prefill_seconds(
        model_config, int(workload.input_tokens * (1 - total_hit)), device_tflops
    )
    if prefill is None:
        return
    bandwidth, _ = NodeKVFacts.from_capabilities(node).pinned_gbps()
    t1_tokens_loaded = (total_hit - gpu_hit) * workload.input_tokens
    load = t1_tokens_loaded * per_token_bytes / (bandwidth * 1e9 * T1_LOAD_EFFICIENCY)
    plan.predicted_ttft_ms = round((prefill + load) * 1e3, 1)
