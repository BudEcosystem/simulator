"""budcluster's sizing arithmetic for vLLM on CUDA (deployment/memory_plan.py), ported once into the planner.

FRD-025 G3: every sizing rule exists once. These functions reproduce ``resolve_max_model_len``,
``resolve_token_budget`` and the CUDA half of ``compute_budget`` exactly (decimal-GB physics terms from budsim,
binary GiB device memory), so the v1 modules render the same pod budcluster renders today (C-GOLD-01).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

GIB = 2**30
MIB = 2**20
CONTEXT_SAFETY = 1.10
L_MIN = 512
DEFAULT_CONTEXT_TOKENS = 8192
MAX_UNVERIFIED_CONTEXT = 32768
T_PREF = 8192
SHM_PER_RANK_MIB, SHM_MIN_MIB, SHM_MAX_MIB = 256, 256, 2048
HOST_PER_RANK_MIB = 2560
DEVICE_CTX_BYTES = 1536 * MIB
FRAMEWORK_FLOOR_MIB = 3072
COMPILE_PEAK_MIB = 2048
GPU_UTIL_MIN, GPU_UTIL_MAX = 0.10, 0.94
T0_GPU_UTIL_FLOOR = 0.92
SLICE_OUTSIDE_RESERVE = 1536 * MIB
SLICE_TP_COMM_RESERVE = 1024 * MIB
T1_DEFAULT_BLOCK_SIZE = 256
T1_MODE_FEATURES = {"register": "cpu_offload_shm", "alloc": "cpu_offload_host_alloc"}
T2_FEATURE = "tiering_fs"
# FRD-023 Phase 4 (memory_plan.plan_t3): the Mooncake store connector's host buffers per rank, its config file, and the
# one transport rendered (RDMA needs FR-RENDER-9).
T3_FEATURE = "mooncake_store"
T3_HOST_MIB_PER_RANK = 1024
T3_DEFAULT_LOCAL_BUFFER_MIB = 512
T3_CONFIG_PATH = "/tmp/bud-mooncake-config.json"
T3_PROTOCOLS = ("tcp",)
_MASTER_RE = re.compile(r"^[a-z0-9]([-a-z0-9.]{0,251}[a-z0-9])?:[0-9]{1,5}$")
_KV_NAMESPACE_RE = re.compile(r"^kvns-[0-9a-f]{16}$")


class SizingError(ValueError):
    """The legacy path would refuse this deployment (DeploymentSizingError)."""


def gb(x: Any) -> int:
    return int(float(x or 0) * 1e9)


def ceil_mib(b: float) -> int:
    return int(math.ceil(float(b) / MIB))


def resolve_max_model_len(
    input_tokens: Any, output_tokens: Any, cap: Optional[int]
) -> Dict[str, Any]:
    if input_tokens and output_tokens:
        requested = int(math.ceil((float(input_tokens) + float(output_tokens)) * CONTEXT_SAFETY))
        source = "requested"
    else:
        requested, source = DEFAULT_CONTEXT_TOKENS, "default"
    if cap is None:
        if source == "default":
            raise SizingError("no token profile and no model context cap")
        resolved = min(requested, MAX_UNVERIFIED_CONTEXT)
        source = "token_profile" if resolved < requested else source
    else:
        resolved = min(requested, int(cap))
        if resolved < requested:
            source = "capped"
    return {"max_model_len": max(resolved, L_MIN), "source": source}


def resolve_token_budget(max_model_len: int, max_num_seqs: int, chunked: bool = True) -> int:
    if chunked:
        return max(min(int(max_model_len), T_PREF), int(max_num_seqs))
    return max(int(max_model_len), int(max_num_seqs))


def shm_bytes(tp: int) -> int:
    return min(max(SHM_PER_RANK_MIB * max(tp, 1), SHM_MIN_MIB), SHM_MAX_MIB) * MIB


#: The pod's memory limit (which the chart also requests) adds the weights' host copy, capped, to its request.
LIMIT_WEIGHT_CAP_MIB = 2048


def limit_headroom_mib(weight_bytes_per_rank: float, tp: int) -> int:
    """MiB a cuda pod's memory limit adds to its request: its weights' host staging copy, capped."""
    return min(ceil_mib(weight_bytes_per_rank * max(int(tp), 1)), LIMIT_WEIGHT_CAP_MIB)


def host_base_bytes(tp: int, t3: bool = False) -> int:
    """Host memory a cuda pod requests besides its T1 region and LoRA scratch, once it requests any (T1 on).

    The KV planner sizes T1 within what is left of the node's free host memory after this (``kv.tiers.t1``).
    """
    tp = max(int(tp), 1)
    return (
        tp * HOST_PER_RANK_MIB * MIB
        + FRAMEWORK_FLOOR_MIB * MIB
        + COMPILE_PEAK_MIB * MIB
        + shm_bytes(tp)
        + (T3_HOST_MIB_PER_RANK * MIB * tp if t3 else 0)
    )


def shared_slice(demand: int, reserve: int) -> int:
    return max(
        math.ceil(demand * 100 / (94 * MIB)), math.ceil((demand + reserve) * 100 / (99 * MIB))
    )


@dataclass
class CudaBudget:
    tp: int
    max_num_seqs: int
    max_model_len: int
    context_source: str
    max_num_batched_tokens: int
    kv_engine_gib: int
    device_demand: int
    device_total: int
    gpu_memory_utilization: float
    gpu_memory_mib: int
    slice_fallback_mib: Optional[int] = None
    slice_fallback_util: Optional[float] = None
    t1_bytes: int = 0
    t1_mode: Optional[str] = None
    t1_block: Optional[int] = None
    t2: Optional[Dict[str, Any]] = None
    t3: Optional[Dict[str, Any]] = None
    kv_cache_dtype: Optional[str] = None
    kv_skip_sliding: Optional[bool] = None
    engine_request_mib: int = 0
    engine_limit_mib: int = 0
    shm_mib: int = 0
    lora_warmup_bytes: int = 0
    notes: List[str] = field(default_factory=list)


def active_kv_plan(
    group: Mapping[str, Any], kv_plan: Optional[Mapping[str, Any]]
) -> Optional[Mapping[str, Any]]:
    return kv_plan if (kv_plan and group.get("kv_features")) else None


def plan_t1(
    group: Mapping[str, Any], plan: Optional[Mapping[str, Any]], gpu_block: int
) -> Dict[str, Any]:
    """FRD-023 T1 gate (memory_plan.plan_t1): the first CPU tier, cuda, generative vLLM, PP 1, a listed mode."""
    out = {"bytes": 0, "mode": None, "block": None}
    if not plan or group.get("type") != "cuda" or int(group.get("pp_size") or 1) != 1:
        return out
    if str(group.get("engine_type") or "vllm") != "vllm" or group.get("is_embedding"):
        return out
    tiers = [
        t
        for t in plan.get("tiers") or []
        if t.get("tier") == "T1" and t.get("backend", "cpu") == "cpu"
    ]
    if not tiers or not float(tiers[0].get("size_gib") or 0) > 0:
        return out
    tier = tiers[0]
    params = tier.get("params") or {}
    mode = str(params.get("host_memory") or "register")
    if T1_MODE_FEATURES.get(mode) not in set(group.get("kv_features") or ()):
        return out
    requested = int(params.get("block_size") or T1_DEFAULT_BLOCK_SIZE)
    block = int(math.ceil(requested / gpu_block) * gpu_block)
    size = int(float(tier["size_gib"]) * GIB) // MIB * MIB
    return {"bytes": size, "mode": mode, "block": block}


def plan_t3(group: Mapping[str, Any], plan: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """FRD-023 T3 gate (memory_plan.plan_t3): a Mooncake tier on cuda, generative vLLM, PP 1, the engine listing
    ``mooncake_store``, TCP, a usable master, and store keys under the plan's KV namespace."""
    if not plan:
        return None
    tier = next(
        (t for t in plan.get("tiers") or [] if t.get("tier") == "T3" and t.get("backend") == "mooncake"),
        None,
    )
    if tier is None or group.get("type") != "cuda" or int(group.get("pp_size") or 1) != 1:
        return None
    if str(group.get("engine_type") or "vllm") != "vllm" or group.get("is_embedding"):
        return None
    if T3_FEATURE not in set(group.get("kv_features") or ()):
        return None
    params = tier.get("params") or {}
    protocol = str(params.get("protocol") or tier.get("transport") or "tcp").lower()
    master = params.get("master_server_address")
    namespace = plan.get("namespace")
    if protocol not in T3_PROTOCOLS or not isinstance(master, str) or not _MASTER_RE.match(master):
        return None
    if not isinstance(namespace, str) or not _KV_NAMESPACE_RE.match(namespace):
        return None
    if params.get("cache_prefix") != namespace:
        return None
    buffer = params.get("local_buffer_size_mib")
    buffer = int(buffer) if isinstance(buffer, (int, float)) and float(buffer).is_integer() and buffer > 0 else T3_DEFAULT_LOCAL_BUFFER_MIB
    tp = max(int(group.get("tp_size") or 1), 1)
    return {
        "bytes": T3_HOST_MIB_PER_RANK * MIB * tp,
        "master": master,
        "protocol": protocol,
        "cache_prefix": namespace,
        "local_buffer_mib": buffer,
    }


def mooncake_config(t3: Mapping[str, Any]) -> Dict[str, Any]:
    """The store client's JSON config (engine_processors.mooncake_config): standalone-store, no own segment."""
    return {
        "mode": "standalone-store",
        "metadata_server": "P2PHANDSHAKE",
        "master_server_address": t3["master"],
        "global_segment_size": 0,
        "local_buffer_size": f"{t3['local_buffer_mib']}MB",
        "protocol": t3["protocol"],
        "device_name": "",
        "enable_offload": False,
    }


def plan_kv_dtype(group: Mapping[str, Any], plan: Optional[Mapping[str, Any]]) -> tuple:
    """memory_plan.plan_kv_cache_dtype, gated on the engine's ``kv_fp8`` feature."""
    if not plan or "kv_fp8" not in set(group.get("kv_features") or ()):
        return None, None
    dtype = plan.get("kv_cache_dtype")
    if not isinstance(dtype, str) or dtype == "auto":
        return None, None
    return dtype, ("sliding_window" if plan.get("skip_layers_sliding_window") is True else None)


def cuda_budget(
    group: Mapping[str, Any], intent: Mapping[str, Any], kv_plan: Optional[Mapping[str, Any]] = None
) -> CudaBudget:
    """memory_plan.compute_budget for a vLLM CUDA node group (single node)."""
    tp = max(int(group.get("tp_size") or 1), 1)
    seqs = max(
        int(group.get("concurrency") or (group.get("labels") or {}).get("concurrency") or 0), 0
    )
    if seqs <= 0:
        raise SizingError("no concurrency on the node group")
    weight = gb(group.get("weight_memory_gb"))
    if weight <= 0:
        raise SizingError("no weight term")
    device_total = int(float(group.get("device_total_memory_gb") or 0) * GIB)
    kv = gb(group.get("kv_cache_memory_gb"))
    act = gb(group.get("activation_memory_gb"))
    state = gb(group.get("state_memory_gb"))
    encoder = str(group.get("engine_type")) == "latentbud" or bool(group.get("is_embedding"))
    if not encoder and kv <= 0:
        raise SizingError("a decoder with no KV term")
    if act <= 0:
        raise SizingError("no activation term")
    ctx = resolve_max_model_len(
        intent.get("input_tokens"),
        intent.get("output_tokens"),
        intent.get("model_max_context_length"),
    )
    max_model_len = ctx["max_model_len"]
    tokens = resolve_token_budget(max_model_len, seqs, True)
    kv_engine_gib = 0 if encoder else max(1, math.ceil(kv / GIB))
    plan = active_kv_plan(group, kv_plan)
    kv_committed = kv_engine_gib * GIB
    logits = 0 if encoder else gb(group.get("sampler_logits_gb"))
    lora = 0
    if str(group.get("engine_type") or "vllm") == "vllm" and group.get("supports_lora"):
        lora = gb(group.get("lora_prefill_scratch_gb"))
        if lora == 0 and not group.get("lora_scratch_zero_by_device"):
            raise SizingError("LoRA warm-up scratch missing")
    gpu_block = int((group.get("args") or {}).get("block-size") or 16)
    t1 = plan_t1(group, plan, gpu_block)
    t3 = plan_t3(group, plan)
    kv_dtype, skip = (
        plan_kv_dtype(group, plan)
        if (str(group.get("engine_type") or "vllm") == "vllm" and not encoder)
        else (None, None)
    )
    shm = shm_bytes(tp)
    demand = weight + kv_committed + act + logits + state + DEVICE_CTX_BYTES
    engine_bytes = host_base_bytes(tp) + lora + t1["bytes"] + (t3["bytes"] if t3 else 0)
    request = ceil_mib(engine_bytes)
    limit = request + limit_headroom_mib(weight, tp)
    if not device_total:
        raise SizingError("no device_total_memory_gb on an accelerator")
    raw = demand / device_total
    shared = (
        group.get("type") == "cuda"
        and group.get("hardware_mode") == "shared"
        and str(group.get("engine_type")) != "latentbud"
    )
    fallback_mib = fallback_util = None
    if raw > GPU_UTIL_MAX and not shared:
        raise SizingError(
            f"computed gpu-memory-utilization {raw:.2f} exceeds the safe maximum {GPU_UTIL_MAX}: the plan does not "
            f"fit on this device ({demand / GIB:.1f} GiB needed of {device_total / GIB:.1f} GiB). Raise tensor "
            "parallelism or lower concurrency."
        )
    if shared:
        reserve = SLICE_OUTSIDE_RESERVE + (SLICE_TP_COMM_RESERVE if tp > 1 else 0)
        slice_mib = shared_slice(demand, reserve)
        util_centi = math.ceil(demand * 100 / (slice_mib * MIB))
        if plan and plan.get("gpu_kv_gib") and float(plan["gpu_kv_gib"]) * GIB > kv_committed:
            grown_demand = demand - kv_committed + int(float(plan["gpu_kv_gib"]) * GIB)
            grown = shared_slice(grown_demand, reserve)
            if grown * MIB <= device_total:
                fallback_mib, fallback_util = slice_mib, util_centi / 100
                slice_mib = grown
                util_centi = math.ceil(grown_demand * 100 / (slice_mib * MIB))
        if slice_mib * MIB > device_total:
            raise SizingError("the shared slice is larger than the card")
        util, gpu_mib = util_centi / 100, slice_mib
    elif (
        group.get("type") == "cuda"
        and str(group.get("engine_type") or "vllm") == "vllm"
        and not encoder
        and group.get("kv_features")
    ):
        demand_centi = math.ceil(demand * 100 / device_total)
        target = (
            round(float(plan["gpu_memory_utilization"]) * 100)
            if plan and plan.get("gpu_memory_utilization")
            else int(T0_GPU_UTIL_FLOOR * 100)
        )
        util, gpu_mib = min(max(demand_centi, target), int(GPU_UTIL_MAX * 100)) / 100, ceil_mib(
            demand
        )
    else:
        util, gpu_mib = math.floor(min(max(raw, GPU_UTIL_MIN), GPU_UTIL_MAX) * 100) / 100, ceil_mib(
            demand
        )
    shm_mib = shm // MIB + (ceil_mib(t1["bytes"]) if t1["mode"] == "register" else 0)
    return CudaBudget(
        tp=tp,
        max_num_seqs=seqs,
        max_model_len=max_model_len,
        context_source=ctx["source"],
        max_num_batched_tokens=tokens,
        kv_engine_gib=kv_engine_gib,
        device_demand=demand,
        device_total=device_total,
        gpu_memory_utilization=util,
        gpu_memory_mib=gpu_mib,
        slice_fallback_mib=fallback_mib,
        slice_fallback_util=fallback_util,
        t1_bytes=t1["bytes"],
        t1_mode=t1["mode"],
        t1_block=t1["block"],
        t3=t3,
        kv_cache_dtype=kv_dtype,
        kv_skip_sliding=skip,
        engine_request_mib=request,
        engine_limit_mib=limit,
        shm_mib=shm_mib,
        lora_warmup_bytes=lora,
    )
