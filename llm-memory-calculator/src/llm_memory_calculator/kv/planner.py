"""The KV planner: one function that makes every KV-cache decision for a deployment.

budsim's heuristic chooses the hardware, node, parallelism and replicas; :func:`plan_kv` takes that
choice, the deployment's own fields and the raw infrastructure records, and returns the plan
budcluster renders (ARCHITECTURE §10). The pipeline:

    gates -> workload -> dtype -> geometry -> T0 -> fits the GPU? -> T1 -> T2 -> T3 -> P/D
          -> routing, namespace, resources -> prediction -> release ceiling

Every step that stops or drops a tier records why. An unknown fact never enables a tier, and the
same inputs always give the same plan.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Optional, Sequence

from . import constants as c
from .context import PlanContext
from .cost import RecomputeModel
from .dtype import choose_kv_dtype
from .facts import ClusterKVFacts, EngineSpec, NodeKVFacts
from .geometry import KVGeometry
from .group import NodeGroup, genz_hardware
from .infra import InfraSnapshot, fit_grown_slice
from .namespace import kv_namespace
from .pd import plan_pd
from .plan import KVTierPlan
from .predict import predict
from .routing import routing_policy
from .tiers import plan_t0, plan_t1, plan_t2, plan_t3
from .workload import DeploymentSpec, KVWorkload

GIB = c.GIB


def _gib(value: Optional[float]) -> float:
    return float(value) if value and value > 0 else 0.0


def _round_gib(value: float) -> float:
    return math.ceil(value * 100) / 100


def plan_kv(
    deployment: DeploymentSpec,
    engine: EngineSpec,
    group: NodeGroup,
    infra: Optional[InfraSnapshot] = None,
    *,
    recompute_fn: Optional[Callable[[int], Optional[float]]] = None,
    workload: Optional[KVWorkload] = None,
) -> KVTierPlan:
    """Plan the KV tiers of one node group of the configuration that deploys.

    ``infra`` is the target node and its cluster from ``cluster_info``. Without it (or without the
    node) the planner has no capability facts and no free memory, which keeps every tier that needs
    them off. ``recompute_fn`` replaces GenZ's prefill estimate (milliseconds for ``n`` tokens), for
    tests and recorded measurements. ``workload`` replaces the reuse profile inferred from the
    deployment's fields, for a measured benchmark whose reuse is known.
    """
    infra = infra or InfraSnapshot()
    host_free = infra.host_free_gib()
    if group.device_type in ("cpu", "cpu_high") and host_free is not None:
        host_free = max(host_free - group.pod_host_gib(), 0.0)
    free_cards = infra.gpu_free_list(group.device_model_key) if group.shared else []
    slice_cap = None
    if group.shared and group.device_type == "cuda" and free_cards:
        committed = int(group.kv_committed_gib * GIB)
        slice_cap = fit_grown_slice(
            group.device_total_memory_gb or max(free_cards),
            device_demand=group.device_demand_bytes(),
            kv_committed=committed,
            engine=group.engine,
            device_total_gib=group.device_total_memory_gb,
            replicas=group.replicas,
            free_gib=free_cards,
        )
        if slice_cap is None:
            slice_cap = committed / GIB  # not even the smallest growth fits: no growth
    hardware, hardware_name = group.hardware, group.hardware_name
    if hardware is None:
        hardware, hardware_name = genz_hardware(group.device_model_key, group.device_type)
    workload = workload or KVWorkload.from_spec(deployment)
    return _plan(
        model_config=deployment.model_config,
        engine=engine,
        device_type=group.device_type,
        hardware_mode="shared" if group.shared else "dedicated",
        device_memory_gib=group.device_total_memory_gb,
        weight_gib_per_rank=group.weight_gib,
        kv_demand_gib_per_rank=group.kv_committed_gib,
        other_device_gib_per_rank=group.other_gib,
        tensor_parallel=group.tp,
        pipeline_parallel=group.pp,
        replicas=group.replicas,
        workload=workload,
        node=infra.node_facts(),
        cluster=infra.cluster_facts(),
        host_free_gib=host_free,
        device_free_gib=max(free_cards) if free_cards else None,
        gpu_kv_cap_gib=slice_cap,
        device_tflops=group.device_tflops,
        device_generation=group.device_generation,
        precision=group.precision,
        data_parallel_attention=group.data_parallel_attention,
        model_id=deployment.model_id,
        revision=deployment.revision,
        weight_quantization=deployment.weight_quantization,
        hardware=hardware,
        hardware_name=hardware_name,
        recompute_fn=recompute_fn,
        cross_node_need=deployment.cross_node_need,
        e2e_latency_s=deployment.e2e_latency_s,
    )


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
    cluster: Optional[Mapping[str, Any]] = None,
    model_id: Optional[str] = None,
    hardware: Optional[Mapping[str, Any]] = None,
    validated_kv_dtypes: Any = frozenset(),
    recompute_fn: Optional[Callable[[int], Optional[float]]] = None,
    cross_node_need: bool = False,
    e2e_latency_s: Optional[float] = None,
) -> KVTierPlan:
    """Plan a deployment's KV tiers from already-derived inputs (the Phase 1 API budsim calls).

    Sizes are GiB. ``*_per_rank`` values are what budsim already sized for one rank (weights, the KV
    demand of the deployment's concurrency and context, and everything else on the device).
    ``host_free_gib`` is the target node's free host memory and ``device_free_gib`` the free memory
    of the GPU a shared slice lands on. ``gpu_kv_cap_gib`` is the most KV per rank a grown slice may
    hold, from the caller's placement check. ``node`` is that node's ``kv_capabilities`` and
    ``cluster`` the cluster's KV capability record. :func:`plan_kv` derives all of these itself.
    """
    spec = EngineSpec(
        name=str(engine or ""),
        kv_features=frozenset(engine_kv_features or ()),
        validated_kv_dtypes=frozenset(validated_kv_dtypes or ()),
        capabilities=engine_capabilities,
    )
    return _plan(
        model_config=model_config,
        engine=spec,
        device_type=device_type,
        hardware_mode=hardware_mode,
        device_memory_gib=device_memory_gib,
        weight_gib_per_rank=weight_gib_per_rank,
        kv_demand_gib_per_rank=kv_demand_gib_per_rank,
        other_device_gib_per_rank=other_device_gib_per_rank,
        tensor_parallel=tensor_parallel,
        pipeline_parallel=pipeline_parallel,
        replicas=replicas,
        workload=workload,
        node=NodeKVFacts.from_capabilities(node),
        cluster=ClusterKVFacts.from_record(cluster),
        host_free_gib=host_free_gib,
        device_free_gib=device_free_gib,
        gpu_kv_cap_gib=gpu_kv_cap_gib,
        device_tflops=device_tflops,
        device_generation=device_generation,
        precision=precision,
        data_parallel_attention=data_parallel_attention,
        model_id=model_id,
        hardware=hardware,
        recompute_fn=recompute_fn,
        cross_node_need=cross_node_need,
        e2e_latency_s=e2e_latency_s,
    )


def _plan(
    *,
    model_config: Mapping[str, Any],
    engine: EngineSpec,
    device_type: str,
    hardware_mode: str,
    device_memory_gib: Optional[float],
    weight_gib_per_rank: float,
    kv_demand_gib_per_rank: float,
    other_device_gib_per_rank: float,
    tensor_parallel: int,
    pipeline_parallel: int,
    replicas: int,
    workload: KVWorkload,
    node: NodeKVFacts,
    cluster: ClusterKVFacts,
    host_free_gib: Optional[float],
    device_free_gib: Optional[float],
    gpu_kv_cap_gib: Optional[float],
    device_tflops: Optional[float],
    device_generation: Optional[str],
    precision: str,
    data_parallel_attention: bool,
    model_id: Optional[str] = None,
    revision: Optional[str] = None,
    weight_quantization: Optional[str] = None,
    hardware: Optional[Mapping[str, Any]] = None,
    hardware_name: Optional[str] = None,
    recompute_fn: Optional[Callable[[int], Optional[float]]] = None,
    cross_node_need: bool = False,
    e2e_latency_s: Optional[float] = None,
) -> KVTierPlan:
    plan = KVTierPlan(workload_profile=workload.profile)
    decisions = plan.decisions
    decisions.append(f"profile: {workload.profile} ({workload.reason})")
    features = frozenset(engine.kv_features or ())
    tp = max(int(tensor_parallel or 1), 1)
    pp = max(int(pipeline_parallel or 1), 1)
    replicas = max(int(replicas or 1), 1)

    # -- 0. gates
    if workload.profile == "off":
        return plan
    if engine.name != "vllm":
        decisions.append(f"no KV plan: engine '{engine.name}' (vLLM only in this release)")
        return plan
    if not features:
        decisions.append("no KV plan: the engine record lists no KV features")
        return plan

    plan.profile = "native"
    plan.events_enabled = "kv_events" in features
    plan.routing = routing_policy(workload.profile)

    # -- 2. dtype (FR-QUANT-2)
    plan.kv_cache_dtype, why = choose_kv_dtype(
        model_config, device_generation, frozenset(engine.validated_kv_dtypes or ())
    )
    decisions.append(why)
    kv_precision = "fp8" if plan.kv_cache_dtype == "fp8" else precision

    # -- 3. geometry
    geometry = KVGeometry.from_model(
        model_config,
        seq_length=workload.input_tokens + workload.output_tokens,
        tensor_parallel=tp,
        precision=kv_precision,
        engine_capabilities=engine.capabilities,
        data_parallel_attention=data_parallel_attention,
    )
    if geometry.bytes_per_token_rank <= 0:
        decisions.append("no KV plan: the model has no per-token KV cache")
        plan.profile = "off"
        return plan
    plan.block_size = geometry.block_size
    if geometry.attention_type == "mla" and tp > 1 and not data_parallel_attention:
        decisions.append("KV: MLA latent counted on every tensor-parallel rank")
    if geometry.hybrid:
        decisions.append(
            f"KV: hybrid model, {geometry.recurrent_layers} recurrent layers with"
            f" {geometry.state_bytes_stored() / 2**20:.0f} MiB of state per cached prefix;"
            f" vLLM blocks of {geometry.block_size} tokens"
        )

    recompute = RecomputeModel(
        model_config=model_config,
        model_id=model_id,
        hardware=hardware,
        device_tflops=device_tflops,
        tensor_parallel=tp,
        precision=precision,
        attention_layers=geometry.kv_layers if geometry.hybrid else None,
        fn=recompute_fn,
    )
    ctx = PlanContext(
        plan=plan,
        model_config=model_config,
        workload=workload,
        features=features,
        geometry=geometry,
        recompute=recompute,
        node=node,
        cluster=cluster,
        device_type=device_type,
        hardware_mode=hardware_mode,
        device_memory_gib=_gib(device_memory_gib),
        weight_gib=_gib(weight_gib_per_rank),
        other_gib=_gib(other_device_gib_per_rank),
        demand_gib=_gib(kv_demand_gib_per_rank),
        tp=tp,
        pp=pp,
        replicas=replicas,
        host_free_gib=host_free_gib,
        device_free_gib=device_free_gib,
        gpu_kv_cap_gib=gpu_kv_cap_gib,
        cross_node_need=cross_node_need,
        e2e_latency_s=e2e_latency_s,
        model_id=model_id,
    )
    ctx.working_set_gib = workload.working_set_tokens * geometry.bytes_per_token_rank / GIB
    plan.working_set_gib = _round_gib(ctx.working_set_gib)
    decisions.append(
        f"working set: {workload.working_set_tokens} tokens ({plan.working_set_gib} GiB per rank)"
        f" from {workload.concurrency} x {workload.sessions_per_slot:g} sessions"
        f" x {workload.reusable_prefix_tokens} reusable tokens"
    )

    # -- 4. T0
    ctx.pool_gib = plan_t0(ctx)
    plan.gpu_pool_gib = _round_gib(ctx.pool_gib)

    # -- 5. fits the GPU? then 6-8. the offload tiers, cheapest first
    if ctx.working_set_gib <= ctx.pool_gib:
        decisions.append("no offload tier: the working set fits the T0 KV pool (FR-PLAN-5)")
    else:
        plan_t1(ctx)
        plan_t2(ctx)
        plan_t3(ctx)

    # -- 9. prefill/decode (withheld above the release ceiling like any tier)
    pd = plan_pd(ctx, hardware_name=hardware_name)
    if pd.get("enabled"):
        if "PD" in c.RELEASED_TIERS:
            plan.pd = pd
            plan.rdma = True
        else:
            plan.withheld.append({"tier": "PD", **pd})
            decisions.append("P/D planned but withheld: not in this package release")

    # -- 10. routing (set above), namespace, hash algorithm, profile, resources
    shared = [t for t in plan.tiers if t["tier"] in ("T2", "T3")]
    plan.hash_algo = "sha256_cbor" if shared else "sha256"
    attention = (
        "mla-replicated"
        if geometry.attention_type == "mla" and not data_parallel_attention
        else geometry.attention_type or "unknown"
    )
    plan.namespace = kv_namespace(
        model_id=model_id,
        revision=revision,
        weight_quantization=weight_quantization,
        kv_cache_dtype=plan.kv_cache_dtype,
        block_size=geometry.block_size,
        tensor_parallel=tp,
        pipeline_parallel=pp,
        attention_layout=attention,
        hash_algo=plan.hash_algo,
        engine_version=engine.version,
    )
    for entry in plan.tiers + plan.withheld:
        if entry["tier"] == "T3" and entry.get("backend") == "mooncake":
            entry["params"]["cache_prefix"] = plan.namespace
    released = {t["tier"] for t in plan.tiers}
    if "T3" in released and plan.tier("T3")["backend"] == "mooncake":
        plan.profile = "native_mooncake" if released & {"T1", "T2"} else "mooncake"

    # -- 11. prediction
    predict(ctx)
    return plan
