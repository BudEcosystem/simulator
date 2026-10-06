"""The full KV planner (Bud FRD-023): plan_kv() from raw records, T2, T3, P/D, the release ceiling,
and the planner held to the measured runs (NFR-9)."""

import dataclasses
import math
import statistics

import pytest

from llm_memory_calculator.kv import (
    ClusterKVFacts,
    DeploymentSpec,
    EngineSpec,
    InfraSnapshot,
    KVGeometry,
    KVWorkload,
    NodeGroup,
    NodeKVFacts,
    constants,
    plan_kv,
)
from llm_memory_calculator.kv import validate as kv_validate
from llm_memory_calculator.kv.calibration import QWEN3_06B, QWEN3_8_27B, QWEN3_8B, geometry_checks
from llm_memory_calculator.kv.cost import (
    RecomputeModel,
    TierLink,
    active_parameters,
    break_even,
    connector_efficiency,
    is_moe,
    prefill_seconds,
)
from llm_memory_calculator.kv.namespace import kv_namespace
from llm_memory_calculator.parameter_counter import UniversalParameterCounter

GIB = 1 << 30
ALL = frozenset({"T1", "T2", "T3", "PD"})
FEATURES = [
    "kv_events",
    "cache_salt",
    "cpu_offload_shm",
    "cpu_offload_host_alloc",
    "tiering_fs",
    "mooncake_store",
    "nixl",
]
BARE_METAL = {
    "pinned_offload_ok": True,
    "vgpu": False,
    "host_pointer_for_registered_mem": True,
    "pinned_copy_gbps": 50.0,
    "pcie_gen": 5,
    "local_nvme_gib": 3500.0,
}
TCS_VGPU = {
    "pinned_offload_ok": True,
    "vgpu": True,
    "host_pointer_for_registered_mem": False,
    "pinned_copy_gbps": 47.0,
    "rdma_nics": 0,
}
RDMA_NODE = {**BARE_METAL, "rdma_nics": 8, "rdma_link_gbps": 400.0, "gpudirect_rdma": True}
STORAGE = {
    "storage_classes": [{"name": "nvme", "node_local": True, "kv_read_gbps": 12.0}],
    "kv_storage": {"class": "nvme", "enabled": True},
}
# Free host memory that keeps T1 well under the working set, so T2 holds more than T1 does: the
# disk tier stores every block T1 stores, so with T1 as large as the working set T2 adds nothing
# and is dropped (the TCS and accubits01 disk rounds made the node RAM-limited the same way).
RAM_LIMITED = 200.0  # T1 100 GiB (half of it) against a 169 GiB working set
TCP_POOL = {"backend": "mooncake", "transport": "tcp", "read_gbps": 1.45, "capacity_gib": 512}
RDMA_POOL = {"backend": "mooncake", "transport": "rdma", "read_gbps": 40.0, "capacity_gib": 4096}


def flat(ms_per_token):
    return lambda n: n * ms_per_token


def _infra(caps=BARE_METAL, cluster=None, host_gb=1000.0, cards=None, pods=0):
    devices = [{"type": "cpu", "available_memory_gb": host_gb}]
    for free in cards or []:
        devices.append(
            {
                "type": "cuda",
                "raw_name": "H100",
                "available_count": 1,
                "mem_per_GPU_in_GB": 80.0,
                "memory_allocated_gb": 80.0 - free,
                "shared_containers_count": pods,
            }
        )
    return InfraSnapshot(
        node={"name": "n1", "kv_capabilities": caps, "devices": devices},
        cluster={"id": "c1", "kv_capability": cluster or {}},
    )


def _group(**over):
    base = dict(
        device_type="cuda",
        weight_memory_gb=16.4,
        kv_cache_memory_gb=4.0,
        activation_memory_gb=1.0,
        device_total_memory_gb=80.0,
        device_model_key="H100",
        device_generation="hopper",
        device_tflops=989.0,
    )
    base.update(over)
    return NodeGroup(**base)


def _deploy(config=QWEN3_8B, **over):
    base = dict(
        model_config=config,
        input_tokens=8000,
        output_tokens=2000,
        concurrency=32,
        model_id="Qwen/Qwen3-8B",
        e2e_latency_s=20.0,
    )
    base.update(over)
    return DeploymentSpec(**base)


def _plan(deploy=None, group=None, infra=None, features=FEATURES, ms_per_token=0.03, **kw):
    return plan_kv(
        deploy or _deploy(),
        EngineSpec.of("vllm", features),
        group or _group(),
        infra or _infra(),
        recompute_fn=flat(ms_per_token),
        **kw,
    )


@pytest.fixture
def released_all(monkeypatch):
    monkeypatch.setattr(constants, "RELEASED_TIERS", ALL)


# ------------------------------------------------------------------------------------- geometry


def test_dense_models_use_vllms_16_token_block_and_64_kib_segments():
    g = KVGeometry.from_model(QWEN3_8B, seq_length=4096)
    assert (g.block_size, g.bytes_per_token_total, g.segment_bytes()) == (16, 147456, 65536)
    assert not g.hybrid and g.state_bytes_stored() == 0


def test_a_hybrids_block_is_the_smallest_aligned_one_whose_page_holds_its_state():
    """vLLM 0.30 gave Qwen3.8-27B 784-token blocks; the state per prefix is 48 padded pages."""
    g = KVGeometry.from_model(QWEN3_8_27B, seq_length=4096)
    assert g.hybrid and (g.kv_layers, g.recurrent_layers) == (16, 48)
    assert g.block_size == 784
    assert g.state_bytes_stored() == 48 * 784 * 4096 == 154140672


@pytest.mark.parametrize("check", geometry_checks(), ids=lambda c: c.name)
def test_every_logged_pool_transfer_size_is_reproduced_exactly(check):
    """Spike S10: bytes Mooncake reported per prefix load, at blocks 784, 1568 and 3136."""
    g = KVGeometry.from_model(check.config, seq_length=check.prompt_tokens + 32)
    _, nbytes = g.prefix_bytes(check.prompt_tokens, block_size=check.block_size, scope="total")
    assert nbytes == check.measured_bytes


def test_a_replicated_mla_latent_moves_once_per_rank_from_a_node_tier():
    mla = {
        "model_type": "deepseek_v2",
        "hidden_size": 2048,
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 128,
        "v_head_dim": 128,
        "intermediate_size": 10944,
        "vocab_size": 102400,
    }
    g = KVGeometry.from_model(mla, seq_length=4096, tensor_parallel=2)
    _, rank = g.prefix_bytes(1024, scope="rank")
    _, total = g.prefix_bytes(1024, scope="total")
    assert total == 2 * rank  # every rank stores the whole latent


# ----------------------------------------------------------------------------------------- cost


def _link(gbps, floor=0.0):
    return TierLink(tier="T3", gbps=gbps, floor_ms=floor, scope="total", source="test")


def test_break_even_keeps_a_fast_tier_and_drops_a_slow_one():
    fast = break_even(
        tier="T3",
        cached_tokens=4096,
        nbytes=1e9,
        link=_link(40.0, 15),
        recompute_ms=500.0,
        recompute_source="t",
    )
    slow = break_even(
        tier="T3",
        cached_tokens=4096,
        nbytes=1e9,
        link=_link(1.0, 15),
        recompute_ms=500.0,
        recompute_source="t",
    )
    assert fast.keep and not slow.keep
    assert fast.reload_ms == pytest.approx(15 + 25)


@pytest.mark.parametrize(
    "load_rate,writes,bound",
    [(0.0, 0.0, "latency"), (0.5, 0.0, "latency"), (2.0, 1e8, "link"), (40.0, 5e9, "link")],
)
def test_the_needed_bandwidth_is_exactly_the_break_even_point(load_rate, writes, bound):
    """The needed rate is the tighter of the two limits: one idle load under 0.8 x recompute, and
    the deployment's loads and writes under MAX_LINK_UTILISATION of the link."""
    kw = dict(
        tier="T3",
        cached_tokens=1,
        nbytes=6e8,
        recompute_ms=800.0,
        recompute_source="t",
        load_rate_per_s=load_rate,
        write_bytes_per_s=writes,
    )
    needed = break_even(link=_link(1.0, 20), **kw).needed_gbps
    at = break_even(link=_link(needed, 20), **kw)
    if bound == "latency":
        assert at.reload_idle_ms == pytest.approx(0.8 * 800.0, rel=1e-6)
    else:
        assert at.utilisation == pytest.approx(constants.MAX_LINK_UTILISATION, rel=1e-6)
    assert not break_even(link=_link(needed * 0.99, 20), **kw).keep
    assert break_even(link=_link(needed * 1.01, 20), **kw).keep


def test_a_busy_link_still_pays_while_it_has_room():
    """Accuracy grid, TCS: Qwen3-0.6B at 16K kept T1 at +29-37% throughput with the link ~50% busy.
    Queueing raises a hit's TTFT (reported) but the GPU serves other requests meanwhile."""
    kw = dict(
        tier="T1", cached_tokens=16384, nbytes=1.88e9, recompute_ms=76.5, recompute_source="t"
    )
    link = TierLink(tier="T1", gbps=46.92, floor_ms=15, scope="rank", source="t", efficiency=0.85)
    busy = break_even(link=link, load_rate_per_s=13.5, **kw)
    assert 0.5 < busy.utilisation < constants.MAX_LINK_UTILISATION
    assert busy.reload_ms > 0.8 * 76.5 > busy.reload_idle_ms and busy.keep
    assert not break_even(link=link, load_rate_per_s=22.0, **kw).keep  # past the link limit


def test_the_needed_bandwidth_is_in_the_measured_unit_not_the_effective_one():
    """A TCP pool's connector reaches half the measured rate at 64 KiB segments, so the pool the
    gate measures has to be twice as fast as the effective link the break-even needs."""
    kw = dict(tier="T3", cached_tokens=1, nbytes=6e8, recompute_ms=800.0, recompute_source="t")
    full = break_even(link=_link(1.0, 20), **kw)
    half = break_even(
        link=TierLink(tier="T3", gbps=0.5, floor_ms=20, scope="total", source="t", efficiency=0.5),
        **kw,
    )
    assert half.needed_gbps == pytest.approx(full.needed_gbps / 0.5, rel=1e-9)


@pytest.mark.parametrize("transport,caps", [("tcp", TCS_VGPU), ("rdma", RDMA_NODE)])
def test_a_pool_measured_at_the_needed_rate_flips_the_decision(released_all, transport, caps):
    def plan_at(gbps):
        pool = {**TCP_POOL, "transport": transport, "read_gbps": gbps}
        return _plan(
            deploy=_deploy(
                input_tokens=16384, output_tokens=64, concurrency=32, cross_node_need=True
            ),
            infra=_infra(caps=caps, cluster={"t3": pool}),
            workload=KVWorkload(
                profile="chat",
                input_tokens=16384,
                output_tokens=64,
                concurrency=32,
                sessions_per_slot=3.0,
                reusable_share=16000 / 16384,
                reason="bench",
            ),
        )

    needed = plan_at(1.0).evaluations["T3"].needed_gbps
    assert (
        needed and not plan_at(needed * 0.98).planned("T3") and plan_at(needed * 1.02).planned("T3")
    )


def test_a_prefix_under_one_offload_block_stores_nothing():
    """The offloading connector moves whole 256-token blocks: a 121-token prefix has none."""
    plan = _plan(
        deploy=_deploy(input_tokens=128, output_tokens=64, concurrency=64),
        workload=KVWorkload(
            profile="chat",
            input_tokens=128,
            output_tokens=64,
            concurrency=64,
            sessions_per_slot=512.0,
            reusable_share=0.95,
            reason="short",
        ),
    )
    t1 = next(d for d in plan.dropped if d["tier"] == "T1")
    assert plan.tier("T1") is None and "nothing to store" in t1["reason"]
    assert "T1" not in plan.evaluations


def test_the_load_rate_counts_only_the_requests_the_gpu_pool_holds():
    """16 concurrent 16K requests can't all run in a small GPU pool: vLLM runs a few and queues the
    rest, so the tier's link sees that load (round 4, Qwen1.5-MoE at 16K: ~40% busy, T1 +54%)."""
    bench = KVWorkload(
        profile="chat",
        input_tokens=16640,
        output_tokens=64,
        concurrency=16,
        sessions_per_slot=4.0,
        reusable_share=16384 / 16640,
        reason="bench",
        uses_per_prefix=8,
    )

    def utilisation(card_gb):
        plan = _plan(group=_group(device_total_memory_gb=card_gb), workload=bench)
        return plan.evaluations["T1"].utilisation

    assert utilisation(30.0) < utilisation(80.0) / 2  # ~4.5 of 16 requests fit a 10 GiB pool


def test_a_saturated_tier_never_pays():
    v = break_even(
        tier="T3",
        cached_tokens=1,
        nbytes=1e9,
        link=_link(1.0),
        recompute_ms=1e6,
        recompute_source="t",
        load_rate_per_s=2.0,
    )
    assert v.utilisation >= 1 and math.isinf(v.reload_ms) and v.keep is False


def test_connector_efficiency_follows_the_measured_points_and_clamps():
    table = constants.T3_CONNECTOR_EFFICIENCY
    assert connector_efficiency(16 * 1024, table) == 0.5
    assert connector_efficiency(64 * 1024, table) == 0.5
    assert connector_efficiency(3 << 20, table) == 1.0
    assert 0.5 < connector_efficiency(512 * 1024, table) < 1.0
    assert connector_efficiency(64 << 20, table) == 1.0


def test_recompute_prefers_genz_and_falls_back_to_the_formula(monkeypatch):
    calls = []

    def fake(**kw):
        calls.append(kw)
        return {"Latency": 123.0}

    import llm_memory_calculator.performance_estimator as pe

    monkeypatch.setattr(pe, "estimate_prefill_performance", fake)
    genz = RecomputeModel(
        model_config=QWEN3_8B,
        model_id="Qwen/Qwen3-8B",
        hardware={"Flops": 989.5},
        device_tflops=989.0,
        tensor_parallel=2,
    )
    assert genz.ms(4096) == (123.0, "GenZ") and calls[0]["tensor_parallel"] == 2
    formula = RecomputeModel(model_config=QWEN3_8B, device_tflops=989.0)
    ms, source = formula.ms(4096)
    assert ms > 0 and "formula" in source
    assert RecomputeModel(model_config=QWEN3_8B).ms(4096)[0] is None


# ---------------------------------------------------------------------------------------- facts


def test_a_moe_model_recomputes_with_its_active_parameters():
    """Qwen3-30B-A3B runs 8 of 128 experts per token: ~3.7B of its ~31B parameters."""
    moe = {
        "model_type": "qwen3_moe",
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "moe_intermediate_size": 768,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "vocab_size": 151936,
    }
    assert 3.0e9 < active_parameters(moe) < 4.5e9
    assert active_parameters(QWEN3_8B) == pytest.approx(
        UniversalParameterCounter().count_parameters(dict(QWEN3_8B))
    )
    dense_equivalent = prefill_seconds({**moe, "num_experts_per_tok": 128}, 4096, 989.5)
    assert prefill_seconds(moe, 4096, 989.5) < dense_equivalent / 5


def test_unknown_or_malformed_facts_read_as_unknown():
    facts = NodeKVFacts.from_capabilities(
        {"pinned_offload_ok": "yes", "pinned_copy_gbps": -1, "rdma_nics": True}
    )
    assert facts.pinned_offload_ok is None and facts.pinned_copy_gbps is None
    assert facts.rdma_nics is None and not facts.gpudirect()


def test_gpudirect_needs_a_probe_or_a_nic_with_dma_buf():
    assert NodeKVFacts.from_capabilities({"rdma_nics": 2, "dma_buf": True}).gpudirect()
    assert not NodeKVFacts.from_capabilities({"rdma_nics": 2}).gpudirect()
    assert not NodeKVFacts.from_capabilities(
        {"rdma_nics": 2, "dma_buf": True, "gpudirect_rdma": False}
    ).gpudirect()


def test_the_kv_storage_class_is_the_named_one_and_only_when_enabled():
    facts = ClusterKVFacts.from_record(STORAGE)
    assert facts.kv_storage().name == "nvme" and facts.kv_storage().read_gbps == 12.0
    off = ClusterKVFacts.from_record({**STORAGE, "kv_storage": {"class": "nvme", "enabled": False}})
    assert off.kv_storage() is None
    assert ClusterKVFacts.from_record({"t3": {"backend": "none"}}).t3 is None


def test_auto_picks_the_fastest_measured_node_local_class():
    record = {
        "storage_classes": [
            {"name": "local-path", "node_local": True, "kv_read_gbps": 2.6},
            {"name": "lvms-nvme", "node_local": True, "kv_read_gbps": 6.1},
            {"name": "ceph-rbd", "node_local": False, "kv_read_gbps": 9.0},
            {"name": "openebs-hostpath", "node_local": True},
        ]
    }
    chosen, how = ClusterKVFacts.from_record(record).kv_storage_choice(2.0)
    assert chosen.name == "lvms-nvme" and how.startswith("Auto")
    # the setting's enabled flag alone (no class) is still Auto
    on = ClusterKVFacts.from_record({**record, "kv_storage": {"enabled": True}})
    assert on.kv_storage(2.0).name == "lvms-nvme"


def test_auto_picks_nothing_below_the_minimum_read_rate():
    record = {"storage_classes": [{"name": "local-path", "node_local": True, "kv_read_gbps": 1.4}]}
    chosen, why = ClusterKVFacts.from_record(record).kv_storage_choice(2.0)
    assert chosen is None and "local-path" in why and "under 2" in why


def test_a_named_class_is_used_even_below_the_auto_minimum_and_a_missing_one_is_reported():
    """FR-SET-4: the admin's choice holds; the break-even still judges it per model."""
    slow = {
        "storage_classes": [{"name": "local-path", "node_local": True, "kv_read_gbps": 1.4}],
        "kv_storage": {"class": "local-path"},
    }
    chosen, how = ClusterKVFacts.from_record(slow).kv_storage_choice(2.0)
    assert chosen.name == "local-path" and "setting" in how
    missing = {**slow, "kv_storage": {"class": "gone"}}
    chosen, why = ClusterKVFacts.from_record(missing).kv_storage_choice(2.0)
    assert chosen is None and "'gone'" in why


# ------------------------------------------------------------------------------- plan_kv from raw


def test_plan_kv_derives_host_memory_from_the_node_record():
    """Free host memory less 6 GiB per GPU engine pod that requests none (budsim's rule)."""
    idle = _plan(infra=_infra(host_gb=200.0, cards=[80.0], pods=0))
    busy = _plan(infra=_infra(host_gb=200.0, cards=[80.0], pods=3))
    assert idle.tier("T1")["size_gib"] > busy.tier("T1")["size_gib"]


def test_plan_kv_caps_a_shared_slice_at_what_the_free_cards_hold():
    group = _group(hardware_mode="shared", kv_cache_memory_gb=2.0)
    roomy = _plan(group=group, infra=_infra(cards=[60.0]))
    tight = _plan(group=group, infra=_infra(cards=[26.0]))
    assert roomy.gpu_kv_gib > tight.gpu_kv_gib > 2


def test_without_a_node_nothing_that_needs_its_facts_is_planned():
    plan = _plan(infra=InfraSnapshot())
    assert plan.tiers == [] and plan.withheld == []
    assert any("not verified" in d for d in plan.decisions)


def test_every_dropped_tier_is_on_the_wire_with_its_reason():
    wire = _plan(
        deploy=_deploy(cross_node_need=True), infra=_infra(cluster={"t3": TCP_POOL})
    ).to_wire()
    t3 = next(d for d in wire["dropped"] if d["tier"] == "T3")
    assert "loses break-even" in t3["reason"] and t3["reload_ms"] > t3["recompute_ms"] * 0.8


# ------------------------------------------------------------------------------- release ceiling


def test_tiers_above_the_release_ceiling_are_planned_but_withheld():
    """This release returns T1 and T2; T3 (the Mooncake pool) is planned but withheld until
    budcluster renders it."""
    plan = _plan(
        deploy=_deploy(
            input_tokens=32000, output_tokens=2000, cross_node_need=True, concurrency=64
        ),
        group=_group(replicas=4),
        infra=_infra(caps=RDMA_NODE, cluster={**FAST_STORAGE, "t3": RDMA_POOL}),
        ms_per_token=0.05,
    )
    assert plan.tier("T1") is not None and plan.tier("T2") is not None and plan.tier("T3") is None
    assert [t["tier"] for t in plan.withheld] == ["T3"]
    assert any("T3 planned but withheld" in d for d in plan.decisions)
    assert all(t["tier"] in constants.RELEASED_TIERS for t in plan.to_wire()["tiers"])


def test_raising_the_ceiling_returns_them(released_all):
    plan = _plan(infra=_infra(cluster=STORAGE, host_gb=RAM_LIMITED))
    assert plan.tier("T2")["storage_class"] == "nvme" and plan.withheld == []
    assert plan.hash_algo == "sha256_cbor"  # a shared tier needs reproducible hashes (FR-RENDER-2)


# -------------------------------------------------------------------------------------------- T2


@pytest.mark.parametrize(
    ("cluster", "features", "reason"),
    [
        ({}, FEATURES, "no node-local storage class has a measured read rate"),
        (
            {
                **STORAGE,
                "storage_classes": [{"name": "nvme", "node_local": False, "kv_read_gbps": 12.0}],
            },
            FEATURES,
            "node-local",
        ),
        (
            {**STORAGE, "storage_classes": [{"name": "nvme", "node_local": True}]},
            FEATURES,
            "measured read rate",
        ),
        (STORAGE, [f for f in FEATURES if f != "tiering_fs"], "tiering_fs"),
    ],
)
def test_t2_needs_a_measured_node_local_class_and_the_engine_feature(
    released_all, cluster, features, reason
):
    plan = _plan(infra=_infra(cluster=cluster), features=features)
    assert plan.tier("T2") is None
    assert any(reason in d["reason"] for d in plan.dropped if d["tier"] == "T2")


def test_a_vgpu_node_gets_no_disk_tier_behind_an_alloc_mode_t1(released_all):
    """vLLM's tiering moves KV through T1's shared registered /dev/shm region. On a vGPU that can't
    use registered memory, alloc mode keeps T1 working but can't carry a disk tier (TCS,
    2026-10-02)."""
    plan = _plan(
        deploy=_deploy(tool_calling=True, input_tokens=24000, output_tokens=1000),
        infra=_infra(caps={**TCS_VGPU, "local_nvme_gib": 500.0}, cluster=STORAGE),
    )
    assert plan.tier("T1")["params"]["host_memory"] == "alloc" and plan.tier("T2") is None
    reason = next(d["reason"] for d in plan.dropped if d["tier"] == "T2")
    assert "shared registered" in reason and constants.T1_PER_COPY_FEATURE in reason


def test_per_copy_transfers_let_a_vgpu_node_run_the_disk_tier(released_all):
    plan = _plan(
        deploy=_deploy(tool_calling=True, input_tokens=24000, output_tokens=1000),
        infra=_infra(caps={**TCS_VGPU, "local_nvme_gib": 500.0}, cluster=STORAGE, host_gb=RAM_LIMITED),
        features=FEATURES + [constants.T1_PER_COPY_FEATURE],
    )
    assert plan.tier("T1")["params"]["host_memory"] == "register" and plan.tier("T2")


def test_per_copy_t1_loads_use_their_own_link(released_all):
    """Per-copy transfers (TCS disk-tier round): 15 ms + bytes / 0.34 of the probe, not register
    mode's batched copy."""
    deploy = _deploy(tool_calling=True, input_tokens=24000, output_tokens=1000)
    infra = _infra(caps=TCS_VGPU)
    per_copy = _plan(
        deploy=deploy, infra=infra, features=FEATURES + [constants.T1_PER_COPY_FEATURE]
    )
    alloc = _plan(deploy=deploy, infra=infra)
    link = per_copy.evaluations["T1"].link
    assert link.efficiency == constants.T1_PER_COPY_LOAD_EFFICIENCY
    assert link.floor_ms == constants.T1_PER_COPY_FLOOR_MS
    assert alloc.evaluations["T1"].link.efficiency == constants.T1_ALLOC_LOAD_EFFICIENCY


def test_a_disk_hit_pays_one_t1_load_after_the_disk_read(released_all):
    """vLLM's tiering promotes the chunks into T1 first, then loads them to the GPU."""
    plan = _plan(infra=_infra(cluster=STORAGE, host_gb=RAM_LIMITED))
    t1, t2 = plan.evaluations["T1"], plan.evaluations["T2"]
    disk_ms = constants.T2_FLOOR_MS + t2.bytes / (12.0 * constants.GB) * 1e3
    assert t2.reload_idle_ms == pytest.approx(disk_ms + t1.reload_idle_ms)


def test_t2_is_kept_for_throughput_unless_ttft_binds(released_all):
    """A disk load slower than one recompute still pays under load (Qwen3-8B 4K c16 on TCS: +14%),
    as for T1; an interactive deployment needs each load to beat its recompute."""
    slow = {
        **STORAGE,
        "storage_classes": [{"name": "nvme", "node_local": True, "kv_read_gbps": 1.5}],
    }
    # 24K-token agentic prompts: a working set past the GPU pool and T1, so T2 holds something
    deploy = _deploy(tool_calling=True, concurrency=16, input_tokens=24000, output_tokens=1000)
    agentic = _plan(deploy=deploy, infra=_infra(cluster=slow, host_gb=RAM_LIMITED))
    verdict = agentic.evaluations["T2"]
    assert verdict.reload_idle_ms > verdict.recompute_ms and verdict.utilisation < 0.8
    assert agentic.tier("T2") is not None
    assert any(d.startswith("T2 kept for throughput") for d in agentic.decisions)
    workload = KVWorkload.from_deployment(
        input_tokens=deploy.input_tokens,
        output_tokens=deploy.output_tokens,
        concurrency=deploy.concurrency,
        enable_tool_calling=True,
    )
    interactive = _plan(
        deploy=deploy,
        infra=_infra(cluster=slow, host_gb=RAM_LIMITED),
        workload=dataclasses.replace(workload, profile="interactive"),
    )
    assert interactive.tier("T2") is None
    assert any(
        d["tier"] == "T2" and "about as much as recomputing" in d["reason"]
        for d in interactive.dropped
    )


def test_t2_is_dropped_when_it_would_only_copy_t1(released_all):
    """With T1 as large as the working set, the disk tier would hold only T1's blocks (it stores
    every one of them) and add none: dropped, so the node isn't written for nothing."""
    plan = _plan(infra=_infra(cluster=STORAGE))
    assert plan.tier("T1") is not None and plan.tier("T2") is None
    assert any(
        d["tier"] == "T2" and "only copies" in d["reason"] for d in plan.dropped
    ), plan.dropped
    limited = _plan(infra=_infra(cluster=STORAGE, host_gb=RAM_LIMITED))
    assert limited.tier("T2")["size_gib"] > limited.tier("T1")["size_gib"]


def test_t2_sits_behind_t1(released_all):
    plan = _plan(infra=_infra(caps={**BARE_METAL, "pinned_offload_ok": False}, cluster=STORAGE))
    assert plan.tier("T1") is None and plan.tier("T2") is None


def test_a_slow_class_loses_break_even(released_all):
    slow = {
        **STORAGE,
        "storage_classes": [{"name": "nvme", "node_local": True, "kv_read_gbps": 0.3}],
    }
    plan = _plan(infra=_infra(cluster=slow, host_gb=RAM_LIMITED), ms_per_token=0.01)
    assert plan.tier("T2") is None and plan.evaluations["T2"].keep is False


# -------------------------------------------------------------------------------------------- T3


def test_t3_is_only_planned_for_a_cross_node_need(released_all):
    plan = _plan(infra=_infra(caps=RDMA_NODE, cluster={**STORAGE, "t3": RDMA_POOL}))
    assert plan.tier("T3") is None
    assert any("no cross-node need" in d["reason"] for d in plan.dropped)


def test_an_rdma_pool_pays_and_renders_the_mooncake_profile(released_all):
    plan = _plan(
        deploy=_deploy(cross_node_need=True),
        infra=_infra(caps=RDMA_NODE, cluster={**STORAGE, "t3": RDMA_POOL}),
    )
    t3 = plan.tier("T3")
    assert t3["backend"] == "mooncake" and t3["transport"] == "rdma"
    assert t3["params"]["cache_prefix"] == plan.namespace and plan.namespace.startswith("kvns-")
    assert t3["params"]["load_failure_policy"] == "recompute"
    assert plan.profile == "native_mooncake" and plan.to_wire()["resources"]["rdma"] is True


def test_an_rdma_pool_needs_gpudirect_on_the_node(released_all):
    plan = _plan(
        deploy=_deploy(cross_node_need=True),
        infra=_infra(caps={**RDMA_NODE, "gpudirect_rdma": False}, cluster={"t3": RDMA_POOL}),
    )
    assert plan.tier("T3") is None
    assert any("GPUDirect" in d["reason"] for d in plan.dropped)


def test_the_tcs_tcp_pool_loses_and_says_what_it_would_need(released_all):
    """Spike S10: on TCS's 2.0 GB/s pod network the pool can't beat an H100 recomputing 8B."""
    plan = _plan(
        deploy=_deploy(cross_node_need=True),
        infra=_infra(caps=TCS_VGPU, cluster={"t3": TCP_POOL}),
        ms_per_token=0.0242,
    )
    t3 = next(d for d in plan.dropped if d["tier"] == "T3")
    assert plan.tier("T3") is None and t3["needed_gbps"] is None  # 300 ms floor > the whole budget


def test_a_hybrid_model_can_use_the_pool_but_not_the_offloading_connector(released_all):
    deploy = _deploy(
        config=QWEN3_8_27B,
        model_id="Qwen/Qwen3.8-27B",
        cross_node_need=True,
        input_tokens=16384,
        output_tokens=256,
    )
    group = _group(weight_memory_gb=55.0, kv_cache_memory_gb=6.0)
    plan = _plan(
        deploy=deploy,
        group=group,
        ms_per_token=0.5,
        infra=_infra(caps=RDMA_NODE, cluster={**STORAGE, "t3": RDMA_POOL}),
    )
    assert plan.tier("T1") is None and plan.tier("T2") is None
    assert plan.tier("T3") is not None and plan.profile == "mooncake"
    assert plan.tier("T3")["params"]["block_size"] == 784


# ------------------------------------------------------------------------------------------- P/D


def test_pd_is_not_proposed_below_its_gates(released_all):
    plan = _plan(infra=_infra(caps=RDMA_NODE))
    assert plan.pd["enabled"] is False
    assert any(d.startswith("P/D: no (") for d in plan.decisions)


# ------------------------------------------------------------------------------------ invariants


def test_the_same_inputs_give_the_same_plan(released_all):
    args = dict(
        deploy=_deploy(cross_node_need=True),
        infra=_infra(caps=RDMA_NODE, cluster={**STORAGE, "t3": RDMA_POOL}),
    )
    assert _plan(**args).to_wire() == _plan(**args).to_wire()


@pytest.mark.parametrize("ms_per_token", [0.01, 0.03, 0.1])
def test_more_pool_bandwidth_never_removes_the_pool(released_all, ms_per_token):
    kept = []
    for gbps in (0.5, 1, 2, 4, 8, 16, 32, 64, 128):
        pool = {**TCP_POOL, "read_gbps": gbps}
        plan = _plan(
            deploy=_deploy(cross_node_need=True),
            infra=_infra(cluster={"t3": pool}),
            ms_per_token=ms_per_token,
        )
        kept.append(plan.tier("T3") is not None)
    assert kept == sorted(kept)  # once kept, kept for every faster pool


def test_unknown_infrastructure_never_enables_a_shared_tier(released_all):
    plan = _plan(deploy=_deploy(cross_node_need=True), infra=_infra(cluster={}))
    assert plan.tier("T2") is None and plan.tier("T3") is None


def test_the_namespace_changes_with_anything_that_changes_the_kv_layout():
    base = dict(
        model_id="m",
        revision="r",
        weight_quantization=None,
        kv_cache_dtype="auto",
        block_size=16,
        tensor_parallel=1,
        pipeline_parallel=1,
        attention_layout="gqa",
        hash_algo="sha256_cbor",
        engine_version="0.10.0",
    )
    first = kv_namespace(**base)
    assert first == kv_namespace(**base)
    for key, value in (
        ("kv_cache_dtype", "fp8"),
        ("block_size", 784),
        ("tensor_parallel", 2),
        ("revision", "r2"),
    ):
        assert kv_namespace(**{**base, key: value}) != first
    assert kv_namespace(**{**base, "model_id": None}) is None


# ------------------------------------------------------------------- reference scenarios (FRD §4)


def test_s_a_coding_agent_on_bare_metal_plans_t1_and_t2_with_maximal_affinity(released_all):
    plan = _plan(
        deploy=_deploy(tool_calling=True, input_tokens=24000, output_tokens=1000),
        group=_group(replicas=2),
        infra=_infra(cluster=STORAGE),
    )
    assert plan.workload_profile == "agentic" and plan.routing["affinity"] == "max"
    assert plan.tier("T1") and plan.tier("T2") and plan.tier("T3") is None


def test_this_release_returns_t2_on_the_auto_class_with_the_fs_tier_params():
    """T2 is released (budcluster renders it, IMPLEMENTATION_PLAN 3.5). With no KV storage
    setting (every cluster until budapp's settings exist) Auto picks the class, and the plan
    carries what the renderer maps."""
    auto = {"storage_classes": STORAGE["storage_classes"]}
    plan = _plan(
        deploy=_deploy(tool_calling=True, input_tokens=24000, output_tokens=1000),
        group=_group(replicas=2),
        infra=_infra(cluster=auto),
    )
    t2 = plan.tier("T2")
    assert t2 and t2["storage_class"] == "nvme" and t2["backend"] == "fs"
    assert t2["params"] == {"block_size": 256, "n_read_threads": 8, "n_write_threads": 4}
    assert any(d.startswith("T2:") and "Auto" in d for d in plan.decisions)
    assert plan.namespace and plan.namespace.startswith("kvns-")


def test_s_c_tcs_chat_plans_alloc_mode_t1_and_no_pool(released_all):
    plan = _plan(infra=_infra(caps=TCS_VGPU, cluster={"t3": TCP_POOL}))
    assert plan.tier("T1")["params"]["host_memory"] == "alloc" and plan.tier("T3") is None


FAST_STORAGE = {
    "storage_classes": [{"name": "lvms-nvme", "node_local": True, "kv_read_gbps": 48.0}],
    "kv_storage": {"class": "lvms-nvme", "enabled": True},
}


def test_s_d_long_context_rag_on_rdma_plans_every_tier(released_all):
    """Four replicas of long-context RAG on an RDMA node with a 4-drive storage class."""
    plan = _plan(
        deploy=_deploy(
            input_tokens=32000, output_tokens=2000, cross_node_need=True, concurrency=64
        ),
        group=_group(replicas=4),
        infra=_infra(caps=RDMA_NODE, cluster={**FAST_STORAGE, "t3": RDMA_POOL}),
        ms_per_token=0.05,
    )
    assert [t["tier"] for t in plan.tiers] == ["T1", "T2", "T3"], plan.decisions


def test_one_nvme_drive_cant_absorb_a_rag_deployments_write_stream(released_all):
    """vLLM writes every computed prompt block to each secondary tier. Eight replicas on one node,
    each running as many 34K-token RAG requests as its GPU pool holds (~12), write more new KV than
    a
    single 12 GB/s drive takes; four replicas keep it ~70% busy, under the link limit."""
    four = _plan(
        deploy=_deploy(input_tokens=32000, output_tokens=2000, concurrency=64),
        group=_group(replicas=4),
        infra=_infra(caps=RDMA_NODE, cluster=STORAGE),
        ms_per_token=0.05,
    )
    assert four.tier("T2") and four.evaluations["T2"].utilisation < constants.MAX_LINK_UTILISATION
    plan = _plan(
        deploy=_deploy(input_tokens=32000, output_tokens=2000, concurrency=64),
        group=_group(replicas=8),
        infra=_infra(caps=RDMA_NODE, cluster=STORAGE),
        ms_per_token=0.05,
    )
    assert plan.tier("T2") is None and plan.evaluations["T2"].utilisation >= 1.0


def test_s_e_embeddings_are_planned_off():
    plan = _plan(deploy=_deploy(is_embedding=True))
    assert plan.profile == "off" and plan.tiers == []


def test_s_b_a_shared_slice_grows_before_anything_offloads():
    plan = _plan(
        deploy=_deploy(target_ttft_ms=300.0, concurrency=4),
        group=_group(hardware_mode="shared", kv_cache_memory_gb=2.0),
        infra=_infra(cards=[40.0]),
    )
    assert plan.workload_profile == "interactive" and plan.routing["affinity"] == "capped"
    assert plan.gpu_kv_gib and plan.gpu_kv_gib > 2


# ------------------------------------------------------------------- measured runs (NFR-9)


@pytest.fixture(scope="module")
def report():
    return kv_validate.run()


def test_the_planner_matches_the_measured_load_runs(report):
    """79 load runs: spike S10, accubits01, the TCS T1 grid (24), hold-out (22) and round 3 (20:
    32K-120K, Qwen3-30B-A3B, Qwen2.5-72B AWQ), round 4 (22: OLMoE, Qwen1.5-MoE, Qwen3-4B/1.7B,
    30B-A3B at 2K) and 6 CPU engines. No tier it turns on loses throughput; the misses are
    conservative drops at 1-4K and concurrency 16 (see test_the_known_misses). Round 4 scored 15/22
    blind; the throughput decision and the MoE guard came after it, so these numbers include it
    in-sample. Plus 5 disk-tier (T2) runs on TCS: 4/5 blind; the staged T1 hop and the throughput
    rule for T2 came after, so they're in-sample too. Plus round 5, a blind MoE hold-out
    (Mixtral-8x7B AWQ, DeepSeek-V2-Lite, granite-3.1-3b-a800m): 17/18, scored as predicted."""
    assert len(report.regrets) == 124
    assert report.decision_accuracy >= 118 / 124
    assert statistics.median(report.regrets) == 0.0 and max(report.regrets) < 0.14
    assert report.enable_precision == 1.0


def _single_errors(report, longer_than_1k):
    reload, recompute = [], []
    for r in report.results:
        f = r.fixture
        if f.kind != "single" or (f.workload.input_tokens > 2048) != longer_than_1k:
            continue
        if f.gate_measured and r.reload_pred_ms:
            reload.append(abs(r.reload_pred_ms - f.reload_ms) / f.reload_ms)
        if f.genz_ms:
            recompute.append(abs(f.genz_ms - f.recompute_ms) / f.recompute_ms)
    return reload, recompute


def test_component_errors_meet_nfr_9_from_4k_up(report):
    reload, _ = _single_errors(report, longer_than_1k=True)
    assert len(reload) >= 38 and statistics.mean(reload) <= 0.20
    dense = [
        abs(r.fixture.genz_ms - r.fixture.recompute_ms) / r.fixture.recompute_ms
        for r in report.results
        if r.fixture.kind == "single"
        and r.fixture.genz_ms
        and r.fixture.workload.input_tokens > 2048
        and not is_moe(r.fixture.deployment.model_config)
    ]
    assert len(dense) >= 29 and statistics.mean(dense) <= 0.15
    assert all(row["error"] == 0 for row in report.geometry)


def test_genz_under_predicts_a_moe_prefill(report):
    """Known gap: GenZ's single-request prefill is below measured for every MoE it models: 24-66%
    for OLMoE, Qwen1.5-MoE and Qwen3-30B-A3B, 32-80% for granite-3.1-3b-a800m and 54% for
    Mixtral-8x7B
    AWQ (kernel overheads it doesn't model; AWQ dequantisation). T1's throughput decision uses the
    batched estimate instead; GenZ's number decides only for interactive deployments. GenZ can't
    model DeepSeek-V2-Lite's MLA, so its recompute is the FLOPs formula, which runs 14-20% high from
    4K up."""
    moe = [
        r.fixture
        for r in report.results
        if r.fixture.kind == "single"
        and r.fixture.genz_ms
        and is_moe(r.fixture.deployment.model_config)
        and "kv_lora_rank" not in r.fixture.deployment.model_config  # MLA: the FLOPs fallback
    ]
    assert len(moe) >= 15 and all(f.genz_ms < f.recompute_ms for f in moe)


def test_short_prefix_reloads_are_overestimated_by_the_t1_floor(report):
    """Known gap: every T1 reload of a 1K-or-shorter prefix on TCS was faster than predicted (7-17
    ms measured; round 5's MoEs 8-11 ms), even with alloc mode's fitted 11 ms floor: load time
    doesn't
    follow bytes alone (a 0.94 GB load took 46 ms on Qwen3-0.6B, 35 ms for 1.07 GB on Qwen3-32B).
    Short-prefix decisions rest on T1_MIN_RECOMPUTE_SAVED_MS instead."""
    short = [
        r
        for r in report.results
        if r.fixture.kind == "single"
        and r.fixture.tier == "T1"
        and r.fixture.workload.input_tokens <= 2048
        and r.fixture.source.startswith("TCS")
    ]
    assert len(short) == 16 and all(r.reload_pred_ms > r.fixture.reload_ms for r in short)


def test_the_known_misses(report):
    """* accubits01's same-node pool: the TCS-fitted 300 ms TCP floor; its health gate must measure
    it.
    * Phi-4-mini at 1K and Qwen3-8B at 512, concurrency 16: one load is slower than recomputing, yet
      T1 raised throughput 7-14% with the GPU saturated. The rule fails safe (a missed gain).
    * Qwen3-8B, one 513-token request: the short-prefix floor above.
    * Round 4 at concurrency 16: Qwen3-4B 1K (+6%), OLMoE 3.5K (+10%), Qwen1.5-MoE 4K (+16%) gain
      with T1 although a hit saves less batched work than the guard asks; at c4 the same cells lose
      or tie. The guard has no notion of how saturated the GPU is.
    * Single requests on OLMoE and Qwen1.5-MoE: GenZ's MoE prefill is low (see above).
    * Qwen3-14B, one 32K request from the disk tier: 1.79 s against a 2.33 s recompute (0.77, under
      the 0.8 cut); the planner has 2.05 s against GenZ's 1.90 s. Under load the throughput rule
      keeps it.
    * Round 5 (blind MoE hold-out): DeepSeek-V2-Lite 4K at c16 gains 16% while a hit saves 53 ms,
      under the 60 ms MoE guard (at c4 it ties) -- the c16 pattern again. Single requests on
      DeepSeek-V2-Lite at 1K and granite-3.1-3b-a800m at 1K and 4K: one T1 load beats recompute,
      which the planner, with GenZ's low small-MoE prefill, has the other way round; under load
      those cells lose with T1, as predicted."""
    assert set(report.misses()) == {
        "tcs-t2-disk-qwen3-14b-32k-single",
        "tcs-t1-deepseek-v2-lite-chat-4k-c16",
        "tcs-t1-deepseek-v2-lite-chat-1k-single",
        "tcs-t1-granite-3.1-3b-a800m-instruct-1k-single",
        "tcs-t1-granite-3.1-3b-a800m-instruct-4k-single",
        "accubits-0.6b-4k-pool-single",
        "tcs-t1-phi-4-mini-instruct-1k-c16",
        "tcs-t1-qwen3-8b-512-c16",
        "tcs-t1-qwen3-8b-512-single",
        "tcs-t1-qwen3-4b-1k-c16",
        "tcs-t1-olmoe-1b-7b-0125-instruct-4k-c16",
        "tcs-t1-qwen1.5-moe-a2.7b-chat-4k-c16",
        "tcs-t1-olmoe-1b-7b-0125-instruct-1k-single",
        "tcs-t1-qwen1.5-moe-a2.7b-chat-1k-single",
        "tcs-t1-qwen1.5-moe-a2.7b-chat-4k-single",
    }


def test_a_moe_at_1k_keeps_no_t1_because_a_hit_saves_too_little():
    """Qwen3-30B-A3B at 1K lost 18-28% with T1 though one load (13 ms) beats one recompute (48 ms):
    batched, its prefill is ~17 ms of FLOPs, less than the tier's own per-request copies cost."""
    moe = next(f for f in kv_validate.fixtures() if f.name == "tcs-t1-qwen3-30b-a3b-1k-c16")
    result = kv_validate.run_fixture(moe)
    assert not result.planned
    t1 = next(d for d in result.plan.dropped if d["tier"] == "T1")
    assert "batched GPU work" in t1["reason"]
    dense = next(f for f in kv_validate.fixtures() if f.name == "tcs-t1-qwen3-8b-1k-c16")
    assert kv_validate.run_fixture(dense).planned  # ~38 ms saved per hit: +28% measured


def test_the_first_use_of_a_prefix_caps_the_predicted_hit_rate():
    """A prefix's first use misses everywhere, so 8 uses per prefix serve at most 7/8 of the reuse
    (TCS bench: predicted hit rates were ~15 points high without this, ~3 with it)."""

    def hit(uses):
        return _plan(
            workload=KVWorkload(
                profile="chat",
                input_tokens=4352,
                output_tokens=32,
                concurrency=8,
                sessions_per_slot=20.0,
                reusable_share=4096 / 4352,
                reason="bench",
                uses_per_prefix=uses,
            )
        ).to_wire()["predicted"]["hit_rate"]

    assert hit(8) == pytest.approx(hit(0) * 7 / 8, abs=0.002)
    assert (
        KVWorkload.from_deployment(
            input_tokens=4096, output_tokens=512, concurrency=8
        ).uses_per_prefix
        > 1
    )


def test_alloc_mode_t1_uses_its_own_measured_load_cost():
    alloc = _plan(infra=_infra(caps=TCS_VGPU)).evaluations["T1"].link
    register = _plan(infra=_infra(caps=BARE_METAL)).evaluations["T1"].link
    assert (alloc.floor_ms, alloc.efficiency) == (
        constants.T1_ALLOC_FLOOR_MS,
        constants.T1_ALLOC_LOAD_EFFICIENCY,
    )
    assert (register.floor_ms, register.efficiency) == (
        constants.T1_FLOOR_MS,
        constants.T1_LOAD_EFFICIENCY,
    )


def test_batched_recompute_counts_a_moes_active_parameters():
    moe = {
        "model_type": "qwen3_moe",
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "moe_intermediate_size": 768,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "vocab_size": 151936,
    }
    batched = RecomputeModel(model_config=moe, device_tflops=989.5).batched_ms(1024)
    assert 12 < batched < 25  # ~17 ms at PREFILL_MFU, not the ~140 ms all 30B parameters would cost
    assert RecomputeModel(model_config=moe).batched_ms(1024) is None  # no device FLOPs


def test_the_report_renders(report):
    text = kv_validate.render(report)
    assert "decision accuracy" in text and "accubits-0.6b-4k-pool-single" in text


def test_a_measured_bench_workload_can_replace_the_inferred_profile():
    bench = KVWorkload(
        profile="chat",
        input_tokens=4352,
        output_tokens=32,
        concurrency=8,
        sessions_per_slot=3.0,
        reusable_share=4096 / 4352,
        reason="bench",
    )
    plan = _plan(workload=bench)
    assert plan.working_set_gib == pytest.approx(24 * 4096 * 147456 / GIB, abs=0.01)


def test_qwen3_06b_is_in_the_calibration_set():
    assert QWEN3_06B["num_hidden_layers"] == 28
