"""KV tier planning (Bud FRD-023 Phase 1): plan_kv_tiers() plans T0 and T1."""

import pytest

from llm_memory_calculator import kv_tiering as kt
from llm_memory_calculator.kv_tiering import KVWorkload, plan_kv_tiers

QWEN3_8B = {
    "model_type": "qwen3",
    "hidden_size": 4096,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 12288,
    "vocab_size": 151936,
    "max_position_embeddings": 40960,
}
QWEN3_06B = {
    "model_type": "qwen3",
    "hidden_size": 1024,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 3072,
    "vocab_size": 151936,
    "tie_word_embeddings": True,
    "max_position_embeddings": 40960,
}
LLAMA_70B = {
    "model_type": "llama",
    "hidden_size": 8192,
    "num_hidden_layers": 80,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "intermediate_size": 28672,
    "vocab_size": 128256,
    "max_position_embeddings": 131072,
}
DEEPSEEK_MLA = {
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
BARE_METAL = {"pinned_offload_ok": True, "vgpu": False, "pinned_copy_gbps": 50.0, "pcie_gen": 5}
TCS_VGPU = {
    "pinned_offload_ok": True,
    "vgpu": True,
    "host_pointer_for_registered_mem": False,
    "pinned_copy_gbps": 47.0,
}
REGISTER = ["kv_events", "cache_salt", "cpu_offload_shm"]
ALLOC = ["kv_events", "cache_salt", "cpu_offload_host_alloc"]
H100_TFLOPS = 989.0


def _chat(concurrency=16, input_tokens=8000, output_tokens=2000, **over):
    return KVWorkload.from_deployment(
        input_tokens=input_tokens, output_tokens=output_tokens, concurrency=concurrency, **over
    )


def _plan(**over):
    args = dict(
        model_config=QWEN3_8B,
        engine="vllm",
        engine_kv_features=REGISTER,
        device_type="cuda",
        hardware_mode="dedicated",
        device_memory_gib=80.0,
        weight_gib_per_rank=15.3,
        kv_demand_gib_per_rank=2.0,
        other_device_gib_per_rank=1.0,
        workload=_chat(),
        node=BARE_METAL,
        host_free_gib=1000.0,
        device_tflops=H100_TFLOPS,
    )
    args.update(over)
    return plan_kv_tiers(**args)


def _t1(plan):
    tiers = [t for t in plan.tiers if t["tier"] == "T1"]
    return tiers[0] if tiers else None


# ------------------------------------------------------------------------------------------------
# workload


@pytest.mark.parametrize(
    ("fields", "profile"),
    [
        ({"is_embedding": True}, "off"),
        ({"enable_tool_calling": True}, "agentic"),
        ({"input_tokens": 10000, "output_tokens": 500}, "rag"),
        ({"target_ttft_ms": 300.0}, "interactive"),
        ({}, "chat"),
    ],
)
def test_profile_is_inferred_from_deployment_fields(fields, profile):
    """Infer the reuse profile from existing deployment fields only (FR-PLAN-3)."""
    base = {"input_tokens": 2000, "output_tokens": 1000, "concurrency": 8}
    assert KVWorkload.from_deployment(**{**base, **fields}).profile == profile


def test_working_set_is_sessions_times_reusable_prefix():
    """Count concurrency x sessions per slot x the reusable part of each prompt."""
    workload = _chat(concurrency=10, input_tokens=1000)
    sessions, share = kt.PROFILE_REUSE["chat"]
    assert workload.working_set_tokens == int(10 * sessions * int(1000 * share))


# ------------------------------------------------------------------------------------------------
# off


def test_embeddings_are_planned_off():
    """Plan nothing for an embedding model (FR-PLAN-4)."""
    plan = _plan(workload=_chat(is_embedding=True))
    assert plan.to_wire()["profile"] == "off" and plan.tiers == []


def test_other_engines_are_planned_off():
    """Plan nothing for an engine other than vLLM in this release."""
    plan = _plan(engine="sglang")
    assert plan.profile == "off" and any("sglang" in d for d in plan.decisions)


def test_an_engine_without_kv_features_is_planned_off():
    """Plan nothing when the engine lists no KV features: budcluster would render none of it."""
    assert _plan(engine_kv_features=[]).profile == "off"


# ------------------------------------------------------------------------------------------------
# T0


def test_dedicated_card_gets_the_t0_fraction():
    """Give a dedicated card the T0 fraction (FR-T0-1)."""
    plan = _plan()
    assert plan.gpu_memory_utilization == kt.T0_GPU_UTIL
    assert plan.gpu_pool_gib == pytest.approx(
        0.92 * 80 - 15.3 - 1.0 - kt.DEVICE_OVERHEAD_GIB, abs=0.01
    )


def test_a_working_set_that_fits_t0_plans_no_offload():
    """Plan no offload tier when the working set fits the T0 pool (FR-PLAN-5)."""
    plan = _plan(workload=_chat(concurrency=1, input_tokens=2000))
    assert plan.tiers == [] and any("fits the T0" in d for d in plan.decisions)


def test_a_shared_slice_grows_toward_the_working_set_capped_by_free_gpu_memory():
    """Grow a slice's KV by at most SLICE_KV_SHARE_MAX of the GPU's free memory (FR-T0-3)."""
    plan = _plan(hardware_mode="shared", device_free_gib=20.0)
    assert plan.gpu_kv_gib == pytest.approx(2.0 + kt.SLICE_KV_SHARE_MAX * 20.0, abs=0.01)
    assert plan.gpu_memory_utilization is None


def test_replicas_share_the_slice_growth_cap():
    """Split the growth cap across replicas pinned to the node: they may land on the same card."""
    plan = _plan(hardware_mode="shared", device_free_gib=20.0, replicas=2)
    assert plan.gpu_kv_gib == pytest.approx(2.0 + kt.SLICE_KV_SHARE_MAX * 20.0 / 2, abs=0.01)


def test_a_shared_slice_does_not_grow_when_the_working_set_fits():
    """Keep the demand-sized slice when the working set already fits it."""
    plan = _plan(
        hardware_mode="shared",
        device_free_gib=20.0,
        workload=_chat(concurrency=1, input_tokens=500),
    )
    assert plan.gpu_kv_gib is None


def test_a_shared_slice_does_not_grow_without_knowing_the_free_gpu_memory():
    """Never grow a slice into memory whose availability is unknown."""
    assert _plan(hardware_mode="shared", device_free_gib=None).gpu_kv_gib is None


def test_cpu_engines_grow_their_kv_space_within_free_host_memory_and_get_no_t1():
    """Grow a CPU engine's KV space toward the working set within free host RAM (FR-T0-4)."""
    plan = _plan(
        device_type="cpu_high",
        device_memory_gib=None,
        host_free_gib=20.0,
        workload=_chat(concurrency=4),
    )
    assert plan.cpu_kv_gib == pytest.approx(2.0 + kt.CPU_KV_SHARE_MAX * 20.0, abs=0.01)
    assert plan.tiers == []


# ------------------------------------------------------------------------------------------------
# T1


def test_bare_metal_plans_register_mode_t1():
    """Plan T1 in register mode on a node whose GPU can use registered memory."""
    plan = _plan()
    tier = _t1(plan)
    assert tier["params"] == {"host_memory": "register", "block_size": kt.T1_BLOCK_SIZE}
    pool = plan.gpu_pool_gib
    assert (
        kt.T1_MIN_POOL_MULTIPLE * pool - 0.3 <= tier["size_gib"] <= kt.T1_MAX_POOL_MULTIPLE * pool
    )
    assert plan.to_wire()["resources"]["shm_gib"] == tier["size_gib"]
    assert plan.to_wire()["resources"]["host_memory_gib"] == tier["size_gib"]


def test_tcs_vgpu_plans_alloc_mode_t1_when_the_engine_has_it():
    """Plan alloc mode where registered memory is unusable (spike S2), with no /dev/shm."""
    plan = _plan(node=TCS_VGPU, engine_kv_features=ALLOC)
    assert _t1(plan)["params"]["host_memory"] == "alloc"
    assert plan.shm_gib == 0.0


def test_tcs_vgpu_plans_no_t1_on_an_engine_without_alloc_mode():
    """Never plan register mode on a node where it crashes the engine."""
    plan = _plan(node=TCS_VGPU, engine_kv_features=REGISTER)
    assert _t1(plan) is None
    assert any("cpu_offload_host_alloc" in d for d in plan.decisions)


def test_an_unprobed_vgpu_node_is_planned_alloc():
    """Treat an unprobed vGPU node as unable to use registered memory."""
    node = {"pinned_offload_ok": True, "vgpu": True}
    assert _t1(_plan(node=node, engine_kv_features=ALLOC))["params"]["host_memory"] == "alloc"
    assert _t1(_plan(node=node, engine_kv_features=REGISTER)) is None


@pytest.mark.parametrize("pinned", [None, False])
def test_t1_needs_verified_pinned_offload(pinned):
    """Plan no T1 unless the node's pinned offload probe passed (FR-T1-2)."""
    assert _t1(_plan(node={**BARE_METAL, "pinned_offload_ok": pinned})) is None


def test_t1_is_bounded_by_free_host_memory_shared_by_colocated_replicas():
    """Size T1 within T1_HOST_RAM_SHARE of free host memory, split across the replicas."""
    tier = _t1(_plan(host_free_gib=400.0, replicas=2))
    assert tier["size_gib"] <= kt.T1_HOST_RAM_SHARE * 400.0 / 2


def test_t1_is_dropped_when_host_memory_cannot_hold_more_than_the_gpu_pool():
    """Drop T1 when the host memory left for it is smaller than the GPU pool it would back."""
    plan = _plan(host_free_gib=50.0)
    assert _t1(plan) is None and any("host memory" in d for d in plan.decisions)


def test_t1_needs_known_free_host_memory():
    """Never size T1 into host memory whose availability is unknown."""
    assert _t1(_plan(host_free_gib=None)) is None


def test_t1_is_single_node_only():
    """Plan no T1 for a pipeline-parallel (Ray) deployment."""
    assert _t1(_plan(pipeline_parallel=2)) is None


def test_break_even_drops_t1_for_a_small_model_on_a_fast_gpu():
    """Drop T1 where loading costs about as much as recomputing (S2: Qwen3-0.6B on an H100)."""
    plan = _plan(
        model_config=QWEN3_06B,
        weight_gib_per_rank=1.4,
        workload=_chat(concurrency=64, input_tokens=4000, output_tokens=1000),
    )
    assert _t1(plan) is None and any("costs about as much" in d for d in plan.decisions)


@pytest.mark.parametrize("model", ["8b", "70b"])
@pytest.mark.parametrize("context", [8192, 32768, 131072])
def test_break_even_keeps_t1_for_large_prefixes_on_gen5(model, context):
    """Keep T1 for 8B and 70B models at 8K, 32K and 128K on a Gen5 node (FRD NFR-2)."""
    config, weights = (QWEN3_8B, 15.3) if model == "8b" else (LLAMA_70B, 65.0)
    # chat-shaped (input:output under 8) and enough sessions that the working set overflows the GPU
    # pool
    workload = _chat(concurrency=32, input_tokens=context, output_tokens=context // 2)
    plan = _plan(
        model_config=config, weight_gib_per_rank=weights, workload=workload, host_free_gib=4000.0
    )
    assert _t1(plan) is not None, plan.decisions


def test_unknown_device_flops_skip_the_break_even_check_but_say_so():
    """Keep T1 when break-even can't be judged, and record that it wasn't checked."""
    plan = _plan(device_tflops=None)
    assert _t1(plan) is not None and any("not checked" in d for d in plan.decisions)


# ------------------------------------------------------------------------------------------------
# KV bytes


def test_mla_latent_is_counted_on_every_tensor_parallel_rank():
    """Count an MLA latent in full on each TP rank, and split GQA KV across ranks."""
    mla_rank, mla_total, attention = kt.kv_bytes_per_token_per_rank(
        DEEPSEEK_MLA, seq_length=4096, tensor_parallel=2
    )
    gqa_rank, gqa_total, _ = kt.kv_bytes_per_token_per_rank(
        QWEN3_8B, seq_length=4096, tensor_parallel=2
    )
    assert attention == "mla" and mla_rank == mla_total
    assert gqa_rank == gqa_total / 2
    dp_rank, _, _ = kt.kv_bytes_per_token_per_rank(
        DEEPSEEK_MLA, seq_length=4096, tensor_parallel=2, data_parallel_attention=True
    )
    assert dp_rank == mla_total / 2


def test_kv_bytes_come_from_the_calculator():
    """Use kv_cache_breakdown's bytes: Qwen3-8B is 144 KiB per token in bf16, as vLLM allocates."""
    _, total, _ = kt.kv_bytes_per_token_per_rank(QWEN3_8B, seq_length=32768)
    assert total == 147456


# ------------------------------------------------------------------------------------------------
# dtype


def test_fp8_only_where_validated(monkeypatch):
    """Keep the model's KV dtype unless (model family, GPU generation) is validated (FR-QUANT-2)."""
    assert _plan(device_generation="hopper").kv_cache_dtype == "auto"
    monkeypatch.setattr(kt, "FP8_KV_VALIDATED", frozenset({("qwen3", "hopper")}))
    assert _plan(device_generation="hopper").kv_cache_dtype == "fp8"


# ------------------------------------------------------------------------------------------------
# plan


def test_the_wire_shape_is_frd_023_section_8_1():
    """Emit exactly the §8.1 keys budcluster parses."""
    wire = _plan().to_wire()
    assert set(wire) == {
        "version",
        "namespace",
        "profile",
        "kv_cache_dtype",
        "skip_layers_sliding_window",
        "hash_algo",
        "gpu_memory_utilization",
        "gpu_kv_gib",
        "cpu_kv_gib",
        "tiers",
        "events",
        "routing",
        "pd",
        "predicted",
        "resources",
        "decisions",
    }
    assert wire["version"] == 1 and wire["events"]["enabled"] is True
    assert set(_t1(_plan())) == {
        "tier",
        "backend",
        "size_gib",
        "storage_class",
        "transport",
        "params",
    }


def test_the_same_inputs_give_the_same_plan():
    """Plan deterministically (NFR-6)."""
    assert _plan().to_wire() == _plan().to_wire()


def test_every_plan_explains_itself():
    """Record the profile, the working set, T0 and T1 in the decisions (FR-PLAN-7)."""
    decisions = " | ".join(_plan().decisions)
    for fragment in ("profile: chat", "working set", "T0 (dedicated)", "T1 mode: register", "T1:"):
        assert fragment in decisions


def test_prediction_credits_t1_hits():
    """Predict a higher hit rate and lower TTFT with T1 than without (FR-PLAN-8)."""
    with_t1 = _plan()
    without = _plan(node={**BARE_METAL, "pinned_offload_ok": False})
    assert with_t1.predicted_hit_rate > without.predicted_hit_rate
    assert with_t1.predicted_ttft_ms < without.predicted_ttft_ms


# ------------------------------------------------------------------------------------------------
# GenZ MemoryModel


def test_genz_memory_model_takes_the_calculators_per_token_kv():
    """Let GenZ's MemoryModel use the calculator's per-token KV, not its GQA formula (FR-PLAN-2)."""
    from types import SimpleNamespace

    from llm_memory_calculator.genz.serving.constants import MemoryTier
    from llm_memory_calculator.genz.serving.memory_model import MemoryModel, MemoryTierConfig

    config = SimpleNamespace(
        num_key_value_heads=16, head_dim=128, num_decoder_layers=27, num_attention_heads=16
    )
    tier = MemoryTierConfig(
        tier=MemoryTier.DEVICE_HBM, capacity_bytes=1 << 34, bandwidth_gbps=3000.0
    )
    per_rank, _, _ = kt.kv_bytes_per_token_per_rank(
        DEEPSEEK_MLA, seq_length=4096, tensor_parallel=2
    )

    legacy = MemoryModel(config, [tier], tensor_parallel=2)
    fixed = MemoryModel(config, [tier], tensor_parallel=2, bytes_per_token_kv=int(per_rank))

    assert fixed.bytes_per_token_kv == int(per_rank)
    assert (
        legacy.bytes_per_token_kv != fixed.bytes_per_token_kv
    )  # the GQA formula misprices an MLA latent


def test_the_callers_placement_cap_bounds_the_slice_growth():
    """A grown slice the cards can't hold per replica is cut to the caller's cap, never rounded past it."""
    free = _plan(hardware_mode="shared", device_free_gib=20.0)
    capped = _plan(
        hardware_mode="shared", device_free_gib=20.0, gpu_kv_cap_gib=free.gpu_kv_gib - 3.337
    )
    assert capped.gpu_kv_gib == pytest.approx(free.gpu_kv_gib - 3.34, abs=1e-9)
    assert any("as much as still fits" in d for d in capped.decisions)
    # T1 is sized against the capped pool, not the uncapped one.
    assert capped.gpu_pool_gib == capped.gpu_kv_gib
    # A cap at or below the demand: no growth at all.
    none = _plan(hardware_mode="shared", device_free_gib=20.0, gpu_kv_cap_gib=0.5)
    assert none.gpu_kv_gib is None
    # A cap above the growth changes nothing.
    loose = _plan(hardware_mode="shared", device_free_gib=20.0, gpu_kv_cap_gib=500.0)
    assert loose.gpu_kv_gib == free.gpu_kv_gib


def test_the_t1_decision_names_what_sized_it():
    """The reason is decided before the quarter-GiB rounding, so rounding never reads as a host-memory limit."""

    def sized_by(plan):
        return next(d for d in plan.decisions if d.startswith("T1: ")).split("sized by ")[1]

    # Qwen3-8B on an 80 GiB card: a ~56 GiB pool; the working set is ~5.3 GiB per unit of concurrency.
    assert (
        sized_by(_plan(workload=_chat(concurrency=64), host_free_gib=2000.0)) == "the working set"
    )
    assert sized_by(_plan(workload=_chat(concurrency=64), host_free_gib=400.0)) == "the host memory"
    assert (
        sized_by(_plan(workload=_chat(concurrency=256), host_free_gib=4000.0)) == "10x the GPU pool"
    )
    assert (
        sized_by(_plan(workload=_chat(concurrency=16), host_free_gib=2000.0)) == "3x the GPU pool"
    )
