"""FRD-027 §8, T-ENG-01…12: the vLLM engine layer -- allocator, limits, boot checks, validate, boot time."""

import pytest

from llm_memory_calculator.device import HardwareMode, catalogue
from llm_memory_calculator.engine import (
    EngineRecord,
    UnsupportedSetting,
    engine_for,
    Workload,
)
from llm_memory_calculator.model import ModelFacts

from .conftest import load_model, load_record

GIB = 1024**3
VGPU = catalogue.get("h100xm-80c-vgpu")


def eng(name="vllm-0.30-upstream"):
    return engine_for(EngineRecord.from_dict(load_record(name)))


def model(name):
    return ModelFacts.from_config(load_model(name))


Q27, Q8, OSS = model("qwen3.8-27b"), model("qwen3-8b"), model("gpt-oss-20b")
STOCK27 = {"batch.max_running": 256}  # the study's "vanilla" 27B


# ---------------------------------------------------------------- T-ENG-01 allocator geometry


@pytest.mark.parametrize(
    "settings, block",
    [
        ({}, 784),
        ({"kv.dtype": "fp8"}, 1568),
        ({"state.dtype": "bfloat16"}, 400),
        ({"kv.dtype": "fp8", "state.dtype": "bfloat16"}, 800),
        ({"spec.method": "mtp", "spec.tokens": 1}, 800),
        ({"spec.method": "mtp", "spec.tokens": 3, "state.dtype": "bfloat16"}, 416),
        (
            {"spec.method": "mtp", "spec.tokens": 2, "kv.dtype": "fp8", "state.dtype": "bfloat16"},
            832,
        ),
        ({"spec.method": "mtp", "spec.tokens": 3, "kv.dtype": "fp8"}, 1600),
    ],
)
def test_hybrid_block_size(settings, block):
    """Measured attn_block_size for every 27B variant in the study (startup.json)."""
    lim = eng().limits({**STOCK27, **settings}, Q27, VGPU)
    assert lim.block_tokens == block


def test_dense_block_is_16():
    assert eng().limits({}, Q8, VGPU).block_tokens == 16


def test_kv_geometry_shim_unchanged():
    from llm_memory_calculator.engine.vllm_geometry import KVGeometry as Moved
    from llm_memory_calculator.kv.geometry import KVGeometry

    assert Moved is KVGeometry
    assert KVGeometry.from_model(load_model("qwen3.8-27b"), seq_length=4096).block_size == 784


# ---------------------------------------------------------------- T-ENG-02 sequence ceiling


def test_sequence_ceiling_matches_the_refusal():
    # study s-seqs512-cg256: "max_num_seqs (512) exceeds available Mamba cache blocks (392)"
    lim = eng().limits({"batch.max_running": 512, "memory.device_fraction": 0.92}, Q27, VGPU)
    assert abs(lim.sequence_ceiling - 392) <= 4
    assert not lim.boots and any(r.check == "sequence_ceiling" for r in lim.failures)


def test_bf16_state_roughly_doubles_the_ceiling():
    fp32 = eng().limits(STOCK27, Q27, VGPU).sequence_ceiling
    bf16 = eng().limits({**STOCK27, "state.dtype": "bfloat16"}, Q27, VGPU).sequence_ceiling
    assert 700 <= bf16 <= 820 and bf16 > 1.8 * fp32


def test_dense_ceiling_from_kv_blocks():
    lim = eng().limits({}, Q8, VGPU)
    assert lim.sequence_ceiling == lim.num_blocks and lim.sequence_ceiling > 20000


def test_visible_memory_moves_the_ceiling():
    bigger = VGPU.__class__(
        **{**VGPU.__dict__, "memory_visible_bytes": VGPU.memory_visible_bytes + GIB}
    )
    base = eng().limits(STOCK27, Q27, VGPU)
    more = eng().limits(STOCK27, Q27, bigger)
    # +1 GiB x 0.92 of KV pool -> about 0.92 GiB / 49 MiB = ~19 more blocks
    assert 15 <= more.sequence_ceiling - base.sequence_ceiling <= 22
    # the bare-metal SKU has the same visible memory but is grounded at the Hopper architecture level (FR-DEV-5)
    bare = eng().limits(STOCK27, Q27, catalogue.get("h100-sxm5-80gb"))
    assert bare.levels["hardware"] == "arch" and base.levels["hardware"] == "sku"


# ---------------------------------------------------------------- T-ENG-03 profiling transient


def test_stock_27b_ooms_on_the_profiling_transient():
    stock = eng().limits({}, Q27, VGPU)  # engine default max_num_seqs 1024 -> 512 captured
    assert stock.profiling_transient_bytes / GIB == pytest.approx(24.5, rel=0.03)
    assert not stock.boots and any(r.check == "profiling_transient" for r in stock.failures)
    assert eng().limits(STOCK27, Q27, VGPU).boots


def test_bf16_state_does_not_shrink_the_profiling_transient():
    """TCS 10-18: the fork 27B with bf16 state, 256 sequences, on a 65,024 MiB HAMi slice OOMed in
    profile_cudagraph_memory with 62.27 GiB allocated (weights 51.1 GiB) -- the transient is sized by the model's own
    (fp32) state page, not the halved bf16 one. On that slice the plan must not boot; on a dedicated card it does."""
    fork = eng("bud-vllm-0.9.0-481e481be")
    s = {"memory.device_fraction": 0.94, "batch.max_running": 256, "context.max_tokens": 95984, "state.dtype": "bfloat16"}
    sliced = VGPU.with_mode(HardwareMode("shared", 65024 * 1024**2))
    lim = fork.limits(s, Q27, sliced)
    assert lim.profiling_transient_bytes / GIB == pytest.approx(12.25, rel=0.05)
    assert not lim.boots and any(r.check == "profiling_transient" for r in lim.failures)
    assert fork.limits(s, Q27, VGPU).boots


def test_the_profiling_transient_sets_a_sequence_ceiling():
    """gtech 10-20: the FP8 27B (28.3 GiB) on a 45 GB L20 at the planning default of 256 sequences needs 12.2 GiB of
    CUDA-graph profiling blocks with 11.4 GiB free. ``transient_ceiling`` is the most sequences whose profiling
    allocation fits; at it the engine boots, one above it does not."""
    from llm_memory_calculator.device import facts_from_node

    l20 = facts_from_node({"product": "NVIDIA-L20", "type": "cuda", "mem_per_gpu_in_gb": 44.99})
    fork = eng("bud-vllm-0.9.0-481e481be")
    q27fp8 = model("qwen3.8-27b-fp8")
    s = {"batch.max_running": 256, "state.dtype": "bfloat16", "context.max_tokens": 32768}
    lim = fork.limits(s, q27fp8, l20)
    assert [c.check for c in lim.failures] == ["profiling_transient"]
    ceiling = lim.transient_ceiling
    assert ceiling is not None and 16 <= ceiling < 256
    assert fork.limits({**s, "batch.max_running": ceiling}, q27fp8, l20).boots
    over = fork.limits({**s, "batch.max_running": ceiling + 8}, q27fp8, l20)
    assert any(c.check == "profiling_transient" for c in over.failures)
    assert lim.to_dict()["transient_ceiling"] == ceiling


def test_the_transient_ceiling_agrees_with_the_stock_27b():
    """The stock 27B OOMs with 512 sequences captured and boots with 256 (T-ENG-03): its ceiling lies between. A dense
    model makes no hybrid profiling allocation, so it has none."""
    ceiling = eng().limits(STOCK27, Q27, VGPU).transient_ceiling
    assert ceiling is not None and 256 <= ceiling < 512
    assert eng().limits({}, Q8, VGPU).transient_ceiling is None


def test_stock_8b_and_oss_boot():
    assert eng().limits({}, Q8, VGPU).boots
    assert eng().limits({}, OSS, VGPU).boots


# ---------------------------------------------------------------- T-ENG-04 utilization ceiling


def test_utilization_ceiling():
    assert not eng().limits({**STOCK27, "memory.device_fraction": 0.97}, Q27, VGPU).boots
    ok = eng().limits({**STOCK27, "memory.device_fraction": 0.95}, Q27, VGPU)
    assert ok.boots
    margin = [c for c in ok.checks if c.check == "device_fraction"][0]
    assert 0 <= margin.margin_bytes < 2 * GIB


def test_low_utilization_cannot_hold_one_request():
    # study 8b-s-util024: 0.04 GiB available, 5.62 GiB needed for 40960
    lim = eng().limits({"memory.device_fraction": 0.24}, Q8, VGPU)
    assert not lim.boots and any(r.check == "one_request_fits" for r in lim.failures)
    assert lim.max_bootable_context < 40960


# ---------------------------------------------------------------- bootable context


def test_mtp_bootable_context_below_262144():
    # study s-mtp1: "the estimated maximum model length is 254400"
    lim = eng().limits({**STOCK27, "spec.method": "mtp", "spec.tokens": 1}, Q27, VGPU)
    assert 245_000 <= lim.max_bootable_context <= 262_000
    assert lim.max_bootable_context < 262144
    refuse = eng().limits(
        {**STOCK27, "spec.method": "mtp", "spec.tokens": 1, "context.max_tokens": 262144}, Q27, VGPU
    )
    assert not refuse.boots


def test_kv_pool_matches_measured():
    """Memory within 3% of the measured KV pool on the base configurations."""
    cases = [
        (Q27, STOCK27, 18.98),
        (Q8, {}, 54.22),
        (OSS, {}, 56.58),
        (model("qwen3.8-27b-fp8"), STOCK27, 41.33),
    ]
    for facts, settings, measured in cases:
        lim = eng().limits(settings, facts, VGPU)
        assert lim.kv_pool_bytes / GIB == pytest.approx(measured, rel=0.03), facts.model_type


# ---------------------------------------------------------------- T-ENG-05..07 validate


def test_validate_mtp_context():
    e = eng()
    refusals = e.validate({"spec.method": "mtp", "context.max_tokens": 262144})
    assert [r.conflict for r in refusals] == ["mtp-context-262144"]
    assert e.validate({"spec.method": "mtp", "context.max_tokens": 131072}) == []


def test_validate_spec_and_prefix_cache():
    refusals = eng().validate({"spec.method": "eagle3", "prefix_cache.enabled": True})
    assert [r.conflict for r in refusals] == ["spec-prefix-cache"]
    rec = load_record("vllm-0.30-upstream")
    rec["features"] = rec["features"] + ["spec_prefix_cache_safe"]
    safe = engine_for(EngineRecord.from_dict(rec))
    assert safe.validate({"spec.method": "eagle3", "prefix_cache.enabled": True}) == []



def test_a_full_attention_waiver_is_scoped_by_the_model():
    """T-ENG-13: ``spec_prefix_cache_safe_full_attention`` lifts the spec-decode/prefix-cache refusal only for a model
    whose layers are all full attention. #53912 lives in the hybrid (Mamba/GDN) cache-hit path and sliding-window
    layers take their own path, so neither is covered by the dense scans that verified the feature."""
    rec = load_record("vllm-0.30-upstream")
    rec["features"] = [f for f in rec["features"] if not f.startswith("spec_prefix_cache_safe")] + [
        "spec_prefix_cache_safe_full_attention"
    ]
    scoped = engine_for(EngineRecord.from_dict(rec))
    s = {"spec.method": "eagle3", "prefix_cache.enabled": True}
    assert scoped.validate(s, Q8) == []
    mtp = {**s, "spec.method": "mtp", "context.max_tokens": 131072}
    assert [r.conflict for r in scoped.validate(mtp, Q27)] == ["spec-prefix-cache"]
    assert [r.conflict for r in scoped.validate(s, OSS)] == ["spec-prefix-cache"]  # sliding-window layers
    assert [r.conflict for r in scoped.validate(s)] == ["spec-prefix-cache"]  # an unknown model is not waived
    assert Q8.full_attention and not Q27.full_attention and not OSS.full_attention
    assert scoped.record.spec_prefix_cache_safe(Q8) and not scoped.record.spec_prefix_cache_safe(Q27)
    assert not eng("bud-vllm-0.9.0-481e481be").record.spec_prefix_cache_safe(Q8)

def test_validate_record_specific_conflict():
    fork, up = eng("bud-vllm-0.9.0-481e481be"), eng()
    s = {"spec.method": "eagle3", "memory.device_fraction": 0.92, "prefix_cache.enabled": False}
    assert [r.conflict for r in fork.validate(s)] == ["fork-eagle3-utilization"]
    assert up.validate(s) == []
    s["memory.device_fraction"] = 0.85
    assert fork.validate(s) == [] and up.validate(s) == []


def test_unknown_setting_in_conflict_is_rejected():
    rec = load_record("vllm-0.30-upstream")
    rec["conflicts"].append(
        {"id": "bad", "when": {"no.such": 1}, "requires": {"context.max_tokens": "< 1"}}
    )
    with pytest.raises(ValueError, match="no.such"):
        EngineRecord.from_dict(rec)


# ---------------------------------------------------------------- T-ENG-08 speculative memory


def test_mtp_state_is_per_recurrent_group():
    lim = eng().limits(
        {**STOCK27, "spec.method": "mtp", "spec.tokens": 3, "context.max_tokens": 131072}, Q27, VGPU
    )
    # (1 + k) blocks of 800 tokens (50 MiB) in each of the 3 recurrent groups per running sequence: 0.586 GiB, as the
    # FRD first said. The engine logs settle it: s-mtp3 never runs more than 24 sequences (310 blocks / (2 + 3 x 4)),
    # s-mtp1 tops out at 46 with 18 waiting (an earlier reading of these runs as "reach c64" was wrong).
    assert lim.block_tokens == 800
    assert lim.state_bytes_per_sequence / GIB == pytest.approx(3 * 4 * 50 / 1024, rel=0.01)
    base = eng().limits({**STOCK27, "context.max_tokens": 131072}, Q27, VGPU)
    assert lim.kv_pool_bytes < base.kv_pool_bytes  # MTP weights + graphs shrink the pool


@pytest.mark.parametrize(
    "record,settings,workload,observed",
    [
        # the Bud fork's 27B stock run (bud-img-vanilla): "Running: 83 reqs, Waiting: 45, KV 100%" at 1280 tokens,
        # 28-29 running at 8448 tokens -- each sequence holds its attention blocks plus one block per recurrent group
        ("bud-vllm-0.9.0-481e481be", {"batch.max_running": 256}, (1024, 256, 128), (83, 83)),
        ("bud-vllm-0.9.0-481e481be", {"batch.max_running": 256}, (8192, 256, 32), (28, 29)),
        # upstream 0.30, MTP k=3 (s-mtp3-l128k): at most 24 running
        (
            "vllm-0.30-upstream",
            {"spec.method": "mtp", "spec.tokens": 3, "context.max_tokens": 131072},
            (1024, 256, 64),
            (21, 25),
        ),
    ],
)
def test_hybrid_running_capacity_matches_the_engine_logs(record, settings, workload, observed):
    """T-ENG-04: the running ceiling from KV blocks is attention blocks + (1 + k) blocks per recurrent group."""
    inp, out, c = workload
    pred = eng(record).predict(
        {**STOCK27, **settings}, Workload(inp, out, c, workload_class="synthetic"), Q27, VGPU
    )
    lo, hi = observed
    assert lo <= pred.running <= hi, pred.running
    assert pred.queueing


def test_eagle3_draft_memory_is_charged():
    draft = ModelFacts.from_config(load_model("eagle3-qwen3-8b-redhatai"))
    base = eng().limits({}, Q8, VGPU)
    spec = eng().limits({"spec.method": "eagle3", "spec.tokens": 3}, Q8, VGPU, draft=draft)
    # measured: 8B + RedHatAI draft loads 17.17 GiB (vs 15.27) and the pool drops 54.22 -> 50.67 GiB
    assert (spec.weights_bytes - base.weights_bytes) / GIB == pytest.approx(1.9, abs=0.4)
    assert spec.kv_pool_bytes / GIB == pytest.approx(50.67, rel=0.04)


# ---------------------------------------------------------------- T-ENG-09 shared mode


def test_shared_mode_predicts_with_the_slice():
    shared = VGPU.with_mode(HardwareMode(kind="shared", slice_bytes=66956 * 1024**2))
    dedicated = eng().limits(
        {"batch.max_running": 64, "memory.device_fraction": 0.94, "context.max_tokens": 65536},
        Q27,
        VGPU,
    )
    sliced = eng().limits(
        {"batch.max_running": 64, "memory.device_fraction": 0.94, "context.max_tokens": 65536},
        Q27,
        shared,
    )
    # study s-slice66956-seqs64-l64k: 122,333 tokens on the slice
    assert sliced.kv_pool_bytes < 0.5 * dedicated.kv_pool_bytes
    assert sliced.boots
    w = Workload(input_tokens=1024, output_tokens=256, concurrency=64)
    p_shared = eng().predict(
        {"batch.max_running": 64, "memory.device_fraction": 0.94, "context.max_tokens": 65536},
        w,
        Q27,
        shared,
    )
    p_dedicated = eng().predict(
        {"batch.max_running": 64, "memory.device_fraction": 0.94, "context.max_tokens": 65536},
        w,
        Q27,
        VGPU,
    )
    assert "contention: unmodelled" in p_shared.notes
    assert p_shared.throughput_tok_s.width >= p_dedicated.throughput_tok_s.width


# ---------------------------------------------------------------- T-ENG-10 boot time


def test_boot_time_monotonic():
    e = eng()
    small = e.boot_time({"batch.max_running": 16}, Q27, VGPU)
    large = e.boot_time({"batch.max_running": 256}, Q27, VGPU)
    assert large.value >= small.value
    assert (
        e.boot_time({"batch.max_running": 256, "context.max_tokens": 262144}, Q27, VGPU).value
        >= e.boot_time({"batch.max_running": 256, "context.max_tokens": 8192}, Q27, VGPU).value
    )
    assert small.low <= small.value <= small.high


# ---------------------------------------------------------------- T-ENG-11 unsupported settings


def test_unsupported_settings():
    sgl = engine_for(EngineRecord.from_dict(load_record("sglang-stub")))
    assert sgl.supports("spec.method") is False
    assert sgl.supports("batch.max_running") is True
    with pytest.raises(UnsupportedSetting):
        sgl.check_supported({"spec.method": "eagle3"})
    assert eng().supports("spec.method") is True


def test_predict_refuses_unsupported_setting_on_vllm_record_without_feature():
    fork = eng("bud-vllm-0.9.0-481e481be")
    w = Workload(input_tokens=1024, output_tokens=256, concurrency=1)
    with pytest.raises(UnsupportedSetting):
        fork.predict({**STOCK27, "spec.method": "mtp", "spec.tokens": 2}, w, Q27, VGPU)


# ---------------------------------------------------------------- prediction sanity


def test_prediction_shape_and_levels():
    w = Workload(input_tokens=1024, output_tokens=256, concurrency=16)
    p = eng().predict(STOCK27, w, Q27, VGPU)
    for est in (p.ttft_ms, p.tpot_ms, p.throughput_tok_s):
        assert est.low <= est.value <= est.high and est.value > 0
    assert set(p.levels) == {"hardware", "engine", "model"}
    assert p.limits.boots


def test_max_running_cap_queues():
    """Bud's legacy render (max-num-seqs 2) at c16: ~96 tok/s and a TTFT of tens of seconds (study §4)."""
    w = Workload(input_tokens=1024, output_tokens=256, concurrency=16)
    capped = eng().predict({"batch.max_running": 2, "memory.device_fraction": 0.92}, w, Q27, VGPU)
    free = eng().predict(STOCK27, w, Q27, VGPU)
    assert capped.queueing and not free.queueing
    assert capped.throughput_tok_s.value < 0.3 * free.throughput_tok_s.value
    assert capped.ttft_ms.value > 10_000


def test_queue_wait_reaches_the_median_ttft_only_when_arrivals_wait():
    """FRD-027 §8: over the KV capacity, a decode-heavy load preempts growing requests and arrivals still get their
    first token; a prefill-heavy load holds its blocks from admission and arrivals wait for completions.
    """
    fork = eng("bud-vllm-0.9.0-481e481be")
    decode = fork.predict(STOCK27, Workload(256, 1024, 128, workload_class="synthetic"), Q27, VGPU)
    assert decode.queueing and decode.running == 83
    assert decode.ttft_ms.value < 5000  # bud-img-vanilla decode c128: median TTFT 2.3 s
    chat = fork.predict(STOCK27, Workload(1024, 256, 256, workload_class="synthetic"), Q27, VGPU)
    assert (
        chat.queueing and chat.ttft_ms.value > 15000
    )  # bud-img-vanilla chat c256: median TTFT 31.5 s
