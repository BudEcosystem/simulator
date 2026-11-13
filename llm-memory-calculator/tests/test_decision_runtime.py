"""Bud spec 031 DM4: sizing Bud-Decision-Runtime replicas (``llm_memory_calculator.decision_runtime``).

TC-DO-07: facts from the family files match the checkpoints; precision follows D-35; the calibration's entries are
reproduced by the tool from the runtime's benchmark results. TC-DO-20 (offline part): a plan calibrated on one load of
the benchmark predicts the other, unseen, load on the same device.
"""

import functools
import importlib.util
import json
import math
import os

import pytest

from llm_memory_calculator import decision_runtime as dr
from llm_memory_calculator.decision_runtime import calibration, families, plan

HERE = os.path.dirname(__file__)
FIX = os.path.join(HERE, "fixtures", "decision_runtime")
BASELINES = os.path.join(FIX, "baselines")

RTX3050 = {"type": "cuda", "name": "NVIDIA GeForce RTX 3050 Laptop GPU", "mem_per_GPU_in_GB": 4}
GTX1650 = {"type": "cuda", "name": "NVIDIA GeForce GTX 1650", "mem_per_GPU_in_GB": 4}
A10 = {"type": "cuda", "name": "NVIDIA A10", "mem_per_GPU_in_GB": 24}
H100 = {"type": "cuda", "name": "NVIDIA H100 80GB HBM3", "mem_per_GPU_in_GB": 80}
XEON = {"type": "cpu", "name": "Intel(R) Xeon(R) 6767P", "cpu_amx": True, "cpu_avx512": True, "physical_cores": 128,
        "numa_nodes": 2}
I7 = {"type": "cpu", "name": "11th Gen Intel(R) Core(TM) i7-11370H", "cpu_amx": False, "cpu_avx512": True,
      "physical_cores": 4, "numa_nodes": 1}
AVX2 = {"type": "cpu_high", "name": "AMD EPYC 7763", "cpu_amx": False, "cpu_avx512": False, "physical_cores": 64}
SPR = {"type": "cpu", "name": "Intel(R) Xeon(R) Platinum 8480+", "cpu_amx": True, "cpu_avx512": True,
       "physical_cores": 112, "numa_nodes": 2}
A100 = {"type": "cuda", "name": "NVIDIA A100-SXM4-40GB", "mem_per_GPU_in_GB": 40}
GIB = 2**30


@functools.lru_cache(maxsize=1)
def _tool():
    """tools/decision_calibration.py, loaded once."""
    spec = importlib.util.spec_from_file_location("decision_calibration",
                                                  os.path.join(HERE, "..", "tools", "decision_calibration.py"))
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def _facts(family):
    if family == "kev":
        return dr.facts(os.path.join(FIX, "kev"), "kev", base_dir=os.path.join(FIX, "qwen2.5-0.5b"))
    return dr.facts(os.path.join(FIX, family), family)


def _wl(**kw):
    out = {"calls": 10, "questions_per_call": 3, "state_tokens": 60, "question_tokens": 20,
           "target_call_latency_ms": 500}
    out.update(kw)
    return out


# ---------------------------------------------------------------------------------------------------------- families

#: basal_engine.models.FAMILIES at v0.1.0, typed from the runtime; "instant" is its built-in test model
RUNTIME_FAMILIES_V010 = ("laya", "julia", "kev", "gliner", "instant")


def test_family_table_matches_the_runtime():
    """Against the installed runtime when there is one; the v0.1.0 snapshot otherwise (it cannot see a rename)."""
    if importlib.util.find_spec("basal_engine") is not None:
        from basal_engine.models import FAMILIES as runtime_families  # noqa: N811 - the runtime's own name
    else:
        runtime_families = RUNTIME_FAMILIES_V010
    assert set(dr.FAMILIES) | set(dr.NOT_SIZED) == set(runtime_families)


def test_untied_embeddings_are_the_default():
    """transformers' Llama / Qwen2 / Qwen3 / Mistral configs default tie_word_embeddings to False."""
    cfg = {"model_type": "llama", "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
           "num_attention_heads": 4, "vocab_size": 1000}
    untied = dr.facts_from_configs("kev", cfg, {})
    tied = dr.facts_from_configs("kev", {**cfg, "tie_word_embeddings": True}, {})
    assert untied.embedding_params - tied.embedding_params == 1000 * 64


@pytest.mark.parametrize("family,checkpoint", [("laya", "laya"), ("julia", "julia"), ("gliner", "gliner")])
def test_params_match_the_checkpoint(family, checkpoint):
    expected = json.load(open(os.path.join(FIX, "checkpoint_params.json")))[checkpoint]
    assert _facts(family).params_total == pytest.approx(expected, rel=0.01)


def test_kev_counts_base_and_adapter():
    expected = json.load(open(os.path.join(FIX, "checkpoint_params.json")))
    f = _facts("kev")
    assert f.embedding_params + f.layer_params == pytest.approx(expected["qwen2.5-0.5b"], rel=0.01)
    assert f.lora_params == pytest.approx(expected["kev-adapter"], rel=0.01)
    assert f.keeps_state and not f.bidirectional and f.layout == families.BRANCH


def test_laya_layers_and_window():
    f = _facts("laya")
    assert (f.full_layers, f.window_layers, f.window, f.max_len) == (10, 18, 128, 512)
    assert f.backbone == "modernbert" and f.layout == families.PER_QUESTION


def test_encoder_attention_is_bidirectional_and_windowed():
    f = _facts("laya")
    width = f.num_heads * f.head_dim
    # a 400-token row: full layers over 400, window layers over 128
    assert f.attention_flops_per_token(400) == 4.0 * width * (10 * 400 + 18 * 128)
    # a decoder token attends to half its row on average
    kev = _facts("kev")
    kw = kev.num_heads * kev.head_dim
    assert kev.attention_flops_per_token(400) == 4.0 * kw * kev.num_layers * 200


def test_deberta_adds_relative_position_scores():
    g = _facts("gliner")
    width = g.num_heads * g.head_dim
    plain = 4.0 * width * g.num_layers * 1000
    assert g.attention_flops_per_token(1000) == plain + 4.0 * width * g.num_layers * min(1000, 512)


def test_missing_files_and_unknown_family_are_refused(tmp_path):
    with pytest.raises(dr.FamilyError, match="rl_agent_config.json is missing"):
        dr.facts(str(tmp_path), "laya")
    with pytest.raises(dr.FamilyError, match="unknown decision family"):
        dr.facts(str(tmp_path), "d3")
    with pytest.raises(dr.FamilyError, match="base model"):
        dr.facts(os.path.join(FIX, "kev"), "kev")


def test_unsupported_backbone_is_refused():
    with pytest.raises(dr.FamilyError, match="not one the runtime serves"):
        dr.facts_from_configs("laya", {"model_type": "bert", "hidden_size": 8, "intermediate_size": 8,
                                       "num_hidden_layers": 1, "num_attention_heads": 1, "vocab_size": 8})


@pytest.mark.parametrize("files,family", [
    (["rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json", "model.safetensors", "config.json"],
     "laya"),
    (["julia_config.json", "encoder/config.json", "tokenizer/tokenizer.json", "model.safetensors", "config.json"],
     "julia"),
    (["config.json", "encoder_config/config.json", "tokenizer.json", "model.safetensors"], "gliner"),
    (["adapter_config.json", "adapter_model.safetensors", "head.pt", "tokenizer.json"], "kev"),
    (["config.json", "model.safetensors", "tokenizer.json"], None),                       # a plain checkpoint
    (["encoder/config.json", "tokenizer/tokenizer.json", "model.safetensors"], None),     # Laya without its config
])
def test_detect(files, family):
    assert dr.detect(files) == family


# --------------------------------------------------------------------------------------------------------- precision

@pytest.mark.parametrize("device,expected", [
    (RTX3050, dr.MIXED16), (A10, dr.MIXED16), (H100, dr.MIXED16), (GTX1650, dr.FLOAT32),
    ({"type": "cuda", "compute_capability": "7.0", "hardware": {"tensor_cores": "gen1"}}, dr.HALF16),
    ({"type": "cuda", "compute_capability": "8.9"}, dr.MIXED16),
    ({"type": "cuda", "name": "mystery card"}, dr.FLOAT32),
    (XEON, dr.MIXED16), (I7, dr.FLOAT32), (AVX2, dr.FLOAT32),
    ({"type": "cpu", "cpu_features": ["avx512f", "amx_bf16"]}, dr.MIXED16),
    ({"type": "cpu", "cpu_amx": "false"}, dr.FLOAT32),
])
def test_precision_rule(device, expected):
    hardware = plan.resolve_hardware(device)
    precision, reason = dr.choose(device, hardware)
    assert precision == expected and reason


def test_int8_is_never_chosen():
    for device in (RTX3050, GTX1650, XEON, I7, AVX2):
        assert dr.choose(device, plan.resolve_hardware(device))[0] != "int8"


@pytest.mark.parametrize("device", [
    {"type": "rocm", "name": "AMD Instinct MI300X", "mem_per_GPU_in_GB": 192},
    {"type": "gpu", "name": "AMD Instinct MI300X", "mem_per_GPU_in_GB": 192},
    {"type": "hpu", "name": "Intel Gaudi 3", "mem_per_GPU_in_GB": 128},
])
def test_devices_the_runtime_does_not_serve_are_refused(device):
    """The runtime serves CUDA and CPU only: an AMD, Gaudi or Intel accelerator is never sized as CUDA."""
    p = dr.size(_facts("laya"), device, workload=_wl())
    assert p.calls_per_replica == 0 and "CUDA GPUs and CPUs" in p.refused
    assert calibration.peak_flops(device, plan.resolve_hardware(device), "float32", None) is None


def test_a_generic_gpu_is_cuda_when_it_is_nvidias():
    device = {"type": "gpu", "name": "NVIDIA GeForce RTX 3050 Laptop GPU", "mem_per_GPU_in_GB": 4}
    p = dr.size(_facts("laya"), device, workload=_wl())
    assert p.refused is None and p.basis == dr.CALIBRATED


# ------------------------------------------------------------------------------------------------------- calibration

def test_calibration_file_is_reproduced_by_the_tool(tmp_path):
    out = tmp_path / "calibration.json"
    results = sorted(os.path.join(BASELINES, f) for f in os.listdir(BASELINES))
    assert _tool().main([*results, "--runtime", "0.1.0", "--out", str(out)]) == 0
    shipped = json.load(open(calibration._PATH))["entries"]
    assert json.load(open(out))["entries"] == shipped


def test_calibration_seeds():
    keys = {e.key for e in calibration.load()}
    assert {"RTX3050_LAPTOP_GPU/mixed16/laya", "RTX3050_LAPTOP_GPU/mixed16/julia", "RTX3050_LAPTOP_GPU/mixed16/gliner",
            "RTX3050_LAPTOP_GPU/mixed16/kev", "cpu-xeon-6767p/mixed16/laya", "cpu-xeon-6767p/mixed16/julia",
            "GTX1650_MOBILE_GPU/float32/laya"} <= keys
    for e in calibration.load():
        assert 0.0 < e.eff < 1.0, e.key           # a share of the peak
        assert e.t_pass_min_ms > 0


def test_shed_loads_make_no_entry():
    """i7-11370H's typical load: 41 requests served, 9,647 shed with 503. Its throughput is not a measurement."""
    tool = _tool()
    assert "41 of 9,688" in tool.unclean(_bench("i7-11370h-4t-laya-new.json")["throughput"]["typical"])
    assert "14 of 12,682" in tool.unclean(_bench("gtx1650-laya-new.json")["throughput"]["long"])
    assert "3,156 of 3,270" in tool.unclean(_bench("rtx3050-laya-new.json")["throughput"]["long"])
    assert tool.unclean(_bench("rtx3050-laya-new.json")["throughput"]["typical"]) is None
    assert not any(e.source.startswith("i7-11370h") for e in calibration.load())


@pytest.mark.parametrize("engine,expected", [
    ("basal auto device=cuda threads=4", "mixed16"),
    ("basal half16 device=cuda threads=4", "half16"),
    ("basal device=cuda threads=4", "mixed16"),
])
def test_the_tool_reads_the_engines_precision(engine, expected):
    assert _tool()._precision(engine, RTX3050, plan.resolve_hardware(RTX3050)) == expected


def test_the_tool_refuses_an_unknown_precision():
    with pytest.raises(ValueError, match="int8"):
        _tool()._precision("basal int8 device=cuda", RTX3050, plan.resolve_hardware(RTX3050))


def test_calibration_runtime_not_newer_than_the_engine_row():
    # TC-DO-07: the bud-connect row lists Bud-Decision-Runtime 0.1.0
    assert calibration._version_key(dr.runtime_version()) <= calibration._version_key("0.1.0")


@pytest.mark.parametrize("device,family,basis", [
    (RTX3050, "laya", dr.CALIBRATED),
    (RTX3050, "julia", dr.CALIBRATED),
    (A10, "laya", dr.SAME_ARCHITECTURE),      # Ampere, from the 3050
    (H100, "laya", dr.ESTIMATED),             # no Hopper entry
    (XEON, "laya", dr.CALIBRATED),
    (XEON, "gliner", dr.CALIBRATED),          # the lowest encoder entry on that CPU model
    (XEON, "kev", dr.ESTIMATED),              # no decoder measured on it: an encoder's entry is only an estimate
    (SPR, "laya", dr.SAME_ARCHITECTURE),      # another AMX CPU: the 6767P's entry, not calibrated on it
    (I7, "laya", dr.ESTIMATED),               # the i7's shed run made no entry
    (AVX2, "laya", dr.ESTIMATED),
])
def test_lookup_basis(device, family, basis):
    f = _facts(family)
    p = dr.size(f, device, workload=_wl(target_call_latency_ms=5000))
    assert p.basis == basis, p


def test_kev_is_not_calibrated_from_an_encoder():
    entry, _, basis = calibration.lookup(XEON, None, "mixed16", "kev", "decoder")
    assert basis == dr.ESTIMATED and entry.kind == "encoder"


def test_measured_activations_come_only_from_the_familys_own_entry():
    """Julia's per-token bytes say nothing about Laya's: another family's entry keeps the analytic figure."""
    f = _facts("laya")
    base = dict(device_class="RTX3050_LAPTOP_GPU", device_type="cuda", architecture="AMPERE", precision="mixed16",
                kind="encoder", eff=0.5, t_pass_min_ms=1.0, act_bytes_per_token=1.0, context_bytes=None,
                threads=None, runtime="0.1.1", source="x")
    other = calibration.Entry(key="x/mixed16/julia", family="julia", **base)
    own = calibration.Entry(key="x/mixed16/laya", family="laya", **base)
    kw = dict(cpu=False, precision="mixed16", basis=dr.CALIBRATED, state_cache_mb=None, row=80)
    assert plan._memory(f, entry=other, **kw)[1] == pytest.approx(
        f.activation_bytes_per_token(2.0) * plan.ACTIVATION_MARGIN)
    assert plan._memory(f, entry=own, **kw)[1] == 1.0


@pytest.mark.parametrize("device,precision,tflops", [
    (A100, "float32", 19.5),                  # GA100: 1/16 of its 312 dense bfloat16
    (H100, "float32", 67.0),
    (A10, "float32", 125 * 0.25),             # GA102: a quarter
    (GTX1650, "float32", 6.4 * 0.5),          # no tensor cores: its float16 is twice its float32
    (A100, "mixed16", 312.0),
])
def test_gpu_peak(device, precision, tflops):
    peak = calibration.peak_flops(device, plan.resolve_hardware(device), precision, None)
    assert peak == pytest.approx(tflops * 1e12, rel=1e-3)


def test_estimate_is_below_every_measured_entry_of_its_type():
    entries = calibration.load()
    low = min(e.eff for e in entries if e.device_type == "cuda" and e.precision == "mixed16")
    _, eff, basis = calibration.lookup(H100, plan.resolve_hardware(H100), "mixed16", "laya", "encoder")
    assert basis == dr.ESTIMATED and eff == pytest.approx(low * calibration.ESTIMATE_FACTOR)


# ------------------------------------------------------------------------------------------------------------ sizing

def _bench(name):
    return json.load(open(os.path.join(BASELINES, name)))


def _bench_workload(load, family):
    return _tool()._workload(_facts(family), load)


@pytest.mark.parametrize("baseline,family,device,under", [
    ("rtx3050-julia-new.json", "julia", RTX3050, 0.20),
    ("rtx3050-gliner-new.json", "gliner", RTX3050, 0.20),
    ("rtx3050-kev-new.json", "kev", RTX3050, 0.20),
    # a small model on a CPU gains from longer rows (fixed per-pass costs), which the model does not count: the
    # plan may be conservative there, never optimistic
    ("xeon6767p-16t-julia-new.json", "julia", dict(XEON, physical_cores=32), 0.40),
])
def test_predicts_the_unseen_long_load(baseline, family, device, under):
    """Calibrated on the ``typical`` load (3 questions of 60 words), the plan's tokens/s for the ``long`` load (8
    questions of 300 words) is at most 20% over what the benchmark measured there, and at most ``under`` below.

    Only a long load the runtime served is compared: RTX 3050 Laya (3,156 of 3,270 shed), GTX 1650 Laya (14 served
    of 12,682) and Xeon Laya (89 of 3,791) measured load shedding, not throughput."""
    result = _bench(baseline)
    assert _tool().unclean(result["throughput"]["long"]) is None
    long_load = result["latency"]["long"]
    wl = dict(_bench_workload(long_load, family), target_call_latency_ms=10_000)
    p = dr.size(_facts(family), device, workload=wl)
    measured = result["throughput"]["long"]["tokens_per_s"]
    assert p.refused is None, p.refused
    assert (1 - under) * measured <= p.tokens_per_second <= 1.2 * measured


def test_reproduces_the_typical_throughput_exactly():
    result = _bench("rtx3050-laya-new.json")
    wl = dict(_bench_workload(result["latency"]["typical"], "laya"), target_call_latency_ms=10_000)
    p = dr.size(_facts("laya"), RTX3050, workload=wl)
    assert p.tokens_per_second == pytest.approx(result["throughput"]["typical"]["tokens_per_s"], rel=0.01)


def test_closed_loop_latency_at_sixteen_calls():
    # 16 clients on the 3050's typical load measured p50 284 ms: 16 calls of 239 tokens at 13.6k tokens/s
    result = _bench("rtx3050-laya-new.json")
    wl = _bench_workload(result["latency"]["typical"], "laya")
    p = dr.size(_facts("laya"), RTX3050, workload=dict(wl, target_call_latency_ms=1000))
    per_call_ms = p.call_tokens / p.tokens_per_second * 1000
    assert 16 * per_call_ms == pytest.approx(result["throughput"]["typical"]["p50_ms"], rel=0.10)


def test_calls_per_replica_formula():
    p = dr.size(_facts("laya"), RTX3050, workload=_wl(target_call_latency_ms=300))
    w = p.call_tokens
    assert w == 3 * 80
    assert p.calls_per_replica == math.floor(0.8 * p.tokens_per_second * 0.3 / w)
    assert p.call_latency_ms <= 300 and p.call_latency_ms >= p.single_call_ms
    assert p.refused is None and p.calls_per_second == pytest.approx(p.tokens_per_second / w)


def test_tighter_target_holds_fewer_calls():
    f = _facts("laya")
    k = [dr.size(f, RTX3050, workload=_wl(target_call_latency_ms=t)).calls_per_replica for t in (100, 300, 1000)]
    assert k[0] < k[1] < k[2]


def test_refused_when_one_call_misses_the_target():
    p = dr.size(_facts("laya"), GTX1650, workload=_wl(questions_per_call=32, state_tokens=480,
                                                      target_call_latency_ms=100))
    assert p.calls_per_replica == 0
    assert p.refused and "over the 100 ms target" in p.refused and "float32" in p.refused


def test_one_call_that_just_fits_gets_one_replica_slot():
    f = _facts("laya")
    single = dr.size(f, RTX3050, workload=_wl(questions_per_call=16, state_tokens=480, target_call_latency_ms=5000))
    target = single.single_call_ms * 1.05
    p = dr.size(f, RTX3050, workload=_wl(questions_per_call=16, state_tokens=480, target_call_latency_ms=target))
    assert p.calls_per_replica == 1 and p.refused is None


def test_no_target_is_refused():
    p = dr.size(_facts("laya"), RTX3050, workload=_wl(target_call_latency_ms=None))
    assert p.calls_per_replica == 0 and "target_call_latency_ms" in p.refused


def test_laya_rows_are_capped_at_max_len():
    p = dr.size(_facts("laya"), RTX3050, workload=_wl(state_tokens=5000, target_call_latency_ms=5000))
    assert p.row_tokens == 512 and p.call_tokens == 3 * 512


def test_gliner_reads_one_sequence_and_refuses_over_its_limit():
    g = _facts("gliner")
    ok = dr.size(g, RTX3050, workload=_wl(state_tokens=100, question_tokens=20, questions_per_call=3))
    assert ok.call_tokens == 160 and ok.row_tokens == 160
    over = dr.size(g, RTX3050, workload=_wl(state_tokens=4000, question_tokens=50, questions_per_call=4))
    assert over.calls_per_replica == 0 and "4,096" in over.refused


def _kev_wl(**kw):
    return _wl(**{"state_tokens": 1000, "question_tokens": 20, "questions_per_call": 8,
                  "target_call_latency_ms": 2000, **kw})


def test_kev_counts_the_state_once():
    p = dr.size(_facts("kev"), A10, workload=_kev_wl())
    assert p.call_tokens == 1000 + 8 * 20
    assert p.state_bytes == plan.DEFAULT_STATE_CACHE_MB * 2**20 and p.state_cache_mb == plan.DEFAULT_STATE_CACHE_MB


@pytest.mark.parametrize("workload,limit", [
    (dict(state_tokens=70_000, question_tokens=10), "65,536-token state limit"),
    (dict(state_tokens=60_000, question_tokens=9_000), "8,192-token question limit"),
])
def test_kev_limits_each_part(workload, limit):
    """Kev takes a state of at most 65,536 tokens and a question of at most 8,192, not any split of their sum."""
    p = dr.size(_facts("kev"), H100, workload=_kev_wl(**workload))
    assert p.calls_per_replica == 0 and limit in p.refused


def test_kev_fits_the_card_it_was_measured_on():
    """rtx3050-kev-new.json ran Kev on the 4 GB RTX 3050: the plan gives its unnamed state cache way first."""
    p = dr.size(_facts("kev"), RTX3050, workload=_kev_wl())
    assert p.refused is None and p.calls_per_replica > 0
    assert plan.MIN_STATE_CACHE_MB <= p.state_cache_mb < plan.DEFAULT_STATE_CACHE_MB
    assert p.state_bytes == p.state_cache_mb * 2**20 and p.max_pass_tokens == plan.MAX_PASS_TOKENS
    assert p.total_bytes <= plan.GPU_UTIL_MAX * 4 * GIB


def test_a_named_state_cache_is_kept():
    k = _facts("kev")
    assert dr.size(k, RTX3050, workload=_kev_wl(), state_cache_mb=0).state_bytes == 0
    p = dr.size(k, RTX3050, workload=_kev_wl(), state_cache_mb=2048)
    assert p.calls_per_replica == 0 and "state cache of 2,048 MiB" in p.refused


def test_cpu_threads_rule():
    assert dr.cpu_threads(XEON) == 16                                  # 64 physical per NUMA node, capped
    assert dr.cpu_threads(I7) == 4
    assert dr.cpu_threads({"type": "cpu", "cores": 8}) == 4            # logical only: half are physical
    assert dr.cpu_threads({"type": "cpu"}) == 16
    assert dr.cpu_threads(XEON, threads=8) == 8


def test_cpu_plan_scales_with_threads():
    f = _facts("laya")
    p16 = dr.size(f, XEON, workload=_wl(target_call_latency_ms=2000))
    p8 = dr.size(f, XEON, workload=_wl(target_call_latency_ms=2000), threads=8)
    assert p16.threads == 16 and p8.threads == 8
    assert p8.tokens_per_second == pytest.approx(p16.tokens_per_second / 2, rel=1e-3)


def test_memory_terms():
    f = _facts("laya")
    gpu = dr.size(f, RTX3050, workload=_wl())
    # mixed16 holds its weights in bfloat16: an assumption until runtime v0.1.1's benchmark records peak memory. If
    # the runtime autocasts over Laya's stored float32 weights instead, this term is half the real one.
    assert gpu.weight_bytes == f.params_total * 2
    assert gpu.state_bytes == 0 and gpu.activation_bytes > 0 and gpu.overhead_bytes > 0
    assert gpu.total_bytes < 4 * 2**30                                  # Laya serves on the 4 GB 3050 (measured)
    f32 = dr.size(f, GTX1650, workload=_wl())
    assert f32.weight_bytes == f.params_total * 4
    cpu = dr.size(f, XEON, workload=_wl())
    assert cpu.weight_bytes == f.params_total * 2 + f.linear_params * 2   # oneDNN's packed copy


def test_a_replica_over_the_gpu_is_refused():
    p = dr.size(_facts("gliner"), {**RTX3050, "mem_per_GPU_in_GB": 1}, workload=_wl())
    assert p.calls_per_replica == 0 and "GiB" in p.refused


def _group(p, *, engine="bud-decision", mode="dedicated", card_gib=4.0):
    """The node group budsim sends budcluster for a plan's terms (decimal GB on the wire)."""
    return {
        "type": "cuda", "hardware_mode": mode, "engine_type": engine, "tp_size": 1, "concurrency": 4,
        "weight_memory_gb": p.weight_bytes / 1e9, "activation_memory_gb": p.activation_bytes / 1e9,
        "state_memory_gb": p.state_bytes / 1e9, "kv_cache_memory_gb": 0, "device_total_memory_gb": card_gib,
    }


@pytest.mark.parametrize("family", ["laya", "julia", "gliner", "kev"])
@pytest.mark.parametrize("mode", ["dedicated", "shared"])
def test_a_plan_the_planner_accepts_the_deployment_accepts(family, mode):
    """The GPU gate is the deployment's: GLiNER on the 4 GB RTX 3050 used to pass the planner (700 MiB context, 100%
    of the card) and fail budcluster's rule (1.5 GiB context, 94%)."""
    from llm_memory_calculator.runtime.modules import legacy

    wl = _gliner_wl() if family == "gliner" else (_kev_wl() if family == "kev" else _wl())
    p = dr.size(_facts(family), RTX3050, workload=wl)
    assert p.refused is None, p.refused
    budget = legacy.cuda_budget(_group(p, mode=mode), {"input_tokens": 128, "output_tokens": 1})
    assert budget.gpu_memory_utilization <= legacy.GPU_UTIL_MAX and budget.kv_engine_gib == 0


def test_unknown_gpu_is_refused_not_raised():
    p = dr.size(_facts("laya"), {"type": "cuda", "name": "Some Future Card"}, workload=_wl())
    assert p.calls_per_replica == 0 and "hardware table" in p.refused


def test_size_never_raises():
    p = dr.size(_facts("laya"), RTX3050, workload=_wl(), precision="int8")
    assert p.calls_per_replica == 0 and "could not be computed" in p.refused


def test_asked_precision_is_used_and_explained():
    p = dr.size(_facts("laya"), RTX3050, workload=_wl(), precision="float32")
    assert p.precision == "float32" and "as asked" in p.precision_reason


@pytest.mark.parametrize("target,expected", [
    (400, {"interactive_queue_ms": 400.0, "max_auto_pass_ms": 200.0}),
    (2000, {"interactive_queue_ms": 2000.0}),                        # T/2 over the runtime's own 300 ms ceiling
    (None, {}),
])
def test_admission_limits(target, expected):
    assert dr.admission_limits(target) == expected


def test_plan_carries_limits_and_serialises():
    p = dr.size(_facts("laya"), RTX3050, workload=_wl(target_call_latency_ms=400))
    d = p.to_dict()
    assert d["limits"] == {"interactive_queue_ms": 400.0, "max_auto_pass_ms": 200.0}
    assert d["total_bytes"] == p.total_bytes and json.dumps(d)
    assert d["calibration"] == "RTX3050_LAPTOP_GPU/mixed16/laya"


# ------------------------------------------------------------------------------------------ memory-bounded pass ceiling

def _gliner_wl():
    return _wl(questions_per_call=4, state_tokens=256, question_tokens=48, target_call_latency_ms=400)


def test_a_card_with_room_keeps_the_runtimes_default_pass_ceiling():
    p = plan.size(_facts("gliner"), A10, workload=_gliner_wl())
    assert p.max_pass_tokens == plan.MAX_PASS_TOKENS
    assert "max_pass_tokens" not in p.limits


def test_the_card_lowers_the_pass_ceiling_without_a_budget():
    """pde-ditto 11-13: GLiNER2.5-Decide on the 4 GB RTX 3050, its activations sized for the runtime's default
    16,384-token pass, does not fit the card; the plan fits it with a lower ``[limits] max_pass_tokens``, from the
    card alone when the caller gives no budget."""
    p = plan.size(_facts("gliner"), RTX3050, workload=_gliner_wl())
    assert p.refused is None
    assert p.weight_bytes + p.activation_bytes + p.state_bytes <= plan.GPU_UTIL_MAX * 4 * GIB - plan.CUDA_CONTEXT_BYTES
    assert p.max_pass_tokens < plan.MAX_PASS_TOKENS and p.limits["max_pass_tokens"] == p.max_pass_tokens


def test_a_memory_budget_lowers_the_pass_ceiling_until_the_replica_fits():
    """The caller's budget bounds the replica on a card that would hold the default (an A10 shared with others)."""
    budget = 2.26 * 2**30
    default = plan.size(_facts("gliner"), A10, workload=_gliner_wl())
    assert default.weight_bytes + default.activation_bytes + default.state_bytes > budget
    p = plan.size(_facts("gliner"), A10, workload=_gliner_wl(), memory_budget_bytes=budget)
    assert p.refused is None
    assert p.weight_bytes + p.activation_bytes + p.state_bytes <= budget
    assert p.max_pass_tokens < plan.MAX_PASS_TOKENS and p.max_pass_tokens % 64 == 0
    assert p.limits["max_pass_tokens"] == p.max_pass_tokens
    assert p.calls_per_replica == default.calls_per_replica


def test_the_pass_ceiling_never_drops_below_the_longest_row():
    """The runtime raises its ceiling to the longest possible row itself, so a lower one would not hold."""
    facts = _facts("gliner")
    floor = max(facts.max_len, 64)
    p = plan.size(facts, RTX3050, workload=_gliner_wl(), memory_budget_bytes=1.0 * 2**30)
    assert p.refused and "GiB" in p.refused
    roomy = plan.size(facts, RTX3050, workload=_gliner_wl(), memory_budget_bytes=2.26 * 2**30)
    assert roomy.max_pass_tokens >= floor


def test_a_budget_that_already_fits_changes_nothing():
    p = plan.size(_facts("laya"), A10, workload=_gliner_wl(), memory_budget_bytes=64 * 2**30)
    assert p.max_pass_tokens == plan.MAX_PASS_TOKENS and "max_pass_tokens" not in p.limits


@pytest.mark.parametrize("budget", [0, -1.0])
def test_no_memory_left_is_refused(budget):
    """A gate that leaves nothing is a refusal, not "no budget"."""
    p = plan.size(_facts("laya"), A10, workload=_wl(), memory_budget_bytes=budget)
    assert p.calls_per_replica == 0 and "0.0 GiB" in p.refused


def test_a_row_that_fits_is_not_lost_to_rounding():
    """The pass ceiling steps by 64 tokens, but never below a row that fits: a Kev row of 1,000 tokens with room
    for 1,010 is served."""
    k = _facts("kev")
    wl = _kev_wl(state_tokens=990, question_tokens=10, questions_per_call=1)
    probe = plan.size(k, A10, workload=wl, state_cache_mb=64)
    per_token = probe.activation_bytes / probe.max_pass_tokens
    budget = probe.weight_bytes + probe.state_bytes + per_token * 1010
    p = plan.size(k, A10, workload=wl, state_cache_mb=64, memory_budget_bytes=budget)
    assert p.refused is None and p.max_pass_tokens == 1000


def test_hardware_is_matched_through_the_hardware_manager(monkeypatch):
    """A card added through the hardware DB is known to the planner as it is to the rest of budsim."""
    from llm_memory_calculator.hardware import manager

    seen = {}

    class Manager:
        def match_cluster_device(self, info):
            seen.update(info)
            return {"name": "DB_ONLY_GPU", "Flops": 100.0, "tensor_cores": "gen3", "compute_capability": "8.6",
                    "architecture": "AMPERE", "manufacturer": "NVIDIA"}

    monkeypatch.setattr(manager, "default_manager", lambda: Manager())
    hw = plan.resolve_hardware({"type": "cuda", "raw_name": "NVIDIA DB Only", "mem_per_GPU_in_GB": 24})
    assert hw["name"] == "DB_ONLY_GPU" and seen == {"raw_name": "NVIDIA DB Only", "memory_mb": 24 * 1024}


# ---------------------------------------------------------------- the runtime-spec memory rule in shared mode


def _shared_group(engine, gib_demand_ex_ctx, state_gb=0):
    return {
        "type": "cuda",
        "hardware_mode": "shared",
        "engine_type": engine,
        "tp_size": 1,
        "concurrency": 4,
        "weight_memory_gb": 1.57,
        "activation_memory_gb": gib_demand_ex_ctx - 1.57 - state_gb,
        "kv_cache_memory_gb": 0,
        "state_memory_gb": state_gb,
        "device_total_memory_gb": 4.0,
    }


def test_a_shared_decision_replica_takes_its_demand_as_its_slice():
    """Bud-Decision-Runtime takes no fraction of the device, so its HAMi slice is its demand, as for LatentBud.

    pde-ditto 2026-11-13: in shared mode a Laya plan sized to fit a 4 GB GTX 1650 was dropped as "the shared slice is
    larger than the card": the rule gave it vLLM's fraction-safe slice, which adds the 1.5 GiB context reserve to a
    demand that already carries it. budsim (shared_slice.py) and budcluster (memory_plan) already treat it as demand.
    """
    from llm_memory_calculator.runtime.modules import legacy

    intent = {"input_tokens": 128, "output_tokens": 1}
    budget = legacy.cuda_budget(_shared_group("bud-decision", 2.0), intent)
    assert budget.gpu_memory_mib == legacy.ceil_mib(budget.device_demand)
    with pytest.raises(legacy.SizingError, match="shared slice is larger than the card"):
        legacy.cuda_budget({**_shared_group("bud-decision", 2.0), "engine_type": "vllm", "is_embedding": True}, intent)


def test_a_decision_replica_has_no_kv_without_is_embedding():
    """Kev keeps states, not KV: a bud-decision group needs no is_embedding flag to be sized without a KV term."""
    from llm_memory_calculator.runtime.modules import legacy

    budget = legacy.cuda_budget(_shared_group("bud-decision", 2.0, state_gb=0.3), {"input_tokens": 128,
                                                                                    "output_tokens": 1})
    assert budget.kv_engine_gib == 0


def test_the_infra_slice_rule_agrees():
    """kv/infra (budsim's placement filter) takes the same demand-only slice as the deployment rule."""
    from llm_memory_calculator.kv import infra
    from llm_memory_calculator.runtime.modules import legacy

    assert legacy.SLICE_DEMAND_ONLY_ENGINES is infra.SLICE_DEMAND_ONLY_ENGINES
    demand = 3 * GIB
    assert infra.slice_mib(demand, "bud-decision") == infra.slice_mib(demand, "latentbud") == (3072, None)
    terms = {"weight_memory": 1.57 * GIB, "activation_memory": 0.4 * GIB, "kv_cache_memory": 0}
    assert infra.wire_slice_mib(terms, "bud-decision") == infra.wire_slice_mib(terms, "latentbud")
    assert infra.placeable_configs([terms], "bud-decision", free_gib=4.0) == [terms]
    assert infra.grown_slice_mib(demand, 0, 2.0, "bud-decision")[2] is False
