"""FRD-025 runtime spec + FRD-027 Part B: composition, settings resolution, legacy equivalence and the modules.

C-GOLD-01 (v1 modules render budcluster's golden pods), C-GOLD-02 (v2 changes exactly the intended keys),
C-GOLD-03 (rung 5 is stock), T-LOOP-03/05/06, T-CAP-01..05, T-GUARD-01, T-FALL-01..02, T-PREC-01..04,
T-HYB-01..03, T-SPEC-01..06, T-REC-01..03, T-VAL-01..04.
"""

import json
from collections import Counter

import pytest

from llm_memory_calculator.device import HardwareMode, catalogue
from llm_memory_calculator.engine import EngineRecord
from llm_memory_calculator.model import ModelFacts
from llm_memory_calculator.runtime import plan
from llm_memory_calculator.runtime.compose import CompositionError, order, resolve_settings
from llm_memory_calculator.runtime.context import (
    Fragment,
    Intent,
    Placement,
    PrecisionCandidate,
    SpeculativeIntent,
    Value,
)
from llm_memory_calculator.runtime.modules import PLANNER, PLANNER_CAPACITY, LEGACY, PlanError, build
from llm_memory_calculator.runtime.modules.base import Module
from llm_memory_calculator.runtime.records import record as vendored_record, record_from_budconnect
from llm_memory_calculator.runtime.spec import RuntimeSpec, flatten_args
from llm_memory_calculator.runtime.validation import (
    ValidationRecord,
    ValidationStore,
    default_validation,
    judge,
)

from .conftest import FIXTURES, load_model, load_record

GIB = 1024**3
VGPU = catalogue.get("h100xm-80c-vgpu")
UP = vendored_record("vllm@0.30.0")
FORK = vendored_record("bud-vllm@0.9.0-481e481be")


def golden(case):
    return json.loads((FIXTURES / "goldens" / f"{case}.json").read_text())


def legacy_view(spec: RuntimeSpec, *, namespace: str, model_path: str):
    eng = spec.engine_container()
    args = flatten_args(eng.args, style=spec.engine.arg_style)
    args = [
        a.replace("$(BUD_MODEL_PATH)", model_path).replace("$(BUD_SERVED_NAME)", namespace)
        for a in args
    ]
    res = eng.resources
    shm = next((v for v in spec.pod.volumes if v.name == "shm"), None)
    return {
        "args": Counter(args),
        "envs": dict(eng.env),
        "gpu_memory_mb": res.limits.get("nvidia.com/gpumem"),
        "memory_mib": res.limits.get("memory_mib"),
        "shm_mib": shm.size_mib if shm else None,
        "node_selector": dict(spec.pod.node_selector),
        "tp_size": res.limits.get("nvidia.com/gpu"),
        "image": spec.engine.image,
        "replicas": spec.replicas,
    }


def golden_spec(case, versions=LEGACY, **group_over):
    g = golden(case)
    group = {**g["group"], **group_over}
    record = record_from_budconnect(
        {
            **group,
            "engine": group["engine_type"],
            "engine_version": group["engine_version"],
            "engine_image": group["image"],
            "kv_features": group.get("kv_features") or [],
        }
    )
    device = catalogue.lookup(group["raw_name"])
    model_name = "qwen3-8b" if "llama" not in group["args"]["model"] else "llama-3.1-8b"
    model = ModelFacts.from_config(load_model(model_name))
    conc = int(group["labels"]["concurrency"])
    intent = Intent(
        model_path=group["args"]["model"],
        input_tokens=g["context"]["input_tokens"],
        output_tokens=g["context"]["output_tokens"],
        model_max_context_length=g["context"]["model_max_context_length"],
        concurrency=conc,
        hardware_mode=group["hardware_mode"],
    )
    placement = Placement(
        device_type="cuda",
        tp=int(group["tp_size"]),
        pp=int(group["pp_size"]),
        replicas=int(group["replicas"]),
        concurrency=conc,
        hardware_mode=group["hardware_mode"],
        raw_name=group["raw_name"],
        group=group,
    )
    ctx = plan.context(
        intent,
        model,
        load_model(model_name),
        record,
        device.with_mode(
            HardwareMode("shared", 9274 * 1024**2)
            if group["hardware_mode"] == "shared"
            else HardwareMode()
        ),
        placement=placement,
        versions=versions,
    )
    return plan.spec(ctx), g


# ====================================================================== C-GOLD-01: v1 == legacy


@pytest.mark.parametrize("case", ["cuda_dedicated_tp1", "cuda_dedicated_tp2", "cuda_shared_tp1"])
def test_v1_spec_renders_the_legacy_golden(case):
    spec, g = golden_spec(case)
    view = legacy_view(
        spec, namespace=g["context"]["namespace"], model_path=g["group"]["args"]["model"]
    )
    exp = g["expected"]
    assert view["args"] == Counter(exp["args"])  # order-insensitive: vLLM's argparse is
    assert view["envs"] == exp["envs"]
    assert view["gpu_memory_mb"] == exp.get("gpu_memory_mb")
    assert view["node_selector"] == exp["node_selector"]
    assert view["tp_size"] == exp["tp_size"]
    assert view["image"] == exp["image"] and view["replicas"] == exp["replicas"]
    assert (
        view["shm_mib"] == 2048 and view["memory_mib"] is None
    )  # no T1: chart default shm, no host request
    assert spec.engine.trailing_args == ["--uvicorn-log-level", "warning"]
    assert spec.pod.scheduler == "hami-scheduler" and spec.pod.runtime_class == "nvidia"
    names = [c.name for c in spec.containers]
    assert names == ["budruntime-container", "aibrix-runtime", "bud-runtime-sidecar"]


def test_v1_with_kv_features_adds_exactly_the_frd023_render():
    base, g = golden_spec("cuda_dedicated_tp1")
    events, _ = golden_spec("cuda_dedicated_tp1", kv_features=["kv_events", "cache_salt"])
    b = legacy_view(base, namespace="ns", model_path="/m")
    e = legacy_view(events, namespace="ns", model_path="/m")
    added = e["args"] - b["args"]
    removed = b["args"] - e["args"]
    assert "--enable-prompt-tokens-details" in added
    assert any(a.startswith("--kv-events-config=") for a in added)
    assert (
        removed == Counter({"--gpu-memory-utilization=0.1": 1})
        and "--gpu-memory-utilization=0.92" in added
    )  # T0
    assert {p.name for p in events.engine_container().ports} == {"serve", "kv-events", "kv-replay"}
    assert events.pod.labels.get("model.aibrix.ai/kv-events-enabled") == "true"


def test_shared_golden_declares_the_hami_check():
    spec, _ = golden_spec("cuda_shared_tp1")
    kinds = {c.kind for c in spec.checks}
    assert "hami_slice" in kinds and "device_count" in kinds
    assert "resources.gpumem" in spec.requires and "check.hami_slice" in spec.requires


# ====================================================================== settings resolution (T-REC-01/02)


def test_settings_resolve_to_the_legacy_flags():
    vals = {
        "batch.max_running": Value(2, "m"),
        "context.max_tokens": Value(4400, "m"),
        "memory.device_fraction": Value(0.92, "m"),
    }
    args, env = resolve_settings(vals, UP)
    assert flatten_args({k: v.value for k, v in args.items()}) == [
        "--max-num-seqs=2",
        "--max-model-len=4400",
        "--gpu-memory-utilization=0.92",
    ]
    assert env == {}


def test_spec_settings_resolve_to_one_json_token():
    args, _ = resolve_settings({"spec.method": Value("mtp", "s"), "spec.tokens": Value(3, "s")}, UP)
    assert flatten_args({k: v.value for k, v in args.items()}) == [
        '--speculative-config={"method":"mtp","num_speculative_tokens":3}'
    ]
    args, _ = resolve_settings(
        {
            "spec.method": Value("eagle3", "s"),
            "spec.tokens": Value(3, "s"),
            "spec.draft": Value("/drafts/x", "s"),
        },
        UP,
    )
    assert args["speculative-config"].value == {
        "method": "eagle3",
        "model": "/drafts/x",
        "num_speculative_tokens": 3,
    }


def test_inverted_and_implied_settings():
    sgl = EngineRecord.from_dict(load_record("sglang-stub"))
    args, _ = resolve_settings({"prefix_cache.enabled": Value(False, "k")}, sgl)
    assert args["disable-radix-cache"].value is True
    args, _ = resolve_settings({"lora.max_adapters": Value(2, "l")}, UP)
    assert args["max-loras"].value == 2 and args["enable-lora"].value is True
    args, _ = resolve_settings({"prefix_cache.enabled": Value(False, "s")}, UP)
    assert list(args) == ["no-enable-prefix-caching"]
    with pytest.raises(CompositionError, match="unsupported"):
        resolve_settings({"spec.method": Value("eagle3", "s")}, sgl)


def test_online_weights_resolve_only_when_online():
    args, _ = resolve_settings({"weights.format": Value("fp8", "w")}, UP)
    assert args == {}
    args, _ = resolve_settings(
        {"weights.format": Value("fp8", "w"), "weights.online": Value(True, "w")}, UP
    )
    assert args["quantization"].value == "fp8"


def test_record_and_settings_schemas():
    from llm_memory_calculator.runtime.schema.generate import settings_schema, spec_schema

    s = settings_schema()
    assert "batch.max_running" in s["properties"]["settings"]["propertyNames"]["enum"]
    assert spec_schema()["title"].startswith("Bud runtime spec v1")
    bad = load_record("vllm-0.30-upstream")
    bad["settings"]["not.a.setting"] = {"arg": "x"}
    with pytest.raises(ValueError, match="not.a.setting"):
        EngineRecord.from_dict(bad)


def test_committed_schemas_are_current():
    from pathlib import Path

    from llm_memory_calculator.runtime.schema.generate import settings_schema, spec_schema
    import llm_memory_calculator.runtime as rt

    root = Path(rt.__file__).parent
    assert json.loads((root / "schema" / "v1.json").read_text()) == json.loads(
        json.dumps(spec_schema())
    )
    assert json.loads((root / "settings" / "v1.json").read_text()) == json.loads(
        json.dumps(settings_schema())
    )


def test_budconnect_row_merges_with_the_vendored_record():
    row = {
        "engine": "vllm",
        "engine_version": "0.9.0",
        "engine_image": "budstudio/vllm-cuda:0.9.0",
        "kv_features": ["kv_events"],
        "supports_lora": False,
    }
    rec = record_from_budconnect(row)
    assert rec.id == "bud-vllm@0.9.0-481e481be"
    assert (
        "kv_events" in rec.features and "kv_fp8" not in rec.features
    )  # KV features: BudConnect's only
    assert (
        "lora" not in rec.features and "spec_mtp" not in rec.features
    )  # the fork has no verified MTP
    new = record_from_budconnect({**row, "engine_version": "0.9.7", "upstream_vllm": "0.30.0"})
    assert new.id == "vllm@0.9.7" and not (
        {"spec_mtp", "spec_eagle3", "kernel_warmup"} & new.features
    )
    assert record_from_budconnect({"engine": "latentbud", "engine_version": "0.1.0"}) is None


# ====================================================================== composition rules (T-LOOP-03/05)


class _A(Module):
    name = "test.a"
    stage = "capacity"

    def plan(self, ctx):
        return Fragment().set("batch.max_running", 7, self.name)


class _B(Module):
    name = "test.b"
    stage = "capacity"
    after = ("test.a",)

    def plan(self, ctx):
        return Fragment().set("batch.max_running", 9, self.name)


class _C(_B):
    name = "test.c"
    overrides = ("test.a",)

    def plan(self, ctx):
        f = Fragment(overrides=self.overrides)
        return f.set("batch.max_running", 9, self.name)


def _ctx(model="qwen3-8b", record=UP, device=VGPU, **intent):
    m = ModelFacts.from_config(
        load_model(model),
        family={
            "qwen3-8b": "Qwen3-8B",
            "qwen3.8-27b": "Qwen3.8-27B",
            "gpt-oss-20b": "gpt-oss-20b",
            "qwen3.8-27b-fp8": "Qwen3.8-27B",
            "qwen3-8b-fp8": "Qwen3-8B",
        }.get(model),
    )
    i = Intent(**{"concurrency": 16, "input_tokens": 1024, "output_tokens": 256, **intent})
    return plan.context(i, m, load_model(model), record, device)


def test_undeclared_override_fails_composition():
    from llm_memory_calculator.runtime.compose import run_stage

    with pytest.raises(CompositionError, match="without declaring overrides"):
        run_stage("capacity", _ctx(), [_A(), _B()])
    comp = run_stage("capacity", _ctx(), [_A(), _C()])
    assert comp.settings["batch.max_running"].value == 9


def test_order_is_topological_then_by_name_and_cycles_fail():
    assert [m.name for m in order([_B(), _A()])] == ["test.a", "test.b"]

    class _X(Module):
        name, after = "x", ("y",)

    class _Y(Module):
        name, after = "y", ("x",)

    with pytest.raises(CompositionError, match="cycle"):
        order([_X(), _Y()])


def test_planner_module_order():
    mods = order([m for m in build(PLANNER) if m.stage == "numerics"])
    assert [m.name for m in mods] == ["weights.precision", "spec_decode", "state.dtype", "kv.dtype"]
    cap = [m.name for m in order([m for m in build(PLANNER) if m.stage == "capacity"])]
    for a, b in (
        ("memory.gpu", "context.batching"),
        ("context.batching", "lora"),
        ("lora", "engine.guard"),
        ("engine.guard", "engine.fallback"),
        ("memory.gpu", "memory.host"),
    ):
        assert cap.index(a) < cap.index(b), (a, b)


def _placement(device, *, conc=16, tp=1, mode="dedicated", replicas=1, group=None):
    g = {
        "weight_memory_gb": 16.4,
        "kv_cache_memory_gb": 20.0,
        "activation_memory_gb": 1.0,
        "state_memory_gb": 0.0,
        "sampler_logits_gb": 0.1,
        "device_total_memory_gb": device.memory_visible_bytes / GIB,
        "type": "cuda",
        "engine_type": "vllm",
        "tp_size": tp,
        "pp_size": 1,
        "raw_name": "NVIDIA-H100XM-80C",
        "name": "NVIDIA-H100XM-80C",
        "supports_lora": True,
        "max_loras": 1,
        "max_lora_rank": 64,
        "lora_prefill_scratch_gb": 0.0,
        "lora_scratch_zero_by_device": True,
        "hardware_mode": mode,
        "image": "vllm/vllm-openai:v0.30.0",
        "args": {
            "model": "/data/models-registry/m",
            "tensor-parallel-size": tp,
            "pipeline-parallel-size": 1,
        },
    }
    g.update(group or {})
    return Placement(
        device_type="cuda",
        tp=tp,
        replicas=replicas,
        concurrency=conc,
        hardware_mode=mode,
        raw_name=g["raw_name"],
        group=g,
    )


def _spec(
    model="qwen3-8b",
    record=UP,
    device=VGPU,
    versions=PLANNER,
    conc=16,
    mode="dedicated",
    group=None,
    **intent,
):
    ctx = _ctx(model, record, device, concurrency=conc, **intent)
    ctx.module_versions = dict(versions)
    dev = (
        device if mode == "dedicated" else device.with_mode(HardwareMode("shared", 66956 * 1024**2))
    )
    ctx.device = dev
    ctx.placement = _placement(dev, conc=conc, mode=mode, group=group)
    return plan.spec(ctx), ctx


def test_spec_is_deterministic_and_pins_versions():
    a, _ = _spec()
    b, _ = _spec()
    assert a.canonical_json() == b.canonical_json()
    assert a.predictor.version.startswith("lmc-") and a.predictor.grounding
    assert a.inputs_digest == b.inputs_digest


def test_each_module_runs_once():
    spec, ctx = _spec()
    from llm_memory_calculator.runtime.compose import run_stage  # noqa: F401

    # the composition log of the last plan: one row per module
    st1 = plan.numerics(_ctx())
    names = [r.module for r in st1.composition.log]
    assert len(names) == len(set(names))


def test_evaluation_budget():
    ctx = _ctx("qwen3.8-27b", concurrency=16, speculative=SpeculativeIntent(enabled=True))
    ctx.placement = _placement(VGPU)
    spec = plan.spec(ctx)
    assert ctx.predictor.evaluations <= 400
    assert spec.settings["spec.method"] == "mtp"


# ====================================================================== T-CAP (v2)


def test_cap_max_running():
    spec, _ = _spec(conc=16)
    assert spec.settings["batch.max_running"] == 256
    spec, _ = _spec(conc=400)
    assert spec.settings["batch.max_running"] == 500
    # bf16 weights and fp32 state: the 392-block ceiling binds
    spec, _ = _spec(
        "qwen3.8-27b",
        conc=400,
        versions={**PLANNER, "state.dtype": "off"},
        weight_precision="original",
    )
    assert (
        abs(spec.settings["batch.max_running"] - 392) <= 8
    )  # within the memory band (2% of the pool)
    assert any("sequence ceiling" in d for d in spec.decisions)
    # with stage 1's bf16 state the sequence ceiling roughly doubles and no longer binds; the CUDA-graph profiling
    # transient does (500 sequences would OOM at startup), and the spec boots
    spec, ctx = _spec("qwen3.8-27b", conc=400)
    assert 400 <= spec.settings["batch.max_running"] < 500
    assert any("profiling transient" in d for d in spec.decisions)
    assert ctx.predictor.limits(spec.settings, ctx.model, ctx.device).boots


def test_cap_max_running_by_the_profiling_transient():
    """gtech 10-20: on a 45 GB L20 the FP8 27B boots only below the CUDA-graph profiling transient's ceiling, so
    context.batching caps max_num_seqs there, says why, and the spec it composes boots (the search places replicas
    for the rest)."""
    from llm_memory_calculator.device import facts_from_node

    l20 = facts_from_node({"product": "NVIDIA-L20", "type": "cuda", "mem_per_gpu_in_gb": 44.99})
    spec, ctx = _spec("qwen3.8-27b-fp8", record=FORK, device=l20, conc=16)
    seqs = spec.settings["batch.max_running"]
    probe = ctx.predictor.limits({**spec.settings, "batch.max_running": 256}, ctx.model, l20)
    assert probe.transient_ceiling and seqs <= probe.transient_ceiling < 256
    assert seqs >= 20  # 16 x 1.25: the planned load still runs at once
    assert any("profiling transient" in d for d in spec.decisions)
    assert ctx.predictor.limits(spec.settings, ctx.model, l20).boots


def test_cap_context():
    spec, _ = _spec("qwen3.8-27b", conc=16)
    assert spec.settings["context.max_tokens"] == 262144
    spec, _ = _spec("qwen3.8-27b", conc=16, max_context_length=32768)
    assert spec.settings["context.max_tokens"] == 32768
    spec, _ = _spec(
        "qwen3.8-27b", conc=16, speculative=SpeculativeIntent(enabled=True, method="mtp")
    )
    assert spec.settings["context.max_tokens"] < 262144
    why = spec.provenance["settings.context.max_tokens"]
    assert "speculative" in why or "MTP" in why


def test_cap_context_refuses_a_request_that_cannot_fit():
    with pytest.raises(PlanError, match="below one request"):
        _spec("qwen3-8b", conc=16, max_context_length=1024, input_tokens=2048, output_tokens=256)


def test_cap_chunk_default_and_long_inputs():
    spec, _ = _spec()
    assert spec.settings["batch.prefill_chunk_tokens"] == 8192
    spec, _ = _spec(input_tokens=32768, output_tokens=512)
    assert spec.settings["batch.prefill_chunk_tokens"] in (8192, 16384)
    assert any("prefill chunk" in d for d in spec.decisions)


def test_cap_memory_fraction():
    spec, _ = _spec()
    assert spec.settings["memory.device_fraction"] == UP.default_fraction(VGPU)
    shared, _ = _spec(mode="shared")
    assert "nvidia.com/gpumem" in shared.engine_container().resources.limits
    assert shared.predicted is not None and shared.predicted.penalty_vs_dedicated is not None
    assert "contention: unmodelled" in shared.predicted.notes


def test_cap_lora():
    spec, _ = _spec()
    args = spec.engine_container().args
    assert "enable-lora" not in args and "max-loras" not in args
    spec, _ = _spec(max_loras=2, lora_rank=32)
    args = spec.engine_container().args
    assert args["max-loras"] == 2 and args["max-lora-rank"] == 32 and args["enable-lora"] is True
    spec, _ = _spec(adapters=[{"name": "a"}, {"name": "b"}, {"name": "c"}])
    assert spec.engine_container().args["max-loras"] == 3


# ====================================================================== C-GOLD-02: the intended diff


@pytest.mark.parametrize("model, conc, fraction", [("qwen3.8-27b", 2, 0.94), ("qwen3-8b", 1, 0.24)])
def test_v2_changes_only_the_intended_keys(model, conc, fraction):
    legacy, _ = _spec(model, versions=LEGACY, conc=conc)
    tuned, _ = _spec(model, versions=PLANNER_CAPACITY, conc=conc)
    a, b = legacy.engine_container().args, tuned.engine_container().args
    changed = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert changed <= {
        "max-num-seqs",
        "gpu-memory-utilization",
        "max-model-len",
        "max-num-batched-tokens",
        "enable-lora",
        "max-loras",
        "max-lora-rank",
    }, changed
    assert b["max-num-seqs"] == 256 and "enable-lora" not in b
    env_a, env_b = legacy.engine_container().env, tuned.engine_container().env
    assert {k for k in set(env_a) | set(env_b) if env_a.get(k) != env_b.get(k)} <= {
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING"
    }


# ====================================================================== guard and fallback


class _Eager(Module):
    name = "test.eager"
    stage = "capacity"

    def plan(self, ctx):
        return Fragment().arg("enforce-eager", True, self.name)


def test_guard_rejects_never_flags():
    from llm_memory_calculator.runtime.compose import run_stage
    from llm_memory_calculator.runtime.modules.guard import EngineGuard

    ctx = _ctx()
    comp = run_stage("capacity", ctx, [_Eager()])
    with pytest.raises(CompositionError, match="test.eager set --enforce-eager"):
        EngineGuard().post_compose(ctx, comp)


def test_guard_text_only_and_defaults_recorded():
    spec, _ = _spec("qwen3.8-27b", text_only=True)
    assert spec.engine_container().args.get("language-model-only") is True
    assert any("engine defaults relied on" in d for d in spec.decisions)
    spec, _ = _spec("qwen3-8b", text_only=True)
    assert "language-model-only" not in spec.engine_container().args


def _rungs(spec):
    return [a.rung for a in spec.alternatives if a.when.get("check") == "boot_failed"]


def test_fallback_ladder_composition():
    spec, _ = _spec("qwen3.8-27b", speculative=SpeculativeIntent(enabled=True, method="mtp"))
    assert _rungs(spec)[0] == 2 and 5 in _rungs(spec)
    capacity_only, _ = _spec("qwen3-8b", versions=PLANNER_CAPACITY)
    assert _rungs(capacity_only) == [4, 5]
    assert (
        any(c.kind == "boot_failed" for c in spec.checks) and "check.boot_failed" in spec.requires
    )


def test_rung5_is_stock():
    s27, _ = _spec("qwen3.8-27b")
    stock = [a for a in s27.alternatives if a.stock_equivalent][0]
    assert stock.settings.get("batch.max_running") == 256  # the 27B's stock default would not boot
    assert stock.settings["memory.device_fraction"] == UP.default_fraction(VGPU)
    s8, _ = _spec("qwen3-8b")
    stock8 = [a for a in s8.alternatives if a.stock_equivalent][0]
    assert "batch.max_running" not in stock8.settings  # stock boots: the engine's own default
    patch = stock8.patch["containers.engine.args"]
    assert patch["max-num-seqs"] is None and patch["max-model-len"] is None


# ====================================================================== stage 1: precision, dtypes, spec


def test_precision_auto_picks_validated_fp8_on_hopper():
    st = plan.numerics(_ctx("qwen3.8-27b"))
    assert st.settings.get("weights.format") == "fp8" and st.settings.get("weights.online") is True
    assert any("chosen" in d for d in st.decisions)


def test_precision_registry_variant_and_path():
    fp8 = load_model("qwen3.8-27b-fp8")
    cand = PrecisionCandidate(
        id="m-fp8", format="fp8", path="/data/models-registry/qwen3.8-27b-fp8", config=fp8
    )
    st = plan.numerics(_ctx("qwen3.8-27b", precision_candidates=[cand]))
    assert st.settings.get("weights.format") == "fp8"
    assert (
        st.model.weight_bytes()
        < 0.6 * ModelFacts.from_config(load_model("qwen3.8-27b")).weight_bytes()
    )


def test_precision_not_on_ampere_without_a_path():
    st = plan.numerics(_ctx("qwen3.8-27b", device=catalogue.get("a100-80gb")))
    assert "weights.format" not in st.settings
    assert any("cannot run fp8" in d for d in st.decisions)


def test_heterogeneous_node_groups_get_one_spec_each():
    """T-LOOP-04: one intent over {h100xm-80c-vgpu, a100-80gb}. The H100 group's spec carries the FP8 variant, the A100
    group's keeps the original weights with the reason, and each group gets its own spec (FR-DEV-4, FR-EXT-2).
    """
    fp8 = load_model("qwen3.8-27b-fp8")
    cand = PrecisionCandidate(
        id="m-fp8", format="fp8", path="/data/models-registry/qwen3.8-27b-fp8", config=fp8
    )
    specs = {}
    for device in (VGPU, catalogue.get("a100-80gb")):
        spec, _ = _spec("qwen3.8-27b", device=device, precision_candidates=[cand])
        specs[device.sku] = spec
    h100, a100 = specs["h100xm-80c-vgpu"], specs["a100-80gb"]
    assert h100.settings.get("weights.format") == "fp8"
    assert "weights.format" not in a100.settings
    assert any("cannot run fp8" in d for d in a100.decisions)
    assert h100.engine_container().args["model"] != a100.engine_container().args["model"]
    assert h100.inputs_digest != a100.inputs_digest
    # an unmeasured SKU is still predicted -- from priors, reported ungrounded -- never left without a prediction
    assert a100.predicted is not None and a100.predicted.ungrounded
    assert not any("not predicted" in d for d in a100.decisions)


@pytest.mark.parametrize(
    "sku",
    [
        s
        for s in catalogue.skus()
        if catalogue.get(s).memory_visible_bytes >= 40 * 1024**3
        and catalogue.get(s).device_class == "cuda"
    ],
)
@pytest.mark.parametrize("model", ["qwen3-8b", "qwen3.8-27b", "gpt-oss-20b"])
def test_every_catalogue_sku_is_planned_with_a_prediction(sku, model):
    """T-DEV-07, FR-DEV-3 / FR-GND-1: a card the model fits on gets a spec with a prediction -- from priors and reported
    ungrounded where nothing is measured on its hardware ladder -- never a spec without one."""
    spec, _ = _spec(model, device=catalogue.get(sku))
    assert spec.predicted is not None, sku
    assert not any("not predicted" in d for d in spec.decisions)


def test_a_card_too_small_is_refused_with_what_it_needs():
    """The 4 GiB laptop card cannot hold the 8B: the plan refuses and says how much it needs and what to change."""
    with pytest.raises(PlanError) as err:
        _spec("qwen3-8b", device=catalogue.get("rtx3050-laptop-4gb"))
    text = str(err.value)
    assert (
        "does not fit on this device" in text
        and "GiB needed of" in text
        and "tensor parallelism" in text
    )


def test_precision_gpt_oss_keeps_mxfp4_and_refuses_online_fp8():
    st = plan.numerics(_ctx("gpt-oss-20b"))
    assert (
        "weights.format" not in st.settings
    )  # native MXFP4 (online fp8 needs a bf16 checkpoint anyway)
    bf16 = PrecisionCandidate(
        id="oss-bf16", format="bf16", path="/m/oss-bf16", config=load_model("gpt-oss-20b-bf16")
    )
    st = plan.numerics(_ctx("gpt-oss-20b", precision_candidates=[bf16]))
    assert "weights.format" not in st.settings  # BF16 loses (bands)


def test_precision_explicit_and_finetune():
    st = plan.numerics(_ctx("qwen3.8-27b", weight_precision="original"))
    assert "weights.format" not in st.settings
    st = plan.numerics(_ctx("qwen3-8b", base_model_relation="finetune"))
    assert "weights.format" not in st.settings
    assert any("needs its own record" in d for d in st.decisions)


def test_state_dtype():
    assert plan.numerics(_ctx("qwen3.8-27b")).settings.get("state.dtype") == "bfloat16"
    assert "state.dtype" not in plan.numerics(_ctx("qwen3-8b")).settings
    fork = plan.numerics(
        _ctx(
            "qwen3.8-27b",
            record=EngineRecord.from_dict(
                {**load_record("bud-vllm-0.9.0-481e481be"), "features": ["kv_fp8"]}
            ),
        )
    )
    assert "state.dtype" not in fork.settings


def test_kv_dtype_rules():
    assert plan.numerics(_ctx("qwen3-8b", enable_kv_fp8=True)).settings.get("kv.dtype") == "fp8"
    # a hybrid needs the 128K needle record on the build: the fork has none
    hybrid = plan.numerics(_ctx("qwen3.8-27b", record=FORK, enable_kv_fp8=True))
    assert "kv.dtype" not in hybrid.settings and any("needle" in d for d in hybrid.decisions)
    assert (
        "kv.dtype" not in plan.numerics(_ctx("qwen3-8b")).settings
    )  # not asked, no calibrated scales
    assert "kv.dtype" not in plan.numerics(_ctx("qwen3-8b", enable_kv_fp8=False)).settings


def test_tcs_needle_run_makes_fp8_kv_eligible_for_the_27b_on_upstream():
    """FRD-027 §10.7 on TCS (10-18): the 27B on vLLM 0.30 with fp8 KV kept GSM8K (95.00 / 97.75 vs bf16 94.54 / 96.75
    on the same engine) and recalled 55/55 needles up to 126K tokens. That record supersedes the 10-16 fail, so an
    explicit fp8 KV request renders; not asked, it stays off (the scales are uncalibrated)."""
    store = default_validation()
    verdict = store.gate("Qwen3.8-27B", None, "base", "vllm@0.30.0", "kv.dtype=fp8")
    assert verdict.ok and verdict.record.evidence["niah_128k_recall"] >= 0.9
    assert plan.numerics(_ctx("qwen3.8-27b", enable_kv_fp8=True)).settings.get("kv.dtype") == "fp8"
    assert "kv.dtype" not in plan.numerics(_ctx("qwen3.8-27b")).settings
    assert not store.gate("Qwen3.8-27B", None, "base", FORK.id, "kv.dtype=fp8").ok


def test_spec_eligibility_and_records():
    on = SpeculativeIntent(enabled=True)
    assert plan.numerics(_ctx("qwen3.8-27b", speculative=on)).settings["spec.method"] == "mtp"
    fork = plan.numerics(_ctx("qwen3.8-27b", record=FORK, speculative=on))
    assert "spec.method" not in fork.settings
    assert (
        "spec.method" not in plan.numerics(_ctx("qwen3-8b", speculative=on)).settings
    )  # no head, no draft
    eagle = SpeculativeIntent(
        enabled=True, method="eagle3", draft_path="/drafts/eagle3-RedHatAI_Qwen3-8B"
    )
    st = plan.numerics(_ctx("qwen3-8b", speculative=eagle))
    assert st.settings["spec.method"] == "eagle3" and st.settings["spec.draft"].startswith(
        "/drafts/"
    )
    assert "spec.method" not in plan.numerics(_ctx("qwen3-8b")).settings  # user-enabled only


def test_spec_prefix_caching_off_unless_safe():
    st = plan.numerics(_ctx("qwen3.8-27b", speculative=SpeculativeIntent(enabled=True)))
    assert st.settings["prefix_cache.enabled"] is False
    rec = EngineRecord.from_dict(
        {
            **load_record("vllm-0.30-upstream"),
            "features": load_record("vllm-0.30-upstream")["features"] + ["spec_prefix_cache_safe"],
        }
    )
    st = plan.numerics(_ctx("qwen3.8-27b", record=rec, speculative=SpeculativeIntent(enabled=True)))
    assert "prefix_cache.enabled" not in st.settings



def test_spec_keeps_prefix_caching_for_a_full_attention_model_on_a_scanned_build():
    """T-SPEC-09: bud-vllm 0.10.0 and vLLM 0.30 were scanned (EAGLE3 on the dense 8B, 1,280 greedy requests each, no
    empty or "!!!!" output), so a full-attention model keeps prefix caching under speculative decoding there. The
    hybrid 27B with MTP and gpt-oss (sliding-window layers) still turn it off (#53912 is open), and so does the fork."""
    eagle = SpeculativeIntent(enabled=True, method="eagle3", draft_path="/drafts/eagle3-RedHatAI_Qwen3-8B")
    oss_eagle = SpeculativeIntent(enabled=True, method="eagle3", draft_path="/drafts/eagle3-RedHatAI_gpt-oss-20b")
    for rec in (UP, vendored_record("bud-vllm@0.10.0")):
        assert "spec_prefix_cache_safe_full_attention" in rec.features
        assert "spec_prefix_cache_safe" not in rec.features
        st = plan.numerics(_ctx("qwen3-8b", record=rec, speculative=eagle))
        assert st.settings["spec.method"] == "eagle3" and "prefix_cache.enabled" not in st.settings
        assert any("prefix caching stays on" in d for d in st.decisions)
        st = plan.numerics(_ctx("qwen3.8-27b", record=rec, speculative=SpeculativeIntent(enabled=True)))
        assert st.settings["prefix_cache.enabled"] is False
        st = plan.numerics(_ctx("gpt-oss-20b", record=rec, speculative=oss_eagle))
        if rec is UP:
            assert st.settings["spec.method"] == "eagle3" and st.settings["prefix_cache.enabled"] is False
        else:  # no GSM8K record for gpt-oss EAGLE3 on 0.10.0: not offered at all
            assert "spec.method" not in st.settings and any("no validation record" in d for d in st.decisions)
    assert "spec_prefix_cache_safe_full_attention" not in FORK.features
    st = plan.numerics(_ctx("qwen3-8b", record=FORK, speculative=eagle))
    assert st.settings.get("prefix_cache.enabled") is False


def test_an_inherited_record_does_not_claim_prefix_cache_safety():
    """T-REC-05: the scan verifies one build; a build that inherits the record by upstream minor drops the claim."""
    rec = record_from_budconnect(
        {
            "engine": "vllm",
            "engine_version": "0.10.1",
            "engine_image": "registry.bud.studio/runtime/vllm-cuda:0.10.1",
            "upstream_vllm": "0.31.0",
        }
    )
    assert rec is not None and "spec_prefix_cache_safe_full_attention" not in rec.features

@pytest.mark.parametrize(
    "sampling, conc, k",
    [("sampled", 16, 2), ("greedy", 16, 3), ("greedy", 64, 1), ("unknown", 16, 2)],
)
def test_spec_default_k(sampling, conc, k):
    st = plan.numerics(
        _ctx(
            "qwen3.8-27b",
            sampling=sampling,
            concurrency=conc,
            speculative=SpeculativeIntent(enabled=True),
        )
    )
    assert st.settings["spec.tokens"] == k


def test_spec_user_k_and_clamp_and_advice():
    st = plan.numerics(
        _ctx("qwen3.8-27b", speculative=SpeculativeIntent(enabled=True, num_speculative_tokens=6))
    )
    assert st.settings["spec.tokens"] == 3 and any("clamped" in d for d in st.decisions)
    advice = st.stage1["spec_decode"]["advice"]
    assert set(advice) == {"1", "2", "3"} and set(advice["3"]) == {"1", "16", "64"}
    assert advice["3"]["1"]["greedy"] > 1.0


# ====================================================================== validation records (T-VAL)


def test_validation_tolerance():
    assert judge(94.9, 94.3)[0] == "pass"
    assert judge(94.9, 93.7)[0] == "fail"
    assert judge(94.9, 95.0, garbage=3)[0] == "fail"


def test_validation_gate_and_applicability():
    store = default_validation()
    assert store.gate("Qwen3-8B", None, "quantized", "vllm@0.30.0", "weights.format=fp8").ok
    assert not store.gate("Qwen3-8B", None, "finetune", "vllm@0.30.0", "weights.format=fp8").ok
    oss = store.gate("gpt-oss-20b", None, "base", "vllm@0.30.0", "weights.online=fp8")
    assert not oss.ok and "garbage" in oss.reason
    # bud-vllm 0.10.0 is upstream 0.31 re-tagged: without its own runs nothing is inherited from 0.30
    assert store.inherit["bud-vllm@0.10.0"] == "vllm@0.31.0"
    without = ValidationStore(
        [r for r in store.records if r.record != "bud-vllm@0.10.0"], inherit=store.inherit
    )
    assert without.gate("Qwen3.8-27B", None, "base", "bud-vllm@0.10.0", "spec.method=mtp").ok is False


@pytest.mark.parametrize(
    "family, setting",
    [
        ("Qwen3-8B", "spec.method=eagle3"),
        ("Qwen3-8B", "weights.online=fp8"),
        ("Qwen3.8-27B", "spec.method=mtp"),
        ("Qwen3.8-27B", "weights.online=fp8"),
        ("Qwen3.8-27B", "state.dtype=bfloat16"),
    ],
)
def test_bud_vllm_0_10_0_numerics_are_validated_on_the_build(family, setting):
    """T-VAL-06: round 7 ran GSM8K on bud-vllm 0.10.0 (the plan's settings against a same-build bf16 baseline), so the
    planner may render them there."""
    verdict = default_validation().gate(family, None, "base", "bud-vllm@0.10.0", setting)
    assert verdict.ok, verdict.reason
    rec = next(
        r
        for r in default_validation().records
        if r.record == "bud-vllm@0.10.0" and r.setting == setting and r.target == f"family:{family}"
    )
    assert rec.provenance["date"] == "2026-10-18" and rec.evidence["baseline_config"].startswith("c-")


def test_validation_inheritance_needs_unchanged_kernels():
    rec = ValidationRecord(
        target="family:F",
        record="vllm@0.30.0",
        setting="state.dtype=bfloat16",
        result="pass",
        kernels_unchanged=("state.dtype",),
    )
    store = ValidationStore([rec], inherit={"vllm@0.31.0": "vllm@0.30.0"})
    assert store.gate("F", None, "base", "vllm@0.31.0", "state.dtype=bfloat16").ok
    rec.kernels_unchanged = ()
    assert not store.gate("F", None, "base", "vllm@0.31.0", "state.dtype=bfloat16").ok


def test_newer_evidence_supersedes_an_older_record():
    """T-VAL-05: a record re-run on the same build replaces the earlier verdict either way (the 27B fp8 KV fail of 10-16 lacked
    only the 128K needle; the TCS run of 10-18 supplies it)."""

    def rec(result, date):
        return ValidationRecord(
            target="family:F",
            record="vllm@0.30.0",
            setting="kv.dtype=fp8",
            result=result,
            reason=f"{result} on {date}",
            provenance={"date": date},
        )

    older_fail, newer_pass = rec("fail", "2026-10-16"), rec("pass", "2026-10-18")
    for records in ([older_fail, newer_pass], [newer_pass, older_fail]):
        verdict = ValidationStore(records).gate("F", None, "base", "vllm@0.30.0", "kv.dtype=fp8")
        assert verdict.ok and verdict.reason == "pass on 2026-10-18"
    newer_fail = rec("fail", "2026-10-19")
    assert (
        not ValidationStore([newer_pass, newer_fail])
        .gate("F", None, "base", "vllm@0.30.0", "kv.dtype=fp8")
        .ok
    )
    # same date: the record loaded last (a later file) wins
    same_day = rec("fail", "2026-10-18")
    assert (
        not ValidationStore([newer_pass, same_day])
        .gate("F", None, "base", "vllm@0.30.0", "kv.dtype=fp8")
        .ok
    )
    # a model-specific record still beats any family record, however new
    model_fail = ValidationRecord(
        target="model:m",
        record="vllm@0.30.0",
        setting="kv.dtype=fp8",
        result="fail",
        provenance={"date": "2000-01-01"},
    )
    assert (
        not ValidationStore([newer_pass, model_fail])
        .gate("F", "m", "base", "vllm@0.30.0", "kv.dtype=fp8")
        .ok
    )


def test_fault_injection_is_off_by_default_and_corrected_by_the_ladder():
    spec, _ = _spec("qwen3-8b", versions=PLANNER_CAPACITY)
    assert spec.settings["memory.device_fraction"] == UP.default_fraction(VGPU)
    ctx = _ctx("qwen3-8b", concurrency=16)
    ctx.module_versions = {**PLANNER_CAPACITY, "test.fault_inject": "on"}
    ctx.cluster["fault_inject"] = {"memory.device_fraction": 0.99}
    ctx.placement = _placement(VGPU)
    faulty = plan.spec(ctx)
    assert faulty.engine_container().args["gpu-memory-utilization"] == 0.99
    stock = [a for a in faulty.alternatives if a.stock_equivalent][0]
    assert stock.patch["containers.engine.args"]["gpu-memory-utilization"] == UP.default_fraction(
        VGPU
    )


# ====================================================================== renderer contract (budcluster review)


@pytest.mark.parametrize("versions", [LEGACY, PLANNER], ids=["legacy", "planner"])
def test_served_name_stays_a_renderer_binding(versions):
    """budsim cannot know the namespace: a node group's served-model-name never replaces $(BUD_SERVED_NAME)."""
    group = {
        "args": {
            "model": "/data/models-registry/m",
            "served-model-name": "bud-kv-golden",
            "tensor-parallel-size": 1,
            "pipeline-parallel-size": 1,
        }
    }
    spec, _ = _spec(versions=versions, group=group)
    assert spec.engine_container().args["served-model-name"] == "$(BUD_SERVED_NAME)"
    assert spec.engine_container().args["model"] == "$(BUD_MODEL_PATH)"


def test_spec_carries_the_records_boot_failure_patterns():
    import re

    spec, _ = _spec()
    wire = spec.to_wire()
    patterns = wire["engine"]["boot_failure_patterns"]
    assert set(patterns) >= {"oom", "refused"}
    # budcluster's pod_health reports a SIGKILL as "exit 137"; the vLLM log says "exit code 137".
    for reason in (
        "container terminated: OOMKilled (exit 137)",
        "process exited with exit code 137",
    ):
        assert any(re.search(p, reason) for p in patterns["oom"]), reason
    assert any(
        re.search(p, "ValueError: max_num_seqs exceeds available Mamba cache blocks")
        for p in patterns["refused"]
    )


def test_requires_names_secret_volumes_and_unknown_resources_exactly():
    """A primitive the renderer may not support must be named as itself, so the renderer refuses it."""
    from llm_memory_calculator.runtime.compose import _requires

    spec, _ = _spec()
    wire = spec.to_wire()
    wire["pod"]["volumes"].append({"name": "token", "kind": "secret"})
    wire["containers"][0]["resources"]["limits"]["example.com/fpga"] = 1
    req = _requires(wire)
    assert "volume.secret" in req and "resources.example.com/fpga" in req
    assert "volume.configMap" not in req


def test_every_rung_carries_its_predicted_memory_margin():
    """A-06 / FR-GND-6: a boot outcome is reported with the margin the plan predicted for the rung that booted (or
    not), so a band can be widened by how wrong it was. Rung 1's margin rides on the boot_failed check.
    """
    spec, ctx = _spec("qwen3.8-27b", conc=16, speculative=SpeculativeIntent(enabled=True))
    (check,) = [c.model_dump() for c in spec.checks if c.kind == "boot_failed"]
    lim = ctx.predictor.limits(spec.settings, ctx.model, ctx.device, draft=None)
    smallest = min(c.margin_bytes for c in lim.checks)
    assert check["margin_bytes"] == int(smallest)
    assert spec.alternatives and all(
        isinstance(a.predicted_margin_bytes, int) for a in spec.alternatives
    )
    stock = [a for a in spec.alternatives if a.stock_equivalent]
    assert stock and stock[0].predicted_margin_bytes is not None
    wire = spec.to_wire()
    assert all("predicted_margin_bytes" in a for a in wire["alternatives"])


def test_budconnect_dev_s_vllm_0_10_0_is_upstream_0_31():
    """T-REC-04: BudConnect dev's new CUDA engine row (vllm 0.10.0, ``registry.bud.studio/runtime/vllm-cuda:0.10.0``,
    no upstream or KV features sent) resolves to the vendored record of that build. On TCS (10-19) the image is
    upstream ``vllm/vllm-openai:v0.31.0`` re-tagged (``VLLM_BUILD_COMMIT`` db9527a4), so it has MTP, EAGLE3 and the
    fused GDN MTP kernel; prefix caching with speculation stays off until the build is validated (vLLM #53912)."""
    from llm_memory_calculator.runtime.records import record_from_budconnect

    row = {
        "engine": "vllm",
        "engine_version": "0.10.0",
        "engine_image": "registry.bud.studio/runtime/vllm-cuda:0.10.0",
        "upstream_vllm": None,
        "kv_features": None,
        "supports_lora": True,
        "reasoning_parser_type": "qwen3",
        "tool_calling_parser_type": "qwen3_coder",
    }
    rec = record_from_budconnect(row)
    assert rec is not None and rec.id == "bud-vllm@0.10.0"
    assert rec.upstream_vllm == "0.31.0"
    assert {"spec_mtp", "spec_eagle3", "fused_gdn_mtp", "mamba_state_bf16", "online_fp8"} <= set(rec.features)
    assert "spec_prefix_cache_safe" not in rec.features
    assert rec.images["cuda"] == "registry.bud.studio/runtime/vllm-cuda:0.10.0"


# ====================================================================== shared mode on the slice (TCS 10-18)

TCS_8B_SHARED_GROUP = {
    # what budsim shipped for Qwen3-8B, shared, c16, 1024/256 on TCS (the legacy terms size a 21353 MiB slice)
    "weight_memory_gb": 16.4,
    "kv_cache_memory_gb": 3.3,
    "activation_memory_gb": 0.49,
    "state_memory_gb": 0.0,
    "sampler_logits_gb": 0.01,
    "supports_lora": False,
}


def _shared_spec(model="qwen3-8b", record=FORK, **intent):
    """As budsim plans shared mode: the device is the whole card in shared mode; the slice is the plan's to size."""
    ctx = _ctx(model, record, VGPU, concurrency=16, hardware_mode="shared", **intent)
    ctx.module_versions = dict(PLANNER)
    card = VGPU.with_mode(HardwareMode("shared", VGPU.memory_visible_bytes))
    ctx.device = card
    ctx.placement = _placement(card, conc=16, mode="shared", group=TCS_8B_SHARED_GROUP)
    return plan.spec(ctx), ctx


def test_shared_slice_holds_the_planned_load_at_the_planned_settings():
    """FRD-027 §10.3: shared mode is planned and predicted on the slice the pod gets.

    TCS 10-18: the slice came from the legacy terms (21353 MiB) while context/batching were planned on the whole card:
    40960 tokens need 2560 blocks and the slice held 1211, so the pod could never boot.
    """
    spec, ctx = _shared_spec()
    eng = spec.engine_container()
    slice_mib = int(eng.resources.limits["nvidia.com/gpumem"])
    on_slice = VGPU.with_mode(HardwareMode("shared", slice_mib * 2**20))
    lim = ctx.predictor.limits(spec.settings, ctx.model, on_slice)
    assert lim.boots, [c.message for c in lim.failures]
    assert lim.kv_tokens >= 16 * (1024 + 256)  # the planned load runs without queueing on its slice
    assert spec.settings["context.max_tokens"] <= lim.max_bootable_context
    (check,) = [c.model_dump() for c in spec.checks if c.kind == "boot_failed"]
    # the context cap leaves the memory band (FR-GND-6), so a pool that much smaller still boots (TCS: margin was 0)
    assert lim.margin("one_request_fits") >= lim.memory_band_bytes > 0
    assert check["margin_bytes"] > 0
    hami = [c.model_dump() for c in spec.checks if c.kind == "hami_slice"][0]
    assert hami["slice_mib"] == slice_mib
    assert slice_mib >= 21353  # never below the legacy (FRD-023) slice
    assert slice_mib * 2**20 < VGPU.memory_visible_bytes  # still a slice, not the card


TCS_OSS_SHARED_GROUP = {
    # what budsim shipped for gpt-oss-20b, shared, c16, 1024/256 on TCS (simulation f537436d; an 18,688 MiB slice)
    "weight_memory_gb": 12.81,
    "kv_cache_memory_gb": 1.03,
    "activation_memory_gb": 0.38,
    "state_memory_gb": 0.0,
    "sampler_logits_gb": 0.01,
    "supports_lora": False,
}


def test_a_slice_s_stock_equivalent_rung_still_boots_on_it():
    """FRD-027 §10.10: every rung of a slice's ladder boots on that slice, the stock-equivalent one included.

    TCS 10-18 (I-RENDER-05 run, gpt-oss on 18,688 MiB): rung 5 dropped max-model-len, so the engine default (131,072
    tokens) needed 3.1 GiB of KV for one request where the slice left 2.31 GiB, and the last rung was refused. On a
    slice the context cap is what makes the pod fit, not a tuning choice: rung 5 keeps the largest one that boots.
    """
    ctx = _ctx("gpt-oss-20b", FORK, VGPU, concurrency=16, hardware_mode="shared")
    ctx.module_versions = dict(PLANNER)
    card = VGPU.with_mode(HardwareMode("shared", VGPU.memory_visible_bytes))
    ctx.device = card
    ctx.placement = _placement(card, conc=16, mode="shared", group=TCS_OSS_SHARED_GROUP)
    spec = plan.spec(ctx)
    slice_mib = int(spec.engine_container().resources.limits["nvidia.com/gpumem"])
    on_slice = VGPU.with_mode(HardwareMode("shared", slice_mib * 2**20))
    wire = spec.to_wire()
    (stock,) = [a for a in wire["alternatives"] if a["stock_equivalent"]]
    for alt in wire["alternatives"]:
        lim = ctx.predictor.limits(alt["settings"], ctx.model, on_slice)
        assert lim.boots, (alt["rung"], [c.message for c in lim.failures])
    assert stock["settings"]["context.max_tokens"] <= ctx.predictor.limits(
        stock["settings"], ctx.model, on_slice
    ).max_bootable_context
    assert "rung 5" in stock["decision"]


def test_shared_prediction_is_on_the_slice():
    spec, ctx = _shared_spec()
    slice_mib = int(spec.engine_container().resources.limits["nvidia.com/gpumem"])
    on_slice = VGPU.with_mode(HardwareMode("shared", slice_mib * 2**20))
    pred = ctx.predictor.predict(spec.settings, ctx.workload(16), ctx.model, on_slice)
    assert spec.to_wire()["predicted"]["point"]["throughput_tok_s"] == round(
        pred.throughput_tok_s.value, 1
    )


def test_tcs_quality_run_validates_online_fp8_on_the_fork():
    """A-04 on TCS (10-18): the fork's online FP8 8B scored GSM8K 93.10 / 93.00 (bf16 92.95 / 93.75), no garbage.

    That record is the FR-VAL evidence. The explicit choice no longer warns; auto now offers the candidate and picks it
    for the 8B, whose predicted gain (TCS measured 1.39x at c16) separates; the 27B, with no record, keeps bf16.
    """
    explicit, _ = _spec("qwen3-8b", record=FORK, versions=PLANNER, weight_precision="online:fp8")
    assert explicit.engine_container().args.get("quantization") == "fp8"
    assert not [d for d in explicit.decisions if "no validation record" in d]
    auto, _ = _spec("qwen3-8b", record=FORK, versions=PLANNER)
    assert auto.engine_container().args.get("quantization") == "fp8"
    (chosen,) = [d for d in auto.decisions if "online:fp8 chosen" in d]
    gain = float(chosen.split("predicted ")[1].split("x")[0])
    assert abs(gain / 1.39 - 1) <= 0.10  # TCS: 2477 vs 1789 tok/s at c16
    big, _ = _spec("qwen3.8-27b", record=FORK, versions=PLANNER)
    assert "quantization" not in big.engine_container().args
