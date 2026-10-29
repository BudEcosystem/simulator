"""Round 18: a shipped layout rule held to the keys its class builds the layout from, the training
optimizer's model size, and the gate's truths for GPT-2-style maps and training activations.

* The layout a transformers 5.12 config class derives ships as a rule
  (``config_class_defaults.json``'s ``layouts``), checked at the class's defaults alone. A config
  that states another value of a key the class builds its layout from got the class's default
  layout: AfmoeConfig's `global_attn_every_n_layers: 2` gave 8 full layers of 32 where the class
  builds 16 (below origin/main and below the class); Llama 4's `no_rope_layer_interval`,
  SmolLM3's `no_rope_layers`, an EXAONE 4 string pattern, LFM2's `full_attn_idxs` and a Qwen2-VL
  / dots1 `max_window_layers` under the window likewise. Each rule now records those keys
  (``depends``: found by building the class at other values) with the values it was checked at;
  any other value is the class's to answer, and without transformers every windowed layer is
  charged full. A class whose `max_window_layers` default (dots1's 62, Qwen2-VL's 80) is deeper
  than the depths the rule was checked at is checked deeper, so its rule is `max_window_layers`.
* ``training.analyze_model_config`` (behind ``optimize_training``, ``compare_training_strategies``
  and ``get_minimum_hardware_for_model``) sized a model from its top-level fields: DeepSeek-V3
  38.6B of its 671B. It reads the decoder where the model keeps it and the routed experts under
  every spelling (``layer_plan``), and its count is held to the model's own, as every other
  training estimator's is. The harness measures it, and the gate holds it.
* ``kv_truth``: the LoRA floor and the cross-attention cache of a GPT-2-style decoder (whose maps
  are transformers' ``Conv1D``) were 0; training_modeling's activation truth was one checkpoint
  a decoder layer, 3-30x below the figure. The truth reads ``Conv1D`` as a linear map, a fused
  cross-attention projection beside its own queries as K and V, an image or audio encoder's
  length from its position table, and the activations of a layer-checkpointed step: one input a
  layer of every stack (the encoder too) and one layer's live maps.
"""

import copy
import json
import os
import sys
import warnings

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")
SCRIPTS = os.path.join(os.path.dirname(HERE), "scripts")
sys.path.insert(0, SCRIPTS)


def _synth(name):
    import kv_corpus

    return copy.deepcopy(kv_corpus.synthetic_configs(FIXTURES)[name])


def _kinds(cfg):
    from llm_memory_calculator.layer_plan import layer_kinds

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k = layer_kinds(copy.deepcopy(cfg))
    return k["attn"].count("full"), k["attn"].count("sliding"), k["attn"].count("chunked")


def _snapshot():
    from llm_memory_calculator.layer_plan import _class_default_snapshot

    return _class_default_snapshot()["layouts"]


@pytest.fixture
def without_transformers(monkeypatch):
    from llm_memory_calculator import layer_plan

    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setattr(layer_plan, "_CLASS_LAYOUT_CACHE", {})
    yield


@pytest.fixture
def with_transformers_5_12(monkeypatch):
    transformers = pytest.importorskip("transformers")
    if not transformers.__version__.startswith("5.12"):
        pytest.skip("the class layouts asserted are transformers 5.12's")
    from llm_memory_calculator import layer_plan

    monkeypatch.setattr(layer_plan, "_CLASS_LAYOUT_CACHE", {})
    yield


def _class_full(cfg):
    """Full-attention layers transformers 5.12's own config class derives for the decoder."""
    from transformers import AutoConfig

    d = copy.deepcopy(cfg)
    d.pop("architectures", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = AutoConfig.for_model(**d)
    tc = c.get_text_config(decoder=True)
    return sum(1 for x in tc.layer_types if str(x) == "full_attention")


# ------------------------------------------------------------ the rules record their layout keys

@pytest.mark.parametrize("model_type,key", [
    ("afmoe", "global_attn_every_n_layers"),
    ("llama4_text", "no_rope_layer_interval"),
    ("llama4_text", "no_rope_layers"),
    ("smollm3", "no_rope_layers"),
    ("exaone4", "sliding_window_pattern"),
    ("lfm2", "full_attn_idxs"),
    ("jamba", "attn_layer_period"),
    ("cohere2", "sliding_window_pattern"),
])
def test_a_shipped_rule_names_every_key_its_class_builds_the_layout_from(model_type, key):
    rule = _snapshot()[model_type]
    assert key in (rule.get("depends") or {}), rule


@pytest.mark.parametrize("model_type", ["dots1", "qwen2_vl_text", "qwen2_5_vl_text"])
def test_a_class_that_windows_from_max_window_layers_ships_that_rule(model_type):
    """Checked only up to 48 layers, dots1's default `max_window_layers` (62) and Qwen2-VL's (80)
    never showed: their rule was "every layer full", and a config that switches the window on
    with `max_window_layers: 8` was charged every layer full where the class windows 16 of 24."""
    rule = _snapshot()[model_type]
    assert rule["rule"] == "max_window_layers", rule


# ----------------------------------------------------- a stated value the rule was not checked at

#: (synthetic config, full layers transformers 5.12's class builds)
STATED_LAYOUT_KEYS = [
    ("afmoe-global-every-2", 16),
    ("llama4-no-rope-interval-2", 24),
    ("llama4-no-rope-layers-half", 24),
    ("smollm3-window-all-rope", 36),
    ("smollm3-window-interval-8", 32),
    ("lfm2-full-attn-idxs", 6),
    ("lfm2-full-attn-idxs-no-window", 6),
    ("dots1-window-from-layer-8", 8),
    ("qwen2-vl-window-from-layer-8", 8),
]


@pytest.mark.parametrize("name,full", STATED_LAYOUT_KEYS)
def test_a_stated_layout_key_is_never_read_below_the_class_without_transformers(
        name, full, without_transformers):
    """Without the class to ask, a layout the rule does not describe is charged full on every
    windowed layer -- above the class, never below it."""
    got, _, _ = _kinds(_synth(name))
    assert got >= full, (name, got, full)


def test_the_afmoe_review_case_is_not_below_the_class_without_transformers(without_transformers):
    """The review's case: 8 full layers of 32 from the rule, where AfmoeConfig builds 16."""
    from llm_memory_calculator.layer_plan import stated_layer_types

    names, source, _ = stated_layer_types(_synth("afmoe-global-every-2"))
    assert source != "config_class"
    assert names.count("full_attention") >= 16


@pytest.mark.parametrize("name,full", STATED_LAYOUT_KEYS)
def test_a_stated_layout_key_takes_the_class_s_layout(name, full, with_transformers_5_12):
    """With the class installed, a windowed config takes the class's own layout. Llama 4 states a
    chunk (`attention_chunk_size`), not a window: its layers are read from a stated `layer_types`
    and, without one, charged full -- above the class's, as before."""
    cfg = _synth(name)
    assert _class_full(cfg) == full
    got, _, _ = _kinds(cfg)
    if name.startswith("llama4"):
        assert got >= full, (name, got)
    else:
        assert got == full, (name, got)


def test_a_string_pattern_the_class_cannot_build_is_charged_full(without_transformers):
    """EXAONE 4's class raises on a string `sliding_window_pattern` (5.12 reads it as a number):
    the rule's fixed period is no answer for it; every layer is charged full."""
    full, sliding, _ = _kinds(_synth("exaone4-string-pattern"))
    assert (full, sliding) == (30, 0)


def test_a_stated_class_default_still_takes_the_rule_without_transformers(without_transformers):
    """A config that states the class's own default (`global_attn_every_n_layers: 4`, a Llama 4
    hub config's `no_rope_layers` as `save_pretrained` writes it) keeps the shipped layout."""
    full, sliding, _ = _kinds(_synth("afmoe-global-every-4"))
    assert (full, sliding) == (8, 24)
    from llm_memory_calculator.layer_plan import layout_rule_applies

    rule = _snapshot()["llama4_text"]
    assert layout_rule_applies(rule, {"num_hidden_layers": 48,
                                      "no_rope_layers": [1, 1, 1, 0] * 12})
    assert not layout_rule_applies(rule, {"num_hidden_layers": 48,
                                          "no_rope_layers": [0, 1] * 24})
    assert not layout_rule_applies(rule, {"num_hidden_layers": 48, "no_rope_layer_interval": 2})


def test_a_qwen2_vl_window_from_max_window_layers_without_transformers(without_transformers):
    full, sliding, _ = _kinds(_synth("qwen2-vl-window-from-layer-8"))
    assert (full, sliding) == (8, 20)


def test_the_reduced_retry_keeps_the_rule_s_layout_keys():
    """When a config does not validate as a whole, the class is asked again with the layout keys
    alone; those include every key the shipped rule depends on (else the retry answers for the
    class's defaults)."""
    from llm_memory_calculator import layer_plan

    src = open(layer_plan.__file__).read()
    assert 'set((rule or {}).get("depends") or ())' in src


def test_every_shipped_rule_records_every_layout_key_of_its_class(with_transformers_5_12):
    """Rebuild each rule's dependencies with the generator's own probe: what the snapshot ships
    is what the 5.12 classes do, and an accepted value gives the rule's list at depths the probe
    did not check."""
    from transformers import CONFIG_MAPPING

    import config_class_defaults as ccd
    from llm_memory_calculator.layer_plan import class_layout_names, layout_rule_applies

    extra = sorted({k for r in _snapshot().values() for k in (r.get("depends") or {})})
    wrong = []
    for mt, rule in sorted(_snapshot().items()):
        if rule["rule"] == "none":
            continue
        cls = CONFIG_MAPPING[mt]
        base = {k: v for k, v in rule.items() if k != "depends"}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            deps = ccd._layout_dependencies(cls, base, cls().to_dict(), extra)
        if sorted(deps) != sorted(rule.get("depends") or {}):
            wrong.append((mt, sorted(deps), sorted(rule.get("depends") or {})))
            continue
        for key, entry in deps.items():
            for value in entry["values"]:
                for n in (7, 30, 61):
                    tc = {"num_hidden_layers": n, key: value}
                    if "switch" in rule:
                        tc["use_sliding_window"] = True
                    assert layout_rule_applies(rule, tc)
                    kwargs = dict(tc)
                    if "sliding_window" in cls().to_dict():
                        kwargs["sliding_window"] = 2048
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            got = [str(x) for x in cls(**kwargs).layer_types]
                    except Exception:  # noqa: BLE001 - the default of a key read from kwargs
                        continue
                    if got != class_layout_names(rule, tc):
                        wrong.append((mt, key, value, n))
    assert not wrong, wrong


# ---------------------------------------------------------------- the training optimizer's size

def _deepseek_v3():
    with open(os.path.join(FIXTURES, "planner", "models", "deepseek-v3.json")) as f:
        return json.load(f)


def test_analyze_model_config_counts_deepseek_v3_s_routed_experts():
    """`n_routed_experts` (256 experts of 2,048) was not read: 38.6B of 671B."""
    from llm_memory_calculator.training import analyze_model_config

    m = analyze_model_config(_deepseek_v3())
    assert m.num_parameters >= 671e9
    assert m.is_moe and m.moe.num_experts == 256 and m.moe.expert_intermediate_size == 2048
    # a token runs through 8 routed experts and the shared one, not every expert
    assert 30e9 < m.active_parameters < 60e9


def test_optimize_training_and_minimum_hardware_size_the_whole_model():
    from llm_memory_calculator.training import optimize_training
    from llm_memory_calculator.training.training_optimizer import get_minimum_hardware_for_model

    plan = optimize_training(_deepseek_v3(), total_tokens=100_000_000, max_gpus=64).to_dict()
    assert plan["model"]["total_params"] >= 671e9
    hw = get_minimum_hardware_for_model(_deepseek_v3(), training_method="lora")
    assert hw["min_memory_gb"] >= 2 * 671


def test_analyze_model_config_reads_a_nested_decoder():
    """A multimodal config keeps its decoder under `text_config`: read at the top level, Qwen3.5
    was a 4,096-wide, 32-layer default."""
    from llm_memory_calculator.training import analyze_model_config

    with open(os.path.join(FIXTURES, "hybrid_configs", "Qwen_Qwen3.6-27B.json")) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config") or cfg
    m = analyze_model_config(cfg)
    assert m.hidden_size == tc["hidden_size"] and m.num_layers == tc["num_hidden_layers"]
    assert m.num_parameters >= 26e9


def test_the_harness_measures_the_training_optimizer():
    import kv_corpus

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = kv_corpus.measure_training_estimators(_deepseek_v3())
    assert out["amc_params"] >= 671e9 and out["topt_params"] >= 671e9
    assert out["minhw_lora_gb"] >= 2 * 671 and out["minhw_full_gb"] >= 14 * 671


def test_the_gate_holds_the_training_optimizer_to_the_model():
    import kv_truth

    p = {"params": 1_000_000_000, "class": "X"}
    for reader, want in (("amc_params", 1e9), ("topt_params", 1e9), ("minhw_lora_gb", 2.0),
                         ("minhw_full_gb", 14.0)):
        truth, _ = kv_truth.figure_truth(p, reader)
        assert truth == pytest.approx(want), reader


# ------------------------------------------------------------------------- the gate's truths

@pytest.fixture(scope="module")
def gpt2_profiles():
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not transformers.__version__.startswith("5.12"):
        pytest.skip("the truths are transformers 5.12's")
    import kv_truth

    return {n: kv_truth.truth_profile(_synth(n)) for n in ("gpt2", "vision-enc-dec-gpt2")}


def test_a_gpt2_decoder_s_conv1d_maps_are_linear_maps(gpt2_profiles):
    """GPT-2's c_attn (768 -> 2,304), c_proj, c_fc (768 -> 3,072) and the MLP's c_proj are
    `Conv1D`: (in + out) a layer 3,072 + 1,536 + 3,840 + 3,840 = 12,288."""
    p = gpt2_profiles["gpt2"]
    assert p["lora_linear_features"] == 12 * 12_288
    assert p["lora_widest_out"] == 3072


def test_a_gpt2_cross_attention_caches_k_and_v(gpt2_profiles):
    """GPT-2's cross-attention fuses K and V in `c_attn` (768 -> 1,536) beside its own `q_attn`;
    the ViT encoder emits 197 tokens (196 patches and the class token)."""
    p = gpt2_profiles["vision-enc-dec-gpt2"]
    layers = len(p["kv"])
    assert p["cross_kv"] == layers * 2 * 768
    assert p["encoder_tokens"] == 197
    import kv_truth

    truth, _ = kv_truth.figure_truth(p, "serving_state_bytes")
    assert truth == 2 * layers * 2 * 768 * 197
    lora, _ = kv_truth.figure_truth(p, "vllm_lora_gb")
    assert lora > 0


def test_the_activation_truth_counts_every_stack_and_a_live_layer():
    """t5-small (6 + 6 layers of 512, d_ff 2,048, not gated): one input a layer of both stacks
    and one layer's live maps -- Q, K, V, O and the FFN's input map -- at 2,048 tokens."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not transformers.__version__.startswith("5.12"):
        pytest.skip("the truths are transformers 5.12's")
    import kv_truth

    p = kv_truth.truth_profile(_synth("t5-small"))
    want = 2048 * (12 * 512 + 4 * 512 + 2048)
    assert p["activation_floor_elements_2048"] == want
    truth, _ = kv_truth.figure_truth(p, "tmodel_full[2]")
    assert truth == pytest.approx(2 * want / 1e9)


# ------------------------------------------------- what the tighter truths found below them

def _tmodel_activation_gb(cfg):
    import contextlib
    import importlib
    import io

    from llm_memory_calculator.genz.Models.get_language_model import get_configs

    tm = importlib.import_module("llm_memory_calculator.genz.LLM_training.training_modeling")
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o = tm.training_modeling(model=get_configs(copy.deepcopy(cfg)), batch_size=1,
                                 seq_length=2048, system_name="A100_80GB_GPU", method="full")
    return float(o.activation_memory_gb)


def test_training_modeling_sizes_an_encoder_at_its_own_width():
    """Pix2Struct's image encoder is 1,536 wide (gated FFN 3,968) beside a 1,024-wide decoder:
    its 18 layers' checkpoints and its live layer at the decoder's width were 0.191 GB, below
    the 0.2464 GB a layer-checkpointed step holds at least."""
    floor = 2 * 2048 * (18 * 1024 + 18 * 1536 + (4 * 1536 + 2 * 3968)) / 1e9
    assert _tmodel_activation_gb(_synth("pix2struct-seq-len")) >= floor


def test_training_modeling_holds_a_layer_wider_than_four_hidden():
    """Qwen3.5's Gated DeltaNet layer projects q|k|v (8,192), z (4,096), a and b (32 each) from a
    2,560-wide stream, and its FFN's gate and up are 9,216 each: 30,784 live elements a token,
    above the 4 x 2,560 + 2 x 9,216 the decoder fields alone give."""
    cfg = _synth("qwen35-list-depth")
    tc = cfg["text_config"]
    n = len(tc["layer_types"])
    floor = 2 * 2048 * (n * 2560 + 8192 + 4096 + 32 + 32 + 2 * 9216) / 1e9
    assert _tmodel_activation_gb(cfg) >= floor


def _lora_adapter_gb(cfg):
    from llm_memory_calculator import calculate_memory

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = calculate_memory(copy.deepcopy(cfg), batch_size=1, seq_length=4096,
                               precision="bf16", max_loras=4, max_lora_rank=16)
    return rep.lora_adapter_memory_bytes / 1e9


def test_the_calculator_s_lora_buffers_cover_a_mixer_the_class_places(with_transformers_5_12):
    """LFM2 with a window and `full_attn_idxs` and no `layer_types`: the class builds 18 of its
    24 layers as short convolutions. The layer plan does not read that layout, so the LoRA
    buffers held the 6 attention layers' maps and none of the convolutions' (0.0755 GB, below
    origin/main's 0.088 and vLLM's 0.109 floor: 854,016 features x 4 x 16 x 2 B)."""
    assert _lora_adapter_gb(_synth("lfm2-full-attn-idxs")) >= 854_016 * 128 / 1e9


def test_the_calculator_s_lora_buffers_read_a_gated_ffn_by_its_family():
    """LFM2 states no activation and builds w1 / w3 / w2: counted with two maps, LFM2-2.6B's
    buffers were 0.156 GB against vLLM's 0.196 GB floor (1,528,832 features)."""
    with open(os.path.join(FIXTURES, "hybrid_configs", "LiquidAI_LFM2-2.6B.json")) as f:
        cfg = json.load(f)
    assert _lora_adapter_gb(cfg) >= 1_528_832 * 128 / 1e9
