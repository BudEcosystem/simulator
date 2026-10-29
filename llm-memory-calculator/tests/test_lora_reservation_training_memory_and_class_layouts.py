"""Round 17: the vLLM planner's LoRA reservation, GenZ training_modeling's memory figures, and the
layout a config class gives a windowed config, each read from the model whatever is installed.

* ``engine.vllm._lora_bytes`` (behind ``VLLMEngine.limits`` / ``predict`` whenever
  ``lora.max_adapters`` is set: the legacy ``Lora`` runtime module sets it for every record that
  supports LoRA) summed the out-dims of three weight roles and 4 x hidden a layer for the in-dims.
  With the facts' roles read right (an encoder-decoder's encoder under ``encoder``, a MoE FFN under
  ``expert``, real FFN widths, decoder layers alone in ``layers.total``), that left out the FFN
  in-dims, the encoder, the experts and every mixer map: whisper-large-v3 0.126 GB against vLLM's
  0.136 GB floor. The facts now state the (in + out) features of every linear map the model's
  layers hold (``ModelFacts.lora_linear_features``), and the reservation is A and B over them.
  Floors: 4 adapters x rank 16 x 2 B x the sum of (in + out) over every ``nn.Linear`` of the
  decoder layers transformers 5.12 builds for the fixture, routed experts left out
  (``kv_truth.truth_profile``'s ``lora_linear_features``).
* ``training_modeling``'s trainable parameters (LoRA's adapters) and its activations counted the
  decoder layers alone; an encoder-decoder's encoder layers hold the same adapters and
  activations (as ``training_parallelization._estimate_memory_per_gpu`` counts them since round
  13): one count of the layers training runs through, ``memory_model.trained_layers``.
* Measuring training_modeling found its parameter count sized every routed expert at the dense
  FFN width (SmallThinker 283.5B of 21.5B); the experts are counted at their own width, and the
  facts' count it is held to now counts a shared expert's gate and a class's default Q / K / V
  biases (Qwen1.5-MoE, Qwen3-Next reach transformers' count).
* A config that states a window but no layout (the Gemma 2 hub config) took its layout from the
  installed transformers' config class: without transformers, or with 4.50, all 42 layers were
  full attention. The class's layout ships with the package as a rule
  (``config_class_defaults.json``'s ``layouts``), so the layout is the same everywhere.
"""

import copy
import json
import os
import sys
import warnings

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))


def _fixture(rel):
    with open(os.path.join(FIXTURES, rel)) as f:
        return json.load(f)


def _config(name):
    if name.startswith("synth:"):
        import kv_corpus

        return copy.deepcopy(kv_corpus.synthetic_configs(FIXTURES)[name[len("synth:"):]])
    return _fixture(name)


def _facts(cfg):
    from llm_memory_calculator.model.facts import ModelFacts

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ModelFacts.from_config(copy.deepcopy(cfg))


# ---------------------------------------------------------------- the vLLM planner's LoRA slots

#: (in + out) features over every nn.Linear of the decoder layers transformers 5.12 builds,
#: routed experts left out (kv_truth.truth_profile, lora_linear_features).
LORA_FLOOR_FEATURES = {
    "planner/models/whisper-large-v3.json": 1_064_960,
    "encoder_decoder/onnx-community_whisper-large-v3-turbo.json": 133_120,
    "hub_as_shipped/llava-hf_llava-1.5-7b-hf.json": 2_498_560,
    "hub_as_shipped/HuggingFaceM4_idefics2-8b.json": 2_621_440,
    "hybrid_configs/tiiuae_Falcon-H1-34B-Instruct.json": 9_365_760,
    "hybrid_configs/LiquidAI_LFM2-2.6B.json": 1_528_832,
    "hub_as_shipped/Salesforce_blip2-flan-t5-xl.json": 1_302_528,
    "encoder_decoder/t5_flan-xl-geometry.json": 1_302_528,
    "cross_attention/facebook_musicgen-small_hub.json": 638_976,
    "cross_attention/mllama_transformers-5.12-default.json": 3_276_800,
    "encoder_decoder/bart_transformers-5.12-default.json": 319_488,
    "hub_as_shipped/microsoft_Florence-2-base-ft.json": 119_808,
    "synth:nllb-moe": 1_168_128,
}


@pytest.mark.parametrize("name", sorted(LORA_FLOOR_FEATURES))
def test_the_vllm_planner_reserves_at_least_vllms_lora_buffers(name):
    from llm_memory_calculator.engine.vllm import _lora_bytes

    facts = _facts(_config(name))
    floor = 2 * 4 * 16 * LORA_FLOOR_FEATURES[name]
    assert _lora_bytes(facts, 4, 16) >= floor, (name, _lora_bytes(facts, 4, 16), floor)


def test_the_reservation_is_a_and_b_over_every_linear_the_facts_state():
    from llm_memory_calculator.engine.vllm import _lora_bytes

    facts = _facts(_config("planner/models/whisper-large-v3.json"))
    assert facts.lora_linear_features > 0
    assert _lora_bytes(facts, 4, 16) == 2.0 * 4 * 16 * facts.lora_linear_features
    assert _lora_bytes(facts, 8, 64) == 2.0 * 8 * 64 * facts.lora_linear_features


def test_a_dense_decoders_features_are_its_layers_maps_exactly():
    """Llama (LLaVA's language model): per layer Q / K / V / O (4 x (4,096 + 4,096)) and the gated
    FFN (3 x (4,096 + 11,008)), 32 layers -- transformers' own count."""
    facts = _facts(_config("hub_as_shipped/llava-hf_llava-1.5-7b-hf.json"))
    assert facts.lora_linear_features == 32 * (4 * 8192 + 3 * (4096 + 11008))


def test_an_encoder_decoders_encoder_and_a_moes_experts_hold_lora_buffers_too():
    """vLLM wraps every linear layer of the model it loads (the encoder's among them) and, since
    its fused-MoE LoRA, every routed expert: both are charged, above the decoder-only floor."""
    whisper = _facts(_config("planner/models/whisper-large-v3.json"))
    decoder_only = LORA_FLOOR_FEATURES["planner/models/whisper-large-v3.json"]
    # 32 encoder layers: Q / K / V / O at 1,280 and the 5,120-wide FFN
    assert whisper.lora_linear_features >= decoder_only + 32 * (4 * 2560 + 2 * (1280 + 5120))
    moe = _facts(_config("synth:nllb-moe"))
    assert moe.lora_linear_features > LORA_FLOOR_FEATURES["synth:nllb-moe"]


def test_the_limits_charge_the_reservation():
    from llm_memory_calculator.device import catalogue
    from llm_memory_calculator.engine import EngineRecord, engine_for

    rec = _fixture("planner/records/vllm-0.30-upstream.json")
    eng, dev = engine_for(EngineRecord.from_dict(rec)), catalogue.get("h100xm-80c-vgpu")
    facts = _facts(_config("planner/models/whisper-large-v3.json"))
    plain = eng.limits({}, facts, dev)
    lora = eng.limits({"lora.max_adapters": 4, "lora.max_rank": 16}, facts, dev)
    assert lora.num_blocks < plain.num_blocks


def test_the_harness_measures_the_reservation_and_the_limits_it_leaves():
    import kv_corpus
    import kv_truth

    row = kv_corpus.measure_readers(_config("planner/models/whisper-large-v3.json"))
    assert row.get("vllm_lora_gb", 0) > 0, row.get("vllm_error")
    assert isinstance(row.get("vllm_limits_lora"), list)
    assert kv_corpus.figure_kind("vllm_lora_gb") == "memory"
    assert kv_corpus.figure_kind("vllm_limits_lora[0]") == "capacity"
    assert "vllm_lora_gb" in kv_truth._held_by("vllm_limits_lora[0]")
    profile = {"params": 1, "lora_linear_features": 1000}
    truth, why = kv_truth.figure_truth(profile, "vllm_lora_gb")
    assert truth == pytest.approx(2 * 4 * 16 * 1000 / 1e9), why


# ---------------------------------------------------------------- GenZ training_modeling's memory


def _modeling(name, method):
    from llm_memory_calculator.genz.LLM_training.training_modeling import training_modeling
    from llm_memory_calculator.genz.Models import get_configs

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return training_modeling(model=get_configs(_config(name)), batch_size=1, seq_length=2048,
                                 system_name="A100_80GB_GPU", method=method)


def test_training_modeling_counts_an_encoders_layers_in_its_activations():
    """Gradient checkpointing keeps one B x S x H input a layer: whisper-large-v3 trains 32
    encoder and 32 decoder layers of 1,280, so at least 2,048 x 1,280 x 2 B x 64 layers."""
    out = _modeling("planner/models/whisper-large-v3.json", "full")
    assert out.activation_memory_gb >= 2048 * 1280 * 2 * 64 / 1e9


def test_training_modeling_puts_lora_adapters_on_an_encoders_layers_too():
    """LoRA's gradients (bf16, 2 B an adapter parameter) cover q / k / v / o / gate / up / down of
    every layer training runs through, the encoder's 32 beside the decoder's 32."""
    from llm_memory_calculator.genz.LLM_training.training_modeling import (
        _calculate_trainable_params,
    )

    out = _modeling("planner/models/whisper-large-v3.json", "lora")
    per_32 = _calculate_trainable_params(1, "lora", 16, 32, 1280, 5120, 20, 20)
    assert out.gradient_memory_gb * 1e9 >= 2 * 2 * per_32 * 0.999


def test_one_count_of_the_layers_training_runs_through():
    from llm_memory_calculator.genz.Models import get_configs
    from llm_memory_calculator.genz.serving.memory_model import trained_layers

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        whisper = get_configs(_config("planner/models/whisper-large-v3.json"))
        llava = get_configs(_config("hub_as_shipped/llava-hf_llava-1.5-7b-hf.json"))
    assert trained_layers(whisper) == 64
    assert trained_layers(llava) == 32


def test_the_harness_measures_training_modelings_memory():
    import kv_corpus
    import kv_truth

    row = kv_corpus.measure_training(_config("planner/models/whisper-large-v3.json"))
    for key in ("tmodel_full", "tmodel_lora"):
        assert isinstance(row.get(key), list) and len(row[key]) == 3, row.get("tmodel_error")
        for i in range(3):
            assert kv_corpus.figure_kind(f"{key}[{i}]") == "memory"
    profile = {"params": 10 ** 9, "activation_floor_elements": 10 ** 6}
    for reader, want in (("tmodel_full[0]", 14.0), ("tmodel_full[1]", 2.0),
                         ("tmodel_lora[0]", 2.0), ("tmodel_lora[1]", 2.0),
                         ("tmodel_full[2]", 2 * 2048 * 10 ** 6 / 1e9),
                         ("tmodel_lora[2]", 2 * 2048 * 10 ** 6 / 1e9)):
        truth, why = kv_truth.figure_truth(profile, reader)
        assert truth == pytest.approx(want), (reader, why)


# ---------------------------------------------------------------- a class's layout, shipped

GEMMA2_9B_HUB = dict(
    model_type="gemma2", architectures=["Gemma2ForCausalLM"], hidden_size=3584,
    num_hidden_layers=42, num_attention_heads=16, num_key_value_heads=8, head_dim=256,
    intermediate_size=14336, sliding_window=4096, vocab_size=256000,
    max_position_embeddings=8192, query_pre_attn_scalar=256, torch_dtype="bfloat16",
)

MISTRAL_V01 = dict(
    model_type="mistral", architectures=["MistralForCausalLM"], hidden_size=4096,
    num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8,
    intermediate_size=14336, sliding_window=4096, vocab_size=32000,
)

QWEN2_WINDOWED = dict(
    model_type="qwen2", architectures=["Qwen2ForCausalLM"], hidden_size=3584,
    num_hidden_layers=40, num_attention_heads=28, num_key_value_heads=4,
    intermediate_size=18944, sliding_window=4096, use_sliding_window=True, vocab_size=152064,
)


@pytest.fixture
def without_transformers(monkeypatch):
    """The readers with transformers absent: `import transformers` fails, and no layout read
    through it earlier is kept."""
    from llm_memory_calculator import layer_plan

    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setattr(layer_plan, "_CLASS_LAYOUT_CACHE", {})
    yield


def test_the_gemma2_hub_config_alternates_without_transformers(without_transformers):
    from llm_memory_calculator.layer_plan import stated_layer_types

    names, source, _ = stated_layer_types(dict(GEMMA2_9B_HUB))
    assert names == ["sliding_attention", "full_attention"] * 21, source
    facts = _facts(GEMMA2_9B_HUB)
    assert facts.layers.full == 21 and facts.layers.sliding == 21


def test_a_class_that_derives_no_layout_keeps_the_window_uniform_without_transformers(
        without_transformers):
    from llm_memory_calculator.layer_plan import stated_layer_types

    names, source, _ = stated_layer_types(dict(MISTRAL_V01))
    assert names is None and source == "uniform"


def test_a_max_window_layers_class_takes_its_default_without_transformers(without_transformers):
    """Qwen2Config windows the layers from `max_window_layers` (28 by default) on."""
    from llm_memory_calculator.layer_plan import stated_layer_types

    names, _, _ = stated_layer_types(dict(QWEN2_WINDOWED))
    assert names == ["full_attention"] * 28 + ["sliding_attention"] * 12
    names, _, _ = stated_layer_types({**QWEN2_WINDOWED, "max_window_layers": 10})
    assert names == ["full_attention"] * 10 + ["sliding_attention"] * 30
    # a config that leaves the switch to the class (off): every layer full, the window unused
    unswitched = {k: v for k, v in QWEN2_WINDOWED.items() if k != "use_sliding_window"}
    names, _, _ = stated_layer_types(unswitched)
    assert names == ["full_attention"] * 40


def test_the_shipped_layout_rules_reproduce_transformers_5_12():
    """Every rule the snapshot ships gives the class's own list at depths and windows the writer
    did not check it at."""
    transformers = pytest.importorskip("transformers")
    if not transformers.__version__.startswith("5.12"):
        pytest.skip("the snapshot is transformers 5.12's")
    from transformers import CONFIG_MAPPING

    from llm_memory_calculator.layer_plan import _class_default_snapshot, class_layout_names

    layouts = _class_default_snapshot()["layouts"]
    assert layouts.get("gemma2", {}).get("rule") == "period"
    assert layouts.get("mistral", {}).get("rule") == "none"
    assert layouts.get("qwen2", {}).get("rule") == "max_window_layers"
    wrong = []
    for mt, rule in sorted(layouts.items()):
        if rule["rule"] == "none":
            continue
        cls = CONFIG_MAPPING[mt]
        stated = cls().to_dict()
        switches = [{}]
        if "use_sliding_window" in stated:
            switches = [{"use_sliding_window": True}, {"use_sliding_window": False}, {}]
        for n in (7, 30, 61):
            for switch in switches:
                kwargs = {"num_hidden_layers": n, **switch}
                if "sliding_window" in stated:
                    kwargs["sliding_window"] = 2048
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    obj = cls(**kwargs)
                want = [str(x) for x in obj.layer_types]
                if class_layout_names(rule, {"num_hidden_layers": n, **switch}) != want:
                    wrong.append((mt, n, switch))
    assert not wrong, wrong


def test_a_head_width_the_class_derives_is_shipped_too(without_transformers):
    """ZambaConfig derives `attention_head_dim = 2 x hidden_size / heads` (its attention reads the
    hidden state concatenated with the embeddings); read from the installed class alone, a Zamba
    config that states no head width was sized at hidden / heads -- half its KV -- without
    transformers."""
    from llm_memory_calculator.layer_plan import attention_head_dim

    zamba = {"model_type": "zamba", "hidden_size": 3712, "num_attention_heads": 16}
    assert attention_head_dim(zamba) == 464
    zamba2 = {"model_type": "zamba2", "hidden_size": 2560, "num_attention_heads": 32}
    assert attention_head_dim(zamba2) == 160


def test_training_modeling_counts_routed_experts_at_their_own_width():
    """`training_parameter_counts` sized every routed expert at the dense `intermediate_size`:
    SmallThinker-21BA3B's 64 experts are 768 wide (`moe_intermediate_size`), and at the default
    11,008 its count was 283.5B of its 21.5B (13x), so training_modeling put 4.7 TB on one GPU.
    The experts are counted at their own width; the total is still held to the model's own."""
    import glob

    from llm_memory_calculator.genz.LLM_training.training_modeling import (
        training_parameter_counts,
    )
    from llm_memory_calculator.genz.Models import get_configs

    path = glob.glob(os.path.join(FIXTURES, "**", "PowerInfer_SmallThinker-21BA3B-Instruct.json"),
                     recursive=True)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        total, active = training_parameter_counts(get_configs(_fixture(os.path.relpath(path, FIXTURES))))
    published = 21_506_562_560  # the checkpoint's tensors (kv_gate truths.json)
    assert published <= total <= 1.1 * published
    assert active < total / 4  # 6 of 64 experts a token


#: transformers 5.12's parameters for the fixture (kv_truth.truth_profile)
MOE_PARAMS = {
    "qwen1.5-moe-a2.7b.json": 14_315_784_192,
    "Qwen_Qwen3-Next-80B-A3B-Instruct.json": 79_674_391_296,
}


@pytest.mark.parametrize("name", sorted(MOE_PARAMS))
def test_the_facts_count_a_shared_experts_gate_and_a_class_default_attention_bias(name):
    """Counting the experts at their own width brought training_modeling's parameters down to the
    model's own count (the facts'), which left out two things transformers builds: the gate that
    scales a shared expert's output (Linear(h, 1) a layer: 49,152 on Qwen1.5-MoE, 98,304 on
    Qwen3-Next) and Q / K / V biases a class builds by default (Qwen2MoeConfig's `qkv_bias`: True;
    147,456). Both are counted, so the facts, and every count held to them, reach the model."""
    import glob

    from llm_memory_calculator.genz.LLM_training.training_modeling import (
        training_parameter_counts,
    )
    from llm_memory_calculator.genz.Models import get_configs

    path = glob.glob(os.path.join(FIXTURES, "**", name), recursive=True)[0]
    cfg = _fixture(os.path.relpath(path, FIXTURES))
    assert _facts(cfg).params_total >= MOE_PARAMS[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        total, _ = training_parameter_counts(get_configs(copy.deepcopy(cfg)))
    assert total >= MOE_PARAMS[name]
