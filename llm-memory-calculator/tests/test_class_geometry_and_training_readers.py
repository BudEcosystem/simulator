"""What a config leaves to its config class is the class's, in every reader; and the training
readers count the whole model.

A hub config's sub-config often states only its model type (BLIP-2's and InstructBLIP's
`vision_config` and `qformer_config`), or leaves the depth, width and heads of its language model
to the class (Qwen2-Audio's and LLaVA-1.5's `text_config`). transformers builds the class's
defaults there; the readers read nothing, so a 985.9M-parameter tower and a 105.1M Q-Former were
counted as 0 and Qwen2-Audio's language model had 0 layers holding KV (only with transformers
5.12 installed). ``layer_plan.with_class_geometry`` fills each role a block leaves out from the
shipped config-class snapshot, the class the composite builds the block as.

The training readers -- ``UniversalParameterCounter.count_parameters`` (TrainingMemoryCalculator,
AdvancedTrainingCalculator, BudSimulator's training and use-case routes) and GenZ's
``training_modeling`` -- counted an encoder-decoder as its decoder alone (t5-small 51.8M of
60.5M, Whisper large-v3 971.8M of 1,543.5M). And a config class that ties its head whatever the
config says (transformers 5.12's T5 family) does not say what its checkpoints hold: an omitted
flag is read untied, and the ground-truth tool counts the head a checkpoint holds apart.

Truths: transformers 5.12's model for the config as the hub ships it, built on the meta device,
with the LM head a T5-family checkpoint holds apart (scripts/kv_truth.py), or the checkpoint's
own safetensors headers (google/flan-t5-xl: 2,849,757,184; Salesforce/blip2-flan-t5-xl:
3,942,446,592).
"""

import copy
import json
import os
import sys

import pytest

from llm_memory_calculator import layer_plan

HERE = os.path.dirname(os.path.abspath(__file__))
HUB = os.path.join(HERE, "fixtures", "hub_as_shipped")
FIXTURES = os.path.join(HERE, "fixtures")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))


def _hub(name):
    with open(os.path.join(HUB, name + ".json")) as f:
        return json.load(f)


def _synth(name):
    import kv_corpus

    return kv_corpus.synthetic_configs(FIXTURES)[name]


@pytest.fixture(params=["installed", "absent"])
def transformers_state(request, monkeypatch):
    """Each figure with transformers as installed and with it absent: the same either way."""
    if request.param == "absent":
        monkeypatch.setitem(sys.modules, "transformers", None)
    for cache in ("_CLASS_DEFAULT_CACHE", "_CLASS_GEOMETRY_CACHE", "_CLASS_LAYOUT_CACHE"):
        getattr(layer_plan, cache, {}).clear()
    yield request.param
    for cache in ("_CLASS_DEFAULT_CACHE", "_CLASS_GEOMETRY_CACHE", "_CLASS_LAYOUT_CACHE"):
        getattr(layer_plan, cache, {}).clear()


def _facts(cfg):
    from llm_memory_calculator.model.facts import ModelFacts

    return ModelFacts.from_config(copy.deepcopy(cfg))


# --- a block that leaves its geometry to its class -------------------------------------------

def test_a_sub_config_that_states_only_its_model_type_is_its_class():
    cfg = layer_plan.with_class_geometry(_hub("Salesforce_blip2-opt-2.7b"))
    vision, qformer = cfg["vision_config"], cfg["qformer_config"]
    assert (vision["num_hidden_layers"], vision["hidden_size"], vision["num_attention_heads"],
            vision["intermediate_size"]) == (39, 1408, 16, 6144)
    assert (qformer["num_hidden_layers"], qformer["hidden_size"],
            qformer["cross_attention_frequency"], qformer["encoder_hidden_size"]) == (12, 768, 2, 1408)


def test_the_class_is_the_one_the_composite_builds_not_the_name_the_block_gives():
    """Idefics2's hub `perceiver_config` names `idefics2`; Idefics2Config builds it as an
    Idefics2PerceiverConfig (3 layers, 64 latents)."""
    cfg = layer_plan.with_class_geometry(_hub("HuggingFaceM4_idefics2-8b"))
    assert cfg["perceiver_config"]["resampler_depth"] == 3
    assert cfg["perceiver_config"]["resampler_n_latents"] == 64
    assert cfg["text_config"]["num_hidden_layers"] == 32  # MistralConfig's


def test_a_role_the_block_states_is_kept_whole():
    """T5 derives `num_decoder_layers` from `num_layers`: a class default must not stand in."""
    t5 = {"model_type": "t5", "num_layers": 24, "d_model": 2048, "num_heads": 32, "d_kv": 64,
          "d_ff": 5120, "vocab_size": 32128}
    assert layer_plan.with_class_geometry(t5) is t5
    qwen = {"model_type": "qwen2", "hidden_size": 896, "num_attention_heads": 14,
            "num_hidden_layers": 24, "intermediate_size": 4864}
    assert "num_key_value_heads" not in layer_plan.with_class_geometry(qwen)


def test_the_heads_and_ffn_are_the_class_s_only_where_its_width_is():
    """FalconConfig sizes `ffn_hidden_size` from its own 4,544 width: beside Falcon-40B's stated
    8,192 the class's 18,176 is another model's, and the counter took 26.96B for 41.3B."""
    from llm_memory_calculator.parameter_counter import UniversalParameterCounter

    falcon = _synth("falcon-40b")
    assert layer_plan.with_class_geometry(falcon) is falcon
    assert UniversalParameterCounter().count_parameters(falcon) >= 41_303_293_952


def test_a_config_is_never_mutated():
    cfg = _hub("Salesforce_blip2-opt-2.7b")
    before = json.dumps(cfg, sort_keys=True)
    layer_plan.with_class_geometry(cfg)
    _facts(cfg)
    assert json.dumps(cfg, sort_keys=True) == before


@pytest.mark.parametrize("name,truth", [
    ("Salesforce_blip2-opt-2.7b", 3_744_761_856),
    ("Salesforce_blip2-flan-t5-xl", 3_942_446_592),  # the checkpoint's tensors
    ("Salesforce_instructblip-flan-t5-xl", 4_022_969_088),
    ("microsoft_Florence-2-base-ft", 231_414_016),
    ("Qwen_Qwen2-Audio-7B-Instruct", 8_397_094_912),
    ("llava-hf_llava-1.5-7b-hf", 7_063_427_072),
    ("HuggingFaceM4_idefics2-8b", 8_402_768_112),
])
def test_the_facts_weights_hold_the_whole_hub_model(transformers_state, name, truth):
    f = _facts(_hub(name))
    assert f.params_total + f.vision_params >= truth


def test_the_serving_path_weights_hold_the_hub_blip2_tower():
    from llm_memory_calculator.genz.Models.get_language_model import get_configs
    from llm_memory_calculator.genz.serving.memory_model import model_parameter_count

    assert model_parameter_count(get_configs(_hub("Salesforce_blip2-flan-t5-xl"))) >= 3_942_446_592


# --- a depth left to the class ----------------------------------------------------------------

def test_a_window_with_the_depth_left_to_the_class_charges_every_layer(transformers_state):
    """Qwen2Config's 32 layers, every one full attention (`use_sliding_window` is False): the
    same with transformers 5.12, an older release or none."""
    f = _facts(_synth("qwen2-window-depth-left-to-class"))
    assert (f.layers.full, f.layers.sliding) == (32, 0)
    assert f.kv_bytes_per_token() == 524_288
    assert layer_plan.depth(_synth("qwen2-window-depth-left-to-class")) == 32


def test_qwen2_audio_as_shipped_has_its_32_kv_layers(transformers_state):
    f = _facts(_hub("Qwen_Qwen2-Audio-7B-Instruct"))
    assert f.layers.full == 32
    assert f.kv_bytes_for_context(4096) == 2_147_483_648


def test_the_vllm_planner_charges_qwen2_audio_its_kv():
    from llm_memory_calculator.device import catalogue
    from llm_memory_calculator.engine import EngineRecord, engine_for
    from llm_memory_calculator.engine.base import Workload

    import kv_corpus

    with open(kv_corpus.PLANNER_RECORD) as fh:
        rec = json.load(fh)
    eng, dev = engine_for(EngineRecord.from_dict(rec)), catalogue.get(kv_corpus.PLANNER_DEVICE)
    f = _facts(_hub("Qwen_Qwen2-Audio-7B-Instruct"))
    running = eng.predict({}, Workload(concurrency=1024, input_tokens=1024, output_tokens=256),
                          f, dev).running
    assert running < 1024


# --- the training readers ---------------------------------------------------------------------

@pytest.mark.parametrize("name,truth", [
    ("google-t5_t5-small", 60_506_624),
    ("google_mt5-small", 300_176_768),
    ("openai_whisper-small", 241_734_912),
    ("Salesforce_blip2-flan-t5-xl", 3_942_446_592),
    ("Qwen_Qwen2-Audio-7B-Instruct", 8_397_094_912),
    ("llava-hf_llava-1.5-7b-hf", 7_063_427_072),
    ("microsoft_Florence-2-base-ft", 231_414_016),
])
def test_the_parameter_counter_holds_the_whole_model(name, truth):
    from llm_memory_calculator.parameter_counter import UniversalParameterCounter

    assert UniversalParameterCounter().count_parameters(_hub(name)) >= truth


def test_the_parameter_counter_holds_a_flat_flan_t5_to_its_checkpoint():
    from llm_memory_calculator.parameter_counter import UniversalParameterCounter

    with open(os.path.join(FIXTURES, "encoder_decoder", "t5_flan-xl-geometry.json")) as f:
        cfg = json.load(f)
    assert UniversalParameterCounter().count_parameters(cfg) >= 2_849_757_184


def test_the_training_memory_calculator_sizes_the_whole_model():
    from llm_memory_calculator.training.calculator import TrainingMemoryCalculator

    t = TrainingMemoryCalculator().calculate_training_memory(
        _hub("google-t5_t5-small"), batch_size=1, seq_length=512, precision="bf16",
        method="full", optimizer="adamw")
    # bf16 weights and their fp32 master copy: 6 B a parameter
    assert t.weight_memory_gb >= 6 * 60_506_624 / 1e9


@pytest.mark.parametrize("config,truth", [
    (lambda: _synth("whisper-large-v3"), 1_543_490_560),
    (lambda: _synth("t5-asym"), 441_932_800),
    (lambda: _synth("prophetnet"), 391_321_600),
    (lambda: _hub("Salesforce_blip2-flan-t5-xl"), 3_942_446_592),
])
def test_genz_training_sizes_the_whole_model(config, truth):
    from llm_memory_calculator.genz.LLM_training.training_modeling import training_parameter_counts
    from llm_memory_calculator.genz.Models.get_language_model import get_configs

    total, active = training_parameter_counts(get_configs(config()))
    assert total >= truth
    assert active <= total


def test_genz_training_keeps_a_decoders_own_count_where_it_is_the_larger():
    """A GQA decoder's fields, read as four full attention matrices, count more than the model:
    that figure stands (no training figure is lowered)."""
    from llm_memory_calculator.genz.LLM_training.training_modeling import (
        _calculate_total_params, training_parameter_counts)
    from llm_memory_calculator.genz.Models.get_language_model import get_configs

    with open(os.path.join(FIXTURES, "qwen2_5_0_5b_config.json")) as f:
        mc = get_configs(json.load(f))
    fields = _calculate_total_params(mc.num_decoder_layers, mc.hidden_size, mc.intermediate_size,
                                     mc.vocab_size, mc.num_attention_heads)
    assert training_parameter_counts(mc) == fields


def test_genz_training_modeling_reads_the_counts():
    """training_modeling sizes its weights from training_parameter_counts (the function the
    corpus harness measures), not a copy of the arithmetic."""
    import importlib
    import inspect

    tm = importlib.import_module("llm_memory_calculator.genz.LLM_training.training_modeling")

    assert "training_parameter_counts(model_config)" in inspect.getsource(tm.training_modeling)


# --- a class that ties whatever the config says ------------------------------------------------

def test_a_class_that_forces_tying_is_read_untied_when_the_flag_is_left_out():
    snap = layer_plan._class_default_snapshot()
    assert {"t5", "mt5", "umt5"} <= set(snap["tie_forced"])
    layer_plan._CLASS_DEFAULT_CACHE.clear()
    assert layer_plan.config_class_default("mt5", "tie_word_embeddings") is False


def test_an_mt5_that_leaves_the_flag_out_counts_its_head():
    from llm_memory_calculator.parameter_counter import UniversalParameterCounter

    cfg = _hub("google_mt5-small")
    cfg.pop("tie_word_embeddings")
    # the checkpoint (google/mt5-small) holds lm_head apart: 300,176,768
    assert UniversalParameterCounter().count_parameters(cfg) >= 300_176_768


def test_the_truth_tool_counts_the_head_a_forced_tie_hides():
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not str(transformers.__version__).startswith("5.12"):
        pytest.skip("the forced tie is transformers 5.12's")
    import kv_truth

    with open(os.path.join(FIXTURES, "encoder_decoder", "t5_flan-xl-geometry.json")) as f:
        cfg = json.load(f)
    cfg["tie_word_embeddings"] = False
    p = kv_truth.truth_profile(cfg)
    assert p["untied_head"] == 32128 * 2048
    assert p["params"] == 2_849_757_184  # google/flan-t5-xl's safetensors


def test_the_corpus_meets_the_hub_shapes():
    """The committed corpus holds the hub configs as shipped, and the filter keeps a
    `text_config` that leaves its whole geometry to its class."""
    import kv_corpus

    for name in os.listdir(HUB):
        with open(os.path.join(HUB, name)) as f:
            assert kv_corpus._is_model_config(json.load(f)), name
    assert kv_corpus._is_model_config(_synth("qwen2-window-depth-left-to-class"))
    for reader in ("counter_params", "trainmem_weight_gb", "traingenz_params"):
        assert reader in kv_corpus.measure_training(_hub("google-t5_t5-small"))


def test_an_mt5_that_leaves_its_flags_to_the_class_is_a_whole_encoder_decoder():
    """synth:mt5-noflag states no `is_encoder_decoder`, `feed_forward_proj` or tying: MT5Config's
    True, `gated-gelu` and (its checkpoints') untied head. The facts counted no encoder and no
    cross-attention (549,995,520 bytes below the 600,353,536 of its 300,176,768 parameters), and
    the encoder's FFN at two matrices."""
    from llm_memory_calculator.parameter_counter import UniversalParameterCounter

    cfg = _synth("mt5-noflag")
    assert layer_plan.stated_gated_ffn(cfg) is True
    f = _facts(cfg)
    assert f.params_total >= 300_176_768
    assert UniversalParameterCounter().count_parameters(cfg) >= 300_176_768


def test_a_tower_that_leaves_its_tiles_to_the_class_is_charged_the_class_s_tiles():
    """MllamaVisionConfig builds 4 tiles of 448 / 14 patches: a tower that states only its width
    and depth holds a cross-attention cache over 4 x 1,025 tokens an image, as transformers
    builds it."""
    cross = layer_plan.cross_attention_cache(_synth("mllama-own-cross-layers-no-tower-geometry"))
    assert cross["tokens"] == 4 * 1025



def test_a_tower_that_is_the_encoder_is_counted_once():
    """Pix2Struct's `vision_config` is its encoder: the facts counted it as the encoder and again
    as a tower beside the model (362.8M of transformers' 282,285,696), and left out its patch
    projection and row / column positions from the encoder."""
    with open(os.path.join(FIXTURES, "cross_attention", "pix2struct_transformers-5.12-default.json")) as fh:
        cfg = json.load(fh)
    assert layer_plan.encoder_is_tower(cfg)
    f = _facts(cfg)
    assert 282_285_696 <= f.params_total + f.vision_params <= 1.01 * 282_285_696
