"""Round 16: the training estimators that sized a model from GenZ's decoder fields, and the LoRA
buffers of a hybrid's recurrent layers.

* ``get_configs`` reads an encoder-decoder's real decoder geometry, so every estimator that counted
  parameters from the ModelConfig's decoder fields -- GenZ's training parallelization
  (``_estimate_memory_per_gpu``, behind ``get_training_parallelization_options``,
  ``get_best_training_parallelization``, the scale optimizer and the cluster ranker), the training
  cluster requirements, the comprehensive config builder and the cluster optimizer's quick checks
  -- left out the encoder and the cross-attention. Each now holds its count to the model's own
  (``memory_model.held_to_the_model``). Truths: transformers 5.12's parameters for the config.
* ``calculate_lora_adapter_memory`` counted adapter buffers on the attention layers alone; vLLM
  wraps every linear layer ("In vLLM, all linear layers support LoRA", lora/utils.py), the
  recurrent and linear-attention layers' maps among them. Floors: rank x 4 adapters x 2 B x the
  sum of (in + out) over every ``nn.Linear`` of the decoder layers transformers 5.12 builds for
  the fixture, its routed experts left out.
"""

import copy
import json
import os
import sys
import warnings

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")
HYBRID = os.path.join(FIXTURES, "hybrid_configs")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))


def _fixture(rel):
    with open(os.path.join(FIXTURES, rel)) as f:
        return json.load(f)


def _synthetic(name):
    import kv_corpus

    return copy.deepcopy(kv_corpus.synthetic_configs(FIXTURES)[name])


#: (config, transformers 5.12's parameters for it)
ENCODER_DECODERS = {
    "whisper-large-v3": (lambda: _fixture("planner/models/whisper-large-v3.json"), 1_543_490_560),
    "whisper-small": (lambda: _fixture("hub_as_shipped/openai_whisper-small.json"), 241_734_912),
    "mt5-small": (lambda: _fixture("hub_as_shipped/google_mt5-small.json"), 300_176_768),
    "whisper-large-v3-turbo": (
        lambda: _fixture("encoder_decoder/onnx-community_whisper-large-v3-turbo.json"),
        808_878_080),
    "flan-t5-xl-geometry": (lambda: _fixture("encoder_decoder/t5_flan-xl-geometry.json"),
                            2_783_959_040),
    "t5-asym": (lambda: _synthetic("t5-asym"), 474_831_872),
    "prophetnet": (lambda: _synthetic("prophetnet"), 391_321_600),
    "nllb-moe": (lambda: _synthetic("nllb-moe"), 54_500_569_088),
}


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.mark.parametrize("name", sorted(ENCODER_DECODERS))
def test_training_parallelization_sizes_the_whole_model(name):
    """Weights (bf16, 2 B), fp32 gradients (4 B) and AdamW's two moments (8 B), with the
    estimator's own 10% overhead, before any activation: whisper-large-v3 16.49 GB on round 15,
    its floor 23.77 GB."""
    from llm_memory_calculator.genz.LLM_training import training_parallelization as tp
    from llm_memory_calculator.genz.Models.get_language_model import get_configs

    cfg, params = ENCODER_DECODERS[name]
    mc = get_configs(cfg())
    gb = tp._estimate_memory_per_gpu(mc, tp.TrainingParallelismConfig(), 1, 2048, "full",
                                     "adamw", "bf16")
    assert gb >= 14 * params * 1.1 / 1e9, (gb, 14 * params * 1.1 / 1e9)


def test_a_model_whose_optimizer_state_alone_overflows_the_gpu_gets_no_option():
    """whisper-large-v3's weights, gradients and Adam state alone are 21.6 GB: no single 20 GB
    GPU holds it (round 15 returned a TP1 / PP1 / DP1 configuration)."""
    from llm_memory_calculator.genz.LLM_training.training_parallelization import (
        get_training_parallelization_options,
    )

    cfg, _ = ENCODER_DECODERS["whisper-large-v3"]
    assert get_training_parallelization_options(
        cfg(), total_gpus=1, gpu_memory_gb=20, batch_size=1, seq_length=2048, method="full",
        optimizer="adamw", bits="bf16") == []


@pytest.mark.parametrize("name", sorted(ENCODER_DECODERS))
def test_cluster_requirements_count_the_whole_model(name):
    from llm_memory_calculator.training.cluster_requirements import _estimate_model_params

    cfg, params = ENCODER_DECODERS[name]
    assert _estimate_model_params(cfg())["total_params"] >= params


@pytest.mark.parametrize("name", sorted(ENCODER_DECODERS))
def test_the_config_builder_counts_the_whole_model(name):
    from llm_memory_calculator.training.comprehensive_config_builder import _estimate_model_size

    cfg, params = ENCODER_DECODERS[name]
    assert _estimate_model_size(cfg()) * 1e9 >= params * (1 - 1e-12)


@pytest.mark.parametrize("name", sorted(ENCODER_DECODERS))
def test_the_cluster_optimizer_quick_checks_count_the_whole_model(name):
    """The quick feasibility check and the GPU-type filter, which reject a cluster before any
    simulation."""
    from llm_memory_calculator.genz.LLM_training.training_parallelization import (
        TrainingParallelismConfig,
    )
    from llm_memory_calculator.training.cluster_optimizer import ClusterOptimizer
    from llm_memory_calculator.training.cluster_optimizer_types import TrainingJobSpec

    cfg, params = ENCODER_DECODERS[name]
    job = TrainingJobSpec(model=cfg(), batch_size=1, avg_sequence_length=2048)
    counted, gb = ClusterOptimizer._quick_memory_estimate_gb(job, TrainingParallelismConfig())
    assert counted >= params
    assert gb >= 14 * params / 1e9
    assert ClusterOptimizer._min_memory_per_gpu_gb(job, 1) >= 2 * params / 1e9


def test_a_dense_decoder_keeps_its_fields_count():
    """Holding to the model never lowers a figure: a decoder-only model whose fields count at
    least the model keeps the fields' figure."""
    from llm_memory_calculator.genz.Models.get_language_model import get_configs
    from llm_memory_calculator.genz.serving.memory_model import (
        held_to_the_model,
        model_parameter_count,
    )

    cfg = _fixture("qwen2_5_0_5b_config.json")
    mc = get_configs(cfg)
    counted = model_parameter_count(mc)
    assert held_to_the_model(mc, counted + 7) == counted + 7
    assert held_to_the_model(mc, 1) == counted


# --------------------------------------------------------------------------------- LoRA


def _lora_gb(cfg):
    from llm_memory_calculator import calculate_memory

    rep = calculate_memory(copy.deepcopy(cfg), batch_size=1, seq_length=4096, precision="bf16",
                           max_loras=4, max_lora_rank=16)
    return rep.lora_adapter_memory_bytes / 1e9


#: sum of (in + out) over every nn.Linear of the decoder layers transformers 5.12 builds (routed
#: experts left out): vLLM's floor is 4 adapters x rank 16 x 2 B x this
LORA_LINEAR_FEATURES = {
    "ai21labs_Jamba-v0.1.json": 2_448_512,
    "ibm-ai-platform_Bamba-9B-v2.json": 2_810_752,
    "ibm-granite_granite-4.0-h-small.json": 1_978_688,
    "Qwen_Qwen3-Next-80B-A3B-Instruct.json": 1_538_352,
    "Qwen_Qwen3.5-4B.json": 2_029_056,
    "Qwen_Qwen3.6-27B.json": 7_295_488,
    "nvidia_NVIDIA-Nemotron-Nano-9B-v2.json": 2_258_944,
    "nvidia_Nemotron-H-8B-Base-8K.json": 2_173_952,
    "tiiuae_Falcon-H1-34B-Instruct.json": 9_365_760,
    "state-spaces_mamba-2.8b-hf.json": 1_988_608,
    "mistralai_Mamba-Codestral-7B-v0.1.json": 2_236_416,
    "EleutherAI_gpt-neo-1.3B.json": 884_736,
    "tencent_Hunyuan-A13B-Instruct.json": 1_673_216,
}


@pytest.mark.parametrize("name", sorted(LORA_LINEAR_FEATURES))
def test_lora_buffers_cover_every_layers_linear_maps(name):
    """Jamba 0.243 GB on round 15 (its 4 attention layers), floor 0.313; Bamba 0.239, floor 0.360;
    GPT-Neo and Hunyuan-A13B raised (a null FFN width, a per-layer list of expert widths)."""
    with open(os.path.join(HYBRID, name)) as f:
        cfg = json.load(f)
    floor = 4 * 16 * 2 * LORA_LINEAR_FEATURES[name] / 1e9
    assert _lora_gb(cfg) >= floor * (1 - 1e-9), (_lora_gb(cfg), floor)


#: origin/main's figure, which counted a share of the layers as attention
ORIGIN_MAIN_LORA_GB = {
    "ai21labs_Jamba-v0.1.json": 0.2936,
    "ibm-ai-platform_Bamba-9B-v2.json": 0.2936,
    "nvidia_NVIDIA-Nemotron-Nano-9B-v2.json": 0.4175,
    "nvidia_Nemotron-H-8B-Base-8K.json": 0.4498,
    "MiniMaxAI_MiniMax-Text-01.json": 0.7235,
    "Qwen_Qwen3-Next-80B-A3B-Instruct.json": 0.1825,
    "Zyphra_Zamba2-2.7B.json": 0.2477,
    "moonshotai_Kimi-Linear-48B-A3B-Instruct.json": 0.1501,
}


@pytest.mark.parametrize("name", sorted(ORIGIN_MAIN_LORA_GB))
def test_no_hybrid_holds_fewer_lora_buffers_than_origin_main(name):
    with open(os.path.join(HYBRID, name)) as f:
        cfg = json.load(f)
    assert _lora_gb(cfg) >= ORIGIN_MAIN_LORA_GB[name]


def test_a_recurrent_layer_adds_its_own_maps():
    """Jamba: 4 attention layers of Q / K / V / O, and 28 Mamba layers of in_proj, conv, x_proj,
    dt_proj and out_proj (``mixer_params.mamba1_linears``)."""
    from llm_memory_calculator.mixer_params import mamba1_linears

    with open(os.path.join(HYBRID, "ai21labs_Jamba-v0.1.json")) as f:
        cfg = json.load(f)
    h = cfg["hidden_size"]
    attn = 4 * 4 * 16 * 4 * 2 * h * 2  # 4 layers x 4 maps x loras x rank x (in + out) x 2 B
    mamba = 28 * 4 * 16 * sum(i + o for i, o in mamba1_linears(cfg, h)) * 2
    ffn_layers, inter = 32, cfg["intermediate_size"]
    ffn = ffn_layers * 4 * 16 * 3 * (h + inter) * 2
    assert _lora_gb(cfg) * 1e9 == pytest.approx(attn + mamba + ffn, rel=1e-9)


def test_a_dense_model_s_lora_buffers_do_not_move():
    """A model without a layer plan keeps its four attention maps and FFN on every layer."""
    cfg = _fixture("qwen2_5_0_5b_config.json")
    h, inter, n = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"]
    expected = n * 4 * 16 * (4 * 2 * h + 3 * (h + inter)) * 2
    assert _lora_gb(cfg) * 1e9 == pytest.approx(expected, rel=1e-9)


def test_the_widest_lora_slice_includes_a_recurrent_map():
    """The prefill scratch is sized from the widest wrapped slice: NVIDIA-Nemotron-Nano-9B-v2's
    Mamba-2 in_proj (2 x 10,240 + 2 x 8 x 128 + 128 = 22,656) is wider than its 15,680-wide FFN,
    granite-4.0-h-small's (16,768) than its 768-wide shared FFN, Jamba's Mamba-1 in_proj (16,384)
    than its 14,336."""
    from llm_memory_calculator.calculator import ModelMemoryCalculator
    from llm_memory_calculator.layer_plan import resolve_layer_plan
    from llm_memory_calculator.mixer_params import recurrent_mixer_linears

    for name in ("nvidia_NVIDIA-Nemotron-Nano-9B-v2.json", "ibm-granite_granite-4.0-h-small.json",
                 "ai21labs_Jamba-v0.1.json", "ibm-ai-platform_Bamba-9B-v2.json"):
        with open(os.path.join(HYBRID, name)) as f:
            cfg = json.load(f)
        widest = max(o for _, o in recurrent_mixer_linears(
            cfg, resolve_layer_plan(cfg), cfg["hidden_size"]))
        assert widest > cfg.get("intermediate_size", 0), name
        assert ModelMemoryCalculator._lora_largest_slice(cfg) >= widest, name


@pytest.mark.parametrize("mech", ["gdn", "kda", "mamba2", "mamba1", "shortconv", "lightning",
                                  "rwkv", "mlstm", "rglru"])
def test_a_mechanisms_parameters_are_its_maps_and_its_vectors(mech):
    """One source for a mixer's shapes: its parameter count is its linear maps' weights plus
    the vectors no map holds, so the counter and the LoRA buffers read the same shapes."""
    import llm_memory_calculator.mixer_params as mp

    cfg = {"hidden_size": 4096, "linear_num_key_heads": 16, "linear_num_value_heads": 32,
           "mamba_d_state": 128, "mamba_n_groups": 8, "mamba_n_heads": 128, "mamba_d_head": 64,
           "num_attention_heads": 32, "num_heads": 8, "lru_width": 4096,
           "linear_attn_config": {"num_heads": 32, "head_dim": 128}}
    linears = getattr(mp, mech + "_linears")(cfg, 4096)
    weights = sum(i * o for i, o in linears)
    params = getattr(mp, mech + "_params")(cfg, 4096)
    assert weights <= params < 2 * weights


# ------------------------------------------------------------------- ChatGLM's KV heads


def _glm4_db_row():
    import sqlite3

    db = os.path.join(os.path.dirname(os.path.dirname(HERE)), "BudSimulator", "data",
                      "prepopulated.db")
    row = sqlite3.connect(db).execute(
        "select config_json from models where model_id = 'THUDM/glm-4-9b-chat'").fetchone()
    if row is None:
        pytest.skip("prepopulated.db holds no glm-4-9b-chat row")
    return json.loads(row[0])


@pytest.mark.parametrize("which,per_token", [("glm4", 40 * 2 * 2 * 128 * 2),
                                             ("chatglm3", 28 * 2 * 2 * 128 * 2)])
def test_chatglm_kv_heads_are_its_query_groups(which, per_token):
    """ChatGLM states its KV heads as `multi_query_group_num` beside `multi_query_attention`.
    Unread, ChatGLM3-6B was charged its 32 query heads (458,752 B a token, 16 x its 28,672), and
    prepopulated.db's glm-4-9b-chat, whose row stores `num_key_value_heads: true`, one head
    (20,480 B, half its 40,960) -- in every reader. Found by the KV tier planner's working set
    (round 16)."""
    from llm_memory_calculator.calculator import ModelMemoryCalculator
    from llm_memory_calculator.genz.Models.get_language_model import get_configs
    from llm_memory_calculator.genz.serving.memory_model import MemoryModel
    from llm_memory_calculator.kv.geometry import KVGeometry
    from llm_memory_calculator.model.facts import ModelFacts

    cfg = _glm4_db_row() if which == "glm4" else _synthetic("chatglm3")
    calc = ModelMemoryCalculator()
    calc.model_type = calc.detect_model_type(cfg)
    assert calc.calculate_kv_cache(copy.deepcopy(cfg), 1, 4096, "bf16") * 1e9 == pytest.approx(
        per_token * 4096)
    assert ModelFacts.from_config(copy.deepcopy(cfg)).kv_bytes_per_token("bf16") == per_token
    assert KVGeometry.from_model(copy.deepcopy(cfg), seq_length=4096).bytes_per_token_total \
        == pytest.approx(per_token)
    mm = MemoryModel(get_configs(copy.deepcopy(cfg)), [], tensor_parallel=1, precision_bytes=2)
    assert mm.bytes_per_token_kv == per_token
