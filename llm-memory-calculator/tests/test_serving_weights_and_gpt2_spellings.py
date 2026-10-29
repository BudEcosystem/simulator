"""The fourteenth review: the serving path's weights, and a decoder spelled the GPT-2 way.

* ``genz.serving.memory_model.estimate_weight_bytes`` is the weight a serving simulation loads
  into the device tier (``ServingSimulator``) and what the cluster and disaggregation ``B_eff``
  take off the device before they size the KV. It counted ``num_decoder_layers`` x (attention +
  3 x hidden x intermediate) + one embedding from the GenZ ModelConfig: no encoder, no
  cross-attention, no tower, no experts. Once the ModelConfig carried an encoder-decoder's own
  decoder (round 7 on) the figure fell below the model for every encoder-decoder (Whisper
  large-v3 1.81 GB of its 3.09 GB) and every model the database converter reads under its
  `text_config` (Qwen3.5-4B 7.48 GB of 9.08), and the cluster admitted more requests than the
  device holds (Whisper large-v3 on a 40 GB A100 at 4k: 44, where the memory holds 43).
  It is now the model's own parameter count -- the FRD-027 facts' (every stack, tower, adapter
  and expert; what the vLLM planner reserves before it sizes the KV) -- of the config the
  ModelConfig was read from; a ModelConfig built by hand (GenZ's built-in dictionary) counts its
  routed experts on the layers it places them.
* A nested decoder spelled the GPT-2 way (`n_layer` / `n_embd` / `n_head`: a ViT-GPT2 captioner)
  was not read: no KV, and GenZ, the serving path and the prefix cache raised. Those are the
  decoder-only names under transformers' `attribute_map` (GPT2Config: `hidden_size` -> `n_embd`,
  `num_hidden_layers` -> `n_layer`, `num_attention_heads` -> `n_head`, `max_position_embeddings`
  -> `n_positions`), read now by every reader, nested or flat.

Truths are transformers 5.12's models built on the meta device (``scripts/kv_truth.py``).
"""

import contextlib
import copy
import io
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import kv_corpus  # noqa: E402

GB = 1024**3


def _config(name):
    if name.startswith("synth:"):
        return copy.deepcopy(kv_corpus.synthetic_configs(FIXTURES)[name[6:]])
    with open(os.path.join(FIXTURES, name)) as f:
        return json.load(f)


def _genz(cfg):
    from llm_memory_calculator.genz.Models.get_language_model import get_configs

    with contextlib.redirect_stdout(io.StringIO()):
        return get_configs(copy.deepcopy(cfg))


# transformers 5.12's parameter count of each config (meta device, `sum(p.numel())`)
MODEL_PARAMS = [
    ("planner/models/whisper-large-v3.json", 1_543_490_560),
    ("encoder_decoder/t5_flan-xl-geometry.json", 2_783_959_040),
    ("encoder_decoder/nari-labs_Dia-1.6B-0626.json", 1_611_160_576),
    ("cross_attention/mllama_transformers-5.12-default.json", 10_642_941_475),
    ("hybrid_configs/Qwen_Qwen3.5-4B.json", 4_539_265_536),
    ("synth:t5-asym", 441_932_800),
    ("synth:prophetnet", 391_321_600),
    ("synth:mt5-noflag", 172_119_424),
    ("synth:marian-noflag", 74_934_784),
    ("synth:nllb-moe", 54_500_569_088),
    ("synth:vision-enc-dec-gpt2", 182_485_248),
]


@pytest.mark.parametrize("name,params", MODEL_PARAMS)
def test_the_serving_weights_hold_the_whole_model(name, params):
    from llm_memory_calculator.genz.serving.memory_model import estimate_weight_bytes

    assert estimate_weight_bytes(_genz(_config(name)), "bf16", 1) >= 2 * params


@pytest.mark.parametrize("name,params", MODEL_PARAMS[:5])
def test_the_simulator_loads_the_same_weights(name, params):
    """`ServingSimulator._estimate_weight_bytes` is the weight load of a simulation."""
    from llm_memory_calculator.genz.serving.memory_model import estimate_weight_bytes
    from llm_memory_calculator.genz.serving.simulator import ServingSimulator

    cfg = _config(name)
    sim = ServingSimulator(model=copy.deepcopy(cfg), hardware="A100_80GB_GPU", precision="bf16")
    mc = _genz(cfg)
    assert sim._estimate_weight_bytes(mc) == estimate_weight_bytes(mc, "bf16", 1) >= 2 * params


def test_the_serving_weights_are_the_models_own_count():
    """One count: the FRD-027 facts' parameters of the config the ModelConfig was read from (the
    decoder, the encoder and cross-attention, the towers and adapters, every expert), at the
    serving precision, split by tensor parallelism."""
    from llm_memory_calculator.genz.serving.memory_model import estimate_weight_bytes
    from llm_memory_calculator.model.facts import ModelFacts

    cfg = _config("planner/models/whisper-large-v3.json")
    facts = ModelFacts.from_config(copy.deepcopy(cfg))
    params = facts.params_total + facts.vision_params
    mc = _genz(cfg)
    assert estimate_weight_bytes(mc, "bf16", 1) == 2 * params
    assert estimate_weight_bytes(mc, "fp8", 1) == params
    assert estimate_weight_bytes(mc, "bf16", 4) == 2 * params // 4


def test_an_audio_tower_beside_the_language_model_is_counted():
    """The towers a model holds beside its language model are its weights too: an audio encoder
    (`audio_config`) as a vision one (`vision_config`). Qwen2-Audio's Whisper-large encoder (32
    layers of 1,280) and its projector are 0.64B the facts left out."""
    from llm_memory_calculator.model.facts import ModelFacts

    cfg = _config("synth:qwen2-audio")
    no_tower = {k: v for k, v in cfg.items() if k != "audio_config"}
    with_tower = ModelFacts.from_config(copy.deepcopy(cfg))
    without = ModelFacts.from_config(no_tower)
    a = cfg["audio_config"]
    h, layers = a["d_model"], a["encoder_layers"]
    blocks = layers * (4 * h * h + 2 * h * a["encoder_ffn_dim"])  # matrices alone: a lower bound
    assert with_tower.vision_params - without.vision_params >= blocks


@pytest.mark.parametrize("name,params", [
    # Kimi Delta Attention: the checkpoint's tensors (moonshotai/Kimi-Linear-48B-A3B-Instruct)
    ("hybrid_configs/moonshotai_Kimi-Linear-48B-A3B-Instruct.json", 49_122_681_728),
    # Mamba-2: transformers 5.12's models of the hub configs
    ("hybrid_configs/ibm-ai-platform_Bamba-9B-v2.json", 9_780_235_392),
    ("hybrid_configs/tiiuae_Falcon-H1-34B-Instruct.json", 33_642_516_224),
    ("hybrid_configs/mistralai_Mamba-Codestral-7B-v0.1.json", 7_285_403_648),
])
def test_the_facts_count_each_recurrent_layer_by_its_own_mechanism(name, params):
    """A recurrent layer's weights are its mechanism's one count (``mixer_params``): the facts had
    counted every non-GDN layer with their own copy of a Mamba-2 formula (no convolution bias), so
    Kimi-Linear's KDA layers -- whose own count also lacked the per-head beta projection `b_proj`
    [32, 2304] -- sat 137.7M below the checkpoint and Mamba-Codestral 393,216 below transformers."""
    from llm_memory_calculator.model.facts import ModelFacts

    f = ModelFacts.from_config(_config(name))
    assert f.params_total + f.vision_params >= params


def test_the_cluster_admits_what_the_memory_holds():
    """B_eff = (capacity - weights) // (KV x context + fixed bytes). Whisper large-v3 on a 40 GB
    A100 at 4,096 tokens: (40 GiB - 3,086,981,120) // (163,840 x 4,096 + 245,760,000) = 43; the
    decoder-only weight figure (1.81 GB) admitted 44."""
    from llm_memory_calculator.genz.serving.cluster import ClusterAnalyzer
    from llm_memory_calculator.genz.serving.disaggregation import DisaggregationAnalyzer

    cfg = _config("planner/models/whisper-large-v3.json")
    for cls in (ClusterAnalyzer, DisaggregationAnalyzer):
        a = cls(model=copy.deepcopy(cfg), hardware="A100_40GB_GPU")
        with contextlib.redirect_stdout(io.StringIO()):
            b = a._build_b_eff(4096, max_batch_size=10**7)
        assert 1 <= b <= (40 * GB - 3_086_981_120) // (163_840 * 4096 + 245_760_000)


def test_a_built_in_moe_model_counts_its_experts():
    """A ModelConfig from GenZ's built-in dictionary has no config to count: its routed experts
    are counted on the layers it places them (Mixtral-8x7B: 32 MoE layers of 8 experts of
    3 x 4,096 x 14,336). The decoder formula counted one."""
    from llm_memory_calculator.genz.Models.get_language_model import get_configs
    from llm_memory_calculator.genz.serving.memory_model import estimate_weight_bytes

    mc = get_configs("mistralai/Mixtral-8x7B")
    experts = 32 * 8 * 3 * 4096 * 14336
    assert estimate_weight_bytes(mc, "bf16", 1) >= 2 * experts


@pytest.mark.parametrize("name,params", [
    # each checkpoint's tensors, read from its safetensors headers (scripts/kv_truth.py --checkpoint)
    ("state-spaces/mamba-130m-hf", 129_135_360),
    ("tiiuae/falcon-mamba-7b", 7_272_665_088),
    # transformers 5.12's MambaForCausalLM of the hub config
    ("state-spaces/mamba-2.8b-hf", 2_768_345_600),
])
def test_a_built_in_mamba_model_counts_its_mixer(name, params):
    """A built-in Mamba layer is a Mamba mixer (``mixer_params.mamba1_params`` over its
    ``mamba_*`` fields), not an attention layer: Falcon-Mamba 7B weighed 4.5B parameters. Its
    built-in vocabulary is the hub config's 65,024 (it said 50,280, state-spaces/mamba's)."""
    from llm_memory_calculator.genz.Models.get_language_model import get_configs
    from llm_memory_calculator.genz.serving.memory_model import estimate_weight_bytes

    assert estimate_weight_bytes(get_configs(name), "bf16", 1) >= 2 * params


def test_a_stand_in_model_config_is_counted_as_before():
    """An object with the decoder fields alone (no source config, no layer placement) keeps the
    decoder formula: Q / O at the heads, K / V at the KV heads, a gated FFN, one embedding."""
    from types import SimpleNamespace

    from llm_memory_calculator.genz.serving.memory_model import estimate_weight_bytes

    mc = SimpleNamespace(hidden_size=4096, num_decoder_layers=32, intermediate_size=11008,
                         vocab_size=32000, num_attention_heads=32, num_key_value_heads=8,
                         head_dim=128)
    params = 32 * (2 * 4096 * 32 * 128 + 2 * 4096 * 8 * 128 + 3 * 4096 * 11008) + 32000 * 4096
    assert estimate_weight_bytes(mc, "bf16", 1) == 2 * params


# ------------------------------------------------------------- a decoder spelled the GPT-2 way


def test_a_nested_gpt2_decoder_is_read():
    """ViT-GPT2's `decoder` block says `n_layer: 6`, `n_embd: 768`, `n_head: 12`: 6 layers of
    2 x 768 K + V a token, and a cross-attention cache over the ViT's 197 tokens (224 / 16 = 14,
    squared, and the class token) on each."""
    from llm_memory_calculator.layer_plan import cross_attention_cache, decoder_view

    cfg = _config("synth:vision-enc-dec-gpt2")
    view = decoder_view(cfg)
    assert (view["num_hidden_layers"], view["hidden_size"], view["num_attention_heads"]) == \
        (6, 768, 12)
    cross = cross_attention_cache(cfg)
    assert cross is not None and cross["layers"] == 6 and cross["tokens"] == 197


def test_every_reader_charges_a_nested_gpt2_decoder():
    from llm_memory_calculator.calculator import ModelMemoryCalculator
    from llm_memory_calculator.genz.serving.memory_model import MemoryModel
    from llm_memory_calculator.genz.serving.prefix_cache import PrefixCacheAnalyzer

    cfg = _config("synth:vision-enc-dec-gpt2")
    self_bytes = 6 * 2 * 768 * 2  # a token's K + V on 6 layers, 2 B
    cross_bytes = 6 * 2 * 768 * 197 * 2
    calc = ModelMemoryCalculator()
    kv = calc.calculate_kv_cache(copy.deepcopy(cfg), 1, 4096, "bf16") * 1e9
    assert kv == pytest.approx(self_bytes * 4096 + cross_bytes, rel=1e-6)
    mc = _genz(cfg)
    assert mc.get_kv_size() * 2 == self_bytes
    mm = MemoryModel(mc, [], tensor_parallel=1, precision_bytes=2)
    assert mm.bytes_per_token_kv == self_bytes
    assert mm.state_bytes_per_sequence == cross_bytes
    with contextlib.redirect_stdout(io.StringIO()):
        assert PrefixCacheAnalyzer(mc)._kv_bytes_per_token == self_bytes


def test_a_flat_gpt2_config_is_read_by_every_reader():
    """A flat GPT-2 config (`n_layer: 12`, `n_embd: 768`, `n_head: 12`): GenZ graphed the
    generic 32 x 4,096 default, and the KV planner charged 0 bytes a token."""
    from llm_memory_calculator.kv.geometry import KVGeometry

    cfg = _config("synth:gpt2")
    per_token = 12 * 2 * 768 * 2
    mc = _genz(cfg)
    assert (mc.num_decoder_layers, mc.hidden_size) == (12, 768)
    assert mc.get_kv_size() * 2 == per_token
    g = KVGeometry.from_model(copy.deepcopy(cfg), seq_length=4096)
    assert g.bytes_per_token_total == per_token


def test_the_aliases_are_transformers_own():
    """The decoder-only names a GPT-2-family config class aliases (`attribute_map`) are each read."""
    transformers = pytest.importorskip("transformers")
    from llm_memory_calculator.layer_plan import DECODER_NAME_ALIASES

    read = {(name, alias) for name, aliases in DECODER_NAME_ALIASES.items() for alias in aliases}
    for model_type in ("gpt2", "gptj", "codegen", "gpt_bigcode", "ctrl", "openai-gpt", "bloom"):
        cls = transformers.CONFIG_MAPPING.get(model_type)
        for name, alias in getattr(cls, "attribute_map", {}).items():
            if name in DECODER_NAME_ALIASES:
                assert (name, alias) in read, (model_type, name, alias)
