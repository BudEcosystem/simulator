"""Round 19: every public figure path measured, and what measuring them found.

* A config that states the key its class builds the layout from but no window (LFM2's
  `full_attn_idxs`) was charged full attention on every layer: the class's own layout is read with
  or without a window (``layer_plan.class_layout``), and every reader takes it from the layer plan
  -- the short-convolution layers hold their conv state (it was 0 even with a window) and no KV.
* The ONNX Runtime engine's activation read the depth from `num_hidden_layers` / `n_layers` and the
  width from `hidden_size` / `d_model` alone: ChatGLM's `num_layers` was a 12-layer default and
  GPT-2's `n_embd` a 768-wide one. It reads them as every other reader does (``layer_plan``).
* The ONNX-GenAI reader (``bud_component.onnx_dir_to_hf_config``) is measured on folders that
  ship with the tests (tests/fixtures/onnx), as the GGUF reader is.
"""

import copy
import json
import os
import sys
import warnings

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))


def _synth(name):
    import kv_corpus

    return copy.deepcopy(kv_corpus.synthetic_configs(FIXTURES)[name])


@pytest.fixture
def with_transformers_5_12(monkeypatch):
    transformers = pytest.importorskip("transformers")
    if not transformers.__version__.startswith("5.12"):
        pytest.skip("the class layouts asserted are transformers 5.12's")
    from llm_memory_calculator import layer_plan

    monkeypatch.setattr(layer_plan, "_CLASS_LAYOUT_CACHE", {})
    yield


@pytest.fixture
def without_transformers(monkeypatch):
    from llm_memory_calculator import layer_plan

    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setattr(layer_plan, "_CLASS_LAYOUT_CACHE", {})
    yield


# ------------------------------------------------------ LFM2's `full_attn_idxs`, with no window

#: Lfm2Config(full_attn_idxs=[3, 7, ..., 23]) over 24 layers: 6 attention layers, 18 short
#: convolutions whose cache is (hidden_size, conv_L_cache) each.
LFM2_ATTENTION, LFM2_CONV, LFM2_WIDTH, LFM2_L = 6, 18, 2048, 3


@pytest.mark.parametrize("name", ["lfm2-full-attn-idxs", "lfm2-full-attn-idxs-no-window"])
def test_lfm2_s_class_layout_holds_kv_on_its_attention_layers_alone(name, with_transformers_5_12):
    from llm_memory_calculator.calculator import ModelMemoryCalculator
    from llm_memory_calculator.model.facts import ModelFacts

    cfg = _synth(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kv = ModelMemoryCalculator().calculate_kv_cache(cfg, 1, 4096, "bf16")
        facts = ModelFacts.from_config(copy.deepcopy(cfg))
    # K + V, 8 heads of 64, 2 B, over the 6 attention layers
    assert kv * 1e9 == pytest.approx(LFM2_ATTENTION * 2 * 8 * 64 * 4096 * 2)
    assert (facts.layers.full, facts.layers.recurrent) == (LFM2_ATTENTION, LFM2_CONV)


@pytest.mark.parametrize("name", ["lfm2-full-attn-idxs", "lfm2-full-attn-idxs-no-window"])
def test_lfm2_s_convolutions_hold_their_conv_state(name, with_transformers_5_12):
    """Placed by the class, not by a stated `layer_types`, the conv layers held no state (0 B) in
    the calculator, the facts and the engine estimate."""
    from llm_memory_calculator.calculator import ModelMemoryCalculator
    from llm_memory_calculator.model.facts import ModelFacts

    cfg = _synth(name)
    truth = LFM2_CONV * LFM2_WIDTH * LFM2_L * 2  # bf16
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        state = ModelMemoryCalculator().calculate_state_memory(copy.deepcopy(cfg), 1, "bf16")
        facts = ModelFacts.from_config(copy.deepcopy(cfg)).state_bytes_per_sequence()
    assert state * 1e9 >= truth
    assert facts >= truth


def test_lfm2_s_layout_key_without_transformers_is_full_on_every_layer(without_transformers):
    """Without the class to ask, the stated key is no layout the rule describes: every layer is
    charged full attention -- above the class, never below it."""
    from llm_memory_calculator.layer_plan import layer_kinds

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kinds = layer_kinds(_synth("lfm2-full-attn-idxs-no-window"))
    assert kinds["attn"].count("full") == 24


# ---------------------------------------------------------- the ONNX Runtime engine's activation

#: ChatGLM3-6B's spelling: `num_layers`, `padded_vocab_size`, `multi_query_group_num`.
CHATGLM = dict(model_type="chatglm", architectures=["ChatGLMModel"], num_layers=28,
               hidden_size=4096, ffn_hidden_size=13696, num_attention_heads=32,
               multi_query_attention=True, multi_query_group_num=2, kv_channels=128,
               padded_vocab_size=65024, seq_length=8192)
#: GPT-2 medium's: `n_layer`, `n_embd`, `n_head`.
GPT2 = dict(model_type="gpt2", architectures=["GPT2LMHeadModel"], n_layer=24, n_embd=1024,
            n_head=16, n_positions=1024, vocab_size=50257)


@pytest.mark.parametrize("cfg,layers,width", [(CHATGLM, 28, 4096), (GPT2, 24, 1024)])
def test_the_onnx_runtime_arena_holds_every_layer_of_the_model(cfg, layers, width):
    """The arena's prefill peak is layers x width x 6 live buffers a token (plus the logits):
    ChatGLM's 28 layers were read as 12, GPT-2's 1,024-wide layers as 768."""
    from llm_memory_calculator.inference_engines import OnnxRuntimeEngine

    got = OnnxRuntimeEngine().activation_bytes(dict(cfg), 1, 4096, 2, 0.0)
    assert got >= 4096 * layers * width * OnnxRuntimeEngine.LAYER_WORK * 2


# --------------------------------------------------------------------- the ONNX-GenAI reader

ONNX = os.path.join(FIXTURES, "onnx")


def _onnx(name):
    sys.path.insert(0, REPO)
    try:
        import bud_component

        return bud_component.onnx_dir_to_hf_config(os.path.join(ONNX, name))
    finally:
        sys.path.remove(REPO)


def test_the_onnx_reader_sizes_a_moe_from_its_source_config():
    """Qwen1.5-MoE-A2.7B: genai_config.json states no experts; the source config.json beside it
    does (60 experts of 1,408, 4 a token, a 5,632-wide shared expert)."""
    cfg, _ = _onnx("qwen1.5-moe-a2.7b-genai")
    assert (cfg["num_experts"], cfg["num_experts_per_tok"], cfg["moe_intermediate_size"]) == (
        60, 4, 1408)
    assert cfg["shared_expert_intermediate_size"] == 5632


def test_the_onnx_reader_refuses_an_unsizable_architecture():
    """A Qwen3.5 hybrid folder without its source config.json: neither dense by type nor sized by
    a config, it is refused rather than modelled dense (bud_component's fail-closed rule)."""
    with pytest.raises(ValueError, match="not a known DENSE"):
        _onnx("qwen3.5-hybrid-genai")


def test_the_onnx_reader_reads_a_dense_folder():
    cfg, _ = _onnx("qwen2.5-0.5b-instruct-genai")
    assert (cfg["num_hidden_layers"], cfg["hidden_size"], cfg["num_key_value_heads"],
            cfg["head_dim"]) == (24, 896, 2, 64)


# ------------------------------------------------- the training characteristics' KV and depth

def test_the_training_characteristics_hold_mla_s_rotary_key_in_a_layer_s_cache():
    """`AttentionConfig.compute_kv_cache_size`: an MLA layer caches its 512-wide latent and the
    64-wide rotary key beside it (576 a token), not the latent alone."""
    from llm_memory_calculator.training.model_characteristics import analyze_model_config

    with open(os.path.join(FIXTURES, "deepseek_v2_lite_chat_config.json")) as f:
        m = analyze_model_config(json.load(f))
    assert m.attention.compute_kv_cache_size(1, 4096) * 1e9 == pytest.approx(4096 * 576 * 2)


def test_the_training_characteristics_read_the_depth_and_vocabulary_under_any_spelling():
    """ChatGLM3's `num_layers` and `padded_vocab_size` were a default 32 layers and 32,000
    tokens."""
    from llm_memory_calculator.training.model_characteristics import analyze_model_config

    m = analyze_model_config(dict(CHATGLM))
    assert (m.num_layers, m.vocab_size) == (28, 65024)


def test_the_training_characteristics_read_a_stated_head_width():
    """Qwen3-0.6B's heads are 128 wide (`head_dim`), not 1,024 / 16 = 64: a layer's KV was half."""
    from llm_memory_calculator.training.model_characteristics import analyze_model_config

    cfg = dict(model_type="qwen3", hidden_size=1024, num_attention_heads=16,
               num_key_value_heads=8, head_dim=128, num_hidden_layers=28,
               intermediate_size=3072, vocab_size=151936)
    m = analyze_model_config(cfg)
    assert m.attention.compute_kv_cache_size(1, 4096) * 1e9 == pytest.approx(4096 * 2 * 8 * 128 * 2)


# ------------------------------------------------- a Nemotron-NAS stack's linear blocks in GenZ

@pytest.mark.parametrize("stage", ["prefill", "decode", "chunked"])
def test_genz_runtime_books_a_linear_ffn_block(stage):
    """A Nemotron-NAS (DeciLM) block whose FFN is `replace_with_linear` graphs one `FFN Linear`
    GEMM; the runtime breakdown knew `Attn Linear` alone, so prefill, decode and chunked modelling
    raised on any such stack (`Layer Name:FFN Linear not found in the breakdown function`)."""
    import kv_corpus

    out = kv_corpus.measure_runtime(_synth("nemotron-nas-linear-blocks"))
    assert f"runtime_{stage}_error" not in out, out.get(f"runtime_{stage}_error")
    latency, weights_mb, kv_mb = out[f"runtime_{stage}"]
    assert latency > 0 and weights_mb > 0
