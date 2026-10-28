"""Readers the KV gate held to their ground truth and found below it (round 12).

Each case is a figure a reader computes for a config, against the model's own count: transformers
5.12's model for the config (meta device), the checkpoint's tensors, or the published
arithmetic the test states. Every reader below takes the fact from layer_plan's one reader for it.
"""

import contextlib
import copy
import io
import json
import os
import sys

import pytest

from llm_memory_calculator import calculate_memory
from llm_memory_calculator.model.facts import ModelFacts

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import kv_corpus  # noqa: E402

MIB = 2**20


def _config(name):
    if name.startswith("synth:"):
        return copy.deepcopy(kv_corpus.synthetic_configs(FIXTURES)[name[6:]])
    with open(os.path.join(FIXTURES, name)) as f:
        return json.load(f)


def _genz(cfg, tmp_path):
    for sub in ("model", "sparsity"):
        (tmp_path / sub).mkdir(exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        return kv_corpus.measure_genz(cfg, str(tmp_path))


@pytest.mark.parametrize("name,layers", [("deepseek_v2_lite_chat_config.json", 27),
                                         ("planner/models/deepseek-v3.json", 61)])
def test_genz_graphs_hold_mla_s_latent_as_the_cache(name, layers, tmp_path):
    """MLA caches one latent of kv_lora_rank + qk_rope_head_dim (576) per token per layer. Sized
    from the expanded per-head K and V its score operators read (192 + 128), the prefill, decode
    and chunked graphs held 320 elements a token."""
    out = _genz(_config(name), tmp_path)
    truth = layers * 576 * 4096 * 2 / MIB
    assert out["genz_decode_kv_mb"] == pytest.approx(truth)
    assert out["genz_prefill_kv_mb"] == pytest.approx(truth)
    assert out["genz_chunked_kv_mb"] >= truth


def test_facts_read_every_spelling_of_the_routed_experts():
    """SmallThinker states `moe_num_primary_experts` (64) of `moe_ffn_hidden_size` (768): read from
    four other spellings, the facts sized it as a dense model (6.0B of 21.5B). Arithmetic: 52 layers
    of attention (28 x 128 queries, 4 x 128 K / V) and 64 gated experts with a router, an untied
    151936 x 2560 embedding and head: 21,506,293,760 matrix parameters."""
    cfg = _config("hybrid_configs/PowerInfer_SmallThinker-21BA3B-Instruct.json")
    facts = ModelFacts.from_config(cfg)
    assert facts.weight_bytes() / 2 >= 21_506_293_760
    assert facts.experts == (64, 6)


@pytest.mark.parametrize("name,truth", [
    # Llama4ForCausalLM on the text config: its class defaults (hidden_act "silu",
    # tie_word_embeddings False) where the config states none
    ("synth:llama4-scout-text", 107_769_861_120),
    ("synth:llama4-maverick-text", 400_711_848_960),
    # NemotronHForCausalLM: a wrapper's text_config whose depth is its per-layer list
    ("synth:nemotronh-list-depth-nested", 8_100_852_736),
    # ChatGLM3-6B's published count: SwiGLU `ffn_hidden_size`, `padded_vocab_size`
    ("synth:chatglm3", 6_243_584_000),
])
def test_reported_parameters_hold_the_model(name, truth):
    rep = calculate_memory(_config(name), batch_size=1, seq_length=4096, precision="bf16")
    assert rep.parameter_count >= truth, f"{rep.parameter_count:,} < {truth:,}"


@pytest.mark.parametrize("name,truth", [
    ("synth:chatglm3", 6_243_221_504),  # its matrices
    ("hybrid_configs/PowerInfer_SmallThinker-21BA3B-Instruct.json", 21_506_293_760),
])
def test_genz_graph_weights_hold_the_model(name, truth, tmp_path):
    out = _genz(_config(name), tmp_path)
    assert out["genz_prefill_weights_mb"] >= truth * 2 / MIB


def test_a_qwen35_gguf_counts_its_gated_deltanet_as_one():
    """llama.cpp's qwen35 writes its Gated DeltaNet with `ssm.*` keys; read as a Mamba-2 mixer, the
    GGUF's parameters fell 26M short of its own tensors (and its attention's output gate went
    uncounted). Read as what it is, the GGUF-shaped config counts what the checkpoint's text model
    holds (transformers 5.12 Qwen3_5ForCausalLM on its text_config: 4,205,751,296)."""
    from llm_memory_calculator.gguf_layers import hf_layer_keys_from_gguf

    tc = _config("hybrid_configs/Qwen_Qwen3.5-4B.json")["text_config"]
    meta = {"attention.head_count": 16, "attention.head_count_kv": 4,
            "full_attention_interval": 4, "ssm.conv_kernel": 4, "ssm.state_size": 128,
            "ssm.group_count": 16, "ssm.time_step_rank": 32, "ssm.inner_size": 128 * 32}
    g = dict(model_type="qwen35", num_hidden_layers=32, hidden_size=tc["hidden_size"],
             intermediate_size=tc["intermediate_size"], vocab_size=tc["vocab_size"],
             num_attention_heads=16, num_key_value_heads=4, head_dim=256, hidden_act="silu",
             tie_word_embeddings=True)
    keys = hf_layer_keys_from_gguf(lambda k: meta.get(k), 32, "qwen35")
    assert keys["linear_num_value_heads"] == 32 and keys["linear_value_head_dim"] == 128
    g.update(keys)
    rep = calculate_memory(g, batch_size=1, seq_length=4096, precision="bf16")
    assert rep.parameter_count >= 4_205_751_296


def test_gemma3_4b_built_in_states_its_head_width(tmp_path):
    """GenZ's built-in Gemma-3-4B left head_dim out (256 in the published config): its KV was
    sized at hidden / heads = 160. 5 full and 29 windowed (1024) layers, 4 x 256 K and V."""
    out = _genz("google/gemma-3-4B", tmp_path)
    truth = (5 * 4096 + 29 * 1024) * 2048 * 2 / MIB
    assert out["genz_decode_kv_mb"] >= truth
