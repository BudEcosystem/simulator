"""Inference activation peak under chunked prefill.

The model this replaced was ``batch x seq_length x hidden x 10..15``, which
priced two things inference does not do: materializing the whole context in one
forward pass, and retaining every layer's activations for a backward pass. At
5 x 100k on Qwen3.6-27B that read 76.8 GB against a real figure under 1 GB, and
it made the reported total roughly double the truth.
"""

import json
from pathlib import Path

import pytest

from llm_memory_calculator import calculate_memory
from llm_memory_calculator.activation_memory import (
    DEFAULT_PREFILL_CHUNK_TOKENS,
    calculate_activation_bytes,
    prefill_tokens_in_flight,
)

FIXTURES = Path(__file__).parent / "fixtures" / "hybrid_configs"

LLAMA31_8B = dict(
    model_type="llama",
    hidden_size=4096,
    intermediate_size=14336,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,
    hidden_act="silu",
    vocab_size=128256,
)


def cfg(name):
    return json.loads((FIXTURES / f"{name}.json").read_text())


# ------------------------------------------------------- tokens in flight


def test_tokens_in_flight_are_bounded_by_the_prefill_chunk():
    """A 100k-context request is not a 100k-token forward pass."""
    assert prefill_tokens_in_flight(5, 100_000) == DEFAULT_PREFILL_CHUNK_TOKENS
    assert prefill_tokens_in_flight(5, 100_000, 2048) == 2048
    # Below the chunk, the real token count wins.
    assert prefill_tokens_in_flight(1, 512) == 512
    assert prefill_tokens_in_flight(4, 100) == 400


def test_activation_is_flat_in_context_length_once_chunked():
    """Doubling the context does not change the peak -- it adds another chunk.

    This is the property the old model got wrong: it scaled activations linearly
    with seq_length forever, so a long-context sizing was dominated by a term
    that does not actually grow.
    """
    a = calculate_activation_bytes(LLAMA31_8B, 1, 32_768, 2)
    b = calculate_activation_bytes(LLAMA31_8B, 1, 262_144, 2)
    assert a == b


def test_activation_is_independent_of_depth():
    """Layer N's activations are freed before layer N+1 allocates."""
    shallow = calculate_activation_bytes(dict(LLAMA31_8B, num_hidden_layers=4), 8, 8192, 2)
    deep = calculate_activation_bytes(dict(LLAMA31_8B, num_hidden_layers=126), 8, 8192, 2)
    assert shallow == deep


def test_activation_scales_linearly_with_the_chunk():
    base = calculate_activation_bytes(LLAMA31_8B, 64, 8192, 2, max_num_batched_tokens=2048)
    doubled = calculate_activation_bytes(LLAMA31_8B, 64, 8192, 2, max_num_batched_tokens=4096)
    assert doubled == pytest.approx(base * 2, rel=1e-12)


# ------------------------------------------------------- absolute geometry


def test_llama31_8b_activation_matches_hand_derivation():
    """3 resident hidden buffers + the widest single-layer working set.

    attn: q(32*128) + k,v(2*8*128) + attn_out(32*128) = 10240
    ffn : gate+up = 2 * 14336 = 28672   <- wider, so it sets the peak
    """
    resident = 3 * 4096
    ffn = 2 * 14336
    expected = (resident + ffn) * 8192 * 2 * 1.2
    got = calculate_activation_bytes(LLAMA31_8B, 64, 8192, 2)
    assert got == pytest.approx(expected, rel=1e-12)
    assert got / 1e6 == pytest.approx(805.3, rel=1e-3)


def test_non_gated_ffn_counts_one_matrix_not_two():
    gated = calculate_activation_bytes(dict(LLAMA31_8B, hidden_act="silu"), 1, 8192, 2)
    plain = calculate_activation_bytes(dict(LLAMA31_8B, hidden_act="gelu"), 1, 8192, 2)
    assert gated > plain


def test_moe_prices_only_the_routed_experts():
    """A 128-expert model does not run 128 FFNs per token; it runs top_k."""
    moe = dict(
        model_type="qwen3_moe",
        hidden_size=4096,
        intermediate_size=14336,
        moe_intermediate_size=1536,
        num_experts=128,
        num_experts_per_tok=8,
        num_hidden_layers=48,
        num_attention_heads=32,
        num_key_value_heads=4,
        hidden_act="silu",
    )
    got = calculate_activation_bytes(moe, 1, 8192, 2)
    expected = (3 * 4096 + 8 * 1536 * 2) * 8192 * 2 * 1.2
    assert got == pytest.approx(expected, rel=1e-12)
    # Charging all 128 experts would be 16x the FFN term.
    assert 128 * 1536 * 2 > 8 * 1536 * 2


def test_gated_attention_output_widens_the_attention_path():
    """Qwen3.5/3.6 set attn_output_gate, so q_proj emits q *and* its gate --
    visible in the shipped q_proj weight, [12288, 5120] for 24 heads x 256."""
    base = dict(
        hidden_size=5120,
        intermediate_size=17408,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        hidden_act="silu",
    )
    ungated = calculate_activation_bytes(base, 1, 8192, 2)
    gated = calculate_activation_bytes(dict(base, attn_output_gate=True), 1, 8192, 2)
    # The FFN (2*17408=34816) still dominates both, so the peak is unchanged...
    assert gated == ungated
    # ...but the attention width itself did grow, which matters for models whose
    # attention path is the wider of the two.
    narrow_ffn = dict(base, intermediate_size=2048)
    assert calculate_activation_bytes(
        dict(narrow_ffn, attn_output_gate=True), 1, 8192, 2
    ) > calculate_activation_bytes(narrow_ffn, 1, 8192, 2)


def test_decode_steady_state_is_negligible():
    """One new token per sequence -- the peak lives in prefill, not decode."""
    assert calculate_activation_bytes(LLAMA31_8B, 64, 1, 2) / 1e6 < 10


# ------------------------------------------------------------ end to end


def test_qwen36_total_is_no_longer_dominated_by_a_phantom_activation_term():
    """The workload that started this: 5 concurrent, 80k in + 20k out.

    Before: activations 76.81 GB, total 301 GB. The activation term alone was
    larger than the entire real footprint.
    """
    r = calculate_memory(
        cfg("Qwen_Qwen3.6-27B"),
        batch_size=5,
        seq_length=100_000,
        precision="bf16",
        max_num_seqs=5,
    )
    assert r.activation_memory_gb < 1.5
    assert r.kv_cache_gb == pytest.approx(32.768, rel=1e-9)
    assert 85 < r.total_memory_gb < 100
    # Activations must no longer be the largest term.
    assert r.activation_memory_gb < r.kv_cache_gb
    assert r.activation_memory_gb < r.weight_memory_gb


def test_max_num_batched_tokens_reaches_the_activation_term():
    """The engine flag has to actually plumb through calculate_memory.

    Uses a text-only model deliberately: on a multimodal config the vision
    encoder's one-shot pass is correctly *independent* of the text chunk, so the
    ratio would not be clean.
    """
    c = cfg("Qwen_Qwen3-Next-80B-A3B-Instruct")
    small = calculate_memory(c, batch_size=5, seq_length=100_000, precision="bf16",
                             max_num_batched_tokens=2048)
    large = calculate_memory(c, batch_size=5, seq_length=100_000, precision="bf16",
                             max_num_batched_tokens=16384)
    assert large.activation_memory_gb == pytest.approx(
        small.activation_memory_gb * 8, rel=1e-9
    )


def test_vision_term_does_not_scale_with_the_text_chunk():
    """The image encoder runs once over its patches, not once per prefill chunk."""
    c = cfg("Qwen_Qwen3.6-27B")
    text_only = {k: v for k, v in c.items() if k != "vision_config"}
    vision_bytes = calculate_activation_bytes(c, 1, 8192, 2) - calculate_activation_bytes(
        text_only, 1, 8192, 2
    )
    vision_bytes_big_chunk = calculate_activation_bytes(
        c, 1, 8192, 2, max_num_batched_tokens=65536
    ) - calculate_activation_bytes(text_only, 1, 8192, 2, max_num_batched_tokens=65536)
    assert vision_bytes == pytest.approx(vision_bytes_big_chunk, rel=1e-12)
    assert vision_bytes > 0


def test_multimodal_adds_a_vision_encoder_term():
    """Gemma-3 is text+vision; the vision tower's one-shot pass is separate."""
    c = cfg("unsloth_gemma-3-27b-it")
    with_vision = calculate_activation_bytes(c, 1, 8192, 2)
    text_only = calculate_activation_bytes(
        {k: v for k, v in c.items() if k != "vision_config"}, 1, 8192, 2
    )
    assert with_vision > text_only
