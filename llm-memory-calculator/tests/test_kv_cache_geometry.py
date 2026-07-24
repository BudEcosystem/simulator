"""KV-cache geometry regression tests.

These pin three families of bug that under- or over-counted the KV cache and, in
a downstream deployment control plane, translated into pods sized against the
wrong context length:

  * explicit ``head_dim`` ignored (2x under-count on Qwen3-family models),
  * interleaved sliding-window attention flattened to a single global window
    (large over/under-count on Gemma-3), and
  * MLA priced as two full heads instead of one shared latent (1.78x over-count
    on DeepSeek-V3).

Numbers below are per-model ground truth, cross-checked against each model's
published config and vLLM's own attention specs.
"""

import pytest

from llm_memory_calculator.calculator import ModelMemoryCalculator
from llm_memory_calculator.config_normalizer import ConfigNormalizer


def kv_bytes(config, batch_size, seq_length, precision="bf16"):
    """Normalize + dispatch, returning KV bytes (the API returns decimal GB)."""
    normalized = ConfigNormalizer.normalize_config(config)
    calc = ModelMemoryCalculator()
    calc.model_type = calc.detect_model_type(normalized)
    calc.attention_type = calc.detect_attention_type(normalized)
    return calc.calculate_kv_cache(normalized, batch_size, seq_length, precision) * 1e9


# --------------------------------------------------------------- explicit head_dim

QWEN3_0_6B = dict(
    model_type="qwen3",
    hidden_size=1024,
    num_hidden_layers=28,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=128,  # NOT hidden_size // num_attention_heads (== 64)
    max_position_embeddings=40960,
)


def test_gqa_reads_explicit_head_dim():
    # 2(K,V) * 8 kv heads * 128 head_dim * 2 bytes = 4096 B per layer per token;
    # 28 layers => 114688 B/token. Deriving head_dim as 1024//16=64 would halve it.
    per_token = kv_bytes(QWEN3_0_6B, batch_size=1, seq_length=1)
    assert per_token == 114688


def test_gqa_head_dim_scales_the_whole_cache():
    assert kv_bytes(QWEN3_0_6B, 1, 8926) == 114688 * 8926


def test_mha_and_gqa_unchanged_when_head_dim_is_implicit():
    # hidden_size == num_heads * head_dim, so reading the explicit key or deriving
    # it must give the identical answer -- guards against a regression that would
    # move every classic model.
    gpt2 = dict(
        model_type="gpt2",
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
    )
    assert kv_bytes(gpt2, 1, 1024) == 2 * 24 * 1024 * 1024 * 2

    llama = dict(
        model_type="llama",
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
    )
    assert kv_bytes(llama, 1, 8192) == 2 * 32 * 8192 * 8 * 128 * 2


# ------------------------------------------------------- interleaved sliding window

GEMMA3_27B = dict(
    model_type="gemma3",
    hidden_size=5376,
    num_hidden_layers=62,
    num_attention_heads=32,
    num_key_value_heads=16,
    head_dim=128,
    max_position_embeddings=131072,
    sliding_window=1024,
    sliding_window_pattern=6,  # 1 global layer every 6 -> 10 global, 52 local
)


def test_sliding_window_pattern_synthesizes_layer_metadata():
    normalized = ConfigNormalizer.normalize_config(GEMMA3_27B)
    meta = normalized["_layer_metadata"]
    assert meta["has_mixed_attention"] is True
    assert meta["num_full_layers"] == 10
    assert meta["num_sliding_layers"] == 52


def test_gemma3_kv_counts_global_and_local_layers_separately():
    # 10 global layers at 131072 tokens + 52 local layers at 1024 tokens,
    # per_token = 2 * 16 kv heads * 128 head_dim * 2 bytes = 8192 B/layer.
    per_layer_token = 2 * 16 * 128 * 2
    expected = per_layer_token * (10 * 131072 + 52 * 1024)
    got = kv_bytes(GEMMA3_27B, 1, 131072)
    assert got == pytest.approx(expected, rel=0.01)
    # Sanity: far larger than the ~0.64 GiB a uniform-1024-window flattening gave.
    assert got > 10 * 2**30


def test_interleave_is_detected_through_a_multimodal_text_config():
    """Gemma-3 ships as multimodal: the attention layout lives in text_config.

    The per-layer metadata is built when text_config is normalized, so reading it
    only from the outer config skips the per-layer path entirely and clamps every
    layer to the sliding window -- a ~21x under-count at 128k on the real config
    shape. The multimodal and flat forms must agree.
    """
    text = {k: v for k, v in GEMMA3_27B.items() if k != "model_type"}
    multimodal = dict(
        model_type="gemma3",
        vision_config=dict(hidden_size=1152, image_size=896, patch_size=14),
        text_config=text,
    )
    assert ModelMemoryCalculator().detect_model_type(multimodal) == "multimodal"
    assert kv_bytes(multimodal, 1, 131072) == pytest.approx(
        kv_bytes(GEMMA3_27B, 1, 131072), rel=1e-9
    )


def test_use_sliding_window_false_disables_the_window():
    # Qwen2.5 ships a sliding_window value with the feature switched off; the
    # cache must be sized at the full context, not clamped to the window.
    qwen25 = dict(
        model_type="qwen2",
        hidden_size=3584,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        sliding_window=32768,
        use_sliding_window=False,
    )
    per_token = 2 * 4 * (3584 // 28) * 2 * 28  # 28 layers, head_dim 128
    # Use a sequence LONGER than the window (32768), so a bug that clamps to the
    # window would show: min(seq, 32768) would cap KV at the window, under-counting
    # ~4x at 131072. The unclamped cache must scale with the full sequence.
    assert kv_bytes(qwen25, 1, 131072) == per_token * 131072


# ------------------------------------------------------------------------- MLA

DEEPSEEK_V3 = dict(
    model_type="deepseek_v3",
    hidden_size=7168,
    num_hidden_layers=61,
    num_attention_heads=128,
    num_key_value_heads=128,
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    max_position_embeddings=163840,
)


def test_mla_caches_one_latent_not_two_heads():
    # One compressed latent per token per layer: (512 + 64) bytes-elements,
    # NO factor of 2. 61 layers * 576 * 2 bytes = 70272 B/token.
    per_token = 61 * (512 + 64) * 2
    assert kv_bytes(DEEPSEEK_V3, 1, 1) == per_token
    assert kv_bytes(DEEPSEEK_V3, 1, 4096) == per_token * 4096


def test_mla_is_detected_and_far_smaller_than_naive_mha():
    normalized = ConfigNormalizer.normalize_config(DEEPSEEK_V3)
    calc = ModelMemoryCalculator()
    assert calc.detect_attention_type(normalized) == "mla"
    # A naive 2 * num_kv_heads * head_dim treatment would be an order larger.
    naive = 2 * 128 * (7168 // 128) * 2 * 61
    assert kv_bytes(DEEPSEEK_V3, 1, 1) < naive / 5


# ------------------------------------------------------------------- batch scaling


@pytest.mark.parametrize("config", [QWEN3_0_6B, GEMMA3_27B, DEEPSEEK_V3])
def test_kv_is_linear_in_batch(config):
    one = kv_bytes(config, 1, 4096)
    eight = kv_bytes(config, 8, 4096)
    assert eight == 8 * one


# ------------------------------------------------- per-rank TP KV replication

LLAMA_GQA = dict(
    model_type="llama",
    hidden_size=4096,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,
    max_position_embeddings=8192,
)


def _report_kv(config, tp):
    return (
        ModelMemoryCalculator()
        .calculate_total_memory(
            config, batch_size=1, seq_length=8192, precision="bf16", tensor_parallel=tp
        )
        .kv_cache_bytes
    )


def test_gqa_kv_shards_evenly_when_tp_divides_kv_heads():
    # 8 kv heads, tp=4 -> 2 heads/rank -> per-rank == full / 4.
    assert _report_kv(LLAMA_GQA, 4) * 4 == pytest.approx(
        _report_kv(LLAMA_GQA, 1), rel=1e-9
    )


def test_gqa_kv_stops_shrinking_past_tp_equals_kv_heads():
    # 8 kv heads: at tp=8 and tp=16 each rank holds exactly one replicated head,
    # so the per-rank cache is identical -- a flat divide-by-tp would halve it.
    assert _report_kv(LLAMA_GQA, 8) == pytest.approx(
        _report_kv(LLAMA_GQA, 16), rel=1e-9
    )


def test_mla_kv_is_replicated_not_sharded_across_ranks():
    # MLA caches one shared latent; every rank holds the whole thing.
    assert _report_kv(DEEPSEEK_V3, 1) == pytest.approx(
        _report_kv(DEEPSEEK_V3, 8), rel=1e-9
    )


# --------------------------------------------------------------- SSM state dtype

FALCON_H1_LIKE = dict(
    model_type="falcon_h1",
    hidden_size=4096,
    num_hidden_layers=44,
    num_attention_heads=32,
    num_key_value_heads=8,
    state_size=256,
    expand=1,
    layer_types=["mamba"] * 44,
)


def test_ssm_recurrent_state_is_priced_at_fp32():
    calc = ModelMemoryCalculator()
    calc.model_type = "hybrid"
    normalized = ConfigNormalizer.normalize_config(FALCON_H1_LIKE)
    state_gb = calc.calculate_state_memory(normalized, batch_size=1, precision="bf16")
    # 44 layers * 256 state * 4096 hidden * 1 expand * 4 bytes (fp32), not 2.
    expected = 44 * 256 * 4096 * 1 * 4 / 1e9
    assert state_gb == pytest.approx(expected, rel=1e-9)


def test_ssm_state_dtype_override_is_honored():
    calc = ModelMemoryCalculator()
    calc.model_type = "hybrid"
    cfg = dict(FALCON_H1_LIKE, mamba_ssm_cache_dtype="bf16")
    state_gb = calc.calculate_state_memory(
        ConfigNormalizer.normalize_config(cfg), batch_size=1, precision="bf16"
    )
    expected = 44 * 256 * 4096 * 1 * 2 / 1e9
    assert state_gb == pytest.approx(expected, rel=1e-9)


# ------------------------------------------------- null-valued optional keys


def test_head_dim_null_is_treated_as_absent_everywhere():
    """HF configs ship optional keys as explicit `null` (the real Qwen3 config has
    `sliding_window: null`, `rope_scaling: null`). `.get(key, default)` returns the
    null rather than the default, so every head_dim reader must use `or`.
    """
    cfg = dict(
        model_type="qwen3",
        hidden_size=1024,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=None,
        vocab_size=151936,
    )
    calc = ModelMemoryCalculator()
    # weight accounting must not raise on a null head_dim ...
    ratio = calc._estimate_skip_ratio(cfg, ["self_attn"], 1_000_000_000)
    assert 0 < ratio < 1
    # ... and the KV geometry must fall back to hidden // heads
    assert kv_bytes(cfg, 1, 1) == 2 * 28 * 8 * (1024 // 16) * 2


def test_disabled_window_with_explicit_layer_types_does_not_crash():
    """A config can carry BOTH layer_types and `use_sliding_window: false`.

    The normalizer scrubs the window to None so the global path stops clamping,
    but the per-layer path then compared that None against an int and raised
    TypeError. "No window" means those layers attend to the full sequence.
    """
    per_layer_token = 2 * 16 * 128 * 2
    base = dict(
        model_type="gemma3",
        hidden_size=5376,
        num_hidden_layers=4,
        num_attention_heads=32,
        num_key_value_heads=16,
        head_dim=128,
        sliding_window=1024,
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
    )
    # window disabled -> every layer full-attention
    assert kv_bytes(dict(base, use_sliding_window=False), 1, 8192) == per_layer_token * 4 * 8192
    # window active -> 2 local at the window + 2 global at full context
    assert kv_bytes(base, 1, 8192) == pytest.approx(per_layer_token * (2 * 1024 + 2 * 8192), rel=1e-9)
