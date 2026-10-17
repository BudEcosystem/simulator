"""An AWQ config with "modules_to_not_convert": null (Qwen's AWQ releases) must normalize, not
crash."""

from llm_memory_calculator.config_normalizer import ConfigNormalizer

QWEN2_5_72B_AWQ = {
    "architectures": ["Qwen2ForCausalLM"],
    "model_type": "qwen2",
    "hidden_size": 8192,
    "intermediate_size": 29568,
    "num_hidden_layers": 80,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "vocab_size": 152064,
    "max_position_embeddings": 32768,
    "quantization_config": {
        "bits": 4,
        "group_size": 128,
        "modules_to_not_convert": None,
        "quant_method": "awq",
        "version": "gemm",
        "zero_point": True,
    },
}


def test_a_null_skip_list_means_no_skipped_modules():
    quant = ConfigNormalizer.normalize_config(QWEN2_5_72B_AWQ)["_quantization"]
    assert quant["skip_modules"] == [] and quant["has_mixed_precision"] is False
