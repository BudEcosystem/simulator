"""BudSimulator's database models are graphed like HF configs (llm-memory-calculator's
docs/LAYER_KINDS.md): the converter reads the depth, the per-layer kinds and the window from the
same table the HF loader reads them from."""

import json
import warnings
from pathlib import Path

import pytest

from src.db.model_converter import ModelConverter
from llm_memory_calculator.genz.Models.get_language_model import get_configs

FIXTURES = (
    Path(__file__).resolve().parents[2]
    / "llm-memory-calculator" / "tests" / "fixtures" / "hybrid_configs"
)


def _cfg(name):
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _both(config):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ModelConverter().hf_to_genz(json.loads(json.dumps(config))), get_configs(
            json.loads(json.dumps(config)))


def _llama4_text_stack():
    """Llama 4's layout: three chunked layers to one full, the chunk in `attention_chunk_size`,
    the decoder under `text_config`."""
    n = 12
    text = dict(model_type="llama4_text", hidden_size=1024, num_hidden_layers=n,
                num_attention_heads=8, num_key_value_heads=2, head_dim=128,
                intermediate_size=4096, vocab_size=32000, attention_chunk_size=8192,
                layer_types=["full_attention" if (i + 1) % 4 == 0 else "chunked_attention"
                             for i in range(n)])
    return dict(model_type="llama4", architectures=["Llama4ForConditionalGeneration"],
                text_config=text, vision_config=dict(hidden_size=768, num_hidden_layers=2))


CASES = {
    "smallthinker": lambda: _cfg("PowerInfer_SmallThinker-21BA3B-Instruct"),  # sliding_window_size
    "llama4-nested": _llama4_text_stack,  # attention_chunk_size, under text_config
    "llama4-flat": lambda: dict(_llama4_text_stack()["text_config"]),
    "gemma-3-27b": lambda: _cfg("unsloth_gemma-3-27b-it"),  # text_config
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_database_model_takes_the_window_its_hf_config_takes(case):
    """The converter passed `sliding_window` alone, so a window spelled `sliding_window_size`
    (SmallThinker) or `attention_chunk_size` (Llama 4's chunked layers), or nested under
    `text_config`, left its windowed layers graphed over the whole context."""
    converted, loaded = _both(CASES[case]())
    assert loaded.sliding_window
    assert converted.sliding_window == loaded.sliding_window
    assert converted.num_decoder_layers == loaded.num_decoder_layers
    assert [lc.attention_type for lc in converted.layer_configs] == [
        lc.attention_type for lc in loaded.layer_configs]


def test_a_database_model_takes_its_depth_from_the_per_layer_list():
    """transformers 5.12 saves Nemotron-H with `layers_block_type` and no `num_hidden_layers`:
    the converter fell back to 32 layers and lost the layout."""
    c = _cfg("nvidia_Nemotron-H-8B-Base-8K")
    blocks = {"M": "mamba", "-": "mlp", "*": "attention"}
    c["layers_block_type"] = [blocks[ch] for ch in c.pop("hybrid_override_pattern")]
    c.pop("num_hidden_layers")
    converted, loaded = _both(c)
    assert converted.num_decoder_layers == loaded.num_decoder_layers == 52
    assert sum(lc.attention_type == "MHA-global" for lc in converted.layer_configs) == 4
    assert converted.get_kv_size() == loaded.get_kv_size() > 0
