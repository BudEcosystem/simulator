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


HYBRIDS = sorted(p.stem for p in FIXTURES.glob("*.json"))


def _figures(mc):
    return dict(layers=mc.num_decoder_layers, kv=mc.get_kv_size(), state=mc.get_state_bytes(2))


@pytest.mark.parametrize("name", HYBRIDS)
def test_a_database_model_holds_the_kv_and_state_its_hf_config_holds(name):
    """The converter computed the recurrent state and dropped it (a Mamba-2 stack fell back to
    Mamba-1's conv buffer, 0.29% low on Mamba-Codestral), and passed no MLA dimensions (DeepSeek
    and Kimi Linear charged full K and V per head, 25x and 8x their latent cache)."""
    converted, loaded = _both(_cfg(name))
    assert _figures(converted) == _figures(loaded)


ENC_DEC = FIXTURES.parent / "encoder_decoder"


def _enc_dec(name, **changes):
    c = json.loads((ENC_DEC / f"{name}.json").read_text())
    c.update(changes)
    return c


# name -> (config, encoder layers, decoder layers, K+V elements a token adds)
ENC_DEC_CASES = {
    "dia": (lambda: _enc_dec("nari-labs_Dia-1.6B-0626"), 12, 18, 2 * 18 * 4 * 128),
    "whisper-turbo": (lambda: _enc_dec("onnx-community_whisper-large-v3-turbo"), 32, 4,
                      2 * 4 * 20 * 64),
    "moonshine": (lambda: _enc_dec("onnx-community_moonshine-base-ONNX"), 8, 8, 2 * 8 * 8 * 52),
    "cohere-asr": (lambda: _enc_dec("onnx-community_cohere-transcribe-03-2026-ONNX"), 48, 8,
                   2 * 8 * 8 * 128),
    "flan-t5-xl": (lambda: _enc_dec("t5_flan-xl-geometry"), 24, 24, 2 * 24 * 32 * 64),
    "t5-asymmetric": (lambda: _enc_dec("t5_flan-xl-geometry", num_layers=12,
                                       num_decoder_layers=4), 12, 4, 2 * 4 * 32 * 64),
    "bart": (lambda: _enc_dec("bart_transformers-5.12-default"), 12, 12, 2 * 12 * 16 * 64),
    # as a config class's `to_dict` writes it, `architectures: null`
    "bart-unnamed": (lambda: _enc_dec("bart_transformers-5.12-default", architectures=None),
                     12, 12, 2 * 12 * 16 * 64),
}


@pytest.mark.parametrize("case", sorted(ENC_DEC_CASES))
def test_a_database_encoder_decoder_is_graphed_with_its_decoder_s_own_depth(case):
    """The converter split the decoder's depth in two (`num_hidden_layers // 2`): Dia's 18
    decoder layers became 9, flan-t5-xl's 24 became 12 with a default 32 heads; Whisper's and
    BART's width and heads were the 7B defaults."""
    make, enc, dec, kv = ENC_DEC_CASES[case]
    converted, loaded = _both(make())
    assert (converted.num_encoder_layers, converted.num_decoder_layers) == (enc, dec)
    assert converted.get_kv_size() == kv
    assert (loaded.num_encoder_layers, loaded.num_decoder_layers, loaded.get_kv_size()) == (
        enc, dec, kv)
