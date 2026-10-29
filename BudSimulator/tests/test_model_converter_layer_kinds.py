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


def _t5_xl():
    """flan-t5-xl as transformers 5.12's T5Config(d_model=2048, d_kv=64, num_heads=32,
    num_layers=24).to_dict() writes it."""
    return dict(model_type="t5", is_encoder_decoder=True, d_model=2048, d_kv=64, d_ff=5120,
                num_heads=32, num_layers=24, num_decoder_layers=24, vocab_size=32128)


def _blip2(kind):
    """Blip2Config / InstructBlipConfig(text_config=<flan-t5-xl>).to_dict() (the
    Salesforce/blip2-flan-t5-xl shape): a whole encoder-decoder under `text_config`, spelled
    `d_model` / `num_heads` / `num_decoder_layers`, beside a 12-layer, 768-wide Q-Former."""
    return dict(model_type=kind, is_encoder_decoder=True, use_decoder_only_language_model=False,
                num_query_tokens=32, text_config=_t5_xl(),
                qformer_config=dict(model_type=kind + "_qformer", hidden_size=768,
                                    num_hidden_layers=12, num_attention_heads=12,
                                    intermediate_size=3072),
                vision_config=dict(hidden_size=1408, num_hidden_layers=39, num_attention_heads=16,
                                   image_size=224, patch_size=14))


def _florence2():
    """Florence2Config().to_dict(): a BART-like encoder-decoder under `text_config`."""
    return dict(model_type="florence2", is_encoder_decoder=True,
                text_config=dict(model_type="florence2_language", d_model=1024, encoder_layers=12,
                                 decoder_layers=12, encoder_attention_heads=16,
                                 decoder_attention_heads=16, encoder_ffn_dim=4096,
                                 decoder_ffn_dim=4096, is_encoder_decoder=True, vocab_size=51289),
                vision_config=dict(model_type="davit"))


# name -> (config, encoder layers, decoder layers, self-attention K+V elements a token adds, the
# cross-attention's per token where the encoder's output length is unstated, its bytes a request
# where it is stated)
ENC_DEC_CASES = {
    "dia": (lambda: _enc_dec("nari-labs_Dia-1.6B-0626"), 12, 18, 2 * 18 * 4 * 128, 0,
            18 * 2 * 16 * 128 * 1024 * 2),
    "whisper-turbo": (lambda: _enc_dec("onnx-community_whisper-large-v3-turbo"), 32, 4,
                      2 * 4 * 20 * 64, 0, 4 * 2 * 20 * 64 * 1500 * 2),
    "moonshine": (lambda: _enc_dec("onnx-community_moonshine-base-ONNX"), 8, 8, 2 * 8 * 8 * 52,
                  2 * 8 * 8 * 52, 0),
    "cohere-asr": (lambda: _enc_dec("onnx-community_cohere-transcribe-03-2026-ONNX"), 48, 8,
                   2 * 8 * 8 * 128, 0, 8 * 2 * 8 * 128 * 5000 * 2),
    "flan-t5-xl": (lambda: _enc_dec("t5_flan-xl-geometry"), 24, 24, 2 * 24 * 32 * 64,
                   2 * 24 * 32 * 64, 0),
    "t5-asymmetric": (lambda: _enc_dec("t5_flan-xl-geometry", num_layers=12,
                                       num_decoder_layers=4), 12, 4, 2 * 4 * 32 * 64,
                      2 * 4 * 32 * 64, 0),
    "bart": (lambda: _enc_dec("bart_transformers-5.12-default"), 12, 12, 2 * 12 * 16 * 64,
             2 * 12 * 16 * 64, 0),
    # as a config class's `to_dict` writes it, `architectures: null`
    "bart-unnamed": (lambda: _enc_dec("bart_transformers-5.12-default", architectures=None),
                     12, 12, 2 * 12 * 16 * 64, 2 * 12 * 16 * 64, 0),
    # a whole encoder-decoder nested under `text_config`
    "blip-2-flan-t5-xl": (lambda: _blip2("blip-2"), 24, 24, 2 * 24 * 32 * 64, 2 * 24 * 32 * 64, 0),
    "instructblip-flan-t5-xl": (lambda: _blip2("instructblip"), 24, 24, 2 * 24 * 32 * 64,
                                2 * 24 * 32 * 64, 0),
    "florence-2": (_florence2, 12, 12, 2 * 12 * 16 * 64, 2 * 12 * 16 * 64, 0),
}


@pytest.mark.parametrize("case", sorted(ENC_DEC_CASES))
def test_a_database_encoder_decoder_is_graphed_with_its_decoder_s_own_depth(case):
    """The converter split the decoder's depth in two (`num_hidden_layers // 2`): Dia's 18
    decoder layers became 9, flan-t5-xl's 24 became 12 with a default 32 heads; Whisper's and
    BART's width and heads were the 7B defaults. A flan-T5 under BLIP-2's or InstructBLIP's
    `text_config` was passed over for the Q-Former: 12 layers of 768, KV 18,432 elements a token
    against the decoder's 98,304 (origin/main's converter: 262,144)."""
    make, enc, dec, kv, cross_per_token, _ = ENC_DEC_CASES[case]
    converted, loaded = _both(make())
    assert (converted.num_encoder_layers, converted.num_decoder_layers) == (enc, dec)
    assert converted.get_kv_size() == kv + cross_per_token
    assert (loaded.num_encoder_layers, loaded.num_decoder_layers, loaded.get_kv_size()) == (
        enc, dec, kv + cross_per_token)


@pytest.mark.parametrize("case", sorted(ENC_DEC_CASES))
def test_a_database_encoder_decoder_holds_its_cross_attention_cache(case):
    """Each decoder layer's cross-attention K / V over the encoder's output: a fixed cost a
    request where the config states the encoder's length (Whisper's 1,500 frames), in the KV a
    token where it does not. The converter charged it nowhere."""
    make, _, _, _, _, fixed = ENC_DEC_CASES[case]
    converted, loaded = _both(make())
    assert converted.get_request_fixed_bytes(2) == loaded.get_request_fixed_bytes(2) == fixed
    assert converted.cross_attention == loaded.cross_attention
    assert converted.encoder == loaded.encoder


CROSS_FIXTURES = FIXTURES.parent / "cross_attention"


@pytest.mark.parametrize(
    "name,truth",
    [
        # 32 self layers x 2 x 8 x 128 x 2 B x 1,024 + 8 cross layers x 2 x 8 x 128 x 2 B x 4,100
        # vision tokens (MllamaConfig().to_dict(), transformers 5.12)
        ("mllama_transformers-5.12-default", 32 * 2 * 8 * 128 * 2 * 1024 + 8 * 2 * 8 * 128 * 2 * 4100),
        # 12 x 2 x 12 x 64 x 2 B x 1,024 self + the same over the image encoder's `seq_len` 4,096
        # (Pix2StructConfig().to_dict())
        ("pix2struct_transformers-5.12-default", 12 * 2 * 12 * 64 * 2 * 1024 + 12 * 2 * 12 * 64 * 2 * 4096),
    ],
)
def test_a_database_model_holds_its_cross_attention_wherever_the_config_puts_it(name, truth):
    """A decoder-only stack's own cross-attention layers (Mllama's `cross_attention_layers`) hold
    K / V over the vision tokens and no self-attention cache; Pix2Struct's decoder cross-attends to
    an image encoder whose length is its `seq_len`. The converter had graphed Mllama's cross layers
    as self-attention (167,772,160 B at 1k) and charged Pix2Struct's cache over the sequence
    (75,497,472 B) -- below origin/main and below the truth."""
    cfg = json.loads((CROSS_FIXTURES / f"{name}.json").read_text())
    conv, loader = _both(cfg)
    for mc in (conv, loader):
        assert mc.get_kv_size() * 2 * 1024 + mc.get_request_fixed_bytes(2) == truth


@pytest.mark.parametrize(
    "path,params",
    [
        # transformers 5.12's parameter count of each config (meta device)
        ("hybrid_configs/Qwen_Qwen3.5-4B.json", 4_539_265_536),
        ("encoder_decoder/t5_flan-xl-geometry.json", 2_783_959_040),
        ("encoder_decoder/nari-labs_Dia-1.6B-0626.json", 1_611_160_576),
        ("cross_attention/mllama_transformers-5.12-default.json", 10_642_941_475),
        ("planner/models/whisper-large-v3.json", 1_543_490_560),
    ],
)
def test_a_database_model_loads_the_weights_its_hf_config_loads(path, params):
    """The serving path's weights (genz.serving.memory_model.estimate_weight_bytes: the
    ServingSimulator's weight load, the cluster's and disaggregation's B_eff) are the model's own
    count of the config the ModelConfig was read from, whichever reader read it. Counted from the
    converter's decoder fields, Qwen3.5-4B was 7.48 GB of its 9.08 and flan-T5-xl 2.45 of 5.57."""
    from llm_memory_calculator.genz.serving.memory_model import estimate_weight_bytes

    cfg = json.loads((FIXTURES.parent / path).read_text())
    converted, loaded = _both(cfg)
    assert estimate_weight_bytes(converted, "bf16", 1) == estimate_weight_bytes(loaded, "bf16", 1)
    assert estimate_weight_bytes(converted, "bf16", 1) >= 2 * params
