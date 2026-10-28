"""Cross-attention in the configs as the hub ships them, priced the same by every reader.

The round-8 fixtures were transformers 5.12's `to_dict()` output. The hub files the simulator
actually reads (huggingface_loader, prepopulated.db) are older and differ in exactly the flags the
encoder-decoder reader keyed on:

* Pix2Struct (google/deplot, pix2struct-docvqa-base, pix2struct-textcaps-base): the top level
  omits `is_encoder_decoder` (Pix2StructConfig's class default, True) and `text_config` says
  `is_encoder_decoder: false`, `is_decoder: true`, `num_heads: 12`, `d_kv: 64`. The decoder still
  cross-attends to the image tower in every layer (Pix2StructTextBlock always builds
  `encoder_decoder_attention`). BLIP's `text_config` is the same shape: `is_decoder: true` beside a
  `vision_config`, and BlipTextLayer builds its `crossattention` exactly when `is_decoder`.
* MusicGen (facebook/musicgen-small): the outer config says `is_encoder_decoder: true`; its nested
  `decoder` block carries the `to_dict` defaults `is_encoder_decoder: false` and
  `add_cross_attention: false`. MusicgenDecoderLayer always builds `encoder_attn`.
* A VisionEncoderDecoder with a BART-family decoder block (Donut, Nougat: `mbart`): the block's
  `num_hidden_layers` is MBartConfig's alias of `encoder_layers` (12); its decoder depth is
  `decoder_layers` (4). Its Swin encoder states `image_size` as [height, width] and four stages
  (`depths`), each after the first merging 2 x 2 patches: Donut hands the decoder
  2,560 / 4 / 8 x 1,920 / 4 / 8 = 80 x 60 = 4,800 tokens, Nougat 28 x 21 = 588.

Truth is hand arithmetic over the configs' own numbers (2-byte K / V), never the code under test.
"""

import json
import warnings
from pathlib import Path

import pytest

from llm_memory_calculator.calculator import ModelMemoryCalculator
from llm_memory_calculator.genz.Models.get_language_model import get_configs
from llm_memory_calculator.layer_plan import (
    cross_attention_cache,
    decoder_view,
    encoder_decoder_stacks,
)

FIX = Path(__file__).parent / "fixtures"


def _load(sub, name):
    return json.loads((FIX / sub / f"{name}.json").read_text())


def _quiet(f, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*a, **kw)


def _per_token(layers, kv_heads, head_dim):
    return layers * 2 * kv_heads * head_dim * 2


# name -> (fixture, (cross layers, cross KV heads, head width, encoder tokens), self layers,
#          self KV bytes per token, one 1,024-token request's self + cross KV)
CASES = {
    # 12 self layers of 12 x 64; cross over the tower's `seq_len` 4,096
    "deplot": ("google_deplot_hub", (12, 12, 64, 4096), 12, _per_token(12, 12, 64),
               _per_token(12, 12, 64) * (1024 + 4096)),  # 188,743,680
    # 12 self layers of 12 x 64; cross over the ViT's (384 / 16)^2 + 1 = 577 tokens
    "blip": ("Salesforce_blip-image-captioning-base_hub", (12, 12, 64, 577), 12,
             _per_token(12, 12, 64), _per_token(12, 12, 64) * (1024 + 577)),  # 59,019,264
    # 24 layers of 16 x 64; a T5 text encoder states no output length: cross over the prompt
    "musicgen-small": ("facebook_musicgen-small_hub", (24, 16, 64, None), 24,
                       _per_token(24, 16, 64), _per_token(24, 16, 64) * (1024 + 1024)),
    # 4 decoder layers of 16 x 64 (d_model 1,024); Swin's 80 x 60 = 4,800 output tokens
    "donut": ("naver-clova-ix_donut-base_hub", (4, 16, 64, 4800), 4, _per_token(4, 16, 64),
              _per_token(4, 16, 64) * (1024 + 4800)),  # 95,420,416
    # 4 decoder layers of 16 x 64; Swin's 28 x 21 = 588 output tokens
    "nougat-small": ("facebook_nougat-small_hub", (4, 16, 64, 588), 4, _per_token(4, 16, 64),
                     _per_token(4, 16, 64) * (1024 + 588)),  # 26,411,008
}


def _cfg(case):
    return _load("cross_attention", CASES[case][0])


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_table_reads_the_hub_shape(case):
    cross = _quiet(cross_attention_cache, _cfg(case))
    assert cross is not None
    assert (cross["layers"], cross["kv_heads"], cross["head_dim"], cross["tokens"]) == CASES[case][1]
    assert encoder_decoder_stacks(_cfg(case))["decoder"] == CASES[case][2]


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_decoder_view_keeps_the_cross_cache(case):
    """Readers that size the decoder from `decoder_view` and ask for the cross cache again get the
    same answer: the view is still the encoder-decoder the config states (the nested block's own
    `is_encoder_decoder: false` describes the block, not the model)."""
    cfg = _cfg(case)
    assert _quiet(cross_attention_cache, decoder_view(cfg)) == _quiet(cross_attention_cache, cfg)
    assert encoder_decoder_stacks(decoder_view(cfg)) == encoder_decoder_stacks(cfg)


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_calculator_charges_self_and_cross(case):
    got = _quiet(ModelMemoryCalculator().calculate_kv_cache, _cfg(case), 1, 1024, "bf16") * 1e9
    assert round(got) == CASES[case][4]


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_calculator_breakdown_shows_the_cross_layers(case):
    bd = _quiet(ModelMemoryCalculator().kv_cache_breakdown, _cfg(case), 1, 1024, "bf16")
    assert bd["bytes"] == pytest.approx(CASES[case][4], rel=1e-6)


@pytest.mark.parametrize("case", sorted(CASES))
def test_model_facts_charge_self_and_cross(case):
    from llm_memory_calculator.model.facts import ModelFacts

    facts = _quiet(ModelFacts.from_config, _cfg(case))
    assert round(facts.kv_bytes_per_token()) == CASES[case][3]
    assert round(facts.kv_bytes_for_context(1024)) == CASES[case][4]


@pytest.mark.parametrize("case", sorted(CASES))
def test_genz_charges_self_and_cross(case):
    mc = _quiet(get_configs, _cfg(case))
    assert mc.get_kv_size() * 2 * 1024 + mc.get_request_fixed_bytes(2) == CASES[case][4]


def test_every_fixture_keeps_its_cross_cache_through_the_decoder_view():
    """The property, over every config fixture the suite ships: viewing a config's decoder never
    changes what the one cross-attention reader answers for it."""
    seen = 0
    for path in sorted(FIX.rglob("*.json")):
        try:
            cfg = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(cfg, dict):
            continue
        assert _quiet(cross_attention_cache, decoder_view(cfg)) == _quiet(
            cross_attention_cache, cfg), path.name
        seen += 1
    assert seen > 20


def test_interleaved_cross_layers_say_they_hold_one_image():
    """Mllama's cross-attention states are per image (transformers reshapes them to
    `num_concurrent_media` x tiles x patches); no config states how many images a request carries,
    so the plan charges one image's 4,100 tower tokens and says so, as the docs do."""
    cfg = _load("cross_attention", "mllama_transformers-5.12-default")
    bd = _quiet(ModelMemoryCalculator().kv_cache_breakdown, cfg, 1, 1024, "bf16")
    assert any("one image's 4100 tower tokens" in n for n in bd["notes"])
