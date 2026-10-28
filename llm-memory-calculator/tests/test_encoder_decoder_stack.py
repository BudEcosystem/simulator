"""An encoder-decoder's KV cache is its DECODER's: the decoder's depth, heads and head width,
read under each family's own spellings (`layer_plan.decoder_view`), by the calculator and GenZ
alike (docs/LAYER_KINDS.md, "Encoder-decoder models").

Fixtures (tests/fixtures/encoder_decoder): four hub configs from the HuggingFace cache (Dia's
nested `decoder_config`, Whisper large-v3-turbo's 32 encoder and 4 decoder layers, Moonshine's
`decoder_num_*` keys, cohere_asr's top-level decoder beside a nested encoder) and two saved by
transformers 5.12's own config classes (flan-t5-xl's geometry with T5's `num_layers` /
`num_decoder_layers` / `num_heads` / `d_kv`, and BART's defaults, which write no
`num_hidden_layers`).
"""

import json
import warnings
from pathlib import Path

import pytest

from llm_memory_calculator.calculator import ModelMemoryCalculator
from llm_memory_calculator.genz.Models.get_language_model import get_configs
from llm_memory_calculator.layer_plan import decoder_view, encoder_decoder_stacks

FIX = Path(__file__).parent / "fixtures"


def _cfg(name):
    return json.loads((FIX / "encoder_decoder" / f"{name}.json").read_text())


def _asymmetric_t5():
    c = _cfg("t5_flan-xl-geometry")
    c.update(num_layers=12, num_decoder_layers=4)
    return c


def _unnamed_t5():
    """As a config class's `to_dict` writes it: `architectures: null`."""
    c = _cfg("t5_flan-xl-geometry")
    c["architectures"] = None
    return c


def _asymmetric_moonshine():
    c = _cfg("onnx-community_moonshine-base-ONNX")
    c.update(decoder_num_hidden_layers=4, encoder_num_hidden_layers=6, num_hidden_layers=6)
    return c


# name -> (config, encoder layers, decoder layers, decoder width, K+V elements a token adds)
CASES = {
    "dia": (lambda: _cfg("nari-labs_Dia-1.6B-0626"), 12, 18, 2048, 2 * 18 * 4 * 128),
    "whisper-turbo": (lambda: _cfg("onnx-community_whisper-large-v3-turbo"), 32, 4, 1280,
                      2 * 4 * 20 * 64),
    "moonshine": (lambda: _cfg("onnx-community_moonshine-base-ONNX"), 8, 8, 416, 2 * 8 * 8 * 52),
    "moonshine-asymmetric": (_asymmetric_moonshine, 6, 4, 416, 2 * 4 * 8 * 52),
    "cohere-asr": (lambda: _cfg("onnx-community_cohere-transcribe-03-2026-ONNX"), 48, 8, 1024,
                   2 * 8 * 8 * 128),
    "flan-t5-xl": (lambda: _cfg("t5_flan-xl-geometry"), 24, 24, 2048, 2 * 24 * 32 * 64),
    "t5-asymmetric": (_asymmetric_t5, 12, 4, 2048, 2 * 4 * 32 * 64),
    "t5-unnamed": (_unnamed_t5, 24, 24, 2048, 2 * 24 * 32 * 64),
    "bart": (lambda: _cfg("bart_transformers-5.12-default"), 12, 12, 1024, 2 * 12 * 16 * 64),
}


def _quiet(f, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*a, **kw)


@pytest.mark.parametrize("case", sorted(CASES))
def test_each_stack_takes_its_own_depth(case):
    make, enc, dec, _, _ = CASES[case]
    assert encoder_decoder_stacks(make()) == {"encoder": enc, "decoder": dec}


@pytest.mark.parametrize("case", sorted(CASES))
def test_genz_graphs_the_decoder_an_encoder_decoder_config_states(case):
    """GenZ refused T5, BART and Whisper (their width is `d_model` and they carry nested
    blocks), and took no encoder depth from any of them."""
    make, enc, dec, width, kv = CASES[case]
    mc = _quiet(get_configs, make())
    assert (mc.num_encoder_layers, mc.num_decoder_layers, mc.hidden_size) == (enc, dec, width)
    assert mc.get_kv_size() == kv


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_calculator_charges_the_decoder_s_kv(case):
    """The calculator charged Whisper, BART and Dia no KV at all (no `num_attention_heads`
    where it looked) and Whisper's decoder the encoder's 32 layers."""
    make, _, dec, _, kv = CASES[case]
    calc = ModelMemoryCalculator()
    seq = 4096
    got = _quiet(calc.calculate_kv_cache, make(), 1, seq, "bf16")
    assert got == pytest.approx(kv * seq * 2 / 1e9)
    bd = _quiet(calc.kv_cache_breakdown, make(), 1, seq, "bf16")
    assert bd["num_layers"] == bd["num_full_layers"] == dec


def test_a_decoder_only_config_is_read_as_it_stands():
    for path in sorted((FIX / "hybrid_configs").glob("*.json")):
        c = json.loads(path.read_text())
        assert encoder_decoder_stacks(c) is None, path.name
        assert decoder_view(c) is c


def test_the_view_is_stable():
    for make, *_ in CASES.values():
        once = decoder_view(make())
        assert decoder_view(once) == once


@pytest.mark.parametrize("case", sorted(CASES))
def test_model_facts_charge_the_decoder_s_kv(case):
    """FRD-027's facts (the planner's KV per token) read Whisper's 32 encoder layers as the
    decoder and T5's head count as a default (16 for flan-t5-xl's 32 heads: half its KV)."""
    from llm_memory_calculator.model.facts import ModelFacts

    make, _, dec, _, kv = CASES[case]
    facts = _quiet(ModelFacts.from_config, make())
    assert facts.layers.full == dec
    assert facts.kv_bytes_per_token("bf16") == kv * 2


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_encoder_decoder_report_charges_the_decoder_s_own_geometry(case):
    """`calculate_memory` sizes an encoder-decoder from its own getters (decoder self-attention,
    cross-attention, encoder). They read Whisper's spellings only: flan-t5-xl's decoder was 4
    layers of 20 heads (a Whisper default) against 24 of 32, Dia's 4 of 20 against 18 of 4 KV
    heads -- 4.4x and 1.8x under."""
    make, enc, dec, _, kv = CASES[case]
    calc = ModelMemoryCalculator()
    c = make()
    seq = 4096
    assert (calc._get_encoder_layers(c), calc._get_decoder_layers(c)) == (enc, dec)
    assert calc.calculate_decoder_kv_cache(c, 1, seq, "bf16") == pytest.approx(kv * seq * 2 / 1e9)
