"""An encoder-decoder's decoder keeps TWO caches a request: its self-attention over the generated
text, and each decoder layer's cross-attention K and V over the encoder's output (transformers'
`EncoderDecoderCache.cross_attention_cache`, vLLM's `CrossAttentionSpec`), computed once and read
at every decode step. `layer_plan.cross_attention_cache` reads the second from the config -- the
decoder's depth, the cross-attention's own K / V width, the encoder's output length -- and every
reader of a request's KV charges it (docs/LAYER_KINDS.md, "Encoder-decoder models").

Before, every reader but `calculate_memory` charged the self-attention alone, so once the decoder
was read correctly (round 8) Whisper's serving admission, the vLLM planner's running sequences
and the prefix figures fell BELOW the truth: whisper-tiny.en was admitted 1,365 1k-token requests
into 8 GiB where self + cross allow ~553. A nested flan-T5 (BLIP-2, InstructBLIP) was read as
its 12-layer Q-Former, and ProphetNet's decoder heads were not read at all.

Truth here is hand arithmetic over the configs' own numbers, never the code under test.
"""

import copy
import json
import os
import warnings
from pathlib import Path

import pytest

from llm_memory_calculator.calculator import ModelMemoryCalculator
from llm_memory_calculator.genz.Models.get_language_model import get_configs
from llm_memory_calculator.layer_plan import encoder_decoder_stacks

FIX = Path(__file__).parent / "fixtures"


def _cfg(name):
    return json.loads((FIX / "encoder_decoder" / f"{name}.json").read_text())


def _quiet(f, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*a, **kw)


def _t5_xl():
    """flan-t5-xl as transformers 5.12's T5Config(d_model=2048, d_kv=64, num_heads=32,
    num_layers=24).to_dict() writes it."""
    return dict(model_type="t5", is_encoder_decoder=True, d_model=2048, d_kv=64, d_ff=5120,
                num_heads=32, num_layers=24, num_decoder_layers=24, vocab_size=32128)


def _blip2(kind="blip-2"):
    """Blip2Config / InstructBlipConfig(text_config=<flan-t5-xl>).to_dict(): the outer config says
    `is_encoder_decoder` and states no stack of its own; the Q-Former is 12 layers of 768."""
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


def _prophetnet():
    """ProphetNetConfig().to_dict()'s spellings: `num_{encoder,decoder}_attention_heads`."""
    return dict(model_type="prophetnet", is_encoder_decoder=True, hidden_size=1024,
                num_encoder_layers=12, num_decoder_layers=12, num_encoder_attention_heads=16,
                num_decoder_attention_heads=16, encoder_ffn_dim=4096, decoder_ffn_dim=4096,
                vocab_size=30522, max_position_embeddings=512)


def _trocr():
    """VisionEncoderDecoderConfig(encoder=ViT 384/16, decoder=TrOCRConfig): the decoder block
    spells its depth `decoder_layers`; the encoder's output is the image's patches."""
    return dict(model_type="vision-encoder-decoder", is_encoder_decoder=True,
                encoder=dict(model_type="vit", hidden_size=768, num_hidden_layers=12,
                             num_attention_heads=12, intermediate_size=3072, image_size=384,
                             patch_size=16),
                decoder=dict(model_type="trocr", d_model=1024, decoder_layers=12,
                             decoder_attention_heads=16, decoder_ffn_dim=4096, vocab_size=50265))


# name -> (config, decoder layers, self-attention K+V elements a token, cross-attention K+V
# elements one encoder token adds over every decoder layer, encoder tokens stated or None)
CASES = {
    "whisper-turbo": (lambda: _cfg("onnx-community_whisper-large-v3-turbo"), 4,
                      2 * 4 * 20 * 64, 2 * 4 * 20 * 64, 1500),
    "dia": (lambda: _cfg("nari-labs_Dia-1.6B-0626"), 18, 2 * 18 * 4 * 128, 2 * 18 * 16 * 128,
            1024),
    "cohere-asr": (lambda: _cfg("onnx-community_cohere-transcribe-03-2026-ONNX"), 8,
                   2 * 8 * 8 * 128, 2 * 8 * 8 * 128, 5000),
    "flan-t5-xl": (lambda: _cfg("t5_flan-xl-geometry"), 24, 2 * 24 * 32 * 64, 2 * 24 * 32 * 64,
                   None),
    "bart": (lambda: _cfg("bart_transformers-5.12-default"), 12, 2 * 12 * 16 * 64,
             2 * 12 * 16 * 64, None),
    "moonshine": (lambda: _cfg("onnx-community_moonshine-base-ONNX"), 8, 2 * 8 * 8 * 52,
                  2 * 8 * 8 * 52, None),
    "blip-2-flan-t5-xl": (_blip2, 24, 2 * 24 * 32 * 64, 2 * 24 * 32 * 64, None),
    "instructblip-flan-t5-xl": (lambda: _blip2("instructblip"), 24, 2 * 24 * 32 * 64,
                                2 * 24 * 32 * 64, None),
    "florence-2": (_florence2, 12, 2 * 12 * 16 * 64, 2 * 12 * 16 * 64, None),
    "prophetnet": (_prophetnet, 12, 2 * 12 * 16 * 64, 2 * 12 * 16 * 64, None),
    "trocr": (_trocr, 12, 2 * 12 * 16 * 64, 2 * 12 * 16 * 64, (384 // 16) ** 2 + 1),
}


def _cross_bytes(case, tokens):
    """One request's cross-attention cache at 2 B: the stated encoder length, else ``tokens``."""
    _, _, _, cross, stated = CASES[case]
    return cross * (stated or tokens) * 2


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_table_reads_the_cross_attention_cache(case):
    from llm_memory_calculator.layer_plan import cross_attention_cache

    make, dec, _, cross, stated = CASES[case]
    c = cross_attention_cache(make())
    assert c["layers"] == dec
    assert c["layers"] * 2 * c["kv_heads"] * c["head_dim"] == cross
    assert c["tokens"] == stated


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_calculator_charges_self_and_cross_attention(case):
    """`calculate_kv_cache`: the decoder's self-attention over the sequence and its cross-attention
    over the encoder's output (unstated: the sequence). ProphetNet was charged 0: its decoder's
    heads (`num_decoder_attention_heads`) were not read. BLIP-2 was read through its Q-Former."""
    make, dec, self_kv, _, _ = CASES[case]
    seq = 4096
    got = _quiet(ModelMemoryCalculator().calculate_kv_cache, make(), 1, seq, "bf16")
    assert got * 1e9 == pytest.approx(self_kv * seq * 2 + _cross_bytes(case, seq))


@pytest.mark.parametrize("case", sorted(CASES))
def test_genz_reads_the_decoder_and_its_cross_cache(case):
    """GenZ's ModelConfig (the serving model, the prefix cache, the disaggregation transfer read
    it; BudSimulator's converter builds the same, tests/test_model_converter_layer_kinds.py): the
    decoder's depth and self-attention, the cross cache a token where the encoder's length is
    unstated, else once a request."""
    make, dec, self_kv, cross, stated = CASES[case]
    mc = _quiet(get_configs, make())
    assert mc.num_decoder_layers == dec
    assert mc.get_kv_size() == self_kv + (0 if stated else cross)
    assert mc.get_cross_kv_bytes(2) == (cross * stated * 2 if stated else 0)
    assert mc.num_encoder_layers == encoder_decoder_stacks(make())["encoder"]


def test_whisper_serving_admits_what_self_and_cross_attention_allow():
    """whisper-large-v3-turbo, a 1,024-token request: 1,024 x 4 x 2 x 1,280 x 2 B of
    self-attention and 4 x 2 x 1,280 x 1,500 x 2 B of cross-attention = 51,691,520 B. Round 8
    charged 20,971,520 (the self-attention alone); origin/main refused the config."""
    from llm_memory_calculator.genz.serving.memory_model import MemoryModel

    mc = _quiet(get_configs, _cfg("onnx-community_whisper-large-v3-turbo"))
    mm = MemoryModel(mc, [], tensor_parallel=1, precision_bytes=2)
    assert mm.request_bytes(1024) == 1024 * 4 * 2 * 1280 * 2 + 4 * 2 * 1280 * 1500 * 2
    assert mm.bytes_per_token_kv == 4 * 2 * 1280 * 2


def test_the_disaggregation_transfer_moves_the_cross_cache():
    """The prefill-to-decode transfer moves the request's KV, its fixed state and its cross cache."""
    mc = _quiet(get_configs, _cfg("onnx-community_whisper-large-v3-turbo"))
    assert mc.get_request_fixed_bytes(2) == 4 * 2 * 1280 * 1500 * 2


@pytest.mark.parametrize("case", ["whisper-turbo", "dia", "flan-t5-xl", "prophetnet"])
def test_model_facts_hold_the_cross_cache_per_sequence(case):
    """The FRD-027 facts: `kv_bytes_for_context` (what a sequence holds and a decode step reads)
    includes the cross cache; `kv_bytes_per_token` stays the self-attention a token adds."""
    from llm_memory_calculator.model.facts import ModelFacts

    make, dec, self_kv, _, _ = CASES[case]
    facts = _quiet(ModelFacts.from_config, make())
    assert facts.layers.full == dec
    assert facts.kv_bytes_per_token("bf16") == self_kv * 2
    assert facts.kv_bytes_for_context(1280, "bf16") == self_kv * 1280 * 2 + _cross_bytes(case, 1280)


def _vllm():
    from llm_memory_calculator.device import catalogue
    from llm_memory_calculator.engine import EngineRecord, engine_for

    rec = json.loads((FIX / "planner" / "records" / "vllm-0.30-upstream.json").read_text())
    return engine_for(EngineRecord.from_dict(rec)), catalogue.get("h100xm-80c-vgpu")


@pytest.mark.parametrize("case", ["dia", "whisper-turbo", "bart"])
def test_the_vllm_planner_holds_each_sequence_s_cross_cache(case):
    """vLLM allocates an encoder-decoder's cross-attention K / V per running sequence beside its
    self-attention blocks. The planner gave Dia 1,024 running 1,280-token sequences on an 80 GB
    card, ignoring 18 x 2 x 16 x 128 x 1,024 x 2 B = 151 MB of cross cache each (origin/main,
    reading the encoder as the decoder: 58)."""
    from llm_memory_calculator.engine.base import Workload
    from llm_memory_calculator.model.facts import ModelFacts

    make, _, self_kv, _, _ = CASES[case]
    engine, device = _vllm()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        facts = ModelFacts.from_config(make())
        lim = engine.limits({}, facts, device)
        p = engine.predict({}, Workload(concurrency=1024, input_tokens=1024, output_tokens=256),
                           facts, device)
    # a text encoder with no stated length is charged the context a request may carry
    per_seq = self_kv * 2 * 1280 + _cross_bytes(case, facts.max_context)
    assert lim.state_bytes_per_sequence == pytest.approx(_cross_bytes(case, facts.max_context))
    capacity = lim.kv_pool_bytes / per_seq
    assert p.running <= capacity
    assert p.running >= min(0.95 * capacity, 1024)


def test_the_kv_planner_geometry_counts_the_decoder_s_layers_and_its_cross_cache():
    """`KVGeometry` took the encoder's 32 layers as Whisper-turbo's KV layers (its per-layer
    segment 8x too small) and had no cross cache. The cache is not a prefix -- no prefix tier
    stores an encoder's output -- so the per-token figure stays the self-attention's."""
    from llm_memory_calculator.kv.geometry import KVGeometry

    g = _quiet(KVGeometry.from_model, _cfg("onnx-community_whisper-large-v3-turbo"),
               seq_length=4096)
    assert g.kv_layers == 4
    assert g.bytes_per_token_total == 4 * 2 * 1280 * 2
    assert g.cross_bytes_per_sequence == 4 * 2 * 1280 * 1500 * 2


def test_the_prefix_cache_prices_a_cached_token_by_the_self_attention():
    """A cached prefix token holds the decoder's self-attention; the cross cache is the
    encoder's output, once a request, not a prefix."""
    from llm_memory_calculator.genz.serving.prefix_cache import PrefixCacheAnalyzer

    pa = PrefixCacheAnalyzer(_quiet(get_configs, _cfg("onnx-community_whisper-large-v3-turbo")))
    assert pa._kv_bytes_per_token == 4 * 2 * 1280 * 2


def test_calculate_memory_prices_the_cross_attention_at_its_own_width():
    """`calculate_memory` sized the cross cache at the ENCODER's width: Dia's is 16 KV heads of
    128 (2,048 wide) against its 1,024-wide encoder -- half. Now the one reader: 151 MB."""
    calc = ModelMemoryCalculator()
    got = calc.calculate_cross_attention_kv_cache(_cfg("nari-labs_Dia-1.6B-0626"), 1, 1024, "bf16")
    assert got * 1e9 == pytest.approx(18 * 2 * 16 * 128 * 1024 * 2)


def _genz_summary(kind, cfg, tmp_path, seq=4096):
    from llm_memory_calculator.genz.analyse_model import get_model_df, get_summary_table
    from llm_memory_calculator.genz.Models.get_language_model import (
        create_full_decode_model,
        create_full_prefill_model,
    )

    for sub in ("model", "sparsity"):
        os.makedirs(tmp_path / sub, exist_ok=True)
    mc = _quiet(get_configs, copy.deepcopy(cfg))
    if kind == "decode":
        path = create_full_decode_model(name=mc, input_sequence_length=seq, output_gen_tokens=0,
                                        data_path=str(tmp_path))
    else:
        path = create_full_prefill_model(name=mc, input_sequence_length=seq,
                                         data_path=str(tmp_path))
    df = _quiet(get_model_df, path, batch_size=1, data_path=str(tmp_path), model_characterstics=True)
    return get_summary_table(df, model_characterstics=True)


def test_the_genz_decode_graph_reads_the_cross_cache(tmp_path):
    """Each decode step reads every decoder layer's cross-attention K / V: whisper-large-v3-turbo
    at 4,096 tokens holds 80 MiB of self-attention and 29.3 MiB of cross-attention."""
    st = _genz_summary("decode", _cfg("onnx-community_whisper-large-v3-turbo"), tmp_path)
    expect = (4 * 2 * 1280 * 4096 * 2 + 4 * 2 * 1280 * 1500 * 2) / 2**20
    assert float(st["KV Cache (MB)"].values[0]) == pytest.approx(expect, rel=1e-6)


@pytest.mark.parametrize("case", ["whisper-turbo", "bart"])
def test_the_genz_prefill_graph_runs_the_encoder(case, tmp_path):
    """An encoder-decoder runs its encoder once a request, at prefill: the prefill graph holds the
    encoder's weights beside the decoder's (whisper-large-v3-turbo: 809M parameters, 32 encoder
    layers of 1,280; BART-large's defaults: 406M). Without it the graph held the decoder alone --
    whisper-turbo's prefill weights were 21% of the model."""
    make = CASES[case][0]
    cfg = make()
    from llm_memory_calculator import calculate_memory

    params = _quiet(calculate_memory, cfg, batch_size=1, seq_length=1024,
                    precision="bf16").parameter_count
    mc = _quiet(get_configs, cfg)
    embedding = mc.vocab_size * mc.hidden_size * 2  # GenZ graphs a tied classifier as its own
    weights = float(_genz_summary("prefill", cfg, tmp_path, 1024)["Total Weights (MB)"].values[0]) * 2**20
    assert 0.97 * params * 2 <= weights <= params * 2 + embedding


def _codec():
    """An audio codec's shape (Qwen3-TTS's speech tokenizer, transformers 4.57): an encoder and a
    decoder nested side by side, no `is_encoder_decoder`, no cross-attention between them."""
    stack = dict(hidden_size=512, num_hidden_layers=8, num_attention_heads=16,
                 num_key_value_heads=16, head_dim=64, intermediate_size=1024,
                 max_position_embeddings=8000)
    return dict(model_type="qwen3_tts_tokenizer_12hz", encoder_config=dict(stack),
                decoder_config=dict(stack))


def test_two_nested_stacks_that_never_meet_hold_no_cross_cache():
    """Two nested stacks make a config an encoder-decoder for its depths (round 8), but only a
    decoder that attends to the encoder holds a cross cache: the config says
    `is_encoder_decoder`, states a decoder stack of its own, or its decoder block says
    `add_cross_attention` / states a cross-attention geometry."""
    from llm_memory_calculator.layer_plan import cross_attention_cache

    assert encoder_decoder_stacks(_codec()) == {"encoder": 8, "decoder": 8}
    assert cross_attention_cache(_codec()) is None
    bd = _quiet(ModelMemoryCalculator().kv_cache_breakdown, _codec(), 1, 4096, "bf16")
    assert bd["num_cross_layers"] == 0
    mc = _quiet(get_configs, _codec())
    assert mc.get_request_fixed_bytes(2) == 0 and mc.encoder is None
    flagged = dict(_codec(), is_encoder_decoder=True)
    assert cross_attention_cache(flagged)["layers"] == 8
