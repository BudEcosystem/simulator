"""Cross-attention, wherever a config puts it, priced and graphed the same by every reader.

A decoder can attend to another stack's output in two ways a config states:

* an encoder-decoder: EVERY decoder layer holds a cross-attention beside its self-attention
  (Whisper, T5, BART, Dia, Pix2Struct, a VisionEncoderDecoder), whose encoder is either a stack of
  its own or, for Pix2Struct, the image tower beside a decoder-only `text_config`;
* cross-attention layers of their own interleaved with self-attention layers in a decoder-only
  stack (Mllama's `cross_attention_layers`): such a layer holds K / V over the encoder's tokens
  and NO self-attention cache.

Either way the cache is over the encoder's output -- a fixed count a request, stated by the
config where it can be (`max_source_positions`, an encoder block's `max_position_embeddings` or
`seq_len`, an image tower's patches and tiles) -- and `layer_plan.cross_attention_cache` is the one
reader. Before this round the interleaved kind reached only the memory calculator: GenZ, the
BudSimulator converter and the FRD-027 facts graphed Mllama's 8 cross layers as self-attention
(per token over the sequence, nothing over its 4,100 vision tokens); Pix2Struct's `seq_len` was
not read; GenZ's runtime modelling raised on every encoder-decoder (its new operators were not
booked); the decode graph's fit check held only the decoder's weights; a flan-T5 nested under
`text_config` was counted with 12 heads and 12 layers; and flan-T5's gated FFN was graphed as two
matrices.

Truth is hand arithmetic over the configs' own numbers, or transformers' own meta-device
parameter count (stated beside each figure), never the code under test.
"""

import copy
import json
import warnings
from pathlib import Path

import pytest

from llm_memory_calculator.calculator import ModelMemoryCalculator
from llm_memory_calculator.genz.Models.get_language_model import (
    create_full_decode_model,
    create_full_prefill_model,
    get_configs,
)
from llm_memory_calculator.genz.analyse_model import get_model_df, get_summary_table
from llm_memory_calculator.layer_plan import cross_attention_cache

FIX = Path(__file__).parent / "fixtures"
MIB = 1024 * 1024


def _load(sub, name):
    return json.loads((FIX / sub / f"{name}.json").read_text())


def _quiet(f, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*a, **kw)


def _mllama():
    """transformers 5.12's MllamaConfig().to_dict(): 40 text layers of which 8
    (`cross_attention_layers`) are cross-attention; 32 heads, 8 KV heads of 128; the vision tower's
    448 / 14 patches + 1 class token, on 4 tiles = 4,100 tokens."""
    return _load("cross_attention", "mllama_transformers-5.12-default")


def _pix2struct():
    """transformers 5.12's Pix2StructConfig().to_dict(): a 12-layer decoder (`text_config`, 12 heads
    of `d_kv` 64) cross-attending to the image tower (`vision_config`, `seq_len` 4,096 patches)."""
    return _load("cross_attention", "pix2struct_transformers-5.12-default")


def _whisper_large_v3():
    """WhisperConfig for large-v3 (32 + 32 layers of 1,280, 20 heads, 1,500 encoder frames)."""
    return dict(model_type="whisper", is_encoder_decoder=True, d_model=1280, encoder_layers=32,
                decoder_layers=32, encoder_attention_heads=20, decoder_attention_heads=20,
                encoder_ffn_dim=5120, decoder_ffn_dim=5120, num_hidden_layers=32,
                max_source_positions=1500, max_target_positions=448, vocab_size=51866,
                num_mel_bins=128)


# Self + cross KV of one 1,024-token request at 2 bytes, by hand:
#   Mllama: 32 self layers x 2 x 8 x 128 x 2 B x 1,024 + 8 cross layers x 2 x 8 x 128 x 2 B x 4,100
#   Pix2Struct: 12 x 2 x 12 x 64 x 2 B x 1,024 (self) + 12 x 2 x 12 x 64 x 2 B x 4,096 (cross)
TRUTH_1K = {
    "mllama": 32 * 2 * 8 * 128 * 2 * 1024 + 8 * 2 * 8 * 128 * 2 * 4100,  # 268,566,528
    "pix2struct": 12 * 2 * 12 * 64 * 2 * 1024 + 12 * 2 * 12 * 64 * 2 * 4096,  # 188,743,680
}
CASES = {"mllama": _mllama, "pix2struct": _pix2struct}


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_table_reads_the_cross_cache_wherever_the_config_puts_it(case):
    cross = _quiet(cross_attention_cache, CASES[case]())
    expect = {"mllama": (8, 8, 128, 4100), "pix2struct": (12, 12, 64, 4096)}[case]
    assert (cross["layers"], cross["kv_heads"], cross["head_dim"], cross["tokens"]) == expect


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_calculator_charges_self_and_cross(case):
    got = _quiet(ModelMemoryCalculator().calculate_kv_cache, CASES[case](), 1, 1024, "bf16") * 1e9
    assert round(got) == TRUTH_1K[case]


@pytest.mark.parametrize("case", sorted(CASES))
def test_genz_charges_self_and_cross(case):
    mc = _quiet(get_configs, CASES[case]())
    assert mc.get_kv_size() * 2 * 1024 + mc.get_request_fixed_bytes(2) == TRUTH_1K[case]


@pytest.mark.parametrize("case", sorted(CASES))
def test_model_facts_charge_self_and_cross(case):
    from llm_memory_calculator.model.facts import ModelFacts

    facts = _quiet(ModelFacts.from_config, CASES[case]())
    assert round(facts.kv_bytes_for_context(1024)) == TRUTH_1K[case]


def test_genz_graphs_an_interleaved_cross_layer_without_a_self_cache(tmp_path):
    """Mllama's 8 cross layers are graphed as cross-attention (no self K / V), the other 32 as
    self-attention: the decode graph's cache at 1,024 tokens is the 32 self layers over the
    context plus the 8 cross layers over the 4,100 vision tokens."""
    mc = _quiet(get_configs, _mllama())
    kinds = [lc.attention_type for lc in mc.layer_configs]
    assert kinds.count("Cross") == 8 and kinds.count("MHA-global") == 32
    path = _quiet(create_full_decode_model, name=mc, input_sequence_length=1024,
                  data_path=str(tmp_path))
    st = get_summary_table(get_model_df(path, batch_size=1, data_path=str(tmp_path),
                                        model_characterstics=True), model_characterstics=True)
    assert st["KV Cache (MB)"].values[0] * MIB == pytest.approx(TRUTH_1K["mllama"], rel=0.01)


# --- GenZ runtime modelling over every cross-attention shape ----------------------------------

SYSTEM = {"Flops": 400, "Memory_size": 200, "Memory_BW": 2000, "ICN": 300, "real_values": True}
RUNTIME_CASES = {
    "whisper-large-v3-turbo": lambda: _load("encoder_decoder", "onnx-community_whisper-large-v3-turbo"),
    "dia-1.6b": lambda: _load("encoder_decoder", "nari-labs_Dia-1.6B-0626"),
    "flan-t5-xl": lambda: _load("encoder_decoder", "t5_flan-xl-geometry"),
    "bart": lambda: _load("encoder_decoder", "bart_transformers-5.12-default"),
    "mllama": _mllama,
    "pix2struct": _pix2struct,
}


@pytest.mark.parametrize("case", sorted(RUNTIME_CASES))
@pytest.mark.parametrize("stage", ["prefill", "decode", "chunked"])
def test_runtime_modelling_runs_every_cross_attention_shape(case, stage):
    """Every operator the graphs emit is booked by `get_runtime_breakdown`: GenZ's prefill, decode
    and chunked runtime modelling (and `estimate_*_performance` above them) answer for an
    encoder-decoder and for interleaved cross layers, as they do on origin/main."""
    from llm_memory_calculator.genz.LLM_inference import (
        chunked_moddeling,
        decode_moddeling,
        prefill_moddeling,
    )

    mc = _quiet(get_configs, RUNTIME_CASES[case]())
    if stage == "prefill":
        out = _quiet(prefill_moddeling, model=mc, batch_size=1, input_tokens=256,
                     system_name=SYSTEM, bits="bf16")
    elif stage == "decode":
        out = _quiet(decode_moddeling, model=mc, batch_size=1, input_tokens=256, output_tokens=32,
                     system_name=SYSTEM, bits="bf16")
    else:
        out = _quiet(chunked_moddeling, model=mc, prefill_kv_sizes=[(0, 256)],
                     decode_kv_sizes=[300], system_name=SYSTEM, bits="bf16")
    assert out.Latency > 0
    assert out.Runtime_breakdown.MHA > 0


# --- the decode fit check holds the resident model --------------------------------------------

def _weights_mib(builder, mc, tmp_path, without_input_embedding=False, **kw):
    path = _quiet(builder, name=mc, data_path=str(tmp_path), **kw)
    df = get_model_df(path, batch_size=1, data_path=str(tmp_path), model_characterstics=True)
    st = get_summary_table(df, model_characterstics=True)
    weights = float(st["Total Weights (MB)"].values[0])
    if without_input_embedding:
        weights -= float(df.loc[df["Layer Name"] == "embeddings", "Input_w (MB)"].sum())
    return weights


@pytest.mark.parametrize("case", ["whisper-large-v3", "flan-t5-xl", "mllama"])
def test_the_decode_graph_weighs_the_resident_model(case, tmp_path):
    """A decode step streams the decoder alone, but the device holds the whole model: the decode
    graph's `Total Weights` -- what `decode_moddeling`'s fit check and
    `platform_size.get_minimum_system_size` add to the KV -- is the prefill graph's, encoder and
    cross K / V projections included (held, at no step time). It had been the decoder's: Whisper
    large-v3 1,526.6 MiB of its ~2,942 MiB. (The input embedding table is the one exception, for
    every model alike: no GenZ decode graph has carried it.)"""
    cfg = {"whisper-large-v3": _whisper_large_v3,
           "flan-t5-xl": lambda: _load("encoder_decoder", "t5_flan-xl-geometry"),
           "mllama": _mllama}[case]()
    mc = _quiet(get_configs, cfg)
    decode = _weights_mib(create_full_decode_model, mc, tmp_path, input_sequence_length=448)
    prefill = _weights_mib(create_full_prefill_model, mc, tmp_path, without_input_embedding=True,
                           input_sequence_length=448)
    assert decode == pytest.approx(prefill, rel=0.001)


def test_a_decode_step_that_does_not_fit_is_refused():
    """Whisper large-v3 (1,543,490,560 parameters, transformers' meta count: ~2,944 MiB in bf16)
    on a 3 GB device: the decode fit check refuses it as the prefill check does."""
    from llm_memory_calculator.genz.LLM_inference import decode_moddeling

    mc = _quiet(get_configs, _whisper_large_v3())
    small = dict(SYSTEM, Memory_size=3)
    with pytest.raises(ValueError, match="would not fit"):
        _quiet(decode_moddeling, model=mc, batch_size=1, input_tokens=448, output_tokens=64,
               system_name=small, bits="bf16")


# --- the parameter counter and the gated FFN --------------------------------------------------

def test_a_nested_flan_t5_is_counted_with_its_own_heads_and_both_stacks():
    """Blip2Config(text_config=T5Config(d_model=2048, d_kv=64, num_heads=32, num_layers=24,
    feed_forward_proj='gated-gelu')).to_dict(): transformers' meta-device count is 3,876,648,448
    (the flan-T5 2,783,959,040 of it). origin/main counted 1,720,216,064, the previous head
    1,595,173,376 (12 heads of 64, 12 layers, one stack). The counter reads T5's `num_heads` beside
    its `d_kv`, both stacks, and the decoder's cross-attention; the Q-Former it does not count
    (107M) is the rest of the gap."""
    from llm_memory_calculator import calculate_memory

    cfg = _load("cross_attention", "blip2_flan-t5-xl_transformers-5.12")
    got = _quiet(calculate_memory, cfg, batch_size=1, seq_length=1024, precision="bf16").parameter_count
    assert 0.95 * 3_876_648_448 <= got <= 1.01 * 3_876_648_448


def test_the_attention_head_count_is_read_with_the_head_width():
    """T5's `num_heads` and `d_kv`: one reader for both, so a T5-spelled block is 32 heads of 64."""
    from llm_memory_calculator.parameter_counter import UniversalParameterCounter

    t5 = _load("cross_attention", "blip2_flan-t5-xl_transformers-5.12")["text_config"]
    got = UniversalParameterCounter()._calculate_attention_params(t5, 24, 2048)
    assert got == 24 * 4 * 2048 * 2048


@pytest.mark.parametrize("spelling", [{"feed_forward_proj": "gated-gelu"}, {"is_gated_act": True}])
def test_a_stated_gated_ffn_is_three_matrices_everywhere(spelling, tmp_path):
    """flan-T5 states its gated FFN as `feed_forward_proj: gated-gelu` (hub configs) and, saved by
    transformers >= 4.25, `is_gated_act`. The fixture's GenZ prefill graph then weighs the model
    at transformers' meta count, 2,783,959,040 parameters (encoder 1.158B, decoder 1.560B, shared
    embedding), plus the 65,798,144-parameter classifier GenZ graphs apart from the tied input
    table (as for every tied model) -- not 2.35B with a 2-matrix FFN."""
    from llm_memory_calculator.parameter_counter import UniversalParameterCounter

    cfg = _load("encoder_decoder", "t5_flan-xl-geometry")
    for k in ("feed_forward_proj", "is_gated_act", "dense_act_fn"):
        cfg.pop(k, None)
    cfg.update(spelling)
    assert UniversalParameterCounter()._is_gated_ffn(cfg)
    mc = _quiet(get_configs, cfg)
    weights = _weights_mib(create_full_prefill_model, mc, tmp_path, input_sequence_length=512)
    assert weights * MIB == pytest.approx((2_783_959_040 + 32128 * 2048) * 2, rel=0.005)


@pytest.mark.parametrize("case,truth", [("mllama", 10_642_941_475), ("pix2struct", 282_285_696)])
def test_a_config_written_by_to_dict_is_counted_not_defaulted(case, truth):
    """transformers' `to_dict()` writes `architectures: null`; the counter iterated it, raised, and
    fell back to a 123,532,032-parameter default for both MllamaConfig() and Pix2StructConfig()
    (on origin/main too). Truth: transformers' meta-device counts. Pix2Struct's encoder is its
    vision tower, counted once (not again as a copy of the decoder)."""
    from llm_memory_calculator import calculate_memory

    got = _quiet(calculate_memory, CASES[case](), batch_size=1, seq_length=1024,
                 precision="bf16").parameter_count
    assert got == pytest.approx(truth, rel=0.10)
