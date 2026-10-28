"""The weights beside a decoder, against transformers' own parameter count.

The FRD-027 facts' `weight_bytes()` is what the vLLM planner reserves before it sizes the KV
blocks, so a facts estimate below the checkpoint admits more KV than the GPU holds. For an
encoder-decoder the encoder is read at its OWN geometry (`stack_params.encoder_params`): a
hierarchical tower's stages (Donut's and Nougat's Swin: 20 blocks over four widths, which was
sized as 4 blocks at the decoder's width), and an adapter stack between a tower and the
language model is counted (`stack_params.adapter_params`: BLIP-2's Q-Former and projection).

Every reader of a config's shape takes it from the same reader: a flat mT5 config that states
its two stacks without `is_encoder_decoder` is an encoder-decoder to the parameter count as it
is to the layer plan.

Truth: transformers 5.12 meta-device parameter counts of each fixture's model class.
"""

import copy
import json
import os
import sys

import pytest

from llm_memory_calculator import calculate_memory
from llm_memory_calculator.calculator import ModelMemoryCalculator
from llm_memory_calculator.layer_plan import encoder_decoder_stacks, encoder_view
from llm_memory_calculator.model.facts import ModelFacts

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import kv_corpus  # noqa: E402

#: transformers 5.12 parameter counts of the whole model (its ForConditionalGeneration class).
TRUTH = {
    "cross_attention/naver-clova-ix_donut-base_hub.json": 201_852_024,
    "cross_attention/facebook_nougat-small_hub.json": 247_383_672,
    "cross_attention/blip2_flan-t5-xl_transformers-5.12.json": 3_876_648_448,
    "cross_attention/google_deplot_hub.json": 282_285_696,
    "cross_attention/pix2struct_transformers-5.12-default.json": 282_285_696,
    "cross_attention/Salesforce_blip-image-captioning-base_hub.json": 223_971_644,
    "encoder_decoder/bart_transformers-5.12-default.json": 406_291_456,
    "encoder_decoder/nari-labs_Dia-1.6B-0626.json": 1_592_212_480,
    "encoder_decoder/onnx-community_moonshine-base-ONNX.json": 61_513_920,
    "encoder_decoder/t5_flan-xl-geometry.json": 2_783_959_040,
    # CohereAsrModel: a 48-layer conformer encoder (two FFNs and a convolution module a layer)
    "encoder_decoder/onnx-community_cohere-transcribe-03-2026-ONNX.json": 2_048_854_272,
    "synth:mt5-noflag": 172_119_424,
    "synth:t5-small": 60_506_624,
    "synth:marian-noflag": 74_934_784,
    "synth:vision-enc-dec": 341_028_352,
}


def _config(name):
    if name.startswith("synth:"):
        return copy.deepcopy(kv_corpus.synthetic_configs(FIXTURES)[name[6:]])
    with open(os.path.join(FIXTURES, name)) as f:
        return json.load(f)


@pytest.mark.parametrize("name", sorted(TRUTH))
def test_facts_weights_hold_the_model(name):
    facts = ModelFacts.from_config(_config(name))
    got = facts.weight_bytes() / 2
    assert got >= TRUTH[name], f"{name}: facts {got / 1e6:,.2f}M < {TRUTH[name] / 1e6:,.2f}M"


#: Where the calculator's count stays below the truth, and why (each no lower than origin/main's,
#: and named in docs/LAYER_KINDS.md's "What errs low").
REPORT_BELOW_TRUTH = {
    # the BERT text decoder's bias vectors, its learned positions and its LM head's transform
    # (0.76M of 224M): origin/main 223.11M
    "cross_attention/Salesforce_blip-image-captioning-base_hub.json",
    # the conformer encoder is counted as plain blocks by the calculator (581M of 2.05B):
    # origin/main 302M
    "encoder_decoder/onnx-community_cohere-transcribe-03-2026-ONNX.json",
}

#: The GenZ graph's weights for an encoder-decoder (the encoder at its own geometry: a
#: hierarchical tower's stages and tables, a tower's patch / frame embedding and positions, a
#: nested encoder's own token table and a composite's projection to the decoder's width; the
#: decoder's codebook tables and learned positions), against transformers' weight matrices for
#: the model GenZ graphs (MusicGen without its EnCodec codec).
GRAPH_TRUTH = {
    "cross_attention/naver-clova-ix_donut-base_hub.json": 201_632_760,
    "cross_attention/facebook_nougat-small_hub.json": 247_164_408,
    "cross_attention/google_deplot_hub.json": 282_237_312,
    "cross_attention/facebook_musicgen-small_hub.json": 529_830_272,
    "encoder_decoder/nari-labs_Dia-1.6B-0626.json": 1_611_022_336,
    "encoder_decoder/onnx-community_whisper-large-v3-turbo.json": 808_291_840,
}


@pytest.mark.parametrize("name", sorted(GRAPH_TRUTH))
def test_genz_graph_weights_hold_the_encoder_decoder(name, tmp_path):
    import contextlib
    import io

    for sub in ("model", "sparsity"):
        (tmp_path / sub).mkdir()
    with contextlib.redirect_stdout(io.StringIO()):
        out = kv_corpus.measure_genz(_config(name), str(tmp_path))
    truth = GRAPH_TRUTH[name] * 2 / 2**20
    assert out["genz_prefill_weights_mb"] >= truth * (1 - 5e-5), (
        f"{name}: {out['genz_prefill_weights_mb']:,.2f} MiB < {truth:,.2f} MiB")


@pytest.mark.parametrize("name", sorted(set(TRUTH) - REPORT_BELOW_TRUTH))
def test_reported_parameters_hold_the_model(name):
    rep = calculate_memory(_config(name), batch_size=1, seq_length=4096, precision="bf16")
    assert rep.parameter_count >= TRUTH[name], (
        f"{name}: {rep.parameter_count / 1e6:,.2f}M < {TRUTH[name] / 1e6:,.2f}M")


def test_a_hierarchical_tower_is_read_per_stage():
    from llm_memory_calculator.layer_plan import encoder_stages

    cfg = _config("cross_attention/naver-clova-ix_donut-base_hub.json")
    stages = encoder_stages(cfg["encoder"])
    assert [s["layers"] for s in stages] == [2, 2, 14, 2]
    assert [s["hidden_size"] for s in stages] == [128, 256, 512, 1024]
    assert [s["num_attention_heads"] for s in stages] == [4, 8, 16, 32]
    assert encoder_decoder_stacks(cfg) == {"encoder": 20, "decoder": 4}
    assert encoder_view(cfg)["stages"] == stages


def test_a_flat_encoder_decoder_without_the_flag_is_one_to_every_reader():
    cfg = _config("synth:mt5-noflag")
    assert "is_encoder_decoder" not in cfg
    assert encoder_decoder_stacks(cfg) == {"encoder": 8, "decoder": 8}
    assert ModelMemoryCalculator().detect_model_type(cfg) == "encoder-decoder"
