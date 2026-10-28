"""The KV gate's ground truth (scripts/kv_truth.py) reads what the model caches, or says it cannot.

A truth of 0 KV lets the gate accept any figure, so a softmax attention whose K / V projections
the truth does not recognise must leave the width undetermined (no truth: the entry fails), never
read as 0; and the decoder's layer list is a list of blocks, never a list of output heads
(MusicGen's `lm_heads`). The figure rules hold each GenZ graph to the weights it graphs.
"""

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import kv_corpus  # noqa: E402
import kv_truth  # noqa: E402

MIB = 2 ** 20


def _built():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not str(transformers.__version__).startswith("5.12"):
        pytest.skip(f"the truths are transformers 5.12's; {transformers.__version__} installed")
    return torch


def _synth(name):
    return kv_corpus.synthetic_configs(FIXTURES)[name]


@pytest.mark.parametrize("config,layers,width", [
    # ProphetNet's ngram self-attention projects K and V with `key_proj` / `value_proj`
    (lambda: _synth("prophetnet"), 12, 2 * 1024),
    # MusicGen's decoder: 24 layers of 16 x 64 K and V; its 4 `lm_heads` are no layer list
    (lambda: _synth("musicgen-small"), 24, 2 * 1024),
    (lambda: json.load(open(os.path.join(FIXTURES, "cross_attention",
                                         "facebook_musicgen-small_hub.json"))), 24, 2 * 1024),
])
def test_the_truth_reads_every_decoder_layers_self_attention_kv(config, layers, width):
    _built()
    prof = kv_truth.truth_profile(config())
    assert "error" not in prof and "kv_error" not in prof, prof
    assert len(prof["kv"]) == layers
    assert kv_truth.kv_elements(prof) == layers * width


def test_an_attention_whose_projections_are_not_recognised_has_no_width():
    """Not 0: a width the truth cannot read leaves the figure without a truth."""
    torch = _built()

    class SelfAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.query_thing = torch.nn.Linear(8, 8)
            self.key_thing = torch.nn.Linear(8, 8)
            self.value_thing = torch.nn.Linear(8, 8)

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = SelfAttention()
            self.mlp = torch.nn.Linear(8, 8)

    assert kv_truth._layer_kv_width(Layer(), 8) is None


def test_a_layer_without_softmax_attention_caches_nothing():
    torch = _built()

    class Mixer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj = torch.nn.Linear(8, 16)

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mixer = Mixer()

    assert kv_truth._layer_kv_width(Layer(), 8) == 0


def test_the_decoder_layer_list_is_a_list_of_blocks():
    torch = _built()

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 4)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = torch.nn.Module()
            self.decoder.layers = torch.nn.ModuleList([Block() for _ in range(3)])
            self.decoder_heads = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(2)])

    m = Model()
    assert kv_truth._decoder_layers(m, 0, decoder_only=True) is m.decoder.layers


PROFILE = {"params": 1000 * MIB, "lm_params": 1000 * MIB, "lm_matrix_params": 900 * MIB,
           "input_embedding": 100 * MIB, "gathered": 104 * MIB, "kv": [["full", None, 8]],
           "class": "X"}


@pytest.mark.parametrize("reader,expected", [
    # the prefill and chunked graphs both look tokens up in, and hold, the input embedding
    ("genz_prefill_weights_mb", 2 * 900),
    ("runtime_prefill[1]", 2 * 900),
    ("runtime_chunked[1]", 2 * 900),
    # a decode step's graph gathers a row from each embedding table (the token embedding, a
    # learned position table) and graphs no operator for it
    ("genz_decode_weights_mb", 2 * 796),
    ("runtime_decode[1]", 2 * 796),
])
def test_each_graph_is_held_to_the_weights_it_graphs(reader, expected):
    truth, _ = kv_truth.figure_truth(PROFILE, reader)
    assert truth == expected


def _hand_truths():
    with open(os.path.join(FIXTURES, "kv_gate", "truths.json")) as f:
        return json.load(f)["profiles"]


def test_a_hand_written_parameter_count_is_arithmetic_or_tensors_not_a_model_card():
    """A truth is transformers' model, a checkpoint's tensors, or the config arithmetic the entry
    states -- never a model card's rounded total, which can sit below the real count."""
    bad = []
    for name, spec in _hand_truths().items():
        if spec.get("base") == "transformers" or "params" not in spec:
            continue
        src = spec.get("source", "").lower()
        if "model card" in src or "hub card" in src:
            bad.append(f"{name}: {spec['source'][:120]}")
        elif not any(w in src for w in ("transformers 5.12", "arithmetic", "tensors",
                                        "checkpoint")):
            bad.append(f"{name}: names neither transformers, arithmetic nor tensors")
    assert not bad, "\n".join(bad)


def test_a_decode_truth_leaves_out_every_table_a_step_gathers_from():
    """Nougat's MBart decoder holds a learned position table (3,586 x 1,024) beside its token
    embedding: a decode step gathers one row of each, as GenZ's decode graph models it."""
    _built()
    cfg = json.load(open(os.path.join(FIXTURES, "cross_attention", "facebook_nougat-small_hub.json")))
    prof = kv_truth.truth_profile(cfg)
    assert prof["gathered"] == prof["input_embedding"] + 3586 * 1024


def test_a_layer_the_class_builds_as_a_mixer_where_the_cache_expects_attention_has_no_truth():
    """LFM2's class reads `full_attention` and builds every other layer name as a short
    convolution, so a `sliding_attention` layer (LFM2.5-Audio's detokenizer) is built without
    the attention its checkpoint holds: the truth is not transformers' (width None), never 0."""
    _built()
    cfg = dict(model_type="lfm2", hidden_size=512, num_attention_heads=16, num_key_value_heads=8,
               num_hidden_layers=4, intermediate_size=2048, vocab_size=1000, sliding_window=30,
               layer_types=["conv", "sliding_attention", "conv", "full_attention"])
    prof = kv_truth.truth_profile(cfg)
    assert prof["kv"][1][2] is None
    assert prof["kv"][3][2] == 2 * 8 * 32
    assert kv_truth.kv_elements(prof) is None
