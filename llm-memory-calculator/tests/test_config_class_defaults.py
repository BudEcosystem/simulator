"""A config class's defaults are read the same with or without transformers installed.

What a config leaves out is its class's default (Llama4TextConfig's untied head, Mllama's tower
of 8 global layers, RwkvConfig's untied head, NemotronH's relu2). Those defaults used to be read
from the installed transformers only: without it, or with a release that predates the class
(4.50 has no Llama 4 or Nemotron-H), the readers fell back to a tied head and a bare tower and
sized the weights below origin/main's and below the model's own count -- and the KV gate passed
only on transformers 5.12. The defaults now ship with the package
(``config_class_defaults.json``, written from transformers 5.12 by
``scripts/config_class_defaults.py``), so every environment reads the same class.
"""

import importlib
import json
import os
import sys

import pytest

from llm_memory_calculator import layer_plan

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(os.path.dirname(HERE), "scripts")


@pytest.fixture
def no_transformers(monkeypatch):
    """The readers with transformers absent (an import of it fails), and their caches empty."""
    monkeypatch.setitem(sys.modules, "transformers", None)
    layer_plan._CLASS_DEFAULT_CACHE.clear()
    yield
    layer_plan._CLASS_DEFAULT_CACHE.clear()


@pytest.mark.parametrize("model_type,name,expected", [
    ("llama4_text", "tie_word_embeddings", False),
    ("llama4_text", "hidden_act", "silu"),
    ("rwkv", "tie_word_embeddings", False),
    ("mllama_vision_model", "num_global_layers", 8),
    ("mllama_vision_model", "vision_output_dim", 7680),
    ("nemotron_h", "mlp_hidden_act", None),  # not a key any reader asks for: no answer
])
def test_a_class_default_is_read_without_transformers(no_transformers, model_type, name,
                                                      expected):
    assert layer_plan.config_class_default(model_type, name) == expected


def test_every_key_a_reader_asks_for_is_in_the_snapshot():
    keys = set(layer_plan.CLASS_DEFAULT_KEYS)
    asked = set(layer_plan.ACTIVATION_KEYS) | set(layer_plan.ROPE_KEYS) | set(
        layer_plan.TOWER_CLASS_KEYS) | {"tie_word_embeddings", "model_type",
                                    "relative_attention_num_buckets"}
    assert asked <= keys, sorted(asked - keys)
    snap = layer_plan._class_default_snapshot()
    assert set(snap["keys"]) == keys, "re-run scripts/config_class_defaults.py"


def _count(cfg):
    """The parameter count three readers give: the calculator, the engine estimate, the facts."""
    from llm_memory_calculator import calculate_memory
    from llm_memory_calculator.inference_engines import estimate_memory_for_engine
    from llm_memory_calculator.model.facts import ModelFacts

    layer_plan._CLASS_DEFAULT_CACHE.clear()
    report = calculate_memory(json.loads(json.dumps(cfg)), batch_size=1, seq_length=4096,
                              precision="bf16")
    eng = estimate_memory_for_engine(json.loads(json.dumps(cfg)), None, batch_size=1,
                                     seq_length=4096, weight_precision="fp16",
                                     runtime_precision="fp16")
    facts = ModelFacts.from_config(json.loads(json.dumps(cfg)))
    return int(report.parameter_count), int(eng.parameter_count), int(facts.weight_bytes())


def _synthetic(name):
    sys.path.insert(0, SCRIPTS)
    try:
        kv_corpus = importlib.import_module("kv_corpus")
    finally:
        sys.path.remove(SCRIPTS)
    return kv_corpus.synthetic_configs(os.path.join(HERE, "fixtures"))[name]


@pytest.mark.parametrize("name", ["mllama-own-cross-layers", "rwkv", "llama4-scout-text"])
def test_the_weights_do_not_depend_on_transformers_being_installed(monkeypatch, name):
    pytest.importorskip("transformers")
    cfg = _synthetic(name)
    with_tf = _count(cfg)
    monkeypatch.setitem(sys.modules, "transformers", None)
    without_tf = _count(cfg)
    layer_plan._CLASS_DEFAULT_CACHE.clear()
    assert without_tf == with_tf


def test_the_snapshot_is_the_installed_transformers_5_12():
    """The shipped defaults are what transformers 5.12's classes say (checked where it is
    installed; elsewhere the snapshot is all there is)."""
    transformers = pytest.importorskip("transformers")
    if not str(transformers.__version__).startswith("5.12"):
        pytest.skip(f"transformers {transformers.__version__}: the snapshot is 5.12's")
    sys.path.insert(0, SCRIPTS)
    try:
        gen = importlib.import_module("config_class_defaults")
    finally:
        sys.path.remove(SCRIPTS)
    fresh = gen.snapshot()
    shipped = layer_plan._class_default_snapshot()
    assert fresh["defaults"] == shipped["defaults"], "re-run scripts/config_class_defaults.py"
