"""Round 20: a model type the shipped snapshot knows takes no geometry from the installed
transformers, and the reader inventory follows private helpers and counts a function measured only
when the harness itself calls it.

* ``layer_plan.class_geometry`` asked the installed transformers for any model type the snapshot
  (``config_class_defaults.json``, transformers 5.12) held no top-level geometry for. Qwen2-VL's and
  Qwen2.5-VL's 5.12 classes keep the language model's geometry in `text_config` and build none at
  the top; transformers 4.50's flat Qwen2VLConfig filled its 72B defaults (80 layers, 8,192 wide,
  8 KV heads) there, and the readers took them over the stated decoder (28 x 1,536): the figures
  depended on the installed release, which the snapshot is there to prevent.
* The inventory (``kv_corpus.figure_readers``) named a public function only when its own body
  named a shape key: one reading the shape through a private helper was never required. It now
  follows private helpers at any depth, reads a wider key set (the parameter-count keys, the KV
  bytes a token, every ModelConfig shape field, the ModelConfig itself), and ``ran_code`` counts a
  function measured only when the harness's own source calls it -- one entered from another
  function of the repository is a helper, named in the exclusion list with the functions it
  returns its figure to, which ``kv_corpus.helper_faults`` checks against the source.
"""

import copy
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import kv_corpus  # noqa: E402

#: transformers 4.50's flat Qwen2VLConfig: the 72B model's geometry at the top level.
FLAT_QWEN2_VL = dict(model_type="qwen2_vl", num_hidden_layers=80, hidden_size=8192,
                     num_attention_heads=64, num_key_value_heads=8, intermediate_size=29568,
                     vocab_size=152064)
#: A model type no 5.12 class has (a release newer than the snapshot).
NEWER = dict(model_type="brand_new_lm", num_hidden_layers=12, hidden_size=768,
             num_attention_heads=12, num_key_value_heads=12, intermediate_size=3072,
             vocab_size=32000)


@pytest.fixture
def installed_transformers(monkeypatch):
    """An installed transformers whose classes are the flat Qwen2VLConfig above and a class the
    snapshot does not know."""
    from llm_memory_calculator import layer_plan

    def config_class(defaults):
        class Config:
            def to_dict(self):
                return dict(defaults)

            def __getattr__(self, name):
                if name in defaults:
                    return defaults[name]
                raise AttributeError(name)
        return Config

    fake = types.ModuleType("transformers")
    fake.__version__ = "4.50.3"
    fake.CONFIG_MAPPING = {"qwen2_vl": config_class(FLAT_QWEN2_VL),
                           "brand_new_lm": config_class(NEWER)}
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setattr(layer_plan, "_CLASS_GEOMETRY_CACHE", {})
    monkeypatch.setattr(layer_plan, "_CLASS_LAYOUT_CACHE", {})
    yield


def test_a_type_the_snapshot_knows_takes_no_installed_geometry(installed_transformers):
    from llm_memory_calculator import layer_plan

    assert layer_plan.class_geometry("qwen2_vl") == {}
    # a type the snapshot does not know at all is still read from the installed release
    assert layer_plan.class_geometry("brand_new_lm")["num_hidden_layers"] == 12


def test_qwen2_vl_reads_its_stated_decoder_whatever_release_is_installed(installed_transformers):
    """The synthetic Qwen2-VL-2B shape (28 layers, 1,536 wide, 2 KV heads of 128, the window from
    layer 8 on): GenZ's reading of it gives 14,336 K+V elements a token under transformers 5.12;
    under the flat 4.50 class it gave 163,840 (80 x 8,192 / 64 x 8 x 2)."""
    from llm_memory_calculator.genz.Models.get_language_model import get_configs
    from llm_memory_calculator.layer_plan import with_class_geometry

    cfg = copy.deepcopy(kv_corpus.synthetic_configs(FIXTURES)["qwen2-vl-window-from-layer-8"])
    filled = with_class_geometry(copy.deepcopy(cfg))
    assert "num_hidden_layers" not in filled and "hidden_size" not in filled
    mc = get_configs(copy.deepcopy(cfg))
    assert (mc.num_decoder_layers, mc.hidden_size, mc.num_key_value_heads) == (28, 1536, 2)
    assert mc.get_kv_size() == 14336


# ------------------------------------------------------------------------- the reader inventory

def _write(tmp_path, files):
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def test_the_inventory_follows_private_helpers_and_reads_the_wider_keys(tmp_path):
    _write(tmp_path, {"llm-memory-calculator/src/pkg/m.py": (
        "def _width(cfg):\n    return cfg['hidden_size']\n"
        "def _twice(cfg):\n    return 2 * _width(cfg)\n"
        "def stored(cfg):\n    return _twice(cfg)\n"
        "def listed(meta):\n    return meta['parameter_count']\n"
        "def run(model_config):\n    return 1\n"
        "def name_only(x):\n    return x['name']\n"
        "def _mixer(cfg):\n    return cfg['d_state']\n"
        "_TABLE = {'m': _mixer}\n"
        "def dispatch(kind, cfg):\n    return _TABLE[kind](cfg)\n")})
    readers = {r["function"]: r for r in kv_corpus.figure_readers(str(tmp_path))}
    assert {"stored", "listed", "run", "_width", "_twice", "_mixer", "dispatch"} <= set(readers)
    assert "name_only" not in readers
    assert readers["stored"]["public"] and readers["stored"]["via"]
    assert not readers["_width"]["public"]


def test_a_function_is_measured_only_when_the_harness_calls_it(tmp_path):
    """``ran_code`` counts what the harness's own source calls (``functions``): a method that
    function calls in turn is entered, not measured."""
    src = tmp_path / "probe_mod_r20.py"
    src.write_text("class C:\n    def m(self):\n        return 1\n"
                   "def f():\n    return C().m()\ndef g():\n    return 2\n")
    sys.path.insert(0, str(tmp_path))
    try:
        import importlib

        mod = importlib.import_module("probe_mod_r20")
        rec = kv_corpus.ran_code(str(tmp_path))
        rec._kind[__file__] = "harness"  # this test calls as the harness does
        with rec:
            mod.f()
        mod.g()
    finally:
        sys.path.remove(str(tmp_path))
    assert ("probe_mod_r20.py", "f") in rec.functions
    assert ("probe_mod_r20.py", "C.m") not in rec.functions
    assert ("probe_mod_r20.py", "C.m") in rec.entered
    assert ("probe_mod_r20.py", "g") not in rec.entered


def test_a_helper_entry_is_checked_against_the_source(tmp_path):
    """A helper the exclusion list names holds only while every function that uses it is
    measured, excluded or such a helper itself, and its ``used_by`` names exactly them."""
    import json

    _write(tmp_path, {"llm-memory-calculator/src/pkg/m.py": (
        "def width(cfg):\n    return cfg['hidden_size']\n"
        "def top(cfg):\n    return width(cfg) * 2\n"
        "def other(cfg):\n    return width(cfg) + cfg['vocab_size']\n")})
    rel = os.path.join("llm-memory-calculator", "src", "pkg", "m.py")
    kv_corpus._FUNCTIONS.pop(str(tmp_path), None)
    kv_corpus._READERS.pop(str(tmp_path), None)

    def faults(entries):
        path = tmp_path / "unmeasured.json"
        path.write_text(json.dumps({"entries": entries}))
        return kv_corpus.helper_faults(str(tmp_path), {(rel, "top")}, str(path))

    helper = {"file": rel, "function": "width", "helper": True, "reason": "r" * 40,
              "used_by": [f"{rel}:other", f"{rel}:top"]}
    assert any("does not hold" in f for f in faults([helper]))
    other = {"file": rel, "function": "other", "reason": "no memory figure " * 4}
    assert faults([helper, other]) == []
    stale = dict(helper, used_by=[f"{rel}:top"])
    assert any("used_by says" in f for f in faults([stale, other]))
