"""THE GATE: no figure of the branch moves in the unsafe direction against origin/main unless the
allowlist names it, with both figures and the ground truth the branch's figure is held to.

The corpus is every config this repository ships: prepopulated.db's rows, every fixture under
tests/fixtures (the hub configs among them, tests/fixtures/gguf's metadata-only GGUFs, read
through the GGUF reader, and tests/fixtures/onnx's ONNX-GenAI folders, read through the ONNX
reader), the harness's synthetic shapes and GenZ's built-in models -- measured by scripts/kv_corpus.py through every reader of a model's memory (the
calculator, the engine estimate, the FRD-027 facts and its vLLM planner, GenZ's graphs and runtime
modelling, the serving and disaggregation paths, the prefix cache, the KV planner and
BudSimulator's converter). tests/fixtures/kv_gate/origin-main.json is the same harness's
measurement of origin/main over the same corpus.

A memory figure (KV, state, weights) below origin/main's, or a capacity figure (vLLM's blocks,
tokens and running sequences, the serving path's requests, the scheduler's peak) above it, fails
unless tests/fixtures/kv_gate/allowlist.json names it -- config, reader, origin/main's and the
branch's figures -- and the branch's figure is at least its ground truth (transformers' own model
for the config, the checkpoint's tensors, or the config arithmetic the entry cites), or, for a
capacity, the memory figures it is sized from are each held to theirs. A figure the branch
reports and origin/main did not is held the same way; a figure origin/main reported and the
branch does not fails unless its entry names the error the branch now records for it. A figure
whose reader cannot import a third-party module it is declared to need
(``kv_corpus.READER_DEPENDENCIES``: the GGUF reader's `gguf`, the converter's `requests`, `lxml`,
`json_repair`) is not measured here, and the test says so; any other missing module fails -- no
reader needs transformers or torch (the config-class defaults ship with the package). The same check runs over
the machine's HF cache and model folders with ``kv_corpus.py --gate``
(docs/proofs/kv-layer-kinds-corpus-diff.md).

Which readers that is, is read from the source: every public function that names a model-shape key
(``kv_corpus.figure_readers``) is entered while the corpus runs (``kv_corpus.ran_code``) or named
with its reason in tests/fixtures/kv_gate/unmeasured.json, else the inventory test fails.
"""

import json
import os
import sys
import warnings

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE = os.path.dirname(HERE)
REPO = os.path.dirname(PACKAGE)
FIXTURES = os.path.join(HERE, "fixtures")
GATE = os.path.join(FIXTURES, "kv_gate")
sys.path.insert(0, os.path.join(PACKAGE, "scripts"))
import kv_corpus  # noqa: E402


def _load(name):
    with open(os.path.join(GATE, name)) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def baseline():
    return [r for r in _load("origin-main.json")["rows"] if not r["name"].startswith("_")]


@pytest.fixture(scope="module")
def allowlist():
    return _load("allowlist.json")


@pytest.fixture(scope="module")
def branch_run(tmp_path_factory):
    """The corpus measured once, in a fresh interpreter, recording every function of the
    repository it enters (``kv_corpus.py --inventory``: ``kv_corpus.ran_code``) for the reader
    inventory. A fresh interpreter, so what the run enters does not depend on what other tests
    imported first (a function a module calls as it is imported runs inside the run or not at
    all), and the rows are what the corpus JSON the allowlist is built from holds."""
    import subprocess

    tmp = tmp_path_factory.mktemp("corpus")
    rows_path, ran_path = str(tmp / "rows.json"), str(tmp / "ran.json")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (os.path.join(PACKAGE, "src"), env.get("PYTHONPATH")) if p)
    subprocess.run(
        [sys.executable, os.path.join(PACKAGE, "scripts", "kv_corpus.py"),
         "--inventory", os.path.join(GATE, "unmeasured.json"),
         "--db", os.path.join(REPO, "BudSimulator", "data", "prepopulated.db"),
         "--fixtures", FIXTURES, "--repo-root", REPO, "--genz-data", str(tmp / "genz"),
         "--out", rows_path, "--ran", ran_path],
        env=env, cwd=str(tmp), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    with open(rows_path) as f:
        rows = json.load(f)
    with open(ran_path) as f:
        ran = {tuple(x) for x in json.load(f)}
    return {"rows": rows, "ran": ran}


@pytest.fixture(scope="module")
def branch_rows(branch_run):
    return branch_run["rows"]


def test_the_branch_measures_every_config_origin_main_did(baseline, branch_rows):
    measured = {r["name"] for r in branch_rows}
    missing = sorted(r["name"] for r in baseline if r["name"] not in measured)
    assert not missing, missing


def test_the_gguf_fixtures_are_read_through_the_gguf_reader(baseline, branch_rows):
    """The committed corpus runs the GGUF reader (bud_component.gguf_to_hf_config with
    gguf_layers.hf_layer_keys_from_gguf) on metadata-only copies of real GGUFs."""
    pytest.importorskip("gguf")
    rows = {r["name"]: r for r in branch_rows}
    ggufs = [r["name"] for r in baseline if r["name"].startswith("gguf:")]
    assert ggufs, "no GGUF fixture in the baseline"
    for name in ggufs:
        assert "gguf_error" not in rows[name], rows[name].get("gguf_error")
        assert rows[name].get("gguf_config_keys"), name


def test_the_onnx_folders_are_read_through_the_onnx_reader(baseline, branch_rows):
    """The committed corpus runs the ONNX-GenAI reader (bud_component.onnx_dir_to_hf_config) on
    tests/fixtures/onnx: a dense folder and a MoE one beside its source config.json are measured
    through every reader; a hybrid folder with no config.json is refused (fail-closed)."""
    rows = {r["name"]: r for r in branch_rows}
    onnx = sorted(r["name"] for r in baseline if r["name"].startswith("onnx:"))
    assert onnx == ["onnx:qwen1.5-moe-a2.7b-genai", "onnx:qwen2.5-0.5b-instruct-genai",
                    "onnx:qwen3.5-hybrid-genai"], onnx
    for name in onnx[:2]:
        assert "onnx_error" not in rows[name], rows[name].get("onnx_error")
        assert rows[name].get("eng_onnxruntime_4096_total_mb"), name
    assert "not a known DENSE" in rows["onnx:qwen3.5-hybrid-genai"]["onnx_error"]


def test_no_unsafe_move_outside_the_allowlist(baseline, branch_rows, allowlist):
    """Every figure is held; a figure whose reader cannot import a third-party module it is
    declared to need (the GGUF reader's `gguf`, the converter's `requests` / `lxml` /
    `json_repair`) is not measured in this environment, and says so -- a reader that raises
    anything else fails."""
    names = [r["name"] for r in baseline]
    skipped = kv_corpus.unavailable_readers(baseline, branch_rows, names)
    if skipped:
        warnings.warn("not measured here (not installed): " + ", ".join(
            f"{m}: {n} figures" for m, n in sorted(skipped.items())))
    failures = kv_corpus.gate(baseline, branch_rows, allowlist, names=names)
    assert not failures, f"{len(failures)} gate failures:\n" + "\n".join(failures[:60])


def test_every_allowlist_entry_names_its_figures_truth_and_cause(allowlist):
    bad = []
    for e in allowlist["entries"]:
        where = f"{e.get('config')} {e.get('reader')}"
        if not (e.get("config") and e.get("reader") and "main" in e and "branch" in e):
            bad.append(f"{where}: config, reader and both figures")
        if not e.get("cause"):
            bad.append(f"{where}: no cause")
        if e.get("change") == "lost":
            if not e.get("error"):
                bad.append(f"{where}: a lost figure names no error")
        elif e.get("kind") == "memory" and kv_corpus.DERIVED_FIGURES.match(e["reader"]) \
                and e.get("held_by"):
            pass  # a tier the planner derives: held by its inputs (gate() checks each)
        elif e.get("kind") == "memory":
            if not isinstance(e.get("truth"), (int, float)) or not e.get("source"):
                bad.append(f"{where}: no ground truth or source")
            elif e["branch"] < e["truth"] * (1 - float(e.get("vectors", 0.0))) - float(
                    e.get("unstated", 0)):
                bad.append(f"{where}: {e['branch']} below its ground truth {e['truth']}")
            elif e.get("unstated") and not e.get("unstated_source"):
                bad.append(f"{where}: an unstated allowance names no source")
        elif not e.get("held_by"):
            bad.append(f"{where}: a capacity entry names no memory figure it follows")
    assert not bad, "\n".join(bad[:60])


def test_the_gate_fails_what_it_should():
    """The gate itself: an unlisted drop fails, a listed one held to its truth passes, a listed
    one below its truth fails, a capacity rise follows its memory figures."""
    main = [{"name": "m", "kv_gb@4096": 1.0, "vllm_running_1024": 10, "facts_weight_bytes": 100}]
    branch = [{"name": "m", "kv_gb@4096": 0.5, "vllm_running_1024": 20, "facts_weight_bytes": 100}]
    assert len(kv_corpus.gate(main, branch, {})) == 2
    entry = {"config": "m", "reader": "kv_gb@4096", "main": 1.0, "branch": 0.5, "truth": 0.4,
             "source": "s", "cause": "c"}
    held = {"config": "m", "reader": "facts_weight_bytes", "main": 100, "branch": 100,
            "truth": 90, "source": "s", "cause": "c"}
    cap = {"config": "m", "reader": "vllm_running_1024", "main": 10, "branch": 20,
           "held_by": ["kv_gb@4096", "facts_weight_bytes"], "cause": "c"}
    assert kv_corpus.gate(main, branch, {"entries": [entry, held, cap]}) == []
    low = dict(entry, truth=0.6)
    assert any("below its ground truth" in f
               for f in kv_corpus.gate(main, branch, {"entries": [low, held, cap]}))
    stale = dict(entry, branch=0.45)
    assert any("allowlist says" in f
               for f in kv_corpus.gate(main, branch, {"entries": [stale, held, cap]}))


def test_the_gate_fails_a_figure_lost_or_new_that_the_allowlist_does_not_hold():
    """A figure origin/main produced that the branch does not (its reader now raises) fails
    unless the allowlist names it with the error the branch records; a figure the branch reports
    that origin/main did not is held to its ground truth like a drop."""
    main = [{"name": "m", "runtime_prefill": [1.0, 100.0, 2.0], "kv_gb@4096": 1.0}]
    branch = [{"name": "m", "runtime_prefill_error": "ValueError: All params would not fit",
               "kv_gb@4096": 1.0, "serving_state_bytes": 50}]
    failures = kv_corpus.gate(main, branch, {})
    assert sum("not in the allowlist" in f for f in failures) == 3, failures
    lost = [{"config": "m", "reader": f"runtime_prefill[{i}]", "main": v, "branch": None,
             "change": "lost", "error": "ValueError: All params would not fit", "cause": "c"}
            for i, v in ((1, 100.0), (2, 2.0))]
    new = {"config": "m", "reader": "serving_state_bytes", "main": None, "branch": 50,
           "change": "new", "kind": "memory", "truth": 40, "source": "s", "cause": "c"}
    assert kv_corpus.gate(main, branch, {"entries": lost + [new]}) == []
    other = [dict(e, error="TypeError") for e in lost]
    assert any("the branch's error is" in f
               for f in kv_corpus.gate(main, branch, {"entries": other + [new]}))
    low = dict(new, truth=60)
    assert any("below its ground truth" in f
               for f in kv_corpus.gate(main, branch, {"entries": lost + [low]}))


def test_a_reader_whose_dependency_is_not_installed_is_not_measured_but_a_crash_fails():
    main = [{"name": "m", "conv_kv_size": 10, "conv_state_bytes": 0}]
    absent = [{"name": "m", "conv_error": "ModuleNotFoundError: No module named 'lxml'"}]
    assert kv_corpus.gate(main, absent, {}) == []
    assert kv_corpus.unavailable_readers(main, absent) == {"lxml": 2}
    own = [{"name": "m", "conv_error": "ModuleNotFoundError: No module named 'llm_memory_calculator.x'"}]
    crash = [{"name": "m", "conv_error": "TypeError: unsupported operand"}]
    assert len(kv_corpus.gate(main, own, {})) == 2
    assert len(kv_corpus.gate(main, crash, {})) == 2


def test_only_a_reader_s_declared_dependency_may_be_missing():
    """A reader may lack only the third-party modules it is declared to need
    (``kv_corpus.READER_DEPENDENCIES``: the GGUF reader's `gguf`, the converter's `requests`,
    `lxml`, `json_repair`). Any other missing module fails -- transformers and torch above all:
    no reader needs them, so a reader that came to depend on one must fail where it is absent,
    not pass as "not measured" (round 13 accepted any module outside the repository)."""
    main = [{"name": "m", "conv_kv_size": 10, "kv_gb@4096": 1.0, "facts_weight_bytes": 100}]
    for module in ("transformers", "torch", "gguf"):
        conv = [{"name": "m", "conv_error": f"ModuleNotFoundError: No module named '{module}'",
                 "kv_gb@4096": 1.0, "facts_weight_bytes": 100}]
        assert len(kv_corpus.gate(main, conv, {})) == 1, module
    calc = [{"name": "m", "conv_kv_size": 10, "kv_gb@4096": 1.0,
             "facts_error": "ModuleNotFoundError: No module named 'transformers'"}]
    assert len(kv_corpus.gate(main, calc, {})) == 1
    assert kv_corpus.unavailable_readers(main, calc) == {}
    gguf_main = [{"name": "gguf:x", "kv_gb@4096": 1.0, "conv_kv_size": 10}]
    no_gguf = [{"name": "gguf:x", "gguf_error": "ModuleNotFoundError: No module named 'gguf'"}]
    assert kv_corpus.gate(gguf_main, no_gguf, {}) == []
    assert kv_corpus.unavailable_readers(gguf_main, no_gguf) == {"gguf": 2}
    no_torch = [{"name": "gguf:x", "gguf_error": "ModuleNotFoundError: No module named 'torch'"}]
    assert len(kv_corpus.gate(gguf_main, no_torch, {})) == 2


def test_an_unstated_module_is_named_and_bounded():
    """A figure may sit below its truth only by the modules the entry names as unstated -- the
    checkpoint holds them and no config key describes them -- with their source; below that, or
    with no source, it fails."""
    main = [{"name": "m", "conv_serving_weight_bytes": 200}]
    branch = [{"name": "m", "conv_serving_weight_bytes": 150}]
    entry = {"config": "m", "reader": "conv_serving_weight_bytes", "main": 200, "branch": 150,
             "truth": 160, "source": "s", "cause": "c", "unstated": 10,
             "unstated_source": "a module no config key describes"}
    assert kv_corpus.gate(main, branch, {"entries": [entry]}) == []
    assert any("below its ground truth" in f for f in kv_corpus.gate(
        main, branch, {"entries": [dict(entry, unstated=5)]}))
    assert any("names no source" in f for f in kv_corpus.gate(
        main, branch, {"entries": [dict(entry, unstated_source="")]}))


def test_a_derived_figure_is_held_by_its_inputs():
    """The KV tier planner's T1 host memory is derived from its working set: a drop passes when
    the entry names the figures it follows and each is held to its truth, and fails when one is
    below its truth or the entry names none (round 16)."""
    main = [{"name": "m", "kvplan_host_gib": 168.75, "kvplan_ws_gib": 57.8}]
    branch = [{"name": "m", "kvplan_host_gib": 0.0, "kvplan_ws_gib": 14.46}]
    assert len(kv_corpus.gate(main, branch, {})) == 2
    ws = {"config": "m", "reader": "kvplan_ws_gib", "main": 57.8, "branch": 14.46, "truth": 14.4,
          "source": "s", "cause": "c"}
    host = {"config": "m", "reader": "kvplan_host_gib", "main": 168.75, "branch": 0.0,
            "kind": "memory", "held_by": ["kvplan_ws_gib"], "cause": "c"}
    assert kv_corpus.gate(main, branch, {"entries": [ws, host]}) == []
    assert any("below its ground truth" in f for f in kv_corpus.gate(
        main, branch, {"entries": [dict(ws, truth=20.0), host]}))
    assert any("no ground truth" in f for f in kv_corpus.gate(
        main, branch, {"entries": [ws, dict(host, held_by=None)]}))


# ------------------------------------------------------------------ the reader inventory

UNMEASURED = os.path.join(GATE, "unmeasured.json")


def test_every_public_figure_path_is_measured_or_excluded_with_its_reason(baseline, branch_run):
    """Every public function of llm-memory-calculator/src, BudSimulator/src, BudSimulator/apis and
    bud_component.py whose body names a model-shape key (``kv_corpus.figure_readers``, read from
    the source with ``ast``) is run by the harness -- the corpus above, recorded as it ran -- or
    named in tests/fixtures/kv_gate/unmeasured.json with the reason it is no memory or capacity
    figure of a model config. A new reader fails here until the harness measures it (and the gate
    holds what it computes) or the list says why not. Where a reader's declared third-party module
    is not installed (``READER_DEPENDENCIES``) the functions only it reaches cannot run: the
    inventory is checked where every reader runs."""
    names = [r["name"] for r in baseline]
    skipped = kv_corpus.unavailable_readers(baseline, branch_run["rows"], names)
    if skipped:
        pytest.skip("a reader's module is not installed here: " + ", ".join(sorted(skipped)))
    readers = kv_corpus.figure_readers(REPO)
    missing = kv_corpus.unmeasured_readers(readers, branch_run["ran"],
                                           kv_corpus.load_exclusions(UNMEASURED))
    assert not missing, "public readers neither measured nor excluded:\n" + "\n".join(
        f"{r['file']}:{r['line']} {r['function']} {r['keys']}" for r in missing)


def test_every_exclusion_names_an_unmeasured_reader_and_why(baseline, branch_run):
    """An entry names a public reader the source holds, gives its reason, and is not run by the
    harness -- a stale entry (the function is gone, renamed, or now measured) fails."""
    names = [r["name"] for r in baseline]
    readers = {(r["file"], r["function"]): r for r in kv_corpus.figure_readers(REPO)}
    bad = []
    for key, reason in kv_corpus.load_exclusions(UNMEASURED).items():
        if key not in readers or not readers[key]["public"]:
            bad.append(f"{key}: no public reader of a shape key by that name")
        elif len(reason) < 40:
            bad.append(f"{key}: no reason")
        elif key in branch_run["ran"] and not kv_corpus.unavailable_readers(
                baseline, branch_run["rows"], names):
            bad.append(f"{key}: the harness runs it -- measured, not excluded")
    assert not bad, "\n".join(bad)


def test_the_inventory_finds_a_reader_and_fails_an_unmeasured_one(tmp_path):
    """The inventory itself: a public function naming a shape key is found (as a string, an
    attribute or a variable), a private one is found but not public, a function naming none is
    not found; a reader the harness did not run fails unless the exclusions name it."""
    pkg = tmp_path / "llm-memory-calculator" / "src" / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "m.py").write_text(
        "def kv(cfg):\n    return cfg['num_key_value_heads']\n"
        "class Graph:\n    def width(self, mc):\n        return mc.hidden_size\n"
        "    def _depth(self, num_layers):\n        return num_layers\n"
        "def other(x):\n    return x['name']\n")
    readers = {r["function"]: r for r in kv_corpus.figure_readers(str(tmp_path))}
    assert set(readers) == {"kv", "Graph.width", "Graph._depth"}
    assert readers["kv"]["public"] and not readers["Graph._depth"]["public"]
    rel = os.path.join("llm-memory-calculator", "src", "pkg", "m.py")
    ran = {(rel, "kv")}
    missing = kv_corpus.unmeasured_readers(list(readers.values()), ran, {})
    assert [r["function"] for r in missing] == ["Graph.width"]
    assert kv_corpus.unmeasured_readers(list(readers.values()), ran,
                                        {(rel, "Graph.width"): "why"}) == []


def test_ran_code_records_the_functions_entered(tmp_path):
    """``ran_code`` names each function of the repository entered inside the block by its file and
    qualified name, and nothing entered after it."""
    src = tmp_path / "probe_mod.py"
    src.write_text("class C:\n    def m(self):\n        return 1\n"
                   "def f():\n    return C().m()\ndef g():\n    return 2\n")
    sys.path.insert(0, str(tmp_path))
    try:
        import importlib

        mod = importlib.import_module("probe_mod")
        with kv_corpus.ran_code(str(tmp_path)) as ran:
            mod.f()
        mod.g()
    finally:
        sys.path.remove(str(tmp_path))
    assert {("probe_mod.py", "f"), ("probe_mod.py", "C.m")} <= ran.functions
    assert ("probe_mod.py", "g") not in ran.functions
