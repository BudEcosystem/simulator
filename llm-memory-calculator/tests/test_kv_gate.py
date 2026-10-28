"""THE GATE: no figure of the branch moves in the unsafe direction against origin/main unless the
allowlist names it, with both figures and the ground truth the branch's figure is held to.

The corpus is every config this repository ships: prepopulated.db's rows, every fixture under
tests/fixtures (the hub configs among them, and tests/fixtures/gguf's metadata-only GGUFs, read
through the GGUF reader), the harness's synthetic shapes and GenZ's built-in models -- measured by scripts/kv_corpus.py through every reader of a model's memory (the
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
whose reader cannot import a third-party module in this environment (`gguf`, `lxml`) is not
measured here, and the test says so; it needs neither transformers nor torch (the config-class
defaults ship with the package). The same check runs over
the machine's HF cache and model folders with ``kv_corpus.py --gate``
(docs/proofs/kv-layer-kinds-corpus-diff.md).
"""

import argparse
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
def branch_rows(tmp_path_factory):
    args = argparse.Namespace(
        db=os.path.join(REPO, "BudSimulator", "data", "prepopulated.db"), fixtures=[FIXTURES],
        hf_cache=None, models=None, repo_root=REPO,
        genz_data=str(tmp_path_factory.mktemp("genz")), no_synthetic=False)
    return kv_corpus.measure_corpus(args)


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


def test_no_unsafe_move_outside_the_allowlist(baseline, branch_rows, allowlist):
    """Every figure is held; a figure whose reader cannot import a third-party module here (the
    GGUF reader's `gguf`, the converter's `lxml`) is not measured in this environment, and says
    so -- a reader that raises anything else fails."""
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
        elif e.get("kind") == "memory":
            if not isinstance(e.get("truth"), (int, float)) or not e.get("source"):
                bad.append(f"{where}: no ground truth or source")
            elif e["branch"] < e["truth"] * (1 - float(e.get("vectors", 0.0))):
                bad.append(f"{where}: {e['branch']} below its ground truth {e['truth']}")
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
