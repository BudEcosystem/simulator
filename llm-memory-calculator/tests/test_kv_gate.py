"""THE GATE: no figure of the branch moves in the unsafe direction against origin/main unless the
allowlist names it, with both figures and the ground truth the branch's figure is held to.

The corpus is every config this repository ships: prepopulated.db's rows, every fixture under
tests/fixtures (the hub configs among them), the harness's synthetic shapes and GenZ's built-in
models -- measured by scripts/kv_corpus.py through every reader of a model's memory (the
calculator, the engine estimate, the FRD-027 facts and its vLLM planner, GenZ's graphs and runtime
modelling, the serving and disaggregation paths, the prefix cache, the KV planner and
BudSimulator's converter). tests/fixtures/kv_gate/origin-main.json is the same harness's
measurement of origin/main over the same corpus.

A memory figure (KV, state, weights) below origin/main's, or a capacity figure (vLLM's blocks,
tokens and running sequences, the serving path's requests, the scheduler's peak) above it, fails
unless tests/fixtures/kv_gate/allowlist.json names it -- config, reader, origin/main's and the
branch's figures -- and the branch's figure is at least its ground truth (transformers' own model
for the config, the checkpoint's tensors, or the config arithmetic the entry cites), or, for a
capacity, the memory figures it is sized from are each held to theirs. The same check runs over
the machine's HF cache and model folders with ``kv_corpus.py --gate``
(docs/proofs/kv-layer-kinds-corpus-diff.md).
"""

import argparse
import json
import os
import sys

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


def test_no_unsafe_move_outside_the_allowlist(baseline, branch_rows, allowlist):
    failures = kv_corpus.gate(baseline, branch_rows, allowlist,
                              names=[r["name"] for r in baseline])
    assert not failures, f"{len(failures)} gate failures:\n" + "\n".join(failures[:60])


def test_every_allowlist_entry_names_its_figures_truth_and_cause(allowlist):
    bad = []
    for e in allowlist["entries"]:
        where = f"{e.get('config')} {e.get('reader')}"
        if not (e.get("config") and e.get("reader") and "main" in e and "branch" in e):
            bad.append(f"{where}: config, reader and both figures")
        if not e.get("cause"):
            bad.append(f"{where}: no cause")
        if e.get("kind") == "memory":
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
