"""FRD-025 Revision 2 (R2-1/R2-2): a vLLM build with no record of its own is planned on the family's baseline record.

Before, ``record_from_budconnect`` returned None for such a build (exact record, then same upstream minor, then
nothing), budsim shipped no runtime spec, and budcluster re-sized the pod with its legacy rules (accubits,
2026-10-29: vLLM 0.32). The baseline holds only what every recorded build of the family shares (launch, argument
syntax, ports, probes, settings, limits, parsers) and the conflicts any of them has; its features are only what every
recorded build verified, minus build-specific ones, plus what BudConnect lists for this build.
"""

import pytest

from llm_memory_calculator.runtime import records
from llm_memory_calculator.runtime.records import BUILD_FEATURES, record_from_budconnect

from .test_runtime_spec import golden_spec

ROW_032 = {
    "engine": "vllm",
    "engine_version": "0.32.0",
    "engine_image": "docker.io/vllm/vllm-openai@sha256:befccf",
    "upstream_vllm": "0.32.0",
    "kv_features": ["kv_events", "cache_salt", "mooncake_store"],
}


def _vllm_records():
    return [r for r in records.vendored().values() if r["engine"] == "vllm"]


def test_a_build_with_no_record_gets_the_baseline():
    rec = record_from_budconnect(ROW_032)
    assert rec is not None
    assert rec.id == "vllm@0.32.0~baseline" and rec.version == "0.32.0"
    assert rec.raw["record_basis"] == "baseline"
    assert rec.images["cuda"] == ROW_032["engine_image"]


def test_the_baseline_holds_what_every_recorded_build_shares():
    rec = record_from_budconnect(ROW_032)
    first = _vllm_records()[0]
    assert dict(rec.runtime) == first["runtime"]
    assert set(rec.settings) == set.intersection(*(set(r["settings"]) for r in _vllm_records()))
    conflicts = {c["id"] for r in _vllm_records() for c in r.get("conflicts") or []}
    assert {c.id for c in rec.conflicts} == conflicts, "every conflict a recorded build has"


def test_the_baseline_claims_only_features_every_build_verified_plus_budconnects():
    rec = record_from_budconnect(ROW_032)
    shared = set.intersection(*(set(r["features"]) for r in _vllm_records())) - BUILD_FEATURES
    assert not (rec.features & BUILD_FEATURES)
    assert set(ROW_032["kv_features"]) <= rec.features
    non_kv = rec.features - set(ROW_032["kv_features"]) - records.KV_FEATURES
    assert non_kv <= shared


@pytest.mark.parametrize(
    ("row", "basis"),
    [
        ({"engine": "vllm", "engine_version": "0.30.0", "upstream_vllm": "0.30.0"}, "exact"),
        ({"engine": "vllm", "engine_version": "0.10.1", "upstream_vllm": "0.31.0"}, "inherited"),
        (ROW_032, "baseline"),
    ],
)
def test_the_record_says_what_it_was_resolved_from(row, basis):
    assert record_from_budconnect(row).raw["record_basis"] == basis


@pytest.mark.parametrize("version", [None, ""])
def test_a_row_that_names_no_build_gets_no_baseline(version):
    """The baseline is for a build BudConnect names but no record covers, not for a row that names no build."""
    assert record_from_budconnect({"engine": "vllm", "engine_version": version}) is None


def test_an_engine_family_with_no_records_has_no_baseline():
    assert record_from_budconnect({"engine": "sglang", "engine_version": "0.5.0"}) is None


def test_a_build_with_no_record_is_planned_on_the_baseline():
    """The spec is composed, names its record and basis, and renders only baseline-backed settings."""
    spec, _ = golden_spec(
        "cuda_dedicated_tp1",
        engine_version="0.32.0",
        upstream_vllm="0.32.0",
        image="docker.io/vllm/vllm-openai@sha256:befccf",
    )
    assert spec.engine.record == "bud-connect:vllm@0.32.0~baseline"
    assert spec.engine.record_basis == "baseline"
