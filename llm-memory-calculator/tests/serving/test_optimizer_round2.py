"""Round-2 golden tests for the serving optimizer + v2 optimizer endpoints.

These pin the PHYSICAL outcomes of solutions_round2.md §5 (Unit: Optimizers):

- R2-OP-A: honest failure (no fabricated fallback) and up-front 404 on unknown hardware for the three
  v2 optimizer endpoints.
- R2-OP-B: ``get_various_parallization`` returns only full-cluster-utilization splits.
- R2-OP-C: end-to-end latency objective; request-completion-rate throughput label; precision dedup by
  multiplier signature; range-based (mu == sigma == 0) Morris effect for the categorical precision factor.

No magic numbers: every assertion is a relation between the optimizer's own outputs.
"""
import os
import sys
from pathlib import Path

import pytest

from llm_memory_calculator.genz.serving.config_optimizer import (
    ConfigOptimizer,
    SearchSpace,
)
from llm_memory_calculator.genz.LLM_inference.best_parallelization import (
    get_various_parallization,
)

# Use the same model/hardware the existing optimizer tests use.
MODEL = "meta-llama/Meta-Llama-3.1-8B"
HARDWARE = "A100_80GB_GPU"
FAKE_HARDWARE = "TOTALLY_FAKE_GPU"

INPUT_TOKENS = 256
OUTPUT_TOKENS = 64


# --------------------------------------------------------------------------------------------------
# R2-OP-C: end-to-end latency, request-rate throughput, precision dedup
# --------------------------------------------------------------------------------------------------

class TestEndToEndLatency:
    def test_e2e_latency_is_prefill_plus_tpot_times_output(self):
        """C1: latency score == prefill + tpot * output_tokens (not prefill + a single decode step)."""
        opt = ConfigOptimizer(MODEL, HARDWARE, num_devices=1)
        r = opt._evaluate_config(
            {"tensor_parallel": 1, "pipeline_parallel": 1, "batch_size": 8, "precision": "bf16"},
            INPUT_TOKENS, OUTPUT_TOKENS,
        )
        assert r.feasible
        expected = r.prefill_latency_ms + r.tpot_ms * OUTPUT_TOKENS
        assert opt._compute_score(r, "latency") == pytest.approx(expected)
        # E2E must be strictly larger than a single-step (prefill + one decode) for output_tokens > 1.
        assert expected > r.prefill_latency_ms + r.decode_latency_ms

    def test_best_latency_equals_e2e(self):
        """The reported best latency score is the true E2E latency of the chosen config."""
        opt = ConfigOptimizer(MODEL, HARDWARE, num_devices=1)
        space = SearchSpace(
            tensor_parallel=[1], pipeline_parallel=[1],
            batch_sizes=[1, 8], precisions=["bf16"],
        )
        result = opt.optimize(
            target="latency", search_space=space, budget=10,
            input_tokens=INPUT_TOKENS, output_tokens=OUTPUT_TOKENS,
        )
        best = result["best_config"]
        assert best is not None
        m = result["best_metrics"]
        expected_e2e = m["prefill_latency_ms"] + m["tpot_ms"] * OUTPUT_TOKENS
        assert result["best_score"] == pytest.approx(expected_e2e)


class TestRequestRateThroughput:
    def test_throughput_rps_times_output_equals_token_tps(self):
        """C2: throughput_rps is a true request-completion rate = token_tps / output_tokens."""
        opt = ConfigOptimizer(MODEL, HARDWARE, num_devices=1)
        r = opt._evaluate_config(
            {"tensor_parallel": 1, "pipeline_parallel": 1, "batch_size": 8, "precision": "bf16"},
            INPUT_TOKENS, OUTPUT_TOKENS,
        )
        assert r.feasible
        assert r.token_throughput_tps > 0
        assert r.throughput_rps * OUTPUT_TOKENS == pytest.approx(r.token_throughput_tps)
        # And it is no longer byte-identical to the token rate (output_tokens > 1).
        assert r.throughput_rps < r.token_throughput_tps


class TestPrecisionDedup:
    def test_fp8_int8_collapse_to_one(self):
        """C3: precisions with identical (compute, mem) multipliers are evaluated once."""
        opt = ConfigOptimizer(MODEL, HARDWARE, num_devices=1)
        # fp8 and int8 share (0.5, 1); bf16 is (1, 2) -> two distinct levels.
        assert opt._dedupe_precisions(["bf16", "fp8", "int8"]) == ["bf16", "fp8"]
        # An all-equivalent list collapses to one.
        assert opt._dedupe_precisions(["fp8", "int8"]) == ["fp8"]

    def test_generated_grid_has_no_duplicate_precision(self):
        opt = ConfigOptimizer(MODEL, HARDWARE, num_devices=1)
        space = SearchSpace(
            tensor_parallel=[1], pipeline_parallel=[1],
            batch_sizes=[1], precisions=["bf16", "fp8", "int8"],
        )
        cfgs = opt._generate_configs(space, budget=100)
        precs = [c["precision"] for c in cfgs]
        assert "int8" not in precs  # int8 deduped against fp8
        assert sorted(set(precs)) == ["bf16", "fp8"]


# --------------------------------------------------------------------------------------------------
# R2-OP-A: honest failure (no fabrication), feasible exclusion
# --------------------------------------------------------------------------------------------------

class TestHonestFailure:
    def test_failed_config_is_infeasible_with_error(self):
        """An unmodelable config (OOM raise) is feasible=False with an error and zeroed metrics."""
        opt = ConfigOptimizer(MODEL, HARDWARE, num_devices=1)
        # A batch this large cannot fit the KV cache on a single A100 -> engine raises ValueError.
        r = opt._evaluate_config(
            {"tensor_parallel": 1, "pipeline_parallel": 1, "batch_size": 1_000_000, "precision": "bf16"},
            INPUT_TOKENS, OUTPUT_TOKENS,
        )
        assert r.feasible is False
        assert r.error
        # No fabricated metrics.
        assert r.prefill_latency_ms == 0.0
        assert r.decode_latency_ms == 0.0
        assert r.throughput_rps == 0.0
        assert r.token_throughput_tps == 0.0

    def test_check_feasibility_rejects_failed_result(self):
        opt = ConfigOptimizer(MODEL, HARDWARE, num_devices=1)
        from llm_memory_calculator.genz.serving.config_optimizer import (
            ConfigResult, OptimizationConstraints,
        )
        failed = ConfigResult(
            config={"tensor_parallel": 1, "pipeline_parallel": 1},
            feasible=False, error="boom",
        )
        assert opt._check_feasibility(failed, OptimizationConstraints()) is False

    def test_failed_config_excluded_from_best(self):
        """An OOM config never wins; the best is drawn only from feasible points."""
        opt = ConfigOptimizer(MODEL, HARDWARE, num_devices=1)
        # Mix a feasible small batch with an OOM-huge batch.
        space = SearchSpace(
            tensor_parallel=[1], pipeline_parallel=[1],
            batch_sizes=[1, 1_000_000], precisions=["bf16"],
        )
        result = opt.optimize(
            target="throughput", search_space=space, budget=10,
            input_tokens=INPUT_TOKENS, output_tokens=OUTPUT_TOKENS,
        )
        assert result["best_config"] is not None
        assert result["best_config"]["batch_size"] == 1
        # The huge-batch point is present in all_results but flagged infeasible.
        huge = [r for r in result["all_results"] if r["config"]["batch_size"] == 1_000_000]
        assert huge and all(not r["feasible"] for r in huge)


# --------------------------------------------------------------------------------------------------
# R2-OP-C (C4): categorical Morris effect
# --------------------------------------------------------------------------------------------------

class TestCategoricalMorris:
    def test_precision_effect_is_range_based(self):
        """C4: precision is qualitative -> mu == sigma == 0; mu_star is the score range (>= 0)."""
        opt = ConfigOptimizer(MODEL, HARDWARE, num_devices=4)
        result = opt.analyze_sensitivity(
            target="throughput", input_tokens=INPUT_TOKENS, output_tokens=OUTPUT_TOKENS,
        )
        prec = result["sensitivity"]["precision"]
        assert prec["mu"] == 0.0
        assert prec["sigma"] == 0.0
        assert prec["mu_star"] >= 0.0

    def test_numeric_params_keep_ee_statistics(self):
        """Numeric factors still use the elementary-effect formula (mu may be nonzero)."""
        opt = ConfigOptimizer(MODEL, HARDWARE, num_devices=4)
        result = opt.analyze_sensitivity(
            target="throughput", input_tokens=INPUT_TOKENS, output_tokens=OUTPUT_TOKENS,
        )
        # batch_size varies throughput, so its mu_star must be a positive importance.
        assert result["sensitivity"]["batch_size"]["mu_star"] > 0.0


# --------------------------------------------------------------------------------------------------
# R2-OP-B: full-cluster-utilization parallelism search
# --------------------------------------------------------------------------------------------------

class TestFullClusterParallelism:
    @pytest.mark.parametrize("n", [2, 4, 8])
    def test_dense_model_uses_full_cluster(self, n):
        """Every (TP, PP) split must use the whole cluster: TP * PP == total_nodes."""
        combos = get_various_parallization("llama2_7b", n)
        assert combos  # non-empty
        assert all(tp * pp == n for (tp, pp) in combos)

    def test_single_node(self):
        assert get_various_parallization("llama2_7b", 1) == {(1, 1)}

    def test_zero_nodes_raises(self):
        with pytest.raises(ValueError):
            get_various_parallization("llama2_7b", 0)

    def test_moe_allows_expert_parallel_factor(self):
        """MoE: TP * PP must divide total_nodes (the remainder is expert parallelism)."""
        n = 8
        combos = get_various_parallization("mixtral_8x7b", n)
        assert combos
        assert all(n % (tp * pp) == 0 for (tp, pp) in combos)


# --------------------------------------------------------------------------------------------------
# R2-OP-A: v2 endpoints return 404 for unknown hardware (FastAPI TestClient)
# --------------------------------------------------------------------------------------------------

os.environ.setdefault("BUD_ALLOWED_HOSTS", "localhost,127.0.0.1,testserver")
_BUDSIM = Path(__file__).resolve().parents[3] / "BudSimulator"
if str(_BUDSIM) not in sys.path:
    sys.path.insert(0, str(_BUDSIM))

try:
    from fastapi.testclient import TestClient
    from apis.main import app as _app
    _APP_OK = True
except Exception as _e:  # pragma: no cover - environment dependent
    _APP_OK = False
    _app = None


@pytest.fixture
def client():
    if not _APP_OK:
        pytest.skip("BudSimulator app could not be imported")
    return TestClient(_app)


class TestUnknownHardware404:
    def test_optimize_config_404(self, client):
        resp = client.post("/api/v2/optimize/config", json={
            "model": MODEL, "hardware": FAKE_HARDWARE,
        })
        assert resp.status_code == 404

    def test_optimize_pareto_404(self, client):
        resp = client.post("/api/v2/optimize/pareto", json={
            "model": MODEL, "hardware": FAKE_HARDWARE,
        })
        assert resp.status_code == 404

    def test_analyze_sensitivity_404(self, client):
        resp = client.post("/api/v2/analyze/sensitivity", json={
            "model": MODEL, "hardware": FAKE_HARDWARE,
        })
        assert resp.status_code == 404

    def test_known_hardware_still_200(self, client):
        """Regression guard: a real device must still succeed (small budget for speed)."""
        resp = client.post("/api/v2/optimize/config", json={
            "model": MODEL, "hardware": HARDWARE,
            "num_devices": 1, "budget": 2,
            "input_tokens": 128, "output_tokens": 16,
        })
        assert resp.status_code == 200
        assert "best_config" in resp.json()
