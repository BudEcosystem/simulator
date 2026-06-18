"""Golden tests for the Round-2 BudEvolve accuracy fixes (R2-BE1/2/3/5 + improvement_pct).

These assert PHYSICAL outcomes, not implementation details:
  - R2-BE5: evaluate_config/evaluate_hardware report throughput_rps == token_tps / output_tokens
  - R2-BE2 (=R2-G1, already landed in genz utils.py): evaluate_hardware applies an eta < 1
            efficiency band (guard test against a future regression to 100% MFU).
  - R2-BE3: tdp_watts/estimated_cost_usd are swept -> cost/power objectives vary; a design
            exceeding the B200 perf/W anchor is rejected.
  - R2-BE1: candidate fitness is candidate-DEPENDENT (a degenerate scheduler that returns []
            scores strictly differently from a working one).
  - improvement_pct (OpenEvolve branch): same-scale (baseline scored through the same bridge).
"""
import os
import tempfile

import pytest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from llm_memory_calculator.budevolve.types import (
    ServingConfig, HardwareSpec, EvalResult,
)


# --------------------------------------------------------------------------- #
# R2-BE5: throughput_rps == token_throughput_tps / output_tokens              #
# --------------------------------------------------------------------------- #

def _moddeling_output(throughput_tok=5120.0):
    # GenZ sets Throughput == Throughput_tokens_per_sec (decode emits 1 tok/req/iter).
    return SimpleNamespace(
        Latency=25.0,
        Throughput=throughput_tok,
        Throughput_tokens_per_sec=throughput_tok,
        summary_table=MagicMock(),
    )


class TestThroughputRelabel:
    def test_evaluate_config_rps_is_tps_over_genlen(self):
        from llm_memory_calculator.budevolve.evaluator import BudSimEvaluator
        out = _moddeling_output(5120.0)
        with patch("llm_memory_calculator.budevolve.evaluator.prefill_moddeling", return_value=out), \
             patch("llm_memory_calculator.budevolve.evaluator.decode_moddeling", return_value=out):
            ev = BudSimEvaluator()
            cfg = ServingConfig(model="m", hardware="H100_GPU")
            r = ev.evaluate_config(cfg, input_tokens=512, output_tokens=128)
            assert r.token_throughput_tps == 5120.0
            assert r.throughput_rps == pytest.approx(5120.0 / 128)
            assert r.throughput_rps == pytest.approx(r.token_throughput_tps / 128)

    def test_evaluate_config_genlen_one_rps_equals_tps(self):
        from llm_memory_calculator.budevolve.evaluator import BudSimEvaluator
        out = _moddeling_output(900.0)
        with patch("llm_memory_calculator.budevolve.evaluator.prefill_moddeling", return_value=out), \
             patch("llm_memory_calculator.budevolve.evaluator.decode_moddeling", return_value=out):
            ev = BudSimEvaluator()
            cfg = ServingConfig(model="m", hardware="H100_GPU")
            r = ev.evaluate_config(cfg, input_tokens=512, output_tokens=1)
            assert r.throughput_rps == pytest.approx(r.token_throughput_tps)

    def test_evaluate_hardware_rps_is_tps_over_genlen(self):
        from llm_memory_calculator.budevolve.evaluator import BudSimEvaluator
        out = _moddeling_output(4096.0)
        with patch("llm_memory_calculator.budevolve.evaluator.prefill_moddeling", return_value=out), \
             patch("llm_memory_calculator.budevolve.evaluator.decode_moddeling", return_value=out), \
             patch("llm_memory_calculator.genz.system.System", return_value=MagicMock()):
            ev = BudSimEvaluator()
            hw = HardwareSpec(flops_tflops=500.0, offchip_mem_bw_gbps=3000.0, off_chip_mem_size_gb=80.0)
            r = ev.evaluate_hardware(hw, model="m", input_tokens=512, output_tokens=256)
            assert r.throughput_rps == pytest.approx(r.token_throughput_tps / 256)


# --------------------------------------------------------------------------- #
# R2-BE2 (=R2-G1): evaluate_hardware applies an eta < 1 efficiency band        #
# --------------------------------------------------------------------------- #

class TestEfficiencyBand:
    def test_evaluate_hardware_applies_efficiency_band(self):
        """token_throughput for a fixed HardwareSpec must be strictly LESS than the
        eta == 1 (100% MFU/MBU) hypothetical -- proving the GenZ System-object path applies
        the C2/C3 bands. Guards against a refactor silently regressing BudEvolve to 100% MFU.
        """
        from llm_memory_calculator.budevolve.evaluator import BudSimEvaluator

        hw = HardwareSpec(
            flops_tflops=312.0, offchip_mem_bw_gbps=2039.0, off_chip_mem_size_gb=80.0,
        )
        ev = BudSimEvaluator()
        banded = ev.evaluate_hardware(
            hw, model="meta-llama/Meta-Llama-3.1-8B",
            input_tokens=256, output_tokens=64, batch_size=8,
        )
        if not banded.feasible:
            pytest.skip("GenZ engine unavailable in this environment")

        # eta == 1 hypothetical: bypass the C2/C3 banding in get_inference_system by patching it
        # to return a System forced to 100% MFU/MBU. The banded path MUST be strictly slower.
        from llm_memory_calculator.genz.system import System
        from llm_memory_calculator.genz.LLM_inference import llm_decode as _ld

        real_get = _ld.get_inference_system

        def _eta1_get(*args, **kwargs):
            system = real_get(*args, **kwargs)
            if isinstance(system, System):
                system.compute_efficiency = 1.0
                system.memory_efficiency = 1.0
            return system

        sk = hw.to_system_kwargs()
        sk["bits"] = "bf16"
        sys100 = System(**sk)
        with patch.object(_ld, "get_inference_system", side_effect=_eta1_get):
            decode100 = _ld.decode_moddeling(
                model="meta-llama/Meta-Llama-3.1-8B", batch_size=8,
                input_tokens=256, output_tokens=64, system_name=sys100, bits="bf16",
            )
        assert banded.token_throughput_tps < decode100.Throughput_tokens_per_sec, (
            "evaluate_hardware must run at eta < 1, not 100% MFU/MBU"
        )


# --------------------------------------------------------------------------- #
# R2-BE3: swept cost/power vary; perf/W ceiling rejects implausible designs    #
# --------------------------------------------------------------------------- #

class TestHardwareDSECoDesign:
    def test_search_space_n_var_includes_tdp_cost(self):
        from llm_memory_calculator.budevolve.numeric.search_spaces import HardwareSearchSpace
        s = HardwareSearchSpace()
        assert s.n_var == 9
        # bounds anchored on _REAL_HARDWARE_SPECS (A100 .. B200)
        assert s.tdp_range == (400.0, 1000.0)
        assert s.cost_range == (10000.0, 40000.0)

    def test_perf_per_watt_ceiling_from_b200_anchor(self):
        from llm_memory_calculator.budevolve.numeric import hardware_explorer as hx
        b200 = hx._REAL_HARDWARE_SPECS["B200_GPU"]
        assert hx._MAX_TFLOPS_PER_WATT == pytest.approx(b200.flops_tflops / b200.tdp_watts)
        assert hx._MAX_TFLOPS_PER_WATT == pytest.approx(2.25)

    def test_design_exceeding_perf_per_watt_is_rejected(self):
        from llm_memory_calculator.budevolve.numeric import hardware_explorer as hx
        # 3000 TFLOPS at 1000 W = 3.0 TFLOPS/W > 2.25 anchor -> infeasible
        bad = HardwareSpec(flops_tflops=3000.0, offchip_mem_bw_gbps=8000.0,
                           off_chip_mem_size_gb=192.0, tdp_watts=1000.0)
        good = HardwareSpec(flops_tflops=2000.0, offchip_mem_bw_gbps=8000.0,
                            off_chip_mem_size_gb=192.0, tdp_watts=1000.0)
        assert hx._is_perf_per_watt_infeasible(bad) is True
        assert hx._is_perf_per_watt_infeasible(good) is False

    def test_best_per_objective_cost_varies_with_swept_cost(self):
        """Two HardwareSpecs differing only in swept cost must produce different
        best_per_objective['cost']. Previously cost was a constant default -> degenerate.
        """
        from llm_memory_calculator.budevolve.numeric.hardware_explorer import HardwareExplorer

        cheap = EvalResult(
            throughput_rps=120.0, ttft_ms=40.0, tpot_ms=10.0, power_w=600.0, feasible=True,
            config={"flops_tflops": 990, "tdp_watts": 800.0, "estimated_cost_usd": 12000.0},
        )
        pricey = EvalResult(
            throughput_rps=200.0, ttft_ms=20.0, tpot_ms=8.0, power_w=750.0, feasible=True,
            config={"flops_tflops": 2000, "tdp_watts": 1000.0, "estimated_cost_usd": 38000.0},
        )
        with patch.object(HardwareExplorer, "_run_search", return_value=[cheap, pricey]):
            ex = HardwareExplorer(model="m")
            res = ex.explore(objectives=["throughput", "cost", "power"], n_generations=2)
            best_cost = res.best_per_objective["cost"]
            best_power = res.best_per_objective["power"]
            best_tput = res.best_per_objective["throughput"]
            # cheapest design wins the cost objective
            assert best_cost["estimated_cost_usd"] == 12000.0
            # lowest power wins the power objective
            assert best_power["power_w"] == 600.0
            # highest throughput wins throughput
            assert best_tput["throughput_rps"] == 200.0
            # The cost objective is NOT pinned to a constant default.
            assert best_cost["estimated_cost_usd"] != best_tput["estimated_cost_usd"]


# --------------------------------------------------------------------------- #
# R2-BE1: candidate-DEPENDENT fitness                                          #
# --------------------------------------------------------------------------- #

_WORKING_SCHEDULER = (
    "def schedule_batch(queue, max_batch_size=256, max_tokens=8192):\n"
    "    if not queue:\n"
    "        return []\n"
    "    batch = []\n"
    "    total = 0\n"
    "    for r in queue:\n"
    "        t = r.input_tokens + r.output_tokens\n"
    "        if len(batch) >= max_batch_size:\n"
    "            break\n"
    "        if total + t > max_tokens:\n"
    "            break\n"
    "        batch.append(r)\n"
    "        total += t\n"
    "    return batch\n"
)

_DEGENERATE_SCHEDULER = (
    "def schedule_batch(queue, max_batch_size=256, max_tokens=8192):\n"
    "    return []\n"
)

_TINY_SCHEDULER = (
    "def schedule_batch(queue, max_batch_size=256, max_tokens=8192):\n"
    "    if not queue:\n"
    "        return []\n"
    "    return [queue[0]]\n"
)

_GOOD_CACHE = (
    "def evict_entries(cache_entries, num_to_evict):\n"
    "    s = sorted(cache_entries, key=lambda e: (e.frequency, -e.last_access_time))\n"
    "    return s[:num_to_evict]\n"
)

_BAD_CACHE = (
    "def evict_entries(cache_entries, num_to_evict):\n"
    "    s = sorted(cache_entries, key=lambda e: (-e.frequency, e.last_access_time))\n"
    "    return s[:num_to_evict]\n"
)


def _bridge_with_fixed_genz():
    """A bridge whose GenZ ceiling is a fixed constant, so any score delta is from E."""
    from llm_memory_calculator.budevolve.evolve.evaluator_bridge import BudSimEvalBridge
    bridge = BudSimEvalBridge(model="m", hardware="H100_GPU")
    bridge._evaluator = MagicMock()
    bridge._evaluator.evaluate_config.return_value = EvalResult(
        throughput_rps=100.0, ttft_ms=50.0, tpot_ms=10.0, slo_compliance=1.0, feasible=True,
    )
    return bridge


def _score_code(bridge, code):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        return bridge.evaluate(path)
    finally:
        os.unlink(path)


class TestCandidateDependentFitness:
    def test_working_vs_degenerate_scheduler_score_differs(self):
        bridge = _bridge_with_fixed_genz()
        good = _score_code(bridge, _WORKING_SCHEDULER)
        degen = _score_code(bridge, _DEGENERATE_SCHEDULER)

        assert good["fitness_basis"] == "executed_policy_efficiency x genz_throughput_ceiling"
        # GenZ ceiling identical, but fitness must differ because the policy differs.
        assert good["throughput_rps"] == degen["throughput_rps"] == 100.0
        assert good["combined_score"] != degen["combined_score"]
        assert good["combined_score"] > degen["combined_score"]
        # Degenerate (empty batch) is floored to eps, not zero.
        assert 0.0 < degen["policy_efficiency"] <= 1e-3
        assert good["policy_efficiency"] > degen["policy_efficiency"]

    def test_fuller_batch_scores_higher(self):
        bridge = _bridge_with_fixed_genz()
        full = _score_code(bridge, _WORKING_SCHEDULER)
        tiny = _score_code(bridge, _TINY_SCHEDULER)
        assert full["policy_efficiency"] > tiny["policy_efficiency"]
        assert full["combined_score"] > tiny["combined_score"]

    def test_valid_candidate_keeps_positive_combined_score(self):
        bridge = _bridge_with_fixed_genz()
        degen = _score_code(bridge, _DEGENERATE_SCHEDULER)
        assert degen["combined_score"] > 0.0  # eps floor preserves mocked-bridge contract

    def test_good_cache_policy_retains_more_hot_than_bad(self):
        bridge = _bridge_with_fixed_genz()
        good = _score_code(bridge, _GOOD_CACHE)
        bad = _score_code(bridge, _BAD_CACHE)
        # Good policy evicts cold (low-freq) entries -> retains more hot -> higher E.
        assert good["policy_efficiency"] > bad["policy_efficiency"]
        assert good["combined_score"] > bad["combined_score"]

    def test_timeout_candidate_floored_and_errored(self):
        bridge = _bridge_with_fixed_genz()
        looping = (
            "def schedule_batch(queue, max_batch_size=256, max_tokens=8192):\n"
            "    while True:\n"
            "        pass\n"
        )
        r = _score_code(bridge, looping)
        assert "error" in r
        assert "timeout" in r["error"]
        assert r["policy_efficiency"] <= 1e-3

    def test_contract_violation_foreign_objects_floored(self):
        bridge = _bridge_with_fixed_genz()
        cheater = (
            "def schedule_batch(queue, max_batch_size=256, max_tokens=8192):\n"
            "    class X: \n"
            "        input_tokens = 99999\n"
            "        output_tokens = 99999\n"
            "    return [X(), X(), X()]\n"
        )
        r = _score_code(bridge, cheater)
        assert "error" in r
        assert "contract violation" in r["error"]
        assert r["policy_efficiency"] <= 1e-3


# --------------------------------------------------------------------------- #
# improvement_pct (OpenEvolve branch) is same-scale                            #
# --------------------------------------------------------------------------- #

class TestOpenEvolveImprovementScale:
    def test_improvement_pct_uses_baseline_combined(self):
        """improvement_pct must compare best_score against the baseline's COMBINED score
        (same scale), not the raw throughput_rps.
        """
        import tempfile as _tf
        from llm_memory_calculator.budevolve.evolve import algorithm_evolver as ae

        # Mock the OpenEvolve package so _run_openevolve returns a known best_score.
        async def _fake_run():
            return {"best_program": "code", "best_score": 60.0}

        fake_instance = MagicMock()
        fake_instance.run = _fake_run
        fake_oe_module = MagicMock()
        fake_oe_module.OpenEvolve = MagicMock(return_value=fake_instance)

        with patch.dict("sys.modules", {"openevolve": fake_oe_module}):
            ev = ae.AlgorithmEvolver(model="m", hardware="H100_GPU")
            # Make the bridge baseline combined-score deterministic.
            with patch.object(ae.AlgorithmEvolver, "_baseline_combined_score", return_value=40.0):
                out_dir = _tf.mkdtemp()
                baseline = EvalResult(throughput_rps=1000.0, feasible=True)  # raw rps far from 40
                bridge = MagicMock()
                res = ev._run_openevolve(
                    base_code="def schedule_batch(q): return q",
                    algo_type="scheduler", baseline=baseline, baseline_metrics={},
                    roofline=None, bridge=bridge, iterations=3, output_dir=out_dir,
                )
                # (60 - 40)/40 * 100 = 50.0 ; NOT (60 - 1000)/1000 (= -94%).
                assert res["baseline_combined"] == 40.0
                assert res["improvement_pct"] == pytest.approx(50.0)
