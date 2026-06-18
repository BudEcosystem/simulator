"""Tests for pymoo-based HardwareExplorer."""
import builtins
import warnings

import pytest
from unittest.mock import patch, MagicMock

from llm_memory_calculator.budevolve.types import (
    EvalResult, ParetoResult, HardwareSpec,
)
from llm_memory_calculator.budevolve.numeric.search_spaces import HardwareSearchSpace


def _make_hw_eval(flops, bw, mem, throughput, ttft, cost):
    return EvalResult(
        throughput_rps=throughput, ttft_ms=ttft,
        tpot_ms=10.0, feasible=True,
        config={"flops_tflops": flops, "offchip_mem_bw_gbps": bw,
                "off_chip_mem_size_gb": mem, "estimated_cost_usd": cost},
    )


class TestHardwareSearchSpace:
    def test_bounds(self):
        space = HardwareSearchSpace()
        # R2-BE3: n_var re-baselined 7 -> 9 (tdp_watts and estimated_cost_usd are now swept
        # decision variables, appended after the 7 physical params).
        assert space.n_var == 9
        assert len(space.xl) == 9
        assert len(space.xu) == 9
        assert all(lo < hi for lo, hi in zip(space.xl, space.xu))
        # Last two vars are the swept TDP and cost; bounds come from _REAL_HARDWARE_SPECS anchors.
        assert space.xl[-2:] == [400.0, 10000.0]
        assert space.xu[-2:] == [1000.0, 40000.0]


class TestHardwareExplorer:
    def test_explore_returns_pareto_result(self):
        from llm_memory_calculator.budevolve.numeric.hardware_explorer import HardwareExplorer

        results = [
            _make_hw_eval(312, 1600, 40, 50.0, 100.0, 10000),
            _make_hw_eval(990, 3350, 80, 120.0, 40.0, 25000),
            _make_hw_eval(1979, 8000, 192, 200.0, 20.0, 40000),
        ]

        with patch.object(HardwareExplorer, "_run_search") as mock_run:
            mock_run.return_value = results
            explorer = HardwareExplorer(model="test-model")
            result = explorer.explore(
                objectives=["throughput", "cost"],
                n_generations=10,
            )
            assert isinstance(result, ParetoResult)
            assert len(result.pareto_front) > 0

    def test_what_if_sweeps_parameter(self):
        from llm_memory_calculator.budevolve.numeric.hardware_explorer import HardwareExplorer

        with patch("llm_memory_calculator.budevolve.numeric.hardware_explorer.BudSimEvaluator") as mock_cls:
            mock_eval = MagicMock()
            mock_eval.evaluate_hardware.return_value = EvalResult(
                throughput_rps=100.0, ttft_ms=50.0, tpot_ms=10.0, feasible=True,
            )
            mock_cls.return_value = mock_eval

            explorer = HardwareExplorer(model="test-model")
            result = explorer.what_if(
                base_hardware="A100_40GB_GPU",
                param="offchip_mem_bw_gbps",
                param_range=(1600, 8000),
                steps=5,
            )
            assert len(result) == 5
            assert all("offchip_mem_bw_gbps" in r for r in result)

    def test_design_for_model(self):
        from llm_memory_calculator.budevolve.numeric.hardware_explorer import HardwareExplorer

        results = [
            _make_hw_eval(312, 1600, 40, 50.0, 100.0, 10000),
            _make_hw_eval(990, 3350, 80, 120.0, 40.0, 25000),
        ]

        with patch.object(HardwareExplorer, "_run_search") as mock_run:
            mock_run.return_value = results
            explorer = HardwareExplorer(model="test-model")
            result = explorer.design_for_model(
                target_throughput_rps=100.0, n_generations=10,
            )
            assert isinstance(result, ParetoResult)
            # Only the second result meets 100 rps target
            for item in result.pareto_front:
                assert item["throughput_rps"] >= 100.0

    def test_compare_vs_real(self):
        from llm_memory_calculator.budevolve.numeric.hardware_explorer import HardwareExplorer

        with patch("llm_memory_calculator.budevolve.numeric.hardware_explorer.BudSimEvaluator") as mock_cls:
            mock_eval = MagicMock()
            mock_eval.evaluate_config.return_value = EvalResult(
                throughput_rps=100.0, ttft_ms=50.0, tpot_ms=10.0, feasible=True,
            )
            mock_eval.evaluate_hardware.return_value = EvalResult(
                throughput_rps=150.0, ttft_ms=30.0, tpot_ms=8.0, feasible=True,
            )
            mock_cls.return_value = mock_eval

            explorer = HardwareExplorer(model="test-model")
            hw = HardwareSpec(flops_tflops=500.0, offchip_mem_bw_gbps=3000.0,
                              off_chip_mem_size_gb=80.0)
            result = explorer.compare_vs_real(hw, real_hardware=["H100_GPU"])
            assert "hypothetical" in result
            assert "H100_GPU" in result


# ---------------------------------------------------------------------------
# R2-BE4: pymoo must be installed so the NSGA-II path actually runs (instead of
# silently falling back to the coarse grid search and producing a degenerate
# Pareto front). The grid fallback must remain present and disclosed.
# ---------------------------------------------------------------------------

# A budget engine evaluation is the dominating cost in these tests, so keep the
# pymoo population/generations small. NSGA-II still meaningfully searches: with
# pop_size=12 over 4 generations it evaluates 48 candidates across the swept
# cost/power/throughput objectives.
_GOLDEN_GEN = 4
_GOLDEN_POP = 12
_GOLDEN_KWARGS = dict(input_tokens=128, output_tokens=32, batch_size=8)
_EXPECTED_FRONT_KEYS = {
    "config", "throughput_rps", "ttft_ms", "tpot_ms",
    "memory_gb", "power_w", "estimated_cost_usd", "feasible",
}


def test_pymoo_is_installed():
    """R2-BE4 root cause: pymoo must import. If this fails, the explorer would
    silently fall back to the coarse grid search."""
    import pymoo  # noqa: F401

    # pymoo 0.6+ ships the NSGA-II / ElementwiseProblem / minimize API used here.
    from pymoo.algorithms.moo.nsga2 import NSGA2  # noqa: F401
    from pymoo.core.problem import ElementwiseProblem  # noqa: F401
    from pymoo.optimize import minimize  # noqa: F401


@pytest.mark.slow
def test_nsga2_runs_without_grid_fallback():
    """Golden: with pymoo installed, explore() takes the NSGA-II path and does
    NOT emit the 'pymoo not installed' fallback warning."""
    from llm_memory_calculator.budevolve.numeric.hardware_explorer import HardwareExplorer

    explorer = HardwareExplorer(model="llama2_7b")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = explorer.explore(
            objectives=["throughput", "cost", "power"],
            n_generations=_GOLDEN_GEN, pop_size=_GOLDEN_POP,
            **_GOLDEN_KWARGS,
        )
    fell_back = any("pymoo not installed" in str(w.message) for w in caught)
    assert not fell_back, "NSGA-II path must run; grid fallback should not trigger"
    assert isinstance(result, ParetoResult)
    # NSGA-II evaluates pop_size candidates per generation.
    assert result.total_evaluations == _GOLDEN_GEN * _GOLDEN_POP
    assert result.num_generations == _GOLDEN_GEN


@pytest.mark.slow
def test_nsga2_pareto_front_is_non_degenerate():
    """Golden physical outcome: the swept cost/power/throughput objectives vary
    across the Pareto front (>1 distinct value on at least one objective). A
    degenerate front (all identical cost AND power) was the original bug when
    cost/power were pinned to dataclass defaults under a grid that did not sweep
    them."""
    from llm_memory_calculator.budevolve.numeric.hardware_explorer import HardwareExplorer

    explorer = HardwareExplorer(model="llama2_7b")
    result = explorer.explore(
        objectives=["throughput", "cost", "power"],
        n_generations=_GOLDEN_GEN, pop_size=_GOLDEN_POP,
        **_GOLDEN_KWARGS,
    )
    front = result.pareto_front
    assert len(front) >= 1
    distinct_cost = {round(d["estimated_cost_usd"], 2) for d in front
                     if d["estimated_cost_usd"] is not None}
    distinct_power = {round(d["power_w"], 2) for d in front}
    distinct_tput = {round(d["throughput_rps"], 4) for d in front}
    # Non-degenerate: at least one swept objective takes multiple values on the
    # Pareto front (a true multi-objective trade-off, not a single point).
    assert (len(distinct_cost) > 1 or len(distinct_power) > 1
            or len(distinct_tput) > 1), (
        f"degenerate front: cost={distinct_cost} power={distinct_power} "
        f"tput={distinct_tput}"
    )
    # Swept cost/power must lie within the sourced anchor bounds (A100..B200).
    space = HardwareSearchSpace()
    for d in front:
        if d["estimated_cost_usd"] is not None:
            assert space.cost_range[0] <= d["estimated_cost_usd"] <= space.cost_range[1]
        # power_w = tdp_watts * 0.75 utilization (evaluator), so it is bounded by
        # 0.75 * tdp_range.
        assert d["power_w"] <= 0.75 * space.tdp_range[1] + 1e-6


@pytest.mark.slow
def test_grid_fallback_still_works_and_is_disclosed():
    """Golden: when pymoo import fails, explore() falls back to the grid search,
    discloses it via a warning, and returns a ParetoResult with the same
    front-entry shape as the NSGA-II path (consistent-shaped fronts)."""
    from llm_memory_calculator.budevolve.numeric.hardware_explorer import HardwareExplorer

    explorer = HardwareExplorer(model="llama2_7b")
    real_import = builtins.__import__

    def _no_pymoo(name, *args, **kwargs):
        if name.startswith("pymoo"):
            raise ImportError("forced: pymoo unavailable")
        return real_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", side_effect=_no_pymoo):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = explorer.explore(
                objectives=["throughput", "cost", "power"],
                n_generations=2, pop_size=6, **_GOLDEN_KWARGS,
            )
    disclosed = any("pymoo not installed" in str(w.message) for w in caught)
    assert disclosed, "grid fallback must be disclosed via warning"
    assert isinstance(result, ParetoResult)
    # Consistent-shaped fronts: each entry has the same keys as NSGA-II output.
    for entry in result.pareto_front:
        assert set(entry.keys()) == _EXPECTED_FRONT_KEYS
