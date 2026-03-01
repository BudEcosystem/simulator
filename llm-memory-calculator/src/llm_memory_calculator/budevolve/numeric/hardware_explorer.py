"""Hardware design space exploration using pymoo NSGA-II."""
import warnings
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from ..types import EvalResult, ParetoResult, HardwareSpec
from ..evaluator import BudSimEvaluator
from .search_spaces import HardwareSearchSpace

# Map real hardware names to approximate specs for comparison
_REAL_HARDWARE_SPECS = {
    "A100_40GB_GPU": HardwareSpec(312, 1600, 40, 40, 18000, 150, 1.41, 1, 400, 10000),
    "A100_80GB_GPU": HardwareSpec(312, 2039, 80, 40, 18000, 150, 1.41, 1, 400, 15000),
    "H100_GPU": HardwareSpec(990, 3350, 80, 50, 33000, 450, 1.98, 1, 700, 25000),
    "H200_GPU": HardwareSpec(990, 4800, 141, 50, 33000, 450, 1.98, 1, 700, 30000),
    "B200_GPU": HardwareSpec(2250, 8000, 192, 64, 50000, 900, 2.1, 1, 1000, 40000),
}


class HardwareExplorer:
    """Discover ideal hardware specs via multi-objective search.

    Uses pymoo NSGA-II to search the hardware parameter space (FLOPS,
    memory BW, capacity, etc.) with BudSim as the evaluation function.
    No LLM required.
    """

    def __init__(
        self,
        model: str,
        evaluator: Optional[BudSimEvaluator] = None,
    ):
        self._model = model
        self._evaluator = evaluator or BudSimEvaluator()

    def explore(
        self,
        objectives: List[str] = None,
        constraints: Optional[Dict[str, float]] = None,
        search_space: Optional[HardwareSearchSpace] = None,
        n_generations: int = 100,
        pop_size: int = 50,
        input_tokens: int = 512,
        output_tokens: int = 128,
        batch_size: int = 32,
    ) -> ParetoResult:
        """Search hardware parameter space for optimal designs.

        Args:
            objectives: e.g. ["throughput", "cost", "power"]
            constraints: e.g. {"min_throughput_rps": 100, "max_power_w": 700}
            search_space: Hardware parameter ranges.
            n_generations: NSGA-II generations.
            pop_size: Population size per generation.
            input_tokens: Representative input length.
            output_tokens: Representative output length.
            batch_size: Batch size for inference.

        Returns:
            ParetoResult with Pareto-optimal hardware designs.
        """
        objectives = objectives or ["throughput", "cost"]
        constraints = constraints or {}
        search_space = search_space or HardwareSearchSpace()

        all_results = self._run_search(
            search_space, objectives, constraints,
            n_generations, pop_size, input_tokens, output_tokens, batch_size,
        )

        feasible = [r for r in all_results if r.feasible]
        min_tput = constraints.get("min_throughput_rps", 0.0)
        max_power = constraints.get("max_power_w", float("inf"))
        feasible = [
            r for r in feasible
            if r.throughput_rps >= min_tput and r.power_w <= max_power
        ]

        if not feasible:
            return ParetoResult(
                pareto_front=[], all_evaluated=[],
                best_per_objective={}, total_evaluations=len(all_results),
            )

        pareto = self._compute_pareto(feasible, objectives)

        best_per_obj = {}
        for obj in objectives:
            if obj == "throughput":
                best = max(feasible, key=lambda r: r.throughput_rps)
            elif obj == "cost":
                best = min(feasible, key=lambda r: r.config.get("estimated_cost_usd", 1e9))
            elif obj == "power":
                best = min(feasible, key=lambda r: r.power_w)
            else:
                best = feasible[0]
            best_per_obj[obj] = self._to_dict(best)

        return ParetoResult(
            pareto_front=[self._to_dict(r) for r in pareto],
            all_evaluated=[self._to_dict(r) for r in feasible[:100]],
            best_per_objective=best_per_obj,
            num_generations=n_generations,
            total_evaluations=len(all_results),
        )

    def what_if(
        self,
        base_hardware: str,
        param: str,
        param_range: Tuple[float, float],
        steps: int = 20,
        input_tokens: int = 512,
        output_tokens: int = 128,
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """Sweep a single hardware parameter.

        Example: what_if("A100_40GB_GPU", "offchip_mem_bw_gbps", (1600, 8000), steps=10)
        answers "what if A100 had up to 8 TB/s memory bandwidth?"

        Args:
            base_hardware: Name of base hardware from _REAL_HARDWARE_SPECS.
            param: HardwareSpec field name to sweep.
            param_range: (lo, hi) range for the parameter.
            steps: Number of sweep points.
            input_tokens: Representative input length.
            output_tokens: Representative output length.
            batch_size: Batch size for inference.

        Returns:
            List of dicts with evaluation results at each sweep point.
        """
        base = _REAL_HARDWARE_SPECS.get(base_hardware)
        if base is None:
            warnings.warn(f"No specs for {base_hardware}, using H100 defaults")
            base = _REAL_HARDWARE_SPECS["H100_GPU"]

        lo, hi = param_range
        values = np.linspace(lo, hi, steps)
        results = []

        for val in values:
            import dataclasses
            hw = dataclasses.replace(base, **{param: float(val)})
            eval_result = self._evaluator.evaluate_hardware(
                hw, model=self._model,
                input_tokens=input_tokens, output_tokens=output_tokens,
                batch_size=batch_size,
            )
            entry = self._to_dict(eval_result)
            entry[param] = float(val)
            results.append(entry)

        return results

    def compare_vs_real(
        self,
        hypothetical: HardwareSpec,
        real_hardware: List[str] = None,
        input_tokens: int = 512,
        output_tokens: int = 128,
        batch_size: int = 32,
    ) -> Dict[str, Dict[str, Any]]:
        """Compare a hypothetical hardware design against real GPUs.

        Args:
            hypothetical: HardwareSpec describing the hypothetical design.
            real_hardware: List of real hardware names to compare against.
            input_tokens: Representative input length.
            output_tokens: Representative output length.
            batch_size: Batch size for inference.

        Returns:
            Dict mapping "hypothetical" and each real hardware name to
            their evaluation result dicts.
        """
        real_hardware = real_hardware or ["A100_40GB_GPU", "H100_GPU"]
        comparison = {}

        hyp_result = self._evaluator.evaluate_hardware(
            hypothetical, model=self._model,
            input_tokens=input_tokens, output_tokens=output_tokens,
            batch_size=batch_size,
        )
        comparison["hypothetical"] = self._to_dict(hyp_result)

        for hw_name in real_hardware:
            from ..types import ServingConfig
            cfg = ServingConfig(model=self._model, hardware=hw_name)
            result = self._evaluator.evaluate_config(
                cfg, input_tokens=input_tokens, output_tokens=output_tokens,
            )
            comparison[hw_name] = self._to_dict(result)

        return comparison

    def _run_search(self, search_space, objectives, constraints,
                    n_gen, pop_size, input_tokens, output_tokens, batch_size):
        """Run pymoo or fallback grid search over hardware params.

        Args:
            search_space: HardwareSearchSpace with parameter bounds.
            objectives: List of objective names.
            constraints: Dict of constraint name to value.
            n_gen: Number of NSGA-II generations.
            pop_size: Population size per generation.
            input_tokens: Representative input length.
            output_tokens: Representative output length.
            batch_size: Batch size for inference.

        Returns:
            List of EvalResult from evaluating candidate hardware designs.
        """
        try:
            return self._run_pymoo_hw(
                search_space, objectives, n_gen, pop_size,
                input_tokens, output_tokens, batch_size,
            )
        except ImportError:
            warnings.warn("pymoo not installed, using grid sample")
            return self._run_grid_hw(
                search_space, input_tokens, output_tokens, batch_size,
            )

    def _run_pymoo_hw(self, search_space, objectives, n_gen, pop_size,
                      input_tokens, output_tokens, batch_size):
        """NSGA-II optimization over hardware parameters using pymoo.

        Defines a continuous-variable problem over the 7 hardware parameters
        (FLOPS, memory BW, memory size, on-chip memory, on-chip BW,
        interchip BW, frequency) and uses BudSimEvaluator.evaluate_hardware
        as the objective function.

        Args:
            search_space: HardwareSearchSpace with parameter bounds.
            objectives: List of objective names.
            n_gen: Number of generations.
            pop_size: Population size.
            input_tokens: Representative input length.
            output_tokens: Representative output length.
            batch_size: Batch size for inference.

        Returns:
            List of all EvalResult produced during the search.
        """
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.optimize import minimize

        evaluator = self._evaluator
        model = self._model
        all_results = []
        n_obj = len(objectives)

        class HWProblem(ElementwiseProblem):
            def __init__(self):
                super().__init__(
                    n_var=search_space.n_var, n_obj=n_obj,
                    xl=np.array(search_space.xl),
                    xu=np.array(search_space.xu),
                )

            def _evaluate(self, x, out, *args, **kwargs):
                hw = HardwareSpec(
                    flops_tflops=x[0], offchip_mem_bw_gbps=x[1],
                    off_chip_mem_size_gb=x[2], on_chip_mem_size_mb=x[3],
                    onchip_mem_bw_gbps=x[4], interchip_link_bw_gbps=x[5],
                    frequency_ghz=x[6],
                )
                result = evaluator.evaluate_hardware(
                    hw, model=model,
                    input_tokens=input_tokens, output_tokens=output_tokens,
                    batch_size=batch_size,
                )
                all_results.append(result)

                obj_vals = []
                for obj in objectives:
                    if obj == "throughput":
                        obj_vals.append(-result.throughput_rps)
                    elif obj == "cost":
                        obj_vals.append(hw.estimated_cost_usd)
                    elif obj == "power":
                        obj_vals.append(hw.tdp_watts)
                    elif obj == "latency":
                        obj_vals.append(result.ttft_ms)
                    else:
                        obj_vals.append(-result.throughput_rps)
                out["F"] = np.array(obj_vals)

        problem = HWProblem()
        algorithm = NSGA2(pop_size=min(pop_size, 20))
        minimize(problem, algorithm, termination=("n_gen", min(n_gen, 30)),
                 seed=42, verbose=False)
        return all_results

    def _run_grid_hw(self, search_space, input_tokens, output_tokens, batch_size):
        """Fallback grid search when pymoo is not available.

        Samples a coarse grid over FLOPS and memory BW dimensions with
        fixed memory size, evaluating each point via BudSimEvaluator.

        Args:
            search_space: HardwareSearchSpace with parameter bounds.
            input_tokens: Representative input length.
            output_tokens: Representative output length.
            batch_size: Batch size for inference.

        Returns:
            List of EvalResult from grid evaluation.
        """
        results = []
        for flops in np.linspace(*search_space.flops_range, 5):
            for bw in np.linspace(*search_space.mem_bw_range, 5):
                hw = HardwareSpec(
                    flops_tflops=flops, offchip_mem_bw_gbps=bw,
                    off_chip_mem_size_gb=80.0,
                )
                result = self._evaluator.evaluate_hardware(
                    hw, model=self._model,
                    input_tokens=input_tokens, output_tokens=output_tokens,
                    batch_size=batch_size,
                )
                results.append(result)
        return results

    def _compute_pareto(self, results, objectives):
        """Compute non-dominated Pareto front from a list of results.

        For each result, checks whether any other result dominates it
        (i.e., is at least as good on all objectives and strictly better
        on at least one). Non-dominated results form the Pareto front.

        Args:
            results: List of feasible EvalResult objects.
            objectives: List of objective names to consider.

        Returns:
            List of EvalResult on the Pareto front.
        """
        def get_vals(r):
            vals = []
            for obj in objectives:
                if obj == "throughput":
                    vals.append(r.throughput_rps)
                elif obj == "cost":
                    vals.append(-r.config.get("estimated_cost_usd", 1e9))
                elif obj == "power":
                    vals.append(-r.power_w)
                elif obj == "latency":
                    vals.append(-r.ttft_ms)
                else:
                    vals.append(r.throughput_rps)
            return vals

        pareto = []
        for i, ri in enumerate(results):
            vi = get_vals(ri)
            dominated = False
            for j, rj in enumerate(results):
                if i == j:
                    continue
                vj = get_vals(rj)
                if (all(vj[k] >= vi[k] for k in range(len(objectives))) and
                        any(vj[k] > vi[k] for k in range(len(objectives)))):
                    dominated = True
                    break
            if not dominated:
                pareto.append(ri)
        return pareto

    def _to_dict(self, r):
        """Convert an EvalResult to a plain dictionary.

        Args:
            r: EvalResult to convert.

        Returns:
            Dict with throughput, latency, memory, power, and feasibility fields.
        """
        return {
            "config": r.config, "throughput_rps": r.throughput_rps,
            "ttft_ms": r.ttft_ms, "tpot_ms": r.tpot_ms,
            "memory_gb": r.memory_gb, "power_w": r.power_w,
            "feasible": r.feasible,
        }
