"""Multi-objective config optimization using pymoo NSGA-II."""
import itertools
import warnings
from typing import Dict, List, Optional, Any

from ..types import EvalResult, ParetoResult, ServingConfig
from ..evaluator import BudSimEvaluator
from .search_spaces import ConfigSearchSpace
from .pareto import compute_pareto_front


class NumericOptimizer:
    """Multi-objective serving config search using pymoo NSGA-II.

    Uses BudSim's GenZ analytical engine (~1-10ms per eval) as the
    objective function. No LLM required.
    """

    def __init__(
        self,
        model: str,
        hardware: str,
        evaluator: Optional[BudSimEvaluator] = None,
    ):
        self._model = model
        self._hardware = hardware
        self._evaluator = evaluator or BudSimEvaluator()

    def optimize(
        self,
        objectives: List[str] = None,
        constraints: Optional[Dict[str, float]] = None,
        search_space: Optional[ConfigSearchSpace] = None,
        n_generations: int = 100,
        pop_size: int = 50,
        input_tokens: int = 512,
        output_tokens: int = 128,
    ) -> ParetoResult:
        """Run multi-objective optimization over serving configs.

        Args:
            objectives: List of objectives, e.g. ["throughput", "latency"].
            constraints: Dict of constraint name -> max value,
                e.g. {"max_ttft_ms": 500.0, "max_tpot_ms": 50.0}.
            search_space: Config parameter ranges.
            n_generations: Number of NSGA-II generations.
            pop_size: Population size per generation.
            input_tokens: Representative input length.
            output_tokens: Representative output length.

        Returns:
            ParetoResult with non-dominated solutions.
        """
        objectives = objectives or ["throughput", "latency"]
        constraints = constraints or {}
        search_space = search_space or ConfigSearchSpace()

        all_results = self._run_pymoo(
            search_space, objectives, constraints,
            n_generations, pop_size, input_tokens, output_tokens,
        )

        # Filter feasible
        feasible = [r for r in all_results if r.feasible]
        if not feasible:
            return ParetoResult(
                pareto_front=[], all_evaluated=[],
                best_per_objective={}, num_generations=n_generations,
                total_evaluations=len(all_results),
            )

        # Apply constraint filtering
        max_ttft = constraints.get("max_ttft_ms", float("inf"))
        max_tpot = constraints.get("max_tpot_ms", float("inf"))
        feasible = [
            r for r in feasible
            if r.ttft_ms <= max_ttft and r.tpot_ms <= max_tpot
        ]

        pareto = self._compute_pareto(feasible, objectives)

        best_per_obj = {}
        for obj in objectives:
            if obj == "throughput":
                best = max(feasible, key=lambda r: r.throughput_rps)
            elif obj == "latency":
                best = min(feasible, key=lambda r: r.ttft_ms + r.tpot_ms)
            else:
                best = feasible[0] if feasible else None
            if best:
                best_per_obj[obj] = self._result_to_dict(best)

        return ParetoResult(
            pareto_front=[self._result_to_dict(r) for r in pareto],
            all_evaluated=[self._result_to_dict(r) for r in feasible[:100]],
            best_per_objective=best_per_obj,
            num_generations=n_generations,
            total_evaluations=len(all_results),
        )

    def _run_pymoo(
        self, search_space, objectives, constraints,
        n_generations, pop_size, input_tokens, output_tokens,
    ) -> List[EvalResult]:
        """Run pymoo NSGA-II optimization.

        Falls back to grid search if pymoo is not installed.
        """
        try:
            return self._run_pymoo_nsga2(
                search_space, objectives, constraints,
                n_generations, pop_size, input_tokens, output_tokens,
            )
        except ImportError:
            warnings.warn("pymoo not installed, falling back to grid search")
            return self._run_grid_search(search_space, input_tokens, output_tokens)

    def _run_pymoo_nsga2(
        self, search_space, objectives, constraints,
        n_generations, pop_size, input_tokens, output_tokens,
    ) -> List[EvalResult]:
        """NSGA-II optimization using pymoo."""
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.optimize import minimize
        from pymoo.operators.sampling.rnd import IntegerRandomSampling
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        import numpy as np

        evaluator = self._evaluator
        model = self._model
        hardware = self._hardware
        all_results = []

        tp_vals = search_space.tensor_parallel
        pp_vals = search_space.pipeline_parallel
        bs_vals = search_space.batch_sizes
        prec_vals = search_space.precisions

        n_obj = len(objectives)

        class ServingProblem(ElementwiseProblem):
            def __init__(self):
                super().__init__(
                    n_var=4,
                    n_obj=n_obj,
                    xl=np.array([0, 0, 0, 0]),
                    xu=np.array([
                        len(tp_vals) - 1,
                        len(pp_vals) - 1,
                        len(bs_vals) - 1,
                        len(prec_vals) - 1,
                    ]),
                    vtype=int,
                )

            def _evaluate(self, x, out, *args, **kwargs):
                tp = tp_vals[int(x[0]) % len(tp_vals)]
                pp = pp_vals[int(x[1]) % len(pp_vals)]
                bs = bs_vals[int(x[2]) % len(bs_vals)]
                prec = prec_vals[int(x[3]) % len(prec_vals)]

                cfg = ServingConfig(
                    model=model, hardware=hardware,
                    tensor_parallel=tp, pipeline_parallel=pp,
                    batch_size=bs, precision=prec,
                )
                result = evaluator.evaluate_config(cfg, input_tokens, output_tokens)
                all_results.append(result)

                obj_values = []
                for obj in objectives:
                    if obj == "throughput":
                        obj_values.append(-result.throughput_rps)  # minimize negative
                    elif obj == "latency":
                        obj_values.append(result.ttft_ms + result.tpot_ms)
                    elif obj == "cost":
                        obj_values.append(result.cost_per_million_tokens)
                    elif obj == "power":
                        obj_values.append(result.power_w)
                    else:
                        obj_values.append(-result.throughput_rps)

                out["F"] = np.array(obj_values)

        problem = ServingProblem()
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=0.9, eta=15, vtype=float, repair=lambda _p, X: np.round(X).astype(int)),
            mutation=PM(eta=20, vtype=float, repair=lambda _p, X: np.round(X).astype(int)),
        )

        res = minimize(
            problem, algorithm,
            termination=("n_gen", n_generations),
            seed=42, verbose=False,
        )

        return all_results

    def _run_grid_search(
        self, search_space, input_tokens, output_tokens,
    ) -> List[EvalResult]:
        """Fallback grid search when pymoo unavailable."""
        results = []
        for tp, pp, bs, prec in itertools.product(
            search_space.tensor_parallel,
            search_space.pipeline_parallel,
            search_space.batch_sizes[:4],
            search_space.precisions[:2],
        ):
            cfg = ServingConfig(
                model=self._model, hardware=self._hardware,
                tensor_parallel=tp, pipeline_parallel=pp,
                batch_size=bs, precision=prec,
            )
            result = self._evaluator.evaluate_config(cfg, input_tokens, output_tokens)
            results.append(result)
        return results

    def _compute_pareto(
        self, results: List[EvalResult], objectives: List[str],
    ) -> List[EvalResult]:
        """Compute non-dominated Pareto front."""
        def get_obj_values(r):
            vals = []
            for obj in objectives:
                if obj == "throughput":
                    vals.append(r.throughput_rps)
                elif obj == "latency":
                    vals.append(-(r.ttft_ms + r.tpot_ms))  # higher = better
                elif obj == "cost":
                    vals.append(-r.cost_per_million_tokens)
                elif obj == "power":
                    vals.append(-r.power_w)
                else:
                    vals.append(r.throughput_rps)
            return vals

        return compute_pareto_front(results, get_obj_values)

    def _result_to_dict(self, r: EvalResult) -> Dict[str, Any]:
        return {
            "config": r.config,
            "throughput_rps": r.throughput_rps,
            "token_throughput_tps": r.token_throughput_tps,
            "ttft_ms": r.ttft_ms,
            "tpot_ms": r.tpot_ms,
            "e2e_latency_ms": r.e2e_latency_ms,
            "memory_gb": r.memory_gb,
            "power_w": r.power_w,
            "feasible": r.feasible,
        }
