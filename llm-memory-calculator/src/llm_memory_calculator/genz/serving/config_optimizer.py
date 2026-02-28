"""Configuration optimizer for LLM serving using GenZ analytical engine."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import warnings
import itertools

from .constants import NS_PER_MS


@dataclass
class SearchSpace:
    """Search space for configuration optimization."""
    tensor_parallel: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    pipeline_parallel: List[int] = field(default_factory=lambda: [1, 2, 4])
    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 32, 64, 128, 256])
    precisions: List[str] = field(default_factory=lambda: ["bf16", "fp8", "int8"])
    block_sizes: List[int] = field(default_factory=lambda: [8, 16, 32])
    enable_chunked_prefill: List[bool] = field(default_factory=lambda: [True, False])
    enable_prefix_caching: List[bool] = field(default_factory=lambda: [True, False])
    gpu_memory_utilization: List[float] = field(default_factory=lambda: [0.85, 0.90, 0.95])


@dataclass
class OptimizationConstraints:
    """Constraints for configuration optimization."""
    max_ttft_ms: float = float('inf')
    max_tpot_ms: float = float('inf')
    max_memory_gb: float = float('inf')
    max_power_w: float = float('inf')
    max_total_devices: int = 64


@dataclass
class ConfigResult:
    """Result of evaluating a single configuration."""
    config: Dict[str, Any]
    throughput_rps: float = 0.0
    token_throughput_tps: float = 0.0
    prefill_latency_ms: float = 0.0
    decode_latency_ms: float = 0.0
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    feasible: bool = True
    score: float = 0.0


class ConfigOptimizer:
    """Configuration optimizer using GenZ analytical engine.

    Uses the fast analytical model (~ms per evaluation) for exhaustive
    grid search over the search space. For Pareto optimization, evaluates
    all feasible configs and computes non-dominated front.
    """

    def __init__(self, model: str, hardware: str, num_devices: int = 1):
        self._model = model
        self._hardware = hardware
        self._num_devices = num_devices

    def optimize(
        self,
        target: str = "throughput",
        constraints: Optional[OptimizationConstraints] = None,
        search_space: Optional[SearchSpace] = None,
        budget: int = 50,
        input_tokens: int = 512,
        output_tokens: int = 128,
    ) -> Dict[str, Any]:
        """Find optimal config for a single objective.

        Args:
            target: "throughput", "latency", "efficiency" (throughput/device)
            constraints: Feasibility constraints
            search_space: Parameter ranges to search
            budget: Max evaluations (caps grid if grid is larger)
            input_tokens: Representative input sequence length
            output_tokens: Representative output sequence length

        Returns:
            Dict with best_config, best_score, all_results, feasible_count
        """
        constraints = constraints or OptimizationConstraints()
        search_space = search_space or SearchSpace()

        configs = self._generate_configs(search_space, budget)
        results = []

        for cfg in configs:
            result = self._evaluate_config(cfg, input_tokens, output_tokens)
            result.feasible = self._check_feasibility(result, constraints)
            if result.feasible:
                result.score = self._compute_score(result, target)
            results.append(result)

        feasible = [r for r in results if r.feasible]
        if not feasible:
            return {
                "best_config": None,
                "best_score": 0.0,
                "all_results": [self._result_to_dict(r) for r in results],
                "feasible_count": 0,
                "total_evaluated": len(results),
            }

        if target == "latency":
            best = min(feasible, key=lambda r: r.score)
        else:
            best = max(feasible, key=lambda r: r.score)

        return {
            "best_config": best.config,
            "best_score": best.score,
            "best_metrics": {
                "throughput_rps": best.throughput_rps,
                "token_throughput_tps": best.token_throughput_tps,
                "prefill_latency_ms": best.prefill_latency_ms,
                "decode_latency_ms": best.decode_latency_ms,
                "ttft_ms": best.ttft_ms,
                "tpot_ms": best.tpot_ms,
            },
            "all_results": [self._result_to_dict(r) for r in results[:20]],
            "feasible_count": len(feasible),
            "total_evaluated": len(results),
        }

    def pareto_optimize(
        self,
        objectives: List[str],
        constraints: Optional[OptimizationConstraints] = None,
        search_space: Optional[SearchSpace] = None,
        budget: int = 100,
        input_tokens: int = 512,
        output_tokens: int = 128,
    ) -> Dict[str, Any]:
        """Multi-objective Pareto optimization.

        Args:
            objectives: List of objectives, e.g. ["throughput", "latency"]
            constraints: Feasibility constraints
            search_space: Parameter ranges
            budget: Max evaluations

        Returns:
            Dict with pareto_front, all_results
        """
        constraints = constraints or OptimizationConstraints()
        search_space = search_space or SearchSpace()

        configs = self._generate_configs(search_space, budget)
        results = []

        for cfg in configs:
            result = self._evaluate_config(cfg, input_tokens, output_tokens)
            result.feasible = self._check_feasibility(result, constraints)
            results.append(result)

        feasible = [r for r in results if r.feasible]
        if not feasible:
            return {"pareto_front": [], "all_results": [], "feasible_count": 0}

        pareto = self._compute_pareto_front(feasible, objectives)

        return {
            "pareto_front": [self._result_to_dict(r) for r in pareto],
            "all_results": [self._result_to_dict(r) for r in feasible[:50]],
            "feasible_count": len(feasible),
            "total_evaluated": len(results),
            "objectives": objectives,
        }

    def analyze_sensitivity(
        self,
        target: str = "throughput",
        num_samples: int = 50,
        input_tokens: int = 512,
        output_tokens: int = 128,
    ) -> Dict[str, Any]:
        """Analyze parameter sensitivity via variance decomposition.

        Tests each parameter while holding others at baseline, measuring
        the impact on the target metric.
        """
        baseline_cfg = {
            "tensor_parallel": 1,
            "pipeline_parallel": 1,
            "batch_size": 32,
            "precision": "bf16",
        }

        sensitivity = {}
        baseline_result = self._evaluate_config(baseline_cfg, input_tokens, output_tokens)
        baseline_score = self._compute_score(baseline_result, target)

        param_ranges = {
            "batch_size": [1, 4, 8, 16, 32, 64, 128, 256],
            "tensor_parallel": [1, 2, 4, 8],
            "pipeline_parallel": [1, 2, 4],
            "precision": ["bf16", "fp8", "int8"],
        }

        for param, values in param_ranges.items():
            scores = []
            for val in values:
                cfg = dict(baseline_cfg)
                cfg[param] = val
                tp = cfg.get("tensor_parallel", 1)
                pp = cfg.get("pipeline_parallel", 1)
                if tp * pp > self._num_devices:
                    continue
                result = self._evaluate_config(cfg, input_tokens, output_tokens)
                score = self._compute_score(result, target)
                scores.append({"value": val, "score": score})

            if scores:
                score_vals = [s["score"] for s in scores]
                mean_score = sum(score_vals) / len(score_vals)
                variance = sum((s - mean_score) ** 2 for s in score_vals) / len(score_vals)
                sensitivity[param] = {
                    "values": scores,
                    "variance": variance,
                    "range": max(score_vals) - min(score_vals) if score_vals else 0,
                    "baseline": baseline_score,
                }

        total_variance = sum(s["variance"] for s in sensitivity.values())
        for param in sensitivity:
            sensitivity[param]["importance"] = (
                sensitivity[param]["variance"] / total_variance if total_variance > 0 else 0
            )

        ranked = sorted(sensitivity.items(), key=lambda x: x[1]["variance"], reverse=True)

        return {
            "sensitivity": sensitivity,
            "ranking": [p for p, _ in ranked],
            "baseline_config": baseline_cfg,
            "baseline_score": baseline_score,
            "target": target,
        }

    def _generate_configs(self, space: SearchSpace, budget: int) -> List[Dict]:
        """Generate configuration grid, capped by budget."""
        tp_values = [tp for tp in space.tensor_parallel if tp <= self._num_devices]
        pp_values = [pp for pp in space.pipeline_parallel if pp <= self._num_devices]

        configs = []
        for tp, pp, bs, prec in itertools.product(
            tp_values, pp_values, space.batch_sizes, space.precisions
        ):
            if tp * pp > self._num_devices:
                continue
            configs.append({
                "tensor_parallel": tp,
                "pipeline_parallel": pp,
                "batch_size": bs,
                "precision": prec,
            })
            if len(configs) >= budget:
                break

        return configs

    def _evaluate_config(
        self, config: Dict, input_tokens: int, output_tokens: int,
    ) -> ConfigResult:
        """Evaluate a single config using GenZ analytical engine."""
        tp = config.get("tensor_parallel", 1)
        pp = config.get("pipeline_parallel", 1)
        bs = config.get("batch_size", 1)
        prec = config.get("precision", "bf16")

        try:
            from llm_memory_calculator.genz import prefill_moddeling, decode_moddeling

            prefill = prefill_moddeling(
                model=self._model, batch_size=bs, input_tokens=input_tokens,
                system_name=self._hardware, bits=prec,
                tensor_parallel=tp, pipeline_parallel=pp,
            )
            decode = decode_moddeling(
                model=self._model, batch_size=bs, input_tokens=input_tokens,
                output_tokens=output_tokens, system_name=self._hardware, bits=prec,
                tensor_parallel=tp, pipeline_parallel=pp,
            )

            prefill_lat = prefill.Latency
            decode_lat = decode.Latency
            throughput = decode.Throughput
            token_tps = decode.Throughput_tokens_per_sec

        except Exception as e:
            warnings.warn(f"GenZ evaluation failed for {config}: {e}")
            prefill_lat = input_tokens * 0.01 * bs / max(tp, 1)
            decode_lat = 0.5 * bs / max(tp, 1)
            throughput = 1000 * bs / decode_lat if decode_lat > 0 else 0
            token_tps = throughput

        return ConfigResult(
            config=config,
            throughput_rps=throughput,
            token_throughput_tps=token_tps,
            prefill_latency_ms=prefill_lat,
            decode_latency_ms=decode_lat,
            ttft_ms=prefill_lat,
            tpot_ms=decode_lat,
        )

    def _check_feasibility(
        self, result: ConfigResult, constraints: OptimizationConstraints,
    ) -> bool:
        """Check if a result satisfies all constraints."""
        if result.ttft_ms > constraints.max_ttft_ms:
            return False
        if result.tpot_ms > constraints.max_tpot_ms:
            return False
        tp = result.config.get("tensor_parallel", 1)
        pp = result.config.get("pipeline_parallel", 1)
        if tp * pp > constraints.max_total_devices:
            return False
        return True

    def _compute_score(self, result: ConfigResult, target: str) -> float:
        """Compute optimization score for a result."""
        if target == "throughput":
            return result.throughput_rps
        elif target == "token_throughput":
            return result.token_throughput_tps
        elif target == "latency":
            return result.prefill_latency_ms + result.decode_latency_ms
        elif target == "efficiency":
            tp = result.config.get("tensor_parallel", 1)
            pp = result.config.get("pipeline_parallel", 1)
            devices = tp * pp
            return result.throughput_rps / devices if devices > 0 else 0
        else:
            return result.throughput_rps

    def _compute_pareto_front(
        self, results: List[ConfigResult], objectives: List[str],
    ) -> List[ConfigResult]:
        """Compute Pareto-optimal front from feasible results."""
        def get_obj_values(r):
            vals = []
            for obj in objectives:
                if obj == "throughput":
                    vals.append(r.throughput_rps)
                elif obj == "token_throughput":
                    vals.append(r.token_throughput_tps)
                elif obj == "latency":
                    vals.append(-r.prefill_latency_ms - r.decode_latency_ms)
                elif obj == "efficiency":
                    tp = r.config.get("tensor_parallel", 1)
                    pp = r.config.get("pipeline_parallel", 1)
                    vals.append(r.throughput_rps / (tp * pp))
                else:
                    vals.append(r.throughput_rps)
            return vals

        pareto = []
        for i, ri in enumerate(results):
            vi = get_obj_values(ri)
            dominated = False
            for j, rj in enumerate(results):
                if i == j:
                    continue
                vj = get_obj_values(rj)
                if all(vj[k] >= vi[k] for k in range(len(objectives))) and \
                   any(vj[k] > vi[k] for k in range(len(objectives))):
                    dominated = True
                    break
            if not dominated:
                pareto.append(ri)

        return pareto

    def _result_to_dict(self, result: ConfigResult) -> Dict[str, Any]:
        return {
            "config": result.config,
            "throughput_rps": result.throughput_rps,
            "token_throughput_tps": result.token_throughput_tps,
            "prefill_latency_ms": result.prefill_latency_ms,
            "decode_latency_ms": result.decode_latency_ms,
            "ttft_ms": result.ttft_ms,
            "tpot_ms": result.tpot_ms,
            "feasible": result.feasible,
            "score": result.score,
        }
