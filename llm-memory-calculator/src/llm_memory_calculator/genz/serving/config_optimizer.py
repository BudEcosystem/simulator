"""Configuration optimizer for LLM serving using GenZ analytical engine."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
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
    output_tokens: int = 1
    feasible: bool = True
    score: float = 0.0
    error: Optional[str] = None


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
        """Analyze parameter sensitivity via Morris Elementary Effects method.

        Morris (1991), Technometrics 33(2): One-at-a-time (OAT) design that
        computes elementary effects for each parameter at multiple levels.

        Returns mu_star (importance), mu (directional effect), sigma (non-linearity/
        interaction indicator) for each parameter. Rank by mu_star.
        """
        import math

        baseline_cfg = {
            "tensor_parallel": 1,
            "pipeline_parallel": 1,
            "batch_size": 32,
            "precision": "bf16",
        }

        # C3: precision levels deduped by multiplier signature (fp8/int8 are one physical level), so the
        # categorical range effect is not inflated by a duplicate evaluation.
        param_ranges: Dict[str, list] = {
            "batch_size": [1, 4, 8, 16, 32, 64, 128, 256],
            "tensor_parallel": [1, 2, 4, 8],
            "pipeline_parallel": [1, 2, 4],
            "precision": self._dedupe_precisions(["bf16", "fp8", "int8"]),
        }
        # Morris treats numeric and qualitative factors differently: numeric factors get the ordinal
        # elementary effect (delta = level step); a qualitative factor has no meaningful ordering, so its
        # effect is the score *range* across its levels.
        categorical_params = {"precision"}

        baseline_result = self._evaluate_config(baseline_cfg, input_tokens, output_tokens)
        baseline_score = self._compute_score(baseline_result, target)

        sensitivity: Dict[str, Any] = {}

        for param, values in param_ranges.items():
            # Evaluate every feasible level once. Missed-regression fix (RC-A): skip infeasible/failed
            # points entirely so the score-0 floor of a failed engine call cannot manufacture a spurious
            # elementary effect (a fabricated jump to/from 0).
            feasible_levels: List[Tuple[Any, int, float]] = []  # (value, level_index, score)
            for i, val in enumerate(values):
                cfg = dict(baseline_cfg)
                cfg[param] = val
                tp = cfg.get("tensor_parallel", 1)
                pp = cfg.get("pipeline_parallel", 1)
                if tp * pp > self._num_devices:
                    continue
                res = self._evaluate_config(cfg, input_tokens, output_tokens)
                if not res.feasible:
                    continue
                feasible_levels.append((val, i, self._compute_score(res, target)))

            if not feasible_levels:
                continue

            all_scores = [{"value": v, "score": s} for (v, _, s) in feasible_levels]
            scores = [s for (_, _, s) in feasible_levels]

            if param in categorical_params:
                # C4: range-based effect for a qualitative factor. There is no ordinal delta, so mu and
                # sigma are 0 (no directional/interaction meaning); mu_star is the max-min score spread.
                mu = 0.0
                sigma = 0.0
                mu_star = (max(scores) - min(scores)) if len(scores) > 1 else 0.0
            else:
                # Ordinal elementary effects between consecutive feasible levels: EE = dScore / dLevel.
                elementary_effects: List[float] = []
                for k in range(1, len(feasible_levels)):
                    _, idx_prev, score_prev = feasible_levels[k - 1]
                    _, idx_cur, score_cur = feasible_levels[k]
                    delta = idx_cur - idx_prev
                    if delta > 0:
                        elementary_effects.append((score_cur - score_prev) / delta)
                if not elementary_effects:
                    continue
                abs_effects = [abs(ee) for ee in elementary_effects]
                mu = sum(elementary_effects) / len(elementary_effects)
                mu_star = sum(abs_effects) / len(abs_effects)
                variance = sum((ee - mu) ** 2 for ee in elementary_effects) / len(elementary_effects)
                sigma = math.sqrt(variance)

            sensitivity[param] = {
                "mu": mu,
                "mu_star": mu_star,
                "sigma": sigma,
                "values": all_scores,
                "baseline": baseline_score,
            }

        # Rank by mu_star (parameter importance)
        ranked = sorted(sensitivity.items(), key=lambda x: x[1]["mu_star"], reverse=True)

        return {
            "sensitivity": sensitivity,
            "ranking": [p for p, _ in ranked],
            "baseline_config": baseline_cfg,
            "baseline_score": baseline_score,
            "target": target,
            "method": "morris_elementary_effects",
        }

    @staticmethod
    def _dedupe_precisions(precisions: List[str]) -> List[str]:
        """C3: collapse precisions that are physically identical to the engine.

        The GenZ ``System`` models a precision only through its ``(compute_multiplier, mem_multiplier)``
        pair, so e.g. fp8 and int8 (both 0.5x compute, 1x memory) produce byte-identical results. Keeping
        both wastes the evaluation budget and double-counts a single level in the Morris analysis. Keep the
        first precision of each multiplier signature; an unknown precision (no table entry) keeps its own
        identity so it is never silently dropped.
        """
        from llm_memory_calculator.genz.system import System
        seen = set()
        deduped: List[str] = []
        for prec in precisions:
            sig = (
                System.compute_multiplier.get(prec, prec),
                System.mem_multiplier.get(prec, prec),
            )
            if sig in seen:
                continue
            seen.add(sig)
            deduped.append(prec)
        return deduped

    def _generate_configs(self, space: SearchSpace, budget: int) -> List[Dict]:
        """Generate configuration grid, capped by budget."""
        tp_values = [tp for tp in space.tensor_parallel if tp <= self._num_devices]
        pp_values = [pp for pp in space.pipeline_parallel if pp <= self._num_devices]
        precisions = self._dedupe_precisions(space.precisions)

        configs = []
        for tp, pp, bs, prec in itertools.product(
            tp_values, pp_values, space.batch_sizes, precisions
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
            decode_lat = decode.Latency  # per-token decode latency (TPOT), in ms
            # token_throughput_tps is decode tokens/s; throughput_rps (request-completion rate) is derived
            # in C2 below from the token rate and the generation length.
            token_tps = decode.Throughput_tokens_per_sec
            # C2: request-completion rate = token rate / tokens-per-request. The engine reports
            # Throughput == Throughput_tokens_per_sec (one token per request per decode step), so reusing
            # decode.Throughput here was byte-identical to token_tps. Divide by the generation length.
            throughput = token_tps / max(output_tokens, 1)

        except Exception as e:
            # RC-A: honest failure. The engine raises ValueError on OOM (weights+KV exceed the device);
            # any other modeling failure is equally unmodelable. Do NOT fabricate plausible latencies /
            # throughput (the old fallback returned feasible-looking numbers indistinguishable from a real
            # estimate). Mark the point infeasible with zeroed metrics and record the error.
            return ConfigResult(
                config=config,
                throughput_rps=0.0,
                token_throughput_tps=0.0,
                prefill_latency_ms=0.0,
                decode_latency_ms=0.0,
                ttft_ms=0.0,
                tpot_ms=0.0,
                output_tokens=output_tokens,
                feasible=False,
                error=str(e),
            )

        return ConfigResult(
            config=config,
            throughput_rps=throughput,
            token_throughput_tps=token_tps,
            prefill_latency_ms=prefill_lat,
            decode_latency_ms=decode_lat,
            ttft_ms=prefill_lat,
            tpot_ms=decode_lat,
            output_tokens=output_tokens,
        )

    def _check_feasibility(
        self, result: ConfigResult, constraints: OptimizationConstraints,
    ) -> bool:
        """Check if a result satisfies all constraints."""
        # RC-A: a result the engine could not produce (OOM / unmodelable) is never feasible. The OOM
        # verdict is delegated to the engine's own ValueError (caught in _evaluate_config) — the single
        # source of truth — rather than a hand-written byte budget here.
        if not result.feasible:
            return False
        if result.ttft_ms > constraints.max_ttft_ms:
            return False
        if result.tpot_ms > constraints.max_tpot_ms:
            return False
        tp = result.config.get("tensor_parallel", 1)
        pp = result.config.get("pipeline_parallel", 1)
        if tp * pp > constraints.max_total_devices:
            return False
        return True

    @staticmethod
    def _e2e_latency_ms(result: ConfigResult) -> float:
        """End-to-end request latency (ms): prefill (TTFT) + per-token TPOT x generated tokens."""
        return result.prefill_latency_ms + result.tpot_ms * max(result.output_tokens, 1)

    def _compute_score(self, result: ConfigResult, target: str) -> float:
        """Compute optimization score for a result."""
        if target == "throughput":
            return result.throughput_rps
        elif target == "token_throughput":
            return result.token_throughput_tps
        elif target == "latency":
            # C1: end-to-end request latency = prefill (TTFT) + per-token TPOT x generated tokens.
            # decode_latency_ms is a single decode step (one token), so prefill + decode_latency_ms
            # understated E2E by ~output_tokens x.
            return self._e2e_latency_ms(result)
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
                    # C1: minimize end-to-end latency -> maximize its negative (Pareto sense is "max").
                    vals.append(-self._e2e_latency_ms(r))
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
            "error": result.error,
        }
