"""Algorithm evolution orchestrator using OpenEvolve + BudSim."""
import os
import warnings
from typing import Optional, Dict, Any

from ..types import ServingConfig
from ..evaluator import BudSimEvaluator
from ..roofline_analyzer import RooflineAnalyzer
from .evaluator_bridge import BudSimEvalBridge
from .prompt_templates import build_scheduler_prompt, build_cache_policy_prompt


# Base algorithm code for seeding evolution
_BASE_SCHEDULER = '''
def schedule_batch(queue, max_batch_size=256, max_tokens=8192):
    """Schedule a batch of requests from the queue.

    Args:
        queue: List of pending requests, each with .input_tokens, .output_tokens
        max_batch_size: Maximum requests per batch.
        max_tokens: Maximum total tokens per batch.

    Returns:
        List of requests to include in the next batch.
    """
    if not queue:
        return []

    batch = []
    total_tokens = 0

    for request in queue:
        req_tokens = request.input_tokens + request.output_tokens
        if len(batch) >= max_batch_size:
            break
        if total_tokens + req_tokens > max_tokens:
            break
        batch.append(request)
        total_tokens += req_tokens

    return batch
'''

_BASE_CACHE_POLICY = '''
def evict_entries(cache_entries, num_to_evict):
    """Select cache entries to evict under memory pressure.

    Args:
        cache_entries: List of cache entries, each with .last_access_time,
                       .frequency, .size_bytes, .prefix_length
        num_to_evict: Number of entries to remove.

    Returns:
        List of entries to evict.
    """
    sorted_entries = sorted(cache_entries, key=lambda e: e.last_access_time)
    return sorted_entries[:num_to_evict]
'''


class AlgorithmEvolver:
    """Evolve scheduling/caching algorithms using OpenEvolve + BudSim.

    Uses OpenEvolve's evolutionary loop with BudSim as the fitness function.
    Roofline analysis is injected into LLM prompts for informed mutations.
    """

    def __init__(
        self,
        model: str,
        hardware: str,
        llm_endpoint: str = "https://api.together.xyz/v1",
        llm_model: str = "openai/gpt-oss-120b",
        llm_api_key: Optional[str] = None,
    ):
        self._model = model
        self._hardware = hardware
        self._llm_endpoint = llm_endpoint
        self._llm_model = llm_model
        self._llm_api_key = llm_api_key or os.environ.get("TOGETHER_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
        self._evaluator = BudSimEvaluator()
        self._roofline = RooflineAnalyzer()

    def get_base_algorithm(self, algorithm_type: str) -> str:
        """Get seed algorithm code for evolution.

        Args:
            algorithm_type: "scheduler" or "cache_policy"
        """
        if algorithm_type == "scheduler":
            return _BASE_SCHEDULER
        elif algorithm_type == "cache_policy":
            return _BASE_CACHE_POLICY
        else:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}")

    def evolve_scheduler(
        self,
        iterations: int = 100,
        input_tokens: int = 512,
        output_tokens: int = 128,
        output_dir: str = "evolved_scheduler",
    ) -> Dict[str, Any]:
        """Evolve a batch scheduling algorithm.

        Args:
            iterations: Number of evolutionary iterations.
            input_tokens: Representative input length.
            output_tokens: Representative output length.
            output_dir: Directory to save evolved algorithms.

        Returns:
            Dict with best_code, fitness_history, improvement_pct.
        """
        base_code = self.get_base_algorithm("scheduler")

        # Get baseline metrics + roofline
        cfg = ServingConfig(model=self._model, hardware=self._hardware)
        baseline = self._evaluator.evaluate_config(cfg, input_tokens, output_tokens)

        try:
            roofline = self._roofline.analyze_config(cfg, input_tokens)
        except Exception:
            roofline = None

        # Try OpenEvolve integration
        try:
            return self._run_openevolve(
                base_code, "scheduler", baseline, roofline,
                iterations, output_dir,
            )
        except ImportError:
            warnings.warn(
                "openevolve not installed. Install with: pip install openevolve\n"
                "Returning base algorithm without evolution."
            )
            return {
                "best_code": base_code,
                "baseline_throughput": baseline.throughput_rps,
                "improvement_pct": 0.0,
                "iterations_run": 0,
                "error": "openevolve not installed",
            }

    def _run_openevolve(self, base_code, algo_type, baseline, roofline,
                        iterations, output_dir):
        """Run OpenEvolve evolutionary loop.

        Attempts OpenEvolve's class-based API first (OpenEvolve + run()),
        then falls back to the convenience function (run_evolution).
        """
        bridge = BudSimEvalBridge(
            model=self._model, hardware=self._hardware,
        )

        prompt_fn = build_scheduler_prompt if algo_type == "scheduler" else build_cache_policy_prompt

        baseline_metrics = {
            "throughput_rps": baseline.throughput_rps,
            "ttft_ms": baseline.ttft_ms,
            "tpot_ms": baseline.tpot_ms,
        }

        # Write base code to a temp file for OpenEvolve
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir=output_dir if os.path.isdir(output_dir) else None) as f:
            f.write(base_code)
            initial_program_path = f.name

        try:
            # Try class-based API first (newer OpenEvolve versions)
            try:
                from openevolve import OpenEvolve
                oe = OpenEvolve(
                    initial_program_path=initial_program_path,
                    evaluator=bridge.evaluate,
                    config={
                        "llm": {
                            "endpoint": self._llm_endpoint,
                            "model": self._llm_model,
                            "api_key": self._llm_api_key,
                        },
                        "iterations": iterations,
                        "output_dir": output_dir,
                    },
                )
                result = oe.run()
            except (ImportError, AttributeError):
                # Fall back to convenience function
                from openevolve import run_evolution
                result = run_evolution(
                    initial_program=base_code,
                    evaluator=bridge.evaluate,
                    iterations=iterations,
                    output_dir=output_dir,
                )
        finally:
            # Clean up temp file
            try:
                os.unlink(initial_program_path)
            except OSError:
                pass

        return {
            "best_code": result.get("best_program", base_code),
            "best_score": result.get("best_score", 0.0),
            "baseline_throughput": baseline.throughput_rps,
            "improvement_pct": (
                (result.get("best_score", 0) - baseline.throughput_rps)
                / max(baseline.throughput_rps, 1e-9) * 100
            ),
            "iterations_run": iterations,
        }
