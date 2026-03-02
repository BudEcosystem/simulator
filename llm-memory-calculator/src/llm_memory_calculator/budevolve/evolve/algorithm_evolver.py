"""Algorithm evolution orchestrator using LLM + BudSim.

Supports two backends:
1. Built-in evolutionary loop (default): Uses any OpenAI-compatible LLM API
   to generate algorithm mutations, BudSim to evaluate them.
2. OpenEvolve (optional): Full MAP-Elites with island model.
"""
import json
import os
import tempfile
import time
import warnings
from typing import Optional, Dict, Any, List

from ..types import ServingConfig
from ..evaluator import BudSimEvaluator
from ..roofline_analyzer import RooflineAnalyzer
from .evaluator_bridge import BudSimEvalBridge
from .prompt_templates import (
    build_scheduler_prompt, build_cache_policy_prompt,
    build_cpu_scheduler_prompt,
)


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

# CPU-optimized base algorithm (from BudEvolve analysis)
_BASE_CPU_SCHEDULER = '''
def schedule_batch_cpu(queue, max_batch_size=64, max_tokens=16384,
                       numa_nodes=2, l3_cache_mb=288.0,
                       target_ttft_ms=200.0, model_hidden_dim=4096,
                       bytes_per_param=1):
    """CPU-optimized batch scheduling for LLM inference.

    Designed for compute-bound CPU inference where batch size is the
    primary throughput lever but TTFT scales linearly with batch size.

    Args:
        queue: List of requests with .input_tokens, .output_tokens
        max_batch_size: Max requests per batch (CPU sweet spot: 32-64).
        max_tokens: Max total tokens per batch.
        numa_nodes: Number of NUMA nodes.
        l3_cache_mb: L3 cache size in MB.
        target_ttft_ms: Target TTFT SLO in ms.
        model_hidden_dim: Model hidden dimension.
        bytes_per_param: Bytes per parameter (1=INT8, 2=BF16).

    Returns:
        List of requests for the next batch.
    """
    if not queue:
        return []

    # KV cache budget from L3 capacity (60% for KV, 40% for weights/activations)
    kv_bytes_per_token = 2 * model_hidden_dim * bytes_per_param
    kv_budget_bytes = l3_cache_mb * 1024 * 1024 * 0.6
    max_kv_tokens = int(kv_budget_bytes / kv_bytes_per_token)

    # TTFT-aware batch size limit
    avg_input = sum(getattr(r, "input_tokens", 512) for r in queue) / len(queue)
    per_request_ms = (avg_input / 512.0) * 114.0
    ttft_limit = max(1, int(target_ttft_ms / max(per_request_ms, 1.0)))
    effective_limit = min(max_batch_size, ttft_limit)

    # Priority + input-length sorting
    scored = []
    for i, req in enumerate(queue):
        priority = getattr(req, "priority", 0)
        input_len = getattr(req, "input_tokens", 512)
        arrival = getattr(req, "arrival_time", i)
        score = priority * 10000 - input_len + 1.0 / (arrival + 1)
        scored.append((score, i, req))
    scored.sort(key=lambda x: -x[0])

    # Greedy bin-packing with KV cache and token constraints
    batch = []
    total_tokens = 0
    total_kv = 0
    for _, _, req in scored:
        if len(batch) >= effective_limit:
            break
        input_t = getattr(req, "input_tokens", 512)
        output_t = getattr(req, "output_tokens", 128)
        req_tokens = input_t + output_t
        if total_tokens + req_tokens > max_tokens:
            continue
        kv_t = input_t + output_t
        if total_kv + kv_t > max_kv_tokens:
            continue
        batch.append(req)
        total_tokens += req_tokens
        total_kv += kv_t

    # NUMA-aware ordering: group similar-length requests per NUMA node
    if numa_nodes > 1 and len(batch) > 1:
        batch.sort(key=lambda r: getattr(r, "input_tokens", 512))
        groups = [[] for _ in range(numa_nodes)]
        for i, req in enumerate(batch):
            groups[i % numa_nodes].append(req)
        batch = [req for g in groups for req in g]

    return batch
'''


def _call_llm_api(endpoint: str, model: str, api_key: str,
                   prompt: str, temperature: float = 0.8) -> str:
    """Call an OpenAI-compatible LLM API to generate code.

    Args:
        endpoint: API base URL (e.g. https://api.together.xyz/v1).
        model: Model name (e.g. openai/gpt-oss-120b).
        api_key: API key.
        prompt: Full prompt text.
        temperature: Sampling temperature (higher = more creative).

    Returns:
        Generated text from the LLM.

    Raises:
        RuntimeError: If the API call fails.
    """
    import urllib.request
    import urllib.error

    url = f"{endpoint.rstrip('/')}/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert algorithm designer. Generate only Python code, no explanations."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": 2048,
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            # Extract code from markdown code blocks if present
            if "```python" in content:
                code = content.split("```python")[1].split("```")[0]
                return code.strip()
            elif "```" in content:
                code = content.split("```")[1].split("```")[0]
                return code.strip()
            return content.strip()
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError,
            json.JSONDecodeError, IndexError) as e:
        raise RuntimeError(f"LLM API call failed: {e}")


class AlgorithmEvolver:
    """Evolve scheduling/caching algorithms using LLM + BudSim.

    Uses an LLM to generate algorithm mutations and BudSim's GenZ engine
    as the fitness function. Roofline analysis is injected into prompts
    so the LLM understands system bottlenecks.

    Supports:
    - Built-in evolutionary loop (no external deps beyond urllib)
    - OpenEvolve integration (optional, for MAP-Elites + island model)
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
        self._llm_api_key = (
            llm_api_key
            or os.environ.get("TOGETHER_API_KEY", "")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        self._evaluator = BudSimEvaluator()
        self._roofline = RooflineAnalyzer()

    def get_base_algorithm(self, algorithm_type: str) -> str:
        """Get seed algorithm code for evolution.

        Args:
            algorithm_type: "scheduler" or "cache_policy"

        Returns:
            String containing the base Python algorithm code.

        Raises:
            ValueError: If algorithm_type is unknown.
        """
        if algorithm_type == "scheduler":
            return _BASE_SCHEDULER
        elif algorithm_type == "cache_policy":
            return _BASE_CACHE_POLICY
        elif algorithm_type == "cpu_scheduler":
            return _BASE_CPU_SCHEDULER
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

        Runs the built-in evolutionary loop:
        1. Evaluate baseline algorithm + get roofline analysis
        2. Build prompt with code, metrics, and bottleneck data
        3. Ask LLM to generate improved version
        4. Validate and score the candidate
        5. Keep the best, repeat

        Falls back gracefully if LLM API is unavailable.

        Args:
            iterations: Number of evolutionary iterations.
            input_tokens: Representative input length.
            output_tokens: Representative output length.
            output_dir: Directory to save evolved algorithms.

        Returns:
            Dict with best_code, best_score, baseline_throughput,
            improvement_pct, iterations_run, fitness_history.
        """
        return self._evolve(
            algorithm_type="scheduler",
            iterations=iterations,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            output_dir=output_dir,
        )

    def evolve_cache_policy(
        self,
        iterations: int = 100,
        input_tokens: int = 512,
        output_tokens: int = 128,
        output_dir: str = "evolved_cache_policy",
    ) -> Dict[str, Any]:
        """Evolve a KV cache eviction policy.

        Same evolutionary loop as evolve_scheduler but operates on
        cache eviction algorithms.

        Args:
            iterations: Number of evolutionary iterations.
            input_tokens: Representative input length.
            output_tokens: Representative output length.
            output_dir: Directory to save evolved algorithms.

        Returns:
            Dict with best_code, best_score, baseline metrics, and history.
        """
        return self._evolve(
            algorithm_type="cache_policy",
            iterations=iterations,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            output_dir=output_dir,
        )

    def evolve_cpu_scheduler(
        self,
        iterations: int = 100,
        input_tokens: int = 512,
        output_tokens: int = 128,
        output_dir: str = "evolved_cpu_scheduler",
    ) -> Dict[str, Any]:
        """Evolve a CPU-optimized batch scheduling algorithm.

        Uses CPU-specific prompt template that includes:
        - Compute-bound nature of CPU inference
        - NUMA topology awareness
        - L3 cache budget constraints
        - TTFT-aware batch sizing
        - ISA-specific optimization hints

        Args:
            iterations: Number of evolutionary iterations.
            input_tokens: Representative input length.
            output_tokens: Representative output length.
            output_dir: Directory to save evolved algorithms.

        Returns:
            Dict with best_code, best_score, baseline metrics, and history.
        """
        return self._evolve(
            algorithm_type="cpu_scheduler",
            iterations=iterations,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            output_dir=output_dir,
        )

    def _evolve(
        self, algorithm_type: str, iterations: int,
        input_tokens: int, output_tokens: int, output_dir: str,
    ) -> Dict[str, Any]:
        """Core evolutionary loop shared by scheduler and cache policy evolution.

        Args:
            algorithm_type: "scheduler" or "cache_policy".
            iterations: Number of evolutionary iterations.
            input_tokens: Representative input length.
            output_tokens: Representative output length.
            output_dir: Directory to save evolved algorithms.

        Returns:
            Dict with evolution results.
        """
        base_code = self.get_base_algorithm(algorithm_type)

        # Get baseline metrics
        cfg = ServingConfig(model=self._model, hardware=self._hardware)
        baseline = self._evaluator.evaluate_config(cfg, input_tokens, output_tokens)
        baseline_metrics = {
            "throughput_rps": baseline.throughput_rps,
            "token_throughput_tps": baseline.token_throughput_tps,
            "ttft_ms": baseline.ttft_ms,
            "tpot_ms": baseline.tpot_ms,
            "memory_gb": baseline.memory_gb,
            "power_w": baseline.power_w,
        }

        # Get roofline analysis
        try:
            roofline = self._roofline.analyze_config(cfg, input_tokens)
        except Exception:
            roofline = None

        # Build the evaluation bridge for scoring candidates
        bridge = BudSimEvalBridge(
            model=self._model, hardware=self._hardware,
            input_tokens=input_tokens, output_tokens=output_tokens,
        )

        # Check if we have an API key
        if not self._llm_api_key:
            warnings.warn(
                "No LLM API key found. Set TOGETHER_API_KEY or OPENAI_API_KEY "
                "environment variable. Returning base algorithm without evolution."
            )
            return {
                "best_code": base_code,
                "best_score": 0.0,
                "baseline_throughput": baseline.throughput_rps,
                "baseline_metrics": baseline_metrics,
                "improvement_pct": 0.0,
                "iterations_run": 0,
                "fitness_history": [],
                "error": "no LLM API key",
            }

        # Try OpenEvolve first if available
        try:
            return self._run_openevolve(
                base_code, algorithm_type, baseline, baseline_metrics,
                roofline, bridge, iterations, output_dir,
            )
        except ImportError:
            pass

        # Built-in evolutionary loop
        return self._run_builtin_evolution(
            base_code, algorithm_type, baseline, baseline_metrics,
            roofline, bridge, iterations, output_dir,
        )

    def _run_builtin_evolution(
        self, base_code: str, algo_type: str,
        baseline, baseline_metrics: Dict, roofline, bridge: BudSimEvalBridge,
        iterations: int, output_dir: str,
    ) -> Dict[str, Any]:
        """Built-in evolutionary loop using direct LLM API calls.

        For each iteration:
        1. Build a prompt with current best code, metrics, and roofline analysis
        2. Call the LLM API to generate a mutated algorithm
        3. Write the candidate to a temp file and evaluate via the bridge
        4. Keep the candidate if it improves on the best score

        Args:
            base_code: Seed algorithm code.
            algo_type: "scheduler" or "cache_policy".
            baseline: EvalResult from baseline evaluation.
            baseline_metrics: Dict of baseline metric values.
            roofline: RooflineReport or None.
            bridge: BudSimEvalBridge for scoring candidates.
            iterations: Number of evolutionary iterations.
            output_dir: Directory for saving results.

        Returns:
            Dict with evolution results.
        """
        if algo_type == "cpu_scheduler":
            prompt_fn = build_cpu_scheduler_prompt
        elif algo_type == "scheduler":
            prompt_fn = build_scheduler_prompt
        else:
            prompt_fn = build_cache_policy_prompt

        best_code = base_code
        best_score = 0.0
        fitness_history: List[Dict] = []

        # Score the baseline
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(base_code)
            base_path = f.name
        try:
            base_result = bridge.evaluate(base_path)
            best_score = base_result.get("combined_score", 0.0)
        finally:
            os.unlink(base_path)

        fitness_history.append({
            "iteration": 0, "score": best_score, "is_baseline": True,
        })

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        for i in range(1, iterations + 1):
            try:
                # Build prompt with current best + roofline insights
                prompt = prompt_fn(
                    current_code=best_code,
                    metrics=baseline_metrics,
                    roofline=roofline,
                )

                # Get LLM mutation
                # Vary temperature: start creative (0.9), get more focused (0.4)
                temp = max(0.4, 0.9 - (i / iterations) * 0.5)
                candidate_code = _call_llm_api(
                    endpoint=self._llm_endpoint,
                    model=self._llm_model,
                    api_key=self._llm_api_key,
                    prompt=prompt,
                    temperature=temp,
                )

                # Evaluate candidate
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".py", delete=False, dir=output_dir,
                ) as f:
                    f.write(candidate_code)
                    candidate_path = f.name

                try:
                    result = bridge.evaluate(candidate_path)
                    score = result.get("combined_score", 0.0)

                    fitness_history.append({
                        "iteration": i, "score": score,
                        "code_quality": result.get("code_quality", 0.0),
                    })

                    if score > best_score:
                        best_score = score
                        best_code = candidate_code
                        # Save the new best
                        best_path = os.path.join(output_dir, f"best_gen_{i}.py")
                        with open(best_path, "w") as bf:
                            bf.write(candidate_code)
                finally:
                    # Clean up temp file (keep saved bests)
                    try:
                        os.unlink(candidate_path)
                    except OSError:
                        pass

            except RuntimeError as e:
                warnings.warn(f"Iteration {i} failed: {e}")
                fitness_history.append({
                    "iteration": i, "score": best_score, "error": str(e),
                })
                continue

        # Save final best
        final_path = os.path.join(output_dir, "best_final.py")
        with open(final_path, "w") as f:
            f.write(best_code)

        initial_score = fitness_history[0]["score"] if fitness_history else 0.0
        improvement = (
            (best_score - initial_score) / max(initial_score, 1e-9) * 100
            if initial_score > 0 else 0.0
        )

        return {
            "best_code": best_code,
            "best_score": best_score,
            "baseline_throughput": baseline.throughput_rps,
            "baseline_metrics": baseline_metrics,
            "improvement_pct": improvement,
            "iterations_run": len([h for h in fitness_history if "error" not in h]) - 1,
            "total_iterations": iterations,
            "fitness_history": fitness_history,
            "output_dir": output_dir,
        }

    def _run_openevolve(
        self, base_code, algo_type, baseline, baseline_metrics,
        roofline, bridge, iterations, output_dir,
    ):
        """Run OpenEvolve evolutionary loop (optional backend).

        OpenEvolve provides MAP-Elites with island model for better
        diversity than the built-in loop. Requires the openevolve package.

        The evaluator bridge is written as a standalone Python file that
        OpenEvolve can import and call.
        """
        from openevolve import OpenEvolve as OEClass

        os.makedirs(output_dir, exist_ok=True)

        # Write the initial program file
        program_path = os.path.join(output_dir, "initial_program.py")
        with open(program_path, "w") as f:
            f.write(base_code)

        # Write the evaluator as a standalone file for OpenEvolve
        evaluator_path = os.path.join(output_dir, "evaluator.py")
        self._write_evaluator_file(evaluator_path, bridge)

        # Write config YAML
        config_path = os.path.join(output_dir, "config.yaml")
        self._write_config_file(config_path, iterations)

        # Run OpenEvolve
        import asyncio
        oe = OEClass(
            initial_program_path=program_path,
            evaluator_path=evaluator_path,
            config=config_path,
        )

        # Handle async run
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(asyncio.run, oe.run()).result()
            else:
                result = loop.run_until_complete(oe.run())
        except RuntimeError:
            result = asyncio.run(oe.run())

        best_program = base_code
        best_score = 0.0
        if isinstance(result, dict):
            best_program = result.get("best_program", base_code)
            best_score = result.get("best_score", 0.0)

        return {
            "best_code": best_program,
            "best_score": best_score,
            "baseline_throughput": baseline.throughput_rps,
            "baseline_metrics": baseline_metrics,
            "improvement_pct": (
                (best_score - baseline.throughput_rps)
                / max(baseline.throughput_rps, 1e-9) * 100
            ),
            "iterations_run": iterations,
            "backend": "openevolve",
            "output_dir": output_dir,
        }

    def _write_evaluator_file(self, path: str, bridge: BudSimEvalBridge):
        """Write a standalone evaluator.py for OpenEvolve.

        OpenEvolve imports the evaluator module and calls its evaluate()
        function with a program_path argument.
        """
        code = f'''"""BudSim evaluator for OpenEvolve."""
import ast
import sys
sys.path.insert(0, "{os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}")

from llm_memory_calculator.budevolve.evolve.evaluator_bridge import BudSimEvalBridge

_bridge = BudSimEvalBridge(
    model="{bridge._model}",
    hardware="{bridge._hardware}",
    input_tokens={bridge._input_tokens},
    output_tokens={bridge._output_tokens},
    batch_size={bridge._batch_size},
    precision="{bridge._precision}",
)

async def evaluate(program_path: str) -> dict:
    """OpenEvolve evaluator entry point."""
    return _bridge.evaluate(program_path)
'''
        with open(path, "w") as f:
            f.write(code)

    def _write_config_file(self, path: str, iterations: int):
        """Write OpenEvolve config YAML."""
        config = {
            "num_iterations": iterations,
            "num_islands": 3,
            "migration_interval": max(iterations // 5, 10),
            "llm": {
                "primary_model": self._llm_model,
                "api_base": self._llm_endpoint,
            },
        }
        # Write as JSON (YAML-compatible subset)
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
