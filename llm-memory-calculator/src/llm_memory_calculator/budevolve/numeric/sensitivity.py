"""Sensitivity analysis for config and hardware parameters."""
import dataclasses
from typing import Dict, List, Optional, Any

from ..types import ServingConfig, HardwareSpec, EvalResult
from ..evaluator import BudSimEvaluator


def _get_hardware_spec(hardware_name: str) -> HardwareSpec:
    """Look up real hardware specs from the hardware config registry.

    Args:
        hardware_name: Hardware name (e.g. "H100_GPU", "A100_80GB_GPU").

    Returns:
        HardwareSpec with real values from the config database.
    """
    try:
        from llm_memory_calculator.hardware.configs import HARDWARE_CONFIGS
        cfg = HARDWARE_CONFIGS.get(hardware_name, {})
        if cfg:
            cost = cfg.get("cost", {})
            return HardwareSpec(
                flops_tflops=float(cfg.get("Flops", 990)),
                offchip_mem_bw_gbps=float(cfg.get("Memory_BW", 3350)),
                off_chip_mem_size_gb=float(cfg.get("Memory_size", 80)),
                on_chip_mem_size_mb=50.0,  # Not in config, use reasonable default
                onchip_mem_bw_gbps=33000.0,
                interchip_link_bw_gbps=float(cfg.get("ICN", 450)),
                frequency_ghz=1.98,
                tdp_watts=float(cost.get("tdp_watts", 700)),
                estimated_cost_usd=float(cost.get("purchase_price_usd", 25000)),
            )
    except ImportError:
        pass

    # Fallback defaults (H100-like)
    return HardwareSpec(
        flops_tflops=990.0, offchip_mem_bw_gbps=3350.0,
        off_chip_mem_size_gb=80.0, on_chip_mem_size_mb=50.0,
        onchip_mem_bw_gbps=33000.0, interchip_link_bw_gbps=450.0,
        frequency_ghz=1.98,
    )


class SensitivityAnalyzer:
    """Parameter sensitivity analysis using one-at-a-time (OAT) method.

    Measures how much each parameter affects the target metric by varying
    one parameter at a time while holding others at baseline.
    """

    def __init__(self, model: str, hardware: str,
                 evaluator: Optional[BudSimEvaluator] = None):
        self._model = model
        self._hardware = hardware
        self._evaluator = evaluator or BudSimEvaluator()

    def analyze(
        self,
        target: str = "throughput",
        input_tokens: int = 512,
        output_tokens: int = 128,
    ) -> Dict[str, Any]:
        """Analyze sensitivity of serving config parameters.

        Varies each parameter independently while holding others at baseline,
        measuring the maximum absolute change in the target metric.

        Args:
            target: Metric to analyze ("throughput" or "latency").
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Dict with 'ranking' (sorted params), 'scores' (max delta per param),
            'details' (per-value scores), and 'baseline'.
        """
        params = {
            "tensor_parallel": [1, 2, 4, 8],
            "pipeline_parallel": [1, 2, 4],
            "batch_size": [1, 8, 32, 64, 128, 256],
            "precision": ["bf16", "fp8", "int8"],
        }

        baseline_cfg = ServingConfig(model=self._model, hardware=self._hardware)
        baseline = self._evaluator.evaluate_config(baseline_cfg, input_tokens, output_tokens)
        baseline_score = self._get_score(baseline, target)

        scores = {}
        details = {}
        for param, values in params.items():
            max_delta = 0.0
            param_scores = []
            for val in values:
                cfg = ServingConfig(model=self._model, hardware=self._hardware)
                if param == "precision":
                    cfg.precision = val
                elif param == "tensor_parallel":
                    cfg.tensor_parallel = val
                elif param == "pipeline_parallel":
                    cfg.pipeline_parallel = val
                elif param == "batch_size":
                    cfg.batch_size = val

                result = self._evaluator.evaluate_config(cfg, input_tokens, output_tokens)
                score = self._get_score(result, target)
                delta = abs(score - baseline_score)
                max_delta = max(max_delta, delta)
                param_scores.append({"value": val, "score": score, "delta": delta})

            scores[param] = max_delta
            details[param] = param_scores

        ranking = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return {
            "ranking": ranking, "scores": scores, "details": details,
            "baseline": baseline_score, "target": target,
        }

    def analyze_hardware(
        self,
        target: str = "throughput",
        input_tokens: int = 512,
        output_tokens: int = 128,
    ) -> Dict[str, Any]:
        """Analyze sensitivity of hardware parameters.

        Uses real hardware specs from the config database (not hardcoded values)
        as the baseline, then sweeps each hardware parameter independently.

        Args:
            target: Metric to analyze ("throughput" or "latency").
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Dict with 'ranking', 'scores', 'details', 'baseline', and
            'base_hardware' name.
        """
        base_hw = _get_hardware_spec(self._hardware)
        baseline = self._evaluator.evaluate_hardware(
            base_hw, model=self._model, input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        baseline_score = self._get_score(baseline, target)

        # Build sweep ranges around the baseline values (0.25x to 4x)
        params = {
            "flops_tflops": _build_sweep(base_hw.flops_tflops, 0.25, 4.0, 5),
            "offchip_mem_bw_gbps": _build_sweep(base_hw.offchip_mem_bw_gbps, 0.25, 4.0, 5),
            "off_chip_mem_size_gb": _build_sweep(base_hw.off_chip_mem_size_gb, 0.5, 3.0, 4),
            "interchip_link_bw_gbps": _build_sweep(base_hw.interchip_link_bw_gbps, 0.25, 4.0, 4),
        }

        scores = {}
        details = {}
        for param, values in params.items():
            max_delta = 0.0
            param_scores = []
            for val in values:
                hw = dataclasses.replace(base_hw, **{param: float(val)})
                result = self._evaluator.evaluate_hardware(
                    hw, model=self._model,
                    input_tokens=input_tokens, output_tokens=output_tokens,
                )
                score = self._get_score(result, target)
                delta = abs(score - baseline_score)
                max_delta = max(max_delta, delta)
                param_scores.append({"value": val, "score": score, "delta": delta})
            scores[param] = max_delta
            details[param] = param_scores

        ranking = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return {
            "ranking": ranking, "scores": scores, "details": details,
            "baseline": baseline_score, "base_hardware": self._hardware,
            "target": target,
        }

    def _get_score(self, result: EvalResult, target: str) -> float:
        if target == "throughput":
            return result.throughput_rps
        elif target == "latency":
            return -(result.ttft_ms + result.tpot_ms)
        elif target == "cost":
            return -result.cost_per_million_tokens
        elif target == "power":
            return -result.power_w
        return result.throughput_rps


def _build_sweep(base_val: float, lo_frac: float, hi_frac: float, n: int) -> list:
    """Build a list of sweep values as fractions of the base value."""
    import numpy as np
    lo = base_val * lo_frac
    hi = base_val * hi_frac
    return [round(v, 2) for v in np.linspace(lo, hi, n).tolist()]
