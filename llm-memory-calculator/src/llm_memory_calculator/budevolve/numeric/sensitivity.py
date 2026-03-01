"""Sensitivity analysis for config and hardware parameters."""
import dataclasses
from typing import Dict, List, Optional, Any

from ..types import ServingConfig, HardwareSpec, EvalResult
from ..evaluator import BudSimEvaluator


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

        Returns dict with 'ranking' (sorted params) and 'scores' (impact per param).
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
        for param, values in params.items():
            max_delta = 0.0
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

            scores[param] = max_delta

        ranking = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return {"ranking": ranking, "scores": scores, "baseline": baseline_score}

    def analyze_hardware(
        self,
        target: str = "throughput",
        input_tokens: int = 512,
        output_tokens: int = 128,
    ) -> Dict[str, Any]:
        """Analyze sensitivity of hardware parameters."""
        base_hw = HardwareSpec(
            flops_tflops=990.0, offchip_mem_bw_gbps=3350.0,
            off_chip_mem_size_gb=80.0, on_chip_mem_size_mb=50.0,
            onchip_mem_bw_gbps=33000.0, interchip_link_bw_gbps=450.0,
            frequency_ghz=1.98,
        )
        baseline = self._evaluator.evaluate_hardware(
            base_hw, model=self._model, input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        baseline_score = self._get_score(baseline, target)

        params = {
            "flops_tflops": [200, 500, 990, 2000, 5000],
            "offchip_mem_bw_gbps": [1000, 2000, 3350, 5000, 8000],
            "off_chip_mem_size_gb": [40, 80, 128, 192],
            "interchip_link_bw_gbps": [50, 150, 450, 900],
        }

        scores = {}
        for param, values in params.items():
            max_delta = 0.0
            for val in values:
                hw = dataclasses.replace(base_hw, **{param: float(val)})
                result = self._evaluator.evaluate_hardware(
                    hw, model=self._model,
                    input_tokens=input_tokens, output_tokens=output_tokens,
                )
                delta = abs(self._get_score(result, target) - baseline_score)
                max_delta = max(max_delta, delta)
            scores[param] = max_delta

        ranking = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return {"ranking": ranking, "scores": scores, "baseline": baseline_score}

    def _get_score(self, result: EvalResult, target: str) -> float:
        if target == "throughput":
            return result.throughput_rps
        elif target == "latency":
            return -(result.ttft_ms + result.tpot_ms)
        return result.throughput_rps
