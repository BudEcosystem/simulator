"""Bridge between OpenEvolve's evaluator interface and BudSim."""
from typing import Dict, Optional

from ..types import ServingConfig, EvalResult
from ..evaluator import BudSimEvaluator
from ..roofline_analyzer import RooflineAnalyzer


class BudSimEvalBridge:
    """OpenEvolve-compatible evaluator that uses BudSim as fitness function.

    OpenEvolve expects an evaluate(program_path) -> dict function.
    This bridge wraps BudSimEvaluator to provide that interface.
    """

    def __init__(
        self,
        model: str,
        hardware: str,
        input_tokens: int = 512,
        output_tokens: int = 128,
        batch_size: int = 32,
        precision: str = "bf16",
    ):
        self._model = model
        self._hardware = hardware
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._batch_size = batch_size
        self._precision = precision
        self._evaluator = BudSimEvaluator()
        self._roofline = RooflineAnalyzer()

    def evaluate(self, program_path: str) -> Dict:
        """Evaluate an evolved algorithm using BudSim.

        Args:
            program_path: Path to evolved Python file (used by OpenEvolve).

        Returns:
            Dict with 'combined_score' and individual metrics.
        """
        cfg = ServingConfig(
            model=self._model, hardware=self._hardware,
            batch_size=self._batch_size, precision=self._precision,
        )
        result = self._evaluator.evaluate_config(
            cfg, input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
        )

        throughput = result.throughput_rps
        slo = result.slo_compliance if result.slo_compliance > 0 else 1.0

        return {
            "combined_score": throughput * slo,
            "throughput_rps": throughput,
            "ttft_ms": result.ttft_ms,
            "tpot_ms": result.tpot_ms,
            "slo_compliance": slo,
            "feasible": 1.0 if result.feasible else 0.0,
        }
