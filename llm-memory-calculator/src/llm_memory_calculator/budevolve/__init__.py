"""BudEvolve: Reverse-optimization system for LLM serving.

Three optimization modes:
- NumericOptimizer: pymoo NSGA-II config search (no LLM)
- HardwareExplorer: pymoo NSGA-II hardware DSE (no LLM)
- AlgorithmEvolver: OpenEvolve algorithm evolution (LLM required)

Usage:
    from llm_memory_calculator.budevolve import NumericOptimizer, HardwareExplorer
    optimizer = NumericOptimizer(model="llama-70b", hardware="H100_GPU")
    result = optimizer.optimize(objectives=["throughput", "latency"])
"""
from .types import (
    ServingConfig, HardwareSpec, EvalResult,
    OperatorInsight, RooflineReport, ParetoResult,
)
from .evaluator import BudSimEvaluator
from .roofline_analyzer import RooflineAnalyzer
from .numeric.config_optimizer import NumericOptimizer
from .numeric.hardware_explorer import HardwareExplorer
from .numeric.sensitivity import SensitivityAnalyzer

__all__ = [
    "ServingConfig", "HardwareSpec", "EvalResult",
    "OperatorInsight", "RooflineReport", "ParetoResult",
    "BudSimEvaluator", "RooflineAnalyzer",
    "NumericOptimizer", "HardwareExplorer", "SensitivityAnalyzer",
]
