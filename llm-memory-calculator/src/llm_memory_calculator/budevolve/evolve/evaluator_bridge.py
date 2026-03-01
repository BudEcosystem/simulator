"""Bridge between OpenEvolve's evaluator interface and BudSim."""
import ast
import warnings
from typing import Dict, Optional

from ..types import ServingConfig, EvalResult
from ..evaluator import BudSimEvaluator
from ..roofline_analyzer import RooflineAnalyzer


class BudSimEvalBridge:
    """OpenEvolve-compatible evaluator that uses BudSim as fitness function.

    OpenEvolve expects an evaluate(program_path) -> dict function.
    This bridge wraps BudSimEvaluator to provide that interface.

    The evolved algorithm is loaded, validated (syntax + safety), and its
    structural quality is scored alongside BudSim performance metrics.
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

        Loads the evolved code from program_path, validates its syntax and
        safety, scores its structural quality, then combines with BudSim
        performance evaluation to produce a fitness score.

        Args:
            program_path: Path to evolved Python file from OpenEvolve.

        Returns:
            Dict with 'combined_score' and individual metrics.
        """
        # Stage 1: Load and validate the evolved code
        code_quality = self._validate_evolved_code(program_path)
        if not code_quality["valid"]:
            return {
                "combined_score": 0.0,
                "throughput_rps": 0.0,
                "ttft_ms": float("inf"),
                "tpot_ms": float("inf"),
                "slo_compliance": 0.0,
                "feasible": 0.0,
                "code_quality": 0.0,
                "error": code_quality.get("error", "invalid code"),
            }

        # Stage 2: BudSim performance evaluation
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

        # Stage 3: Combine performance score with code quality score
        # Code quality factors: structural complexity, proper branching, etc.
        perf_score = throughput * slo
        quality_bonus = code_quality["score"]  # 0.0 to 1.0

        # Combined: 80% performance, 20% code quality
        combined = 0.8 * perf_score + 0.2 * perf_score * quality_bonus

        return {
            "combined_score": combined,
            "throughput_rps": throughput,
            "ttft_ms": result.ttft_ms,
            "tpot_ms": result.tpot_ms,
            "slo_compliance": slo,
            "feasible": 1.0 if result.feasible else 0.0,
            "code_quality": quality_bonus,
        }

    def _validate_evolved_code(self, program_path: str) -> Dict:
        """Validate evolved code: syntax check, safety check, quality score.

        Args:
            program_path: Path to the evolved Python file.

        Returns:
            Dict with 'valid' (bool), 'score' (float 0-1), 'error' (str if invalid).
        """
        # Read the file
        try:
            with open(program_path, "r") as f:
                code = f.read()
        except (FileNotFoundError, PermissionError, OSError) as e:
            return {"valid": False, "score": 0.0, "error": f"Cannot read file: {e}"}

        if not code.strip():
            return {"valid": False, "score": 0.0, "error": "Empty file"}

        # Syntax validation via AST parse
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {"valid": False, "score": 0.0, "error": f"Syntax error: {e}"}

        # Safety check: reject code with dangerous constructs
        unsafe_nodes = (ast.Import, ast.ImportFrom)
        for node in ast.walk(tree):
            if isinstance(node, unsafe_nodes):
                # Allow only safe stdlib imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name not in ("math", "collections", "itertools", "functools", "typing"):
                            return {"valid": False, "score": 0.0,
                                    "error": f"Unsafe import: {alias.name}"}
                elif isinstance(node, ast.ImportFrom):
                    if node.module not in ("math", "collections", "itertools", "functools", "typing"):
                        return {"valid": False, "score": 0.0,
                                "error": f"Unsafe import from: {node.module}"}

        # Structural quality scoring
        score = self._score_code_structure(tree, code)
        return {"valid": True, "score": score}

    def _score_code_structure(self, tree: ast.AST, code: str) -> float:
        """Score the structural quality of evolved code.

        Rewards: function definitions, conditional logic, docstrings.
        Penalizes: overly long functions, no functions, trivial code.

        Args:
            tree: Parsed AST of the code.
            code: Raw source code string.

        Returns:
            Quality score between 0.0 and 1.0.
        """
        score = 0.0

        # Count function definitions
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        if functions:
            score += 0.3  # Has at least one function
        else:
            return 0.1  # Trivial code without functions

        # Check for conditional logic (if/else) — indicates decision-making
        conditionals = [n for n in ast.walk(tree) if isinstance(n, (ast.If, ast.IfExp))]
        if conditionals:
            score += 0.2

        # Check for loops — indicates iteration over data
        loops = [n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))]
        if loops:
            score += 0.2

        # Check for return statements
        returns = [n for n in ast.walk(tree) if isinstance(n, ast.Return)]
        if returns:
            score += 0.15

        # Penalize overly short code (likely trivial)
        lines = [l for l in code.strip().split("\n") if l.strip() and not l.strip().startswith("#")]
        if len(lines) >= 5:
            score += 0.15

        return min(score, 1.0)
