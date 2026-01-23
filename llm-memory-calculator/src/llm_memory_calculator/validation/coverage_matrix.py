"""
Coverage Matrix for Ground Truth Dataset.

Tracks benchmark coverage across multiple dimensions:
- Training type (Pretraining, SFT, DPO, PPO, LoRA, QLoRA, etc.)
- Quantization (BF16, FP16, FP8, INT8, NF4)
- GPU scale (1-8, 8-64, 64-256, 256-1000, 1000+)
- Hardware (H100, A100, MI300X, TPU, Gaudi)
- Model size (<10B, 10-50B, 50-100B, 100-500B, >500B)
- Optimizer (AdamW, 8-bit AdamW, GaLore, Lion, etc.)

Identifies gaps and generates collection priorities.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .benchmark_schema import (
    ExtendedBenchmark,
    GPUScale,
    HardwareType,
    ModelSizeCategory,
    OptimizerType,
    QuantizationMethod,
    TrainingType,
)

logger = logging.getLogger(__name__)


# Target coverage per cell
DEFAULT_MIN_COVERAGE = 5  # Minimum benchmarks per important cell
CRITICAL_MIN_COVERAGE = 10  # For critical combinations


@dataclass
class CoverageCell:
    """Represents a single cell in the coverage matrix."""
    dimensions: Dict[str, str]  # Dimension name -> value
    benchmarks: List[str] = field(default_factory=list)  # Benchmark IDs
    count: int = 0
    is_critical: bool = False

    @property
    def coverage_met(self) -> bool:
        """Check if minimum coverage is met."""
        target = CRITICAL_MIN_COVERAGE if self.is_critical else DEFAULT_MIN_COVERAGE
        return self.count >= target

    @property
    def gap(self) -> int:
        """Number of benchmarks needed to meet target."""
        target = CRITICAL_MIN_COVERAGE if self.is_critical else DEFAULT_MIN_COVERAGE
        return max(0, target - self.count)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dimensions": self.dimensions,
            "count": self.count,
            "benchmark_ids": self.benchmarks,
            "is_critical": self.is_critical,
            "coverage_met": self.coverage_met,
            "gap": self.gap,
        }


@dataclass
class CoverageReport:
    """Coverage analysis report."""
    total_benchmarks: int = 0
    total_cells: int = 0
    covered_cells: int = 0
    partially_covered_cells: int = 0
    uncovered_cells: int = 0

    coverage_by_dimension: Dict[str, Dict[str, int]] = field(default_factory=dict)
    gaps: List[CoverageCell] = field(default_factory=list)
    priorities: List[Dict[str, Any]] = field(default_factory=list)

    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def overall_coverage(self) -> float:
        """Overall coverage percentage."""
        if self.total_cells == 0:
            return 0.0
        return self.covered_cells / self.total_cells

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_benchmarks": self.total_benchmarks,
            "total_cells": self.total_cells,
            "covered_cells": self.covered_cells,
            "partially_covered_cells": self.partially_covered_cells,
            "uncovered_cells": self.uncovered_cells,
            "overall_coverage": self.overall_coverage,
            "coverage_by_dimension": self.coverage_by_dimension,
            "gaps": [g.to_dict() for g in self.gaps],
            "priorities": self.priorities,
            "generated_at": self.generated_at,
        }


# Critical combinations that require higher coverage
CRITICAL_COMBINATIONS: List[Dict[str, str]] = [
    # High-priority pretraining benchmarks
    {"training_type": "pretraining", "gpu_scale": "hyperscale"},
    {"training_type": "pretraining", "model_size": "xlarge"},
    {"training_type": "pretraining", "model_size": "ultra"},

    # Fine-tuning benchmarks
    {"training_type": "sft", "hardware": "h100_sxm"},
    {"training_type": "lora", "hardware": "a100_80gb"},
    {"training_type": "qlora", "model_size": "large"},

    # RLHF benchmarks
    {"training_type": "ppo", "hardware": "h100_sxm"},
    {"training_type": "dpo", "hardware": "h100_sxm"},

    # Quantization benchmarks
    {"quantization": "fp8_e4m3", "hardware": "h100_sxm"},
    {"quantization": "nf4", "training_type": "qlora"},
]


class CoverageMatrix:
    """
    Tracks and analyzes benchmark coverage across dimensions.

    Dimensions tracked:
    - training_type: Type of training (pretraining, sft, dpo, etc.)
    - quantization: Precision/quantization method
    - gpu_scale: Number of GPUs
    - hardware: Hardware type
    - model_size: Model size category
    - optimizer: Optimizer type
    """

    DIMENSIONS = [
        "training_type",
        "quantization",
        "gpu_scale",
        "hardware",
        "model_size",
        "optimizer",
    ]

    def __init__(
        self,
        benchmarks: Optional[List[ExtendedBenchmark]] = None,
        min_coverage: int = DEFAULT_MIN_COVERAGE,
    ):
        """
        Initialize coverage matrix.

        Args:
            benchmarks: Initial list of benchmarks
            min_coverage: Minimum benchmarks per cell
        """
        self.min_coverage = min_coverage
        self._benchmarks: Dict[str, ExtendedBenchmark] = {}
        self._coverage: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

        if benchmarks:
            for benchmark in benchmarks:
                self.add_benchmark(benchmark)

    def add_benchmark(self, benchmark: ExtendedBenchmark) -> None:
        """Add a benchmark to the coverage matrix."""
        bid = benchmark.benchmark_id
        self._benchmarks[bid] = benchmark

        # Update coverage for each dimension
        dims = self._get_dimensions(benchmark)
        for dim, value in dims.items():
            self._coverage[dim][value].add(bid)

    def remove_benchmark(self, benchmark_id: str) -> None:
        """Remove a benchmark from the coverage matrix."""
        if benchmark_id not in self._benchmarks:
            return

        benchmark = self._benchmarks[benchmark_id]
        dims = self._get_dimensions(benchmark)

        for dim, value in dims.items():
            self._coverage[dim][value].discard(benchmark_id)

        del self._benchmarks[benchmark_id]

    def _get_dimensions(self, benchmark: ExtendedBenchmark) -> Dict[str, str]:
        """Extract dimension values from a benchmark."""
        return {
            "training_type": benchmark.training_type.value,
            "quantization": benchmark.quantization.model_precision.value,
            "gpu_scale": benchmark.gpu_scale.value,
            "hardware": benchmark.hardware_type.value,
            "model_size": benchmark.model_size_category.value,
            "optimizer": benchmark.optimizer.value,
        }

    def get_coverage_by_dimension(self, dimension: str) -> Dict[str, int]:
        """Get coverage counts for a single dimension."""
        if dimension not in self._coverage:
            return {}
        return {k: len(v) for k, v in self._coverage[dimension].items()}

    def get_coverage_matrix_2d(
        self,
        dim1: str,
        dim2: str,
    ) -> Dict[Tuple[str, str], int]:
        """
        Get 2D coverage matrix for two dimensions.

        Args:
            dim1: First dimension
            dim2: Second dimension

        Returns:
            Dictionary mapping (dim1_value, dim2_value) -> count
        """
        matrix = defaultdict(int)

        for benchmark in self._benchmarks.values():
            dims = self._get_dimensions(benchmark)
            key = (dims.get(dim1, "unknown"), dims.get(dim2, "unknown"))
            matrix[key] += 1

        return dict(matrix)

    def identify_gaps(
        self,
        dimensions: Optional[List[str]] = None,
    ) -> List[CoverageCell]:
        """
        Identify coverage gaps.

        Args:
            dimensions: Dimensions to check (default: all)

        Returns:
            List of cells with insufficient coverage
        """
        dims = dimensions or self.DIMENSIONS
        gaps = []

        for dim in dims:
            # Get all possible values for this dimension
            all_values = self._get_all_values_for_dimension(dim)

            for value in all_values:
                count = len(self._coverage[dim].get(value, set()))

                if count < self.min_coverage:
                    cell = CoverageCell(
                        dimensions={dim: value},
                        benchmarks=list(self._coverage[dim].get(value, set())),
                        count=count,
                        is_critical=self._is_critical_cell({dim: value}),
                    )
                    gaps.append(cell)

        # Sort by gap size (largest first) and criticality
        gaps.sort(key=lambda x: (-x.is_critical, -x.gap))
        return gaps

    def _get_all_values_for_dimension(self, dimension: str) -> List[str]:
        """Get all possible values for a dimension."""
        value_map = {
            "training_type": [t.value for t in TrainingType],
            "quantization": [q.value for q in QuantizationMethod],
            "gpu_scale": [g.value for g in GPUScale],
            "hardware": [h.value for h in HardwareType],
            "model_size": [m.value for m in ModelSizeCategory],
            "optimizer": [o.value for o in OptimizerType],
        }
        return value_map.get(dimension, [])

    def _is_critical_cell(self, dimensions: Dict[str, str]) -> bool:
        """Check if a cell is marked as critical."""
        for critical in CRITICAL_COMBINATIONS:
            if all(dimensions.get(k) == v for k, v in critical.items()):
                return True
        return False

    def generate_priorities(self, top_n: int = 20) -> List[Dict[str, Any]]:
        """
        Generate prioritized list of gaps to fill.

        Args:
            top_n: Number of top priorities to return

        Returns:
            List of priority items with suggested actions
        """
        gaps = self.identify_gaps()
        priorities = []

        for gap in gaps[:top_n]:
            priority = {
                "dimensions": gap.dimensions,
                "current_count": gap.count,
                "target_count": CRITICAL_MIN_COVERAGE if gap.is_critical else self.min_coverage,
                "gap": gap.gap,
                "is_critical": gap.is_critical,
                "suggested_sources": self._suggest_sources(gap.dimensions),
                "priority_score": self._calculate_priority_score(gap),
            }
            priorities.append(priority)

        return priorities

    def _suggest_sources(self, dimensions: Dict[str, str]) -> List[str]:
        """Suggest data sources for filling a gap."""
        suggestions = []

        training_type = dimensions.get("training_type")
        hardware = dimensions.get("hardware")

        # Suggest based on training type
        if training_type in ("pretraining", "continued_pretraining"):
            suggestions.extend([
                "arXiv papers (LLaMA, Mistral, DeepSeek)",
                "MLPerf Training submissions",
                "Company technical reports",
            ])
        elif training_type in ("sft", "lora", "qlora"):
            suggestions.extend([
                "HuggingFace model cards",
                "GitHub: LLaMA-Factory, TRL",
                "Community benchmarks",
            ])
        elif training_type in ("dpo", "ppo", "grpo", "kto"):
            suggestions.extend([
                "GitHub: OpenRLHF, TRL",
                "RLHF papers (Anthropic, OpenAI)",
                "Post-training technical reports",
            ])

        # Suggest based on hardware
        if hardware and "tpu" in hardware:
            suggestions.append("Google research papers and TPU blogs")
        elif hardware and "mi" in hardware:
            suggestions.append("AMD ROCm documentation and benchmarks")
        elif hardware and "gaudi" in hardware:
            suggestions.append("Intel Habana documentation")

        return suggestions

    def _calculate_priority_score(self, gap: CoverageCell) -> float:
        """Calculate priority score for a gap (higher = more important)."""
        base_score = gap.gap * 10

        # Critical cells get 2x weight
        if gap.is_critical:
            base_score *= 2

        # Common dimensions get higher priority
        dim = list(gap.dimensions.keys())[0]
        dim_weights = {
            "training_type": 1.5,
            "hardware": 1.3,
            "model_size": 1.2,
            "quantization": 1.1,
            "optimizer": 1.0,
            "gpu_scale": 1.0,
        }
        base_score *= dim_weights.get(dim, 1.0)

        return base_score

    def generate_report(self) -> CoverageReport:
        """Generate comprehensive coverage report."""
        report = CoverageReport(
            total_benchmarks=len(self._benchmarks),
        )

        # Calculate coverage by dimension
        for dim in self.DIMENSIONS:
            report.coverage_by_dimension[dim] = self.get_coverage_by_dimension(dim)

        # Count cells
        gaps = self.identify_gaps()
        total_cells = sum(len(self._get_all_values_for_dimension(d)) for d in self.DIMENSIONS)

        covered = sum(
            1 for dim in self.DIMENSIONS
            for value, bids in self._coverage[dim].items()
            if len(bids) >= self.min_coverage
        )
        partial = sum(
            1 for dim in self.DIMENSIONS
            for value, bids in self._coverage[dim].items()
            if 0 < len(bids) < self.min_coverage
        )

        report.total_cells = total_cells
        report.covered_cells = covered
        report.partially_covered_cells = partial
        report.uncovered_cells = total_cells - covered - partial
        report.gaps = gaps
        report.priorities = self.generate_priorities()

        return report

    def print_report(self, detailed: bool = False) -> None:
        """Print coverage report to console."""
        report = self.generate_report()

        print("\n" + "=" * 60)
        print("COVERAGE MATRIX REPORT")
        print("=" * 60)
        print(f"\nTotal benchmarks: {report.total_benchmarks}")
        print(f"Overall coverage: {report.overall_coverage:.1%}")
        print(f"  - Fully covered cells: {report.covered_cells}")
        print(f"  - Partially covered: {report.partially_covered_cells}")
        print(f"  - Uncovered: {report.uncovered_cells}")

        print("\n--- Coverage by Dimension ---")
        for dim, values in report.coverage_by_dimension.items():
            print(f"\n{dim.upper()}:")
            for value, count in sorted(values.items(), key=lambda x: -x[1]):
                status = "OK" if count >= self.min_coverage else f"NEED +{self.min_coverage - count}"
                print(f"  {value}: {count} [{status}]")

        if detailed and report.gaps:
            print("\n--- Top Coverage Gaps ---")
            for i, gap in enumerate(report.gaps[:10], 1):
                crit = "[CRITICAL]" if gap.is_critical else ""
                print(f"  {i}. {gap.dimensions} - need +{gap.gap} {crit}")

        if detailed and report.priorities:
            print("\n--- Collection Priorities ---")
            for i, priority in enumerate(report.priorities[:5], 1):
                print(f"  {i}. {priority['dimensions']}")
                print(f"     Gap: {priority['gap']}, Score: {priority['priority_score']:.1f}")
                print(f"     Sources: {', '.join(priority['suggested_sources'][:2])}")

        print("\n" + "=" * 60)

    @classmethod
    def from_database(cls, benchmarks: List[ExtendedBenchmark]) -> "CoverageMatrix":
        """Create coverage matrix from benchmark database."""
        return cls(benchmarks=benchmarks)


def analyze_coverage(
    benchmarks: List[ExtendedBenchmark],
    print_report: bool = True,
) -> CoverageReport:
    """
    Analyze coverage of a benchmark collection.

    Args:
        benchmarks: List of benchmarks to analyze
        print_report: Whether to print report to console

    Returns:
        CoverageReport with analysis results
    """
    matrix = CoverageMatrix(benchmarks=benchmarks)

    if print_report:
        matrix.print_report(detailed=True)

    return matrix.generate_report()


def identify_collection_priorities(
    benchmarks: List[ExtendedBenchmark],
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """
    Identify top priorities for benchmark collection.

    Args:
        benchmarks: Current benchmark collection
        top_n: Number of priorities to return

    Returns:
        List of priority items
    """
    matrix = CoverageMatrix(benchmarks=benchmarks)
    return matrix.generate_priorities(top_n=top_n)
