"""
Benchmark Data Collectors.

This module provides collectors for gathering training benchmarks from various sources:
- arXiv papers
- MLPerf submissions
- HuggingFace model cards
- GitHub repositories
- Engineering blogs

Each collector implements the BaseCollector interface and produces ExtendedBenchmark objects.
"""

from .base_collector import (
    BaseCollector,
    CollectorResult,
    CollectorError,
    ExtractionMethod,
)

from .arxiv_collector import ArxivCollector
from .mlperf_collector import MLPerfCollector
from .huggingface_collector import HuggingFaceCollector
from .github_collector import GitHubCollector

__all__ = [
    # Base
    "BaseCollector",
    "CollectorResult",
    "CollectorError",
    "ExtractionMethod",
    # Collectors
    "ArxivCollector",
    "MLPerfCollector",
    "HuggingFaceCollector",
    "GitHubCollector",
]
