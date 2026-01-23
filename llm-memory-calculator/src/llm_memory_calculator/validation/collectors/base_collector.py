"""
Base Collector Interface for Benchmark Data Collection.

Provides abstract base class and common utilities for all benchmark collectors.
"""

import abc
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Tuple

from ..benchmark_schema import (
    ExtendedBenchmark,
    SourceProvenance,
    SourceType,
    VerificationStatus,
)

logger = logging.getLogger(__name__)


class ExtractionMethod(Enum):
    """Methods used to extract benchmark data."""
    MANUAL = "manual"
    API = "api"
    REGEX = "regex"
    LLM_EXTRACTION = "llm_extraction"
    TABLE_PARSING = "table_parsing"
    HTML_SCRAPING = "html_scraping"
    JSON_PARSING = "json_parsing"


class CollectorError(Exception):
    """Base exception for collector errors."""
    pass


class RateLimitError(CollectorError):
    """Rate limit exceeded."""
    pass


class AuthenticationError(CollectorError):
    """Authentication failed."""
    pass


class SourceNotFoundError(CollectorError):
    """Source document/resource not found."""
    pass


class ExtractionError(CollectorError):
    """Failed to extract benchmark data."""
    pass


@dataclass
class CollectorResult:
    """Result from a collector operation."""

    # Collected benchmarks
    benchmarks: List[ExtendedBenchmark] = field(default_factory=list)

    # Source information
    source_url: Optional[str] = None
    source_title: Optional[str] = None

    # Collection metadata
    collected_at: str = field(default_factory=lambda: datetime.now().isoformat())
    extraction_method: ExtractionMethod = ExtractionMethod.MANUAL
    extraction_duration_sec: float = 0.0

    # Quality indicators
    raw_data: Optional[Dict[str, Any]] = None
    extraction_confidence: float = 1.0  # 0-1 confidence in extraction quality

    # Errors/warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if collection was successful."""
        return len(self.benchmarks) > 0 and len(self.errors) == 0

    @property
    def num_benchmarks(self) -> int:
        """Number of benchmarks collected."""
        return len(self.benchmarks)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "benchmarks": [b.to_dict() for b in self.benchmarks],
            "source_url": self.source_url,
            "source_title": self.source_title,
            "collected_at": self.collected_at,
            "extraction_method": self.extraction_method.value,
            "extraction_duration_sec": self.extraction_duration_sec,
            "extraction_confidence": self.extraction_confidence,
            "num_benchmarks": self.num_benchmarks,
            "errors": self.errors,
            "warnings": self.warnings,
            "success": self.success,
        }


class BaseCollector(abc.ABC):
    """
    Abstract base class for benchmark data collectors.

    All collectors must implement:
    - collect(): Main collection method
    - get_source_type(): Return the SourceType for this collector

    Optional methods to override:
    - validate_source(): Check if source is valid
    - preprocess_data(): Clean/normalize raw data
    - postprocess_benchmarks(): Validate/enrich collected benchmarks
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        rate_limit_delay: float = 1.0,
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        """
        Initialize collector.

        Args:
            cache_dir: Directory for caching collected data
            rate_limit_delay: Delay between requests (seconds)
            max_retries: Maximum retry attempts on failure
            timeout: Request timeout (seconds)
        """
        self.cache_dir = cache_dir
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.timeout = timeout
        self.logger = logging.getLogger(self.__class__.__name__)

    @abc.abstractmethod
    def get_source_type(self) -> SourceType:
        """Return the source type for this collector."""
        pass

    @abc.abstractmethod
    def collect(
        self,
        source_identifier: str,
        **kwargs,
    ) -> CollectorResult:
        """
        Collect benchmarks from a source.

        Args:
            source_identifier: Identifier for the source (URL, paper ID, etc.)
            **kwargs: Additional collector-specific arguments

        Returns:
            CollectorResult with collected benchmarks
        """
        pass

    def validate_source(self, source_identifier: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that a source identifier is valid.

        Args:
            source_identifier: The source to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not source_identifier:
            return False, "Empty source identifier"
        return True, None

    def preprocess_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess raw data before extraction.

        Override this to clean or normalize raw data.

        Args:
            raw_data: Raw data from source

        Returns:
            Preprocessed data
        """
        return raw_data

    def postprocess_benchmarks(
        self,
        benchmarks: List[ExtendedBenchmark],
    ) -> List[ExtendedBenchmark]:
        """
        Postprocess collected benchmarks.

        Override this to add validation, enrichment, or deduplication.

        Args:
            benchmarks: Collected benchmarks

        Returns:
            Processed benchmarks
        """
        # Default: validate each benchmark
        valid_benchmarks = []
        for benchmark in benchmarks:
            is_valid, issues = benchmark.validate()
            if is_valid:
                valid_benchmarks.append(benchmark)
            else:
                self.logger.warning(
                    f"Benchmark {benchmark.name} has validation issues: {issues}"
                )
                # Still include but mark as unverified
                benchmark.verification_status = VerificationStatus.UNVERIFIED
                benchmark.verification_notes = f"Validation issues: {', '.join(issues)}"
                valid_benchmarks.append(benchmark)
        return valid_benchmarks

    def create_provenance(
        self,
        source_url: Optional[str] = None,
        source_title: Optional[str] = None,
        authors: Optional[List[str]] = None,
        organization: Optional[str] = None,
        publication_date: Optional[str] = None,
        extraction_method: ExtractionMethod = ExtractionMethod.MANUAL,
        **kwargs,
    ) -> SourceProvenance:
        """
        Create a SourceProvenance object with common fields populated.

        Args:
            source_url: URL of the source
            source_title: Title of the source document
            authors: List of author names
            organization: Organization name
            publication_date: Publication date (YYYY-MM-DD)
            extraction_method: How data was extracted
            **kwargs: Additional fields for SourceProvenance

        Returns:
            SourceProvenance object
        """
        return SourceProvenance(
            source_type=self.get_source_type(),
            source_url=source_url,
            source_title=source_title,
            authors=authors or [],
            organization=organization,
            publication_date=publication_date,
            extraction_method=extraction_method.value,
            extraction_date=datetime.now().strftime("%Y-%m-%d"),
            **kwargs,
        )


class PatternExtractor:
    """
    Utility class for extracting benchmark data using regex patterns.

    Common patterns for:
    - MFU values
    - Throughput (tokens/sec, samples/sec)
    - Memory usage
    - Model configurations
    - GPU counts and types
    """

    # MFU patterns
    MFU_PATTERNS = [
        r"(?:MFU|model\s*flops?\s*utilization)\s*[=:]\s*(\d+\.?\d*)%?",
        r"(\d+\.?\d*)%?\s*(?:MFU|model\s*flops?\s*utilization)",
        r"(?:achieves?|achieved|reaching)\s*(\d+\.?\d*)%?\s*MFU",
    ]

    # Throughput patterns
    THROUGHPUT_PATTERNS = [
        r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:tokens?/s|tok/s|TPS)",
        r"(?:throughput|tokens\s*per\s*second)\s*[=:]\s*(\d+(?:,\d{3})*(?:\.\d+)?)",
        r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*samples?/s(?:ec)?",
    ]

    # Memory patterns
    MEMORY_PATTERNS = [
        r"(\d+(?:\.\d+)?)\s*GB\s*(?:per\s*GPU|/GPU|memory)",
        r"(?:memory|GPU\s*memory)\s*[=:]\s*(\d+(?:\.\d+)?)\s*GB",
        r"(?:peak|max)\s*memory\s*[=:]\s*(\d+(?:\.\d+)?)\s*GB",
    ]

    # GPU count patterns
    GPU_COUNT_PATTERNS = [
        r"(\d+(?:,\d{3})*)\s*(?:GPUs?|accelerators?|devices?)",
        r"(?:on|using|with)\s*(\d+(?:,\d{3})*)\s*(?:GPUs?|A100|H100|V100)",
        r"(\d+)x\s*(?:A100|H100|V100|GPU)",
    ]

    # Batch size patterns
    BATCH_SIZE_PATTERNS = [
        r"(?:global\s*)?batch\s*size\s*[=:]\s*(\d+(?:,\d{3})*)",
        r"(\d+(?:,\d{3})*)\s*(?:global\s*)?batch\s*size",
    ]

    # Sequence length patterns
    SEQ_LENGTH_PATTERNS = [
        r"(?:sequence|seq|context)\s*length\s*[=:]\s*(\d+(?:,\d{3})*)",
        r"(\d+(?:,\d{3})*)\s*(?:tokens?|sequence|context)\s*length",
    ]

    # Model size patterns
    MODEL_SIZE_PATTERNS = [
        r"(\d+(?:\.\d+)?)\s*[Bb]illion\s*param",
        r"(\d+(?:\.\d+)?)[Bb]\s*(?:param|model)",
        r"(\d+(?:\.\d+)?)\s*[Bb]\s*LLM",
    ]

    @classmethod
    def extract_mfu(cls, text: str) -> Optional[float]:
        """Extract MFU value from text."""
        for pattern in cls.MFU_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1).replace(",", ""))
                # Normalize to 0-1 if given as percentage
                if value > 1:
                    value = value / 100
                return value
        return None

    @classmethod
    def extract_throughput(cls, text: str) -> Optional[float]:
        """Extract throughput (tokens/sec) from text."""
        for pattern in cls.THROUGHPUT_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1).replace(",", ""))
        return None

    @classmethod
    def extract_memory(cls, text: str) -> Optional[float]:
        """Extract memory usage (GB) from text."""
        for pattern in cls.MEMORY_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1).replace(",", ""))
        return None

    @classmethod
    def extract_gpu_count(cls, text: str) -> Optional[int]:
        """Extract GPU count from text."""
        for pattern in cls.GPU_COUNT_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1).replace(",", ""))
        return None

    @classmethod
    def extract_batch_size(cls, text: str) -> Optional[int]:
        """Extract batch size from text."""
        for pattern in cls.BATCH_SIZE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1).replace(",", ""))
        return None

    @classmethod
    def extract_seq_length(cls, text: str) -> Optional[int]:
        """Extract sequence length from text."""
        for pattern in cls.SEQ_LENGTH_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1).replace(",", ""))
        return None

    @classmethod
    def extract_model_size(cls, text: str) -> Optional[float]:
        """Extract model size (billions) from text."""
        for pattern in cls.MODEL_SIZE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1).replace(",", ""))
        return None

    @classmethod
    def extract_all(cls, text: str) -> Dict[str, Any]:
        """
        Extract all metrics from text.

        Returns:
            Dictionary with extracted values (keys are metric names)
        """
        return {
            "mfu": cls.extract_mfu(text),
            "tokens_per_second": cls.extract_throughput(text),
            "memory_gb": cls.extract_memory(text),
            "num_gpus": cls.extract_gpu_count(text),
            "batch_size": cls.extract_batch_size(text),
            "seq_length": cls.extract_seq_length(text),
            "model_size_b": cls.extract_model_size(text),
        }


class TableParser:
    """
    Utility class for parsing benchmark tables from various formats.
    """

    @staticmethod
    def parse_markdown_table(text: str) -> List[Dict[str, str]]:
        """
        Parse a markdown-formatted table.

        Args:
            text: Text containing markdown table

        Returns:
            List of dictionaries (one per row)
        """
        lines = text.strip().split("\n")
        rows = []

        # Find header row
        header_line = None
        data_start = 0
        for i, line in enumerate(lines):
            if "|" in line and not line.strip().startswith("|--") and not line.strip().startswith("|-"):
                if header_line is None:
                    header_line = line
                    # Skip separator line
                    data_start = i + 2
                    break

        if header_line is None:
            return rows

        # Parse header
        headers = [h.strip() for h in header_line.split("|") if h.strip()]

        # Parse data rows
        for line in lines[data_start:]:
            if "|" in line and not line.strip().startswith("|--"):
                cells = [c.strip() for c in line.split("|") if c.strip()]
                if len(cells) >= len(headers):
                    row = {headers[i]: cells[i] for i in range(len(headers))}
                    rows.append(row)

        return rows

    @staticmethod
    def parse_csv_text(text: str, delimiter: str = ",") -> List[Dict[str, str]]:
        """
        Parse CSV-formatted text.

        Args:
            text: CSV text
            delimiter: Field delimiter

        Returns:
            List of dictionaries (one per row)
        """
        import csv
        import io

        rows = []
        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
        for row in reader:
            rows.append(dict(row))
        return rows


def deduplicate_benchmarks(
    benchmarks: List[ExtendedBenchmark],
    key_fields: Optional[List[str]] = None,
) -> List[ExtendedBenchmark]:
    """
    Remove duplicate benchmarks based on key fields.

    Args:
        benchmarks: List of benchmarks
        key_fields: Fields to use for deduplication (default: model+gpus+training_type)

    Returns:
        Deduplicated list
    """
    if key_fields is None:
        key_fields = ["model_name", "model_params_b", "num_gpus", "training_type"]

    seen = set()
    unique = []

    for benchmark in benchmarks:
        key = tuple(getattr(benchmark, f, None) for f in key_fields)
        if key not in seen:
            seen.add(key)
            unique.append(benchmark)

    return unique


def merge_benchmarks(
    existing: ExtendedBenchmark,
    new: ExtendedBenchmark,
) -> ExtendedBenchmark:
    """
    Merge two benchmarks, preferring higher confidence values.

    Args:
        existing: Existing benchmark
        new: New benchmark to merge

    Returns:
        Merged benchmark
    """
    # Keep the one with higher confidence
    if new.confidence > existing.confidence:
        base = new
        other = existing
    else:
        base = existing
        other = new

    # Merge missing fields from other
    if base.metrics.mfu is None and other.metrics.mfu is not None:
        base.metrics.mfu = other.metrics.mfu

    if base.metrics.tokens_per_second is None and other.metrics.tokens_per_second is not None:
        base.metrics.tokens_per_second = other.metrics.tokens_per_second

    if base.metrics.memory_per_gpu_gb is None and other.metrics.memory_per_gpu_gb is not None:
        base.metrics.memory_per_gpu_gb = other.metrics.memory_per_gpu_gb

    # Add verification source
    if other.provenance.source_url and other.provenance.source_url not in base.verification_sources:
        base.verification_sources.append(other.provenance.source_url)

    # Update verification status if now cross-checked
    if len(base.verification_sources) > 0:
        base.verification_status = VerificationStatus.CROSS_CHECKED

    base.updated_at = datetime.now().isoformat()

    return base
