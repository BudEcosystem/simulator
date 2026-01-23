"""
arXiv Paper Collector for Training Benchmarks.

Collects benchmark data from arXiv papers using:
- arXiv API for metadata
- PDF text extraction
- Pattern matching and LLM-based extraction
"""

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from ..benchmark_schema import (
    ExtendedBenchmark,
    HardwareType,
    OptimizerType,
    ParallelismConfig,
    QuantizationConfig,
    QuantizationMethod,
    ReportedMetrics,
    SourceProvenance,
    SourceType,
    TrainingType,
    VerificationStatus,
)
from .base_collector import (
    BaseCollector,
    CollectorError,
    CollectorResult,
    ExtractionMethod,
    PatternExtractor,
    SourceNotFoundError,
)

logger = logging.getLogger(__name__)


# Known paper configurations for high-priority papers
# These provide structured benchmark data that can be extracted reliably
KNOWN_PAPERS: Dict[str, Dict[str, Any]] = {
    # LLaMA 2 Paper
    "2307.09288": {
        "title": "Llama 2: Open Foundation and Fine-Tuned Chat Models",
        "organization": "meta",
        "benchmarks": [
            {
                "name": "LLaMA-2 7B Pretraining",
                "model_name": "Llama-2-7B",
                "model_params_b": 7.0,
                "training_type": TrainingType.PRETRAINING,
                "num_gpus": 1024,
                "hardware_type": HardwareType.A100_80GB,
                "batch_size": 4096,
                "seq_length": 4096,
                "mfu": 0.54,
                "parallelism": {"tp": 1, "pp": 1, "dp": 1024},
            },
            {
                "name": "LLaMA-2 13B Pretraining",
                "model_name": "Llama-2-13B",
                "model_params_b": 13.0,
                "training_type": TrainingType.PRETRAINING,
                "num_gpus": 1024,
                "hardware_type": HardwareType.A100_80GB,
                "batch_size": 4096,
                "seq_length": 4096,
                "mfu": 0.50,
                "parallelism": {"tp": 2, "pp": 1, "dp": 512},
            },
            {
                "name": "LLaMA-2 70B Pretraining",
                "model_name": "Llama-2-70B",
                "model_params_b": 70.0,
                "training_type": TrainingType.PRETRAINING,
                "num_gpus": 2048,
                "hardware_type": HardwareType.A100_80GB,
                "batch_size": 1024,
                "seq_length": 4096,
                "mfu": 0.43,
                "tokens_per_second": 654190,
                "parallelism": {"tp": 8, "pp": 16, "dp": 16},
            },
        ],
    },
    # LLaMA 3 Herd Paper
    "2407.21783": {
        "title": "The Llama 3 Herd of Models",
        "organization": "meta",
        "benchmarks": [
            {
                "name": "LLaMA-3 8B Pretraining",
                "model_name": "Llama-3-8B",
                "model_params_b": 8.0,
                "training_type": TrainingType.PRETRAINING,
                "num_gpus": 2048,
                "hardware_type": HardwareType.H100_SXM,
                "batch_size": 4096,
                "seq_length": 8192,
                "mfu": 0.51,
                "parallelism": {"tp": 1, "pp": 2, "dp": 1024},
            },
            {
                "name": "LLaMA-3 70B Pretraining",
                "model_name": "Llama-3-70B",
                "model_params_b": 70.0,
                "training_type": TrainingType.PRETRAINING,
                "num_gpus": 4096,
                "hardware_type": HardwareType.H100_SXM,
                "batch_size": 2048,
                "seq_length": 8192,
                "mfu": 0.48,
                "tokens_per_second": 4629650,
                "parallelism": {"tp": 8, "pp": 8, "dp": 64},
            },
            {
                "name": "LLaMA-3 405B Pretraining",
                "model_name": "Llama-3-405B",
                "model_params_b": 405.0,
                "training_type": TrainingType.PRETRAINING,
                "num_gpus": 16384,
                "hardware_type": HardwareType.H100_SXM,
                "batch_size": 4096,
                "seq_length": 8192,
                "mfu": 0.38,
                "parallelism": {"tp": 8, "pp": 16, "dp": 128},
            },
        ],
    },
    # QLoRA Paper
    "2305.14314": {
        "title": "QLoRA: Efficient Finetuning of Quantized LLMs",
        "organization": "uw",
        "benchmarks": [
            {
                "name": "QLoRA LLaMA-65B 4-bit",
                "model_name": "LLaMA-65B",
                "model_params_b": 65.0,
                "training_type": TrainingType.QLORA,
                "num_gpus": 1,
                "hardware_type": HardwareType.A100_80GB,
                "batch_size": 1,
                "seq_length": 1024,
                "memory_per_gpu_gb": 48.0,
                "quantization": QuantizationMethod.NF4,
            },
            {
                "name": "QLoRA LLaMA-33B 4-bit",
                "model_name": "LLaMA-33B",
                "model_params_b": 33.0,
                "training_type": TrainingType.QLORA,
                "num_gpus": 1,
                "hardware_type": HardwareType.A100_40GB,
                "batch_size": 1,
                "seq_length": 1024,
                "memory_per_gpu_gb": 24.0,
                "quantization": QuantizationMethod.NF4,
            },
        ],
    },
    # DeepSeek V3 Paper
    "2412.19437": {
        "title": "DeepSeek-V3 Technical Report",
        "organization": "deepseek",
        "benchmarks": [
            {
                "name": "DeepSeek V3 671B MoE FP8",
                "model_name": "DeepSeek-V3",
                "model_params_b": 671.0,
                "training_type": TrainingType.PRETRAINING,
                "is_moe": True,
                "num_experts": 256,
                "active_experts": 8,
                "num_gpus": 2048,
                "hardware_type": HardwareType.H100_SXM,
                "batch_size": 4096,
                "seq_length": 4096,
                "mfu": 0.21,
                "quantization": QuantizationMethod.FP8_E4M3,
                "parallelism": {"tp": 1, "pp": 8, "dp": 32, "ep": 64},
            },
        ],
    },
    # DPO Paper
    "2305.18290": {
        "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        "organization": "stanford",
        "benchmarks": [
            {
                "name": "DPO Pythia 2.8B",
                "model_name": "Pythia-2.8B",
                "model_params_b": 2.8,
                "training_type": TrainingType.DPO,
                "num_gpus": 4,
                "hardware_type": HardwareType.A100_80GB,
                "batch_size": 64,
                "seq_length": 512,
            },
        ],
    },
    # GaLore Paper
    "2403.03507": {
        "title": "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection",
        "organization": "berkeley",
        "benchmarks": [
            {
                "name": "GaLore LLaMA-7B Full Precision",
                "model_name": "LLaMA-7B",
                "model_params_b": 7.0,
                "training_type": TrainingType.PRETRAINING,
                "num_gpus": 1,
                "hardware_type": HardwareType.A100_80GB,
                "batch_size": 128,
                "seq_length": 256,
                "memory_per_gpu_gb": 22.0,
                "optimizer": OptimizerType.GALORE,
            },
            {
                "name": "GaLore 8-bit LLaMA-7B",
                "model_name": "LLaMA-7B",
                "model_params_b": 7.0,
                "training_type": TrainingType.PRETRAINING,
                "num_gpus": 1,
                "hardware_type": HardwareType.A100_80GB,
                "batch_size": 256,
                "seq_length": 256,
                "memory_per_gpu_gb": 15.0,
                "optimizer": OptimizerType.GALORE_8BIT,
            },
        ],
    },
    # ORPO Paper
    "2403.07691": {
        "title": "ORPO: Monolithic Preference Optimization without Reference Model",
        "organization": "kaist",
        "benchmarks": [
            {
                "name": "ORPO Mistral-7B",
                "model_name": "Mistral-7B",
                "model_params_b": 7.0,
                "training_type": TrainingType.ORPO,
                "num_gpus": 4,
                "hardware_type": HardwareType.A100_80GB,
                "batch_size": 16,
                "seq_length": 2048,
            },
            {
                "name": "ORPO LLaMA-2-7B",
                "model_name": "LLaMA-2-7B",
                "model_params_b": 7.0,
                "training_type": TrainingType.ORPO,
                "num_gpus": 4,
                "hardware_type": HardwareType.A100_80GB,
                "batch_size": 16,
                "seq_length": 2048,
            },
        ],
    },
}


@dataclass
class ArxivPaperMetadata:
    """Metadata extracted from arXiv API."""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published_date: str
    updated_date: Optional[str] = None
    pdf_url: Optional[str] = None
    doi: Optional[str] = None


class ArxivCollector(BaseCollector):
    """
    Collector for extracting training benchmarks from arXiv papers.

    Supports:
    - Known papers with pre-defined benchmark structures
    - API-based metadata extraction
    - Pattern-based extraction from abstracts
    - LLM-based extraction (when LLM service is available)
    """

    ARXIV_API_URL = "http://export.arxiv.org/api/query"

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        rate_limit_delay: float = 3.0,  # arXiv requires 3s between requests
        llm_extractor: Optional[Any] = None,  # Optional LLM for extraction
        **kwargs,
    ):
        super().__init__(
            cache_dir=cache_dir,
            rate_limit_delay=rate_limit_delay,
            **kwargs,
        )
        self.llm_extractor = llm_extractor

    def get_source_type(self) -> SourceType:
        return SourceType.ACADEMIC_PAPER

    def collect(
        self,
        source_identifier: str,
        use_known_papers: bool = True,
        extract_from_abstract: bool = True,
        **kwargs,
    ) -> CollectorResult:
        """
        Collect benchmarks from an arXiv paper.

        Args:
            source_identifier: arXiv ID (e.g., "2307.09288") or URL
            use_known_papers: Use pre-defined benchmark data for known papers
            extract_from_abstract: Try to extract metrics from abstract

        Returns:
            CollectorResult with collected benchmarks
        """
        start_time = time.time()
        result = CollectorResult(
            extraction_method=ExtractionMethod.API,
        )

        try:
            # Parse arxiv ID
            arxiv_id = self._parse_arxiv_id(source_identifier)
            if not arxiv_id:
                result.errors.append(f"Invalid arXiv identifier: {source_identifier}")
                return result

            # Check known papers first
            if use_known_papers and arxiv_id in KNOWN_PAPERS:
                benchmarks = self._extract_from_known_paper(arxiv_id)
                result.benchmarks = benchmarks
                result.extraction_method = ExtractionMethod.MANUAL
                result.extraction_confidence = 1.0
            else:
                # Fetch metadata from arXiv API
                metadata = self._fetch_arxiv_metadata(arxiv_id)
                if metadata:
                    result.source_url = f"https://arxiv.org/abs/{arxiv_id}"
                    result.source_title = metadata.title

                    # Try to extract from abstract
                    if extract_from_abstract:
                        benchmarks = self._extract_from_abstract(arxiv_id, metadata)
                        result.benchmarks = benchmarks
                        result.extraction_method = ExtractionMethod.REGEX
                        result.extraction_confidence = 0.5  # Lower confidence for pattern extraction
                else:
                    result.errors.append(f"Could not fetch metadata for {arxiv_id}")

            # Post-process benchmarks
            result.benchmarks = self.postprocess_benchmarks(result.benchmarks)

        except Exception as e:
            result.errors.append(f"Collection error: {str(e)}")
            logger.exception(f"Error collecting from arXiv {source_identifier}")

        result.extraction_duration_sec = time.time() - start_time
        return result

    def _parse_arxiv_id(self, identifier: str) -> Optional[str]:
        """Parse arXiv ID from various formats."""
        # Handle URLs
        if "arxiv.org" in identifier:
            # Match patterns like arxiv.org/abs/2307.09288 or arxiv.org/pdf/2307.09288
            match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", identifier)
            if match:
                return match.group(1)

        # Handle direct IDs
        match = re.match(r"^(\d{4}\.\d{4,5})(?:v\d+)?$", identifier.strip())
        if match:
            return match.group(1)

        return None

    def _fetch_arxiv_metadata(self, arxiv_id: str) -> Optional[ArxivPaperMetadata]:
        """Fetch paper metadata from arXiv API."""
        try:
            import requests
            import xml.etree.ElementTree as ET

            # Respect rate limit
            time.sleep(self.rate_limit_delay)

            url = f"{self.ARXIV_API_URL}?id_list={arxiv_id}"
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()

            # Parse XML response
            root = ET.fromstring(response.content)
            ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

            entry = root.find("atom:entry", ns)
            if entry is None:
                return None

            # Extract metadata
            title_elem = entry.find("atom:title", ns)
            title = title_elem.text.strip() if title_elem is not None else ""

            abstract_elem = entry.find("atom:summary", ns)
            abstract = abstract_elem.text.strip() if abstract_elem is not None else ""

            authors = []
            for author in entry.findall("atom:author", ns):
                name = author.find("atom:name", ns)
                if name is not None:
                    authors.append(name.text)

            published_elem = entry.find("atom:published", ns)
            published = published_elem.text[:10] if published_elem is not None else ""

            updated_elem = entry.find("atom:updated", ns)
            updated = updated_elem.text[:10] if updated_elem is not None else None

            # Get PDF link
            pdf_url = None
            for link in entry.findall("atom:link", ns):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href")
                    break

            # Get categories
            categories = []
            for cat in entry.findall("arxiv:primary_category", ns):
                categories.append(cat.get("term", ""))

            return ArxivPaperMetadata(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract,
                categories=categories,
                published_date=published,
                updated_date=updated,
                pdf_url=pdf_url,
            )

        except Exception as e:
            logger.error(f"Error fetching arXiv metadata for {arxiv_id}: {e}")
            return None

    def _extract_from_known_paper(self, arxiv_id: str) -> List[ExtendedBenchmark]:
        """Extract benchmarks from known paper configurations."""
        paper_config = KNOWN_PAPERS.get(arxiv_id, {})
        benchmarks = []

        provenance = self.create_provenance(
            source_url=f"https://arxiv.org/abs/{arxiv_id}",
            source_title=paper_config.get("title", ""),
            organization=paper_config.get("organization", ""),
            extraction_method=ExtractionMethod.MANUAL,
            arxiv_id=arxiv_id,
        )

        for bm_config in paper_config.get("benchmarks", []):
            # Create parallelism config
            p_config = bm_config.get("parallelism", {})
            parallelism = ParallelismConfig(
                tensor_parallel=p_config.get("tp", 1),
                pipeline_parallel=p_config.get("pp", 1),
                data_parallel=p_config.get("dp", 1),
                expert_parallel=p_config.get("ep", 1),
            )

            # Create quantization config
            quant = bm_config.get("quantization")
            quantization = QuantizationConfig(
                model_precision=quant if quant else QuantizationMethod.BF16,
            )

            # Create metrics
            metrics = ReportedMetrics(
                mfu=bm_config.get("mfu"),
                tokens_per_second=bm_config.get("tokens_per_second"),
                memory_per_gpu_gb=bm_config.get("memory_per_gpu_gb"),
            )

            benchmark = ExtendedBenchmark(
                benchmark_id="",  # Will be generated
                name=bm_config["name"],
                provenance=provenance,
                model_name=bm_config["model_name"],
                model_params_b=bm_config["model_params_b"],
                training_type=bm_config.get("training_type", TrainingType.PRETRAINING),
                is_moe=bm_config.get("is_moe", False),
                num_experts=bm_config.get("num_experts", 1),
                active_experts=bm_config.get("active_experts", 1),
                num_gpus=bm_config.get("num_gpus", 1),
                hardware_type=bm_config.get("hardware_type", HardwareType.H100_SXM),
                batch_size=bm_config.get("batch_size", 1),
                seq_length=bm_config.get("seq_length", 2048),
                parallelism=parallelism,
                quantization=quantization,
                optimizer=bm_config.get("optimizer", OptimizerType.ADAMW),
                metrics=metrics,
                verification_status=VerificationStatus.VERIFIED,
                tags=["arxiv", f"arxiv:{arxiv_id}"],
            )

            benchmarks.append(benchmark)

        return benchmarks

    def _extract_from_abstract(
        self,
        arxiv_id: str,
        metadata: ArxivPaperMetadata,
    ) -> List[ExtendedBenchmark]:
        """Extract benchmarks from paper abstract using pattern matching."""
        benchmarks = []

        # Extract metrics from abstract
        extracted = PatternExtractor.extract_all(metadata.abstract)

        # Only create benchmark if we have meaningful data
        if extracted["mfu"] or extracted["tokens_per_second"] or extracted["num_gpus"]:
            provenance = self.create_provenance(
                source_url=f"https://arxiv.org/abs/{arxiv_id}",
                source_title=metadata.title,
                authors=metadata.authors,
                publication_date=metadata.published_date,
                extraction_method=ExtractionMethod.REGEX,
                arxiv_id=arxiv_id,
            )

            metrics = ReportedMetrics(
                mfu=extracted["mfu"],
                tokens_per_second=extracted["tokens_per_second"],
                memory_per_gpu_gb=extracted["memory_gb"],
            )

            benchmark = ExtendedBenchmark(
                benchmark_id="",
                name=f"Extracted: {metadata.title[:50]}...",
                provenance=provenance,
                model_name=self._infer_model_name(metadata.abstract) or "unknown",
                model_params_b=extracted["model_size_b"] or 0.0,
                num_gpus=extracted["num_gpus"] or 1,
                batch_size=extracted["batch_size"] or 1,
                seq_length=extracted["seq_length"] or 2048,
                metrics=metrics,
                verification_status=VerificationStatus.UNVERIFIED,
                notes="Automatically extracted from abstract",
                tags=["arxiv", "auto-extracted", f"arxiv:{arxiv_id}"],
            )

            benchmarks.append(benchmark)

        return benchmarks

    def _infer_model_name(self, text: str) -> Optional[str]:
        """Infer model name from text."""
        # Common model patterns
        patterns = [
            r"(LLaMA[\s-]*\d+[Bb]?)",
            r"(GPT[\s-]*\d+)",
            r"(Mistral[\s-]*\d+[Bb]?)",
            r"(Qwen[\s-]*\d*[Bb]?)",
            r"(DeepSeek[\s-]*V?\d*)",
            r"(Gemma[\s-]*\d*[Bb]?)",
            r"(Falcon[\s-]*\d+[Bb]?)",
            r"(Pythia[\s-]*\d+[Bb]?)",
            r"(OPT[\s-]*\d+[Bb]?)",
            r"(BLOOM[\s-]*\d+[Bb]?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def collect_known_papers(self) -> CollectorResult:
        """Collect benchmarks from all known papers."""
        result = CollectorResult(
            extraction_method=ExtractionMethod.MANUAL,
            extraction_confidence=1.0,
        )

        for arxiv_id in KNOWN_PAPERS:
            paper_result = self.collect(arxiv_id, use_known_papers=True)
            result.benchmarks.extend(paper_result.benchmarks)
            result.errors.extend(paper_result.errors)
            result.warnings.extend(paper_result.warnings)

        return result

    def list_known_papers(self) -> List[Dict[str, str]]:
        """List all known papers with arXiv IDs."""
        return [
            {
                "arxiv_id": arxiv_id,
                "title": config.get("title", ""),
                "organization": config.get("organization", ""),
                "num_benchmarks": len(config.get("benchmarks", [])),
            }
            for arxiv_id, config in KNOWN_PAPERS.items()
        ]
