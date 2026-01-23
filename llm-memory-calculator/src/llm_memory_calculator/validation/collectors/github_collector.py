"""
GitHub Repository Collector for Training Benchmarks.

Collects training benchmark data from GitHub repositories, including:
- Training scripts and configurations
- Benchmark results in READMEs
- Training logs and reports
- Optimizer and library repositories (GaLore, APOLLO, OpenRLHF, TRL)
"""

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..benchmark_schema import (
    ExtendedBenchmark,
    HardwareType,
    MultiModelConfig,
    OptimizerType,
    ParallelismConfig,
    PEFTConfig,
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
    CollectorResult,
    ExtractionMethod,
    PatternExtractor,
)

logger = logging.getLogger(__name__)


# Known repositories with training benchmarks
KNOWN_REPOSITORIES: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # Optimizer Libraries
    # =========================================================================
    "jiaweizzhao/GaLore": {
        "name": "GaLore",
        "organization": "berkeley",
        "description": "Gradient Low-Rank Projection for memory-efficient LLM training",
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
                "notes": "GaLore rank 128, 4x memory reduction vs AdamW",
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
                "notes": "GaLore 8-bit with bitsandbytes",
            },
            {
                "name": "GaLore LLaMA-1B Single GPU",
                "model_name": "LLaMA-1B",
                "model_params_b": 1.0,
                "training_type": TrainingType.PRETRAINING,
                "num_gpus": 1,
                "hardware_type": HardwareType.A100_40GB,
                "batch_size": 256,
                "seq_length": 256,
                "memory_per_gpu_gb": 6.8,
                "optimizer": OptimizerType.GALORE,
            },
        ],
    },
    "zhuzilin/APOLLO": {
        "name": "APOLLO",
        "organization": "community",
        "description": "An Optimized LLM Training Framework",
        "benchmarks": [
            {
                "name": "APOLLO LLaMA-7B",
                "model_name": "LLaMA-7B",
                "model_params_b": 7.0,
                "training_type": TrainingType.PRETRAINING,
                "num_gpus": 1,
                "hardware_type": HardwareType.A100_80GB,
                "batch_size": 128,
                "seq_length": 256,
                "memory_per_gpu_gb": 18.0,
                "optimizer": OptimizerType.APOLLO,
                "notes": "APOLLO with rank-1 update",
            },
        ],
    },
    "huggingface/bitsandbytes": {
        "name": "bitsandbytes",
        "organization": "huggingface",
        "description": "8-bit CUDA functions for PyTorch",
        "benchmarks": [
            {
                "name": "8-bit AdamW LLaMA-7B",
                "model_name": "LLaMA-7B",
                "model_params_b": 7.0,
                "training_type": TrainingType.SFT,
                "num_gpus": 1,
                "hardware_type": HardwareType.A100_80GB,
                "batch_size": 8,
                "seq_length": 2048,
                "memory_per_gpu_gb": 28.0,
                "optimizer": OptimizerType.ADAMW_8BIT,
                "notes": "75% optimizer memory reduction",
            },
            {
                "name": "8-bit AdamW LLaMA-13B",
                "model_name": "LLaMA-13B",
                "model_params_b": 13.0,
                "training_type": TrainingType.SFT,
                "num_gpus": 1,
                "hardware_type": HardwareType.A100_80GB,
                "batch_size": 4,
                "seq_length": 2048,
                "memory_per_gpu_gb": 48.0,
                "optimizer": OptimizerType.ADAMW_8BIT,
            },
        ],
    },
    # =========================================================================
    # RLHF/Post-Training Libraries
    # =========================================================================
    "OpenRLHF/OpenRLHF": {
        "name": "OpenRLHF",
        "organization": "community",
        "description": "High-performance RLHF training framework",
        "benchmarks": [
            {
                "name": "OpenRLHF PPO LLaMA-7B",
                "model_name": "LLaMA-7B",
                "model_params_b": 7.0,
                "training_type": TrainingType.PPO,
                "num_gpus": 8,
                "hardware_type": HardwareType.A100_80GB,
                "batch_size": 128,
                "seq_length": 2048,
                "memory_per_gpu_gb": 65.0,
                "multi_model": {
                    "num_models": 4,
                    "has_policy_model": True,
                    "has_reference_model": True,
                    "has_reward_model": True,
                    "has_value_model": True,
                },
                "notes": "Full PPO with 4 models",
            },
            {
                "name": "OpenRLHF PPO LLaMA-70B",
                "model_name": "LLaMA-70B",
                "model_params_b": 70.0,
                "training_type": TrainingType.PPO,
                "num_gpus": 64,
                "hardware_type": HardwareType.H100_SXM,
                "batch_size": 512,
                "seq_length": 2048,
                "memory_per_gpu_gb": 75.0,
                "parallelism": {"tp": 4, "pp": 1, "dp": 16},
                "multi_model": {
                    "num_models": 4,
                    "has_policy_model": True,
                    "has_reference_model": True,
                    "has_reward_model": True,
                    "has_value_model": True,
                    "reference_model_offloaded": False,
                },
                "notes": "Large-scale PPO training",
            },
            {
                "name": "OpenRLHF DPO LLaMA-7B",
                "model_name": "LLaMA-7B",
                "model_params_b": 7.0,
                "training_type": TrainingType.DPO,
                "num_gpus": 4,
                "hardware_type": HardwareType.A100_80GB,
                "batch_size": 64,
                "seq_length": 2048,
                "memory_per_gpu_gb": 45.0,
                "multi_model": {
                    "num_models": 2,
                    "has_policy_model": True,
                    "has_reference_model": True,
                    "reference_model_frozen": True,
                },
            },
        ],
    },
    "huggingface/trl": {
        "name": "TRL",
        "organization": "huggingface",
        "description": "Transformer Reinforcement Learning",
        "benchmarks": [
            {
                "name": "TRL SFT LLaMA-7B",
                "model_name": "LLaMA-7B",
                "model_params_b": 7.0,
                "training_type": TrainingType.SFT,
                "num_gpus": 4,
                "hardware_type": HardwareType.A100_80GB,
                "batch_size": 32,
                "seq_length": 2048,
                "memory_per_gpu_gb": 35.0,
            },
            {
                "name": "TRL DPO Mistral-7B",
                "model_name": "Mistral-7B",
                "model_params_b": 7.0,
                "training_type": TrainingType.DPO,
                "num_gpus": 4,
                "hardware_type": HardwareType.A100_80GB,
                "batch_size": 16,
                "seq_length": 2048,
                "memory_per_gpu_gb": 50.0,
                "multi_model": {
                    "num_models": 2,
                    "has_policy_model": True,
                    "has_reference_model": True,
                },
            },
            {
                "name": "TRL PPO GPT-2",
                "model_name": "GPT-2",
                "model_params_b": 0.12,
                "training_type": TrainingType.PPO,
                "num_gpus": 1,
                "hardware_type": HardwareType.A100_40GB,
                "batch_size": 256,
                "seq_length": 512,
                "memory_per_gpu_gb": 12.0,
                "multi_model": {
                    "num_models": 4,
                    "has_policy_model": True,
                    "has_reference_model": True,
                    "has_reward_model": True,
                    "has_value_model": True,
                },
            },
            {
                "name": "TRL ORPO Mistral-7B",
                "model_name": "Mistral-7B",
                "model_params_b": 7.0,
                "training_type": TrainingType.ORPO,
                "num_gpus": 4,
                "hardware_type": HardwareType.A100_80GB,
                "batch_size": 16,
                "seq_length": 2048,
                "memory_per_gpu_gb": 32.0,
                "notes": "Reference-free preference optimization",
            },
        ],
    },
    # =========================================================================
    # Training Frameworks
    # =========================================================================
    "hiyouga/LLaMA-Factory": {
        "name": "LLaMA-Factory",
        "organization": "community",
        "description": "Unified efficient fine-tuning framework",
        "benchmarks": [
            {
                "name": "LLaMA-Factory SFT LLaMA-7B",
                "model_name": "LLaMA-7B",
                "model_params_b": 7.0,
                "training_type": TrainingType.SFT,
                "num_gpus": 1,
                "hardware_type": HardwareType.A100_80GB,
                "batch_size": 4,
                "seq_length": 4096,
                "memory_per_gpu_gb": 28.0,
            },
            {
                "name": "LLaMA-Factory LoRA LLaMA-7B",
                "model_name": "LLaMA-7B",
                "model_params_b": 7.0,
                "training_type": TrainingType.LORA,
                "num_gpus": 1,
                "hardware_type": HardwareType.A100_40GB,
                "batch_size": 8,
                "seq_length": 2048,
                "memory_per_gpu_gb": 18.0,
                "peft": {
                    "method": "lora",
                    "lora_rank": 64,
                    "lora_alpha": 128,
                },
            },
            {
                "name": "LLaMA-Factory QLoRA LLaMA-70B",
                "model_name": "LLaMA-70B",
                "model_params_b": 70.0,
                "training_type": TrainingType.QLORA,
                "num_gpus": 1,
                "hardware_type": HardwareType.A100_80GB,
                "batch_size": 1,
                "seq_length": 2048,
                "memory_per_gpu_gb": 48.0,
                "quantization": QuantizationMethod.NF4,
                "peft": {
                    "method": "qlora",
                    "lora_rank": 64,
                    "qlora_bits": 4,
                },
            },
            {
                "name": "LLaMA-Factory DPO LLaMA-7B",
                "model_name": "LLaMA-7B",
                "model_params_b": 7.0,
                "training_type": TrainingType.DPO,
                "num_gpus": 2,
                "hardware_type": HardwareType.A100_80GB,
                "batch_size": 4,
                "seq_length": 2048,
                "memory_per_gpu_gb": 55.0,
                "multi_model": {
                    "num_models": 2,
                    "has_policy_model": True,
                    "has_reference_model": True,
                },
            },
        ],
    },
    "microsoft/DeepSpeed": {
        "name": "DeepSpeed",
        "organization": "microsoft",
        "description": "Deep learning optimization library",
        "benchmarks": [
            {
                "name": "DeepSpeed ZeRO-3 LLaMA-7B",
                "model_name": "LLaMA-7B",
                "model_params_b": 7.0,
                "training_type": TrainingType.SFT,
                "num_gpus": 8,
                "hardware_type": HardwareType.A100_40GB,
                "batch_size": 64,
                "seq_length": 2048,
                "memory_per_gpu_gb": 28.0,
                "parallelism": {"zero_stage": 3},
                "notes": "ZeRO-3 with optimizer offload",
            },
            {
                "name": "DeepSpeed ZeRO-3 LLaMA-70B",
                "model_name": "LLaMA-70B",
                "model_params_b": 70.0,
                "training_type": TrainingType.SFT,
                "num_gpus": 16,
                "hardware_type": HardwareType.A100_80GB,
                "batch_size": 32,
                "seq_length": 2048,
                "memory_per_gpu_gb": 70.0,
                "parallelism": {"zero_stage": 3},
            },
            {
                "name": "DeepSpeed ZeRO++ LLaMA-13B",
                "model_name": "LLaMA-13B",
                "model_params_b": 13.0,
                "training_type": TrainingType.PRETRAINING,
                "num_gpus": 8,
                "hardware_type": HardwareType.H100_SXM,
                "batch_size": 128,
                "seq_length": 4096,
                "memory_per_gpu_gb": 65.0,
                "mfu": 0.52,
                "parallelism": {"zero_stage": 3},
                "notes": "ZeRO++ with hierarchical partitioning",
            },
        ],
    },
    "facebookresearch/llama-recipes": {
        "name": "llama-recipes",
        "organization": "meta",
        "description": "Meta's official LLaMA fine-tuning examples",
        "benchmarks": [
            {
                "name": "LLaMA-recipes SFT LLaMA-2-7B",
                "model_name": "LLaMA-2-7B",
                "model_params_b": 7.0,
                "training_type": TrainingType.SFT,
                "num_gpus": 8,
                "hardware_type": HardwareType.A100_80GB,
                "batch_size": 32,
                "seq_length": 4096,
                "memory_per_gpu_gb": 42.0,
            },
            {
                "name": "LLaMA-recipes FSDP LLaMA-2-70B",
                "model_name": "LLaMA-2-70B",
                "model_params_b": 70.0,
                "training_type": TrainingType.SFT,
                "num_gpus": 8,
                "hardware_type": HardwareType.A100_80GB,
                "batch_size": 8,
                "seq_length": 4096,
                "memory_per_gpu_gb": 75.0,
                "parallelism": {"fsdp": True},
            },
        ],
    },
}


class GitHubCollector(BaseCollector):
    """
    Collector for GitHub repository training benchmarks.

    Extracts training data from:
    - Repository READMEs
    - Training configurations
    - Benchmark results
    - Known repository benchmark databases
    """

    GITHUB_API_URL = "https://api.github.com"

    def __init__(
        self,
        github_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(cache_dir=cache_dir, **kwargs)
        self.github_token = github_token

    def get_source_type(self) -> SourceType:
        return SourceType.GITHUB_REPO

    def collect(
        self,
        source_identifier: str,
        use_known_repos: bool = True,
        **kwargs,
    ) -> CollectorResult:
        """
        Collect benchmarks from a GitHub repository.

        Args:
            source_identifier: Repository in "owner/repo" format
                             or "all" to collect all known repos
            use_known_repos: Use pre-defined benchmark data

        Returns:
            CollectorResult with collected benchmarks
        """
        start_time = time.time()
        result = CollectorResult(
            extraction_method=ExtractionMethod.MANUAL,
        )

        try:
            if source_identifier.lower() == "all":
                # Collect all known repos
                for repo_id, repo_config in KNOWN_REPOSITORIES.items():
                    benchmarks = self._extract_from_known_repo(repo_id, repo_config)
                    result.benchmarks.extend(benchmarks)
                result.extraction_confidence = 0.8
            elif use_known_repos and source_identifier in KNOWN_REPOSITORIES:
                repo_config = KNOWN_REPOSITORIES[source_identifier]
                benchmarks = self._extract_from_known_repo(source_identifier, repo_config)
                result.benchmarks.extend(benchmarks)
                result.extraction_confidence = 0.8
            else:
                # Try to fetch from GitHub API
                api_benchmarks = self._fetch_from_github(source_identifier)
                result.benchmarks.extend(api_benchmarks)
                result.extraction_confidence = 0.5

            result.benchmarks = self.postprocess_benchmarks(result.benchmarks)

        except Exception as e:
            result.errors.append(f"Collection error: {str(e)}")
            logger.exception(f"Error collecting from GitHub {source_identifier}")

        result.extraction_duration_sec = time.time() - start_time
        return result

    def _extract_from_known_repo(
        self,
        repo_id: str,
        repo_config: Dict[str, Any],
    ) -> List[ExtendedBenchmark]:
        """Extract benchmarks from known repository configuration."""
        benchmarks = []

        for bm_config in repo_config.get("benchmarks", []):
            # Create parallelism config
            p_config = bm_config.get("parallelism", {})
            parallelism = ParallelismConfig(
                tensor_parallel=p_config.get("tp", 1),
                pipeline_parallel=p_config.get("pp", 1),
                data_parallel=p_config.get("dp", bm_config.get("num_gpus", 1)),
                zero_stage=p_config.get("zero_stage", 0),
                fsdp_enabled=p_config.get("fsdp", False),
            )

            # Create quantization config
            quant_method = bm_config.get("quantization")
            quantization = QuantizationConfig(
                model_precision=quant_method if quant_method else QuantizationMethod.BF16,
            )

            # Create multi-model config if present
            multi_model = None
            mm_config = bm_config.get("multi_model")
            if mm_config:
                multi_model = MultiModelConfig(
                    num_models=mm_config.get("num_models", 1),
                    has_policy_model=mm_config.get("has_policy_model", True),
                    has_reference_model=mm_config.get("has_reference_model", False),
                    has_reward_model=mm_config.get("has_reward_model", False),
                    has_value_model=mm_config.get("has_value_model", False),
                    reference_model_frozen=mm_config.get("reference_model_frozen", True),
                    reference_model_offloaded=mm_config.get("reference_model_offloaded", False),
                )

            # Create PEFT config if present
            peft = None
            peft_config = bm_config.get("peft")
            if peft_config:
                peft = PEFTConfig(
                    method=peft_config.get("method", "lora"),
                    lora_rank=peft_config.get("lora_rank", 8),
                    lora_alpha=peft_config.get("lora_alpha", 16),
                    qlora_bits=peft_config.get("qlora_bits", 4),
                )

            # Create metrics
            metrics = ReportedMetrics(
                mfu=bm_config.get("mfu"),
                tokens_per_second=bm_config.get("tokens_per_second"),
                memory_per_gpu_gb=bm_config.get("memory_per_gpu_gb"),
            )

            # Create provenance
            provenance = SourceProvenance(
                source_type=SourceType.GITHUB_REPO,
                source_url=f"https://github.com/{repo_id}",
                source_title=f"GitHub: {repo_config.get('name', repo_id)}",
                organization=repo_config.get("organization", ""),
                extraction_method="manual",
                extraction_date=datetime.now().strftime("%Y-%m-%d"),
                notes=repo_config.get("description", ""),
            )

            benchmark = ExtendedBenchmark(
                benchmark_id="",
                name=bm_config["name"],
                provenance=provenance,
                model_name=bm_config.get("model_name", "Unknown"),
                model_params_b=bm_config.get("model_params_b", 0.0),
                training_type=bm_config.get("training_type", TrainingType.SFT),
                num_gpus=bm_config.get("num_gpus", 1),
                hardware_type=bm_config.get("hardware_type", HardwareType.A100_80GB),
                batch_size=bm_config.get("batch_size", 1),
                seq_length=bm_config.get("seq_length", 2048),
                parallelism=parallelism,
                quantization=quantization,
                multi_model=multi_model,
                peft=peft,
                optimizer=bm_config.get("optimizer", OptimizerType.ADAMW),
                metrics=metrics,
                verification_status=VerificationStatus.UNVERIFIED,
                notes=bm_config.get("notes", ""),
                tags=["github", f"github:{repo_id}", repo_config.get("name", "")],
            )

            benchmarks.append(benchmark)

        return benchmarks

    def _fetch_from_github(
        self,
        repo_id: str,
    ) -> List[ExtendedBenchmark]:
        """Fetch and parse repository README for benchmarks."""
        try:
            import requests

            headers = {"Accept": "application/vnd.github.v3+json"}
            if self.github_token:
                headers["Authorization"] = f"token {self.github_token}"

            # Fetch README
            url = f"{self.GITHUB_API_URL}/repos/{repo_id}/readme"
            response = requests.get(url, headers=headers, timeout=self.timeout)

            if response.status_code == 404:
                logger.warning(f"Repository not found: {repo_id}")
                return []

            response.raise_for_status()
            readme_data = response.json()

            # Decode README content
            import base64
            content = base64.b64decode(readme_data.get("content", "")).decode("utf-8")

            # Extract benchmarks using pattern matching
            return self._extract_from_readme(repo_id, content)

        except Exception as e:
            logger.error(f"Error fetching from GitHub: {e}")
            return []

    def _extract_from_readme(
        self,
        repo_id: str,
        readme_content: str,
    ) -> List[ExtendedBenchmark]:
        """Extract benchmark data from README content."""
        benchmarks = []

        # Look for benchmark tables or sections
        extracted = PatternExtractor.extract_all(readme_content)

        # Only create benchmark if we have meaningful data
        if any(v for v in extracted.values() if v is not None):
            provenance = SourceProvenance(
                source_type=SourceType.GITHUB_REPO,
                source_url=f"https://github.com/{repo_id}",
                source_title=f"GitHub: {repo_id}",
                extraction_method="regex",
                extraction_date=datetime.now().strftime("%Y-%m-%d"),
            )

            metrics = ReportedMetrics(
                mfu=extracted.get("mfu"),
                tokens_per_second=extracted.get("tokens_per_second"),
                memory_per_gpu_gb=extracted.get("memory_gb"),
            )

            benchmark = ExtendedBenchmark(
                benchmark_id="",
                name=f"GitHub: {repo_id} (auto-extracted)",
                provenance=provenance,
                model_name="Unknown",
                model_params_b=extracted.get("model_size_b") or 0.0,
                num_gpus=extracted.get("num_gpus") or 1,
                batch_size=extracted.get("batch_size") or 1,
                seq_length=extracted.get("seq_length") or 2048,
                metrics=metrics,
                verification_status=VerificationStatus.UNVERIFIED,
                notes="Automatically extracted from README",
                tags=["github", "auto-extracted", f"github:{repo_id}"],
            )

            benchmarks.append(benchmark)

        return benchmarks

    def collect_all_known(self) -> CollectorResult:
        """Collect all known GitHub repository benchmarks."""
        return self.collect("all")

    def collect_by_category(self, category: str) -> CollectorResult:
        """
        Collect benchmarks by category.

        Args:
            category: "optimizer", "rlhf", "framework", or "all"
        """
        result = CollectorResult(
            extraction_method=ExtractionMethod.MANUAL,
            extraction_confidence=0.8,
        )

        category_map = {
            "optimizer": ["jiaweizzhao/GaLore", "zhuzilin/APOLLO", "huggingface/bitsandbytes"],
            "rlhf": ["OpenRLHF/OpenRLHF", "huggingface/trl"],
            "framework": ["hiyouga/LLaMA-Factory", "microsoft/DeepSpeed", "facebookresearch/llama-recipes"],
        }

        repos = category_map.get(category.lower(), list(KNOWN_REPOSITORIES.keys()))

        for repo_id in repos:
            if repo_id in KNOWN_REPOSITORIES:
                benchmarks = self._extract_from_known_repo(repo_id, KNOWN_REPOSITORIES[repo_id])
                result.benchmarks.extend(benchmarks)

        result.benchmarks = self.postprocess_benchmarks(result.benchmarks)
        return result

    def list_known_repos(self) -> List[Dict[str, Any]]:
        """List all known repositories."""
        return [
            {
                "repo_id": repo_id,
                "name": config.get("name", ""),
                "organization": config.get("organization", ""),
                "description": config.get("description", ""),
                "num_benchmarks": len(config.get("benchmarks", [])),
            }
            for repo_id, config in KNOWN_REPOSITORIES.items()
        ]
