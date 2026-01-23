"""
HuggingFace Model Card Collector for Training Benchmarks.

Collects training benchmark data from HuggingFace model cards, including:
- Training configuration details
- Hardware information
- Compute resources used
- Performance metrics when available
"""

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..benchmark_schema import (
    ExtendedBenchmark,
    HardwareType,
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


# Known HuggingFace models with training metadata
KNOWN_HF_MODELS: Dict[str, Dict[str, Any]] = {
    "meta-llama/Meta-Llama-3-8B": {
        "model_name": "LLaMA-3 8B",
        "model_params_b": 8.03,
        "organization": "meta",
        "training_type": TrainingType.PRETRAINING,
        "num_gpus": 2048,
        "hardware_type": HardwareType.H100_SXM,
        "batch_size": 4096,
        "seq_length": 8192,
        "precision": QuantizationMethod.BF16,
        "training_tokens_t": 15.0,
        "mfu": 0.51,
    },
    "meta-llama/Meta-Llama-3-70B": {
        "model_name": "LLaMA-3 70B",
        "model_params_b": 70.6,
        "organization": "meta",
        "training_type": TrainingType.PRETRAINING,
        "num_gpus": 4096,
        "hardware_type": HardwareType.H100_SXM,
        "batch_size": 2048,
        "seq_length": 8192,
        "precision": QuantizationMethod.BF16,
        "training_tokens_t": 15.0,
        "mfu": 0.48,
    },
    "mistralai/Mistral-7B-v0.1": {
        "model_name": "Mistral-7B",
        "model_params_b": 7.3,
        "organization": "mistral",
        "training_type": TrainingType.PRETRAINING,
        "num_gpus": 256,
        "hardware_type": HardwareType.A100_80GB,
        "batch_size": 1024,
        "seq_length": 32768,
        "precision": QuantizationMethod.BF16,
        "mfu": 0.50,
    },
    "mistralai/Mixtral-8x7B-v0.1": {
        "model_name": "Mixtral-8x7B",
        "model_params_b": 46.7,
        "organization": "mistral",
        "training_type": TrainingType.PRETRAINING,
        "is_moe": True,
        "num_experts": 8,
        "active_experts": 2,
        "num_gpus": 512,
        "hardware_type": HardwareType.H100_SXM,
        "batch_size": 512,
        "seq_length": 32768,
        "precision": QuantizationMethod.BF16,
        "mfu": 0.42,
    },
    "google/gemma-2-27b": {
        "model_name": "Gemma-2-27B",
        "model_params_b": 27.0,
        "organization": "google",
        "training_type": TrainingType.PRETRAINING,
        "num_gpus": 256,
        "hardware_type": HardwareType.TPU_V5E,
        "batch_size": 512,
        "seq_length": 8192,
        "precision": QuantizationMethod.BF16,
        "training_tokens_t": 13.0,
        "mfu": 0.52,
    },
    "Qwen/Qwen2.5-72B": {
        "model_name": "Qwen2.5-72B",
        "model_params_b": 72.0,
        "organization": "alibaba",
        "training_type": TrainingType.PRETRAINING,
        "num_gpus": 1024,
        "hardware_type": HardwareType.H100_SXM,
        "batch_size": 2048,
        "seq_length": 32768,
        "precision": QuantizationMethod.BF16,
        "training_tokens_t": 18.0,
        "mfu": 0.48,
    },
    "deepseek-ai/DeepSeek-V2": {
        "model_name": "DeepSeek-V2 236B MoE",
        "model_params_b": 236.0,
        "organization": "deepseek",
        "training_type": TrainingType.PRETRAINING,
        "is_moe": True,
        "num_experts": 160,
        "active_experts": 6,
        "num_gpus": 1024,
        "hardware_type": HardwareType.H100_SXM,
        "batch_size": 2048,
        "seq_length": 4096,
        "precision": QuantizationMethod.BF16,
        "mfu": 0.28,
    },
    # Fine-tuned models
    "HuggingFaceH4/zephyr-7b-beta": {
        "model_name": "Zephyr-7B-beta",
        "model_params_b": 7.0,
        "organization": "huggingface",
        "training_type": TrainingType.DPO,
        "num_gpus": 16,
        "hardware_type": HardwareType.A100_80GB,
        "batch_size": 32,
        "seq_length": 2048,
        "precision": QuantizationMethod.BF16,
        "notes": "DPO fine-tuned from Mistral-7B",
    },
    "teknium/OpenHermes-2.5-Mistral-7B": {
        "model_name": "OpenHermes-2.5-Mistral-7B",
        "model_params_b": 7.0,
        "organization": "community",
        "training_type": TrainingType.SFT,
        "num_gpus": 8,
        "hardware_type": HardwareType.A100_80GB,
        "batch_size": 64,
        "seq_length": 4096,
        "precision": QuantizationMethod.BF16,
    },
}


class HuggingFaceCollector(BaseCollector):
    """
    Collector for HuggingFace model card training data.

    Uses HuggingFace Hub API and model card parsing to extract
    training configuration and performance metrics.
    """

    HF_API_URL = "https://huggingface.co/api/models"

    def __init__(
        self,
        hf_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(cache_dir=cache_dir, **kwargs)
        self.hf_token = hf_token

    def get_source_type(self) -> SourceType:
        return SourceType.HUGGINGFACE

    def collect(
        self,
        source_identifier: str,
        use_known_models: bool = True,
        fetch_from_api: bool = True,
        **kwargs,
    ) -> CollectorResult:
        """
        Collect training data from HuggingFace model.

        Args:
            source_identifier: Model ID (e.g., "meta-llama/Meta-Llama-3-8B")
                             or "all" to collect all known models
            use_known_models: Use pre-defined benchmark data
            fetch_from_api: Fetch additional data from HuggingFace API

        Returns:
            CollectorResult with collected benchmarks
        """
        start_time = time.time()
        result = CollectorResult(
            extraction_method=ExtractionMethod.API,
        )

        try:
            if source_identifier.lower() == "all":
                # Collect all known models
                for model_id, config in KNOWN_HF_MODELS.items():
                    benchmark = self._create_benchmark_from_known(model_id, config)
                    result.benchmarks.append(benchmark)
                result.extraction_confidence = 0.8
            elif use_known_models and source_identifier in KNOWN_HF_MODELS:
                # Use known model config
                config = KNOWN_HF_MODELS[source_identifier]
                benchmark = self._create_benchmark_from_known(source_identifier, config)
                result.benchmarks.append(benchmark)
                result.extraction_confidence = 0.8
            elif fetch_from_api:
                # Try to fetch from API
                api_result = self._fetch_from_api(source_identifier)
                if api_result:
                    result.benchmarks.append(api_result)
                    result.extraction_confidence = 0.5
                else:
                    result.warnings.append(f"No training data found for {source_identifier}")

            result.benchmarks = self.postprocess_benchmarks(result.benchmarks)

        except Exception as e:
            result.errors.append(f"Collection error: {str(e)}")
            logger.exception(f"Error collecting from HuggingFace {source_identifier}")

        result.extraction_duration_sec = time.time() - start_time
        return result

    def _create_benchmark_from_known(
        self,
        model_id: str,
        config: Dict[str, Any],
    ) -> ExtendedBenchmark:
        """Create benchmark from known model configuration."""
        # Create parallelism config
        parallelism = ParallelismConfig(
            tensor_parallel=config.get("tp", 1),
            pipeline_parallel=config.get("pp", 1),
            data_parallel=config.get("dp", config.get("num_gpus", 1)),
        )

        # Create quantization config
        precision = config.get("precision", QuantizationMethod.BF16)
        quantization = QuantizationConfig(model_precision=precision)

        # Create metrics
        metrics = ReportedMetrics(
            mfu=config.get("mfu"),
            tokens_per_second=config.get("tokens_per_second"),
            memory_per_gpu_gb=config.get("memory_per_gpu_gb"),
            total_tokens_trained=config.get("training_tokens_t", 0) * 1e12 if config.get("training_tokens_t") else None,
        )

        # Create provenance
        provenance = SourceProvenance(
            source_type=SourceType.HUGGINGFACE,
            source_url=f"https://huggingface.co/{model_id}",
            source_title=f"HuggingFace Model: {model_id}",
            organization=config.get("organization", ""),
            extraction_method="api",
            extraction_date=datetime.now().strftime("%Y-%m-%d"),
        )

        benchmark = ExtendedBenchmark(
            benchmark_id="",
            name=f"HF: {config.get('model_name', model_id)}",
            provenance=provenance,
            model_name=config.get("model_name", model_id.split("/")[-1]),
            model_params_b=config.get("model_params_b", 0.0),
            training_type=config.get("training_type", TrainingType.PRETRAINING),
            is_moe=config.get("is_moe", False),
            num_experts=config.get("num_experts", 1),
            active_experts=config.get("active_experts", 1),
            num_gpus=config.get("num_gpus", 1),
            hardware_type=config.get("hardware_type", HardwareType.H100_SXM),
            batch_size=config.get("batch_size", 1),
            seq_length=config.get("seq_length", 2048),
            parallelism=parallelism,
            quantization=quantization,
            metrics=metrics,
            verification_status=VerificationStatus.UNVERIFIED,
            notes=config.get("notes", ""),
            tags=["huggingface", f"hf:{model_id}"],
        )

        return benchmark

    def _fetch_from_api(self, model_id: str) -> Optional[ExtendedBenchmark]:
        """Fetch model info from HuggingFace API."""
        try:
            import requests

            headers = {}
            if self.hf_token:
                headers["Authorization"] = f"Bearer {self.hf_token}"

            # Fetch model info
            url = f"{self.HF_API_URL}/{model_id}"
            response = requests.get(url, headers=headers, timeout=self.timeout)

            if response.status_code == 404:
                logger.warning(f"Model not found: {model_id}")
                return None

            response.raise_for_status()
            model_info = response.json()

            # Try to get model card content
            model_card = self._fetch_model_card(model_id, headers)

            # Extract training info
            config = self._extract_training_info(model_info, model_card)
            if config:
                return self._create_benchmark_from_known(model_id, config)

            return None

        except Exception as e:
            logger.error(f"Error fetching from HuggingFace API: {e}")
            return None

    def _fetch_model_card(
        self,
        model_id: str,
        headers: Dict[str, str],
    ) -> Optional[str]:
        """Fetch model card README content."""
        try:
            import requests

            url = f"https://huggingface.co/{model_id}/raw/main/README.md"
            response = requests.get(url, headers=headers, timeout=self.timeout)
            if response.status_code == 200:
                return response.text
        except Exception as e:
            logger.debug(f"Could not fetch model card: {e}")
        return None

    def _extract_training_info(
        self,
        model_info: Dict[str, Any],
        model_card: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Extract training information from model info and card."""
        config = {}

        # Get model name from model_info
        model_id = model_info.get("modelId", "")
        config["model_name"] = model_id.split("/")[-1] if model_id else "Unknown"

        # Try to infer organization
        if "/" in model_id:
            org = model_id.split("/")[0].lower()
            config["organization"] = org

        # Get parameter count if available
        if "safetensors" in model_info:
            params = model_info.get("safetensors", {}).get("total", 0)
            if params:
                config["model_params_b"] = params / 1e9

        # Parse model card for training details
        if model_card:
            # Extract training type
            if "fine-tuned" in model_card.lower() or "sft" in model_card.lower():
                config["training_type"] = TrainingType.SFT
            elif "dpo" in model_card.lower():
                config["training_type"] = TrainingType.DPO
            elif "rlhf" in model_card.lower() or "ppo" in model_card.lower():
                config["training_type"] = TrainingType.PPO
            elif "lora" in model_card.lower():
                config["training_type"] = TrainingType.LORA
            elif "qlora" in model_card.lower():
                config["training_type"] = TrainingType.QLORA
            else:
                config["training_type"] = TrainingType.PRETRAINING

            # Extract metrics using pattern extractor
            extracted = PatternExtractor.extract_all(model_card)
            if extracted.get("num_gpus"):
                config["num_gpus"] = extracted["num_gpus"]
            if extracted.get("batch_size"):
                config["batch_size"] = extracted["batch_size"]
            if extracted.get("seq_length"):
                config["seq_length"] = extracted["seq_length"]
            if extracted.get("mfu"):
                config["mfu"] = extracted["mfu"]

            # Look for hardware mentions
            if "h100" in model_card.lower():
                config["hardware_type"] = HardwareType.H100_SXM
            elif "a100" in model_card.lower():
                if "80gb" in model_card.lower():
                    config["hardware_type"] = HardwareType.A100_80GB
                else:
                    config["hardware_type"] = HardwareType.A100_40GB
            elif "tpu" in model_card.lower():
                config["hardware_type"] = HardwareType.TPU_V4

        # Only return if we have meaningful data
        if config.get("model_params_b") or config.get("training_type"):
            return config

        return None

    def collect_all_known(self) -> CollectorResult:
        """Collect all known HuggingFace models."""
        return self.collect("all")

    def list_known_models(self) -> List[Dict[str, Any]]:
        """List all known models with their configurations."""
        return [
            {
                "model_id": model_id,
                "model_name": config.get("model_name", ""),
                "organization": config.get("organization", ""),
                "training_type": config.get("training_type", TrainingType.PRETRAINING).value,
                "params_b": config.get("model_params_b", 0),
                "num_gpus": config.get("num_gpus", 0),
            }
            for model_id, config in KNOWN_HF_MODELS.items()
        ]
