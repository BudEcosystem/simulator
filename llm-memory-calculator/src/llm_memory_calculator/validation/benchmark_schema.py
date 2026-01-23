"""
Extended Benchmark Schema for Ground Truth Dataset.

This module defines comprehensive data structures for training benchmarks
with enhanced provenance tracking, confidence scoring, and validation.

Key Components:
- SourceType: Categorizes data sources (paper, mlperf, github, blog)
- TrainingType: All supported training types (pretraining, sft, dpo, ppo, etc.)
- QuantizationMethod: Quantization approaches (fp16, bf16, fp8, int8, nf4)
- VerificationStatus: Data verification state (verified, cross_checked, unverified)
- SourceProvenance: Tracks data origin and extraction method
- ExtendedBenchmark: Full benchmark with confidence scoring
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import datetime
import hashlib
import json


class SourceType(Enum):
    """Type of data source for benchmark extraction."""
    ACADEMIC_PAPER = "academic_paper"  # arXiv, peer-reviewed papers
    TECHNICAL_REPORT = "technical_report"  # Official company reports
    MLPERF = "mlperf"  # MLPerf official submissions
    GITHUB_REPO = "github_repo"  # GitHub repository data
    HUGGINGFACE = "huggingface"  # HuggingFace model cards
    ENGINEERING_BLOG = "engineering_blog"  # Company engineering blogs
    COMMUNITY = "community"  # Community benchmarks
    MANUAL_ENTRY = "manual_entry"  # Manually curated data


class TrainingType(Enum):
    """All supported training types."""
    PRETRAINING = "pretraining"
    CONTINUED_PRETRAINING = "continued_pretraining"
    SFT = "sft"  # Supervised Fine-Tuning
    FULL_FINETUNE = "full_finetune"
    LORA = "lora"  # Low-Rank Adaptation
    QLORA = "qlora"  # Quantized LoRA
    DORA = "dora"  # Weight-Decomposed LoRA
    DPO = "dpo"  # Direct Preference Optimization
    ORPO = "orpo"  # Odds Ratio Preference Optimization
    SIMPO = "simpo"  # Simple Preference Optimization
    KTO = "kto"  # Kahneman-Tversky Optimization
    PPO = "ppo"  # Proximal Policy Optimization (RLHF)
    GRPO = "grpo"  # Group Relative Policy Optimization
    REINFORCE = "reinforce"
    REWARD_MODELING = "reward_modeling"
    DISTILLATION = "distillation"
    PRUNING = "pruning"


class QuantizationMethod(Enum):
    """Quantization methods and precisions."""
    FP32 = "fp32"
    TF32 = "tf32"
    BF16 = "bf16"
    FP16 = "fp16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    INT8 = "int8"
    INT8_DYNAMIC = "int8_dynamic"
    INT4 = "int4"
    NF4 = "nf4"  # Normal Float 4-bit (QLoRA)
    GPTQ = "gptq"
    AWQ = "awq"
    GGUF_Q4 = "gguf_q4"
    GGUF_Q8 = "gguf_q8"


class OptimizerType(Enum):
    """Optimizer types with memory characteristics."""
    ADAMW = "adamw"  # 8 bytes/param
    ADAMW_8BIT = "adamw_8bit"  # 2 bytes/param
    ADAMW_FUSED = "adamw_fused"  # 8 bytes/param with kernel fusion
    ADAM = "adam"  # 8 bytes/param
    ADAM_8BIT = "adam_8bit"  # 2 bytes/param
    SGD = "sgd"  # 4 bytes/param
    SGD_MOMENTUM = "sgd_momentum"  # 4 bytes/param
    LION = "lion"  # 4 bytes/param (sign-based)
    ADAFACTOR = "adafactor"  # ~4 bytes/param (factorized)
    CAME = "came"  # Similar to Adafactor
    GALORE = "galore"  # Low-rank (~0.25x AdamW)
    GALORE_8BIT = "galore_8bit"  # Low-rank 8-bit
    APOLLO = "apollo"  # Low-rank optimizer
    ADAM_MINI = "adam_mini"  # Memory-efficient Adam
    Q_ADAM_MINI = "q_adam_mini"  # Quantized Adam-mini
    MUON = "muon"  # Hybrid optimizer
    BADAM = "badam"  # Block-wise Adam
    SOPHIA = "sophia"  # Second-order optimizer
    SHAMPOO = "shampoo"  # Preconditioned optimizer
    LOMO = "lomo"  # Low-Memory Optimization
    ADALOMO = "adalomo"  # Adaptive LOMO


class VerificationStatus(Enum):
    """Verification status of benchmark data."""
    VERIFIED = "verified"  # Verified against multiple sources
    CROSS_CHECKED = "cross_checked"  # Checked against one other source
    UNVERIFIED = "unverified"  # Single source, not verified
    DISPUTED = "disputed"  # Conflicting data from sources


class HardwareType(Enum):
    """Hardware types for training."""
    # NVIDIA GPUs
    A100_40GB = "a100_40gb"
    A100_80GB = "a100_80gb"
    H100_SXM = "h100_sxm"
    H100_PCIE = "h100_pcie"
    H200 = "h200"
    GH200 = "gh200"
    B100 = "b100"
    B200 = "b200"
    GB200 = "gb200"
    # AMD GPUs
    MI250X = "mi250x"
    MI300X = "mi300x"
    MI300A = "mi300a"
    MI325X = "mi325x"
    # Intel
    GAUDI2 = "gaudi2"
    GAUDI3 = "gaudi3"
    # Google
    TPU_V4 = "tpu_v4"
    TPU_V5E = "tpu_v5e"
    TPU_V5P = "tpu_v5p"
    TPU_V6 = "tpu_v6"
    # Other
    OTHER = "other"


class GPUScale(Enum):
    """GPU scale tiers for coverage tracking."""
    SINGLE = "single"  # 1 GPU
    SMALL = "small"  # 2-8 GPUs
    MEDIUM = "medium"  # 8-64 GPUs
    LARGE = "large"  # 64-256 GPUs
    XLARGE = "xlarge"  # 256-1000 GPUs
    HYPERSCALE = "hyperscale"  # 1000+ GPUs


class ModelSizeCategory(Enum):
    """Model size categories."""
    TINY = "tiny"  # < 1B
    SMALL = "small"  # 1B - 10B
    MEDIUM = "medium"  # 10B - 50B
    LARGE = "large"  # 50B - 100B
    XLARGE = "xlarge"  # 100B - 500B
    ULTRA = "ultra"  # > 500B


# Organization reputation scores for confidence calculation
ORGANIZATION_REPUTATION: Dict[str, int] = {
    # Tier 1: Top AI Labs (score: 25)
    "meta": 25,
    "nvidia": 25,
    "google": 25,
    "deepmind": 25,
    "anthropic": 25,
    "openai": 25,

    # Tier 2: Major AI Companies (score: 22)
    "microsoft": 22,
    "amazon": 22,
    "alibaba": 22,
    "tencent": 22,
    "baidu": 22,
    "bytedance": 22,

    # Tier 3: Notable AI Orgs (score: 18)
    "huggingface": 18,
    "mistral": 18,
    "cohere": 18,
    "together": 18,
    "deepseek": 18,
    "01ai": 18,
    "databricks": 18,
    "tii": 18,  # Falcon

    # Tier 4: Research Institutions (score: 15)
    "stanford": 15,
    "berkeley": 15,
    "cmu": 15,
    "mit": 15,
    "princeton": 15,
    "uw": 15,

    # Tier 5: Community/Other (score: 10)
    "community": 10,
    "other": 10,
}


# Source type weights for confidence calculation
SOURCE_TYPE_WEIGHTS: Dict[SourceType, int] = {
    SourceType.MLPERF: 30,
    SourceType.ACADEMIC_PAPER: 25,
    SourceType.TECHNICAL_REPORT: 23,
    SourceType.GITHUB_REPO: 18,
    SourceType.HUGGINGFACE: 16,
    SourceType.ENGINEERING_BLOG: 20,
    SourceType.COMMUNITY: 12,
    SourceType.MANUAL_ENTRY: 10,
}


# Metric completeness scores
METRIC_COMPLETENESS_SCORES: Dict[str, int] = {
    "mfu": 10,
    "tokens_per_second": 8,
    "samples_per_second": 6,
    "step_time_ms": 7,
    "memory_usage_gb": 8,
    "peak_memory_gb": 9,
    "throughput_tflops": 7,
    "training_time_hours": 5,
}


@dataclass
class SourceProvenance:
    """Tracks the origin and extraction method of benchmark data."""

    # Source identification
    source_type: SourceType
    source_url: Optional[str] = None
    source_title: Optional[str] = None

    # Authorship
    authors: List[str] = field(default_factory=list)
    organization: Optional[str] = None

    # Publication info
    publication_date: Optional[str] = None  # YYYY-MM-DD
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None

    # Extraction metadata
    extraction_method: str = "manual"  # manual, llm_extraction, regex, api
    extraction_date: Optional[str] = None  # YYYY-MM-DD
    extracted_by: Optional[str] = None  # tool/person name

    # Data location in source
    page_numbers: Optional[str] = None
    table_reference: Optional[str] = None
    figure_reference: Optional[str] = None
    section_reference: Optional[str] = None

    # Quality indicators
    raw_text_excerpt: Optional[str] = None  # Original text for verification
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_type": self.source_type.value,
            "source_url": self.source_url,
            "source_title": self.source_title,
            "authors": self.authors,
            "organization": self.organization,
            "publication_date": self.publication_date,
            "arxiv_id": self.arxiv_id,
            "doi": self.doi,
            "extraction_method": self.extraction_method,
            "extraction_date": self.extraction_date,
            "extracted_by": self.extracted_by,
            "page_numbers": self.page_numbers,
            "table_reference": self.table_reference,
            "figure_reference": self.figure_reference,
            "section_reference": self.section_reference,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceProvenance":
        """Create from dictionary."""
        return cls(
            source_type=SourceType(data.get("source_type", "manual_entry")),
            source_url=data.get("source_url"),
            source_title=data.get("source_title"),
            authors=data.get("authors", []),
            organization=data.get("organization"),
            publication_date=data.get("publication_date"),
            arxiv_id=data.get("arxiv_id"),
            doi=data.get("doi"),
            extraction_method=data.get("extraction_method", "manual"),
            extraction_date=data.get("extraction_date"),
            extracted_by=data.get("extracted_by"),
            page_numbers=data.get("page_numbers"),
            table_reference=data.get("table_reference"),
            figure_reference=data.get("figure_reference"),
            section_reference=data.get("section_reference"),
            notes=data.get("notes"),
        )


@dataclass
class QuantizationConfig:
    """Detailed quantization configuration."""

    # Primary precision
    model_precision: QuantizationMethod = QuantizationMethod.BF16
    compute_precision: Optional[QuantizationMethod] = None  # e.g., TF32 for matmuls

    # Quantization details
    weights_quantized: bool = False
    activations_quantized: bool = False
    gradients_quantized: bool = False
    optimizer_quantized: bool = False

    # Specific methods
    weight_quant_method: Optional[QuantizationMethod] = None
    activation_quant_method: Optional[QuantizationMethod] = None

    # Memory-related
    bytes_per_param: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_precision": self.model_precision.value,
            "compute_precision": self.compute_precision.value if self.compute_precision else None,
            "weights_quantized": self.weights_quantized,
            "activations_quantized": self.activations_quantized,
            "gradients_quantized": self.gradients_quantized,
            "optimizer_quantized": self.optimizer_quantized,
            "weight_quant_method": self.weight_quant_method.value if self.weight_quant_method else None,
            "activation_quant_method": self.activation_quant_method.value if self.activation_quant_method else None,
            "bytes_per_param": self.bytes_per_param,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuantizationConfig":
        """Create from dictionary."""
        return cls(
            model_precision=QuantizationMethod(data.get("model_precision", "bf16")),
            compute_precision=QuantizationMethod(data["compute_precision"]) if data.get("compute_precision") else None,
            weights_quantized=data.get("weights_quantized", False),
            activations_quantized=data.get("activations_quantized", False),
            gradients_quantized=data.get("gradients_quantized", False),
            optimizer_quantized=data.get("optimizer_quantized", False),
            weight_quant_method=QuantizationMethod(data["weight_quant_method"]) if data.get("weight_quant_method") else None,
            activation_quant_method=QuantizationMethod(data["activation_quant_method"]) if data.get("activation_quant_method") else None,
            bytes_per_param=data.get("bytes_per_param"),
        )


@dataclass
class ParallelismConfig:
    """Parallelism configuration for distributed training."""

    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    data_parallel: int = 1
    expert_parallel: int = 1
    context_parallel: int = 1
    sequence_parallel: bool = False

    # ZeRO stages
    zero_stage: int = 0  # 0, 1, 2, 3
    zero_offload_optimizer: bool = False
    zero_offload_param: bool = False

    # FSDP
    fsdp_enabled: bool = False
    fsdp_sharding_strategy: Optional[str] = None

    @property
    def total_parallelism(self) -> int:
        """Total parallelism degree."""
        return self.tensor_parallel * self.pipeline_parallel * self.data_parallel * self.expert_parallel

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tensor_parallel": self.tensor_parallel,
            "pipeline_parallel": self.pipeline_parallel,
            "data_parallel": self.data_parallel,
            "expert_parallel": self.expert_parallel,
            "context_parallel": self.context_parallel,
            "sequence_parallel": self.sequence_parallel,
            "zero_stage": self.zero_stage,
            "zero_offload_optimizer": self.zero_offload_optimizer,
            "zero_offload_param": self.zero_offload_param,
            "fsdp_enabled": self.fsdp_enabled,
            "fsdp_sharding_strategy": self.fsdp_sharding_strategy,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParallelismConfig":
        """Create from dictionary."""
        return cls(
            tensor_parallel=data.get("tensor_parallel", 1),
            pipeline_parallel=data.get("pipeline_parallel", 1),
            data_parallel=data.get("data_parallel", 1),
            expert_parallel=data.get("expert_parallel", 1),
            context_parallel=data.get("context_parallel", 1),
            sequence_parallel=data.get("sequence_parallel", False),
            zero_stage=data.get("zero_stage", 0),
            zero_offload_optimizer=data.get("zero_offload_optimizer", False),
            zero_offload_param=data.get("zero_offload_param", False),
            fsdp_enabled=data.get("fsdp_enabled", False),
            fsdp_sharding_strategy=data.get("fsdp_sharding_strategy"),
        )


@dataclass
class MultiModelConfig:
    """Configuration for multi-model training (PPO, DPO, GRPO)."""

    # Number of model instances
    num_models: int = 1

    # Model roles
    has_policy_model: bool = True
    has_reference_model: bool = False
    has_reward_model: bool = False
    has_value_model: bool = False
    has_critic_model: bool = False

    # Model configurations
    reference_model_frozen: bool = True
    reward_model_frozen: bool = True
    reference_model_quantized: bool = False
    reward_model_as_lora: bool = False

    # Memory strategies
    reference_model_offloaded: bool = False
    share_base_weights: bool = False  # e.g., policy and value share backbone

    # GRPO specific
    grpo_num_groups: int = 1
    grpo_samples_per_group: int = 1

    def get_memory_multiplier(self) -> float:
        """Calculate memory multiplier relative to single model training."""
        multiplier = 1.0

        if self.has_reference_model:
            if self.reference_model_offloaded:
                multiplier += 0.0  # No GPU memory
            elif self.reference_model_quantized:
                multiplier += 0.3  # Quantized inference
            else:
                multiplier += 0.7  # Eval mode (no grads)

        if self.has_reward_model:
            if self.reward_model_as_lora:
                multiplier += 0.1  # LoRA adapters only
            else:
                multiplier += 0.7  # Full model in eval mode

        if self.has_value_model:
            if self.share_base_weights:
                multiplier += 0.2  # Just value head
            else:
                multiplier += 1.0  # Full model with grads

        return multiplier

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_models": self.num_models,
            "has_policy_model": self.has_policy_model,
            "has_reference_model": self.has_reference_model,
            "has_reward_model": self.has_reward_model,
            "has_value_model": self.has_value_model,
            "has_critic_model": self.has_critic_model,
            "reference_model_frozen": self.reference_model_frozen,
            "reward_model_frozen": self.reward_model_frozen,
            "reference_model_quantized": self.reference_model_quantized,
            "reward_model_as_lora": self.reward_model_as_lora,
            "reference_model_offloaded": self.reference_model_offloaded,
            "share_base_weights": self.share_base_weights,
            "grpo_num_groups": self.grpo_num_groups,
            "grpo_samples_per_group": self.grpo_samples_per_group,
            "memory_multiplier": self.get_memory_multiplier(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiModelConfig":
        """Create from dictionary."""
        return cls(
            num_models=data.get("num_models", 1),
            has_policy_model=data.get("has_policy_model", True),
            has_reference_model=data.get("has_reference_model", False),
            has_reward_model=data.get("has_reward_model", False),
            has_value_model=data.get("has_value_model", False),
            has_critic_model=data.get("has_critic_model", False),
            reference_model_frozen=data.get("reference_model_frozen", True),
            reward_model_frozen=data.get("reward_model_frozen", True),
            reference_model_quantized=data.get("reference_model_quantized", False),
            reward_model_as_lora=data.get("reward_model_as_lora", False),
            reference_model_offloaded=data.get("reference_model_offloaded", False),
            share_base_weights=data.get("share_base_weights", False),
            grpo_num_groups=data.get("grpo_num_groups", 1),
            grpo_samples_per_group=data.get("grpo_samples_per_group", 1),
        )


@dataclass
class PEFTConfig:
    """Parameter-Efficient Fine-Tuning configuration."""

    method: str = "full"  # full, lora, qlora, dora, prefix, prompt, adapter

    # LoRA specific
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_rs_lora: bool = False  # Rank-stabilized LoRA

    # DoRA specific
    dora_enabled: bool = False

    # QLoRA specific
    qlora_bits: int = 4
    qlora_double_quant: bool = True

    # Prefix/Prompt tuning
    prefix_length: int = 0
    prompt_length: int = 0

    # Adapter
    adapter_hidden_size: int = 0

    @property
    def trainable_param_fraction(self) -> float:
        """Estimated fraction of trainable parameters."""
        if self.method == "full":
            return 1.0
        elif self.method in ("lora", "qlora", "dora"):
            # Rough estimate: rank * 2 * target_layers / total_params
            # Typically 0.1% - 5% depending on configuration
            return 0.01  # Conservative 1% estimate
        elif self.method in ("prefix", "prompt"):
            return 0.001  # Usually < 0.1%
        elif self.method == "adapter":
            return 0.005  # Typically 0.5%
        return 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
            "lora_rs_lora": self.lora_rs_lora,
            "dora_enabled": self.dora_enabled,
            "qlora_bits": self.qlora_bits,
            "qlora_double_quant": self.qlora_double_quant,
            "prefix_length": self.prefix_length,
            "prompt_length": self.prompt_length,
            "adapter_hidden_size": self.adapter_hidden_size,
            "trainable_param_fraction": self.trainable_param_fraction,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PEFTConfig":
        """Create from dictionary."""
        return cls(
            method=data.get("method", "full"),
            lora_rank=data.get("lora_rank", 8),
            lora_alpha=data.get("lora_alpha", 16),
            lora_dropout=data.get("lora_dropout", 0.0),
            lora_target_modules=data.get("lora_target_modules", ["q_proj", "v_proj"]),
            lora_rs_lora=data.get("lora_rs_lora", False),
            dora_enabled=data.get("dora_enabled", False),
            qlora_bits=data.get("qlora_bits", 4),
            qlora_double_quant=data.get("qlora_double_quant", True),
            prefix_length=data.get("prefix_length", 0),
            prompt_length=data.get("prompt_length", 0),
            adapter_hidden_size=data.get("adapter_hidden_size", 0),
        )


@dataclass
class ReportedMetrics:
    """Reported performance metrics from benchmark."""

    # Throughput metrics
    mfu: Optional[float] = None  # Model FLOPs Utilization (0-1)
    hfu: Optional[float] = None  # Hardware FLOPs Utilization (0-1)
    tokens_per_second: Optional[float] = None  # Global tokens/sec
    tokens_per_second_per_gpu: Optional[float] = None
    samples_per_second: Optional[float] = None

    # Timing metrics
    step_time_ms: Optional[float] = None
    forward_time_ms: Optional[float] = None
    backward_time_ms: Optional[float] = None
    optimizer_time_ms: Optional[float] = None
    communication_time_ms: Optional[float] = None

    # Memory metrics
    memory_usage_gb: Optional[float] = None  # Reported memory usage
    peak_memory_gb: Optional[float] = None
    memory_per_gpu_gb: Optional[float] = None
    activation_memory_gb: Optional[float] = None

    # Training metrics
    training_time_hours: Optional[float] = None
    total_tokens_trained: Optional[float] = None
    final_loss: Optional[float] = None

    # Compute metrics
    throughput_tflops: Optional[float] = None  # Achieved TFLOP/s
    throughput_tflops_per_gpu: Optional[float] = None

    def get_completeness_score(self) -> int:
        """Calculate metric completeness score."""
        score = 0
        if self.mfu is not None:
            score += METRIC_COMPLETENESS_SCORES.get("mfu", 0)
        if self.tokens_per_second is not None:
            score += METRIC_COMPLETENESS_SCORES.get("tokens_per_second", 0)
        if self.samples_per_second is not None:
            score += METRIC_COMPLETENESS_SCORES.get("samples_per_second", 0)
        if self.step_time_ms is not None:
            score += METRIC_COMPLETENESS_SCORES.get("step_time_ms", 0)
        if self.memory_usage_gb is not None:
            score += METRIC_COMPLETENESS_SCORES.get("memory_usage_gb", 0)
        if self.peak_memory_gb is not None:
            score += METRIC_COMPLETENESS_SCORES.get("peak_memory_gb", 0)
        if self.throughput_tflops is not None:
            score += METRIC_COMPLETENESS_SCORES.get("throughput_tflops", 0)
        if self.training_time_hours is not None:
            score += METRIC_COMPLETENESS_SCORES.get("training_time_hours", 0)
        return score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding None values)."""
        result = {}
        for key, value in {
            "mfu": self.mfu,
            "hfu": self.hfu,
            "tokens_per_second": self.tokens_per_second,
            "tokens_per_second_per_gpu": self.tokens_per_second_per_gpu,
            "samples_per_second": self.samples_per_second,
            "step_time_ms": self.step_time_ms,
            "forward_time_ms": self.forward_time_ms,
            "backward_time_ms": self.backward_time_ms,
            "optimizer_time_ms": self.optimizer_time_ms,
            "communication_time_ms": self.communication_time_ms,
            "memory_usage_gb": self.memory_usage_gb,
            "peak_memory_gb": self.peak_memory_gb,
            "memory_per_gpu_gb": self.memory_per_gpu_gb,
            "activation_memory_gb": self.activation_memory_gb,
            "training_time_hours": self.training_time_hours,
            "total_tokens_trained": self.total_tokens_trained,
            "final_loss": self.final_loss,
            "throughput_tflops": self.throughput_tflops,
            "throughput_tflops_per_gpu": self.throughput_tflops_per_gpu,
        }.items():
            if value is not None:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReportedMetrics":
        """Create from dictionary."""
        return cls(
            mfu=data.get("mfu"),
            hfu=data.get("hfu"),
            tokens_per_second=data.get("tokens_per_second"),
            tokens_per_second_per_gpu=data.get("tokens_per_second_per_gpu"),
            samples_per_second=data.get("samples_per_second"),
            step_time_ms=data.get("step_time_ms"),
            forward_time_ms=data.get("forward_time_ms"),
            backward_time_ms=data.get("backward_time_ms"),
            optimizer_time_ms=data.get("optimizer_time_ms"),
            communication_time_ms=data.get("communication_time_ms"),
            memory_usage_gb=data.get("memory_usage_gb"),
            peak_memory_gb=data.get("peak_memory_gb"),
            memory_per_gpu_gb=data.get("memory_per_gpu_gb"),
            activation_memory_gb=data.get("activation_memory_gb"),
            training_time_hours=data.get("training_time_hours"),
            total_tokens_trained=data.get("total_tokens_trained"),
            final_loss=data.get("final_loss"),
            throughput_tflops=data.get("throughput_tflops"),
            throughput_tflops_per_gpu=data.get("throughput_tflops_per_gpu"),
        )


@dataclass
class ExtendedBenchmark:
    """
    Extended benchmark with full provenance and confidence scoring.

    This is the primary data structure for ground truth benchmarks.
    """

    # Identification
    benchmark_id: str  # Unique identifier
    name: str  # Human-readable name

    # Source provenance
    provenance: SourceProvenance

    # Model configuration
    model_name: str
    model_params_b: float
    model_architecture: str = "transformer"  # transformer, mamba, hybrid
    is_moe: bool = False
    num_experts: int = 1
    active_experts: int = 1

    # Training configuration
    training_type: TrainingType = TrainingType.PRETRAINING
    batch_size: int = 1
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    seq_length: int = 2048

    # Hardware configuration
    num_gpus: int = 1
    hardware_type: HardwareType = HardwareType.H100_SXM
    hardware_name: str = ""  # Free text hardware name
    gpu_memory_gb: float = 80.0

    # Parallelism
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)

    # Quantization
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)

    # Multi-model (for PPO, DPO, etc.)
    multi_model: Optional[MultiModelConfig] = None

    # PEFT
    peft: Optional[PEFTConfig] = None

    # Optimizer
    optimizer: OptimizerType = OptimizerType.ADAMW
    optimizer_details: Optional[Dict[str, Any]] = None

    # Reported metrics
    metrics: ReportedMetrics = field(default_factory=ReportedMetrics)

    # Verification
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    verification_sources: List[str] = field(default_factory=list)
    verification_notes: Optional[str] = None

    # Additional info
    gradient_checkpointing: bool = True
    flash_attention: bool = False
    notes: str = ""
    tags: List[str] = field(default_factory=list)

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())

    def __post_init__(self):
        """Generate benchmark_id if not provided."""
        if not self.benchmark_id:
            # Generate deterministic ID from key fields
            key_str = f"{self.name}_{self.model_name}_{self.model_params_b}_{self.num_gpus}"
            self.benchmark_id = hashlib.md5(key_str.encode()).hexdigest()[:12]

    @property
    def gpu_scale(self) -> GPUScale:
        """Determine GPU scale tier."""
        if self.num_gpus == 1:
            return GPUScale.SINGLE
        elif self.num_gpus <= 8:
            return GPUScale.SMALL
        elif self.num_gpus <= 64:
            return GPUScale.MEDIUM
        elif self.num_gpus <= 256:
            return GPUScale.LARGE
        elif self.num_gpus <= 1000:
            return GPUScale.XLARGE
        else:
            return GPUScale.HYPERSCALE

    @property
    def model_size_category(self) -> ModelSizeCategory:
        """Determine model size category."""
        if self.model_params_b < 1:
            return ModelSizeCategory.TINY
        elif self.model_params_b < 10:
            return ModelSizeCategory.SMALL
        elif self.model_params_b < 50:
            return ModelSizeCategory.MEDIUM
        elif self.model_params_b < 100:
            return ModelSizeCategory.LARGE
        elif self.model_params_b < 500:
            return ModelSizeCategory.XLARGE
        else:
            return ModelSizeCategory.ULTRA

    def calculate_confidence_score(self) -> float:
        """
        Calculate confidence score (0-100) based on:
        - Source type weight
        - Organization reputation
        - Metric completeness
        - Verification status
        """
        # Source type weight (max 30)
        source_score = SOURCE_TYPE_WEIGHTS.get(self.provenance.source_type, 10)

        # Organization reputation (max 25)
        org = (self.provenance.organization or "").lower()
        org_score = 10  # Default
        for org_key, score in ORGANIZATION_REPUTATION.items():
            if org_key in org:
                org_score = score
                break

        # Metric completeness (max 30)
        completeness_score = min(self.metrics.get_completeness_score(), 30)

        # Verification bonus (max 15)
        verification_scores = {
            VerificationStatus.VERIFIED: 15,
            VerificationStatus.CROSS_CHECKED: 10,
            VerificationStatus.UNVERIFIED: 0,
            VerificationStatus.DISPUTED: -5,
        }
        verification_score = verification_scores.get(self.verification_status, 0)

        # Total score (max 100)
        total = source_score + org_score + completeness_score + verification_score
        return min(max(total, 0), 100)

    @property
    def confidence(self) -> float:
        """Get normalized confidence (0-1)."""
        return self.calculate_confidence_score() / 100.0

    @property
    def is_high_confidence(self) -> bool:
        """Check if benchmark has high confidence (>= 0.8)."""
        return self.confidence >= 0.8

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate benchmark data plausibility.

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # MFU bounds check
        if self.metrics.mfu is not None:
            if not (0.10 <= self.metrics.mfu <= 0.75):
                issues.append(f"MFU {self.metrics.mfu} outside typical range [0.10, 0.75]")

        # Memory plausibility
        if self.metrics.memory_per_gpu_gb is not None:
            if self.metrics.memory_per_gpu_gb > self.gpu_memory_gb:
                issues.append(f"Memory {self.metrics.memory_per_gpu_gb}GB exceeds GPU capacity {self.gpu_memory_gb}GB")

        # Parallelism check
        total_parallel = self.parallelism.total_parallelism
        if total_parallel > self.num_gpus:
            issues.append(f"Parallelism {total_parallel} exceeds num_gpus {self.num_gpus}")

        # Model size vs GPU count
        min_gpus_for_model = int(self.model_params_b * 2 / self.gpu_memory_gb)  # Rough estimate
        if self.peft is None or self.peft.method == "full":
            if self.num_gpus < min_gpus_for_model and self.parallelism.zero_stage < 3:
                issues.append(f"Model {self.model_params_b}B may not fit on {self.num_gpus} GPUs without sharding")

        # Batch size consistency
        expected_effective_batch = self.micro_batch_size * self.gradient_accumulation_steps * self.parallelism.data_parallel
        if abs(expected_effective_batch - self.batch_size) > self.batch_size * 0.1:
            issues.append(f"Batch size {self.batch_size} inconsistent with micro_batch*grad_accum*dp={expected_effective_batch}")

        return len(issues) == 0, issues

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "benchmark_id": self.benchmark_id,
            "name": self.name,
            "provenance": self.provenance.to_dict(),
            "model_name": self.model_name,
            "model_params_b": self.model_params_b,
            "model_architecture": self.model_architecture,
            "is_moe": self.is_moe,
            "num_experts": self.num_experts,
            "active_experts": self.active_experts,
            "training_type": self.training_type.value,
            "batch_size": self.batch_size,
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "seq_length": self.seq_length,
            "num_gpus": self.num_gpus,
            "hardware_type": self.hardware_type.value,
            "hardware_name": self.hardware_name,
            "gpu_memory_gb": self.gpu_memory_gb,
            "parallelism": self.parallelism.to_dict(),
            "quantization": self.quantization.to_dict(),
            "multi_model": self.multi_model.to_dict() if self.multi_model else None,
            "peft": self.peft.to_dict() if self.peft else None,
            "optimizer": self.optimizer.value,
            "optimizer_details": self.optimizer_details,
            "metrics": self.metrics.to_dict(),
            "verification_status": self.verification_status.value,
            "verification_sources": self.verification_sources,
            "verification_notes": self.verification_notes,
            "gradient_checkpointing": self.gradient_checkpointing,
            "flash_attention": self.flash_attention,
            "notes": self.notes,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "confidence_score": self.calculate_confidence_score(),
            "gpu_scale": self.gpu_scale.value,
            "model_size_category": self.model_size_category.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtendedBenchmark":
        """Create from dictionary."""
        return cls(
            benchmark_id=data.get("benchmark_id", ""),
            name=data["name"],
            provenance=SourceProvenance.from_dict(data.get("provenance", {})),
            model_name=data["model_name"],
            model_params_b=data["model_params_b"],
            model_architecture=data.get("model_architecture", "transformer"),
            is_moe=data.get("is_moe", False),
            num_experts=data.get("num_experts", 1),
            active_experts=data.get("active_experts", 1),
            training_type=TrainingType(data.get("training_type", "pretraining")),
            batch_size=data.get("batch_size", 1),
            micro_batch_size=data.get("micro_batch_size", 1),
            gradient_accumulation_steps=data.get("gradient_accumulation_steps", 1),
            seq_length=data.get("seq_length", 2048),
            num_gpus=data.get("num_gpus", 1),
            hardware_type=HardwareType(data.get("hardware_type", "h100_sxm")),
            hardware_name=data.get("hardware_name", ""),
            gpu_memory_gb=data.get("gpu_memory_gb", 80.0),
            parallelism=ParallelismConfig.from_dict(data.get("parallelism", {})),
            quantization=QuantizationConfig.from_dict(data.get("quantization", {})),
            multi_model=MultiModelConfig.from_dict(data["multi_model"]) if data.get("multi_model") else None,
            peft=PEFTConfig.from_dict(data["peft"]) if data.get("peft") else None,
            optimizer=OptimizerType(data.get("optimizer", "adamw")),
            optimizer_details=data.get("optimizer_details"),
            metrics=ReportedMetrics.from_dict(data.get("metrics", {})),
            verification_status=VerificationStatus(data.get("verification_status", "unverified")),
            verification_sources=data.get("verification_sources", []),
            verification_notes=data.get("verification_notes"),
            gradient_checkpointing=data.get("gradient_checkpointing", True),
            flash_attention=data.get("flash_attention", False),
            notes=data.get("notes", ""),
            tags=data.get("tags", []),
            created_at=data.get("created_at", datetime.datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.datetime.now().isoformat()),
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "ExtendedBenchmark":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


# Pre-defined configurations for common training types
def create_ppo_config(
    reward_model_type: str = "full",
    reference_model_quantized: bool = False,
) -> MultiModelConfig:
    """Create configuration for PPO/RLHF training."""
    return MultiModelConfig(
        num_models=3,
        has_policy_model=True,
        has_reference_model=True,
        has_reward_model=True,
        has_value_model=True,
        reference_model_frozen=True,
        reward_model_frozen=True,
        reference_model_quantized=reference_model_quantized,
        reward_model_as_lora=(reward_model_type == "lora"),
        share_base_weights=True,  # Policy and value share backbone
    )


def create_dpo_config(
    reference_model_quantized: bool = False,
    reference_model_offloaded: bool = False,
) -> MultiModelConfig:
    """Create configuration for DPO training."""
    return MultiModelConfig(
        num_models=2,
        has_policy_model=True,
        has_reference_model=True,
        has_reward_model=False,
        has_value_model=False,
        reference_model_frozen=True,
        reference_model_quantized=reference_model_quantized,
        reference_model_offloaded=reference_model_offloaded,
    )


def create_grpo_config(
    num_groups: int = 4,
    samples_per_group: int = 4,
    reference_model_quantized: bool = False,
) -> MultiModelConfig:
    """Create configuration for GRPO training."""
    return MultiModelConfig(
        num_models=2,
        has_policy_model=True,
        has_reference_model=True,
        has_reward_model=False,
        has_value_model=False,
        reference_model_frozen=True,
        reference_model_quantized=reference_model_quantized,
        grpo_num_groups=num_groups,
        grpo_samples_per_group=samples_per_group,
    )


def create_lora_config(
    rank: int = 8,
    alpha: int = 16,
    target_modules: Optional[List[str]] = None,
) -> PEFTConfig:
    """Create LoRA configuration."""
    return PEFTConfig(
        method="lora",
        lora_rank=rank,
        lora_alpha=alpha,
        lora_target_modules=target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"],
    )


def create_qlora_config(
    rank: int = 8,
    alpha: int = 16,
    bits: int = 4,
    target_modules: Optional[List[str]] = None,
) -> PEFTConfig:
    """Create QLoRA configuration."""
    return PEFTConfig(
        method="qlora",
        lora_rank=rank,
        lora_alpha=alpha,
        qlora_bits=bits,
        qlora_double_quant=True,
        lora_target_modules=target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"],
    )
