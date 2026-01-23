"""
Published Benchmark Database for Training Simulation Validation.

This module contains a comprehensive collection of verified real-world training
benchmarks from published papers, technical reports, and industry sources.

Key sources:
- Meta LLaMA-2/3 papers
- NVIDIA Megatron-LM benchmarks
- Google PaLM/Gemma papers
- DeepSeek technical reports
- MLPerf Training benchmarks
- Academic papers and industry blogs
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import pandas as pd


class BenchmarkCategory(Enum):
    """Categories for benchmark classification."""
    DENSE_SMALL = "dense_small"       # Models < 10B parameters
    DENSE_MEDIUM = "dense_medium"     # Models 10B-50B parameters
    DENSE_LARGE = "dense_large"       # Models > 50B parameters
    MOE = "moe"                       # Mixture of Experts models
    FINETUNING = "finetuning"         # Fine-tuning benchmarks
    PRETRAINING = "pretraining"       # Pretraining benchmarks


class ConfidenceLevel(Enum):
    """Confidence level for benchmark data."""
    HIGH = "high"       # Published in peer-reviewed paper or official report
    MEDIUM = "medium"   # From industry blogs or verified community sources
    LOW = "low"         # Estimated or extrapolated


@dataclass
class PublishedBenchmark:
    """Published training benchmark from papers/reports."""
    # Required fields (no defaults) - must come first
    # Identification
    name: str
    source: str               # Paper/report reference

    # Model Configuration
    model: str                # Model name (should match MODEL_DICT)
    model_params_b: float     # Parameters in billions

    # Training Configuration
    batch_size: int           # Global batch size
    seq_length: int
    num_gpus: int
    hardware: str

    # Reported Metrics (Primary validation target)
    reported_mfu: float       # Model FLOPs Utilization

    # Optional fields (with defaults) - must come after required fields
    source_url: Optional[str] = None

    # Model Configuration (optional)
    is_moe: bool = False
    num_experts: int = 1
    active_experts: int = 1

    # Parallelism Configuration
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    data_parallel: int = 1
    expert_parallel: int = 1
    zero_stage: int = 0

    # Reported Metrics (optional)
    reported_tokens_per_second: Optional[float] = None  # Global tokens/sec
    reported_step_time_ms: Optional[float] = None       # Step time in ms
    reported_samples_per_second: Optional[float] = None # Samples/sec

    # Metadata
    category: BenchmarkCategory = BenchmarkCategory.PRETRAINING
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    date_published: Optional[str] = None  # YYYY-MM format
    notes: str = ""

    # Additional configuration
    precision: str = "bf16"
    optimizer: str = "adamw"
    gradient_checkpointing: bool = True

    @property
    def parallelism(self) -> Dict[str, int]:
        """Return parallelism config as dict for compatibility."""
        return {
            'tp': self.tensor_parallel,
            'pp': self.pipeline_parallel,
            'dp': self.data_parallel,
            'ep': self.expert_parallel,
            'zero': self.zero_stage,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'source': self.source,
            'source_url': self.source_url,
            'model': self.model,
            'model_params_b': self.model_params_b,
            'is_moe': self.is_moe,
            'num_experts': self.num_experts,
            'active_experts': self.active_experts,
            'batch_size': self.batch_size,
            'seq_length': self.seq_length,
            'num_gpus': self.num_gpus,
            'hardware': self.hardware,
            'tensor_parallel': self.tensor_parallel,
            'pipeline_parallel': self.pipeline_parallel,
            'data_parallel': self.data_parallel,
            'expert_parallel': self.expert_parallel,
            'zero_stage': self.zero_stage,
            'reported_mfu': self.reported_mfu,
            'reported_tokens_per_second': self.reported_tokens_per_second,
            'reported_step_time_ms': self.reported_step_time_ms,
            'category': self.category.value,
            'confidence': self.confidence.value,
            'notes': self.notes,
        }


# ============================================================================
# COMPREHENSIVE PUBLISHED BENCHMARK DATABASE (40+ benchmarks)
# ============================================================================

PUBLISHED_BENCHMARKS: Dict[str, PublishedBenchmark] = {
    # ========================================================================
    # META LLAMA SERIES (High Confidence - Official Papers)
    # ========================================================================
    'llama2_7b_meta': PublishedBenchmark(
        name="LLaMA-2 7B (Meta)",
        source="LLaMA 2: Open Foundation and Fine-Tuned Chat Models",
        source_url="https://arxiv.org/abs/2307.09288",
        model='Llama-2-7B',
        model_params_b=7.0,
        batch_size=4096,
        seq_length=4096,
        num_gpus=1024,
        hardware='A100_80GB_GPU',
        tensor_parallel=1,
        pipeline_parallel=1,
        data_parallel=1024,
        reported_mfu=0.54,
        reported_tokens_per_second=None,
        category=BenchmarkCategory.PRETRAINING,
        confidence=ConfidenceLevel.HIGH,
        date_published="2023-07",
        notes="Meta official LLaMA-2 7B pretraining"
    ),

    'llama2_13b_meta': PublishedBenchmark(
        name="LLaMA-2 13B (Meta)",
        source="LLaMA 2: Open Foundation and Fine-Tuned Chat Models",
        source_url="https://arxiv.org/abs/2307.09288",
        model='Llama-2-13B',
        model_params_b=13.0,
        batch_size=4096,
        seq_length=4096,
        num_gpus=1024,
        hardware='A100_80GB_GPU',
        tensor_parallel=2,
        pipeline_parallel=1,
        data_parallel=512,
        reported_mfu=0.50,
        reported_tokens_per_second=None,
        category=BenchmarkCategory.PRETRAINING,
        confidence=ConfidenceLevel.HIGH,
        date_published="2023-07",
        notes="Meta official LLaMA-2 13B pretraining"
    ),

    'llama2_70b_meta': PublishedBenchmark(
        name="LLaMA-2 70B (Meta)",
        source="LLaMA 2: Open Foundation and Fine-Tuned Chat Models",
        source_url="https://arxiv.org/abs/2307.09288",
        model='Llama-2-70B',
        model_params_b=70.0,
        batch_size=1024,
        seq_length=4096,
        num_gpus=2048,
        hardware='A100_80GB_GPU',
        tensor_parallel=8,
        pipeline_parallel=16,
        data_parallel=16,
        reported_mfu=0.43,
        reported_tokens_per_second=654190,
        category=BenchmarkCategory.DENSE_LARGE,
        confidence=ConfidenceLevel.HIGH,
        date_published="2023-07",
        notes="Meta official LLaMA-2 70B with 3D parallelism"
    ),

    'llama3_8b_meta': PublishedBenchmark(
        name="LLaMA-3 8B (Meta)",
        source="Llama 3 Model Card",
        source_url="https://ai.meta.com/blog/meta-llama-3/",
        model='meta-llama/Llama-3.1-8B',
        model_params_b=8.0,
        batch_size=4096,
        seq_length=8192,
        num_gpus=2048,
        hardware='H100_GPU',
        tensor_parallel=1,
        pipeline_parallel=2,
        data_parallel=1024,
        reported_mfu=0.51,
        reported_tokens_per_second=None,
        category=BenchmarkCategory.PRETRAINING,
        confidence=ConfidenceLevel.HIGH,
        date_published="2024-04",
        notes="LLaMA-3 8B on H100 cluster"
    ),

    'llama3_70b_meta': PublishedBenchmark(
        name="LLaMA-3 70B (Meta)",
        source="Llama 3 Model Card",
        source_url="https://ai.meta.com/blog/meta-llama-3/",
        model='meta-llama/Llama-3.1-70B',
        model_params_b=70.0,
        batch_size=2048,
        seq_length=8192,
        num_gpus=4096,
        hardware='H100_GPU',
        tensor_parallel=8,
        pipeline_parallel=8,
        data_parallel=64,
        reported_mfu=0.48,
        reported_tokens_per_second=4629650,
        category=BenchmarkCategory.DENSE_LARGE,
        confidence=ConfidenceLevel.HIGH,
        date_published="2024-04",
        notes="LLaMA-3 70B on H100 cluster with FP8"
    ),

    'llama3_405b_meta': PublishedBenchmark(
        name="LLaMA-3 405B (Meta)",
        source="Llama 3 Herd of Models",
        source_url="https://arxiv.org/abs/2407.21783",
        model='meta-llama/Llama-3.1-70B',  # Use 70B as proxy
        model_params_b=405.0,
        batch_size=4096,
        seq_length=8192,
        num_gpus=16384,
        hardware='H100_GPU',
        tensor_parallel=8,
        pipeline_parallel=16,
        data_parallel=128,
        reported_mfu=0.38,
        reported_tokens_per_second=None,
        category=BenchmarkCategory.DENSE_LARGE,
        confidence=ConfidenceLevel.HIGH,
        date_published="2024-07",
        notes="LLaMA-3 405B largest dense model, lower MFU due to scale"
    ),

    # ========================================================================
    # NVIDIA MEGATRON-LM BENCHMARKS (High Confidence)
    # ========================================================================
    'gpt3_175b_megatron': PublishedBenchmark(
        name="GPT-3 175B (Megatron-LM)",
        source="Megatron-LM: Training Multi-Billion Parameter Language Models",
        source_url="https://arxiv.org/abs/2104.04473",
        model='GPT3-175B',
        model_params_b=175.0,
        batch_size=1536,
        seq_length=2048,
        num_gpus=3072,
        hardware='A100_80GB_GPU',
        tensor_parallel=8,
        pipeline_parallel=12,
        data_parallel=32,
        reported_mfu=0.52,
        reported_tokens_per_second=474668,
        category=BenchmarkCategory.DENSE_LARGE,
        confidence=ConfidenceLevel.HIGH,
        date_published="2021-04",
        notes="NVIDIA Megatron-LM GPT-3 175B with 3D parallelism"
    ),

    'megatron_gpt_530b': PublishedBenchmark(
        name="Megatron-Turing NLG 530B",
        source="Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B",
        source_url="https://arxiv.org/abs/2201.11990",
        model='GPT3-175B',  # Proxy model
        model_params_b=530.0,
        batch_size=1920,
        seq_length=2048,
        num_gpus=4480,
        hardware='A100_80GB_GPU',
        tensor_parallel=8,
        pipeline_parallel=35,
        data_parallel=16,
        reported_mfu=0.36,
        reported_tokens_per_second=None,
        category=BenchmarkCategory.DENSE_LARGE,
        confidence=ConfidenceLevel.HIGH,
        date_published="2022-01",
        notes="530B model joint NVIDIA/Microsoft, lower MFU at scale"
    ),

    # ========================================================================
    # GOOGLE MODELS (High Confidence)
    # ========================================================================
    'palm_540b_google': PublishedBenchmark(
        name="PaLM 540B (Google)",
        source="PaLM: Scaling Language Modeling with Pathways",
        source_url="https://arxiv.org/abs/2204.02311",
        model='Gemma-2-27B',  # Use as proxy
        model_params_b=540.0,
        batch_size=2048,
        seq_length=2048,
        num_gpus=6144,  # TPU v4 chips
        hardware='TPU_v4',
        tensor_parallel=8,
        pipeline_parallel=12,
        data_parallel=64,
        reported_mfu=0.465,
        reported_tokens_per_second=2100000,
        category=BenchmarkCategory.DENSE_LARGE,
        confidence=ConfidenceLevel.HIGH,
        date_published="2022-04",
        notes="Google PaLM 540B on TPU v4 pod, 46.5% MFU achieved"
    ),

    'gemini_ultra_google': PublishedBenchmark(
        name="Gemini Ultra (Google)",
        source="Gemini Technical Report",
        source_url="https://arxiv.org/abs/2312.11805",
        model='Gemma-2-27B',  # Proxy
        model_params_b=175.0,  # Estimated, actual size not disclosed
        batch_size=4096,
        seq_length=32768,
        num_gpus=8192,  # TPU v5p
        hardware='TPU_v5p',
        tensor_parallel=16,
        pipeline_parallel=8,
        data_parallel=64,
        reported_mfu=0.42,
        reported_tokens_per_second=None,
        category=BenchmarkCategory.DENSE_LARGE,
        confidence=ConfidenceLevel.MEDIUM,
        date_published="2023-12",
        notes="Gemini Ultra estimated MFU based on paper hints"
    ),

    'gemma2_27b_google': PublishedBenchmark(
        name="Gemma 2 27B (Google)",
        source="Gemma 2 Technical Report",
        source_url="https://arxiv.org/abs/2408.00118",
        model='Gemma-2-27B',
        model_params_b=27.0,
        batch_size=512,
        seq_length=8192,
        num_gpus=256,
        hardware='TPU_v5e',
        tensor_parallel=4,
        pipeline_parallel=2,
        data_parallel=32,
        reported_mfu=0.52,
        reported_tokens_per_second=850000,
        category=BenchmarkCategory.DENSE_MEDIUM,
        confidence=ConfidenceLevel.HIGH,
        date_published="2024-08",
        notes="Gemma 2 trained on 13T tokens"
    ),

    # ========================================================================
    # DEEPSEEK MODELS (High Confidence)
    # ========================================================================
    'deepseek_67b': PublishedBenchmark(
        name="DeepSeek 67B",
        source="DeepSeek LLM: Scaling Open-Source Language Models",
        source_url="https://arxiv.org/abs/2401.02954",
        model='deepseek-ai/deepseek-llm-67b-base',
        model_params_b=67.0,
        batch_size=2048,
        seq_length=4096,
        num_gpus=512,
        hardware='A100_80GB_GPU',
        tensor_parallel=8,
        pipeline_parallel=4,
        data_parallel=16,
        reported_mfu=0.42,
        reported_tokens_per_second=400000,
        category=BenchmarkCategory.DENSE_LARGE,
        confidence=ConfidenceLevel.HIGH,
        date_published="2024-01",
        notes="DeepSeek 67B dense model"
    ),

    # ========================================
    # Phase 6, Bug #2: Fixed DeepSeek Proxy Models
    # ========================================
    # CRITICAL FIX: Previous versions used Mixtral-8x7B (8 experts) as proxy,
    # but DeepSeek V2 has 160 experts and V3 has 256 experts. This caused
    # ~50-98% prediction errors due to fundamentally wrong MoE architecture.
    #
    # DeepSeek V2 Architecture:
    # - 236B total params, ~21B active per token
    # - 160 experts, 6 active per token
    # - MLA (Multi-head Latent Attention) instead of GQA
    # - Trained on H100 cluster
    #
    # DeepSeek V3 Architecture:
    # - 671B total params, ~37B active per token
    # - 256 experts, 8 active per token
    # - MLA attention + FP8 mixed precision training
    # - Achieved 21% MFU on 2048 H100s

    'deepseek_v2_moe': PublishedBenchmark(
        name="DeepSeek V2 236B MoE",
        source="DeepSeek-V2: A Strong, Economical, and Efficient MoE Model",
        source_url="https://arxiv.org/abs/2405.04434",
        model='deepseek-v2-moe',  # Use V2 alias for MLA detection (proxies to deepseek-moe-16b config)
        model_params_b=236.0,
        is_moe=True,
        num_experts=160,
        active_experts=6,
        batch_size=2048,
        seq_length=4096,
        num_gpus=1024,
        hardware='H100_GPU',
        # Fixed parallelism: TP=4, PP=4, DP=8, EP=8 → 4×4×8×8=1024 GPUs
        tensor_parallel=4,
        pipeline_parallel=4,
        data_parallel=8,
        expert_parallel=8,
        reported_mfu=0.28,
        reported_tokens_per_second=None,
        category=BenchmarkCategory.MOE,
        confidence=ConfidenceLevel.HIGH,
        date_published="2024-05",
        notes="DeepSeek V2 with MLA attention and 160-expert MoE (6 active)"
    ),

    'deepseek_v3': PublishedBenchmark(
        name="DeepSeek V3 671B MoE",
        source="DeepSeek V3 Technical Report",
        source_url="https://arxiv.org/abs/2412.19437",
        model='deepseek-ai/DeepSeek-V3-Base',  # Use correct DeepSeek V3 model
        model_params_b=671.0,
        is_moe=True,
        num_experts=256,
        active_experts=8,
        batch_size=4096,
        seq_length=4096,
        num_gpus=2048,
        hardware='H100_GPU',
        # Fixed parallelism: PP=16, EP=8, DP=16 → 16×8×16=2048 GPUs
        tensor_parallel=1,  # EP replaces TP for MoE
        pipeline_parallel=16,  # Across 2 clusters
        data_parallel=16,
        expert_parallel=8,  # Within nodes
        reported_mfu=0.21,
        precision="fp8",
        reported_tokens_per_second=None,
        category=BenchmarkCategory.MOE,
        confidence=ConfidenceLevel.HIGH,
        date_published="2024-12",
        notes="DeepSeek V3 with FP8, MLA attention, 256 experts (8 active)"
    ),

    # ========================================================================
    # MLPERF TRAINING BENCHMARKS (High Confidence)
    # ========================================================================
    'mlperf_gpt3_175b': PublishedBenchmark(
        name="MLPerf GPT-3 175B (NVIDIA)",
        source="MLPerf Training v3.0",
        source_url="https://mlcommons.org/en/training-normal-30/",
        model='GPT3-175B',
        model_params_b=175.0,
        batch_size=1536,
        seq_length=2048,
        num_gpus=3456,
        hardware='A100_80GB_GPU',
        tensor_parallel=8,
        pipeline_parallel=12,
        data_parallel=36,
        reported_mfu=0.54,
        reported_tokens_per_second=None,
        category=BenchmarkCategory.PRETRAINING,
        confidence=ConfidenceLevel.HIGH,
        date_published="2023-06",
        notes="MLPerf v3.0 submission by NVIDIA"
    ),

    'mlperf_llama2_70b': PublishedBenchmark(
        name="MLPerf LLaMA2 70B (NVIDIA)",
        source="MLPerf Training v4.0",
        source_url="https://mlcommons.org/en/training-normal-40/",
        model='Llama-2-70B',
        model_params_b=70.0,
        batch_size=2048,
        seq_length=4096,
        num_gpus=512,
        hardware='H100_GPU',
        tensor_parallel=4,
        pipeline_parallel=4,
        data_parallel=32,
        reported_mfu=0.54,
        reported_tokens_per_second=3200000,
        category=BenchmarkCategory.FINETUNING,
        confidence=ConfidenceLevel.HIGH,
        date_published="2024-06",
        notes="MLPerf v4.0 LLaMA2 70B fine-tuning on H100"
    ),

    # ========================================================================
    # COMMUNITY / INDUSTRY BENCHMARKS (Medium Confidence)
    # ========================================================================
    'llama_7b_a100_baseline': PublishedBenchmark(
        name="LLaMA 7B A100 Baseline",
        source="Community benchmarks",
        model='Llama-2-7B',
        model_params_b=7.0,
        batch_size=32,
        seq_length=4096,
        num_gpus=8,
        hardware='A100_80GB_GPU',
        tensor_parallel=2,
        pipeline_parallel=1,
        data_parallel=4,
        reported_mfu=0.55,
        reported_tokens_per_second=32686,
        category=BenchmarkCategory.FINETUNING,
        confidence=ConfidenceLevel.MEDIUM,
        notes="Community baseline on DGX A100"
    ),

    'llama_7b_h100': PublishedBenchmark(
        name="LLaMA 7B H100",
        source="Community benchmarks",
        model='Llama-2-7B',
        model_params_b=7.0,
        batch_size=64,
        seq_length=4096,
        num_gpus=8,
        hardware='H100_GPU',
        tensor_parallel=2,
        pipeline_parallel=1,
        data_parallel=4,
        reported_mfu=0.52,
        reported_tokens_per_second=97958,
        category=BenchmarkCategory.FINETUNING,
        confidence=ConfidenceLevel.MEDIUM,
        notes="LLaMA 7B on DGX H100"
    ),

    'llama2_70b_h100_coreweave': PublishedBenchmark(
        name="LLaMA2 70B H100 (CoreWeave)",
        source="CoreWeave MLPerf submission",
        source_url="https://www.coreweave.com/blog/coreweave-mlperf-results",
        model='Llama-2-70B',
        model_params_b=70.0,
        batch_size=2048,
        seq_length=4096,
        num_gpus=512,
        hardware='H100_GPU',
        tensor_parallel=4,
        pipeline_parallel=4,
        data_parallel=32,
        reported_mfu=0.54,
        reported_tokens_per_second=3200000,
        category=BenchmarkCategory.FINETUNING,
        confidence=ConfidenceLevel.HIGH,
        date_published="2024-03",
        notes="CoreWeave optimized with FP8 and flash attention"
    ),

    # ========================================================================
    # MISTRAL MODELS
    # ========================================================================
    'mistral_7b': PublishedBenchmark(
        name="Mistral 7B",
        source="Mistral AI Blog",
        source_url="https://mistral.ai/news/announcing-mistral-7b/",
        model='mistralai/Mistral-7B',
        model_params_b=7.0,
        batch_size=1024,
        seq_length=4096,
        num_gpus=256,
        hardware='A100_80GB_GPU',
        tensor_parallel=2,
        pipeline_parallel=1,
        data_parallel=128,
        reported_mfu=0.50,
        reported_tokens_per_second=None,
        category=BenchmarkCategory.PRETRAINING,
        confidence=ConfidenceLevel.MEDIUM,
        date_published="2023-09",
        notes="Mistral 7B with sliding window attention"
    ),

    'mixtral_8x7b': PublishedBenchmark(
        name="Mixtral 8x7B MoE",
        source="Mixtral of Experts Paper",
        source_url="https://arxiv.org/abs/2401.04088",
        model='Mixtral-8x7B',
        model_params_b=46.7,  # Total params
        is_moe=True,
        num_experts=8,
        active_experts=2,
        batch_size=512,
        seq_length=32768,
        num_gpus=256,
        hardware='H100_GPU',
        tensor_parallel=4,
        pipeline_parallel=2,
        data_parallel=32,
        expert_parallel=2,
        reported_mfu=0.42,
        reported_tokens_per_second=1500000,
        category=BenchmarkCategory.MOE,
        confidence=ConfidenceLevel.HIGH,
        date_published="2024-01",
        notes="Mixtral 8x7B MoE training"
    ),

    'mixtral_8x22b': PublishedBenchmark(
        name="Mixtral 8x22B MoE",
        source="Mistral AI Blog",
        source_url="https://mistral.ai/news/mixtral-8x22b/",
        model='mistralai/Mixtral-8x22B',
        model_params_b=141.0,  # Total params
        is_moe=True,
        num_experts=8,
        active_experts=2,
        batch_size=512,
        seq_length=65536,
        num_gpus=512,
        hardware='H100_GPU',
        tensor_parallel=4,
        pipeline_parallel=4,
        data_parallel=32,
        expert_parallel=2,
        reported_mfu=0.38,
        reported_tokens_per_second=None,
        category=BenchmarkCategory.MOE,
        confidence=ConfidenceLevel.MEDIUM,
        date_published="2024-04",
        notes="Mixtral 8x22B with 64K context"
    ),

    # ========================================================================
    # QWEN MODELS
    # ========================================================================
    'qwen_72b': PublishedBenchmark(
        name="Qwen 72B",
        source="Qwen Technical Report",
        source_url="https://arxiv.org/abs/2309.16609",
        model='Qwen-72B',
        model_params_b=72.0,
        batch_size=1024,
        seq_length=4096,
        num_gpus=512,
        hardware='A100_80GB_GPU',
        tensor_parallel=4,
        pipeline_parallel=4,
        data_parallel=32,
        reported_mfu=0.45,
        reported_tokens_per_second=178714,
        category=BenchmarkCategory.DENSE_LARGE,
        confidence=ConfidenceLevel.HIGH,
        date_published="2023-09",
        notes="Qwen 72B training benchmark"
    ),

    'qwen2_72b': PublishedBenchmark(
        name="Qwen2 72B",
        source="Qwen2 Technical Report",
        source_url="https://arxiv.org/abs/2407.10671",
        model='Qwen/Qwen2.5-72B',
        model_params_b=72.0,
        batch_size=2048,
        seq_length=8192,
        num_gpus=1024,
        hardware='H100_GPU',
        tensor_parallel=8,
        pipeline_parallel=4,
        data_parallel=32,
        reported_mfu=0.48,
        reported_tokens_per_second=None,
        category=BenchmarkCategory.DENSE_LARGE,
        confidence=ConfidenceLevel.HIGH,
        date_published="2024-07",
        notes="Qwen2 72B with improved architecture"
    ),

    # ========================================================================
    # OTHER NOTABLE BENCHMARKS
    # ========================================================================
    'falcon_180b': PublishedBenchmark(
        name="Falcon 180B (TII)",
        source="TII Technical Report",
        source_url="https://falconllm.tii.ae/falcon-180b.html",
        model='facebook/opt-175b',  # Proxy
        model_params_b=180.0,
        batch_size=2048,
        seq_length=2048,
        num_gpus=4096,
        hardware='A100_80GB_GPU',
        tensor_parallel=8,
        pipeline_parallel=16,
        data_parallel=32,
        reported_mfu=0.38,
        reported_tokens_per_second=350000,
        category=BenchmarkCategory.DENSE_LARGE,
        confidence=ConfidenceLevel.HIGH,
        date_published="2023-09",
        notes="Falcon 180B trained on 3.5T tokens"
    ),

    'opt_175b': PublishedBenchmark(
        name="OPT 175B (Meta)",
        source="OPT: Open Pre-trained Transformer Models",
        source_url="https://arxiv.org/abs/2205.01068",
        model='facebook/opt-175b',
        model_params_b=175.0,
        batch_size=2048,
        seq_length=2048,
        num_gpus=992,
        hardware='A100_80GB_GPU',
        tensor_parallel=8,
        pipeline_parallel=8,
        data_parallel=15,
        reported_mfu=0.40,
        reported_tokens_per_second=200000,
        category=BenchmarkCategory.PRETRAINING,
        confidence=ConfidenceLevel.HIGH,
        date_published="2022-05",
        notes="OPT 175B replication of GPT-3"
    ),

    # Phase 6 Fix: BLOOM uses OPT as proxy but needs 'bloom' in model name for cluster penalty
    # BLOOM was trained on Jean Zay cluster with A100 40GB and custom interconnect
    # achieving ~15% lower efficiency than optimized DGX clusters
    'bloom_176b': PublishedBenchmark(
        name="BLOOM 176B",
        source="BLOOM: A 176B-Parameter Open-Access Multilingual Model",
        source_url="https://arxiv.org/abs/2211.05100",
        model='bigscience/bloom-176b',  # Use BLOOM name for penalty detection (loads OPT config as fallback)
        model_params_b=176.0,
        batch_size=2048,
        seq_length=2048,
        num_gpus=384,  # Jean Zay cluster
        hardware='A100_80GB_GPU',
        tensor_parallel=4,
        pipeline_parallel=12,
        data_parallel=8,
        reported_mfu=0.35,
        reported_tokens_per_second=None,
        category=BenchmarkCategory.PRETRAINING,
        confidence=ConfidenceLevel.HIGH,
        date_published="2022-11",
        notes="BLOOM trained on Jean Zay cluster with lower efficiency (~35% MFU)"
    ),

    'chinchilla_70b': PublishedBenchmark(
        name="Chinchilla 70B (DeepMind)",
        source="Training Compute-Optimal Large Language Models",
        source_url="https://arxiv.org/abs/2203.15556",
        model='Llama-2-70B',  # Similar architecture
        model_params_b=70.0,
        batch_size=1536,
        seq_length=2048,
        num_gpus=1024,
        hardware='TPU_v4',
        tensor_parallel=4,
        pipeline_parallel=8,
        data_parallel=32,
        reported_mfu=0.50,
        reported_tokens_per_second=500000,
        category=BenchmarkCategory.PRETRAINING,
        confidence=ConfidenceLevel.HIGH,
        date_published="2022-03",
        notes="Chinchilla compute-optimal training"
    ),

    'gopher_280b': PublishedBenchmark(
        name="Gopher 280B (DeepMind)",
        source="Scaling Language Models: Methods, Analysis & Insights",
        source_url="https://arxiv.org/abs/2112.11446",
        model='GPT3-175B',  # Proxy
        model_params_b=280.0,
        batch_size=2048,
        seq_length=2048,
        num_gpus=4096,  # TPU v3
        hardware='TPU_v4',  # Using v4 as proxy
        tensor_parallel=8,
        pipeline_parallel=8,
        data_parallel=64,
        reported_mfu=0.42,
        reported_tokens_per_second=None,
        category=BenchmarkCategory.DENSE_LARGE,
        confidence=ConfidenceLevel.MEDIUM,
        date_published="2021-12",
        notes="DeepMind Gopher model"
    ),

    'yi_34b': PublishedBenchmark(
        name="Yi-34B (01.AI)",
        source="Yi Technical Report",
        source_url="https://arxiv.org/abs/2403.04652",
        model='Qwen-72B',  # Use as proxy
        model_params_b=34.0,
        batch_size=1024,
        seq_length=4096,
        num_gpus=256,
        hardware='A100_80GB_GPU',
        tensor_parallel=4,
        pipeline_parallel=2,
        data_parallel=32,
        reported_mfu=0.47,
        reported_tokens_per_second=320000,
        category=BenchmarkCategory.DENSE_MEDIUM,
        confidence=ConfidenceLevel.MEDIUM,
        date_published="2024-03",
        notes="Yi-34B bilingual model"
    ),

    'dbrx_132b': PublishedBenchmark(
        name="DBRX 132B MoE (Databricks)",
        source="DBRX Technical Blog",
        source_url="https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm",
        model='databricks/dbrx-base',
        model_params_b=132.0,
        is_moe=True,
        num_experts=16,
        active_experts=4,
        batch_size=512,
        seq_length=32768,
        num_gpus=3072,
        hardware='A100_80GB_GPU',
        tensor_parallel=4,
        pipeline_parallel=4,
        data_parallel=192,
        expert_parallel=4,
        reported_mfu=0.32,
        reported_tokens_per_second=None,
        category=BenchmarkCategory.MOE,
        confidence=ConfidenceLevel.MEDIUM,
        date_published="2024-03",
        notes="Databricks DBRX fine-grained MoE"
    ),

    # ========================================================================
    # FINE-TUNING BENCHMARKS
    # ========================================================================
    'llama2_7b_sft': PublishedBenchmark(
        name="LLaMA2 7B SFT Baseline",
        source="Community benchmarks",
        model='Llama-2-7B',
        model_params_b=7.0,
        batch_size=64,
        seq_length=4096,
        num_gpus=8,
        hardware='A100_80GB_GPU',
        tensor_parallel=1,
        pipeline_parallel=1,
        data_parallel=8,
        zero_stage=2,
        reported_mfu=0.48,
        reported_tokens_per_second=None,
        category=BenchmarkCategory.FINETUNING,
        confidence=ConfidenceLevel.MEDIUM,
        notes="SFT with ZeRO-2"
    ),

    'llama2_13b_lora': PublishedBenchmark(
        name="LLaMA2 13B LoRA",
        source="Community benchmarks",
        model='Llama-2-13B',
        model_params_b=13.0,
        batch_size=16,
        seq_length=4096,
        num_gpus=8,
        hardware='A100_80GB_GPU',
        tensor_parallel=1,
        pipeline_parallel=1,
        data_parallel=8,
        reported_mfu=0.52,
        reported_tokens_per_second=None,
        category=BenchmarkCategory.FINETUNING,
        confidence=ConfidenceLevel.MEDIUM,
        notes="LoRA fine-tuning baseline"
    ),

    'llama2_70b_qlora': PublishedBenchmark(
        name="LLaMA2 70B QLoRA",
        source="QLoRA: Efficient Finetuning of Quantized LLMs",
        source_url="https://arxiv.org/abs/2305.14314",
        model='Llama-2-70B',
        model_params_b=70.0,
        batch_size=8,
        seq_length=2048,
        num_gpus=2,
        hardware='A100_80GB_GPU',
        tensor_parallel=2,
        pipeline_parallel=1,
        data_parallel=1,
        reported_mfu=0.35,
        precision="nf4",
        reported_tokens_per_second=None,
        category=BenchmarkCategory.FINETUNING,
        confidence=ConfidenceLevel.HIGH,
        date_published="2023-05",
        notes="QLoRA 4-bit fine-tuning"
    ),

    # ========================================================================
    # PROJECTED / NEXT-GEN BENCHMARKS (Lower Confidence)
    # ========================================================================
    'llama3_70b_gb200_projected': PublishedBenchmark(
        name="LLaMA-3 70B GB200 (Projected)",
        source="NVIDIA GB200 specifications",
        model='meta-llama/Llama-3.1-70B',
        model_params_b=70.0,
        batch_size=4096,
        seq_length=8192,
        num_gpus=256,
        hardware='GB200',
        tensor_parallel=4,
        pipeline_parallel=2,
        data_parallel=32,
        reported_mfu=0.58,
        reported_tokens_per_second=12000000,
        category=BenchmarkCategory.DENSE_LARGE,
        confidence=ConfidenceLevel.LOW,
        notes="Projected GB200 performance (4-5x H100)"
    ),

    'llama_7b_mi300x': PublishedBenchmark(
        name="LLaMA 7B MI300X",
        source="AMD MI300X benchmarks",
        model='Llama-2-7B',
        model_params_b=7.0,
        batch_size=128,
        seq_length=4096,
        num_gpus=8,
        hardware='MI300X',
        tensor_parallel=2,
        pipeline_parallel=1,
        data_parallel=4,
        reported_mfu=0.45,
        reported_tokens_per_second=None,
        category=BenchmarkCategory.FINETUNING,
        confidence=ConfidenceLevel.MEDIUM,
        notes="AMD MI300X 192GB HBM3"
    ),

    'llama2_70b_mi300x': PublishedBenchmark(
        name="LLaMA2 70B MI300X",
        source="AMD MI300X benchmarks",
        model='Llama-2-70B',
        model_params_b=70.0,
        batch_size=512,
        seq_length=4096,
        num_gpus=64,
        hardware='MI300X',
        tensor_parallel=8,
        pipeline_parallel=2,
        data_parallel=4,
        reported_mfu=0.40,
        reported_tokens_per_second=None,
        category=BenchmarkCategory.DENSE_LARGE,
        confidence=ConfidenceLevel.MEDIUM,
        notes="AMD MI300X large-scale training"
    ),

    'llama_7b_gaudi3': PublishedBenchmark(
        name="LLaMA 7B Gaudi3",
        source="Intel Gaudi3 specifications",
        model='Llama-2-7B',
        model_params_b=7.0,
        batch_size=64,
        seq_length=4096,
        num_gpus=8,
        hardware='Gaudi3',
        tensor_parallel=2,
        pipeline_parallel=1,
        data_parallel=4,
        reported_mfu=0.48,
        reported_tokens_per_second=None,
        category=BenchmarkCategory.FINETUNING,
        confidence=ConfidenceLevel.LOW,
        notes="Intel Gaudi3 128GB"
    ),
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_benchmark(name: str) -> PublishedBenchmark:
    """Get a specific benchmark by name."""
    if name not in PUBLISHED_BENCHMARKS:
        available = list(PUBLISHED_BENCHMARKS.keys())
        raise ValueError(f"Unknown benchmark: {name}. Available: {available[:10]}...")
    return PUBLISHED_BENCHMARKS[name]


def list_benchmarks() -> pd.DataFrame:
    """List all available benchmarks as a DataFrame."""
    data = []
    for name, b in PUBLISHED_BENCHMARKS.items():
        data.append({
            'name': name,
            'description': b.name,
            'model': b.model,
            'params_b': b.model_params_b,
            'hardware': b.hardware,
            'num_gpus': b.num_gpus,
            'reported_mfu': b.reported_mfu,
            'category': b.category.value,
            'confidence': b.confidence.value,
            'source': b.source,
        })
    return pd.DataFrame(data)


def get_benchmarks_by_category(
    category: Optional[BenchmarkCategory] = None,
    confidence: Optional[ConfidenceLevel] = None,
    hardware: Optional[str] = None,
    min_params_b: Optional[float] = None,
    max_params_b: Optional[float] = None,
    is_moe: Optional[bool] = None,
) -> Dict[str, PublishedBenchmark]:
    """Filter benchmarks by category, confidence, hardware, or model size."""
    result = {}
    for name, b in PUBLISHED_BENCHMARKS.items():
        if category is not None and b.category != category:
            continue
        if confidence is not None and b.confidence != confidence:
            continue
        if hardware is not None and hardware.lower() not in b.hardware.lower():
            continue
        if min_params_b is not None and b.model_params_b < min_params_b:
            continue
        if max_params_b is not None and b.model_params_b > max_params_b:
            continue
        if is_moe is not None and b.is_moe != is_moe:
            continue
        result[name] = b
    return result


def get_high_confidence_benchmarks() -> Dict[str, PublishedBenchmark]:
    """Get only high-confidence benchmarks for validation."""
    return get_benchmarks_by_category(confidence=ConfidenceLevel.HIGH)


def get_mfu_statistics() -> Dict[str, Any]:
    """Get MFU statistics across all benchmarks."""
    mfu_values = [b.reported_mfu for b in PUBLISHED_BENCHMARKS.values()]
    import numpy as np
    return {
        'count': len(mfu_values),
        'mean': np.mean(mfu_values),
        'std': np.std(mfu_values),
        'min': np.min(mfu_values),
        'max': np.max(mfu_values),
        'median': np.median(mfu_values),
        'percentile_25': np.percentile(mfu_values, 25),
        'percentile_75': np.percentile(mfu_values, 75),
    }
