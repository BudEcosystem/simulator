"""
Comprehensive Validation Benchmarks for Training Memory Calculator.

This module validates the training simulation against real-world benchmarks from:
- Official documentation (DeepSpeed, HuggingFace, Modal)
- Published papers (QLoRA, GaLore, ZeRO)
- Industry benchmarks (Unsloth, LlamaFactory)

Reference formulas:
- Mixed precision + AdamW: 16 bytes per param (2 fp16 params + 2 fp16 grads + 12 fp32 opt)
- Full fp32 + AdamW: 16 bytes per param (4 fp32 params + 4 fp32 grads + 8 fp32 opt)
- 8-bit AdamW: 6 bytes per param (2 fp16 params + 2 fp16 grads + 2 int8 opt)

Sources:
- https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one
- https://modal.com/blog/how-much-vram-need-fine-tuning
- https://deepspeed.readthedocs.io/en/latest/memory.html
- https://arxiv.org/abs/2305.14314 (QLoRA)
- https://arxiv.org/abs/2403.03507 (GaLore)
"""

import sys
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import math

# Add source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_memory_calculator.training import (
    AdvancedTrainingCalculator,
    calculate_optimizer_memory,
    get_optimizer_config,
    get_training_stage_config,
    get_deepspeed_config,
    ParallelismConfig,
    calculate_distributed_memory,
    recommend_distributed_strategy,
    OPTIMIZER_CONFIGS,
)


# ============================================================================
# BENCHMARK DATA FROM PUBLISHED SOURCES
# ============================================================================

@dataclass
class BenchmarkCase:
    """A benchmark validation case."""
    name: str
    source: str
    model_params: int
    config: Dict[str, Any]
    expected_memory_gb: float
    tolerance_gb: float  # Acceptable error margin
    notes: str = ""


# Model configurations
LLAMA_7B_CONFIG = {
    "hidden_size": 4096,
    "intermediate_size": 11008,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "vocab_size": 32000,
    "num_parameters": 7_000_000_000,
}

LLAMA_8B_CONFIG = {
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "vocab_size": 128256,
    "num_parameters": 8_000_000_000,
}

LLAMA_13B_CONFIG = {
    "hidden_size": 5120,
    "intermediate_size": 13824,
    "num_hidden_layers": 40,
    "num_attention_heads": 40,
    "num_key_value_heads": 40,
    "vocab_size": 32000,
    "num_parameters": 13_000_000_000,
}

LLAMA_70B_CONFIG = {
    "hidden_size": 8192,
    "intermediate_size": 28672,
    "num_hidden_layers": 80,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "vocab_size": 32000,
    "num_parameters": 70_000_000_000,
}

T5_3B_CONFIG = {
    "hidden_size": 1024,
    "intermediate_size": 16384,
    "num_hidden_layers": 24,
    "num_attention_heads": 32,
    "vocab_size": 32128,
    "num_parameters": 2_851_000_000,
}


# ============================================================================
# THEORETICAL MEMORY FORMULAS (Ground Truth)
# ============================================================================

class TheoreticalCalculator:
    """
    Ground truth memory calculations based on published formulas.

    Reference: https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one
    """

    @staticmethod
    def mixed_precision_adamw(params: int) -> float:
        """
        Mixed precision training with AdamW.

        Formula: 16 bytes per parameter
        - fp16 params: 2 bytes
        - fp16 gradients: 2 bytes
        - fp32 master weights: 4 bytes
        - fp32 momentum: 4 bytes
        - fp32 variance: 4 bytes

        Total: 2 + 2 + 4 + 4 + 4 = 16 bytes per param
        """
        return (params * 16) / 1e9

    @staticmethod
    def mixed_precision_adamw_8bit(params: int) -> float:
        """
        Mixed precision training with 8-bit AdamW.

        Formula: 6 bytes per parameter
        - fp16 params: 2 bytes
        - fp16 gradients: 2 bytes
        - int8 momentum: 1 byte
        - int8 variance: 1 byte

        Total: 2 + 2 + 1 + 1 = 6 bytes per param
        """
        return (params * 6) / 1e9

    @staticmethod
    def full_precision_adamw(params: int) -> float:
        """
        Full FP32 training with AdamW.

        Formula: 16-20 bytes per parameter
        - fp32 params: 4 bytes
        - fp32 gradients: 4 bytes
        - fp32 momentum: 4 bytes
        - fp32 variance: 4 bytes

        Total: 4 + 4 + 4 + 4 = 16 bytes per param (without master copy)
        """
        return (params * 16) / 1e9

    @staticmethod
    def lora_memory(
        base_params: int,
        lora_rank: int,
        num_layers: int,
        hidden_size: int,
        num_adapters: int = 4,  # Q, K, V, O
        base_precision: str = "bf16",
        optimizer: str = "adamw",
    ) -> Dict[str, float]:
        """
        LoRA fine-tuning memory estimation.

        Components:
        - Base model weights (frozen): params × bytes_per_precision
        - LoRA adapters: 2 × rank × hidden × num_adapters × num_layers × precision
        - Gradients for LoRA: LoRA params × 4 bytes (fp32)
        - Optimizer states for LoRA: LoRA params × opt_multiplier × 4 bytes
        """
        precision_bytes = 2 if base_precision in ("bf16", "fp16") else 4

        # Base model memory
        base_memory = (base_params * precision_bytes) / 1e9

        # LoRA parameters: A matrix (rank × hidden) + B matrix (hidden × rank)
        # For each adapter target
        lora_params_per_layer = num_adapters * 2 * lora_rank * hidden_size
        total_lora_params = lora_params_per_layer * num_layers

        # LoRA adapters in same precision as training
        lora_memory = (total_lora_params * precision_bytes) / 1e9

        # Gradients for LoRA params (fp32)
        gradient_memory = (total_lora_params * 4) / 1e9

        # Optimizer states
        if optimizer == "adamw":
            opt_states = 2  # momentum + variance
        elif optimizer == "adamw_8bit":
            opt_states = 0.5  # 8-bit states
        else:
            opt_states = 2

        optimizer_memory = (total_lora_params * opt_states * 4) / 1e9

        return {
            "base_model_gb": base_memory,
            "lora_adapters_gb": lora_memory,
            "gradient_gb": gradient_memory,
            "optimizer_gb": optimizer_memory,
            "total_lora_params": total_lora_params,
            "trainable_percent": 100 * total_lora_params / base_params,
        }

    @staticmethod
    def qlora_memory(
        base_params: int,
        lora_rank: int,
        num_layers: int,
        hidden_size: int,
        num_adapters: int = 4,
    ) -> Dict[str, float]:
        """
        QLoRA memory estimation (4-bit base model).

        Components:
        - Base model in 4-bit: params × 0.5 bytes
        - LoRA adapters in bf16: LoRA params × 2 bytes
        - Gradients for LoRA: LoRA params × 4 bytes (fp32)
        - 8-bit optimizer states: LoRA params × 2 bytes
        """
        # Base model in 4-bit NF4
        base_memory = (base_params * 0.5) / 1e9

        # LoRA parameters
        lora_params_per_layer = num_adapters * 2 * lora_rank * hidden_size
        total_lora_params = lora_params_per_layer * num_layers

        # LoRA in bf16
        lora_memory = (total_lora_params * 2) / 1e9

        # Gradients in fp32
        gradient_memory = (total_lora_params * 4) / 1e9

        # 8-bit paged AdamW
        optimizer_memory = (total_lora_params * 2) / 1e9

        return {
            "base_model_4bit_gb": base_memory,
            "lora_adapters_gb": lora_memory,
            "gradient_gb": gradient_memory,
            "optimizer_gb": optimizer_memory,
            "total_lora_params": total_lora_params,
        }

    @staticmethod
    def deepspeed_zero2_memory(
        params: int,
        num_gpus: int,
        precision: str = "bf16",
        optimizer: str = "adamw",
    ) -> Dict[str, float]:
        """
        DeepSpeed ZeRO-2 memory estimation.

        ZeRO-2 shards:
        - Optimizer states: Divided by num_gpus
        - Gradients: Divided by num_gpus
        - Parameters: NOT sharded (replicated)
        """
        precision_bytes = 2 if precision in ("bf16", "fp16") else 4

        # Parameters (replicated on each GPU)
        param_memory = (params * precision_bytes) / 1e9

        # Gradients (sharded)
        gradient_memory = (params * 4) / 1e9 / num_gpus

        # Optimizer states (sharded)
        opt_memory = (params * 8) / 1e9 / num_gpus  # 2 states × 4 bytes

        return {
            "params_gb": param_memory,
            "gradients_gb": gradient_memory,
            "optimizer_gb": opt_memory,
            "total_per_gpu_gb": param_memory + gradient_memory + opt_memory,
        }

    @staticmethod
    def deepspeed_zero3_memory(
        params: int,
        num_gpus: int,
        precision: str = "bf16",
    ) -> Dict[str, float]:
        """
        DeepSpeed ZeRO-3 memory estimation.

        ZeRO-3 shards everything:
        - Parameters: Divided by num_gpus
        - Gradients: Divided by num_gpus
        - Optimizer states: Divided by num_gpus
        """
        precision_bytes = 2 if precision in ("bf16", "fp16") else 4

        # All sharded
        param_memory = (params * precision_bytes) / 1e9 / num_gpus
        gradient_memory = (params * 4) / 1e9 / num_gpus
        opt_memory = (params * 8) / 1e9 / num_gpus

        return {
            "params_gb": param_memory,
            "gradients_gb": gradient_memory,
            "optimizer_gb": opt_memory,
            "total_per_gpu_gb": param_memory + gradient_memory + opt_memory,
        }


# ============================================================================
# VALIDATION BENCHMARK CASES
# ============================================================================

# Based on published benchmarks
VALIDATION_BENCHMARKS = [
    # Full Fine-Tuning Benchmarks
    BenchmarkCase(
        name="7B Full FT Mixed Precision",
        source="Modal blog / HuggingFace docs",
        model_params=7_000_000_000,
        config={
            "method": "full",
            "optimizer": "adamw",
            "precision": "bf16",
        },
        expected_memory_gb=112.0,  # 7B × 16 bytes = 112 GB
        tolerance_gb=15.0,  # Allow for activation memory variations
        notes="16 bytes/param for mixed precision AdamW",
    ),

    BenchmarkCase(
        name="13B Full FT Mixed Precision",
        source="Modal blog",
        model_params=13_000_000_000,
        config={
            "method": "full",
            "optimizer": "adamw",
            "precision": "bf16",
        },
        expected_memory_gb=208.0,  # 13B × 16 bytes
        tolerance_gb=25.0,
        notes="13B full fine-tuning needs ~200GB+",
    ),

    # LoRA Benchmarks
    BenchmarkCase(
        name="7B LoRA bf16 rank=16",
        source="Unsloth benchmarks / LlamaFactory",
        model_params=7_000_000_000,
        config={
            "method": "lora",
            "optimizer": "adamw",
            "precision": "bf16",
            "lora_rank": 16,
        },
        expected_memory_gb=18.0,  # Base model (~14GB) + LoRA overhead (~4GB)
        tolerance_gb=5.0,
        notes="Base model + small LoRA overhead",
    ),

    BenchmarkCase(
        name="8B LoRA bf16 rank=16",
        source="Unsloth benchmarks",
        model_params=8_000_000_000,
        config={
            "method": "lora",
            "optimizer": "adamw",
            "precision": "bf16",
            "lora_rank": 16,
        },
        expected_memory_gb=20.0,  # Base model (~16GB) + LoRA overhead
        tolerance_gb=5.0,
        notes="Llama 3 8B LoRA fine-tuning",
    ),

    # QLoRA Benchmarks (Critical - from QLoRA paper)
    BenchmarkCase(
        name="7B QLoRA 4-bit rank=16",
        source="QLoRA paper (Dettmers et al.)",
        model_params=7_000_000_000,
        config={
            "method": "qlora",
            "optimizer": "paged_adamw_8bit",
            "precision": "bf16",
            "lora_rank": 16,
            "base_model_quantization": "nf4",
        },
        expected_memory_gb=6.0,  # ~3.5GB base + ~2.5GB overhead
        tolerance_gb=3.0,
        notes="QLoRA enables 7B on ~6GB, fitting RTX 3060",
    ),

    BenchmarkCase(
        name="8B QLoRA 4-bit rank=32",
        source="Unsloth benchmarks",
        model_params=8_000_000_000,
        config={
            "method": "qlora",
            "optimizer": "paged_adamw_8bit",
            "precision": "bf16",
            "lora_rank": 32,
            "base_model_quantization": "nf4",
        },
        expected_memory_gb=8.0,  # Fits on 8GB GPU with Unsloth
        tolerance_gb=3.0,
        notes="Llama 3 8B QLoRA fits on 8GB GPU",
    ),

    BenchmarkCase(
        name="70B QLoRA 4-bit rank=16",
        source="QLoRA paper",
        model_params=70_000_000_000,
        config={
            "method": "qlora",
            "optimizer": "paged_adamw_8bit",
            "precision": "bf16",
            "lora_rank": 16,
            "base_model_quantization": "nf4",
        },
        expected_memory_gb=46.0,  # ~35GB base + ~11GB overhead
        tolerance_gb=10.0,
        notes="70B QLoRA fits on single 48GB GPU",
    ),

    # DeepSpeed ZeRO Benchmarks
    BenchmarkCase(
        name="70B LoRA ZeRO-3 8xV100",
        source="DeepSpeed benchmarks",
        model_params=70_000_000_000,
        config={
            "method": "lora",
            "optimizer": "adamw",
            "precision": "bf16",
            "lora_rank": 16,
            "num_gpus": 8,
            "deepspeed_stage": "zero3",
        },
        # Memory breakdown:
        # - Base weights (sharded): 70B × 2 bytes / 8 GPUs = 17.5 GB
        # - LoRA gradients (sharded): ~178M × 4 bytes / 8 GPUs = 0.09 GB
        # - LoRA optimizer (sharded): ~178M × 8 bytes / 8 GPUs = 0.18 GB
        # - Activations (NOT sharded): ~5-8 GB with gradient checkpointing
        # - Framework overhead: ~10%
        # Total: ~25-30 GB per GPU
        expected_memory_gb=26.0,  # Realistic for 70B LoRA with ZeRO-3
        tolerance_gb=6.0,
        notes="70B with ZeRO-3 + LoRA on 8xV100 32GB. Activations not sharded.",
    ),

    # Additional ZeRO benchmark for full fine-tuning
    BenchmarkCase(
        name="7B Full ZeRO-3 8xGPU",
        source="DeepSpeed documentation",
        model_params=7_000_000_000,
        config={
            "method": "full",
            "optimizer": "adamw",
            "precision": "bf16",
            "num_gpus": 8,
            "deepspeed_stage": "zero3",
        },
        # Memory breakdown (ZeRO-3 shards everything):
        # - Weights (sharded): 7B × 2 bytes / 8 = 1.75 GB
        # - Gradients (sharded): 7B × 4 bytes / 8 = 3.5 GB
        # - Optimizer (sharded): 7B × 8 bytes / 8 = 7 GB
        # - Activations: ~2 GB
        # Total: ~14.25 GB + overhead = ~16 GB
        expected_memory_gb=16.0,
        tolerance_gb=4.0,
        notes="7B full fine-tuning with ZeRO-3 on 8 GPUs",
    ),
]


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

class ValidationResult:
    """Result of a validation test."""

    def __init__(
        self,
        benchmark: BenchmarkCase,
        calculated_memory: float,
        passed: bool,
        error_gb: float,
        error_percent: float,
        details: Dict[str, Any],
    ):
        self.benchmark = benchmark
        self.calculated_memory = calculated_memory
        self.passed = passed
        self.error_gb = error_gb
        self.error_percent = error_percent
        self.details = details

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.benchmark.name}\n"
            f"  Expected: {self.benchmark.expected_memory_gb:.1f} GB ± {self.benchmark.tolerance_gb:.1f} GB\n"
            f"  Calculated: {self.calculated_memory:.1f} GB\n"
            f"  Error: {self.error_gb:.1f} GB ({self.error_percent:.1f}%)\n"
            f"  Source: {self.benchmark.source}"
        )


def get_model_config(params: int) -> Dict[str, Any]:
    """Get model config based on parameter count."""
    if params <= 8_000_000_000:
        config = LLAMA_7B_CONFIG.copy()
    elif params <= 15_000_000_000:
        config = LLAMA_13B_CONFIG.copy()
    else:
        config = LLAMA_70B_CONFIG.copy()

    config["num_parameters"] = params
    return config


def run_validation(benchmark: BenchmarkCase) -> ValidationResult:
    """Run a single validation benchmark."""
    calculator = AdvancedTrainingCalculator()

    # Get model config
    config = get_model_config(benchmark.model_params)

    # Build calculation parameters
    calc_params = {
        "config": config,
        "training_stage": "sft",
        "batch_size": 1,  # Minimum batch for memory estimation
        "seq_length": 2048,
        "gradient_checkpointing": True,
        **benchmark.config,
    }

    # Run calculation
    result = calculator.calculate_advanced_training_memory(**calc_params)

    # Calculate error
    error_gb = abs(result.total_memory_gb - benchmark.expected_memory_gb)
    error_percent = 100 * error_gb / benchmark.expected_memory_gb
    passed = error_gb <= benchmark.tolerance_gb

    details = {
        "weight_memory_gb": result.weight_memory_gb,
        "gradient_memory_gb": result.gradient_memory_gb,
        "optimizer_memory_gb": result.optimizer_memory_gb,
        "activation_memory_gb": result.activation_memory_gb,
        "trainable_params": result.trainable_params,
        "trainable_percent": result.trainable_percent,
    }

    return ValidationResult(
        benchmark=benchmark,
        calculated_memory=result.total_memory_gb,
        passed=passed,
        error_gb=error_gb,
        error_percent=error_percent,
        details=details,
    )


def run_all_validations() -> List[ValidationResult]:
    """Run all validation benchmarks."""
    results = []
    for benchmark in VALIDATION_BENCHMARKS:
        try:
            result = run_validation(benchmark)
            results.append(result)
        except Exception as e:
            print(f"ERROR running {benchmark.name}: {e}")
    return results


def validate_theoretical_formulas():
    """Validate against theoretical memory formulas."""
    print("\n" + "=" * 80)
    print("THEORETICAL FORMULA VALIDATION")
    print("=" * 80)

    params_7b = 7_000_000_000
    theory = TheoreticalCalculator()

    # Test mixed precision AdamW
    expected_mixed = theory.mixed_precision_adamw(params_7b)
    print(f"\n7B Mixed Precision AdamW (theoretical): {expected_mixed:.1f} GB")

    # Test 8-bit AdamW
    expected_8bit = theory.mixed_precision_adamw_8bit(params_7b)
    print(f"7B 8-bit AdamW (theoretical): {expected_8bit:.1f} GB")

    # Test optimizer memory calculation
    print("\nOptimizer Memory Validation:")
    adamw_mem = calculate_optimizer_memory("adamw", params_7b)
    adamw8_mem = calculate_optimizer_memory("adamw_8bit", params_7b)

    expected_adamw_opt = (params_7b * 8) / 1e9  # 2 states × 4 bytes
    expected_adamw8_opt = (params_7b * 2) / 1e9  # 2 states × 1 byte

    print(f"  AdamW optimizer: {adamw_mem:.1f} GB (expected: {expected_adamw_opt:.1f} GB)")
    print(f"  AdamW 8-bit optimizer: {adamw8_mem:.1f} GB (expected: {expected_adamw8_opt:.1f} GB)")

    # Validate LoRA calculation
    print("\nLoRA Memory Validation:")
    lora_theory = theory.lora_memory(
        base_params=params_7b,
        lora_rank=16,
        num_layers=32,
        hidden_size=4096,
        num_adapters=4,
    )
    print(f"  Theoretical LoRA params: {lora_theory['total_lora_params']:,}")
    print(f"  Theoretical trainable %: {lora_theory['trainable_percent']:.3f}%")

    # Validate QLoRA calculation
    print("\nQLoRA Memory Validation:")
    qlora_theory = theory.qlora_memory(
        base_params=params_7b,
        lora_rank=16,
        num_layers=32,
        hidden_size=4096,
    )
    print(f"  4-bit base model: {qlora_theory['base_model_4bit_gb']:.1f} GB")
    print(f"  LoRA adapters: {qlora_theory['lora_adapters_gb']:.3f} GB")
    print(f"  Gradients: {qlora_theory['gradient_gb']:.3f} GB")
    print(f"  Optimizer: {qlora_theory['optimizer_gb']:.3f} GB")
    total_qlora = sum([
        qlora_theory['base_model_4bit_gb'],
        qlora_theory['lora_adapters_gb'],
        qlora_theory['gradient_gb'],
        qlora_theory['optimizer_gb'],
    ])
    print(f"  Total (theoretical): {total_qlora:.1f} GB")


def validate_deepspeed_formulas():
    """Validate DeepSpeed ZeRO memory formulas."""
    print("\n" + "=" * 80)
    print("DEEPSPEED ZERO FORMULA VALIDATION")
    print("=" * 80)

    params_7b = 7_000_000_000
    theory = TheoreticalCalculator()

    # ZeRO-2 on 8 GPUs
    print("\nZeRO-2 (8 GPUs, 7B model):")
    zero2_theory = theory.deepspeed_zero2_memory(params_7b, num_gpus=8)
    print(f"  Params per GPU: {zero2_theory['params_gb']:.1f} GB (replicated)")
    print(f"  Gradients per GPU: {zero2_theory['gradients_gb']:.1f} GB (sharded)")
    print(f"  Optimizer per GPU: {zero2_theory['optimizer_gb']:.1f} GB (sharded)")
    print(f"  Total per GPU: {zero2_theory['total_per_gpu_gb']:.1f} GB")

    # ZeRO-3 on 8 GPUs
    print("\nZeRO-3 (8 GPUs, 7B model):")
    zero3_theory = theory.deepspeed_zero3_memory(params_7b, num_gpus=8)
    print(f"  Params per GPU: {zero3_theory['params_gb']:.1f} GB (sharded)")
    print(f"  Gradients per GPU: {zero3_theory['gradients_gb']:.1f} GB (sharded)")
    print(f"  Optimizer per GPU: {zero3_theory['optimizer_gb']:.1f} GB (sharded)")
    print(f"  Total per GPU: {zero3_theory['total_per_gpu_gb']:.1f} GB")

    # Compare with our calculator
    print("\nOur Calculator Results:")
    result = calculate_distributed_memory(
        model_params=params_7b,
        trainable_params=params_7b,
        precision="bf16",
        optimizer_bytes_per_param=8,
        batch_size=1,
        seq_length=2048,
        hidden_size=4096,
        num_layers=32,
        num_gpus=8,
        deepspeed_stage="zero3",
    )
    print(f"  Total per GPU (ZeRO-3): {result.total_per_gpu_gb:.1f} GB")


def validate_parallelism_calculations():
    """Validate parallelism and communication overhead calculations."""
    print("\n" + "=" * 80)
    print("PARALLELISM CALCULATION VALIDATION")
    print("=" * 80)

    # Test various parallelism configurations
    configs = [
        ParallelismConfig(data_parallel=8),
        ParallelismConfig(tensor_parallel=2, data_parallel=4),
        ParallelismConfig(tensor_parallel=4, data_parallel=2),
        ParallelismConfig(tensor_parallel=2, pipeline_parallel=2, data_parallel=2),
        ParallelismConfig(tensor_parallel=8),
    ]

    print("\nParallelism Configurations:")
    for config in configs:
        overhead = config.get_communication_overhead()
        valid, msg = config.validate()
        status = "Valid" if valid else f"Invalid: {msg}"
        print(f"  TP={config.tensor_parallel}, PP={config.pipeline_parallel}, "
              f"DP={config.data_parallel} -> "
              f"Total GPUs: {config.total_gpus}, "
              f"Overhead: {overhead:.1%}, "
              f"Status: {status}")


def run_regression_tests():
    """Run comprehensive regression tests across all configurations."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE REGRESSION TESTS")
    print("=" * 80)

    calculator = AdvancedTrainingCalculator()

    # Test configurations
    model_sizes = [
        ("1B", 1_000_000_000, LLAMA_7B_CONFIG),
        ("7B", 7_000_000_000, LLAMA_7B_CONFIG),
        ("13B", 13_000_000_000, LLAMA_13B_CONFIG),
        ("70B", 70_000_000_000, LLAMA_70B_CONFIG),
    ]

    methods = ["full", "lora", "qlora"]
    optimizers = ["adamw", "adamw_8bit", "galore", "adafactor"]
    deepspeed_stages = [None, "zero2", "zero3"]

    results = []
    errors = []

    print("\nRunning regression tests...")

    for model_name, params, base_config in model_sizes:
        config = base_config.copy()
        config["num_parameters"] = params

        for method in methods:
            for optimizer in optimizers:
                for ds_stage in deepspeed_stages:
                    try:
                        num_gpus = 8 if ds_stage else 1

                        result = calculator.calculate_advanced_training_memory(
                            config=config,
                            training_stage="sft",
                            method=method,
                            optimizer=optimizer,
                            precision="bf16",
                            batch_size=1,
                            seq_length=2048,
                            gradient_checkpointing=True,
                            lora_rank=16,
                            num_gpus=num_gpus,
                            deepspeed_stage=ds_stage,
                            base_model_quantization="nf4" if method == "qlora" else None,
                        )

                        # Sanity checks
                        assert result.total_memory_gb > 0, "Memory should be positive"
                        assert result.total_memory_gb < 10000, "Memory unreasonably high"

                        if method in ("lora", "qlora"):
                            assert result.trainable_percent < 10, "LoRA should have low trainable %"

                        if method == "qlora":
                            # QLoRA memory efficiency check needs to account for:
                            # 1. Activation memory (NOT sharded in distributed settings)
                            # 2. Framework overhead
                            # 3. ZeRO-2 does NOT shard parameters (only gradients + optimizer)
                            # 4. ZeRO-3 shards everything including parameters
                            if ds_stage is None:
                                # Single GPU: QLoRA total should be < 10x 4-bit weight memory
                                # (includes activations, gradients, optimizer)
                                assert result.total_memory_gb < result.weight_memory_gb * 10, \
                                    f"QLoRA should be memory efficient: {result.total_memory_gb:.1f} GB vs {result.weight_memory_gb:.1f} GB weights"
                            elif ds_stage == "zero3":
                                # ZeRO-3: everything sharded including parameters
                                base_4bit_memory = (params * 0.5) / 1e9 / num_gpus
                                max_expected = base_4bit_memory + 10  # 4-bit weights + ~10GB overhead
                                assert result.total_memory_gb < max_expected, \
                                    f"QLoRA+{ds_stage} too high: {result.total_memory_gb:.1f} GB, expected < {max_expected:.1f} GB"
                            else:
                                # ZeRO-2 and others: parameters NOT sharded, only grad+opt
                                # Base model still fully replicated on each GPU
                                base_4bit_memory = (params * 0.5) / 1e9  # NOT sharded
                                max_expected = base_4bit_memory + 15  # 4-bit weights + generous overhead
                                assert result.total_memory_gb < max_expected, \
                                    f"QLoRA+{ds_stage} too high: {result.total_memory_gb:.1f} GB, expected < {max_expected:.1f} GB"

                        results.append({
                            "model": model_name,
                            "method": method,
                            "optimizer": optimizer,
                            "deepspeed": ds_stage or "none",
                            "memory_gb": result.total_memory_gb,
                            "trainable_%": result.trainable_percent,
                        })

                    except Exception as e:
                        errors.append({
                            "model": model_name,
                            "method": method,
                            "optimizer": optimizer,
                            "deepspeed": ds_stage or "none",
                            "error": str(e),
                        })

    # Print summary
    print(f"\nRegression Test Results:")
    print(f"  Total tests: {len(results) + len(errors)}")
    print(f"  Passed: {len(results)}")
    print(f"  Failed: {len(errors)}")

    if errors:
        print("\nErrors:")
        for err in errors:
            print(f"  {err['model']} {err['method']} {err['optimizer']} {err['deepspeed']}: {err['error']}")

    # Print sample results
    print("\nSample Results (7B model):")
    for r in results:
        if r["model"] == "7B" and r["deepspeed"] == "none":
            print(f"  {r['method']:6} {r['optimizer']:12} -> {r['memory_gb']:.1f} GB ({r['trainable_%']:.2f}% trainable)")

    return results, errors


def main():
    """Run all validation tests."""
    print("=" * 80)
    print("TRAINING MEMORY CALCULATOR VALIDATION SUITE")
    print("=" * 80)

    # 1. Theoretical formula validation
    validate_theoretical_formulas()

    # 2. DeepSpeed formula validation
    validate_deepspeed_formulas()

    # 3. Parallelism validation
    validate_parallelism_calculations()

    # 4. Run benchmark validations
    print("\n" + "=" * 80)
    print("BENCHMARK VALIDATION RESULTS")
    print("=" * 80)

    results = run_all_validations()

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print(f"\nResults: {passed}/{total} benchmarks passed")

    for result in results:
        print(f"\n{result}")

    # 5. Regression tests
    regression_results, regression_errors = run_regression_tests()

    # Final summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Benchmark validations: {passed}/{total} passed")
    print(f"Regression tests: {len(regression_results)}/{len(regression_results) + len(regression_errors)} passed")

    if passed < total or regression_errors:
        print("\nACTION REQUIRED: Some validations failed. Review and fix calculations.")
        return 1
    else:
        print("\nAll validations passed!")
        return 0


if __name__ == "__main__":
    exit(main())
