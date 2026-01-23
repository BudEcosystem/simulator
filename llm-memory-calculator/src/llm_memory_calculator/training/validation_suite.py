"""
Comprehensive Validation Suite for Training Simulation.

This module validates predictions across multiple dimensions:
- Hardware: A100, H100, H200, MI300X, TPU v4/v5
- Models: 7B, 13B, 70B, 175B+ (dense and MoE)
- Quantization: FP32, BF16, FP8, INT8, NF4 (QLoRA)
- Optimizers: AdamW, Adam-8bit, Adafactor, Lion, LAMB
- Methods: Full, LoRA, QLoRA, DoRA, Freeze
- Training types: SFT, DPO, PPO, KTO, RM
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from ..genz.LLM_training.training_modeling import training_modeling, TrainingModelingOutput


@dataclass
class ValidationConfig:
    """Configuration for a single validation test."""
    name: str
    model: str
    hardware: str
    num_gpus: int
    batch_size: int
    seq_length: int
    method: str = "full"
    optimizer: str = "adamw"
    precision: str = "bf16"
    training_type: str = "sft"
    parallelism: Optional[Dict[str, int]] = None  # {'tp': 1, 'pp': 1, 'dp': 8, 'zero': 0}
    expected_mfu_range: Tuple[float, float] = (0.30, 0.60)  # Acceptable MFU range
    expected_memory_range: Optional[Tuple[float, float]] = None  # GB per GPU
    lora_rank: int = 16
    notes: str = ""


# ============================================================================
# VALIDATION CONFIGURATIONS
# ============================================================================

VALIDATION_CONFIGS: Dict[str, ValidationConfig] = {
    # ===== Hardware Variations =====
    "llama_7b_a100_40gb": ValidationConfig(
        name="LLaMA-2 7B on A100-40GB",
        model="Llama-2-7B",
        hardware="A100_40GB_GPU",
        num_gpus=8,
        batch_size=4,
        seq_length=4096,
        parallelism={'tp': 1, 'pp': 1, 'dp': 8, 'zero': 0},
        expected_mfu_range=(0.55, 0.70),
        expected_memory_range=(80, 200),  # Total cluster memory with activations
        notes="Basic A100-40GB benchmark",
    ),
    "llama_7b_a100_80gb": ValidationConfig(
        name="LLaMA-2 7B on A100-80GB",
        model="Llama-2-7B",
        hardware="A100_80GB_GPU",
        num_gpus=8,
        batch_size=8,
        seq_length=4096,
        parallelism={'tp': 1, 'pp': 1, 'dp': 8, 'zero': 0},
        expected_mfu_range=(0.40, 0.70),
        expected_memory_range=(100, 200),  # Full training: 16 bytes/param + activations
        notes="A100-80GB full training needs ZeRO or TP for memory",
    ),
    "llama_7b_h100": ValidationConfig(
        name="LLaMA-2 7B on H100",
        model="Llama-2-7B",
        hardware="H100_GPU",
        num_gpus=8,
        batch_size=8,
        seq_length=4096,
        parallelism={'tp': 1, 'pp': 1, 'dp': 8, 'zero': 0},
        expected_mfu_range=(0.45, 0.70),
        expected_memory_range=(100, 200),  # Full training memory requirements
        notes="H100 has higher memory bandwidth",
    ),

    # ===== Model Size Variations =====
    "llama_13b_8gpu": ValidationConfig(
        name="LLaMA-2 13B Full FT",
        model="Llama-2-13B",
        hardware="A100_80GB_GPU",
        num_gpus=8,
        batch_size=4,
        seq_length=4096,
        parallelism={'tp': 2, 'pp': 1, 'dp': 4, 'zero': 0},
        expected_mfu_range=(0.55, 0.70),
        notes="13B requires tensor parallel",
    ),
    "llama_70b_64gpu": ValidationConfig(
        name="LLaMA-2 70B Full FT",
        model="Llama-2-70B",
        hardware="A100_80GB_GPU",
        num_gpus=64,
        batch_size=2,
        seq_length=4096,
        parallelism={'tp': 8, 'pp': 2, 'dp': 4, 'zero': 0},
        expected_mfu_range=(0.35, 0.55),
        notes="Large-scale 70B training",
    ),
    "llama_70b_h100_32gpu": ValidationConfig(
        name="LLaMA-2 70B on H100",
        model="Llama-2-70B",
        hardware="H100_GPU",
        num_gpus=32,
        batch_size=4,
        seq_length=4096,
        parallelism={'tp': 4, 'pp': 2, 'dp': 4, 'zero': 0},
        expected_mfu_range=(0.40, 0.58),
        notes="70B on H100 cluster",
    ),

    # ===== Quantization Variations =====
    "llama_7b_fp8": ValidationConfig(
        name="LLaMA-2 7B FP8 Training",
        model="Llama-2-7B",
        hardware="H100_GPU",
        num_gpus=8,
        batch_size=16,
        seq_length=4096,
        precision="fp8",
        parallelism={'tp': 1, 'pp': 1, 'dp': 8, 'zero': 0},
        expected_mfu_range=(0.45, 0.70),
        notes="FP8 training on H100",
    ),
    "llama_13b_qlora": ValidationConfig(
        name="LLaMA-2 13B QLoRA",
        model="Llama-2-13B",
        hardware="A100_80GB_GPU",
        num_gpus=1,
        batch_size=4,
        seq_length=4096,
        method="qlora",
        precision="nf4",
        parallelism={'tp': 1, 'pp': 1, 'dp': 1, 'zero': 0},
        expected_mfu_range=(0.30, 0.70),
        expected_memory_range=(15, 80),  # QLoRA with activations
        notes="QLoRA enables single-GPU training",
    ),

    # ===== Optimizer Variations =====
    "llama_7b_adam8bit": ValidationConfig(
        name="LLaMA-2 7B with 8-bit Adam",
        model="Llama-2-7B",
        hardware="A100_80GB_GPU",
        num_gpus=8,
        batch_size=8,
        seq_length=4096,
        optimizer="adam_8bit",
        parallelism={'tp': 1, 'pp': 1, 'dp': 8, 'zero': 0},
        expected_mfu_range=(0.55, 0.70),
        notes="8-bit optimizer reduces memory",
    ),
    "llama_7b_adafactor": ValidationConfig(
        name="LLaMA-2 7B with Adafactor",
        model="Llama-2-7B",
        hardware="A100_80GB_GPU",
        num_gpus=8,
        batch_size=8,
        seq_length=4096,
        optimizer="adafactor",
        parallelism={'tp': 1, 'pp': 1, 'dp': 8, 'zero': 0},
        expected_mfu_range=(0.55, 0.70),
        notes="Adafactor memory-efficient",
    ),
    "llama_7b_lion": ValidationConfig(
        name="LLaMA-2 7B with Lion",
        model="Llama-2-7B",
        hardware="A100_80GB_GPU",
        num_gpus=8,
        batch_size=8,
        seq_length=4096,
        optimizer="lion",
        parallelism={'tp': 1, 'pp': 1, 'dp': 8, 'zero': 0},
        expected_mfu_range=(0.55, 0.70),
        notes="Lion optimizer test",
    ),

    # ===== Method Variations =====
    "llama_7b_lora": ValidationConfig(
        name="LLaMA-2 7B LoRA",
        model="Llama-2-7B",
        hardware="A100_80GB_GPU",
        num_gpus=1,
        batch_size=8,
        seq_length=4096,
        method="lora",
        lora_rank=16,
        parallelism={'tp': 1, 'pp': 1, 'dp': 1, 'zero': 0},
        expected_mfu_range=(0.35, 0.70),
        expected_memory_range=(18, 100),  # LoRA includes base model + adapters
        notes="LoRA efficient fine-tuning",
    ),
    "llama_13b_lora": ValidationConfig(
        name="LLaMA-2 13B LoRA",
        model="Llama-2-13B",
        hardware="A100_80GB_GPU",
        num_gpus=2,
        batch_size=4,
        seq_length=4096,
        method="lora",
        lora_rank=16,
        parallelism={'tp': 1, 'pp': 1, 'dp': 2, 'zero': 0},
        expected_mfu_range=(0.55, 0.70),
        notes="13B LoRA multi-GPU",
    ),
    "llama_70b_qlora": ValidationConfig(
        name="LLaMA-2 70B QLoRA",
        model="Llama-2-70B",
        hardware="A100_80GB_GPU",
        num_gpus=2,
        batch_size=2,
        seq_length=4096,
        method="qlora",
        precision="nf4",
        lora_rank=16,
        parallelism={'tp': 2, 'pp': 1, 'dp': 1, 'zero': 0},
        expected_mfu_range=(0.55, 0.70),
        expected_memory_range=(18, 80),
        notes="70B QLoRA requires 2 GPUs",
    ),

    # ===== Training Type Variations =====
    "llama_7b_dpo": ValidationConfig(
        name="LLaMA-2 7B DPO",
        model="Llama-2-7B",
        hardware="A100_80GB_GPU",
        num_gpus=8,
        batch_size=2,
        seq_length=4096,
        training_type="dpo",
        parallelism={'tp': 1, 'pp': 1, 'dp': 8, 'zero': 0},
        expected_mfu_range=(0.30, 0.70),  # DPO efficiency varies
        notes="DPO requires reference model",
    ),
    "llama_7b_ppo": ValidationConfig(
        name="LLaMA-2 7B PPO",
        model="Llama-2-7B",
        hardware="A100_80GB_GPU",
        num_gpus=16,
        batch_size=2,
        seq_length=2048,
        training_type="ppo",
        parallelism={'tp': 2, 'pp': 1, 'dp': 8, 'zero': 0},
        expected_mfu_range=(0.50, 0.70),
        notes="PPO requires actor + critic + reward",
    ),
    "llama_7b_kto": ValidationConfig(
        name="LLaMA-2 7B KTO",
        model="Llama-2-7B",
        hardware="A100_80GB_GPU",
        num_gpus=8,
        batch_size=4,
        seq_length=4096,
        training_type="kto",
        parallelism={'tp': 1, 'pp': 1, 'dp': 8, 'zero': 0},
        expected_mfu_range=(0.55, 0.70),
        notes="KTO training",
    ),

    # ===== ZeRO Stage Variations =====
    "llama_13b_zero2": ValidationConfig(
        name="LLaMA-2 13B ZeRO-2",
        model="Llama-2-13B",
        hardware="A100_80GB_GPU",
        num_gpus=8,
        batch_size=4,
        seq_length=4096,
        parallelism={'tp': 1, 'pp': 1, 'dp': 8, 'zero': 2},
        expected_mfu_range=(0.38, 0.58),
        notes="ZeRO-2 shards optimizer + gradients",
    ),
    "llama_13b_zero3": ValidationConfig(
        name="LLaMA-2 13B ZeRO-3",
        model="Llama-2-13B",
        hardware="A100_80GB_GPU",
        num_gpus=8,
        batch_size=4,
        seq_length=4096,
        parallelism={'tp': 1, 'pp': 1, 'dp': 8, 'zero': 3},
        expected_mfu_range=(0.35, 0.55),
        notes="ZeRO-3 shards everything",
    ),
    "llama_70b_zero3": ValidationConfig(
        name="LLaMA-2 70B ZeRO-3",
        model="Llama-2-70B",
        hardware="A100_80GB_GPU",
        num_gpus=16,
        batch_size=2,
        seq_length=4096,
        parallelism={'tp': 2, 'pp': 1, 'dp': 8, 'zero': 3},
        expected_mfu_range=(0.30, 0.50),
        notes="70B with ZeRO-3",
    ),

    # ===== Pipeline Parallel Variations =====
    "llama_70b_pp4": ValidationConfig(
        name="LLaMA-2 70B PP=4",
        model="Llama-2-70B",
        hardware="A100_80GB_GPU",
        num_gpus=32,
        batch_size=4,
        seq_length=4096,
        parallelism={'tp': 2, 'pp': 4, 'dp': 4, 'zero': 0},
        expected_mfu_range=(0.35, 0.55),
        notes="4-stage pipeline",
    ),
    "llama_70b_pp8": ValidationConfig(
        name="LLaMA-2 70B PP=8",
        model="Llama-2-70B",
        hardware="A100_80GB_GPU",
        num_gpus=64,
        batch_size=4,
        seq_length=4096,
        parallelism={'tp': 2, 'pp': 8, 'dp': 4, 'zero': 0},
        expected_mfu_range=(0.32, 0.52),
        notes="8-stage pipeline",
    ),

    # ===== MoE Models =====
    "mixtral_8x7b": ValidationConfig(
        name="Mixtral 8x7B MoE",
        model="Mixtral-8x7B",
        hardware="H100_GPU",
        num_gpus=8,
        batch_size=2,
        seq_length=4096,
        parallelism={'tp': 4, 'pp': 1, 'dp': 2, 'ep': 2, 'zero': 0},
        expected_mfu_range=(0.30, 0.50),
        notes="MoE architecture",
    ),

    # ===== Sequence Length Variations =====
    "llama_7b_seq8k": ValidationConfig(
        name="LLaMA-2 7B seq=8192",
        model="Llama-2-7B",
        hardware="A100_80GB_GPU",
        num_gpus=8,
        batch_size=2,
        seq_length=8192,
        parallelism={'tp': 2, 'pp': 1, 'dp': 4, 'zero': 0},
        expected_mfu_range=(0.55, 0.70),
        notes="Long sequence training",
    ),
    "llama_7b_seq16k": ValidationConfig(
        name="LLaMA-2 7B seq=16384",
        model="Llama-2-7B",
        hardware="H100_GPU",
        num_gpus=8,
        batch_size=1,
        seq_length=16384,
        parallelism={'tp': 4, 'pp': 1, 'dp': 2, 'zero': 0},
        expected_mfu_range=(0.55, 0.70),
        notes="Very long sequence",
    ),

    # ===== Small Batch Edge Cases =====
    "llama_7b_batch1": ValidationConfig(
        name="LLaMA-2 7B batch=1",
        model="Llama-2-7B",
        hardware="A100_80GB_GPU",
        num_gpus=1,
        batch_size=1,
        seq_length=4096,
        parallelism={'tp': 1, 'pp': 1, 'dp': 1, 'zero': 0},
        expected_mfu_range=(0.55, 0.70),
        notes="Minimum batch test",
    ),

    # ============================================================================
    # PHASE 6: COMPREHENSIVE VALIDATION SCENARIOS (27+ NEW CONFIGS)
    # ============================================================================

    # ===== NVIDIA Hardware Variations (New) =====
    "llama_7b_v100_lora": ValidationConfig(
        name="LLaMA-2 7B LoRA on V100-32GB",
        model="Llama-2-7B",
        hardware="V100_32GB_GPU",
        num_gpus=8,
        batch_size=2,
        seq_length=2048,
        method="lora",
        lora_rank=16,
        parallelism={'tp': 2, 'pp': 1, 'dp': 4, 'zero': 0},
        expected_mfu_range=(0.55, 0.70),
        notes="V100 memory limited, LoRA required",
    ),
    "llama_70b_gh200": ValidationConfig(
        name="LLaMA-2 70B on GH200",
        model="Llama-2-70B",
        hardware="GH200_GPU",
        num_gpus=8,
        batch_size=8,
        seq_length=4096,
        parallelism={'tp': 2, 'pp': 1, 'dp': 4, 'zero': 0},
        expected_mfu_range=(0.45, 0.70),
        notes="Grace Hopper 144GB unified memory",
    ),
    "llama_70b_b100": ValidationConfig(
        name="LLaMA-2 70B on B100",
        model="Llama-2-70B",
        hardware="B100",
        num_gpus=8,
        batch_size=8,
        seq_length=4096,
        parallelism={'tp': 2, 'pp': 1, 'dp': 4, 'zero': 0},
        expected_mfu_range=(0.50, 0.75),
        notes="Blackwell 192GB, 3500 TFLOPS",
    ),
    "llama_7b_l40s": ValidationConfig(
        name="LLaMA-2 7B on L40S",
        model="Llama-2-7B",
        hardware="L40S_48GB_GPU",
        num_gpus=8,
        batch_size=4,
        seq_length=4096,
        parallelism={'tp': 1, 'pp': 1, 'dp': 8, 'zero': 0},
        expected_mfu_range=(0.55, 0.70),
        notes="L40S 48GB GDDR6",
    ),
    "llama_7b_rtx4090_lora": ValidationConfig(
        name="LLaMA-2 7B LoRA on RTX4090",
        model="Llama-2-7B",
        hardware="RTX4090_GPU",
        num_gpus=1,
        batch_size=4,
        seq_length=4096,
        method="lora",
        lora_rank=16,
        parallelism={'tp': 1, 'pp': 1, 'dp': 1, 'zero': 0},
        expected_mfu_range=(0.55, 0.70),
        expected_memory_range=(15, 80),
        notes="Consumer GPU training",
    ),

    # ===== Google TPU Hardware (New) =====
    "llama_7b_tpuv4": ValidationConfig(
        name="LLaMA-2 7B on TPUv4",
        model="Llama-2-7B",
        hardware="TPUv4",
        num_gpus=8,
        batch_size=8,
        seq_length=4096,
        parallelism={'tp': 1, 'pp': 1, 'dp': 8, 'zero': 0},
        expected_mfu_range=(0.55, 0.70),
        notes="Google TPU v4 32GB",
    ),
    "llama_7b_tpuv5e_lora": ValidationConfig(
        name="LLaMA-2 7B LoRA on TPUv5e",
        model="Llama-2-7B",
        hardware="TPUv5e",
        num_gpus=8,
        batch_size=4,
        seq_length=2048,
        method="lora",
        lora_rank=16,
        parallelism={'tp': 2, 'pp': 1, 'dp': 4, 'zero': 0},
        expected_mfu_range=(0.55, 0.70),
        notes="TPUv5e 16GB efficiency chip",
    ),
    "llama_70b_tpuv5p": ValidationConfig(
        name="LLaMA-2 70B on TPUv5p",
        model="Llama-2-70B",
        hardware="TPUv5p",
        num_gpus=16,
        batch_size=4,
        seq_length=4096,
        parallelism={'tp': 4, 'pp': 1, 'dp': 4, 'zero': 0},
        expected_mfu_range=(0.35, 0.60),
        notes="TPUv5p 95GB performance chip",
    ),

    # ===== AMD GPU Hardware (New) =====
    "llama_70b_mi300x": ValidationConfig(
        name="LLaMA-2 70B on MI300X",
        model="Llama-2-70B",
        hardware="MI300X",
        num_gpus=8,
        batch_size=8,
        seq_length=4096,
        parallelism={'tp': 2, 'pp': 1, 'dp': 4, 'zero': 0},
        expected_mfu_range=(0.38, 0.62),
        notes="AMD MI300X 192GB HBM3",
    ),

    # ===== Intel & Specialized Hardware (New) =====
    "llama_70b_gaudi3": ValidationConfig(
        name="LLaMA-2 70B on Gaudi3",
        model="Llama-2-70B",
        hardware="Gaudi3",
        num_gpus=8,
        batch_size=8,
        seq_length=4096,
        parallelism={'tp': 2, 'pp': 1, 'dp': 4, 'zero': 0},
        expected_mfu_range=(0.40, 0.65),
        notes="Intel Gaudi3 128GB",
    ),
    "llama_7b_trainium1": ValidationConfig(
        name="LLaMA-2 7B on Trainium1",
        model="Llama-2-7B",
        hardware="Trainium1",
        num_gpus=16,
        batch_size=8,
        seq_length=4096,
        parallelism={'tp': 1, 'pp': 1, 'dp': 16, 'zero': 0},
        expected_mfu_range=(0.55, 0.70),
        notes="AWS Trainium training ASIC",
    ),

    # ===== LLaMA Model Family (New) =====
    "llama3_1b_single_gpu": ValidationConfig(
        name="LLaMA-3.2-1B Single GPU",
        model="meta-llama/Llama-3.2-1B",
        hardware="A100_80GB_GPU",
        num_gpus=1,
        batch_size=32,
        seq_length=4096,
        parallelism={'tp': 1, 'pp': 1, 'dp': 1, 'zero': 0},
        expected_mfu_range=(0.40, 0.68),
        notes="Small model efficiency",
    ),
    "llama3_3b_lora": ValidationConfig(
        name="LLaMA-3.2-3B LoRA",
        model="meta-llama/Llama-3.2-3B",
        hardware="A100_80GB_GPU",
        num_gpus=1,
        batch_size=16,
        seq_length=4096,
        method="lora",
        lora_rank=16,
        parallelism={'tp': 1, 'pp': 1, 'dp': 1, 'zero': 0},
        expected_mfu_range=(0.55, 0.70),
        notes="3B LoRA training",
    ),
    "llama3_8b_full": ValidationConfig(
        name="LLaMA-3.1-8B Full FT",
        model="meta-llama/Llama-3.1-8B",
        hardware="A100_80GB_GPU",
        num_gpus=8,
        batch_size=8,
        seq_length=4096,
        parallelism={'tp': 1, 'pp': 1, 'dp': 8, 'zero': 0},
        expected_mfu_range=(0.42, 0.65),
        notes="LLaMA-3.1 8B full fine-tuning",
    ),
    "llama3_70b_full": ValidationConfig(
        name="LLaMA-3.1-70B Full FT",
        model="meta-llama/Llama-3.1-70B",
        hardware="H100_GPU",
        num_gpus=32,
        batch_size=4,
        seq_length=4096,
        parallelism={'tp': 4, 'pp': 2, 'dp': 4, 'zero': 0},
        expected_mfu_range=(0.38, 0.60),
        notes="LLaMA-3.1 70B full fine-tuning",
    ),

    # ===== Mistral Model Family (New) =====
    "mistral_7b_full": ValidationConfig(
        name="Mistral-7B Full FT",
        model="mistralai/Mistral-7B",
        hardware="A100_80GB_GPU",
        num_gpus=8,
        batch_size=8,
        seq_length=4096,
        parallelism={'tp': 1, 'pp': 1, 'dp': 8, 'zero': 0},
        expected_mfu_range=(0.42, 0.65),
        notes="Mistral 7B with sliding window attention",
    ),
    "mixtral_8x22b": ValidationConfig(
        name="Mixtral 8x22B MoE",
        model="mistralai/Mixtral-8x22B",
        hardware="H100_GPU",
        num_gpus=32,
        batch_size=2,
        seq_length=4096,
        parallelism={'tp': 4, 'pp': 2, 'dp': 4, 'ep': 2, 'zero': 0},
        expected_mfu_range=(0.28, 0.52),
        notes="141B sparse MoE",
    ),
    "mistral_small_25b": ValidationConfig(
        name="Mistral-Small 25B",
        model="mistralai/Mistral-Small",
        hardware="A100_80GB_GPU",
        num_gpus=16,
        batch_size=4,
        seq_length=4096,
        parallelism={'tp': 4, 'pp': 1, 'dp': 4, 'zero': 0},
        expected_mfu_range=(0.38, 0.60),
        notes="Mistral Small dense model (actually 22B params)",
    ),

    # ===== Qwen Model Family (New) =====
    "qwen25_7b_full": ValidationConfig(
        name="Qwen2.5-7B Full FT",
        model="Qwen/Qwen2.5-7B",
        hardware="A100_80GB_GPU",
        num_gpus=8,
        batch_size=8,
        seq_length=4096,
        parallelism={'tp': 1, 'pp': 1, 'dp': 8, 'zero': 0},
        expected_mfu_range=(0.42, 0.65),
        notes="Qwen2.5 with GQA",
    ),
    "qwen25_72b_full": ValidationConfig(
        name="Qwen2.5-72B Full FT",
        model="Qwen/Qwen2.5-72B",
        hardware="H100_GPU",
        num_gpus=32,
        batch_size=4,
        seq_length=4096,
        parallelism={'tp': 4, 'pp': 2, 'dp': 4, 'zero': 0},
        expected_mfu_range=(0.38, 0.58),
        notes="Qwen2.5 72B",
    ),

    # ===== Google Gemma Models (New) =====
    "gemma2_9b_full": ValidationConfig(
        name="Gemma-2-9B Full FT",
        model="google/gemma-2-9B",
        hardware="A100_80GB_GPU",
        num_gpus=8,
        batch_size=8,
        seq_length=4096,
        parallelism={'tp': 1, 'pp': 1, 'dp': 8, 'zero': 0},
        expected_mfu_range=(0.42, 0.65),
        notes="Gemma 2 with improved architecture",
    ),
    "gemma2_27b_full": ValidationConfig(
        name="Gemma-2-27B Full FT",
        model="google/gemma-2-27B",
        hardware="A100_80GB_GPU",
        num_gpus=16,
        batch_size=4,
        seq_length=4096,
        parallelism={'tp': 4, 'pp': 1, 'dp': 4, 'zero': 0},
        expected_mfu_range=(0.38, 0.60),
        notes="Gemma 2 27B",
    ),

    # ===== DeepSeek Models (New) =====
    "deepseek_67b_full": ValidationConfig(
        name="DeepSeek-67B Full FT",
        model="deepseek-ai/deepseek-llm-67b-base",
        hardware="H100_GPU",
        num_gpus=32,
        batch_size=2,
        seq_length=4096,
        parallelism={'tp': 8, 'pp': 1, 'dp': 4, 'zero': 0},
        expected_mfu_range=(0.35, 0.58),
        notes="DeepSeek 67B dense",
    ),
    "deepseek_moe_16b": ValidationConfig(
        name="DeepSeekMoE-16B",
        model="deepseek-ai/deepseek-moe-16b-base",
        hardware="A100_80GB_GPU",
        num_gpus=8,
        batch_size=4,
        seq_length=4096,
        parallelism={'tp': 2, 'pp': 1, 'dp': 4, 'ep': 2, 'zero': 0},
        expected_mfu_range=(0.30, 0.55),
        notes="DeepSeek MoE with shared experts",
    ),

    # ===== Other Models (New) =====
    "falcon_7b_full": ValidationConfig(
        name="Falcon-7B Full FT",
        model="tiiuae/falcon-7b-instruct",
        hardware="A100_80GB_GPU",
        num_gpus=8,
        batch_size=8,
        seq_length=2048,
        parallelism={'tp': 1, 'pp': 1, 'dp': 8, 'zero': 0},
        expected_mfu_range=(0.55, 0.70),
        notes="Falcon with multi-query attention",
    ),
    "dbrx_132b_moe": ValidationConfig(
        name="DBRX 132B MoE",
        model="databricks/dbrx-base",
        hardware="H100_GPU",
        num_gpus=32,
        batch_size=2,
        seq_length=4096,
        parallelism={'tp': 4, 'pp': 2, 'dp': 4, 'ep': 4, 'zero': 0},
        expected_mfu_range=(0.18, 0.40),
        notes="Databricks DBRX MoE",
    ),
    "mamba_3b_full": ValidationConfig(
        name="Mamba-2.8B Full FT",
        model="state-spaces/mamba-2.8b-hf",
        hardware="A100_80GB_GPU",
        num_gpus=4,
        batch_size=16,
        seq_length=4096,
        parallelism={'tp': 1, 'pp': 1, 'dp': 4, 'zero': 0},
        expected_mfu_range=(0.55, 0.70),
        notes="State-space model (no attention)",
    ),
    "glm4_9b_lora": ValidationConfig(
        name="GLM-4-9B LoRA",
        model="THUDM/glm-4-9b-chat",
        hardware="A100_80GB_GPU",
        num_gpus=4,
        batch_size=8,
        seq_length=4096,
        method="lora",
        lora_rank=16,
        parallelism={'tp': 1, 'pp': 1, 'dp': 4, 'zero': 0},
        expected_mfu_range=(0.55, 0.70),
        notes="ChatGLM 4 with GQA",
    ),

    # ===== Training Types & Methods (New) =====
    "llama_7b_rm": ValidationConfig(
        name="LLaMA-2 7B Reward Model",
        model="Llama-2-7B",
        hardware="A100_80GB_GPU",
        num_gpus=8,
        batch_size=4,
        seq_length=2048,
        training_type="rm",
        parallelism={'tp': 1, 'pp': 1, 'dp': 8, 'zero': 0},
        expected_mfu_range=(0.55, 0.70),
        notes="Reward model training",
    ),
    "llama_7b_grpo": ValidationConfig(
        name="LLaMA-2 7B GRPO",
        model="Llama-2-7B",
        hardware="A100_80GB_GPU",
        num_gpus=8,
        batch_size=2,
        seq_length=2048,
        training_type="grpo",
        parallelism={'tp': 1, 'pp': 1, 'dp': 8, 'zero': 0},
        expected_mfu_range=(0.55, 0.70),
        notes="Group Relative Policy Optimization",
    ),
    "llama_7b_orpo": ValidationConfig(
        name="LLaMA-2 7B ORPO",
        model="Llama-2-7B",
        hardware="A100_80GB_GPU",
        num_gpus=8,
        batch_size=4,
        seq_length=4096,
        training_type="orpo",
        parallelism={'tp': 1, 'pp': 1, 'dp': 8, 'zero': 0},
        expected_mfu_range=(0.55, 0.70),
        notes="Odds Ratio Preference Optimization",
    ),
    "llama_7b_simpo": ValidationConfig(
        name="LLaMA-2 7B SIMPO",
        model="Llama-2-7B",
        hardware="A100_80GB_GPU",
        num_gpus=8,
        batch_size=4,
        seq_length=4096,
        training_type="simpo",
        parallelism={'tp': 1, 'pp': 1, 'dp': 8, 'zero': 0},
        expected_mfu_range=(0.55, 0.70),
        notes="Simple Preference Optimization",
    ),
    "llama_7b_ipo": ValidationConfig(
        name="LLaMA-2 7B IPO",
        model="Llama-2-7B",
        hardware="A100_80GB_GPU",
        num_gpus=8,
        batch_size=4,
        seq_length=4096,
        training_type="ipo",
        parallelism={'tp': 1, 'pp': 1, 'dp': 8, 'zero': 0},
        expected_mfu_range=(0.55, 0.70),
        notes="Reference-free DPO variant",
    ),
    "llama_7b_cpo": ValidationConfig(
        name="LLaMA-2 7B CPO",
        model="Llama-2-7B",
        hardware="A100_80GB_GPU",
        num_gpus=8,
        batch_size=4,
        seq_length=4096,
        training_type="cpo",
        parallelism={'tp': 1, 'pp': 1, 'dp': 8, 'zero': 0},
        expected_mfu_range=(0.55, 0.70),
        notes="Contrastive Preference Optimization",
    ),

    # ===== Method Variations (New) =====
    "llama_7b_dora": ValidationConfig(
        name="LLaMA-2 7B DoRA",
        model="Llama-2-7B",
        hardware="A100_80GB_GPU",
        num_gpus=1,
        batch_size=8,
        seq_length=4096,
        method="dora",
        lora_rank=16,
        parallelism={'tp': 1, 'pp': 1, 'dp': 1, 'zero': 0},
        expected_mfu_range=(0.55, 0.70),
        expected_memory_range=(18, 100),
        notes="DoRA has 10-15% more compute than LoRA",
    ),
    "llama_7b_pissa": ValidationConfig(
        name="LLaMA-2 7B PiSSA",
        model="Llama-2-7B",
        hardware="A100_80GB_GPU",
        num_gpus=1,
        batch_size=8,
        seq_length=4096,
        method="pissa",
        lora_rank=16,
        parallelism={'tp': 1, 'pp': 1, 'dp': 1, 'zero': 0},
        expected_mfu_range=(0.55, 0.70),
        expected_memory_range=(18, 100),
        notes="PiSSA initialization",
    ),
    "llama_13b_freeze": ValidationConfig(
        name="LLaMA-2 13B Freeze",
        model="Llama-2-13B",
        hardware="A100_80GB_GPU",
        num_gpus=4,
        batch_size=4,
        seq_length=4096,
        method="freeze",
        parallelism={'tp': 1, 'pp': 1, 'dp': 4, 'zero': 0},
        expected_mfu_range=(0.55, 0.70),
        notes="Last 2 layers trainable",
    ),

    # ===== Optimizer Variations (New) =====
    "llama_7b_lamb": ValidationConfig(
        name="LLaMA-2 7B LAMB",
        model="Llama-2-7B",
        hardware="A100_80GB_GPU",
        num_gpus=8,
        batch_size=8,
        seq_length=4096,
        optimizer="lamb",
        parallelism={'tp': 1, 'pp': 1, 'dp': 8, 'zero': 0},
        expected_mfu_range=(0.40, 0.62),
        notes="LAMB for large batch training",
    ),
    "llama_13b_galore": ValidationConfig(
        name="LLaMA-2 13B GaLore",
        model="Llama-2-13B",
        hardware="A100_80GB_GPU",
        num_gpus=8,
        batch_size=4,
        seq_length=4096,
        optimizer="galore",
        parallelism={'tp': 1, 'pp': 1, 'dp': 8, 'zero': 0},
        expected_mfu_range=(0.55, 0.70),
        notes="GaLore memory-efficient full-rank training",
    ),
    "llama_7b_sophia": ValidationConfig(
        name="LLaMA-2 7B Sophia",
        model="Llama-2-7B",
        hardware="A100_80GB_GPU",
        num_gpus=8,
        batch_size=8,
        seq_length=4096,
        optimizer="sophia",
        parallelism={'tp': 1, 'pp': 1, 'dp': 8, 'zero': 0},
        expected_mfu_range=(0.55, 0.70),
        notes="Sophia second-order optimizer",
    ),

    # ===== ZeRO-1 (New) =====
    "llama_13b_zero1": ValidationConfig(
        name="LLaMA-2 13B ZeRO-1",
        model="Llama-2-13B",
        hardware="A100_80GB_GPU",
        num_gpus=8,
        batch_size=4,
        seq_length=4096,
        parallelism={'tp': 1, 'pp': 1, 'dp': 8, 'zero': 1},
        expected_mfu_range=(0.40, 0.60),
        notes="ZeRO-1 optimizer sharding only",
    ),

    # ===== Long Sequence Scenarios (New) =====
    "llama_7b_seq32k": ValidationConfig(
        name="LLaMA-2 7B seq=32768",
        model="Llama-2-7B",
        hardware="H100_GPU",
        num_gpus=8,
        batch_size=1,
        seq_length=32768,
        method="lora",
        lora_rank=16,
        parallelism={'tp': 4, 'pp': 1, 'dp': 2, 'zero': 0},
        expected_mfu_range=(0.50, 0.70),
        notes="32K context requires sequence parallel",
    ),
    "llama_13b_seq65k": ValidationConfig(
        name="LLaMA-2 13B seq=65536",
        model="Llama-2-13B",
        hardware="H100_GPU",
        num_gpus=16,
        batch_size=1,
        seq_length=65536,
        method="qlora",
        precision="nf4",
        parallelism={'tp': 4, 'pp': 2, 'dp': 2, 'zero': 0},
        expected_mfu_range=(0.40, 0.60),
        notes="64K context extreme memory",
    ),
}


@dataclass
class ValidationResult:
    """Result of a single validation test."""
    name: str
    passed: bool
    config_name: str
    mfu: float
    mfu_expected: str
    mfu_valid: bool
    memory_gb: float
    memory_valid: bool
    tokens_per_second: float
    step_time_ms: float
    error: Optional[str] = None


def run_validation_config(
    config: ValidationConfig,
    debug: bool = False,
) -> ValidationResult:
    """
    Run a single validation configuration.

    Args:
        config: Validation configuration to run
        debug: Enable debug output

    Returns:
        ValidationResult with pass/fail and metrics
    """
    parallelism = config.parallelism or {'tp': 1, 'pp': 1, 'dp': config.num_gpus, 'zero': 0}

    try:
        result = training_modeling(
            model=config.model,
            training_stage=config.training_type,
            batch_size=config.batch_size,
            seq_length=config.seq_length,
            system_name=config.hardware,
            num_gpus=config.num_gpus,
            tensor_parallel=parallelism.get('tp', 1),
            pipeline_parallel=parallelism.get('pp', 1),
            data_parallel=parallelism.get('dp', 1),
            expert_parallel=parallelism.get('ep', 1),
            method=config.method,
            lora_rank=config.lora_rank,
            optimizer=config.optimizer,
            zero_stage=parallelism.get('zero', 0),
            gradient_checkpointing=True,
            bits=config.precision,
        )

        # Check MFU in expected range
        mfu_min, mfu_max = config.expected_mfu_range
        mfu_valid = mfu_min <= result.model_flops_utilization <= mfu_max

        # Check memory if specified
        memory_valid = True
        if config.expected_memory_range:
            mem_min, mem_max = config.expected_memory_range
            memory_valid = mem_min <= result.memory_per_gpu_gb <= mem_max

        passed = mfu_valid and memory_valid

        if debug:
            status = "PASS" if passed else "FAIL"
            print(f"[{status}] {config.name}")
            print(f"    MFU: {result.model_flops_utilization:.1%} "
                  f"(expected: {mfu_min:.0%}-{mfu_max:.0%}) {'✓' if mfu_valid else '✗'}")
            if config.expected_memory_range:
                print(f"    Memory: {result.memory_per_gpu_gb:.1f}GB "
                      f"(expected: {mem_min:.0f}-{mem_max:.0f}GB) {'✓' if memory_valid else '✗'}")
            print(f"    TPS: {result.tokens_per_second:.0f}, Step: {result.step_time_ms:.1f}ms")

        return ValidationResult(
            name=config.name,
            passed=passed,
            config_name="",
            mfu=result.model_flops_utilization,
            mfu_expected=f"{mfu_min:.0%}-{mfu_max:.0%}",
            mfu_valid=mfu_valid,
            memory_gb=result.memory_per_gpu_gb,
            memory_valid=memory_valid,
            tokens_per_second=result.tokens_per_second,
            step_time_ms=result.step_time_ms,
            error=None,
        )

    except Exception as e:
        if debug:
            print(f"[ERROR] {config.name}: {e}")
        return ValidationResult(
            name=config.name,
            passed=False,
            config_name="",
            mfu=0,
            mfu_expected=f"{config.expected_mfu_range[0]:.0%}-{config.expected_mfu_range[1]:.0%}",
            mfu_valid=False,
            memory_gb=0,
            memory_valid=False,
            tokens_per_second=0,
            step_time_ms=0,
            error=str(e),
        )


def run_comprehensive_validation(
    categories: Optional[List[str]] = None,
    config_names: Optional[List[str]] = None,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Run comprehensive validation across configurations.

    Args:
        categories: Filter by category keywords (e.g., ['a100', 'h100', 'lora', 'zero'])
        config_names: Specific configuration names to run
        debug: Enable debug output

    Returns:
        DataFrame with validation results
    """
    configs_to_run = VALIDATION_CONFIGS.copy()

    # Filter by specific config names
    if config_names:
        configs_to_run = {k: v for k, v in configs_to_run.items() if k in config_names}

    # Filter by categories
    if categories:
        filtered = {}
        for name, config in configs_to_run.items():
            for cat in categories:
                if cat.lower() in name.lower() or cat.lower() in config.notes.lower():
                    filtered[name] = config
                    break
        configs_to_run = filtered

    if debug:
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE VALIDATION - {len(configs_to_run)} configs")
        print(f"{'='*60}\n")

    results = []
    for name, config in configs_to_run.items():
        result = run_validation_config(config, debug)
        result.config_name = name
        results.append({
            'config_name': name,
            'name': config.name,
            'model': config.model,
            'hardware': config.hardware,
            'num_gpus': config.num_gpus,
            'method': config.method,
            'training_type': config.training_type,
            'passed': result.passed,
            'mfu': result.mfu,
            'mfu_expected': result.mfu_expected,
            'mfu_valid': result.mfu_valid,
            'memory_gb': result.memory_gb,
            'memory_valid': result.memory_valid,
            'tokens_per_second': result.tokens_per_second,
            'step_time_ms': result.step_time_ms,
            'error': result.error,
        })

    df = pd.DataFrame(results)

    if debug:
        print(f"\n{'='*60}")
        print("VALIDATION SUMMARY")
        print(f"{'='*60}")
        passed = df['passed'].sum()
        total = len(df)
        print(f"Passed: {passed}/{total} ({passed/total*100:.0f}%)")
        print(f"Mean MFU: {df['mfu'].mean():.1%}")
        print(f"Mean Memory: {df['memory_gb'].mean():.1f}GB")

        failed = df[~df['passed']]
        if len(failed) > 0:
            print(f"\nFailed configs:")
            for _, row in failed.iterrows():
                reason = row['error'] if row['error'] else (
                    "MFU out of range" if not row['mfu_valid'] else "Memory out of range"
                )
                print(f"  - {row['name']}: {reason}")

    return df


def analyze_correlations(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze correlations and patterns in validation results.

    Returns:
        Dict with analysis results including:
        - MFU statistics by hardware
        - MFU statistics by method
        - MFU statistics by training type
        - Overall statistics
    """
    analysis = {}

    # MFU by hardware
    hardware_stats = {}
    for _, row in df.iterrows():
        hw = row['hardware']
        if hw not in hardware_stats:
            hardware_stats[hw] = {'mfu': [], 'tps': [], 'memory': []}
        hardware_stats[hw]['mfu'].append(row['mfu'])
        hardware_stats[hw]['tps'].append(row['tokens_per_second'])
        hardware_stats[hw]['memory'].append(row['memory_gb'])

    analysis['mfu_by_hardware'] = {
        k: {
            'mean_mfu': np.mean(v['mfu']),
            'std_mfu': np.std(v['mfu']),
            'mean_tps': np.mean(v['tps']),
            'mean_memory': np.mean(v['memory']),
            'count': len(v['mfu']),
        }
        for k, v in hardware_stats.items()
    }

    # MFU by method
    method_stats = {}
    for _, row in df.iterrows():
        method = row['method']
        if method not in method_stats:
            method_stats[method] = {'mfu': [], 'memory': []}
        method_stats[method]['mfu'].append(row['mfu'])
        method_stats[method]['memory'].append(row['memory_gb'])

    analysis['mfu_by_method'] = {
        k: {
            'mean_mfu': np.mean(v['mfu']),
            'mean_memory': np.mean(v['memory']),
            'count': len(v['mfu']),
        }
        for k, v in method_stats.items()
    }

    # MFU by training type
    training_stats = {}
    for _, row in df.iterrows():
        ttype = row['training_type']
        if ttype not in training_stats:
            training_stats[ttype] = {'mfu': []}
        training_stats[ttype]['mfu'].append(row['mfu'])

    analysis['mfu_by_training_type'] = {
        k: {'mean_mfu': np.mean(v['mfu']), 'count': len(v['mfu'])}
        for k, v in training_stats.items()
    }

    # MFU by model size
    model_stats = {}
    for _, row in df.iterrows():
        model = row['model']
        # Extract size from model name (e.g., "7B", "13B", "70B")
        size = "unknown"
        if "7B" in model or "8B" in model:
            size = "7-8B"
        elif "13B" in model:
            size = "13B"
        elif "70B" in model:
            size = "70B"
        elif "8x7B" in model or "MoE" in model:
            size = "MoE"

        if size not in model_stats:
            model_stats[size] = {'mfu': [], 'tps': []}
        model_stats[size]['mfu'].append(row['mfu'])
        model_stats[size]['tps'].append(row['tokens_per_second'])

    analysis['mfu_by_model_size'] = {
        k: {
            'mean_mfu': np.mean(v['mfu']),
            'mean_tps': np.mean(v['tps']),
            'count': len(v['mfu']),
        }
        for k, v in model_stats.items()
    }

    # Overall statistics
    analysis['overall'] = {
        'mean_mfu': df['mfu'].mean(),
        'std_mfu': df['mfu'].std(),
        'median_mfu': df['mfu'].median(),
        'mean_memory_gb': df['memory_gb'].mean(),
        'mean_tps': df['tokens_per_second'].mean(),
        'pass_rate': df['passed'].mean(),
        'total_configs': len(df),
    }

    return analysis


def run_quick_validation(debug: bool = False) -> pd.DataFrame:
    """
    Run quick validation on a subset of configurations.

    Uses small models and configurations for fast testing.
    """
    quick_configs = [
        'llama_7b_a100_80gb',
        'llama_7b_h100',
        'llama_7b_lora',
        'llama_13b_qlora',
        'llama_7b_dpo',
    ]
    return run_comprehensive_validation(config_names=quick_configs, debug=debug)


def list_validation_configs() -> pd.DataFrame:
    """List all available validation configurations."""
    data = []
    for name, config in VALIDATION_CONFIGS.items():
        data.append({
            'name': name,
            'description': config.name,
            'model': config.model,
            'hardware': config.hardware,
            'num_gpus': config.num_gpus,
            'method': config.method,
            'training_type': config.training_type,
            'expected_mfu': f"{config.expected_mfu_range[0]:.0%}-{config.expected_mfu_range[1]:.0%}",
        })
    return pd.DataFrame(data)


def get_validation_config(name: str) -> ValidationConfig:
    """Get a specific validation configuration by name."""
    if name not in VALIDATION_CONFIGS:
        raise ValueError(f"Unknown config: {name}. Use list_validation_configs() to see available.")
    return VALIDATION_CONFIGS[name]


# ============================================================================
# CATEGORY-BASED ERROR ANALYSIS
# ============================================================================

def run_category_analysis(
    df: Optional[pd.DataFrame] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Run detailed category-based error analysis.

    Analyzes systematic biases by:
    - Hardware generation (Ampere vs Hopper vs Blackwell)
    - Model size (7B vs 70B vs 175B+)
    - Parallelism strategy (DP-only vs TP+PP vs ZeRO)
    - Training method (full vs LoRA vs QLoRA)
    - Training type (SFT vs DPO vs PPO)

    Args:
        df: Validation results DataFrame (runs validation if None)
        debug: Enable debug output

    Returns:
        Dict with category-based analysis results
    """
    if df is None:
        df = run_comprehensive_validation(debug=debug)

    analysis = {}

    # Hardware generation analysis
    hardware_generations = {
        'Ampere': ['A100', 'A100_40GB', 'A100_80GB'],
        'Hopper': ['H100', 'H200', 'GH200'],
        'Blackwell': ['B100', 'B200', 'GB200'],
        'Volta': ['V100'],
        'Ada': ['L40S', 'RTX4090', 'RTX3090'],
        'TPU': ['TPU', 'TPUv4', 'TPUv5'],
        'AMD': ['MI300X', 'MI250X'],
        'Intel': ['Gaudi'],
    }

    def get_generation(hardware: str) -> str:
        for gen, prefixes in hardware_generations.items():
            for prefix in prefixes:
                if prefix.lower() in hardware.lower():
                    return gen
        return 'Other'

    df_copy = df.copy()
    df_copy['hw_generation'] = df_copy['hardware'].apply(get_generation)

    gen_stats = df_copy.groupby('hw_generation').agg({
        'mfu': ['mean', 'std', 'count'],
        'passed': 'mean',
        'tokens_per_second': 'mean',
    }).round(3)
    gen_stats.columns = ['mean_mfu', 'std_mfu', 'count', 'pass_rate', 'mean_tps']
    analysis['by_hardware_generation'] = gen_stats.to_dict('index')

    # Model size analysis
    def get_size_bucket(model: str) -> str:
        model_lower = model.lower()
        if '1b' in model_lower or '3b' in model_lower:
            return '1-3B'
        elif '7b' in model_lower or '8b' in model_lower:
            return '7-8B'
        elif '13b' in model_lower:
            return '13B'
        elif '27b' in model_lower or '34b' in model_lower:
            return '27-34B'
        elif '70b' in model_lower or '72b' in model_lower:
            return '70-72B'
        elif '175b' in model_lower or '180b' in model_lower:
            return '175B+'
        elif '8x' in model_lower or 'moe' in model_lower:
            return 'MoE'
        return 'Other'

    df_copy['size_bucket'] = df_copy['model'].apply(get_size_bucket)

    size_stats = df_copy.groupby('size_bucket').agg({
        'mfu': ['mean', 'std', 'count'],
        'passed': 'mean',
        'memory_gb': 'mean',
    }).round(3)
    size_stats.columns = ['mean_mfu', 'std_mfu', 'count', 'pass_rate', 'mean_memory']
    analysis['by_model_size'] = size_stats.to_dict('index')

    # Training method analysis
    method_stats = df_copy.groupby('method').agg({
        'mfu': ['mean', 'std', 'count'],
        'passed': 'mean',
        'memory_gb': 'mean',
    }).round(3)
    method_stats.columns = ['mean_mfu', 'std_mfu', 'count', 'pass_rate', 'mean_memory']
    analysis['by_training_method'] = method_stats.to_dict('index')

    # Training type analysis
    type_stats = df_copy.groupby('training_type').agg({
        'mfu': ['mean', 'std', 'count'],
        'passed': 'mean',
    }).round(3)
    type_stats.columns = ['mean_mfu', 'std_mfu', 'count', 'pass_rate']
    analysis['by_training_type'] = type_stats.to_dict('index')

    # Identify systematic biases
    analysis['systematic_biases'] = []

    # Check for hardware generation biases
    for gen, stats in analysis['by_hardware_generation'].items():
        if stats['count'] >= 3:
            if stats['pass_rate'] < 0.5:
                analysis['systematic_biases'].append({
                    'category': 'hardware_generation',
                    'value': gen,
                    'issue': f"Low pass rate ({stats['pass_rate']:.0%}) for {gen} GPUs",
                    'suggestion': f"Review efficiency factors for {gen} architecture"
                })

    # Check for model size biases
    for size, stats in analysis['by_model_size'].items():
        if stats['count'] >= 3:
            if stats['pass_rate'] < 0.5:
                analysis['systematic_biases'].append({
                    'category': 'model_size',
                    'value': size,
                    'issue': f"Low pass rate ({stats['pass_rate']:.0%}) for {size} models",
                    'suggestion': f"Review size scaling factors for {size} range"
                })

    # Check for method biases
    for method, stats in analysis['by_training_method'].items():
        if stats['count'] >= 2:
            if stats['pass_rate'] < 0.5:
                analysis['systematic_biases'].append({
                    'category': 'training_method',
                    'value': method,
                    'issue': f"Low pass rate ({stats['pass_rate']:.0%}) for {method} training",
                    'suggestion': f"Review method efficiency for {method}"
                })

    if debug:
        print("\n" + "=" * 60)
        print("CATEGORY-BASED ERROR ANALYSIS")
        print("=" * 60)

        print("\nBy Hardware Generation:")
        for gen, stats in analysis['by_hardware_generation'].items():
            print(f"  {gen}: MFU={stats['mean_mfu']:.1%} "
                  f"(pass={stats['pass_rate']:.0%}, n={stats['count']})")

        print("\nBy Model Size:")
        for size, stats in analysis['by_model_size'].items():
            print(f"  {size}: MFU={stats['mean_mfu']:.1%} "
                  f"(pass={stats['pass_rate']:.0%}, n={stats['count']})")

        print("\nBy Training Method:")
        for method, stats in analysis['by_training_method'].items():
            print(f"  {method}: MFU={stats['mean_mfu']:.1%} "
                  f"(pass={stats['pass_rate']:.0%}, n={stats['count']})")

        if analysis['systematic_biases']:
            print("\nSystematic Biases Detected:")
            for bias in analysis['systematic_biases']:
                print(f"  - {bias['issue']}")
                print(f"    Suggestion: {bias['suggestion']}")

    return analysis


def get_differential_validation_results(
    baseline_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compare two validation runs to measure improvement.

    Useful for validating that calibration or code changes
    improve accuracy.

    Args:
        baseline_df: Baseline validation results
        comparison_df: Comparison validation results

    Returns:
        Dict with improvement metrics
    """
    # Find common configs
    common_configs = set(baseline_df['config_name']) & set(comparison_df['config_name'])

    if len(common_configs) == 0:
        raise ValueError("No common configurations found between baseline and comparison")

    baseline = baseline_df[baseline_df['config_name'].isin(common_configs)]
    comparison = comparison_df[comparison_df['config_name'].isin(common_configs)]

    # Calculate improvement metrics
    baseline_pass_rate = baseline['passed'].mean()
    comparison_pass_rate = comparison['passed'].mean()

    baseline_mean_mfu = baseline['mfu'].mean()
    comparison_mean_mfu = comparison['mfu'].mean()

    # Per-config improvement
    merged = baseline.merge(
        comparison,
        on='config_name',
        suffixes=('_baseline', '_comparison')
    )

    improved_configs = []
    degraded_configs = []

    for _, row in merged.iterrows():
        baseline_valid = row['mfu_valid_baseline']
        comparison_valid = row['mfu_valid_comparison']

        if comparison_valid and not baseline_valid:
            improved_configs.append(row['config_name'])
        elif baseline_valid and not comparison_valid:
            degraded_configs.append(row['config_name'])

    return {
        'n_common_configs': len(common_configs),
        'baseline_pass_rate': baseline_pass_rate,
        'comparison_pass_rate': comparison_pass_rate,
        'pass_rate_improvement': comparison_pass_rate - baseline_pass_rate,
        'baseline_mean_mfu': baseline_mean_mfu,
        'comparison_mean_mfu': comparison_mean_mfu,
        'improved_configs': improved_configs,
        'degraded_configs': degraded_configs,
        'net_improvement': len(improved_configs) - len(degraded_configs),
    }


def validate_scaling_behavior(debug: bool = False) -> Dict[str, Any]:
    """
    Validate relative accuracy (scaling behavior).

    Tests that the simulator correctly predicts:
    1. Hardware scaling: H100 should be ~2x faster than A100
    2. GPU scaling: 64 GPUs should scale appropriately vs 8 GPUs
    3. Parallelism tradeoffs: TP=8 vs TP=4 overhead should be reasonable

    Returns:
        Dict with scaling validation results
    """
    results = {}

    # Test 1: Hardware scaling (A100 vs H100)
    a100_configs = [c for c in VALIDATION_CONFIGS if 'a100' in c.lower() and '7b' in c.lower()]
    h100_configs = [c for c in VALIDATION_CONFIGS if 'h100' in c.lower() and '7b' in c.lower()]

    if a100_configs and h100_configs:
        a100_df = run_comprehensive_validation(config_names=a100_configs[:3], debug=False)
        h100_df = run_comprehensive_validation(config_names=h100_configs[:3], debug=False)

        a100_tps = a100_df['tokens_per_second'].mean()
        h100_tps = h100_df['tokens_per_second'].mean()
        speedup = h100_tps / a100_tps if a100_tps > 0 else 0

        # Expected: H100 should be 1.5-3x faster than A100
        expected_range = (1.5, 3.0)
        valid = expected_range[0] <= speedup <= expected_range[1]

        results['hardware_scaling'] = {
            'a100_tps': a100_tps,
            'h100_tps': h100_tps,
            'speedup': speedup,
            'expected_range': expected_range,
            'valid': valid,
        }

    # Test 2: GPU scaling
    small_scale_configs = [c for c in VALIDATION_CONFIGS if VALIDATION_CONFIGS[c].num_gpus <= 8]
    large_scale_configs = [c for c in VALIDATION_CONFIGS if VALIDATION_CONFIGS[c].num_gpus >= 32]

    if small_scale_configs and large_scale_configs:
        small_df = run_comprehensive_validation(
            config_names=small_scale_configs[:3], debug=False
        )
        large_df = run_comprehensive_validation(
            config_names=large_scale_configs[:3], debug=False
        )

        small_mfu = small_df['mfu'].mean()
        large_mfu = large_df['mfu'].mean()

        # Large scale should have similar MFU (within 20%)
        mfu_ratio = large_mfu / small_mfu if small_mfu > 0 else 0
        valid = 0.7 <= mfu_ratio <= 1.1  # Allow some efficiency loss

        results['gpu_scaling'] = {
            'small_scale_mfu': small_mfu,
            'large_scale_mfu': large_mfu,
            'mfu_ratio': mfu_ratio,
            'valid': valid,
        }

    if debug:
        print("\n" + "=" * 60)
        print("SCALING BEHAVIOR VALIDATION")
        print("=" * 60)

        if 'hardware_scaling' in results:
            hs = results['hardware_scaling']
            print(f"\nHardware Scaling (A100 vs H100):")
            print(f"  A100 TPS: {hs['a100_tps']:.0f}")
            print(f"  H100 TPS: {hs['h100_tps']:.0f}")
            print(f"  Speedup: {hs['speedup']:.2f}x")
            print(f"  Expected: {hs['expected_range'][0]:.1f}x - {hs['expected_range'][1]:.1f}x")
            print(f"  Valid: {'YES' if hs['valid'] else 'NO'}")

        if 'gpu_scaling' in results:
            gs = results['gpu_scaling']
            print(f"\nGPU Scaling:")
            print(f"  Small scale MFU: {gs['small_scale_mfu']:.1%}")
            print(f"  Large scale MFU: {gs['large_scale_mfu']:.1%}")
            print(f"  MFU ratio: {gs['mfu_ratio']:.2f}")
            print(f"  Valid: {'YES' if gs['valid'] else 'NO'}")

    return results
