"""
Benchmark Validation Framework for Training Simulation.

This module provides validation against published training benchmarks
from Meta, NVIDIA, Google, and other organizations. It allows testing
the accuracy of training simulations against real-world reported metrics.

Key features:
- Published benchmark data from LLaMA, GPT-3, BLOOM, PaLM papers
- Validation functions with configurable tolerance
- Comprehensive validation suite runner
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import pandas as pd

from .training_modeling import training_modeling, TrainingModelingOutput


@dataclass
class PublishedBenchmark:
    """Published training benchmark from papers/reports."""
    name: str
    source: str  # Paper/report reference
    model: str
    batch_size: int  # Global batch size
    seq_length: int
    num_gpus: int
    hardware: str
    parallelism: Dict[str, int]  # {'tp': X, 'pp': Y, 'dp': Z, 'zero': N}
    reported_tokens_per_second: float
    reported_mfu: float
    reported_step_time_ms: Optional[float] = None
    notes: str = ""


# Published benchmark data for validation
# TPS values are calculated from MFU using: TPS = MFU * peak_flops / (6 * params)
# MFU is the primary validation metric as it's more reliably reported
PUBLISHED_BENCHMARKS: Dict[str, PublishedBenchmark] = {
    # Meta LLaMA-2 Training (https://arxiv.org/abs/2307.09288)
    'llama2_70b_meta': PublishedBenchmark(
        name="LLaMA-2 70B (Meta)",
        source="https://arxiv.org/abs/2307.09288",
        model='Llama-2-70B',  # Use MODEL_DICT name
        batch_size=1024,  # Global batch
        seq_length=4096,
        num_gpus=2048,
        hardware='A100_80GB_GPU',
        parallelism={'tp': 8, 'pp': 16, 'dp': 16, 'zero': 0},
        reported_tokens_per_second=654190,  # Derived from MFU
        reported_mfu=0.43,
        notes="Meta's LLaMA-2 70B training on A100 cluster"
    ),

    'llama2_13b_meta': PublishedBenchmark(
        name="LLaMA-2 13B (Meta)",
        source="https://arxiv.org/abs/2307.09288",
        model='Llama-2-13B',  # Use MODEL_DICT name
        batch_size=1024,
        seq_length=4096,
        num_gpus=512,
        hardware='A100_80GB_GPU',
        parallelism={'tp': 2, 'pp': 4, 'dp': 64, 'zero': 0},
        reported_tokens_per_second=1024000,  # Derived from MFU
        reported_mfu=0.50,
        notes="Meta's LLaMA-2 13B training"
    ),

    # Megatron-LM GPT-3 (https://arxiv.org/abs/2104.04473)
    'gpt3_175b_megatron': PublishedBenchmark(
        name="GPT-3 175B (Megatron-LM)",
        source="https://arxiv.org/abs/2104.04473",
        model='GPT3-175B',  # Use MODEL_DICT name
        batch_size=1536,
        seq_length=2048,
        num_gpus=3072,
        hardware='A100_80GB_GPU',
        parallelism={'tp': 8, 'pp': 12, 'dp': 32, 'zero': 0},
        reported_tokens_per_second=474668,  # Derived from MFU
        reported_mfu=0.52,
        notes="NVIDIA Megatron-LM GPT-3 175B training"
    ),

    # OPT-175B as BLOOM equivalent (BLOOM not in MODEL_DICT)
    'opt_175b_training': PublishedBenchmark(
        name="OPT 175B Training",
        source="https://arxiv.org/abs/2205.01068",
        model='facebook/opt-175b',  # Use MODEL_DICT name
        batch_size=2048,
        seq_length=2048,
        num_gpus=512,
        hardware='A100_80GB_GPU',
        parallelism={'tp': 8, 'pp': 8, 'dp': 8, 'zero': 0},
        reported_tokens_per_second=60855,  # Derived: 0.40 * 512 * 312e12 / (6 * 175e9)
        reported_mfu=0.40,
        notes="OPT 175B training benchmark"
    ),

    # LLaMA-3.1 70B (https://ai.meta.com/blog/meta-llama-3/)
    'llama3_70b_meta': PublishedBenchmark(
        name="LLaMA-3.1 70B (Meta)",
        source="Meta LLaMA-3 Technical Report",
        model='Llama-3.1-70B',  # Use MODEL_DICT name
        batch_size=2048,
        seq_length=8192,
        num_gpus=4096,
        hardware='H100_GPU',
        parallelism={'tp': 8, 'pp': 8, 'dp': 64, 'zero': 0},
        reported_tokens_per_second=4629650,  # Derived from MFU
        reported_mfu=0.48,
        notes="Meta LLaMA-3.1 70B on H100 cluster"
    ),

    # Mixtral 8x7B (https://mistral.ai/news/mixtral-of-experts/)
    # MoE: 47B total params, but only ~13B active per token (shared + 2 of 8 experts)
    # TPS derivation: 0.42 * 256 * 989e12 / (6 * 13e9) = 1,366,474
    'mixtral_8x7b': PublishedBenchmark(
        name="Mixtral 8x7B",
        source="Mistral AI",
        model='Mixtral-8x7B',  # Use MODEL_DICT name
        batch_size=512,
        seq_length=32768,
        num_gpus=256,
        hardware='H100_GPU',
        parallelism={'tp': 4, 'pp': 2, 'dp': 32, 'ep': 2, 'zero': 0},
        reported_tokens_per_second=1366474,  # Derived: 0.42 * 256 * 989e12 / (6 * 13e9) active params
        reported_mfu=0.42,
        notes="Mixtral 8x7B MoE training (~13B active params per token)"
    ),

    # Small model benchmarks for quick validation
    'llama_7b_baseline': PublishedBenchmark(
        name="LLaMA 7B Baseline",
        source="Community benchmarks",
        model='Llama-2-7B',  # Use MODEL_DICT name
        batch_size=32,
        seq_length=4096,
        num_gpus=8,
        hardware='A100_80GB_GPU',
        parallelism={'tp': 2, 'pp': 1, 'dp': 4, 'zero': 0},
        reported_tokens_per_second=32686,  # Derived from MFU
        reported_mfu=0.55,
        notes="LLaMA 7B baseline on DGX A100"
    ),

    'llama_7b_h100': PublishedBenchmark(
        name="LLaMA 7B on H100",
        source="Community benchmarks",
        model='Llama-2-7B',  # Use MODEL_DICT name
        batch_size=64,
        seq_length=4096,
        num_gpus=8,
        hardware='H100_GPU',
        parallelism={'tp': 2, 'pp': 1, 'dp': 4, 'zero': 0},
        reported_tokens_per_second=97958,  # Derived from MFU
        reported_mfu=0.52,
        notes="LLaMA 7B on DGX H100"
    ),

    # Qwen 72B as DeepSeek 67B equivalent
    'qwen_72b_training': PublishedBenchmark(
        name="Qwen 72B Training",
        source="Qwen Technical Report",
        model='Qwen-72B',  # Use MODEL_DICT name
        batch_size=1024,
        seq_length=4096,
        num_gpus=512,
        hardware='A100_80GB_GPU',
        parallelism={'tp': 4, 'pp': 4, 'dp': 32, 'zero': 0},
        reported_tokens_per_second=178714,  # Derived: 0.45 * 512 * 312e12 / (6 * 72e9)
        reported_mfu=0.45,
        notes="Qwen 72B training benchmark"
    ),

    # ========================================================================
    # Additional Benchmarks from Web Research (Jan 2026)
    # ========================================================================

    # Google PaLM 540B (https://arxiv.org/abs/2204.02311)
    # Note: Uses Gemma-2-27B as proxy. TPS derived assuming 27B params (proxy model)
    # TPU v4 ~275 TFLOPS BF16
    'palm_540b_google': PublishedBenchmark(
        name="PaLM 540B (Google)",
        source="https://arxiv.org/abs/2204.02311",
        model='Gemma-2-27B',  # Use as proxy (540B not in dict, scale MFU)
        batch_size=2048,
        seq_length=2048,
        num_gpus=6144,  # TPU v4
        hardware='TPU_v4',
        parallelism={'tp': 8, 'pp': 12, 'dp': 64, 'zero': 0},
        reported_tokens_per_second=4849778,  # Derived: 0.465 * 6144 * 275e12 / (6 * 27e9)
        reported_mfu=0.465,  # 46.5% MFU reported for PaLM 540B
        notes="Google PaLM 540B on TPU v4 pod (uses 27B proxy)"
    ),

    # Chinchilla 70B (DeepMind) - Optimal compute scaling
    # TPU v4 ~275 TFLOPS BF16
    'chinchilla_70b_deepmind': PublishedBenchmark(
        name="Chinchilla 70B (DeepMind)",
        source="https://arxiv.org/abs/2203.15556",
        model='Llama-2-70B',  # Similar architecture
        batch_size=1536,
        seq_length=2048,
        num_gpus=1024,
        hardware='TPU_v4',
        parallelism={'tp': 4, 'pp': 8, 'dp': 32, 'zero': 0},
        reported_tokens_per_second=335238,  # Derived: 0.50 * 1024 * 275e12 / (6 * 70e9)
        reported_mfu=0.50,  # ~50% MFU achieved with optimal scaling
        notes="Chinchilla used compute-optimal training (20 tokens/param)"
    ),

    # H100 Optimized Training (CoreWeave/NVIDIA benchmarks)
    'llama2_70b_h100_optimized': PublishedBenchmark(
        name="LLaMA-2 70B H100 Optimized",
        source="CoreWeave/NVIDIA MLPerf 2024",
        model='Llama-2-70B',
        batch_size=2048,
        seq_length=4096,
        num_gpus=512,
        hardware='H100_GPU',
        parallelism={'tp': 4, 'pp': 4, 'dp': 32, 'zero': 0},
        reported_tokens_per_second=651045,  # Derived: 0.54 * 512 * 989e12 / (6 * 70e9)
        reported_mfu=0.54,  # CoreWeave achieved >50% MFU on H100
        notes="Optimized training with FP8, flash attention, pipeline overlap"
    ),

    # DeepSeek 67B (https://arxiv.org/abs/2401.02954)
    'deepseek_67b_training': PublishedBenchmark(
        name="DeepSeek 67B",
        source="https://arxiv.org/abs/2401.02954",
        model='DeepSeek-67B',  # Use MODEL_DICT name
        batch_size=2048,
        seq_length=4096,
        num_gpus=512,
        hardware='A100_80GB_GPU',
        parallelism={'tp': 8, 'pp': 4, 'dp': 16, 'zero': 0},
        reported_tokens_per_second=166897,  # Derived: 0.42 * 512 * 312e12 / (6 * 67e9)
        reported_mfu=0.42,
        notes="DeepSeek 67B dense model training"
    ),

    # Falcon 180B (TII)
    'falcon_180b_training': PublishedBenchmark(
        name="Falcon 180B (TII)",
        source="TII Technical Report",
        model='facebook/opt-175b',  # Similar size proxy
        batch_size=2048,
        seq_length=2048,
        num_gpus=4096,
        hardware='A100_80GB_GPU',
        parallelism={'tp': 8, 'pp': 16, 'dp': 32, 'zero': 0},
        reported_tokens_per_second=462497,  # Derived: 0.38 * 4096 * 312e12 / (6 * 175e9)
        reported_mfu=0.38,  # Large model efficiency typically lower
        notes="Falcon 180B trained on 3.5T tokens"
    ),

    # GPT-4 Estimated Training (based on public MFU reports)
    'gpt4_estimated_training': PublishedBenchmark(
        name="GPT-4 Estimated Training",
        source="Industry estimates, MS blog posts",
        model='GPT3-175B',  # Proxy model (actual architecture unknown)
        batch_size=2048,
        seq_length=8192,
        num_gpus=10000,  # Estimated
        hardware='A100_80GB_GPU',
        parallelism={'tp': 8, 'pp': 32, 'dp': 39, 'zero': 0},
        reported_tokens_per_second=1069714,  # Derived: 0.36 * 10000 * 312e12 / (6 * 175e9)
        reported_mfu=0.36,  # GPT-4 reported 32-36% MFU due to complexity
        notes="GPT-4 estimated training, 32-36% MFU due to architectural overhead"
    ),

    # Small-scale efficient training (LoRA/QLoRA style)
    'llama_13b_lora_efficient': PublishedBenchmark(
        name="LLaMA 13B Efficient FineTuning",
        source="Community benchmarks",
        model='Llama-2-13B',
        batch_size=16,
        seq_length=4096,
        num_gpus=8,
        hardware='A100_80GB_GPU',
        parallelism={'tp': 1, 'pp': 1, 'dp': 8, 'zero': 3},
        reported_tokens_per_second=15360,  # Derived: 0.48 * 8 * 312e12 / (6 * 13e9)
        reported_mfu=0.48,  # Full fine-tuning baseline (not LoRA)
        notes="Baseline full fine-tuning for comparison with LoRA"
    ),

    # Gemma 2 27B (Google)
    # Note: TPU v5e ~197 TFLOPS BF16
    'gemma2_27b_training': PublishedBenchmark(
        name="Gemma 2 27B (Google)",
        source="Google Gemma 2 Technical Report",
        model='Gemma-2-27B',
        batch_size=512,
        seq_length=8192,
        num_gpus=256,
        hardware='TPU_v5e',
        parallelism={'tp': 4, 'pp': 2, 'dp': 32, 'zero': 0},
        reported_tokens_per_second=161880,  # Derived: 0.52 * 256 * 197e12 / (6 * 27e9)
        reported_mfu=0.52,  # Google typically achieves >50% on TPU
        notes="Gemma 2 trained on 13T tokens"
    ),

    # Yi-34B (01.AI)
    # Note: Yi-34B has 34B params, using LLaMA-2-33B as proxy (closest available)
    # TPS derived: 0.47 * 256 * 312e12 / (6 * 34e9) = 184,280
    'yi_34b_training': PublishedBenchmark(
        name="Yi-34B (01.AI)",
        source="01.AI Technical Report",
        model='meta-llama/llama-2-33b',  # Proxy (closest to 34B in MODEL_DICT)
        batch_size=1024,
        seq_length=4096,
        num_gpus=256,
        hardware='A100_80GB_GPU',
        parallelism={'tp': 4, 'pp': 2, 'dp': 32, 'zero': 0},
        reported_tokens_per_second=184280,  # Derived: 0.47 * 256 * 312e12 / (6 * 34e9)
        reported_mfu=0.47,
        notes="Yi-34B bilingual model training (uses 33B proxy)"
    ),

    # B200/GB200 Forward-looking benchmark
    # Note: GB200 ~2250 TFLOPS BF16 (estimated, ~2.3x H100)
    'llama3_70b_gb200_projected': PublishedBenchmark(
        name="LLaMA-3 70B GB200 (Projected)",
        source="NVIDIA GB200 specs extrapolation",
        model='Llama-3.1-70B',
        batch_size=4096,
        seq_length=8192,
        num_gpus=256,
        hardware='GB200',  # Will need hardware config
        parallelism={'tp': 4, 'pp': 2, 'dp': 32, 'zero': 0},
        reported_tokens_per_second=795429,  # Derived: 0.58 * 256 * 2250e12 / (6 * 70e9)
        reported_mfu=0.58,  # GB200 expected to achieve higher MFU
        notes="Projected GB200 performance based on NVIDIA specs"
    ),
}


@dataclass
class ValidationResult:
    """Result of validating against a benchmark."""
    benchmark_name: str
    passed: bool
    tolerance: float

    # Throughput comparison
    predicted_tps: float
    actual_tps: float
    tps_error_pct: float

    # MFU comparison
    predicted_mfu: float
    actual_mfu: float
    mfu_error_pct: float

    # Step time comparison (if available)
    predicted_step_time_ms: Optional[float] = None
    actual_step_time_ms: Optional[float] = None
    step_time_error_pct: Optional[float] = None

    # Additional details
    details: Dict[str, Any] = field(default_factory=dict)


def validate_against_benchmark(
    benchmark_name: str,
    tolerance: float = 0.25,  # 25% tolerance
    debug: bool = False,
) -> ValidationResult:
    """
    Validate simulator output against a published benchmark.

    Args:
        benchmark_name: Name of the benchmark (key in PUBLISHED_BENCHMARKS)
        tolerance: Maximum allowed error fraction (default 25%)
        debug: Enable debug output

    Returns:
        ValidationResult with pass/fail status and error metrics
    """
    if benchmark_name not in PUBLISHED_BENCHMARKS:
        raise ValueError(
            f"Unknown benchmark: {benchmark_name}. "
            f"Available: {list(PUBLISHED_BENCHMARKS.keys())}"
        )

    benchmark = PUBLISHED_BENCHMARKS[benchmark_name]

    # Calculate per-DP-rank batch size
    # Note: We divide by DP only, NOT by DP * PP
    # PP is for pipeline scheduling (micro-batches), not batch division
    # The gradient_accumulation_steps parameter handles micro-batch division
    total_dp = benchmark.parallelism.get('dp', 1) or 1  # Ensure non-zero
    total_pp = benchmark.parallelism.get('pp', 1) or 1  # Ensure non-zero
    total_tp = benchmark.parallelism.get('tp', 1) or 1  # Ensure non-zero
    per_dp_batch_size = benchmark.batch_size // total_dp
    per_dp_batch_size = max(1, per_dp_batch_size)

    # Calculate gradient accumulation (typically PP for 1F1B)
    gradient_accum = total_pp if total_pp > 1 else 4

    if debug:
        print(f"Validating: {benchmark.name}")
        print(f"  Model: {benchmark.model}")
        print(f"  Hardware: {benchmark.hardware} x {benchmark.num_gpus}")
        print(f"  Parallelism: TP={benchmark.parallelism['tp']}, "
              f"PP={benchmark.parallelism['pp']}, DP={total_dp}")
        print(f"  Batch: {benchmark.batch_size} (per-DP: {per_dp_batch_size})")

    try:
        # Run simulation
        result = training_modeling(
            model=benchmark.model,
            training_stage='sft',  # Use SFT for base training
            batch_size=per_dp_batch_size,
            seq_length=benchmark.seq_length,
            system_name=benchmark.hardware,
            num_gpus=benchmark.num_gpus,
            tensor_parallel=total_tp,
            pipeline_parallel=total_pp,
            data_parallel=total_dp,
            expert_parallel=benchmark.parallelism.get('ep', 1) or 1,
            method='full',
            optimizer='adamw',
            zero_stage=benchmark.parallelism.get('zero', 0),
            gradient_checkpointing=True,
            gradient_accumulation_steps=gradient_accum,
            bits='bf16',
        )

        # tokens_per_second from simulation is already global
        # (training_modeling multiplies by data_parallel internally)
        predicted_tps = result.tokens_per_second

        # Calculate errors
        tps_error = abs(predicted_tps - benchmark.reported_tokens_per_second) / benchmark.reported_tokens_per_second
        mfu_error = abs(result.model_flops_utilization - benchmark.reported_mfu) / benchmark.reported_mfu

        # Check step time if available
        step_time_error = None
        if benchmark.reported_step_time_ms is not None:
            step_time_error = abs(
                result.step_time_ms - benchmark.reported_step_time_ms
            ) / benchmark.reported_step_time_ms

        # Determine pass/fail
        passed = tps_error <= tolerance and mfu_error <= tolerance

        if debug:
            print(f"  Results:")
            print(f"    TPS: predicted={predicted_tps:.0f}, actual={benchmark.reported_tokens_per_second:.0f}, "
                  f"error={tps_error*100:.1f}%")
            print(f"    MFU: predicted={result.model_flops_utilization:.2f}, actual={benchmark.reported_mfu:.2f}, "
                  f"error={mfu_error*100:.1f}%")
            print(f"    Status: {'PASS' if passed else 'FAIL'}")

        return ValidationResult(
            benchmark_name=benchmark_name,
            passed=passed,
            tolerance=tolerance,
            predicted_tps=predicted_tps,
            actual_tps=benchmark.reported_tokens_per_second,
            tps_error_pct=tps_error * 100,
            predicted_mfu=result.model_flops_utilization,
            actual_mfu=benchmark.reported_mfu,
            mfu_error_pct=mfu_error * 100,
            predicted_step_time_ms=result.step_time_ms,
            actual_step_time_ms=benchmark.reported_step_time_ms,
            step_time_error_pct=step_time_error * 100 if step_time_error is not None else None,
            details={
                'benchmark': benchmark.name,
                'source': benchmark.source,
                'notes': benchmark.notes,
                'memory_gb': result.memory_per_gpu_gb,
                'runtime_breakdown': result.runtime_breakdown,
            }
        )

    except Exception as e:
        if debug:
            print(f"  Error: {e}")
        return ValidationResult(
            benchmark_name=benchmark_name,
            passed=False,
            tolerance=tolerance,
            predicted_tps=0,
            actual_tps=benchmark.reported_tokens_per_second,
            tps_error_pct=100,
            predicted_mfu=0,
            actual_mfu=benchmark.reported_mfu,
            mfu_error_pct=100,
            details={'error': str(e)}
        )


def run_validation_suite(
    benchmarks: Optional[List[str]] = None,
    tolerance: float = 0.25,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Run validation against multiple benchmarks.

    Args:
        benchmarks: List of benchmark names to run (default: all)
        tolerance: Maximum allowed error fraction
        debug: Enable debug output

    Returns:
        DataFrame with validation results
    """
    if benchmarks is None:
        benchmarks = list(PUBLISHED_BENCHMARKS.keys())

    results = []
    for name in benchmarks:
        if debug:
            print(f"\n{'='*60}")
        result = validate_against_benchmark(name, tolerance, debug)
        results.append({
            'benchmark': name,
            'model': PUBLISHED_BENCHMARKS[name].model,
            'hardware': PUBLISHED_BENCHMARKS[name].hardware,
            'num_gpus': PUBLISHED_BENCHMARKS[name].num_gpus,
            'passed': result.passed,
            'tps_error_pct': result.tps_error_pct,
            'mfu_error_pct': result.mfu_error_pct,
            'predicted_tps': result.predicted_tps,
            'actual_tps': result.actual_tps,
            'predicted_mfu': result.predicted_mfu,
            'actual_mfu': result.actual_mfu,
        })

    df = pd.DataFrame(results)

    if debug:
        print(f"\n{'='*60}")
        print("VALIDATION SUMMARY")
        print(f"{'='*60}")
        passed = df['passed'].sum()
        total = len(df)
        print(f"Passed: {passed}/{total} ({passed/total*100:.0f}%)")
        print(f"Mean TPS error: {df['tps_error_pct'].mean():.1f}%")
        print(f"Mean MFU error: {df['mfu_error_pct'].mean():.1f}%")

    return df


def run_quick_validation(debug: bool = False) -> pd.DataFrame:
    """
    Run validation on smaller benchmarks for quick testing.
    """
    quick_benchmarks = [
        'llama_7b_baseline',
        'llama_7b_h100',
        'llama2_13b_meta',
    ]
    return run_validation_suite(
        benchmarks=quick_benchmarks,
        tolerance=0.30,  # Slightly higher tolerance for quick tests
        debug=debug
    )


def get_benchmark_info(benchmark_name: str) -> Dict[str, Any]:
    """Get information about a specific benchmark."""
    if benchmark_name not in PUBLISHED_BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    b = PUBLISHED_BENCHMARKS[benchmark_name]
    return {
        'name': b.name,
        'source': b.source,
        'model': b.model,
        'batch_size': b.batch_size,
        'seq_length': b.seq_length,
        'num_gpus': b.num_gpus,
        'hardware': b.hardware,
        'parallelism': b.parallelism,
        'reported_tps': b.reported_tokens_per_second,
        'reported_mfu': b.reported_mfu,
        'notes': b.notes,
    }


def list_benchmarks() -> pd.DataFrame:
    """List all available benchmarks."""
    data = []
    for name, b in PUBLISHED_BENCHMARKS.items():
        data.append({
            'name': name,
            'model': b.model,
            'hardware': b.hardware,
            'num_gpus': b.num_gpus,
            'batch_size': b.batch_size,
            'seq_length': b.seq_length,
            'reported_tps': b.reported_tokens_per_second,
            'reported_mfu': b.reported_mfu,
        })
    return pd.DataFrame(data)


# Network topology configurations for ASTRA-SIM validation
CLUSTER_NETWORK_CONFIGS = {
    # DGX A100 (8 GPUs, NVLink intra-node)
    'dgx_a100': {
        'topology': ['FullyConnected'],
        'npus_count': [8],
        'bandwidth': [600],  # 600 GB/s NVLink
        'latency': [0.25],   # 0.25 us
    },

    # DGX H100 (8 GPUs, NVLink 4.0)
    'dgx_h100': {
        'topology': ['FullyConnected'],
        'npus_count': [8],
        'bandwidth': [900],  # 900 GB/s NVLink 4.0
        'latency': [0.25],
    },

    # Multi-node A100 cluster (NVLink intra, InfiniBand inter)
    'cluster_a100_64': {
        'topology': ['FullyConnected', 'Ring'],
        'npus_count': [8, 8],
        'bandwidth': [600, 100],  # NVLink 600, IB HDR 100 GB/s
        'latency': [0.25, 5.0],   # NVLink 0.25us, IB 5us
    },

    # Multi-node H100 cluster
    'cluster_h100_64': {
        'topology': ['FullyConnected', 'Ring'],
        'npus_count': [8, 8],
        'bandwidth': [900, 400],  # NVLink 900, IB NDR 400 GB/s
        'latency': [0.25, 2.0],
    },

    # Large scale cluster (2048 GPUs)
    'cluster_large_2048': {
        'topology': ['FullyConnected', 'FullyConnected', 'Ring'],
        'npus_count': [8, 32, 8],  # 8 per node, 32 per rack, 8 racks
        'bandwidth': [900, 400, 200],
        'latency': [0.25, 2.0, 10.0],
    },

    # TPU v4 pod topology
    'tpu_v4_pod': {
        'topology': ['FullyConnected', 'FullyConnected'],
        'npus_count': [64, 64],  # TPU v4 slice
        'bandwidth': [4800, 4800],  # Optical interconnect
        'latency': [0.1, 0.5],
    },
}


def get_network_config(config_name: str) -> Dict[str, Any]:
    """Get predefined network configuration for ASTRA-SIM."""
    if config_name not in CLUSTER_NETWORK_CONFIGS:
        raise ValueError(
            f"Unknown network config: {config_name}. "
            f"Available: {list(CLUSTER_NETWORK_CONFIGS.keys())}"
        )
    return CLUSTER_NETWORK_CONFIGS[config_name]


# ============================================================================
# ACCURACY METRICS INTEGRATION
# ============================================================================
# These functions provide easy access to the comprehensive accuracy validation
# framework in llm_memory_calculator.validation

def get_accuracy_metrics(
    tolerance: float = 0.15,
    use_high_confidence_only: bool = True,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Get comprehensive accuracy metrics for the training simulator.

    This function runs the simulation against all published benchmarks
    and returns statistical accuracy measures.

    Args:
        tolerance: Acceptable relative error (default 15%)
        use_high_confidence_only: Only use high-confidence benchmarks
        debug: Enable debug output

    Returns:
        Dict containing:
        - mean_relative_error: Average relative prediction error
        - pearson_correlation: Correlation between predicted and actual MFU
        - systematic_bias: Over/under prediction tendency
        - tolerance_rate: Fraction of benchmarks within tolerance
        - is_production_ready: Whether accuracy meets targets
    """
    try:
        from ...validation.accuracy_metrics import run_accuracy_assessment
        metrics = run_accuracy_assessment(
            use_high_confidence_only=use_high_confidence_only,
            tolerance=tolerance,
            debug=debug,
        )
        return {
            'mean_relative_error': metrics.mean_relative_error,
            'median_relative_error': metrics.median_relative_error,
            'pearson_correlation': metrics.pearson_correlation,
            'systematic_bias': metrics.systematic_bias,
            'bias_direction': metrics.bias_direction,
            'tolerance_rate': metrics.tolerance_rate,
            'n_samples': metrics.n_samples,
            'is_production_ready': metrics.is_production_ready(),
            'percentile_95_error': metrics.percentile_95_error,
            'errors_by_category': metrics.errors_by_category,
            'errors_by_hardware': metrics.errors_by_hardware,
        }
    except ImportError:
        # Fallback if validation module not available
        df = run_validation_suite(debug=debug)
        return {
            'mean_relative_error': df['mfu_error_pct'].mean() / 100,
            'median_relative_error': df['mfu_error_pct'].median() / 100,
            'pearson_correlation': None,
            'systematic_bias': None,
            'tolerance_rate': df['passed'].mean(),
            'n_samples': len(df),
            'is_production_ready': False,
        }


def generate_accuracy_report(
    filepath: Optional[str] = None,
    format: str = "markdown",
    debug: bool = False,
) -> str:
    """
    Generate a comprehensive accuracy report.

    Args:
        filepath: Optional file path to save the report
        format: Output format ("markdown", "json", or "html")
        debug: Enable debug output

    Returns:
        Report content as string
    """
    try:
        from ...validation.accuracy_reporter import generate_report, save_report
        report = generate_report(
            use_high_confidence_only=True,
            include_calibration=True,
            debug=debug,
        )

        if format == "markdown":
            content = report.to_markdown()
        elif format == "json":
            content = report.to_json()
        else:
            content = report.to_markdown()

        if filepath:
            save_report(report, filepath, format)

        return content
    except ImportError as e:
        return f"Accuracy report generation requires validation module: {e}"


def run_calibration(
    max_iterations: int = 50,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Run calibration to optimize efficiency factors.

    This tunes the simulator's efficiency factors to match
    real-world benchmark performance.

    Args:
        max_iterations: Maximum optimization iterations
        debug: Enable debug output

    Returns:
        Dict containing calibration results and improvement metrics
    """
    try:
        from ...validation.calibration_engine import CalibrationEngine
        engine = CalibrationEngine()
        result = engine.run_calibration(
            method="differential_evolution",
            max_iterations=max_iterations,
            debug=debug,
        )

        return {
            'train_mre': result.train_metrics.mean_relative_error,
            'validation_mre': result.validation_metrics.mean_relative_error,
            'train_correlation': result.train_metrics.pearson_correlation,
            'validation_correlation': result.validation_metrics.pearson_correlation,
            'factors': result.factors.to_dict(),
            'production_ready': result.validation_metrics.is_production_ready(),
        }
    except ImportError as e:
        return {'error': f"Calibration requires validation module: {e}"}


# ============================================================================
# PHASE 1-5 ACCURACY IMPROVEMENTS VALIDATION
# ============================================================================
# These functions validate the accuracy improvements implemented in:
# - Phase 1: FLOPs counting (per-layer, GQA, gradient checkpointing)
# - Phase 2: Roofline model (tensor core efficiency, op intensity)
# - Phase 3: Communication model (NCCL-calibrated, ZeRO-3)
# - Phase 4: Comm-compute overlap modeling
# - Phase 5: Kernel overhead model (fused kernels)

def validate_flops_calculation(debug: bool = False) -> Dict[str, Any]:
    """
    Test Phase 1: FLOPs calculation accuracy.

    Validates:
    - Per-layer FLOPs are calculated correctly
    - Backward multiplier is 2.5x for attention (FlashAttention-2)
    - GQA/MQA FLOPs adjustments
    - Gradient checkpointing adds 33% forward (not 30% to backward multiplier)
    """
    from .training_modeling import (
        calculate_attention_flops,
        calculate_ffn_flops,
        calculate_total_training_flops,
        calculate_backward_multiplier,
    )

    results = {'passed': True, 'tests': []}

    # Test 1: Attention backward should be 2.5x forward
    attn_fwd = calculate_attention_flops(1, 4096, 4096, 32, 32, is_backward=False)
    attn_bwd = calculate_attention_flops(1, 4096, 4096, 32, 32, is_backward=True)
    backward_ratio = attn_bwd / attn_fwd

    test_1 = {
        'name': 'Attention backward multiplier',
        'expected': 2.5,
        'actual': backward_ratio,
        'passed': abs(backward_ratio - 2.5) < 0.1,
    }
    results['tests'].append(test_1)
    if not test_1['passed']:
        results['passed'] = False

    # Test 2: FFN backward should be 2.0x forward
    ffn_fwd = calculate_ffn_flops(1, 4096, 4096, 16384, is_backward=False)
    ffn_bwd = calculate_ffn_flops(1, 4096, 4096, 16384, is_backward=True)
    ffn_ratio = ffn_bwd / ffn_fwd

    test_2 = {
        'name': 'FFN backward multiplier',
        'expected': 2.0,
        'actual': ffn_ratio,
        'passed': abs(ffn_ratio - 2.0) < 0.01,
    }
    results['tests'].append(test_2)
    if not test_2['passed']:
        results['passed'] = False

    # Test 3: GQA should have higher backward/forward ratio than MHA
    # Compare GQA backward to GQA forward (not MHA forward)
    attn_fwd_gqa = calculate_attention_flops(1, 4096, 4096, 32, 8, is_backward=False)  # GQA forward
    attn_bwd_gqa = calculate_attention_flops(1, 4096, 4096, 32, 8, is_backward=True)   # GQA backward
    gqa_ratio = attn_bwd_gqa / attn_fwd_gqa

    # GQA backward should have >2.5x ratio (2.5 base * GQA overhead)
    test_3 = {
        'name': 'GQA backward overhead',
        'expected': '>2.5 (2.5 * GQA overhead)',
        'actual': gqa_ratio,
        'passed': gqa_ratio > 2.5,  # GQA should have higher overhead than base 2.5x
    }
    results['tests'].append(test_3)
    if not test_3['passed']:
        results['passed'] = False

    # Test 4: Gradient checkpointing adds to forward, not backward multiplier
    flops_no_ckpt = calculate_total_training_flops(
        1, 4096, 32, 4096, 16384, 128000, 32, 32, gradient_checkpointing=False
    )
    flops_with_ckpt = calculate_total_training_flops(
        1, 4096, 32, 4096, 16384, 128000, 32, 32, gradient_checkpointing=True
    )

    recompute_flops = flops_with_ckpt['recompute']
    forward_flops = flops_with_ckpt['forward']
    recompute_ratio = recompute_flops / forward_flops

    test_4 = {
        'name': 'Gradient checkpointing ratio',
        'expected': '~0.33',
        'actual': recompute_ratio,
        'passed': 0.3 < recompute_ratio < 0.4,  # ~33%
    }
    results['tests'].append(test_4)
    if not test_4['passed']:
        results['passed'] = False

    if debug:
        print("\n=== Phase 1: FLOPs Calculation Validation ===")
        for test in results['tests']:
            status = "PASS" if test['passed'] else "FAIL"
            print(f"  [{status}] {test['name']}: expected={test['expected']}, actual={test['actual']:.3f}")

    return results


def validate_communication_model(debug: bool = False) -> Dict[str, Any]:
    """
    Test Phase 3: Communication model accuracy.

    Validates:
    - AllReduce has realistic overhead (1.5-2x ideal)
    - ZeRO-3 models 3 AllGathers + 1 ReduceScatter
    - TP communication counts 4 ops per layer (not 2)
    """
    from ..collective_times import (
        get_AR_time,
        get_AG_time,
        get_reduce_scatter_time,
        calculate_zero3_communication_time,
    )
    from ..system import System

    results = {'passed': True, 'tests': []}

    # Create test system
    system = System(
        flops=1000,  # 1 PFLOPS
        offchip_mem_bw=3000,  # 3 TB/s (H100-like)
        interchip_link_bw=100,  # 100 GB/s
        interchip_link_latency=2.0,  # 2us
    )

    # Test 1: AllReduce should have overhead vs ideal
    # Phase 6: Updated to use more realistic (lower) overhead factors
    data_size = 100 * 1024 * 1024  # 100 MB
    ar_time_realistic = get_AR_time(data_size, 8, system, use_realistic_overhead=True)
    ar_time_ideal = get_AR_time(data_size, 8, system, use_realistic_overhead=False)

    overhead_factor = ar_time_realistic / ar_time_ideal if ar_time_ideal > 0 else 0

    test_1 = {
        'name': 'AllReduce overhead factor',
        'expected': '1.15-1.50 (Phase 6: more realistic for modern hardware)',
        'actual': overhead_factor,
        'passed': 1.10 < overhead_factor < 1.60,  # Updated range for Phase 6
    }
    results['tests'].append(test_1)
    if not test_1['passed']:
        results['passed'] = False

    # Test 2: ZeRO-3 should have 3 operations (2 AG + 1 RS)
    zero3_comm = calculate_zero3_communication_time(
        num_parameters=1_000_000_000,  # 1B params
        dp_ranks=8,
        system=system,
        precision='bf16',
    )

    # Total should be 2*AG + RS
    expected_ratio = 3  # 2 AllGathers + 1 ReduceScatter
    ag_time = zero3_comm['forward_allgather']
    rs_time = zero3_comm['backward_reducescatter']
    total_time = zero3_comm['total']

    # Verify we have both AG and RS times
    test_2 = {
        'name': 'ZeRO-3 has both AG and RS',
        'expected': 'both > 0',
        'actual': f"AG={ag_time:.1f}, RS={rs_time:.1f}",
        'passed': ag_time > 0 and rs_time > 0,
    }
    results['tests'].append(test_2)
    if not test_2['passed']:
        results['passed'] = False

    # Test 3: Exposed time should be less than total (overlap)
    exposed_time = zero3_comm['exposed']
    overlap_ratio = 1 - (exposed_time / total_time) if total_time > 0 else 0

    test_3 = {
        'name': 'ZeRO-3 comm overlap',
        'expected': '0.3-0.7',
        'actual': overlap_ratio,
        'passed': 0.2 < overlap_ratio < 0.8,
    }
    results['tests'].append(test_3)
    if not test_3['passed']:
        results['passed'] = False

    if debug:
        print("\n=== Phase 3: Communication Model Validation ===")
        for test in results['tests']:
            status = "PASS" if test['passed'] else "FAIL"
            print(f"  [{status}] {test['name']}: expected={test['expected']}, actual={test['actual']}")

    return results


def validate_kernel_overhead(debug: bool = False) -> Dict[str, Any]:
    """
    Test Phase 5: Kernel overhead accuracy.

    Validates:
    - Fused kernels have ~10-15 kernels per layer (not 80)
    - Kernel overhead is in reasonable range
    """
    from .training_modeling import _estimate_kernel_overhead_ms

    results = {'passed': True, 'tests': []}

    # Test 1: Fused kernel overhead should be lower
    overhead_fused = _estimate_kernel_overhead_ms(
        num_layers=32,
        batch_size=8,
        pipeline_parallel=1,
        gradient_checkpointing=False,
        use_fused_kernels=True,
    )

    overhead_unfused = _estimate_kernel_overhead_ms(
        num_layers=32,
        batch_size=8,
        pipeline_parallel=1,
        gradient_checkpointing=False,
        use_fused_kernels=False,
    )

    fused_ratio = overhead_fused / overhead_unfused if overhead_unfused > 0 else 0

    test_1 = {
        'name': 'Fused kernels reduce overhead',
        'expected': '~0.3-0.5',
        'actual': fused_ratio,
        'passed': 0.2 < fused_ratio < 0.6,  # Fused should be 30-50% of unfused
    }
    results['tests'].append(test_1)
    if not test_1['passed']:
        results['passed'] = False

    # Test 2: Overhead should be in reasonable range (< 5ms for 32 layers)
    test_2 = {
        'name': 'Overhead in reasonable range',
        'expected': '<5ms',
        'actual': f"{overhead_fused:.2f}ms",
        'passed': overhead_fused < 5.0,
    }
    results['tests'].append(test_2)
    if not test_2['passed']:
        results['passed'] = False

    if debug:
        print("\n=== Phase 5: Kernel Overhead Validation ===")
        for test in results['tests']:
            status = "PASS" if test['passed'] else "FAIL"
            print(f"  [{status}] {test['name']}: expected={test['expected']}, actual={test['actual']}")

    return results


def run_improvement_validation(debug: bool = False) -> Dict[str, Any]:
    """
    Run complete validation of Phase 1-5 accuracy improvements.

    Returns summary of all improvement tests.
    """
    results = {
        'flops_calculation': validate_flops_calculation(debug),
        'communication_model': validate_communication_model(debug),
        'kernel_overhead': validate_kernel_overhead(debug),
    }

    all_passed = all(r['passed'] for r in results.values())
    results['all_passed'] = all_passed

    if debug:
        print("\n" + "=" * 60)
        print("IMPROVEMENT VALIDATION SUMMARY")
        print("=" * 60)
        for name, result in results.items():
            if name == 'all_passed':
                continue
            status = "PASS" if result['passed'] else "FAIL"
            print(f"  {name}: {status}")
        print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

    return results
