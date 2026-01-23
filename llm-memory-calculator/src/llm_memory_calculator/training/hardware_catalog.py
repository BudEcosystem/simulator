"""
Hardware Catalog for Training Optimization.

Comprehensive hardware database with:
- Compute capabilities (TFLOPS by precision)
- Memory specifications (capacity, bandwidth)
- Network capabilities (intra-node, inter-node)
- Cost information (hourly rates by provider)
- Optimal configurations and constraints

Sources:
- NVIDIA documentation
- Cloud provider pricing (AWS, GCP, Azure, Lambda Labs)
- Published benchmarks
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import math


class GPUGeneration(Enum):
    """GPU architecture generations."""
    VOLTA = "volta"         # V100
    AMPERE = "ampere"       # A100, A10, A6000
    ADA = "ada"            # L40S, RTX 4090
    HOPPER = "hopper"      # H100, H200, GH200
    BLACKWELL = "blackwell" # B100, B200, GB200
    MI300 = "mi300"        # AMD MI300X


class MemoryType(Enum):
    """Memory technology types."""
    HBM2 = "hbm2"
    HBM2E = "hbm2e"
    HBM3 = "hbm3"
    HBM3E = "hbm3e"
    GDDR6X = "gddr6x"


class InterconnectType(Enum):
    """GPU interconnect types."""
    PCIE_4 = "pcie4"       # ~64 GB/s bidirectional
    PCIE_5 = "pcie5"       # ~128 GB/s bidirectional
    NVLINK_3 = "nvlink3"   # 600 GB/s (A100)
    NVLINK_4 = "nvlink4"   # 900 GB/s (H100)
    NVLINK_5 = "nvlink5"   # 1.8 TB/s (B200)
    INFINITY_FABRIC = "infinity_fabric"  # AMD MI300X


@dataclass
class ComputeCapability:
    """Compute capability in TFLOPS by precision."""
    fp32: float           # FP32 tensor core TFLOPS
    fp16: float           # FP16 tensor core TFLOPS
    bf16: float           # BF16 tensor core TFLOPS
    tf32: float           # TF32 tensor core TFLOPS
    fp8: float = 0.0      # FP8 tensor core TFLOPS (Hopper+)
    int8: float = 0.0     # INT8 tensor core TFLOPS

    # Sparsity-enabled TFLOPS (2:4 structured sparsity)
    fp16_sparse: float = 0.0
    bf16_sparse: float = 0.0
    fp8_sparse: float = 0.0


@dataclass
class GPUSpec:
    """Complete GPU specification."""

    name: str
    display_name: str
    generation: GPUGeneration

    # Compute
    compute: ComputeCapability
    cuda_cores: int
    tensor_cores: int
    sm_count: int

    # Memory
    memory_gb: float
    memory_type: MemoryType
    memory_bandwidth_gbps: float

    # Interconnect
    interconnect: InterconnectType

    # Power and thermal (required)
    tdp_watts: int
    max_power_watts: int

    # Optional interconnect details
    nvlink_bandwidth_gbps: float = 0.0  # Per-GPU NVLink bandwidth
    max_nvlink_gpus: int = 0            # Max GPUs in NVSwitch domain

    # Training-specific features
    supports_fp8: bool = False
    supports_transformer_engine: bool = False
    flash_attention_optimized: bool = False

    # Constraints
    min_batch_size: int = 1
    max_sequence_length: int = 128000

    def get_compute_tflops(self, precision: str) -> float:
        """Get compute TFLOPS for given precision."""
        precision = precision.lower()
        mapping = {
            'fp32': self.compute.fp32,
            'float32': self.compute.fp32,
            'fp16': self.compute.fp16,
            'float16': self.compute.fp16,
            'bf16': self.compute.bf16,
            'bfloat16': self.compute.bf16,
            'tf32': self.compute.tf32,
            'fp8': self.compute.fp8,
            'int8': self.compute.int8,
        }
        return mapping.get(precision, self.compute.bf16)

    def estimate_mfu(self, model_size_b: float, batch_size: int) -> float:
        """Estimate Model FLOPS Utilization."""
        # Base MFU depends on model size and batch size
        # Larger models and batches generally achieve higher MFU

        base_mfu = 0.35  # Conservative baseline

        # Model size scaling (larger models are more compute-bound)
        if model_size_b >= 70:
            base_mfu += 0.10
        elif model_size_b >= 13:
            base_mfu += 0.05

        # Batch size scaling
        if batch_size >= 16:
            base_mfu += 0.08
        elif batch_size >= 4:
            base_mfu += 0.04

        # Generation bonus (newer architectures have better utilization)
        if self.generation in (GPUGeneration.HOPPER, GPUGeneration.BLACKWELL):
            base_mfu += 0.05

        return min(base_mfu, 0.55)  # Cap at 55% (very good)


@dataclass
class GPUCost:
    """Cost information for a GPU."""

    gpu_name: str

    # Hourly rates by provider (USD)
    aws_on_demand: float = 0.0
    aws_spot: float = 0.0
    gcp_on_demand: float = 0.0
    gcp_preemptible: float = 0.0
    azure_on_demand: float = 0.0
    azure_spot: float = 0.0
    lambda_labs: float = 0.0
    coreweave: float = 0.0
    vast_ai: float = 0.0
    runpod: float = 0.0

    # Purchase cost (for on-prem)
    purchase_price_usd: float = 0.0

    def get_best_cloud_rate(self, allow_spot: bool = True) -> Tuple[float, str]:
        """Get best available cloud rate."""
        rates = []

        if self.lambda_labs > 0:
            rates.append((self.lambda_labs, "Lambda Labs"))
        if self.coreweave > 0:
            rates.append((self.coreweave, "CoreWeave"))
        if self.runpod > 0:
            rates.append((self.runpod, "RunPod"))
        if self.vast_ai > 0:
            rates.append((self.vast_ai, "Vast.ai"))

        if allow_spot:
            if self.aws_spot > 0:
                rates.append((self.aws_spot, "AWS Spot"))
            if self.gcp_preemptible > 0:
                rates.append((self.gcp_preemptible, "GCP Preemptible"))
            if self.azure_spot > 0:
                rates.append((self.azure_spot, "Azure Spot"))

        if self.aws_on_demand > 0:
            rates.append((self.aws_on_demand, "AWS On-Demand"))
        if self.gcp_on_demand > 0:
            rates.append((self.gcp_on_demand, "GCP On-Demand"))
        if self.azure_on_demand > 0:
            rates.append((self.azure_on_demand, "Azure On-Demand"))

        if not rates:
            return (0.0, "Unknown")

        return min(rates, key=lambda x: x[0])


@dataclass
class ClusterSpec:
    """Specification for a GPU cluster."""

    name: str
    gpu_spec: GPUSpec
    num_gpus: int

    # Network topology
    gpus_per_node: int = 8
    intra_node_bandwidth_gbps: float = 600.0  # NVLink
    inter_node_bandwidth_gbps: float = 400.0  # InfiniBand/RoCE

    # Storage
    storage_bandwidth_gbps: float = 100.0

    @property
    def num_nodes(self) -> int:
        return math.ceil(self.num_gpus / self.gpus_per_node)

    @property
    def total_memory_gb(self) -> float:
        return self.num_gpus * self.gpu_spec.memory_gb

    @property
    def total_compute_tflops(self) -> float:
        return self.num_gpus * self.gpu_spec.compute.bf16

    def get_effective_bandwidth(self, parallelism_config: Dict[str, int]) -> float:
        """Get effective bandwidth considering parallelism."""
        tp = parallelism_config.get('tp', 1)
        pp = parallelism_config.get('pp', 1)
        dp = parallelism_config.get('dp', 1)

        # TP communication is within-node (NVLink)
        # PP and DP communication may cross nodes

        if tp > self.gpus_per_node:
            # TP crosses nodes - use inter-node bandwidth
            return min(self.intra_node_bandwidth_gbps, self.inter_node_bandwidth_gbps)

        if dp > 1 or pp > 1:
            # Cross-node communication needed
            return self.inter_node_bandwidth_gbps

        return self.intra_node_bandwidth_gbps


# =============================================================================
# GPU HARDWARE DATABASE
# =============================================================================

GPU_SPECS: Dict[str, GPUSpec] = {
    # NVIDIA Volta
    "v100_16gb": GPUSpec(
        name="v100_16gb",
        display_name="NVIDIA V100 16GB",
        generation=GPUGeneration.VOLTA,
        compute=ComputeCapability(
            fp32=15.7, fp16=125.0, bf16=0.0, tf32=0.0, int8=62.5
        ),
        cuda_cores=5120,
        tensor_cores=640,
        sm_count=80,
        memory_gb=16,
        memory_type=MemoryType.HBM2,
        memory_bandwidth_gbps=900,
        interconnect=InterconnectType.NVLINK_3,
        nvlink_bandwidth_gbps=300,
        max_nvlink_gpus=8,
        tdp_watts=250,
        max_power_watts=300,
    ),

    "v100_32gb": GPUSpec(
        name="v100_32gb",
        display_name="NVIDIA V100 32GB",
        generation=GPUGeneration.VOLTA,
        compute=ComputeCapability(
            fp32=15.7, fp16=125.0, bf16=0.0, tf32=0.0, int8=62.5
        ),
        cuda_cores=5120,
        tensor_cores=640,
        sm_count=80,
        memory_gb=32,
        memory_type=MemoryType.HBM2,
        memory_bandwidth_gbps=900,
        interconnect=InterconnectType.NVLINK_3,
        nvlink_bandwidth_gbps=300,
        max_nvlink_gpus=8,
        tdp_watts=250,
        max_power_watts=300,
    ),

    # NVIDIA Ampere
    "a100_40gb": GPUSpec(
        name="a100_40gb",
        display_name="NVIDIA A100 40GB",
        generation=GPUGeneration.AMPERE,
        compute=ComputeCapability(
            fp32=19.5, fp16=312.0, bf16=312.0, tf32=156.0, int8=624.0,
            fp16_sparse=624.0, bf16_sparse=624.0
        ),
        cuda_cores=6912,
        tensor_cores=432,
        sm_count=108,
        memory_gb=40,
        memory_type=MemoryType.HBM2E,
        memory_bandwidth_gbps=1555,
        interconnect=InterconnectType.NVLINK_3,
        nvlink_bandwidth_gbps=600,
        max_nvlink_gpus=8,
        tdp_watts=400,
        max_power_watts=400,
        flash_attention_optimized=True,
    ),

    "a100_80gb": GPUSpec(
        name="a100_80gb",
        display_name="NVIDIA A100 80GB",
        generation=GPUGeneration.AMPERE,
        compute=ComputeCapability(
            fp32=19.5, fp16=312.0, bf16=312.0, tf32=156.0, int8=624.0,
            fp16_sparse=624.0, bf16_sparse=624.0
        ),
        cuda_cores=6912,
        tensor_cores=432,
        sm_count=108,
        memory_gb=80,
        memory_type=MemoryType.HBM2E,
        memory_bandwidth_gbps=2039,
        interconnect=InterconnectType.NVLINK_3,
        nvlink_bandwidth_gbps=600,
        max_nvlink_gpus=8,
        tdp_watts=400,
        max_power_watts=400,
        flash_attention_optimized=True,
    ),

    # NVIDIA Ada Lovelace
    "rtx_4090": GPUSpec(
        name="rtx_4090",
        display_name="NVIDIA RTX 4090",
        generation=GPUGeneration.ADA,
        compute=ComputeCapability(
            fp32=82.6, fp16=330.0, bf16=330.0, tf32=165.0, int8=660.0
        ),
        cuda_cores=16384,
        tensor_cores=512,
        sm_count=128,
        memory_gb=24,
        memory_type=MemoryType.GDDR6X,
        memory_bandwidth_gbps=1008,
        interconnect=InterconnectType.PCIE_4,
        nvlink_bandwidth_gbps=0,
        max_nvlink_gpus=1,
        tdp_watts=450,
        max_power_watts=600,
        flash_attention_optimized=True,
    ),

    "l40s": GPUSpec(
        name="l40s",
        display_name="NVIDIA L40S",
        generation=GPUGeneration.ADA,
        compute=ComputeCapability(
            fp32=91.6, fp16=366.0, bf16=366.0, tf32=183.0, fp8=733.0, int8=733.0,
            fp16_sparse=733.0, bf16_sparse=733.0, fp8_sparse=1466.0
        ),
        cuda_cores=18176,
        tensor_cores=568,
        sm_count=142,
        memory_gb=48,
        memory_type=MemoryType.GDDR6X,
        memory_bandwidth_gbps=864,
        interconnect=InterconnectType.PCIE_4,
        nvlink_bandwidth_gbps=0,
        max_nvlink_gpus=1,
        tdp_watts=350,
        max_power_watts=350,
        supports_fp8=True,
        flash_attention_optimized=True,
    ),

    # NVIDIA Hopper
    "h100_sxm": GPUSpec(
        name="h100_sxm",
        display_name="NVIDIA H100 SXM",
        generation=GPUGeneration.HOPPER,
        compute=ComputeCapability(
            fp32=66.9, fp16=1979.0, bf16=1979.0, tf32=989.0, fp8=3958.0, int8=3958.0,
            fp16_sparse=3958.0, bf16_sparse=3958.0, fp8_sparse=7916.0
        ),
        cuda_cores=16896,
        tensor_cores=528,
        sm_count=132,
        memory_gb=80,
        memory_type=MemoryType.HBM3,
        memory_bandwidth_gbps=3352,
        interconnect=InterconnectType.NVLINK_4,
        nvlink_bandwidth_gbps=900,
        max_nvlink_gpus=8,
        tdp_watts=700,
        max_power_watts=700,
        supports_fp8=True,
        supports_transformer_engine=True,
        flash_attention_optimized=True,
    ),

    "h100_pcie": GPUSpec(
        name="h100_pcie",
        display_name="NVIDIA H100 PCIe",
        generation=GPUGeneration.HOPPER,
        compute=ComputeCapability(
            fp32=51.2, fp16=1513.0, bf16=1513.0, tf32=756.0, fp8=3026.0, int8=3026.0,
            fp16_sparse=3026.0, bf16_sparse=3026.0, fp8_sparse=6052.0
        ),
        cuda_cores=14592,
        tensor_cores=456,
        sm_count=114,
        memory_gb=80,
        memory_type=MemoryType.HBM3,
        memory_bandwidth_gbps=2000,
        interconnect=InterconnectType.PCIE_5,
        nvlink_bandwidth_gbps=0,
        max_nvlink_gpus=1,
        tdp_watts=350,
        max_power_watts=350,
        supports_fp8=True,
        supports_transformer_engine=True,
        flash_attention_optimized=True,
    ),

    "h200": GPUSpec(
        name="h200",
        display_name="NVIDIA H200",
        generation=GPUGeneration.HOPPER,
        compute=ComputeCapability(
            fp32=66.9, fp16=1979.0, bf16=1979.0, tf32=989.0, fp8=3958.0, int8=3958.0,
            fp16_sparse=3958.0, bf16_sparse=3958.0, fp8_sparse=7916.0
        ),
        cuda_cores=16896,
        tensor_cores=528,
        sm_count=132,
        memory_gb=141,
        memory_type=MemoryType.HBM3E,
        memory_bandwidth_gbps=4800,
        interconnect=InterconnectType.NVLINK_4,
        nvlink_bandwidth_gbps=900,
        max_nvlink_gpus=8,
        tdp_watts=700,
        max_power_watts=700,
        supports_fp8=True,
        supports_transformer_engine=True,
        flash_attention_optimized=True,
    ),

    # NVIDIA Blackwell
    "b100": GPUSpec(
        name="b100",
        display_name="NVIDIA B100",
        generation=GPUGeneration.BLACKWELL,
        compute=ComputeCapability(
            fp32=60.0, fp16=3500.0, bf16=3500.0, tf32=1750.0, fp8=7000.0, int8=7000.0,
            fp16_sparse=7000.0, bf16_sparse=7000.0, fp8_sparse=14000.0
        ),
        cuda_cores=18000,  # Estimated
        tensor_cores=600,   # Estimated
        sm_count=150,       # Estimated
        memory_gb=192,
        memory_type=MemoryType.HBM3E,
        memory_bandwidth_gbps=8000,
        interconnect=InterconnectType.NVLINK_5,
        nvlink_bandwidth_gbps=1800,
        max_nvlink_gpus=8,
        tdp_watts=700,
        max_power_watts=1000,
        supports_fp8=True,
        supports_transformer_engine=True,
        flash_attention_optimized=True,
    ),

    "b200": GPUSpec(
        name="b200",
        display_name="NVIDIA B200",
        generation=GPUGeneration.BLACKWELL,
        compute=ComputeCapability(
            fp32=80.0, fp16=4500.0, bf16=4500.0, tf32=2250.0, fp8=9000.0, int8=9000.0,
            fp16_sparse=9000.0, bf16_sparse=9000.0, fp8_sparse=18000.0
        ),
        cuda_cores=20000,   # Estimated
        tensor_cores=700,    # Estimated
        sm_count=160,        # Estimated
        memory_gb=192,
        memory_type=MemoryType.HBM3E,
        memory_bandwidth_gbps=8000,
        interconnect=InterconnectType.NVLINK_5,
        nvlink_bandwidth_gbps=1800,
        max_nvlink_gpus=8,
        tdp_watts=1000,
        max_power_watts=1200,
        supports_fp8=True,
        supports_transformer_engine=True,
        flash_attention_optimized=True,
    ),

    # AMD
    "mi300x": GPUSpec(
        name="mi300x",
        display_name="AMD MI300X",
        generation=GPUGeneration.MI300,
        compute=ComputeCapability(
            fp32=163.4, fp16=1307.0, bf16=1307.0, tf32=653.7, fp8=2614.0, int8=2614.0
        ),
        cuda_cores=0,  # AMD uses CUs
        tensor_cores=304,  # Matrix cores
        sm_count=304,  # Compute Units
        memory_gb=192,
        memory_type=MemoryType.HBM3,
        memory_bandwidth_gbps=5300,
        interconnect=InterconnectType.INFINITY_FABRIC,
        nvlink_bandwidth_gbps=0,
        max_nvlink_gpus=8,
        tdp_watts=750,
        max_power_watts=750,
        supports_fp8=True,
        flash_attention_optimized=True,
    ),
}


# =============================================================================
# GPU COST DATABASE
# =============================================================================

GPU_COSTS: Dict[str, GPUCost] = {
    "v100_16gb": GPUCost(
        gpu_name="v100_16gb",
        aws_on_demand=0.90,
        aws_spot=0.27,
        gcp_on_demand=0.74,
        gcp_preemptible=0.22,
        lambda_labs=0.50,
        vast_ai=0.25,
        runpod=0.39,
        purchase_price_usd=3000,
    ),

    "v100_32gb": GPUCost(
        gpu_name="v100_32gb",
        aws_on_demand=1.21,
        aws_spot=0.36,
        gcp_on_demand=0.99,
        gcp_preemptible=0.30,
        lambda_labs=0.60,
        vast_ai=0.35,
        runpod=0.49,
        purchase_price_usd=5000,
    ),

    "a100_40gb": GPUCost(
        gpu_name="a100_40gb",
        aws_on_demand=2.21,
        aws_spot=0.80,
        gcp_on_demand=2.06,
        gcp_preemptible=0.72,
        azure_on_demand=2.42,
        azure_spot=0.73,
        lambda_labs=1.10,
        coreweave=1.28,
        vast_ai=0.80,
        runpod=0.79,
        purchase_price_usd=10000,
    ),

    "a100_80gb": GPUCost(
        gpu_name="a100_80gb",
        aws_on_demand=3.67,
        aws_spot=1.40,
        gcp_on_demand=3.67,
        gcp_preemptible=1.10,
        azure_on_demand=3.67,
        azure_spot=1.28,
        lambda_labs=1.29,
        coreweave=1.85,
        vast_ai=1.10,
        runpod=1.19,
        purchase_price_usd=15000,
    ),

    "rtx_4090": GPUCost(
        gpu_name="rtx_4090",
        lambda_labs=0.50,
        vast_ai=0.34,
        runpod=0.44,
        purchase_price_usd=2000,
    ),

    "l40s": GPUCost(
        gpu_name="l40s",
        aws_on_demand=1.98,
        gcp_on_demand=1.70,
        lambda_labs=0.99,
        coreweave=1.14,
        runpod=0.99,
        purchase_price_usd=8000,
    ),

    "h100_sxm": GPUCost(
        gpu_name="h100_sxm",
        aws_on_demand=4.76,
        aws_spot=2.00,
        gcp_on_demand=4.76,
        gcp_preemptible=1.90,
        azure_on_demand=5.12,
        azure_spot=2.05,
        lambda_labs=2.49,
        coreweave=2.85,
        vast_ai=2.00,
        runpod=2.39,
        purchase_price_usd=30000,
    ),

    "h100_pcie": GPUCost(
        gpu_name="h100_pcie",
        aws_on_demand=3.98,
        gcp_on_demand=3.98,
        lambda_labs=1.99,
        coreweave=2.23,
        runpod=1.99,
        purchase_price_usd=25000,
    ),

    "h200": GPUCost(
        gpu_name="h200",
        aws_on_demand=6.50,
        gcp_on_demand=6.50,
        lambda_labs=3.99,
        coreweave=3.85,
        purchase_price_usd=40000,
    ),

    "b100": GPUCost(
        gpu_name="b100",
        # Estimated pricing (not yet widely available)
        lambda_labs=5.00,
        coreweave=5.50,
        purchase_price_usd=50000,
    ),

    "b200": GPUCost(
        gpu_name="b200",
        # Estimated pricing (not yet widely available)
        lambda_labs=7.00,
        coreweave=7.50,
        purchase_price_usd=70000,
    ),

    "mi300x": GPUCost(
        gpu_name="mi300x",
        aws_on_demand=3.50,
        gcp_on_demand=3.50,
        coreweave=2.50,
        purchase_price_usd=20000,
    ),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_gpu_spec(name: str) -> GPUSpec:
    """Get GPU specification by name."""
    name = name.lower().replace("-", "_").replace(" ", "_")
    if name in GPU_SPECS:
        return GPU_SPECS[name]

    # Try to find partial match
    for gpu_name, spec in GPU_SPECS.items():
        if name in gpu_name or gpu_name in name:
            return spec

    raise ValueError(f"Unknown GPU: {name}. Available: {list(GPU_SPECS.keys())}")


def get_gpu_cost(name: str) -> GPUCost:
    """Get GPU cost by name."""
    name = name.lower().replace("-", "_").replace(" ", "_")
    if name in GPU_COSTS:
        return GPU_COSTS[name]

    # Try to find partial match
    for gpu_name, cost in GPU_COSTS.items():
        if name in gpu_name or gpu_name in name:
            return cost

    raise ValueError(f"Unknown GPU cost: {name}. Available: {list(GPU_COSTS.keys())}")


def list_gpus(
    min_memory_gb: float = 0,
    generation: Optional[GPUGeneration] = None,
    supports_fp8: Optional[bool] = None,
) -> List[GPUSpec]:
    """List GPUs matching criteria."""
    results = []
    for spec in GPU_SPECS.values():
        if spec.memory_gb < min_memory_gb:
            continue
        if generation is not None and spec.generation != generation:
            continue
        if supports_fp8 is not None and spec.supports_fp8 != supports_fp8:
            continue
        results.append(spec)

    return sorted(results, key=lambda x: x.memory_gb)


def estimate_cluster_cost(
    gpu_name: str,
    num_gpus: int,
    hours: float,
    allow_spot: bool = True,
) -> Dict[str, Any]:
    """Estimate cluster cost."""
    cost = get_gpu_cost(gpu_name)
    rate, provider = cost.get_best_cloud_rate(allow_spot)

    total_cost = rate * num_gpus * hours

    return {
        "gpu": gpu_name,
        "num_gpus": num_gpus,
        "hours": hours,
        "hourly_rate_per_gpu": rate,
        "provider": provider,
        "total_cost_usd": total_cost,
    }


def get_optimal_gpu_for_model(
    model_params_b: float,
    training_method: str = "lora",
    precision: str = "bf16",
    min_batch_size: int = 1,
) -> List[Dict[str, Any]]:
    """
    Get ranked list of GPUs optimal for training a model.

    Returns GPUs sorted by cost-efficiency.
    """
    precision_bytes = 2 if precision in ('bf16', 'fp16') else 4

    # Estimate memory requirements
    if training_method == "qlora":
        # 4-bit base + small overhead
        weight_memory_gb = (model_params_b * 1e9 * 0.5) / 1e9
        total_estimate = weight_memory_gb * 2  # Conservative with activations
    elif training_method == "lora":
        # Full precision base + small LoRA overhead
        weight_memory_gb = (model_params_b * 1e9 * precision_bytes) / 1e9
        total_estimate = weight_memory_gb * 1.5
    else:
        # Full fine-tuning: weights + gradients + optimizer
        weight_memory_gb = (model_params_b * 1e9 * precision_bytes) / 1e9
        total_estimate = weight_memory_gb * 8  # Mixed precision AdamW

    results = []

    for gpu_name, spec in GPU_SPECS.items():
        if spec.memory_gb < total_estimate * 0.8:  # Need at least 80% headroom
            continue

        cost = GPU_COSTS.get(gpu_name)
        if cost is None:
            continue

        rate, provider = cost.get_best_cloud_rate()
        if rate == 0:
            continue

        # Calculate cost-efficiency: TFLOPS per dollar
        tflops = spec.get_compute_tflops(precision)
        efficiency = tflops / rate if rate > 0 else 0

        results.append({
            "gpu": gpu_name,
            "display_name": spec.display_name,
            "memory_gb": spec.memory_gb,
            "compute_tflops": tflops,
            "hourly_rate": rate,
            "provider": provider,
            "tflops_per_dollar": efficiency,
            "fits_model": spec.memory_gb >= total_estimate,
            "needs_multi_gpu": spec.memory_gb < total_estimate,
            "min_gpus_needed": max(1, math.ceil(total_estimate / (spec.memory_gb * 0.85))),
        })

    # Sort by cost-efficiency (TFLOPS per dollar)
    return sorted(results, key=lambda x: x["tflops_per_dollar"], reverse=True)


def calculate_communication_time(
    data_size_gb: float,
    bandwidth_gbps: float,
    operation: str = "allreduce",
) -> float:
    """
    Calculate communication time in seconds.

    Args:
        data_size_gb: Data size in GB
        bandwidth_gbps: Bandwidth in GB/s
        operation: Type of collective operation

    Returns:
        Time in seconds
    """
    # Convert to bits and bytes correctly
    data_bits = data_size_gb * 8  # GB to Gb

    # AllReduce has 2x communication overhead (ring algorithm)
    if operation == "allreduce":
        effective_data = data_bits * 2
    elif operation == "allgather":
        effective_data = data_bits
    elif operation == "reduce_scatter":
        effective_data = data_bits
    else:
        effective_data = data_bits

    return effective_data / bandwidth_gbps
