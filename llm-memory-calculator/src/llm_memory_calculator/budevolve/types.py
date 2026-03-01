"""Core data types for BudEvolve optimization."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class ServingConfig:
    """Serving configuration to evaluate."""
    model: str
    hardware: str
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    batch_size: int = 32
    precision: str = "bf16"
    max_num_batched_tokens: int = 8192
    enable_chunked_prefill: bool = False
    chunk_size: int = 512
    enable_prefix_caching: bool = False
    gpu_memory_utilization: float = 0.90


@dataclass
class HardwareSpec:
    """Hypothetical hardware specification for design exploration."""
    flops_tflops: float
    offchip_mem_bw_gbps: float
    off_chip_mem_size_gb: float
    on_chip_mem_size_mb: float = 40.0
    onchip_mem_bw_gbps: float = 18000.0
    interchip_link_bw_gbps: float = 150.0
    frequency_ghz: float = 1.0
    num_nodes: int = 1
    tdp_watts: float = 700.0
    estimated_cost_usd: float = 25000.0

    def to_system_kwargs(self) -> Dict[str, Any]:
        """Convert to kwargs for genz.system.System() constructor."""
        return {
            "flops": self.flops_tflops,
            "offchip_mem_bw": self.offchip_mem_bw_gbps,
            "off_chip_mem_size": self.off_chip_mem_size_gb * 1024,
            "on_chip_mem_size": self.on_chip_mem_size_mb,
            "onchip_mem_bw": self.onchip_mem_bw_gbps,
            "interchip_link_bw": self.interchip_link_bw_gbps,
            "frequency": self.frequency_ghz * 1000,
            "num_nodes": self.num_nodes,
        }


@dataclass
class OperatorInsight:
    """Per-operator roofline analysis result."""
    name: str
    bottleneck: str
    arithmetic_intensity: float
    compute_time_ms: float
    memory_time_ms: float
    pct_of_total: float


@dataclass
class RooflineReport:
    """Roofline analysis report from BudSim."""
    overall_bottleneck: str
    compute_utilization: float
    memory_bw_utilization: float
    interconnect_utilization: float
    per_operator: List[OperatorInsight]
    recommendations: List[str]


@dataclass
class EvalResult:
    """Result from evaluating a config/hardware/algorithm through BudSim."""
    throughput_rps: float = 0.0
    token_throughput_tps: float = 0.0
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    e2e_latency_ms: float = 0.0
    memory_gb: float = 0.0
    power_w: float = 0.0
    slo_compliance: float = 0.0
    cost_per_million_tokens: float = 0.0
    roofline: Optional[RooflineReport] = None
    feasible: bool = True
    config: Optional[Dict[str, Any]] = None


@dataclass
class ParetoResult:
    """Result from multi-objective optimization."""
    pareto_front: List[Dict[str, Any]]
    all_evaluated: List[Dict[str, Any]]
    best_per_objective: Dict[str, Any]
    sensitivity: Dict[str, float] = field(default_factory=dict)
    num_generations: int = 0
    total_evaluations: int = 0
