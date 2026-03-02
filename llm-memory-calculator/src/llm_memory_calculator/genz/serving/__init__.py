"""Serving simulation subsystem for LLM inference workload modeling."""


def get_modeling_functions(hardware: str):
    """Return ``(prefill_fn, decode_fn)`` appropriate for the hardware type.

    For CPU hardware, routes to the CPU-aware wrappers that enhance operators
    with ISA-specific implementations.  For GPU/TPU/ASIC hardware, returns
    the standard GenZ modeling functions.
    """
    from llm_memory_calculator.hardware.configs import is_cpu_hardware
    if is_cpu_hardware(hardware):
        from llm_memory_calculator.genz.cpu import (
            cpu_aware_prefill_moddeling,
            cpu_aware_decode_moddeling,
        )
        return cpu_aware_prefill_moddeling, cpu_aware_decode_moddeling
    else:
        from llm_memory_calculator.genz import prefill_moddeling, decode_moddeling
        return prefill_moddeling, decode_moddeling


from .constants import (
    MemoryTier,
    EvictionPolicy,
    RequestStatus,
    PowerState,
    PowerComponent,
    GB_TO_BYTES,
    MB_TO_BYTES,
    NS_PER_MS,
    NS_PER_S,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_PRECISION_BYTES,
    SERVER_POWER_FRACTION_GPU,
    SERVER_POWER_FRACTION_CPU,
    SERVER_POWER_FRACTION_DRAM,
    SERVER_POWER_FRACTION_NVLINK,
    SERVER_POWER_FRACTION_NIC,
    SERVER_POWER_FRACTION_COOLING,
    SERVER_POWER_FRACTION_SSD,
    GPU_IDLE_FRACTION,
    GPU_ACTIVE_FRACTION,
    GPU_STANDBY_FRACTION,
    DRAM_ENERGY_PJ_PER_BIT_HBM2E,
    DRAM_ENERGY_PJ_PER_BIT_HBM3,
    DRAM_ACTIVITY_FACTOR,
    PUE_HYPERSCALE,
    PUE_ENTERPRISE,
    PUE_INDUSTRY_AVG,
    PUE_DEFAULT,
)
from .request import Request, Batch
from .slo_tracker import SLOTargets, SLOTracker
from .memory_model import MemoryTierConfig, MemoryModel
from .power_model import ComponentPowerConfig, PowerConfig, PowerModel
from .workload import WorkloadConfig, WorkloadGenerator
from .batch_scheduler import SchedulerConfig, BatchScheduler
from .prefix_cache import RadixTreeNode, RadixCache, PrefixCacheAnalyzer
from .config_optimizer import SearchSpace, OptimizationConstraints, ConfigResult, ConfigOptimizer
from .disaggregation import DisaggregationAnalyzer
from .cluster import ClusterAnalyzer
from .simulator import ServingSimulationResult, ServingSimulator

__all__ = [
    # Routing helper
    "get_modeling_functions",
    # Constants & Enums
    "MemoryTier",
    "EvictionPolicy",
    "RequestStatus",
    "PowerState",
    "PowerComponent",
    "GB_TO_BYTES",
    "MB_TO_BYTES",
    "NS_PER_MS",
    "NS_PER_S",
    "DEFAULT_BLOCK_SIZE",
    "DEFAULT_PRECISION_BYTES",
    # Request
    "Request",
    "Batch",
    # SLO
    "SLOTargets",
    "SLOTracker",
    # Memory
    "MemoryTierConfig",
    "MemoryModel",
    # Power
    "ComponentPowerConfig",
    "PowerConfig",
    "PowerModel",
    # Workload
    "WorkloadConfig",
    "WorkloadGenerator",
    # Scheduler
    "SchedulerConfig",
    "BatchScheduler",
    # Prefix Cache
    "RadixTreeNode",
    "RadixCache",
    "PrefixCacheAnalyzer",
    # Config Optimizer
    "SearchSpace",
    "OptimizationConstraints",
    "ConfigResult",
    "ConfigOptimizer",
    # Disaggregation
    "DisaggregationAnalyzer",
    # Cluster
    "ClusterAnalyzer",
    # Simulator
    "ServingSimulationResult",
    "ServingSimulator",
]
