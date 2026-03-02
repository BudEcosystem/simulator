"""Constants and enums for the serving simulation subsystem."""
from enum import Enum


class MemoryTier(str, Enum):
    """Memory hierarchy tiers ordered by proximity to compute."""
    DEVICE_HBM = "device_hbm"
    DEVICE_DRAM = "device_dram"  # CPU primary memory (DDR4/DDR5)
    HOST_DDR = "host_ddr"
    CXL = "cxl"
    NVME = "nvme"


class EvictionPolicy(str, Enum):
    """Cache eviction policies for KV block management."""
    LRU = "lru"
    LFU = "lfu"


class RequestStatus(str, Enum):
    """Lifecycle states for an inference request."""
    ARRIVED = "arrived"
    QUEUED = "queued"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    COMPLETE = "complete"
    FAILED = "failed"


class PowerState(str, Enum):
    """Power states for hardware components."""
    IDLE = "idle"
    ACTIVE = "active"
    STANDBY = "standby"


class PowerComponent(str, Enum):
    """Hardware components tracked for power modeling."""
    ACCELERATOR = "accelerator"
    DRAM = "dram"
    INTERCONNECT = "interconnect"
    HOST_CPU = "host_cpu"
    COOLING = "cooling"
    STORAGE = "storage"
    MISC = "misc"


# Unit conversion constants
GB_TO_BYTES: int = 1024 ** 3
MB_TO_BYTES: int = 1024 ** 2
NS_PER_MS: int = 1_000_000
NS_PER_S: int = 1_000_000_000

# Default configuration values
DEFAULT_BLOCK_SIZE: int = 16
DEFAULT_PRECISION_BYTES: int = 2

# ---------------------------------------------------------------------------
# Physics-based power constants derived from DGX H100 measured power rails
# (10.2 kW total system with 8× H100 SXM5 GPUs)
# ---------------------------------------------------------------------------

# Server power fractions (fraction of total system power per component)
# Source: NVIDIA DGX H100 System Architecture Whitepaper, power rail instrumentation
SERVER_POWER_FRACTION_GPU: float = 0.55       # GPU compute subsystem
SERVER_POWER_FRACTION_CPU: float = 0.11       # Host CPUs (2× AMD EPYC)
SERVER_POWER_FRACTION_DRAM: float = 0.065     # System DRAM (2 TB DDR5)
SERVER_POWER_FRACTION_NVLINK: float = 0.09    # NVLink/NVSwitch interconnect
SERVER_POWER_FRACTION_NIC: float = 0.04       # Network interfaces (8× ConnectX-7)
SERVER_POWER_FRACTION_COOLING: float = 0.125  # Cooling subsystem (fans, liquid cooling)
SERVER_POWER_FRACTION_SSD: float = 0.02       # Storage subsystem (NVMe SSDs)

# GPU power state fractions (fraction of TDP)
# Source: nvidia-smi power measurements across A100/H100 under various workloads
# Idle: leakage-current dominated (A100 measured: ~50W / 400W TDP)
# Active: inference is memory-BW bound, cannot sustain peak FLOPS (~70% TDP)
# Standby: clock-gated, only memory controllers active (~20% TDP)
GPU_IDLE_FRACTION: float = 0.125
GPU_ACTIVE_FRACTION: float = 0.70
GPU_STANDBY_FRACTION: float = 0.20

# DRAM energy per bit (picojoules)
# Source: Micron HBM2e / HBM3 datasheets
DRAM_ENERGY_PJ_PER_BIT_HBM2E: float = 3.7
DRAM_ENERGY_PJ_PER_BIT_HBM3: float = 2.5

# DRAM activity factor: accounts for refresh cycles and row activation overhead
# Source: Micron HBM2e datasheet, effective bandwidth utilization
DRAM_ACTIVITY_FACTOR: float = 0.80  # range 0.7-0.9 in practice

# ---------------------------------------------------------------------------
# CPU power constants derived from SPECpower measurements across Xeon/EPYC
# ---------------------------------------------------------------------------

# CPU power state fractions (fraction of TDP)
# Source: SPECpower measurements across Xeon/EPYC under various workloads
CPU_IDLE_FRACTION: float = 0.30
CPU_ACTIVE_FRACTION: float = 0.85
CPU_STANDBY_FRACTION: float = 0.45

# DRAM energy per bit for DDR5/DDR4 (picojoules)
# Source: Micron DDR5-4800 and DDR4-3200 datasheets
DRAM_ENERGY_PJ_PER_BIT_DDR5: float = 3.7
DRAM_ENERGY_PJ_PER_BIT_DDR4: float = 5.0

# CPU server power fractions (CPU is the compute element, no separate GPU)
SERVER_POWER_FRACTION_CPU_COMPUTE: float = 0.45
SERVER_POWER_FRACTION_CPU_DRAM: float = 0.25
SERVER_POWER_FRACTION_CPU_COOLING: float = 0.15
SERVER_POWER_FRACTION_CPU_NIC: float = 0.05
SERVER_POWER_FRACTION_CPU_SSD: float = 0.05
SERVER_POWER_FRACTION_CPU_MISC: float = 0.05

# Power Usage Effectiveness (PUE) defaults
# PUE = Total Facility Power / IT Equipment Power
# Source: Uptime Institute 2023 Global Data Center Survey
PUE_HYPERSCALE: float = 1.10   # Google, Meta, Microsoft tier
PUE_ENTERPRISE: float = 1.30   # Well-run enterprise DC
PUE_INDUSTRY_AVG: float = 1.50 # Industry average
PUE_DEFAULT: float = 1.20      # Modern well-run DC
