"""Constants and enums for the serving simulation subsystem."""
from enum import Enum


class MemoryTier(str, Enum):
    """Memory hierarchy tiers ordered by proximity to compute."""
    DEVICE_HBM = "device_hbm"
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
