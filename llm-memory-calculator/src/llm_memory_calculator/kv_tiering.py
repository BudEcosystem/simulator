"""KV cache tier planning (Bud FRD-023). Kept for the Phase 1 import path; the planner now lives in
:mod:`llm_memory_calculator.kv`."""

from .kv.constants import BREAK_EVEN_RATIO as T1_BREAK_EVEN_RATIO  # noqa: F401
from .kv.constants import (  # noqa: F401
    CPU_KV_SHARE_MAX,
    DEFAULT_PINNED_GBPS,
    DEVICE_OVERHEAD_GIB,
    EVENTS_ENDPOINT,
    EVENTS_REPLAY,
    FP8_KV_VALIDATED,
    GIB,
    GPU_UTIL_MAX,
    INTERACTIVE_TTFT_MS,
    PCIE_GBPS_BY_GEN,
    PLAN_VERSION,
    PREFILL_MFU,
    PREFIX_HEAVY_RATIO,
    PROFILE_REUSE,
    SLICE_KV_SHARE_MAX,
    T0_GPU_UTIL,
    T1_BLOCK_SIZE,
    T1_HOST_RAM_SHARE,
    T1_LOAD_EFFICIENCY,
    T1_MAX_POOL_MULTIPLE,
    T1_MIN_POOL_MULTIPLE,
    T1_MODE_FEATURES,
)
from .kv.cost import prefill_seconds  # noqa: F401
from .kv.facts import NodeKVFacts  # noqa: F401
from .kv.geometry import kv_bytes_per_token_per_rank  # noqa: F401
from .kv.plan import KVTierPlan  # noqa: F401
from .kv.planner import plan_kv_tiers  # noqa: F401
from .kv.workload import KVWorkload  # noqa: F401
