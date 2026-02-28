"""Multi-tier memory model for KV cache management in LLM serving.

Tracks memory allocation across tiers (HBM, DDR, CXL, NVME),
handles KV block allocation/deallocation, and supports eviction
and spilling between tiers.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import OrderedDict

from .constants import (
    MemoryTier, EvictionPolicy,
    GB_TO_BYTES, DEFAULT_BLOCK_SIZE, DEFAULT_PRECISION_BYTES,
)


@dataclass
class MemoryTierConfig:
    """Configuration for a single memory tier."""
    tier: MemoryTier
    capacity_bytes: int
    bandwidth_gbps: float = 0.0
    latency_ns: float = 0.0


class MemoryModel:
    """Multi-tier memory model for KV cache management in LLM serving.

    Tracks memory allocation across tiers (HBM, DDR, CXL, NVME),
    handles KV block allocation/deallocation, and supports eviction
    and spilling between tiers.
    """

    def __init__(
        self,
        model_config,
        tier_configs: List[MemoryTierConfig],
        block_size: int = DEFAULT_BLOCK_SIZE,
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
        tensor_parallel: int = 1,
        precision_bytes: int = DEFAULT_PRECISION_BYTES,
    ):
        self._model_config = model_config
        self._tier_configs = {tc.tier: tc for tc in tier_configs}
        self._block_size = block_size
        self._eviction_policy = eviction_policy
        self._tensor_parallel = tensor_parallel
        self._precision_bytes = precision_bytes

        # Per-tier tracking: {tier: {"capacity": int, "used": int, "weight_bytes": int}}
        self._tier_state: Dict[MemoryTier, Dict] = {}
        for tc in tier_configs:
            self._tier_state[tc.tier] = {
                "capacity": tc.capacity_bytes,
                "used": 0,
                "weight_bytes": 0,
            }

        # Per-request block tracking: {request_id: [(tier, num_blocks), ...]}
        self._request_blocks: Dict[int, List[Tuple[MemoryTier, int]]] = {}

        # LRU tracking: OrderedDict of request_id for eviction ordering
        self._lru_order: OrderedDict = OrderedDict()

        # Resolve model config attributes
        num_kv_heads = getattr(model_config, 'num_key_value_heads', None)
        if num_kv_heads is None:
            num_kv_heads = getattr(model_config, 'num_attention_heads', 32)
        self._num_kv_heads = num_kv_heads

        head_dim = getattr(model_config, 'head_dim', None)
        if head_dim is None:
            hidden = getattr(model_config, 'hidden_size', 4096)
            n_heads = getattr(model_config, 'num_attention_heads', 32)
            head_dim = hidden // n_heads
        self._head_dim = head_dim

        self._num_layers = getattr(model_config, 'num_decoder_layers', 32)

    @property
    def bytes_per_token_kv(self) -> int:
        """KV cache bytes per token.

        Formula: 2 (K+V) * kv_heads * head_dim * layers * precision / TP
        """
        return (2 * self._num_kv_heads * self._head_dim
                * self._num_layers * self._precision_bytes // self._tensor_parallel)

    @property
    def bytes_per_block(self) -> int:
        """Bytes per KV block = bytes_per_token_kv * block_size."""
        return self.bytes_per_token_kv * self._block_size

    def load_weights(self, weight_bytes: int, tier: MemoryTier = MemoryTier.DEVICE_HBM) -> None:
        """Load model weights into a memory tier."""
        if tier not in self._tier_state:
            raise ValueError(f"Tier {tier} not configured")
        self._tier_state[tier]["used"] += weight_bytes
        self._tier_state[tier]["weight_bytes"] += weight_bytes

    def allocate_kv_blocks(
        self, request, num_tokens: int, tier: MemoryTier = MemoryTier.DEVICE_HBM
    ) -> Tuple[int, MemoryTier]:
        """Allocate KV cache blocks for a request.

        Tries the requested tier first, then falls back to lower tiers.
        Returns (blocks_allocated, tier_used).
        Raises MemoryError if no tier has sufficient capacity.
        """
        num_blocks = (num_tokens + self._block_size - 1) // self._block_size
        required_bytes = num_blocks * self.bytes_per_block

        # Build an ordered list of tiers to try: requested tier first, then hierarchy order
        tier_order = [MemoryTier.DEVICE_HBM, MemoryTier.HOST_DDR, MemoryTier.CXL, MemoryTier.NVME]
        candidates = []
        if tier in self._tier_state:
            candidates.append(tier)
        for t in tier_order:
            if t != tier and t in self._tier_state:
                candidates.append(t)

        for candidate_tier in candidates:
            available = self._tier_state[candidate_tier]["capacity"] - self._tier_state[candidate_tier]["used"]
            if available >= required_bytes:
                self._tier_state[candidate_tier]["used"] += required_bytes
                rid = getattr(request, 'request_id', request)
                if rid not in self._request_blocks:
                    self._request_blocks[rid] = []
                self._request_blocks[rid].append((candidate_tier, num_blocks))
                self._lru_order[rid] = True
                self._lru_order.move_to_end(rid)
                return num_blocks, candidate_tier

        raise MemoryError(
            f"Cannot allocate {num_blocks} blocks ({required_bytes} bytes) in any tier"
        )

    def deallocate_kv_blocks(self, request) -> int:
        """Free all KV cache blocks for a request. Returns total blocks freed."""
        rid = getattr(request, 'request_id', request)
        if rid not in self._request_blocks:
            return 0

        total_freed = 0
        for tier, num_blocks in self._request_blocks[rid]:
            freed_bytes = num_blocks * self.bytes_per_block
            self._tier_state[tier]["used"] -= freed_bytes
            total_freed += num_blocks

        del self._request_blocks[rid]
        if rid in self._lru_order:
            del self._lru_order[rid]
        return total_freed

    def evict_blocks(self, tier: MemoryTier, num_blocks: int) -> int:
        """Evict blocks from a tier using the configured policy.

        Returns the number of blocks actually evicted.
        """
        if tier not in self._tier_state:
            return 0

        evicted = 0
        if self._eviction_policy == EvictionPolicy.LRU:
            # Iterate from least recently used (oldest first)
            to_evict = list(self._lru_order.keys())
            for rid in to_evict:
                if evicted >= num_blocks:
                    break
                if rid not in self._request_blocks:
                    continue
                new_allocs = []
                for alloc_tier, alloc_blocks in self._request_blocks[rid]:
                    if alloc_tier == tier and evicted < num_blocks:
                        can_evict = min(alloc_blocks, num_blocks - evicted)
                        freed_bytes = can_evict * self.bytes_per_block
                        self._tier_state[tier]["used"] -= freed_bytes
                        evicted += can_evict
                        remaining = alloc_blocks - can_evict
                        if remaining > 0:
                            new_allocs.append((alloc_tier, remaining))
                    else:
                        new_allocs.append((alloc_tier, alloc_blocks))
                if new_allocs:
                    self._request_blocks[rid] = new_allocs
                else:
                    del self._request_blocks[rid]
                    if rid in self._lru_order:
                        del self._lru_order[rid]
        return evicted

    def spill_to_next_tier(
        self, from_tier: MemoryTier, num_blocks: int
    ) -> Tuple[int, MemoryTier]:
        """Move blocks from one tier to the next lower tier.

        Returns (blocks_moved, target_tier). If no lower tier is
        available, returns (0, from_tier).
        """
        tier_order = [MemoryTier.DEVICE_HBM, MemoryTier.HOST_DDR, MemoryTier.CXL, MemoryTier.NVME]

        try:
            idx = tier_order.index(from_tier)
        except ValueError:
            return 0, from_tier

        for next_idx in range(idx + 1, len(tier_order)):
            next_tier = tier_order[next_idx]
            if next_tier not in self._tier_state:
                continue

            available = self._tier_state[next_tier]["capacity"] - self._tier_state[next_tier]["used"]
            can_move = min(num_blocks, available // self.bytes_per_block)

            if can_move > 0:
                moved_bytes = can_move * self.bytes_per_block
                self._tier_state[from_tier]["used"] -= moved_bytes
                self._tier_state[next_tier]["used"] += moved_bytes
                return can_move, next_tier

        return 0, from_tier

    def memory_snapshot(self) -> Dict[str, Dict]:
        """Return current memory state for all tiers."""
        snapshot = {}
        for tier, state in self._tier_state.items():
            tc = self._tier_configs[tier]
            snapshot[tier.value] = {
                "capacity_bytes": state["capacity"],
                "used_bytes": state["used"],
                "free_bytes": state["capacity"] - state["used"],
                "utilization": state["used"] / state["capacity"] if state["capacity"] > 0 else 0.0,
                "weight_bytes": state["weight_bytes"],
                "kv_cache_bytes": state["used"] - state["weight_bytes"],
                "bandwidth_gbps": tc.bandwidth_gbps,
                "latency_ns": tc.latency_ns,
            }
        return snapshot

    @classmethod
    def from_system(
        cls,
        model_config,
        system,
        host_ddr_gb: float = 0,
        cxl_gb: float = 0,
        nvme_gb: float = 0,
        block_size: int = DEFAULT_BLOCK_SIZE,
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
        tensor_parallel: int = 1,
        precision_bytes: int = DEFAULT_PRECISION_BYTES,
    ) -> 'MemoryModel':
        """Create MemoryModel from GenZ System object.

        The System stores off_chip_mem_size in raw bytes (after unit conversion
        from MB) and offchip_mem_bw in raw bytes/sec (after unit conversion
        from GB/s).
        """
        tier_configs = []

        # HBM from system: off_chip_mem_size is in raw bytes, offchip_mem_bw
        # is in raw bytes/sec. Use the system's unit helpers to get human units.
        hbm_bytes = int(system.off_chip_mem_size)
        hbm_bw = float(system.get_offchip_mem_bw())
        tier_configs.append(MemoryTierConfig(
            tier=MemoryTier.DEVICE_HBM,
            capacity_bytes=hbm_bytes,
            bandwidth_gbps=hbm_bw,
        ))

        if host_ddr_gb > 0:
            tier_configs.append(MemoryTierConfig(
                tier=MemoryTier.HOST_DDR,
                capacity_bytes=int(host_ddr_gb * GB_TO_BYTES),
            ))

        if cxl_gb > 0:
            tier_configs.append(MemoryTierConfig(
                tier=MemoryTier.CXL,
                capacity_bytes=int(cxl_gb * GB_TO_BYTES),
            ))

        if nvme_gb > 0:
            tier_configs.append(MemoryTierConfig(
                tier=MemoryTier.NVME,
                capacity_bytes=int(nvme_gb * GB_TO_BYTES),
            ))

        return cls(
            model_config=model_config,
            tier_configs=tier_configs,
            block_size=block_size,
            eviction_policy=eviction_policy,
            tensor_parallel=tensor_parallel,
            precision_bytes=precision_bytes,
        )
