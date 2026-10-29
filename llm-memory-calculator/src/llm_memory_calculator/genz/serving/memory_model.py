"""Multi-tier memory model for KV cache management in LLM serving.

Tracks memory allocation across tiers (HBM, DDR, CXL, NVME),
handles KV block allocation/deallocation, and supports eviction
and spilling between tiers.

A request costs its KV blocks, which grow with its tokens, and fixed bytes that do not
(``ModelConfig.get_request_fixed_bytes``): in a model with recurrent layers (Mamba, Gated
DeltaNet, short convolution, ...) their per-sequence state, and in an encoder-decoder the
cross-attention cache over the encoder's output. They are charged once, with the request's
first blocks, on the same tier, and freed with the request.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
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


def _precision_bytes(precision: str) -> int:
    return 2 if precision in ("bf16", "fp16") else (1 if precision in ("int8", "fp8") else 4)


def model_parameter_count(model_config) -> int:
    """The parameters a serving device holds for ``model_config``.

    A ModelConfig read from a model config (the HF loader's, BudSimulator's converter's) carries
    it (``source_config``), and its parameters are the FRD-027 facts' count of it -- every stack
    (an encoder-decoder's encoder and cross-attention), every tower and adapter beside the
    language model, every routed expert -- the count the vLLM planner reserves before it sizes the
    KV. A ModelConfig built by hand (GenZ's built-in dictionary), or one the facts cannot read, is
    counted from its own fields (``_decoder_parameter_count``).

    The decoder fields alone are not the model: an encoder-decoder's ModelConfig holds its
    DECODER's depth and width, a multimodal model's its language model's, so counting them
    charged Whisper large-v3 1.81 GB of its 3.09 GB and Qwen3.5-4B 7.48 GB of 9.08, and the
    cluster admitted more requests than the device holds.
    """
    source = getattr(model_config, "source_config", None)
    if isinstance(source, dict):
        try:
            from llm_memory_calculator.model.facts import ModelFacts

            facts = ModelFacts.from_config(source)
            params = int(facts.params_total + facts.vision_params)
            if params > 0:
                return params
        except Exception:  # noqa: BLE001 - a config the facts cannot read is counted below
            pass
    return _decoder_parameter_count(model_config)


def _decoder_parameter_count(model_config) -> int:
    """A ModelConfig's parameters from its own fields, for one with no config to count (GenZ's
    built-in dictionary, a hand-built one): per attention layer Q / O at the heads and K / V at
    the KV heads; per Mamba layer (``layer_type``: "Mamba") its mixer (``mixer_params.mamba1_params``
    over the ``mamba_*`` fields: in / out projections, convolution, x / dt projections, A and D);
    a gated FFN on every layer of a stack that has attention layers -- on the layers it places
    experts ("MoE"), every routed expert at ``moe_intermediate_size``, its shared experts and the
    router -- and none in a stack of Mamba layers alone; the token embedding and an LM head (a
    ModelConfig states no tying: an upper bound for a tied model) and two norms a layer. An object
    with the decoder fields alone (no layer placement) is a dense attention stack.

    Counted as attention layers, GenZ's built-in Mamba models weighed the wrong mixer (Falcon-Mamba
    7B: 4.5B parameters, where its checkpoint holds 7.27B), and an MoE layer one expert (Mixtral
    8x7B: 7.1B of 46.7B)."""
    hidden = getattr(model_config, "hidden_size", 4096)
    layers = getattr(model_config, "num_decoder_layers", 32)
    intermediate = getattr(model_config, "intermediate_size", 11008)
    vocab = getattr(model_config, "vocab_size", 32000)
    n_heads = getattr(model_config, "num_attention_heads", 32)
    head_dim = getattr(model_config, "head_dim", None)
    if head_dim is None:
        head_dim = hidden // n_heads
    num_kv_heads = getattr(model_config, "num_key_value_heads", n_heads)

    layer_type = getattr(model_config, "layer_type", None)
    placed = isinstance(layer_type, list) and bool(layer_type)
    reps = layers / len(layer_type) if placed else 0

    def count(test):
        return int(round(sum(reps for lt in layer_type if test(lt)))) if placed else 0

    mamba_layers = count(lambda lt: lt[0] == "Mamba") if getattr(model_config, "is_mamba", False) else 0
    attn_layers = layers - mamba_layers
    attn_params = (
        2 * hidden * n_heads * head_dim +        # Q and O projections (full heads)
        2 * hidden * num_kv_heads * head_dim     # K and V projections (kv heads)
    ) * attn_layers
    mixer_params = 0
    if mamba_layers:
        from llm_memory_calculator.mixer_params import mamba1_params

        dims = {"expand": getattr(model_config, "mamba_expand", 2) or 2,
                "state_size": getattr(model_config, "mamba_d_state", 16) or 16,
                "conv_kernel": getattr(model_config, "mamba_d_conv", 4) or 4,
                "time_step_rank": getattr(model_config, "mamba_dt_rank", None)}
        mixer_params = mamba1_params(dims, hidden) * mamba_layers
    experts = int(getattr(model_config, "num_experts", 1) or 1)
    moe_layers = count(lambda lt: len(lt) > 1 and lt[1] == "MoE") if experts > 1 else 0
    ffn_layers = layers if attn_layers else 0
    dense_ffn = 3 * hidden * intermediate
    moe_ffn = 0
    if moe_layers and ffn_layers:
        expert_width = getattr(model_config, "moe_intermediate_size", None) or intermediate
        shared = int(getattr(model_config, "n_shared_experts", 0) or 0)
        shared_width = getattr(model_config, "shared_expert_intermediate_size", None) or intermediate
        moe_ffn = (experts * 3 * hidden * expert_width + shared * 3 * hidden * shared_width
                   + hidden * experts)  # the router
    else:
        moe_layers = 0
    ffn_params = dense_ffn * (ffn_layers - moe_layers) + moe_ffn * moe_layers
    if not placed:
        # the decoder fields alone: the formula callers built on (one embedding, no head, no norms)
        return int(attn_params + ffn_params + vocab * hidden)
    embed_params = 2 * vocab * hidden
    norms = (2 * layers + 1) * hidden
    return int(attn_params + mixer_params + ffn_params + embed_params + norms)


def estimate_weight_bytes(model_config, precision: str = "bf16", tensor_parallel: int = 1) -> int:
    """The model's weight bytes on one device: ``model_parameter_count`` at the serving precision
    (bf16 / fp16 2 B, int8 / fp8 1 B, else 4 B), sharded by ``tensor_parallel``.

    The one figure for the weights a serving simulation loads into the device tier
    (``ServingSimulator._estimate_weight_bytes``) and that the cluster and disaggregation ``B_eff``
    take off the device before they size the KV.
    """
    total_params = model_parameter_count(model_config)
    return total_params * _precision_bytes(precision) // max(tensor_parallel, 1)


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
        bytes_per_token_kv: Optional[int] = None,
        state_bytes_per_sequence: Optional[int] = None,
    ):
        """Build the model.

        ``bytes_per_token_kv`` is the per-rank KV cost of one token. Pass it from
        :func:`llm_memory_calculator.kv_tiering.kv_bytes_per_token_per_rank`, which reads
        ``ModelMemoryCalculator.kv_cache_breakdown`` -- the one source of per-token KV, aware of MLA, windowed layers
        and KV-free layers. Left out, the model falls back to its own GQA-only formula, kept only for existing callers.

        ``state_bytes_per_sequence`` is the per-rank fixed state of one request (recurrent layers).
        Left out, it is ``ModelConfig.get_request_fixed_bytes`` at ``precision_bytes``, split by TP.
        """
        self._bytes_per_token_kv_override = int(bytes_per_token_kv) if bytes_per_token_kv else None
        self._state_bytes_override = (
            int(state_bytes_per_sequence) if state_bytes_per_sequence is not None else None
        )
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
                "state_bytes": 0,
            }

        # Per-request block tracking: {request_id: [(tier, num_blocks), ...]}
        self._request_blocks: Dict[int, List[Tuple[MemoryTier, int]]] = {}
        # Per-request fixed state: {request_id: (tier, bytes)}
        self._request_state: Dict[int, Tuple[MemoryTier, int]] = {}

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

        Formula: ``ModelConfig.get_kv_size()`` (K+V elements per token over the layers that hold a
        cache) * precision / TP, unless the caller passed the calculator's figure (see ``__init__``).
        """
        if self._bytes_per_token_kv_override:
            return self._bytes_per_token_kv_override
        get_kv_size = getattr(self._model_config, 'get_kv_size', None)
        if callable(get_kv_size):
            # Only the layers that hold a cache (ModelConfig.get_kv_size), MLA as its latent.
            return int(get_kv_size() * self._precision_bytes // self._tensor_parallel)
        return (2 * self._num_kv_heads * self._head_dim
                * self._num_layers * self._precision_bytes // self._tensor_parallel)

    @property
    def state_bytes_per_sequence(self) -> int:
        """Fixed state bytes one request holds on this rank, whatever its length.

        ``ModelConfig.get_request_fixed_bytes`` / TP, unless the caller passed it: the recurrent
        layers' state (the matrix at its state dtype, the conv buffer at ``precision_bytes``)
        and an encoder-decoder's cross-attention cache over its encoder's stated output. 0 for a
        model with neither.
        """
        if self._state_bytes_override is None:
            # Computed once: the model and its precision do not change under the scheduler.
            # A request's fixed bytes: the recurrent state and an encoder-decoder's
            # cross-attention cache over its encoder's output (ModelConfig.get_request_fixed_bytes).
            fixed = getattr(self._model_config, 'get_request_fixed_bytes', None)
            if not callable(fixed):
                fixed = getattr(self._model_config, 'get_state_bytes', None)
            self._state_bytes_override = (
                int(fixed(self._precision_bytes) // max(self._tensor_parallel, 1))
                if callable(fixed) else 0
            )
        return self._state_bytes_override

    def request_bytes(self, num_tokens: int) -> int:
        """Bytes a new request of ``num_tokens`` tokens takes: its KV blocks and its state."""
        num_blocks = (num_tokens + self._block_size - 1) // self._block_size
        return num_blocks * self.bytes_per_block + self.state_bytes_per_sequence

    @property
    def bytes_per_block(self) -> int:
        """Bytes per KV block = bytes_per_token_kv * block_size."""
        return self.bytes_per_token_kv * self._block_size

    @property
    def primary_tier(self) -> MemoryTier:
        """Return the first (highest-priority) configured memory tier."""
        return next(iter(self._tier_state))

    @property
    def primary_capacity_bytes(self) -> int:
        """Total bytes of the primary (highest-priority) tier (weights + KV share it)."""
        return self._tier_state[self.primary_tier]["capacity"]

    @property
    def total_capacity_bytes(self) -> int:
        """Sum of every configured tier's capacity (HBM + DRAM + HOST_DDR + CXL + NVME).

        SV1/SV2: a request whose KV footprint exceeds this combined budget can never be
        admitted on any tier, so the scheduler fails it instead of re-queuing forever.
        """
        return sum(state["capacity"] for state in self._tier_state.values())

    def load_weights(self, weight_bytes: int, tier: MemoryTier = None) -> None:
        """Load model weights into a memory tier.

        If *tier* is ``None``, loads into the primary (first configured) tier.
        """
        if tier is None:
            tier = self.primary_tier
        if tier not in self._tier_state:
            raise ValueError(f"Tier {tier} not configured")
        self._tier_state[tier]["used"] += weight_bytes
        self._tier_state[tier]["weight_bytes"] += weight_bytes

    def allocate_kv_blocks(
        self, request, num_tokens: int, tier: MemoryTier = None
    ) -> Tuple[int, MemoryTier]:
        """Allocate KV cache blocks for a request.

        Tries the requested tier first, then falls back to lower tiers.
        If *tier* is ``None``, starts from the primary configured tier.
        Returns (blocks_allocated, tier_used).
        Raises MemoryError if no tier has sufficient capacity.
        """
        if tier is None:
            tier = self.primary_tier
        num_blocks = (num_tokens + self._block_size - 1) // self._block_size
        rid = getattr(request, 'request_id', request)
        # A request's first blocks bring its fixed state with them, on the same tier.
        state_bytes = 0 if rid in self._request_state else self.state_bytes_per_sequence
        required_bytes = num_blocks * self.bytes_per_block + state_bytes

        # Build an ordered list of tiers to try: requested tier first, then hierarchy order
        tier_order = [
            MemoryTier.DEVICE_HBM, MemoryTier.DEVICE_DRAM,
            MemoryTier.HOST_DDR, MemoryTier.CXL, MemoryTier.NVME,
        ]
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
                if state_bytes:
                    self._tier_state[candidate_tier]["state_bytes"] += state_bytes
                    self._request_state[rid] = (candidate_tier, state_bytes)
                if rid not in self._request_blocks:
                    self._request_blocks[rid] = []
                self._request_blocks[rid].append((candidate_tier, num_blocks))
                self._lru_order[rid] = True
                self._lru_order.move_to_end(rid)
                return num_blocks, candidate_tier

        raise MemoryError(
            f"Cannot allocate {num_blocks} blocks ({required_bytes} bytes) in any tier"
        )

    def _free_state(self, rid) -> None:
        """Release a request's fixed state, wherever it was placed."""
        placed = self._request_state.pop(rid, None)
        if placed is not None:
            tier, nbytes = placed
            self._tier_state[tier]["used"] -= nbytes
            self._tier_state[tier]["state_bytes"] -= nbytes

    def deallocate_kv_blocks(self, request) -> int:
        """Free all KV cache blocks for a request, and its state. Returns total blocks freed."""
        rid = getattr(request, 'request_id', request)
        self._free_state(rid)
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

    def evict_blocks(self, tier: MemoryTier, num_blocks: int) -> Tuple[int, List[int]]:
        """Evict blocks from a tier using the configured policy.

        Returns tuple of (blocks_evicted, list_of_fully_evicted_request_ids).
        """
        if tier not in self._tier_state:
            return 0, []

        evicted = 0
        evicted_request_ids = []
        if self._eviction_policy == EvictionPolicy.LRU:
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
                    # A request evicted whole loses its state too: it re-runs prefill.
                    self._free_state(rid)
                    evicted_request_ids.append(rid)
        return evicted, evicted_request_ids

    def spill_to_next_tier(
        self, from_tier: MemoryTier, num_blocks: int
    ) -> Tuple[int, MemoryTier]:
        """Move blocks from one tier to the next lower tier.

        Returns (blocks_moved, target_tier). If no lower tier is
        available, returns (0, from_tier).
        """
        tier_order = [
            MemoryTier.DEVICE_HBM, MemoryTier.DEVICE_DRAM,
            MemoryTier.HOST_DDR, MemoryTier.CXL, MemoryTier.NVME,
        ]

        try:
            idx = tier_order.index(from_tier)
        except ValueError:
            return 0, from_tier
        if self.bytes_per_block <= 0:
            return 0, from_tier  # no KV to move (a pure state-space model)

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
                "kv_cache_bytes": state["used"] - state["weight_bytes"] - state["state_bytes"],
                "state_bytes": state["state_bytes"],
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
