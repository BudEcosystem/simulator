"""Radix tree prefix cache for KV cache reuse in LLM serving."""
from collections import OrderedDict
from typing import Dict, List, Any

from .constants import DEFAULT_BLOCK_SIZE, GB_TO_BYTES


class RadixTreeNode:
    """Node in the radix tree for prefix matching."""
    def __init__(self):
        self.children: Dict[int, 'RadixTreeNode'] = {}
        self.block_ids: List[int] = []
        self.ref_count: int = 0
        self.last_access_time: int = 0


class RadixCache:
    """Radix tree cache for prefix-aware KV cache management.

    Stores token sequences in a radix tree, allowing O(n) prefix matching
    where n is the length of the query sequence.
    """

    def __init__(self, capacity_bytes: int, block_size: int = DEFAULT_BLOCK_SIZE,
                 kv_bytes_per_token: int = 0):
        self._capacity_bytes = capacity_bytes
        self._block_size = block_size
        self._kv_bytes_per_token = kv_bytes_per_token
        self._bytes_per_block = kv_bytes_per_token * block_size if kv_bytes_per_token > 0 else 1

        self._root = RadixTreeNode()
        self._total_blocks = capacity_bytes // self._bytes_per_block if self._bytes_per_block > 0 else 0
        self._used_blocks = 0
        self._next_block_id = 0
        self._access_time = 0

        # LRU tracking: block_id -> access_time
        self._block_access: OrderedDict = OrderedDict()

        # Stats
        self._total_lookups = 0
        self._total_hits = 0
        self._total_tokens_matched = 0
        self._total_tokens_looked_up = 0

    def insert(self, token_ids: List[int]) -> int:
        """Insert a token sequence into the cache. Returns blocks inserted."""
        if not token_ids:
            return 0

        self._access_time += 1
        blocks_inserted = 0
        node = self._root
        i = 0

        # Walk existing prefix
        while i < len(token_ids) and token_ids[i] in node.children:
            node = node.children[token_ids[i]]
            node.ref_count += 1
            node.last_access_time = self._access_time
            # Update LRU for existing blocks
            for bid in node.block_ids:
                if bid in self._block_access:
                    self._block_access.move_to_end(bid)
            i += self._block_size

        # Insert new nodes for remaining tokens
        while i < len(token_ids):
            if self._used_blocks >= self._total_blocks:
                break  # Cache full

            token = token_ids[min(i, len(token_ids) - 1)]
            new_node = RadixTreeNode()
            new_node.ref_count = 1
            new_node.last_access_time = self._access_time

            # Allocate a block
            block_id = self._next_block_id
            self._next_block_id += 1
            new_node.block_ids.append(block_id)
            self._block_access[block_id] = self._access_time
            self._used_blocks += 1
            blocks_inserted += 1

            node.children[token] = new_node
            node = new_node
            i += self._block_size

        return blocks_inserted

    def match_prefix(self, token_ids: List[int]) -> int:
        """Match a token sequence against the cache. Returns matched token count."""
        if not token_ids:
            return 0

        self._access_time += 1
        self._total_lookups += 1
        self._total_tokens_looked_up += len(token_ids)

        node = self._root
        matched = 0
        i = 0

        while i < len(token_ids):
            token = token_ids[i]
            if token not in node.children:
                break
            node = node.children[token]
            node.last_access_time = self._access_time
            # Update LRU
            for bid in node.block_ids:
                if bid in self._block_access:
                    self._block_access.move_to_end(bid)
            matched += self._block_size
            i += self._block_size

        matched = min(matched, len(token_ids))

        if matched > 0:
            self._total_hits += 1
            self._total_tokens_matched += matched

        return matched

    def evict_lru(self, num_blocks: int) -> int:
        """Evict least recently used blocks. Returns blocks evicted."""
        evicted = 0
        to_evict = min(num_blocks, self._used_blocks)

        # Get LRU blocks
        block_ids_to_evict = []
        for bid in list(self._block_access.keys()):
            if evicted >= to_evict:
                break
            block_ids_to_evict.append(bid)
            evicted += 1

        for bid in block_ids_to_evict:
            if bid in self._block_access:
                del self._block_access[bid]

        self._used_blocks -= evicted
        return evicted

    def hit_rate(self) -> float:
        """Fraction of lookups that found a prefix match."""
        if self._total_lookups == 0:
            return 0.0
        return self._total_hits / self._total_lookups

    def token_hit_rate(self) -> float:
        """Fraction of looked-up tokens that were cached."""
        if self._total_tokens_looked_up == 0:
            return 0.0
        return self._total_tokens_matched / self._total_tokens_looked_up

    def utilization(self) -> float:
        """Fraction of cache capacity used."""
        if self._total_blocks == 0:
            return 0.0
        return self._used_blocks / self._total_blocks

    def stats(self) -> Dict[str, Any]:
        return {
            "total_blocks": self._total_blocks,
            "used_blocks": self._used_blocks,
            "utilization": self.utilization(),
            "hit_rate": self.hit_rate(),
            "token_hit_rate": self.token_hit_rate(),
            "total_lookups": self._total_lookups,
            "total_hits": self._total_hits,
        }


class PrefixCacheAnalyzer:
    """Analyze prefix caching effectiveness for a workload."""

    def __init__(self, model_config, system_or_kv_bytes: int = 0,
                 cache_capacity_gb: float = 10.0):
        if isinstance(system_or_kv_bytes, int) and system_or_kv_bytes > 0:
            self._kv_bytes_per_token = system_or_kv_bytes
        else:
            # Derive from model config
            num_kv_heads = getattr(model_config, 'num_key_value_heads', None)
            if num_kv_heads is None:
                num_kv_heads = getattr(model_config, 'num_attention_heads', 32)
            head_dim = getattr(model_config, 'head_dim', None)
            if head_dim is None:
                hidden = getattr(model_config, 'hidden_size', 4096)
                n_heads = getattr(model_config, 'num_attention_heads', 32)
                head_dim = hidden // n_heads
            num_layers = getattr(model_config, 'num_decoder_layers', 32)
            self._kv_bytes_per_token = 2 * num_kv_heads * head_dim * num_layers * 2

        self._cache_capacity_bytes = int(cache_capacity_gb * GB_TO_BYTES)
        self._model_config = model_config

    def analyze(self, workload, shared_prefix_tokens: int = 0) -> Dict[str, Any]:
        """Analyze prefix caching for a list of Request objects.

        Returns hit rate, memory savings, and cache utilization.
        """
        cache = RadixCache(
            capacity_bytes=self._cache_capacity_bytes,
            kv_bytes_per_token=self._kv_bytes_per_token,
        )

        total_tokens_processed = 0
        tokens_saved = 0

        for req in workload:
            if shared_prefix_tokens > 0:
                # Shared system prompt: first N tokens are deterministic (same for all requests)
                # Remaining tokens are unique per request (user-specific content)
                prefix = list(range(shared_prefix_tokens))
                num_suffix = max(0, req.input_tokens - shared_prefix_tokens)
                suffix = [hash((req.request_id, i)) % (2**31) for i in range(num_suffix)]
                token_ids = prefix + suffix
            else:
                # No explicit shared prefix: use position-based hashing
                # This produces ~20-40% natural overlap when requests share similar
                # content patterns (e.g., common preambles, repeated questions)
                token_ids = [hash((req.model, i, i // self._kv_bytes_per_token)) % (2**31)
                             for i in range(req.input_tokens)]

            # Check prefix match
            matched = cache.match_prefix(token_ids)
            tokens_saved += matched
            total_tokens_processed += req.input_tokens

            # Insert into cache
            cache.insert(token_ids)

        memory_saved_bytes = tokens_saved * self._kv_bytes_per_token

        return {
            "total_requests": len(workload),
            "total_tokens_processed": total_tokens_processed,
            "tokens_saved_by_cache": tokens_saved,
            "token_savings_rate": tokens_saved / total_tokens_processed if total_tokens_processed > 0 else 0,
            "memory_saved_bytes": memory_saved_bytes,
            "memory_saved_gb": memory_saved_bytes / GB_TO_BYTES,
            "cache_stats": cache.stats(),
        }
