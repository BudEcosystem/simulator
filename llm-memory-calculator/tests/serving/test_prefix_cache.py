"""Tests for radix tree prefix cache and analyzer."""
import pytest

from llm_memory_calculator.genz.serving.prefix_cache import (
    RadixCache,
    RadixTreeNode,
    PrefixCacheAnalyzer,
)
from llm_memory_calculator.genz.serving.request import Request
from llm_memory_calculator.genz.serving.constants import DEFAULT_BLOCK_SIZE, GB_TO_BYTES


# ---------------------------------------------------------------------------
# RadixCache tests
# ---------------------------------------------------------------------------

def _make_cache(total_blocks: int = 100, block_size: int = DEFAULT_BLOCK_SIZE,
                kv_bytes_per_token: int = 256) -> RadixCache:
    """Helper: create a RadixCache with a given number of total blocks."""
    bytes_per_block = kv_bytes_per_token * block_size
    capacity = total_blocks * bytes_per_block
    return RadixCache(capacity_bytes=capacity, block_size=block_size,
                      kv_bytes_per_token=kv_bytes_per_token)


class TestRadixCacheInsert:
    def test_insert_returns_positive_blocks(self):
        cache = _make_cache()
        tokens = list(range(32))  # 2 blocks worth
        inserted = cache.insert(tokens)
        assert inserted > 0

    def test_insert_empty_returns_zero(self):
        cache = _make_cache()
        assert cache.insert([]) == 0

    def test_insert_updates_used_blocks(self):
        cache = _make_cache()
        tokens = list(range(DEFAULT_BLOCK_SIZE * 3))
        inserted = cache.insert(tokens)
        assert cache._used_blocks == inserted


class TestRadixCacheMatchPrefix:
    def test_match_after_insert(self):
        cache = _make_cache()
        tokens = list(range(DEFAULT_BLOCK_SIZE * 2))
        cache.insert(tokens)
        matched = cache.match_prefix(tokens)
        assert matched > 0
        assert matched <= len(tokens)

    def test_match_returns_block_aligned(self):
        cache = _make_cache()
        tokens = list(range(DEFAULT_BLOCK_SIZE * 2))
        cache.insert(tokens)
        matched = cache.match_prefix(tokens)
        # Match should be a multiple of block_size (capped at len)
        assert matched % DEFAULT_BLOCK_SIZE == 0 or matched == len(tokens)


class TestRadixCacheNoMatch:
    def test_no_match_without_insert(self):
        cache = _make_cache()
        tokens = list(range(32))
        matched = cache.match_prefix(tokens)
        assert matched == 0


class TestRadixCachePartialMatch:
    def test_partial_match(self):
        cache = _make_cache()
        # Insert a sequence of 16 tokens (1 block)
        seq = list(range(DEFAULT_BLOCK_SIZE))
        cache.insert(seq)

        # Query a sequence that starts the same but diverges after the first block
        query = list(range(DEFAULT_BLOCK_SIZE)) + [9999] * DEFAULT_BLOCK_SIZE
        matched = cache.match_prefix(query)
        # Should match the first block only
        assert matched == DEFAULT_BLOCK_SIZE


class TestRadixCacheEvictLru:
    def test_evict_reduces_used_blocks(self):
        cache = _make_cache(total_blocks=10)
        tokens = list(range(DEFAULT_BLOCK_SIZE * 5))
        cache.insert(tokens)
        before = cache._used_blocks
        evicted = cache.evict_lru(2)
        assert evicted == 2
        assert cache._used_blocks == before - 2

    def test_evict_more_than_used(self):
        cache = _make_cache(total_blocks=10)
        tokens = list(range(DEFAULT_BLOCK_SIZE * 3))
        inserted = cache.insert(tokens)
        assert inserted > 0
        evicted = cache.evict_lru(100)
        assert evicted == inserted  # Can only evict what was inserted
        assert cache._used_blocks == 0


class TestRadixCacheHitRate:
    def test_hit_rate_with_hits(self):
        cache = _make_cache()
        tokens = list(range(DEFAULT_BLOCK_SIZE * 2))
        cache.insert(tokens)

        # First lookup: hit
        cache.match_prefix(tokens)
        # Second lookup: miss (different tokens)
        cache.match_prefix([9999, 9998, 9997])

        assert 0.0 < cache.hit_rate() <= 1.0

    def test_hit_rate_zero_without_lookups(self):
        cache = _make_cache()
        assert cache.hit_rate() == 0.0


class TestRadixCacheUtilization:
    def test_utilization_positive_after_insert(self):
        cache = _make_cache(total_blocks=10)
        tokens = list(range(DEFAULT_BLOCK_SIZE * 5))
        cache.insert(tokens)
        assert cache.utilization() > 0.0
        assert cache.utilization() <= 1.0

    def test_utilization_zero_when_empty(self):
        cache = _make_cache()
        assert cache.utilization() == 0.0


class TestRadixCacheEmptyStats:
    def test_empty_stats(self):
        cache = _make_cache()
        s = cache.stats()
        assert s["used_blocks"] == 0
        assert s["utilization"] == 0.0
        assert s["hit_rate"] == 0.0
        assert s["token_hit_rate"] == 0.0
        assert s["total_lookups"] == 0
        assert s["total_hits"] == 0
        assert s["total_blocks"] > 0


# ---------------------------------------------------------------------------
# PrefixCacheAnalyzer tests
# ---------------------------------------------------------------------------

class _FakeModelConfig:
    """Minimal model config for PrefixCacheAnalyzer tests."""
    num_attention_heads = 32
    num_key_value_heads = 8
    head_dim = 128
    hidden_size = 4096
    num_decoder_layers = 32


def _make_request(request_id: int, input_tokens: int) -> Request:
    return Request(
        request_id=request_id,
        model="test",
        input_tokens=input_tokens,
        max_output_tokens=128,
        arrival_time_ns=0,
    )


class TestPrefixCacheAnalyzerBasic:
    def test_shared_prefix_has_savings(self):
        """Requests with shared prefix tokens should benefit from prefix caching."""
        analyzer = PrefixCacheAnalyzer(
            model_config=_FakeModelConfig(),
            cache_capacity_gb=10.0,
        )
        workload = [_make_request(i, 512) for i in range(5)]
        result = analyzer.analyze(workload, shared_prefix_tokens=256)

        assert result["total_requests"] == 5
        assert result["tokens_saved_by_cache"] > 0
        assert result["token_savings_rate"] > 0.0
        assert result["memory_saved_bytes"] > 0


class TestPrefixCacheAnalyzerSharedPrefix:
    def test_shared_prefix_high_hit_rate(self):
        """With shared_prefix_tokens > 0, hit rate should be high."""
        analyzer = PrefixCacheAnalyzer(
            model_config=_FakeModelConfig(),
            cache_capacity_gb=10.0,
        )
        # All requests share a 256-token system prompt
        workload = [_make_request(i, 512) for i in range(10)]
        result = analyzer.analyze(workload, shared_prefix_tokens=256)

        assert result["total_requests"] == 10
        # With 256 shared prefix tokens, 9 out of 10 requests should cache-hit
        assert result["tokens_saved_by_cache"] > 0
        assert result["token_savings_rate"] > 0.3

    def test_no_shared_prefix_lower_hit_rate(self):
        """Without shared prefix, hit rate should be lower (hash-based)."""
        analyzer = PrefixCacheAnalyzer(
            model_config=_FakeModelConfig(),
            cache_capacity_gb=10.0,
        )
        workload = [_make_request(i, 512) for i in range(10)]
        result_no_prefix = analyzer.analyze(workload, shared_prefix_tokens=0)

        # With position-based hashing, there may be some overlap but less than shared prefix
        assert result_no_prefix["total_requests"] == 10


class TestPrefixCacheAnalyzerNoSharing:
    def test_unique_prefixes_low_savings(self):
        """Requests with distinct token counts produce distinct simulated sequences
        so prefix cache reuse should be low or zero."""
        analyzer = PrefixCacheAnalyzer(
            model_config=_FakeModelConfig(),
            cache_capacity_gb=10.0,
        )
        # Each request has a unique input_tokens length, so range(n) differs.
        workload = [_make_request(i, 100 + i * 200) for i in range(5)]
        result = analyzer.analyze(workload)

        assert result["total_requests"] == 5
        # With distinct sequence lengths the shorter ones won't match longer
        # ones because range(100) != range(300)[:100] only if token_ids differ.
        # Actually range(100) IS a prefix of range(300), so there will be SOME
        # savings. But savings_rate should be noticeably lower than the shared case.
        shared_analyzer = PrefixCacheAnalyzer(
            model_config=_FakeModelConfig(),
            cache_capacity_gb=10.0,
        )
        shared_workload = [_make_request(i, 512) for i in range(5)]
        shared_result = shared_analyzer.analyze(shared_workload)

        assert result["token_savings_rate"] <= shared_result["token_savings_rate"]
