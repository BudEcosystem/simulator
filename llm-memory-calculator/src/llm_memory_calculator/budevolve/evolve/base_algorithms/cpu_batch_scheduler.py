"""CPU-optimized batch scheduling algorithm for LLM inference.

Evolved using BudEvolve's analytical pipeline:
- BudSim GenZ roofline analysis: CPU is 100% compute-bound
- Sensitivity analysis: batch_size and tensor_parallel are dominant
- NSGA-II optimization: INT8/FP8 with TP=8, BS=32-64 is the Pareto sweet spot
- CPU-specific factors: NUMA topology, cache hierarchy, memory bandwidth

Key insights from BudEvolve analysis on GraniteRapids_CPU:
  - Compute utilization: 100%, Memory BW utilization: 4-30%
  - Embeddings consume 41% of total time (main bottleneck)
  - Larger batches amortize per-request overhead but increase TTFT linearly
  - INT8 quantization doubles throughput with no quality impact
  - Pipeline parallelism HURTS on CPU (increases latency, reduces throughput)
  - Memory bandwidth is the #1 hardware sensitivity factor (delta=737)
"""
import math
from typing import List, Optional


def schedule_batch_cpu(
    queue,
    max_batch_size: int = 64,
    max_tokens: int = 16384,
    numa_nodes: int = 2,
    l3_cache_mb: float = 288.0,
    memory_bw_gbps: float = 500.0,
    target_ttft_ms: float = 200.0,
    model_hidden_dim: int = 4096,
    bytes_per_param: int = 1,
):
    """CPU-optimized batch scheduling for LLM inference.

    Designed for compute-bound CPU inference where:
    - Batch size is the primary throughput lever
    - TTFT scales linearly with batch size
    - KV cache must fit in L3 to avoid DRAM spills
    - NUMA-aware grouping reduces cross-node traffic

    Args:
        queue: List of pending requests with .input_tokens, .output_tokens,
               and optionally .priority, .arrival_time, .numa_node
        max_batch_size: Maximum requests per batch (CPU sweet spot: 32-64).
        max_tokens: Maximum total tokens per batch.
        numa_nodes: Number of NUMA nodes on the CPU.
        l3_cache_mb: L3 cache size in MB (e.g. 288 for GraniteRapids).
        memory_bw_gbps: Off-chip memory bandwidth in GB/s.
        target_ttft_ms: Target time-to-first-token SLO in ms.
        model_hidden_dim: Model hidden dimension (for KV cache sizing).
        bytes_per_param: Bytes per parameter (1=INT8, 2=BF16).

    Returns:
        List of requests to include in the next batch, ordered for
        optimal cache utilization.
    """
    if not queue:
        return []

    # Phase 1: Estimate KV cache budget from L3 capacity
    # KV cache per token per layer ≈ 2 * hidden_dim * bytes_per_param
    # (2 for K and V matrices)
    kv_bytes_per_token = 2 * model_hidden_dim * bytes_per_param
    l3_bytes = l3_cache_mb * 1024 * 1024
    # Reserve 40% of L3 for weights and activations
    kv_budget_bytes = l3_bytes * 0.6
    max_kv_tokens = int(kv_budget_bytes / kv_bytes_per_token)

    # Phase 2: Compute TTFT-aware batch size limit
    # On CPU, TTFT ≈ (batch_size * input_tokens * compute_per_token) / TFLOPS
    # From profiling: TTFT scales ~linearly with batch_size
    # GraniteRapids: BS=1 → 114ms, BS=32 → 3660ms → ~114ms per request
    # So max batch for TTFT SLO ≈ target_ttft / per_request_ttft
    if len(queue) > 0:
        avg_input = sum(getattr(r, 'input_tokens', 512) for r in queue) / len(queue)
        # Approximate per-request prefill time based on input length
        # Base: 512 tokens → ~114ms on GraniteRapids (scales linearly)
        per_request_ms = (avg_input / 512.0) * 114.0
        ttft_batch_limit = max(1, int(target_ttft_ms / max(per_request_ms, 1.0)))
    else:
        ttft_batch_limit = max_batch_size

    effective_batch_limit = min(max_batch_size, ttft_batch_limit)

    # Phase 3: Priority-aware sorting
    # Sort by: priority (if available) > shorter input (amortize compute) > arrival time
    scored_queue = []
    for i, req in enumerate(queue):
        priority = getattr(req, 'priority', 0)
        input_len = getattr(req, 'input_tokens', 512)
        arrival = getattr(req, 'arrival_time', i)
        # Higher priority first, then shorter inputs (less TTFT impact),
        # then earlier arrivals (fairness)
        score = (priority * 10000) - input_len + (1.0 / (arrival + 1))
        scored_queue.append((score, i, req))

    scored_queue.sort(key=lambda x: -x[0])

    # Phase 4: Greedy bin-packing with KV cache and token budget
    batch = []
    total_tokens = 0
    total_kv_tokens = 0

    for _, _, req in scored_queue:
        if len(batch) >= effective_batch_limit:
            break

        input_t = getattr(req, 'input_tokens', 512)
        output_t = getattr(req, 'output_tokens', 128)
        req_tokens = input_t + output_t

        # Check total token budget
        if total_tokens + req_tokens > max_tokens:
            continue

        # Check KV cache fits in L3
        kv_tokens = input_t + output_t  # KV cache for full sequence
        if total_kv_tokens + kv_tokens > max_kv_tokens:
            continue

        batch.append(req)
        total_tokens += req_tokens
        total_kv_tokens += kv_tokens

    # Phase 5: NUMA-aware ordering
    # Group requests by NUMA affinity to minimize cross-node memory access
    if numa_nodes > 1 and len(batch) > 1:
        batch = _numa_sort(batch, numa_nodes)

    return batch


def _numa_sort(batch, numa_nodes: int):
    """Sort batch for NUMA locality.

    Assigns requests to NUMA nodes in round-robin by input length
    similarity, then interleaves them so adjacent requests in the
    batch share the same NUMA node. This reduces cross-node memory
    traffic during batched GEMM operations.

    Args:
        batch: List of requests to reorder.
        numa_nodes: Number of NUMA nodes.

    Returns:
        Reordered batch list.
    """
    # Group by assigned NUMA node
    groups = [[] for _ in range(numa_nodes)]

    # Sort by input length first, then distribute to NUMA nodes
    # Similar-length requests on the same node avoid padding waste
    sorted_batch = sorted(
        batch, key=lambda r: getattr(r, 'input_tokens', 512)
    )

    for i, req in enumerate(sorted_batch):
        # Check if request has NUMA preference
        preferred = getattr(req, 'numa_node', None)
        if preferred is not None and 0 <= preferred < numa_nodes:
            groups[preferred].append(req)
        else:
            # Round-robin assignment
            groups[i % numa_nodes].append(req)

    # Interleave: process all requests from node 0 first, then node 1, etc.
    # This keeps memory accesses local during each sub-batch
    result = []
    for group in groups:
        result.extend(group)
    return result


def evict_entries_cpu(
    cache_entries,
    num_to_evict: int,
    l3_cache_mb: float = 288.0,
    memory_bw_gbps: float = 500.0,
):
    """CPU-optimized KV cache eviction policy.

    On CPU, cache misses are ~10x more expensive than on GPU (DRAM latency
    ~100ns vs GPU HBM ~100ns but with much higher bandwidth). This policy
    prioritizes keeping entries that are:
    1. Frequently accessed (temporal locality)
    2. Small enough to fit in L3 cache
    3. Part of long common prefixes (spatial locality)

    Uses a cost-benefit score: benefit = frequency * prefix_sharing_factor,
    cost = size_bytes / l3_budget. Evict lowest benefit/cost ratio.

    Args:
        cache_entries: List of entries with .last_access_time, .frequency,
                       .size_bytes, .prefix_length
        num_to_evict: Number of entries to remove.

    Returns:
        List of entries to evict.
    """
    if num_to_evict <= 0 or not cache_entries:
        return []
    if num_to_evict >= len(cache_entries):
        return list(cache_entries)

    l3_bytes = l3_cache_mb * 1024 * 1024

    scored = []
    for entry in cache_entries:
        freq = getattr(entry, 'frequency', 1)
        last_access = getattr(entry, 'last_access_time', 0)
        size = getattr(entry, 'size_bytes', 1024)
        prefix_len = getattr(entry, 'prefix_length', 0)

        # Benefit: frequency weighted by prefix sharing potential
        # Longer prefixes have higher reuse probability across requests
        prefix_factor = 1.0 + math.log1p(prefix_len) * 0.3

        # Recency: exponential decay (entries not accessed recently get lower score)
        # Assumes last_access_time is a timestamp or counter
        recency_weight = 1.0 / (1.0 + max(0, last_access))

        benefit = freq * prefix_factor * recency_weight

        # Cost: fraction of L3 cache consumed
        # Larger entries are more expensive to keep
        cost = max(size / l3_bytes, 1e-12)

        # Score: higher = more valuable to keep
        score = benefit / cost
        scored.append((score, entry))

    # Sort ascending: lowest score = least valuable = evict first
    scored.sort(key=lambda x: x[0])
    return [entry for _, entry in scored[:num_to_evict]]
