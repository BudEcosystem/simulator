# BudEvolve: CPU Inference Algorithm Report

## Summary

BudEvolve was used to analyze and create a CPU-optimized batch scheduling algorithm
for LLM inference. The analysis pipeline consisted of:

1. **Multi-CPU benchmarking** across 6 CPU platforms (Intel, AMD, ARM)
2. **NSGA-II multi-objective optimization** (1,483 evaluations, 50 generations)
3. **Sensitivity analysis** (config and hardware parameters)
4. **Roofline analysis** (compute vs memory bottleneck characterization)
5. **Algorithm creation** informed by all analytical results

## Key Findings

### CPU Inference is Compute-Bound

Unlike GPU decode (which is memory-bandwidth-bound), CPU inference is **100% compute-bound**
across all batch sizes and hardware platforms. Memory bandwidth utilization ranges from
5% (BS=32) to 30% (BS=1).

### Performance Comparison (Llama-3.1-8B, BF16)

| CPU | BS=1 Throughput | BS=1 TTFT | BS=32 Throughput | BS=32 TTFT |
|-----|----------------|-----------|-----------------|------------|
| Turin (AMD Zen5) | 42.7/s | 100ms | 1,178/s | 3,212ms |
| GraniteRapids (Intel) | 35.6/s | 114ms | 982/s | 3,660ms |
| Grace (NVIDIA ARM) | 35.6/s | 133ms | 912/s | 4,253ms |
| Graviton4 (AWS ARM) | 32.0/s | 246ms | 520/s | 7,868ms |
| EmeraldRapids (Intel) | 24.9/s | 209ms | 586/s | 6,697ms |
| SapphireRapids (Intel) | 12.8/s | 298ms | 354/s | 9,537ms |

### Optimal Configuration (INT8, TP=8, BS=32)

| CPU | Throughput | TTFT | TPOT |
|-----|-----------|------|------|
| Turin | 7,500/s | 268ms | 4.3ms |
| GraniteRapids | 6,948/s | 296ms | 4.6ms |

### Parameter Sensitivity Ranking

**Config parameters** (by throughput impact):
1. Tensor Parallel: delta=3,940 (most sensitive)
2. Precision: delta=982 (INT8 doubles throughput)
3. Batch Size: delta=946
4. Pipeline Parallel: delta=354 (hurts throughput)

**Hardware parameters**:
1. Memory Bandwidth: delta=737
2. FLOPS: delta=691
3. Memory Size: delta=0 (not a bottleneck for 8B model)
4. Interconnect: delta=0

### NSGA-II Pareto Front (Throughput vs Latency)

Best configs from 1,483 evaluated candidates:
- Best throughput: TP=8, PP=4, BS=256, FP8 -> 23,248 rps (TTFT=594ms)
- Best latency: TP=8, PP=1, BS=1, FP8 -> 214 rps (TTFT=11ms)
- Sweet spot: TP=8, PP=4, BS=32, INT8 -> 4,744 rps (TTFT=75ms, TPOT=3.9ms)

## Algorithm: CPU-Optimized Batch Scheduler

Location: `budevolve/evolve/base_algorithms/cpu_batch_scheduler.py`

### Design Decisions (informed by BudEvolve analysis)

1. **TTFT-aware batch sizing**: Since TTFT scales linearly with batch size (~114ms/request
   at 512 input tokens), the scheduler dynamically limits batch size based on TTFT SLO.

2. **KV cache budget from L3**: CPU has deep cache hierarchy. The scheduler reserves 60%
   of L3 for KV cache and 40% for weights/activations. Requests whose KV cache would
   exceed the L3 budget are deferred to avoid expensive DRAM spills.

3. **Priority-aware sorting**: Higher-priority requests are scheduled first. Tie-breaking
   favors shorter inputs (less TTFT impact) then earlier arrivals (fairness).

4. **NUMA-aware ordering**: Similar-length requests are grouped onto the same NUMA node.
   This reduces cross-node memory traffic during batched GEMM operations, which is
   critical on multi-socket CPUs (2-node on Intel, up to 8-node CCD topology on AMD).

5. **Greedy bin-packing**: Requests are packed respecting both total token budget and
   KV cache budget constraints, using a "skip if doesn't fit" approach.

### CPU Eviction Policy

Also provides `evict_entries_cpu()` for KV cache management:
- Score = (frequency x prefix_sharing_bonus x recency_weight) / (size / L3_budget)
- Entries with long shared prefixes get a bonus (high reuse probability)
- Large entries are penalized (consume precious L3 space)
- Evicts lowest-scored entries first

## Usage

```bash
# Run CPU scheduler evolution (requires TOGETHER_API_KEY or OPENAI_API_KEY)
budsim-evolve evolve-cpu-scheduler \
    --model meta-llama/Meta-Llama-3.1-8B \
    --hardware GraniteRapids_CPU \
    --iterations 100

# Run CPU sensitivity analysis
budsim-evolve sensitivity \
    --model meta-llama/Meta-Llama-3.1-8B \
    --hardware GraniteRapids_CPU \
    --mode both

# Run multi-objective optimization
budsim-evolve optimize \
    --model meta-llama/Meta-Llama-3.1-8B \
    --hardware GraniteRapids_CPU \
    --objectives throughput,latency \
    --generations 50
```

## Recommendations

1. **Use INT8 quantization** - doubles throughput with AMX (Intel) or AVX-512
2. **Maximize tensor parallelism** - TP=8 across sockets
3. **Never use pipeline parallelism** on CPU (PP=1 always)
4. **Batch size 32-64** for throughput/latency balance
5. **Target Turin (AMD)** for best absolute performance
6. **Use GraniteRapids (Intel)** for best AMX acceleration
7. **Keep KV cache in L3** to avoid DRAM latency penalties
