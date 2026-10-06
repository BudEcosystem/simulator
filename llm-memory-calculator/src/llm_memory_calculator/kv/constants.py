"""Tunables of the KV tier planner (Bud FRD-023), each with where its value comes from.

``[MEASURE]`` marks a default no measurement backs yet. The values marked measured are checked
against the runs they came from by :mod:`llm_memory_calculator.kv.validate`. Bandwidths are decimal
GB/s (1e9 bytes per second); sizes are GiB.
"""

from typing import Dict, FrozenSet, Tuple

GIB = 1 << 30
MIB = 1 << 20
KIB = 1 << 10
GB = 10**9
PLAN_VERSION = 1

# ----------------------------------------------------------------------------- release ceiling
# The tiers a plan returns. T0 is always planned. Every tier is evaluated and tested; the ones not
# listed here are withheld from the plan (and named in its decisions) until budcluster can render
# them. Raising this is a package release, never configuration (FRD-023 §7).
RELEASED_TIERS: FrozenSet[str] = frozenset({"T1", "T2"})
OFFLOAD_TIERS: Tuple[str, ...] = ("T1", "T2", "T3")

# -------------------------------------------------------------------------------------------- T0
# A dedicated card's KV pool gets what the plan doesn't need, up to this fraction (FR-T0-1; the
# same floor budcluster applies).
T0_GPU_UTIL = 0.92
GPU_UTIL_MAX = 0.94
# Non-torch memory vLLM counts inside its fraction (CUDA context, graph pool). [MEASURE] per image.
DEVICE_OVERHEAD_GIB = 1.0
# FR-T0-3: a shared slice's KV part grows by at most this share of the GPU's free memory. [MEASURE]
SLICE_KV_SHARE_MAX = 0.5
# FR-T0-4: a CPU engine's KV space grows by at most this share of the node's free host memory.
# [MEASURE]
CPU_KV_SHARE_MAX = 0.5

# ------------------------------------------------------------------------------------ break-even
# A tier is kept only if reloading the cached prefix takes under this share of recomputing it
# (ARCHITECTURE §10.3 rule 7).
BREAK_EVEN_RATIO = 0.8
# ...and only while the deployment's loads and write-through stay under this share of the tier's
# link. Below it, a load waiting its turn costs that request time but not throughput: the GPU serves
# other requests meanwhile (TCS accuracy grid, 2026-10-01: Qwen3-0.6B at 16K gained 29-37% from T1
# with the link ~50% busy, which a queueing-inflated reload had judged a loss). [MEASURE]
MAX_LINK_UTILISATION = 0.8

# -------------------------------------------------------------------------------------------- T1
# FR-T1-1: T1 = clamp(working set, 3x, 10x the GPU KV pool), within the node's free host memory.
T1_MIN_POOL_MULTIPLE = 3.0
T1_MAX_POOL_MULTIPLE = 10.0
# At most this share of the node's free host memory, shared by the replicas pinned to it. [MEASURE]
T1_HOST_RAM_SHARE = 0.5
# Offload chunk, in tokens. Spike S2 (H100): 16 -> 256 took a 32K-token hit from 5.5x to 7.9x.
T1_BLOCK_SIZE = 256
# A T1 load reaches this share of the probed pinned-copy rate. S2 (TCS H100): ~47 of 55 GB/s.
# accubits01 (x4 link) reached ~3 of 4.7 GB/s; its fixture checks the effect on the decision.
T1_LOAD_EFFICIENCY = 0.85
# Fixed cost of a T1 load beyond the bytes. S2: a 32K hit took 155 ms, ~101 ms of it transfer and
# ~25 ms the GPU-hit baseline. [MEASURE]
T1_FLOOR_MS = 15.0
# alloc mode (pinned host memory the driver allocates): the TCS accuracy runs' T1 loads fit 11 ms +
# bytes / 40 GB/s against a 55.2 GB/s probe (23 single requests, Qwen3 0.6B-32B and others,
# 1K-120K). Measured with spike S2's stand-in, which copies each layer's buffer separately;
# re-measure on the Bud engine's alloc patch (one region). [MEASURE]
T1_ALLOC_FLOOR_MS = 11.0
T1_ALLOC_LOAD_EFFICIENCY = 0.72
# register mode on a GPU that can't use registered memory at its host address, with every copy its
# own cudaMemcpyAsync (T1_PER_COPY_FEATURE): the TCS disk-tier round's T1 loads fit 15 ms + bytes /
# 18.5 GB/s against the 55.2 GB/s probe (Qwen3-8B 4K and 16K, Qwen2.5-72B-AWQ 4K). Measured with the
# spike's Python stand-in (one ctypes call per block and layer); re-measure on the fork's native
# version. [MEASURE]
T1_PER_COPY_FLOOR_MS = 15.0
T1_PER_COPY_LOAD_EFFICIENCY = 0.34
# T1 pays only if a hit saves at least this much batched GPU work (the cached prefix's prefill FLOPs
# at PREFILL_MFU over the model's active parameters). The tier's own store and load copies stall
# decode on every request; below this, they cost more than the recompute they save. TCS accuracy
# runs: every cell saving <= 19 ms per hit lost or tied (Qwen3-0.6B/Qwen2.5-1.5B at 1K,
# Qwen3-30B-A3B at 1K -18 to -28%, Phi-4-mini 1K, Qwen3-8B 512), every cell saving >= 28 ms won.
# [MEASURE]
T1_MIN_RECOMPUTE_SAVED_MS = 25.0
# MoE models need more: Qwen3-30B-A3B at 2K (36 ms) and Qwen1.5-MoE-A2.7B at 4K (44 ms) lost or tied
# at c4, while every MoE cell from 75 ms won (rounds 3-4). The extra cost isn't modelled yet.
# [MEASURE]
T1_MIN_RECOMPUTE_SAVED_MS_MOE = 60.0
# Pinned-copy bandwidth (GB/s) by PCIe generation, when the node has no measurement.
PCIE_GBPS_BY_GEN: Dict[int, float] = {3: 12.0, 4: 25.0, 5: 50.0, 6: 100.0}
DEFAULT_PINNED_GBPS = 12.0  # neither measured nor a known link
# The host-memory modes, and the engine feature each needs (FR-T1-2).
T1_MODE_FEATURES = {"register": "cpu_offload_shm", "alloc": "cpu_offload_host_alloc"}
# An engine that copies the registered /dev/shm region one cudaMemcpyAsync at a time, which works
# where the GPU maps registered memory at a different address (TCS vGPU, spike S2; vLLM 0.30's
# batched copy and Triton load kernel fault there). It lets such a node run T1 in register mode,
# which vLLM's tiering needs: the disk tier's manager reads and writes T1 through that shared
# region, so alloc mode (private per rank) can't carry a disk tier. Verified with a stand-in patch
# on TCS, 2026-10-02.
T1_PER_COPY_FEATURE = "cpu_offload_shm_per_copy"

# -------------------------------------------------------------------------------------------- T2
T2_FEATURE = "tiering_fs"
# Fixed cost of a T2 load. [MEASURE]: the storage-class measurement (IMPLEMENTATION_PLAN 3.2).
T2_FLOOR_MS = 15.0
# A deployment's T2 directory takes at most this share of the node's local disk. [MEASURE]
T2_DISK_SHARE = 0.25
# Auto picks no class below this read rate (FR-SET-3).
T2_MIN_READ_GBPS = 2.0
# T2 is dropped when it would serve no more than this share of requests beyond the GPU pool and
# T1: it would only hold copies of their blocks and still write every one to disk.
T2_MIN_SERVED_SHARE = 0.005
# vLLM's fs tier I/O threads (its defaults are 16 and 16). The TCS and accubits01 rounds ran 8
# readers and 4 writers; one reader stream already matches the tier's effective rate (Bud's
# spikes/accuracy "Disk tier").
T2_READ_THREADS = 8
T2_WRITE_THREADS = 4

# -------------------------------------------------------------------------------------------- T3
T3_FEATURE = "mooncake_store"
# Fixed cost of one pool load, when the pool's health gate didn't measure it. TCP: fitted on TCS
# (spike S10): load time = 0.27-0.42 s + bytes / rate for Qwen3-8B and Qwen3.8-27B. RDMA:
# [MEASURE], no RDMA hardware yet.
T3_FLOOR_MS = {"tcp": 300.0, "rdma": 15.0}
# Share of the pool's measured transfer rate that vLLM's Mooncake connector reaches, by per-layer
# segment size (bytes). TCS (S10): the transfer engine moved 1.3-1.6 GB/s; through the connector the
# 8B's 64 KiB segments loaded at ~0.73 GB/s and the 27B's ~3 MiB segments at ~1.45 GB/s. Between
# points the share is interpolated on log(segment size). TCP only; RDMA [MEASURE] at 1.0.
T3_CONNECTOR_EFFICIENCY: Tuple[Tuple[int, float], ...] = ((64 * KIB, 0.5), (3 * MIB, 1.0))
T3_RDMA_EFFICIENCY = 1.0
# One deployment plans on at most this share of the pool's capacity. [MEASURE]
T3_POOL_SHARE_MAX = 0.25
# FR-INFRA-4: an RDMA pool qualifies from this rate per GPU.
RDMA_MIN_GBPS_PER_GPU = 10.0

# -------------------------------------------------------------------------------- load model
# Decode time per output token, used only to estimate a replica's request rate when the caller
# doesn't pass its predicted end-to-end latency. TCS measured 8.4 ms (Qwen3-8B) and 22 ms
# (Qwen3.8-27B) on an H100. [MEASURE]
DECODE_MS_PER_TOKEN = 20.0

# ---------------------------------------------------------------------------------- workload
# profile -> (sessions alive per concurrency slot within the reuse window, reusable share of each
# prompt). [MEASURE]: no real-user reuse data yet; IMPLEMENTATION_PLAN 1.1r names the traces.
PROFILE_REUSE: Dict[str, Tuple[float, float]] = {
    "agentic": (4.0, 0.95),  # tool definitions and the whole history come back every turn
    "chat": (6.0, 0.8),  # multi-turn history
    "interactive": (6.0, 0.8),  # chat with a tight TTFT target
    "rag": (3.0, 0.3),  # a shared system prompt; retrieved documents vary
}
# How many times a reusable prefix is used over its life, by profile: its first use always misses,
# so no cache serves more than (1 - 1/uses) of the reusable tokens. A shared-prefix bench with 8
# prompts per prefix measured ~15 points below a model without this term. [MEASURE: per-profile
# traces]
PROFILE_USES: Dict[str, float] = {
    "agentic": 20.0,  # many turns of one session, each resending its history
    "chat": 6.0,  # the turns of a conversation
    "interactive": 6.0,
    "rag": 50.0,  # one system prompt shared by many requests
}
PREFIX_HEAVY_RATIO = 8.0  # input:output at or above this is prefix-heavy (ARCHITECTURE §10.2)
INTERACTIVE_TTFT_MS = 500.0

# ---------------------------------------------------------------------------------- recompute
# Share of peak dense FLOPs a prefill reaches, for the formula used only when GenZ can't estimate
# the model on the hardware. On TCS's H100 GenZ was within 4-17% of measured prefill; this formula
# at 0.45 was 55% high for Qwen3-8B at 4K.
PREFILL_MFU = 0.45

# ---------------------------------------------------------------------------------- geometry
VLLM_BLOCK_SIZE = 16  # vLLM's default KV block on CUDA
VLLM_KERNEL_BLOCK_ALIGNMENT = 16  # attention backends take block sizes in multiples of 16

# ------------------------------------------------------------------------------------- dtype
# fp8 KV only where validated (FR-QUANT-2), as (model_type, GPU generation). budeval's verdicts
# arrive through EngineSpec.validated_kv_dtypes; this is the package's own list (empty).
FP8_KV_VALIDATED: FrozenSet[Tuple[str, str]] = frozenset()

# ------------------------------------------------------------------------------------- P/D
# FR-PD-1 gates.
PD_FEATURE = "nixl"
PD_MIN_PARAMS = 70e9
PD_MIN_INPUT_TOKENS = 4096
PD_MIN_REQUEST_RATE = 2.0

EVENTS_ENDPOINT = "tcp://*:5557"
EVENTS_REPLAY = "tcp://*:5558"
