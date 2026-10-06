"""Measured runs the planner is held to (FRD-023 NFR-9; spike S10, spike S2, accubits01 Phase 1).

Each :class:`Fixture` is one measured scenario: the inputs the planner would get, and what the run
showed. Two kinds:

* **load** fixtures come from a shared-prefix benchmark run with and without the tier. The oracle is
  the configuration with the higher measured throughput (goodput), and regret is measured against
  it.
* **single** fixtures come from one request after its prefix was evicted from the GPU. They check
  the parts of the break-even: the reload time (tier TTFT minus GPU-hit TTFT), the recompute time
  (TTFT with the prefix cache missed, also minus GPU-hit TTFT: the overhead every request pays), and
  GenZ's estimate of it. Their oracle applies the planner's own margin to the measured times: keep
  only if reload < 0.8 x recompute.

Recompute for a fixture comes from GenZ estimates recorded for that model on that GPU (interpolated
on log-log between the recorded lengths) or, where GenZ has no entry for the GPU (the RTX 3050),
from the run's own measured prefill times.

Sources: ``specs/023-kv-cache-tiering/spikes/lmcache/README.md`` (TCS and accubits01 tables),
``spikes/s2`` (the 32K T1 hit), IMPLEMENTATION_PLAN "Bare-metal check: accubits01".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .constants import T1_PER_COPY_FEATURE
from .facts import EngineSpec
from .geometry import KVGeometry
from .group import NodeGroup
from .infra import InfraSnapshot
from .workload import DeploymentSpec, KVWorkload

GIB = 1 << 30
# Throughput differences under this share are within run-to-run noise (FRD-023 "Planner accuracy").
TIE_FLOOR = 0.03

QWEN3_8B: Dict[str, Any] = {
    "model_type": "qwen3",
    "hidden_size": 4096,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 12288,
    "vocab_size": 151936,
    "max_position_embeddings": 40960,
}
QWEN3_06B: Dict[str, Any] = {
    "model_type": "qwen3",
    "hidden_size": 1024,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 3072,
    "vocab_size": 151936,
    "tie_word_embeddings": True,
    "max_position_embeddings": 40960,
}
# Qwen3.8-27B as served on TCS: 16 full-attention layers of 64, the rest Gated DeltaNet.
QWEN3_8_27B: Dict[str, Any] = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "tie_word_embeddings": False,
    "text_config": {
        "model_type": "qwen3_5_text",
        "attn_output_gate": True,
        "dtype": "bfloat16",
        "full_attention_interval": 4,
        "head_dim": 256,
        "hidden_act": "silu",
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "layer_types": (
            ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 16
        ),
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_value_head_dim": 128,
        "mamba_ssm_dtype": "float32",
        "max_position_embeddings": 262144,
        "num_attention_heads": 24,
        "num_hidden_layers": 64,
        "num_key_value_heads": 4,
        "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1e-06,
        "tie_word_embeddings": False,
        "vocab_size": 248320,
    },
    "vision_config": {
        "depth": 27,
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "model_type": "qwen3_5",
        "num_heads": 16,
        "out_hidden_size": 5120,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
}

# Qwen/Qwen3-14B (config.json, fields the planner reads)
QWEN3_14B: Dict[str, Any] = {
    "architectures": ["Qwen3ForCausalLM"],
    "head_dim": 128,
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "max_position_embeddings": 40960,
    "max_window_layers": 40,
    "model_type": "qwen3",
    "num_attention_heads": 40,
    "num_hidden_layers": 40,
    "num_key_value_heads": 8,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "use_sliding_window": False,
    "vocab_size": 151936,
}
# Qwen/Qwen3-32B (config.json, fields the planner reads)
QWEN3_32B: Dict[str, Any] = {
    "architectures": ["Qwen3ForCausalLM"],
    "head_dim": 128,
    "hidden_size": 5120,
    "intermediate_size": 25600,
    "max_position_embeddings": 40960,
    "max_window_layers": 64,
    "model_type": "qwen3",
    "num_attention_heads": 64,
    "num_hidden_layers": 64,
    "num_key_value_heads": 8,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "use_sliding_window": False,
    "vocab_size": 151936,
}
# Qwen/Qwen2.5-1.5B-Instruct (config.json, fields the planner reads)
QWEN2_5_1_5B: Dict[str, Any] = {
    "architectures": ["Qwen2ForCausalLM"],
    "hidden_size": 1536,
    "intermediate_size": 8960,
    "max_position_embeddings": 32768,
    "max_window_layers": 21,
    "model_type": "qwen2",
    "num_attention_heads": 12,
    "num_hidden_layers": 28,
    "num_key_value_heads": 2,
    "sliding_window": 32768,
    "tie_word_embeddings": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
}
# Qwen/Qwen2.5-7B-Instruct (config.json, fields the planner reads)
QWEN2_5_7B: Dict[str, Any] = {
    "architectures": ["Qwen2ForCausalLM"],
    "hidden_size": 3584,
    "intermediate_size": 18944,
    "max_position_embeddings": 32768,
    "max_window_layers": 28,
    "model_type": "qwen2",
    "num_attention_heads": 28,
    "num_hidden_layers": 28,
    "num_key_value_heads": 4,
    "sliding_window": 131072,
    "tie_word_embeddings": False,
    "use_sliding_window": False,
    "vocab_size": 152064,
}
# microsoft/Phi-4-mini-instruct (config.json, fields the planner reads)
PHI4_MINI: Dict[str, Any] = {
    "_name_or_path": "Phi-4-mini-instruct",
    "architectures": ["Phi3ForCausalLM"],
    "hidden_size": 3072,
    "intermediate_size": 8192,
    "max_position_embeddings": 131072,
    "model_type": "phi3",
    "num_attention_heads": 24,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "partial_rotary_factor": 0.75,
    "sliding_window": 262144,
    "tie_word_embeddings": True,
    "vocab_size": 200064,
}

# Qwen/Qwen3-30B-A3B (config.json, fields the planner reads)
QWEN3_30B_A3B: Dict[str, Any] = {
    "architectures": ["Qwen3MoeForCausalLM"],
    "decoder_sparse_step": 1,
    "head_dim": 128,
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "max_position_embeddings": 40960,
    "max_window_layers": 48,
    "mlp_only_layers": [],
    "model_type": "qwen3_moe",
    "moe_intermediate_size": 768,
    "num_attention_heads": 32,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 48,
    "num_key_value_heads": 4,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "use_sliding_window": False,
    "vocab_size": 151936,
}
# Qwen/Qwen2.5-72B-Instruct-AWQ (config.json, fields the planner reads)
QWEN2_5_72B_AWQ: Dict[str, Any] = {
    "architectures": ["Qwen2ForCausalLM"],
    "hidden_size": 8192,
    "intermediate_size": 29696,
    "max_position_embeddings": 32768,
    "max_window_layers": 70,
    "model_type": "qwen2",
    "num_attention_heads": 64,
    "num_hidden_layers": 80,
    "num_key_value_heads": 8,
    "quantization_config": {
        "bits": 4,
        "group_size": 128,
        "modules_to_not_convert": None,
        "quant_method": "awq",
        "version": "gemm",
        "zero_point": True,
    },
    "sliding_window": 131072,
    "tie_word_embeddings": False,
    "use_sliding_window": False,
    "vocab_size": 152064,
}

# Qwen/Qwen3-1.7B (config.json, fields the planner reads)
QWEN3_17B: Dict[str, Any] = {
    "architectures": ["Qwen3ForCausalLM"],
    "head_dim": 128,
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "max_position_embeddings": 40960,
    "model_type": "qwen3",
    "num_attention_heads": 16,
    "num_hidden_layers": 28,
    "num_key_value_heads": 8,
    "tie_word_embeddings": True,
    "vocab_size": 151936,
}

# allenai/OLMoE-1B-7B-0125-Instruct (config.json, fields the planner reads)
OLMOE_1B_7B: Dict[str, Any] = {
    "_name_or_path": "allenai/open_instruct_dev",
    "architectures": ["OlmoeForCausalLM"],
    "hidden_size": 2048,
    "intermediate_size": 1024,
    "max_position_embeddings": 4096,
    "model_type": "olmoe",
    "num_attention_heads": 16,
    "num_experts": 64,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 16,
    "num_key_value_heads": 16,
    "tie_word_embeddings": False,
    "vocab_size": 50304,
}
# TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ (config.json, fields the planner reads; sliding_window as
# mistralai's own config: Mixtral has no SWA)
MIXTRAL_8X7B_AWQ: Dict[str, Any] = {
    "architectures": ["MixtralForCausalLM"],
    "hidden_act": "silu",
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "max_position_embeddings": 32768,
    "model_type": "mixtral",
    "num_attention_heads": 32,
    "num_experts_per_tok": 2,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "num_local_experts": 8,
    "quantization_config": {
        "bits": 4,
        "group_size": 128,
        "modules_to_not_convert": ["gate"],
        "quant_method": "awq",
        "version": "gemm",
        "zero_point": True,
    },
    # the parameter counter reads it to classify the architecture
    "router_aux_loss_coef": 0.02,
    "rope_theta": 1000000.0,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "vocab_size": 32000,
}

# deepseek-ai/DeepSeek-V2-Lite-Chat (config.json, fields the planner reads)
DEEPSEEK_V2_LITE: Dict[str, Any] = {
    # the parameter counter reads these to classify the architecture
    "aux_loss_alpha": 0.001,
    "seq_aux": True,
    "architectures": ["DeepseekV2ForCausalLM"],
    "first_k_dense_replace": 1,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "intermediate_size": 10944,
    "kv_lora_rank": 512,
    "max_position_embeddings": 163840,
    "model_type": "deepseek_v2",
    "moe_intermediate_size": 1408,
    "moe_layer_freq": 1,
    "n_routed_experts": 64,
    "n_shared_experts": 2,
    "num_attention_heads": 16,
    "num_experts_per_tok": 6,
    "num_hidden_layers": 27,
    "num_key_value_heads": 16,
    "q_lora_rank": None,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "rope_theta": 10000,
    "tie_word_embeddings": False,
    "v_head_dim": 128,
    "vocab_size": 102400,
}

# ibm-granite/granite-3.1-3b-a800m-instruct (config.json, fields the planner reads)
GRANITE_31_3B_A800M: Dict[str, Any] = {
    "architectures": ["GraniteMoeForCausalLM"],
    "hidden_act": "silu",
    "hidden_size": 1536,
    "intermediate_size": 512,
    "max_position_embeddings": 131072,
    "model_type": "granitemoe",
    "num_attention_heads": 24,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "num_local_experts": 40,
    "rope_theta": 10000000.0,
    "tie_word_embeddings": True,
    "vocab_size": 49155,
}
# Qwen/Qwen1.5-MoE-A2.7B-Chat (config.json, fields the planner reads)
QWEN1_5_MOE_A2_7B: Dict[str, Any] = {
    "architectures": ["Qwen2MoeForCausalLM"],
    "decoder_sparse_step": 1,
    "hidden_size": 2048,
    "intermediate_size": 5632,
    "max_position_embeddings": 32768,
    "max_window_layers": 21,
    "model_type": "qwen2_moe",
    "moe_intermediate_size": 1408,
    "num_attention_heads": 16,
    "num_experts": 60,
    "num_experts_per_tok": 4,
    "num_hidden_layers": 24,
    "num_key_value_heads": 16,
    "shared_expert_intermediate_size": 5632,
    "sliding_window": 32768,
    "tie_word_embeddings": False,
    "use_sliding_window": False,
    "vocab_size": 151936,
}
# Qwen/Qwen3-4B (config.json, fields the planner reads)
QWEN3_4B: Dict[str, Any] = {
    "architectures": ["Qwen3ForCausalLM"],
    "head_dim": 128,
    "hidden_size": 2560,
    "intermediate_size": 9728,
    "max_position_embeddings": 40960,
    "max_window_layers": 36,
    "model_type": "qwen3",
    "num_attention_heads": 32,
    "num_hidden_layers": 36,
    "num_key_value_heads": 8,
    "sliding_window": None,
    "tie_word_embeddings": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
}

# GenZ prefill estimates recorded on TCS's H100 (Flops 989.5, 3,350 GB/s), ms by prompt tokens.
GENZ_H100_MS: Dict[str, Sequence[Tuple[int, float]]] = {
    "qwen3-8b": (
        (512, 13.6),
        (1024, 25.3),
        (2400, 57.4),
        (3585, 86.3),
        (4097, 99.2),
        (8192, 209.6),
        (16129, 461.5),
        (16384, 470.4),
        (32769, 1152.1),
    ),
    "qwen3-0.6b": (
        (512, 3.0),
        (1024, 4.2),
        (4096, 12.7),
        (8192, 28.8),
        (16128, 74.7),
        (16384, 76.5),
    ),
    "qwen3-14b": (
        (512, 22.9),
        (1024, 43.7),
        (4096, 174.5),
        (8192, 365.3),
        (16128, 787.7),
        (16384, 802.4),
    ),
    "qwen3-32b": (
        (512, 48.8),
        (1024, 95.3),
        (4096, 388.3),
        (8192, 821.0),
        (16128, 1794.4),
        (16384, 1828.8),
    ),
    "qwen2.5-1.5b-instruct": (
        (512, 4.4),
        (1024, 6.9),
        (4096, 23.4),
        (8192, 48.9),
        (16128, 109.2),
        (16384, 111.4),
    ),
    "qwen2.5-7b-instruct": (
        (512, 12.5),
        (1024, 23.2),
        (4096, 90.5),
        (8192, 188.1),
        (16128, 403.2),
        (16384, 410.7),
    ),
    "phi-4-mini-instruct": (
        (512, 8.3),
        (1024, 14.6),
        (4096, 55.5),
        (8192, 118.0),
        (16128, 264.3),
        (16384, 269.6),
    ),
    "qwen3.8-27b": ((4097, 289.6), (16129, 1337.2)),
}
# Round 3 (2026-10-02): longer prompts and the MoE / AWQ models, GenZ with the MoE expert width
# fixed.
_GENZ_ROUND3: Dict[str, Sequence[Tuple[int, float]]] = {
    "qwen3-14b": ((32768, 1899.0),),
    "phi-4-mini-instruct": ((65536, 1925.9), (122880, 5476.0)),
    "qwen3-30b-a3b": ((1024, 28.5), (4096, 51.5), (16384, 293.6)),
    "qwen2.5-72b-instruct-awq": ((1024, 207.2), (4096, 838.4), (16384, 3698.8)),
    "qwen3-8b": ((32768, 1152.1),),
}
_GENZ_ROUND4: Dict[str, Sequence[Tuple[int, float]]] = {
    "olmoe-1b-7b-0125-instruct": ((1024, 7.8), (2048, 9.8), (3584, 15.2)),
    "qwen1.5-moe-a2.7b-chat": ((1024, 13.9), (4096, 25.4), (16384, 119.9)),
    "qwen3-30b-a3b": ((2048, 35.0),),
    "qwen3-4b": ((1024, 14.8), (2048, 28.2)),
    "qwen3-1.7b": ((2048, 13.8), (4096, 26.9)),
}
# The planner's recompute at prediction time (GenZ; DeepSeek-V2-Lite's MLA falls back to the FLOPs
# formula).
_GENZ_ROUND5: Dict[str, Sequence[Tuple[int, float]]] = {
    "granite-3.1-3b-a800m-instruct": ((1024, 5.8), (4096, 16.5), (16384, 99.8)),
    "mixtral-8x7b-instruct-v0.1-awq": ((1024, 38.3), (4096, 150.6), (16384, 667.1)),
    "deepseek-v2-lite-chat": ((1024, 12.6), (4096, 53.5), (16384, 263.9)),
}
_GENZ_ROUND3 = {
    k: tuple(
        dict([*_GENZ_ROUND3.get(k, ()), *_GENZ_ROUND4.get(k, ()), *_GENZ_ROUND5.get(k, ())]).items()
    )
    for k in {*_GENZ_ROUND3, *_GENZ_ROUND4, *_GENZ_ROUND5}
}
for _key, _points in _GENZ_ROUND3.items():
    GENZ_H100_MS[_key] = tuple(sorted(dict([*GENZ_H100_MS.get(_key, ()), *_points]).items()))
# accubits01's RTX 3050 has no GenZ entry: its measured cold prefill times stand in.
MEASURED_RTX3050_MS: Dict[str, Sequence[Tuple[int, float]]] = {
    "qwen3-0.6b": ((3585, 414.0 - 28.5), (4097, 517.5 - 23.5)),
}


def per_token(points: Sequence[Tuple[int, float]]) -> Callable[[int], Optional[float]]:
    """Milliseconds for ``n`` tokens at the points' mean per-token rate: for a small model whose
    prefill is linear at these lengths, where two nearby points can't fix a curve's slope."""
    rate = sum(ms / tokens for tokens, ms in points) / len(points)

    def ms(tokens: int) -> Optional[float]:
        return max(tokens, 0) * rate

    return ms


def interpolated(points: Sequence[Tuple[int, float]]) -> Callable[[int], Optional[float]]:
    """Milliseconds for ``n`` tokens, interpolated (and extrapolated) on log-log between points."""
    pts = sorted(points)

    def ms(tokens: int) -> Optional[float]:
        if tokens <= 0:
            return 0.0
        if len(pts) == 1:
            return pts[0][1] * tokens / pts[0][0]
        lo, hi = pts[0], pts[1]
        for a, b in zip(pts, pts[1:]):
            lo, hi = a, b
            if tokens <= b[0]:
                break
        slope = (math.log(hi[1]) - math.log(lo[1])) / (math.log(hi[0]) - math.log(lo[0]))
        return math.exp(math.log(lo[1]) + slope * (math.log(tokens) - math.log(lo[0])))

    return ms


@dataclass(frozen=True)
class Fixture:
    """One measured scenario and the planner inputs that describe it."""

    name: str
    kind: str  # load | single
    tier: str  # the tier the run measured
    source: str
    deployment: DeploymentSpec
    engine: EngineSpec
    group: NodeGroup
    infra: InfraSnapshot
    workload: KVWorkload
    recompute: Callable[[int], Optional[float]]
    # load fixtures: measured throughput (req/s) without and with the tier
    goodput_without: Optional[float] = None
    goodput_with: Optional[float] = None
    # single fixtures: measured times, ms
    reload_ms: Optional[float] = None  # tier TTFT minus GPU-hit TTFT
    recompute_ms: Optional[float] = None  # TTFT with the prefix missed
    genz_ms: Optional[float] = None  # GenZ's estimate for the same prompt, when it has one
    # The pool rate and floor came from a health-gate-style measurement (False: inferred or
    # defaulted, so the fixture is excluded from the reload error).
    gate_measured: bool = True
    # Expected bytes a tier load moves, when the run logged it
    measured_bytes: Optional[int] = None
    # load fixtures: the run's repeat spread (relative). Configurations closer than this, or than
    # TIE_FLOOR, tie: either decision is right.
    noise: float = 0.0
    notes: str = ""

    @property
    def oracle_keep(self) -> bool:
        if self.kind == "load":
            return (self.goodput_with or 0.0) > (self.goodput_without or 0.0)
        return (self.reload_ms or math.inf) < 0.8 * (self.recompute_ms or 0.0)

    @property
    def tie(self) -> bool:
        if self.kind != "load" or not self.goodput_without:
            return False
        delta = abs((self.goodput_with or 0.0) - self.goodput_without) / self.goodput_without
        return delta <= max(TIE_FLOOR, self.noise)


# ------------------------------------------------------------------------------------------ helpers


def _pool_gib_device(pool_gib: float, weight_gib: float, other_gib: float = 0.0) -> float:
    """Card memory that makes T0's dedicated pool exactly ``pool_gib`` (the runs fixed the GPU KV
    pool with --kv-cache-memory-bytes)."""
    return (pool_gib + weight_gib + other_gib + 1.0) / 0.92


def _gb(gib: float) -> float:
    """GiB -> the decimal GB budsim puts on the wire."""
    return gib * GIB / 1e9


def _bench(
    *,
    prefix: int,
    suffix: int,
    prefixes: int,
    concurrency: int,
    output: int = 32,
    uses: float = 8.0,
) -> KVWorkload:
    """A shared-prefix benchmark (``vllm bench serve --dataset-name prefix_repetition``) as a
    workload: every prompt reuses one of ``prefixes`` prefixes, each sent ``uses`` times."""
    total = prefix + suffix
    return KVWorkload(
        profile="chat",
        input_tokens=total,
        output_tokens=output,
        concurrency=concurrency,
        sessions_per_slot=prefixes / concurrency,
        reusable_share=prefix / total,
        reason="measured shared-prefix benchmark",
        uses_per_prefix=uses,
    )


def _single(prompt: int, *, sessions: float) -> KVWorkload:
    """One request whose whole prompt is the reusable prefix, among enough others (``sessions``
    prompts) that the GPU pool can't keep it: the runs flooded it out of GPU memory first."""
    return KVWorkload(
        profile="chat",
        input_tokens=prompt,
        output_tokens=1,
        concurrency=1,
        sessions_per_slot=sessions,
        reusable_share=(prompt - 1) / prompt,
        reason="measured single request after eviction",
    )


TCS_NODE = {
    "pinned_offload_ok": True,
    "vgpu": True,
    "host_pointer_for_registered_mem": False,
    "pinned_copy_gbps": 55.2,
    "rdma_nics": 0,
}
TCS_POOL = {
    "backend": "mooncake",
    "transport": "tcp",
    # Mooncake's transfer engine between the two TCS nodes, RAM to RAM (1.26-1.60 GB/s measured)
    "read_gbps": 1.45,
    "capacity_gib": 32.0,
}
# upstream vllm/vllm-openai:v0.30.0 as run on TCS: no alloc-mode T1, Mooncake in the image
UPSTREAM_030 = ["kv_events", "cache_salt", "cpu_offload_shm", "tiering_fs", "mooncake_store"]
BUD_010 = UPSTREAM_030 + ["cpu_offload_host_alloc"]


def _tcs(
    *,
    model: str,
    config: Mapping[str, Any],
    weight_gib: float,
    pool_gib: float,
    features: Sequence[str],
    e2e_s: Optional[float],
    pool: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    cluster = {"kv_capability": {"t3": dict(pool)}} if pool else {"kv_capability": {}}
    node = {
        "name": "cscps3ee8ubu02",
        "kv_capabilities": TCS_NODE,
        "devices": [{"type": "cpu", "available_memory_gb": 220.0}],
    }
    return dict(
        deployment=DeploymentSpec(
            model_config=config,
            input_tokens=0,
            output_tokens=0,
            concurrency=1,
            model_id=model,
            e2e_latency_s=e2e_s,
            cross_node_need=pool is not None,
        ),
        engine=EngineSpec.of("vllm", features),
        group=NodeGroup(
            device_type="cuda",
            weight_memory_gb=_gb(weight_gib),
            kv_cache_memory_gb=_gb(1.0),
            device_total_memory_gb=_pool_gib_device(pool_gib, weight_gib),
            device_model_key="H100XM-80C",
            device_generation="hopper",
            device_tflops=989.5,
            hardware={"Flops": 989.5, "Memory_size": 80, "Memory_BW": 3350, "ICN": 450},
        ),
        infra=InfraSnapshot(node=node, cluster=cluster),
    )


def _accubits(
    *, features: Sequence[str], e2e_s: Optional[float], pool: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    cluster = {"kv_capability": {"t3": dict(pool)}} if pool else {"kv_capability": {}}
    node = {
        "name": "accubits",
        "kv_capabilities": {
            "pinned_offload_ok": True,
            "vgpu": False,
            "host_pointer_for_registered_mem": True,
            "pinned_copy_gbps": 4.7,
            "pcie_gen": 4,
        },
        "devices": [{"type": "cpu", "available_memory_gb": 20.0}],
    }
    return dict(
        deployment=DeploymentSpec(
            model_config=QWEN3_06B,
            input_tokens=0,
            output_tokens=0,
            concurrency=1,
            e2e_latency_s=e2e_s,
            cross_node_need=pool is not None,
        ),
        engine=EngineSpec.of("vllm", features),
        group=NodeGroup(
            device_type="cuda",
            weight_memory_gb=_gb(1.2),
            kv_cache_memory_gb=_gb(1.0),
            device_total_memory_gb=_pool_gib_device(1.3, 1.2),
            device_model_key="RTX 3050 Laptop",
            device_tflops=None,
        ),
        infra=InfraSnapshot(node=node, cluster=cluster),
    )


# -------------------------------------------------- TCS T1 accuracy grid and hold-out (2026-10-01)
# One dedicated H100 vGPU per engine on cscps3ee8ubu02, the GPU KV pool fixed
# (--kv-cache-memory-bytes), a shared-prefix benchmark (prefix + 256 tokens, 32 out, ~2.5x the pool
# in prefixes, 8 prompts per prefix, two rounds) run side by side with no connector and with T1 in
# alloc mode (spike S2's stand-in), T1 sized by the planner (3x the pool). The grid was predicted
# blind; its two misses (Qwen3-0.6B at 16K) moved the break-even to the idle reload plus a link
# limit. The hold-out (new models and lengths) was then predicted blind under that rule. Harness:
# spikes/lmcache (run_grid).
T1_GRID_SIZES: Dict[str, Tuple[Dict[str, Any], float, float]] = {  # config, weights GiB, pool GiB
    "Qwen/Qwen3-0.6B": (QWEN3_06B, 1.4, 6.0),
    "Qwen/Qwen3-8B": (QWEN3_8B, 15.27, 8.0),
    "Qwen/Qwen3-14B": (QWEN3_14B, 27.5, 9.0),
    "Qwen/Qwen3-32B": (QWEN3_32B, 61.0, 9.0),
    "Qwen/Qwen2.5-1.5B-Instruct": (QWEN2_5_1_5B, 2.9, 2.0),
    "Qwen/Qwen2.5-7B-Instruct": (QWEN2_5_7B, 14.2, 4.0),
    "microsoft/Phi-4-mini-instruct": (PHI4_MINI, 7.2, 7.0),
}
# (model, prefix, concurrency, prefixes, no-connector req/s, T1 req/s, repeat spread)
T1_GRID: Sequence[Tuple[str, int, int, int, float, float, float]] = (
    ("Qwen/Qwen3-0.6B", 1024, 4, 137, 36.54, 32.30, 0.013),
    ("Qwen/Qwen3-0.6B", 1024, 16, 137, 93.52, 73.91, 0.055),
    ("Qwen/Qwen3-0.6B", 4096, 4, 34, 29.02, 25.84, 0.022),
    ("Qwen/Qwen3-0.6B", 4096, 16, 34, 53.07, 48.16, 0.011),
    ("Qwen/Qwen3-0.6B", 16384, 4, 9, 9.71, 12.48, 0.057),
    ("Qwen/Qwen3-0.6B", 16384, 16, 9, 10.50, 14.35, 0.052),
    ("Qwen/Qwen3-14B", 1024, 4, 144, 7.43, 8.31, 0.012),
    ("Qwen/Qwen3-14B", 1024, 16, 144, 15.39, 21.80, 0.010),
    ("Qwen/Qwen3-14B", 4096, 4, 36, 4.46, 6.91, 0.029),
    ("Qwen/Qwen3-14B", 4096, 16, 36, 6.36, 14.59, 0.011),
    ("Qwen/Qwen3-14B", 16384, 4, 9, 1.44, 3.53, 0.111),
    ("Qwen/Qwen3-14B", 16384, 16, 9, 1.40, 3.63, 0.036),
    ("Qwen/Qwen3-32B", 1024, 4, 90, 3.46, 3.96, 0.009),
    ("Qwen/Qwen3-32B", 1024, 16, 90, 7.09, 9.96, 0.023),
    ("Qwen/Qwen3-32B", 4096, 4, 22, 1.98, 3.41, 0.032),
    ("Qwen/Qwen3-32B", 4096, 16, 22, 2.61, 5.80, 0.038),
    ("Qwen/Qwen3-32B", 16384, 4, 6, 0.59, 1.66, 0.079),
    ("Qwen/Qwen3-32B", 16384, 16, 6, 0.57, 1.67, 0.157),
    ("Qwen/Qwen3-8B", 1024, 4, 142, 13.75, 14.47, 0.011),
    ("Qwen/Qwen3-8B", 1024, 16, 142, 28.06, 36.03, 0.028),
    ("Qwen/Qwen3-8B", 4096, 4, 36, 7.82, 11.25, 0.031),
    ("Qwen/Qwen3-8B", 4096, 16, 36, 11.31, 22.50, 0.028),
    ("Qwen/Qwen3-8B", 16384, 4, 9, 2.39, 5.21, 0.084),
    ("Qwen/Qwen3-8B", 16384, 16, 9, 2.37, 5.52, 0.047),
)
T1_HOLDOUT: Sequence[Tuple[str, int, int, int, float, float, float]] = (
    ("Qwen/Qwen2.5-1.5B-Instruct", 1024, 4, 183, 34.11, 31.42, 0.014),
    ("Qwen/Qwen2.5-1.5B-Instruct", 1024, 16, 183, 85.57, 76.24, 0.012),
    ("Qwen/Qwen2.5-1.5B-Instruct", 4096, 4, 46, 23.85, 25.39, 0.017),
    ("Qwen/Qwen2.5-1.5B-Instruct", 4096, 16, 46, 41.99, 51.11, 0.035),
    ("Qwen/Qwen2.5-1.5B-Instruct", 16384, 4, 11, 8.43, 14.53, 0.070),
    ("Qwen/Qwen2.5-1.5B-Instruct", 16384, 16, 11, 9.65, 18.65, 0.026),
    ("Qwen/Qwen2.5-7B-Instruct", 1024, 4, 183, 14.05, 15.32, 0.002),
    ("Qwen/Qwen2.5-7B-Instruct", 1024, 16, 183, 29.67, 40.66, 0.012),
    ("Qwen/Qwen2.5-7B-Instruct", 4096, 4, 46, 8.50, 12.77, 0.027),
    ("Qwen/Qwen2.5-7B-Instruct", 4096, 16, 46, 12.24, 27.47, 0.051),
    ("Qwen/Qwen2.5-7B-Instruct", 16384, 4, 11, 2.82, 6.69, 0.092),
    ("Qwen/Qwen2.5-7B-Instruct", 16384, 16, 11, 2.98, 7.67, 0.030),
    ("Qwen/Qwen3-0.6B", 8192, 4, 17, 18.41, 18.56, 0.017),
    ("Qwen/Qwen3-0.6B", 8192, 16, 17, 24.80, 26.11, 0.099),
    ("Qwen/Qwen3-8B", 512, 4, 284, 15.79, 15.64, 0.019),
    ("Qwen/Qwen3-8B", 512, 16, 284, 38.14, 40.84, 0.007),
    ("microsoft/Phi-4-mini-instruct", 1024, 4, 140, 22.17, 21.99, 0.008),
    ("microsoft/Phi-4-mini-instruct", 1024, 16, 140, 49.25, 56.15, 0.012),
    ("microsoft/Phi-4-mini-instruct", 4096, 4, 35, 13.45, 16.91, 0.023),
    ("microsoft/Phi-4-mini-instruct", 4096, 16, 35, 20.34, 33.84, 0.018),
    ("microsoft/Phi-4-mini-instruct", 16384, 4, 9, 4.09, 7.37, 0.068),
    ("microsoft/Phi-4-mini-instruct", 16384, 16, 9, 4.22, 8.17, 0.072),
)
# One request after a flood evicted it from the GPU, T1 engine: (model, prompt tokens, cold TTFT,
# T1-hit TTFT, GPU-hit TTFT), medians, ms. Reload and recompute are both net of the GPU-hit TTFT.
T1_SINGLES: Sequence[Tuple[str, int, float, float, float]] = (
    ("Qwen/Qwen3-0.6B", 1025, 16.7, 20.0, 10.0),
    ("Qwen/Qwen3-0.6B", 4097, 23.6, 33.6, 12.9),
    ("Qwen/Qwen3-0.6B", 16129, 104.9, 75.2, 23.6),
    ("Qwen/Qwen3-14B", 1025, 52.3, 27.5, 16.6),
    ("Qwen/Qwen3-14B", 4097, 183.2, 43.1, 19.7),
    ("Qwen/Qwen3-14B", 16129, 832.0, 101.0, 32.0),
    ("Qwen/Qwen3-32B", 1025, 128.3, 44.1, 30.3),
    ("Qwen/Qwen3-32B", 4097, 413.0, 68.5, 33.5),
    ("Qwen/Qwen3-32B", 16129, 2047.6, 146.6, 44.9),
    ("Qwen/Qwen3-8B", 1025, 29.3, 24.2, 12.1),
    ("Qwen/Qwen3-8B", 4097, 104.8, 42.8, 15.1),
    ("Qwen/Qwen3-8B", 16129, 491.0, 113.3, 26.1),
    ("Qwen/Qwen2.5-1.5B-Instruct", 1025, 18.1, 18.4, 10.5),
    ("Qwen/Qwen2.5-1.5B-Instruct", 4097, 31.4, 25.1, 13.1),
    ("Qwen/Qwen2.5-1.5B-Instruct", 16385, 138.5, 50.6, 24.1),
    ("Qwen/Qwen2.5-7B-Instruct", 1025, 27.5, 21.4, 11.7),
    ("Qwen/Qwen2.5-7B-Instruct", 4097, 94.4, 32.4, 14.4),
    ("Qwen/Qwen2.5-7B-Instruct", 16385, 448.3, 77.9, 25.4),
    ("Qwen/Qwen3-0.6B", 8193, 64.6, 65.0, 17.0),
    ("Qwen/Qwen3-8B", 513, 22.3, 18.8, 11.5),
    ("microsoft/Phi-4-mini-instruct", 1025, 20.4, 24.2, 11.4),
    ("microsoft/Phi-4-mini-instruct", 4097, 57.1, 44.5, 14.8),
    ("microsoft/Phi-4-mini-instruct", 16385, 306.8, 113.1, 27.1),
)


# ------------------------------------------- TCS T1 round 3: long context and large models
# (2026-10-02) Same protocol as the grid. Predicted blind under the current rule
# (spikes/accuracy/round3*.json). The bf16 Qwen2.5-72B at TP2 couldn't start on TCS (NCCL creates a
# CUDA memory pool, which the vGPU refuses), so its AWQ build on one card stands in. Model ->
# (config, weights GiB, GPU KV pool GiB).
T1_ROUND3_SIZES: Dict[str, Tuple[Dict[str, Any], float, float]] = {
    "Qwen/Qwen3-8B": (QWEN3_8B, 15.27, 8.0),
    "Qwen/Qwen3-14B": (QWEN3_14B, 27.5, 9.0),
    "microsoft/Phi-4-mini-instruct": (PHI4_MINI, 7.2, 18.0),
    "Qwen/Qwen3-30B-A3B": (QWEN3_30B_A3B, 56.9, 9.0),
    "Qwen/Qwen2.5-72B-Instruct-AWQ": (QWEN2_5_72B_AWQ, 38.5, 9.0),
}
T1_ROUND3: Sequence[Tuple[str, int, int, int, float, float, float]] = (
    ("Qwen/Qwen2.5-72B-Instruct-AWQ", 1024, 4, 72, 2.36, 2.85, 0.018),
    ("Qwen/Qwen2.5-72B-Instruct-AWQ", 1024, 16, 72, 3.57, 5.23, 0.017),
    ("Qwen/Qwen2.5-72B-Instruct-AWQ", 4096, 4, 18, 1.13, 2.30, 0.124),
    ("Qwen/Qwen2.5-72B-Instruct-AWQ", 4096, 16, 18, 1.26, 3.04, 0.063),
    ("Qwen/Qwen2.5-72B-Instruct-AWQ", 16384, 4, 4, 0.30, 0.78, 0.200),
    ("Qwen/Qwen2.5-72B-Instruct-AWQ", 16384, 16, 4, 0.30, 0.81, 0.033),
    ("Qwen/Qwen3-14B", 32768, 2, 4, 0.61, 1.38, 0.033),
    ("Qwen/Qwen3-14B", 32768, 8, 4, 0.66, 1.42, 0.137),
    ("Qwen/Qwen3-30B-A3B", 1024, 4, 240, 11.31, 9.23, 0.007),
    ("Qwen/Qwen3-30B-A3B", 1024, 16, 240, 25.77, 18.52, 0.021),
    ("Qwen/Qwen3-30B-A3B", 4096, 4, 60, 8.09, 8.41, 0.019),
    ("Qwen/Qwen3-30B-A3B", 4096, 16, 60, 13.15, 18.00, 0.022),
    ("Qwen/Qwen3-30B-A3B", 16384, 4, 15, 2.87, 5.12, 0.031),
    ("Qwen/Qwen3-30B-A3B", 16384, 16, 15, 2.89, 6.24, 0.032),
    ("Qwen/Qwen3-8B", 32768, 2, 4, 1.02, 2.18, 0.078),
    ("Qwen/Qwen3-8B", 32768, 8, 4, 1.02, 2.27, 0.068),
    ("microsoft/Phi-4-mini-instruct", 65536, 2, 6, 0.58, 1.46, 0.034),
    ("microsoft/Phi-4-mini-instruct", 65536, 8, 6, 0.67, 1.79, 0.135),
    ("microsoft/Phi-4-mini-instruct", 122880, 2, 3, 0.22, 0.68, 0.091),
    ("microsoft/Phi-4-mini-instruct", 122880, 8, 3, 0.22, 0.69, 0.140),
)
T1_ROUND3_SINGLES: Sequence[Tuple[str, int, float, float, float]] = (
    ("Qwen/Qwen2.5-72B-Instruct-AWQ", 1025, 256.9, 46.1, 29.6),
    ("Qwen/Qwen2.5-72B-Instruct-AWQ", 4097, 953.5, 76.7, 32.7),
    ("Qwen/Qwen2.5-72B-Instruct-AWQ", 16385, 4337.0, 177.9, 48.1),
    ("Qwen/Qwen3-14B", 32769, 2135.5, 232.3, 46.0),
    ("Qwen/Qwen3-30B-A3B", 1025, 61.3, 25.9, 13.1),
    ("Qwen/Qwen3-30B-A3B", 4097, 85.6, 38.6, 16.7),
    ("Qwen/Qwen3-30B-A3B", 16385, 446.2, 81.2, 28.1),
    ("Qwen/Qwen3-8B", 32769, 1299.8, 151.8, 41.5),
    ("microsoft/Phi-4-mini-instruct", 65537, 2225.8, 392.6, 73.7),
    ("microsoft/Phi-4-mini-instruct", 122881, 6447.1, 731.8, 131.2),
)


# ------------------------------------------------ TCS CPU engines: KV space (FR-T0-4), 2026-10-02
# Two upstream vLLM 0.30 CPU engines (vllm/vllm-openai-cpu:v0.30.0), 32 cores each, one per NUMA
# node of cscps3ee8ubu02, side by side: budsim's demand-sized VLLM_CPU_KVCACHE_SPACE against the
# planner's cpu_kv_gib (whole GiB, as budcluster renders it), on the shared-prefix bench with a
# working set ~4x demand. Predicted blind (spikes/accuracy/cpu_predictions.json). (model, prefix,
# concurrency, prefixes, demand GiB, demand-space req/s, planned-space req/s, spread)
TCS_CPU: Sequence[Tuple[str, int, int, int, int, float, float, float]] = (
    ("Qwen/Qwen3-0.6B", 1024, 4, 37, 1, 4.18, 5.38, 0.098),
    ("Qwen/Qwen3-0.6B", 1024, 8, 74, 2, 6.09, 6.81, 0.076),
    ("Qwen/Qwen3-0.6B", 4096, 4, 19, 2, 1.60, 2.88, 0.035),
    ("Qwen/Qwen3-0.6B", 4096, 8, 37, 4, 2.02, 3.33, 0.035),
    ("Qwen/Qwen3-1.7B", 1024, 4, 37, 1, 2.23, 2.99, 0.045),
    ("Qwen/Qwen3-1.7B", 4096, 4, 19, 2, 1.02, 1.78, 0.034),
)
TCS_CPU_MODELS: Dict[str, Tuple[Dict[str, Any], float]] = {  # config, weights GiB
    "Qwen/Qwen3-0.6B": (QWEN3_06B, 1.4),
    "Qwen/Qwen3-1.7B": (QWEN3_17B, 3.8),
}


def _tcs_cpu_fixtures() -> List[Fixture]:
    out: List[Fixture] = []
    for model, prefix, conc, prefixes, demand, without, grown, spread in TCS_CPU:
        config, weight = TCS_CPU_MODELS[model]
        node = {
            "name": "cscps3ee8ubu02",
            "kv_capabilities": {},
            "devices": [{"type": "cpu", "available_memory_gb": 220.0}],
        }
        out.append(
            Fixture(
                name=f"tcs-cpu-{model.split('/')[1].lower()}-{_length(prefix)}-c{conc}",
                kind="load",
                tier="CPU",
                source="TCS CPU engines, KV space demand vs planned",
                deployment=DeploymentSpec(
                    model_config=config,
                    input_tokens=prefix + 256,
                    output_tokens=32,
                    concurrency=conc,
                    model_id=model,
                ),
                engine=EngineSpec.of("vllm", ["kv_events", "cache_salt"]),
                group=NodeGroup(
                    device_type="cpu_high",
                    weight_memory_gb=_gb(weight),
                    kv_cache_memory_gb=_gb(demand),
                    device_model_key="Xeon",
                ),
                infra=InfraSnapshot(node=node, cluster={"kv_capability": {}}),
                workload=_bench(prefix=prefix, suffix=256, prefixes=prefixes, concurrency=conc),
                recompute=lambda n: None,
                goodput_without=without,
                goodput_with=grown,
                noise=spread,
            )
        )
    return out


# ------------------------------------ TCS T1 round 4: blind hold-out of the round-3 fixes
# (2026-10-02) T1_MIN_RECOMPUTE_SAVED_MS, alloc mode's load cost and the first-use hit term were
# fitted on rounds 1-3; round 4 tests them on models none of the fits saw (OLMoE-1B-7B,
# Qwen1.5-MoE-A2.7B, Qwen3-4B, Qwen3-1.7B) and Qwen3-30B-A3B at 2K, predicted blind
# (spikes/accuracy/round4_predictions.json).
T1_ROUND4_SIZES: Dict[str, Tuple[Dict[str, Any], float, float]] = {
    "allenai/OLMoE-1B-7B-0125-Instruct": (OLMOE_1B_7B, 12.9, 6.0),
    "Qwen/Qwen1.5-MoE-A2.7B-Chat": (QWEN1_5_MOE_A2_7B, 26.7, 8.0),
    "Qwen/Qwen3-30B-A3B": (QWEN3_30B_A3B, 56.9, 9.0),
    "Qwen/Qwen3-4B": (QWEN3_4B, 7.5, 6.0),
    "Qwen/Qwen3-1.7B": (QWEN3_17B, 3.8, 6.0),
}
T1_ROUND4: Sequence[Tuple[str, int, int, int, float, float, float]] = (
    ("Qwen/Qwen1.5-MoE-A2.7B-Chat", 1024, 4, 107, 20.45, 19.13, 0.059),
    ("Qwen/Qwen1.5-MoE-A2.7B-Chat", 1024, 16, 107, 51.02, 40.48, 0.099),
    ("Qwen/Qwen1.5-MoE-A2.7B-Chat", 4096, 4, 27, 19.34, 17.18, 0.043),
    ("Qwen/Qwen1.5-MoE-A2.7B-Chat", 4096, 16, 27, 28.07, 32.44, 0.025),
    ("Qwen/Qwen1.5-MoE-A2.7B-Chat", 16384, 4, 7, 5.92, 8.96, 0.101),
    ("Qwen/Qwen1.5-MoE-A2.7B-Chat", 16384, 16, 7, 5.92, 9.12, 0.058),
    ("Qwen/Qwen3-1.7B", 2048, 4, 69, 28.06, 26.51, 0.005),
    ("Qwen/Qwen3-1.7B", 2048, 16, 69, 56.48, 56.67, 0.024),
    ("Qwen/Qwen3-1.7B", 4096, 4, 34, 21.23, 21.97, 0.009),
    ("Qwen/Qwen3-1.7B", 4096, 16, 34, 36.31, 42.91, 0.018),
    ("Qwen/Qwen3-30B-A3B", 2048, 4, 120, 10.21, 9.02, 0.004),
    ("Qwen/Qwen3-30B-A3B", 2048, 16, 120, 20.32, 19.88, 0.051),
    ("Qwen/Qwen3-4B", 1024, 4, 107, 18.90, 17.72, 0.050),
    ("Qwen/Qwen3-4B", 1024, 16, 107, 40.97, 43.52, 0.048),
    ("Qwen/Qwen3-4B", 2048, 4, 53, 15.54, 16.61, 0.011),
    ("Qwen/Qwen3-4B", 2048, 16, 53, 28.11, 37.95, 0.030),
    ("allenai/OLMoE-1B-7B-0125-Instruct", 1024, 4, 120, 26.27, 24.78, 0.025),
    ("allenai/OLMoE-1B-7B-0125-Instruct", 1024, 16, 120, 55.98, 49.36, 0.037),
    ("allenai/OLMoE-1B-7B-0125-Instruct", 2048, 4, 60, 25.01, 23.16, 0.018),
    ("allenai/OLMoE-1B-7B-0125-Instruct", 2048, 16, 60, 46.93, 46.00, 0.014),
    ("allenai/OLMoE-1B-7B-0125-Instruct", 3584, 4, 34, 20.99, 20.89, 0.030),
    ("allenai/OLMoE-1B-7B-0125-Instruct", 3584, 16, 34, 35.16, 38.51, 0.020),
)
T1_ROUND4_SINGLES: Sequence[Tuple[str, int, float, float, float]] = (
    ("Qwen/Qwen1.5-MoE-A2.7B-Chat", 1025, 51.4, 23.1, 10.2),
    ("Qwen/Qwen1.5-MoE-A2.7B-Chat", 4097, 56.9, 40.1, 12.8),
    ("Qwen/Qwen1.5-MoE-A2.7B-Chat", 16385, 275.4, 101.7, 24.3),
    ("Qwen/Qwen3-1.7B", 2049, 19.5, 27.9, 11.4),
    ("Qwen/Qwen3-1.7B", 4097, 42.3, 39.0, 13.1),
    ("Qwen/Qwen3-30B-A3B", 2049, 63.3, 30.3, 15.4),
    ("Qwen/Qwen3-4B", 1025, 24.0, 22.0, 11.0),
    ("Qwen/Qwen3-4B", 2049, 34.1, 27.9, 12.2),
    ("allenai/OLMoE-1B-7B-0125-Instruct", 1025, 27.1, 18.1, 8.3),
    ("allenai/OLMoE-1B-7B-0125-Instruct", 2049, 26.6, 24.0, 9.7),
    ("allenai/OLMoE-1B-7B-0125-Instruct", 3585, 32.0, 30.0, 11.9),
)


def _length(tokens: int) -> str:
    return f"{round(tokens / 1024)}k" if tokens >= 1024 else str(tokens)


# ------------------------------------ TCS T1 round 5: blind MoE hold-out (2026-10-03) The MoE rules
# (T1 on throughput, a hit saving >= T1_MIN_RECOMPUTE_SAVED_MS_MOE of batched work, active
# parameters at top-k of E) on three MoE families none of the fits saw: Mixtral-8x7B (AWQ),
# DeepSeek-V2-Lite (MLA) and granite-3.1-3b-a800m, predicted blind
# (spikes/accuracy/round5_predictions.json): 17/18, precision 1.00. The miss is DeepSeek-V2-Lite 4K
# c16 (+16% with T1; a hit saves 53 ms, under the 60 ms MoE guard).
T1_ROUND5_SIZES: Dict[str, Tuple[Dict[str, Any], float, float]] = {
    "TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ": (MIXTRAL_8X7B_AWQ, 22.96, 6.0),
    "deepseek-ai/DeepSeek-V2-Lite-Chat": (DEEPSEEK_V2_LITE, 29.25, 1.5),
    "ibm-granite/granite-3.1-3b-a800m-instruct": (GRANITE_31_3B_A800M, 6.15, 3.0),
}
T1_ROUND5: Sequence[Tuple[str, int, int, int, float, float, float]] = (
    ("TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ", 1024, 4, 120, 6.80, 8.57, 0.015),
    ("TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ", 1024, 16, 120, 10.28, 15.50, 0.037),
    ("TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ", 4096, 4, 30, 3.29, 6.73, 0.009),
    ("TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ", 4096, 16, 30, 3.51, 9.93, 0.045),
    ("TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ", 16384, 4, 8, 0.89, 2.91, 0.022),
    ("TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ", 16384, 16, 8, 0.95, 3.06, 0.095),
    ("deepseek-ai/DeepSeek-V2-Lite-Chat", 1024, 4, 126, 11.55, 11.80, 0.012),
    ("deepseek-ai/DeepSeek-V2-Lite-Chat", 1024, 16, 126, 27.24, 23.42, 0.029),
    ("deepseek-ai/DeepSeek-V2-Lite-Chat", 4096, 4, 32, 11.03, 11.11, 0.015),
    ("deepseek-ai/DeepSeek-V2-Lite-Chat", 4096, 16, 32, 18.36, 21.24, 0.057),
    ("deepseek-ai/DeepSeek-V2-Lite-Chat", 16384, 4, 8, 4.29, 7.41, 0.009),
    ("deepseek-ai/DeepSeek-V2-Lite-Chat", 16384, 16, 8, 4.68, 9.09, 0.009),
    ("ibm-granite/granite-3.1-3b-a800m-instruct", 1024, 4, 120, 18.53, 17.23, 0.017),
    ("ibm-granite/granite-3.1-3b-a800m-instruct", 1024, 16, 120, 40.73, 31.43, 0.041),
    ("ibm-granite/granite-3.1-3b-a800m-instruct", 4096, 4, 30, 17.27, 14.96, 0.010),
    ("ibm-granite/granite-3.1-3b-a800m-instruct", 4096, 16, 30, 30.62, 25.16, 0.027),
    ("ibm-granite/granite-3.1-3b-a800m-instruct", 16384, 4, 8, 6.50, 8.87, 0.014),
    ("ibm-granite/granite-3.1-3b-a800m-instruct", 16384, 16, 8, 6.99, 9.70, 0.019),
)
T1_ROUND5_SINGLES: Sequence[Tuple[str, int, float, float, float]] = (
    ("ibm-granite/granite-3.1-3b-a800m-instruct", 1025, 40.3, 20.2, 10.9),
    ("ibm-granite/granite-3.1-3b-a800m-instruct", 4097, 47.4, 30.7, 14.3),
    ("ibm-granite/granite-3.1-3b-a800m-instruct", 16385, 171.3, 67.6, 24.5),
    ("deepseek-ai/DeepSeek-V2-Lite-Chat", 1025, 50.7, 19.9, 11.8),
    ("deepseek-ai/DeepSeek-V2-Lite-Chat", 4097, 59.6, 28.1, 15.2),
    ("deepseek-ai/DeepSeek-V2-Lite-Chat", 16385, 258.6, 59.0, 27.1),
    ("TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ", 1025, 97.4, 23.6, 12.6),
    ("TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ", 4097, 345.4, 38.5, 17.0),
    ("TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ", 16385, 1471.7, 89.4, 31.2),
)


def _t1_grid_fixtures() -> List[Fixture]:
    out: List[Fixture] = []
    for table, label, sizes, min_prompts in (
        (T1_GRID, "grid", T1_GRID_SIZES, 64),
        (T1_HOLDOUT, "hold-out", T1_GRID_SIZES, 64),
        (T1_ROUND3, "round 3", T1_ROUND3_SIZES, 16),
        (T1_ROUND4, "round 4", T1_ROUND4_SIZES, 16),
        (T1_ROUND5, "round 5", T1_ROUND5_SIZES, 16),
    ):
        for model, prefix, conc, prefixes, without, with_t1, spread in table:
            uses = max(8 * prefixes, min_prompts) / prefixes
            config, weight, pool = sizes[model]
            base = model.split("/")[1].lower()
            out.append(
                Fixture(
                    name=f"tcs-t1-{base}-{_length(prefix)}-c{conc}",
                    kind="load",
                    tier="T1",
                    source=f"TCS T1 accuracy {label}, H100 vGPU, alloc mode",
                    recompute=interpolated(GENZ_H100_MS[base]),
                    workload=_bench(
                        prefix=prefix, suffix=256, prefixes=prefixes, concurrency=conc, uses=uses
                    ),
                    goodput_without=without,
                    goodput_with=with_t1,
                    noise=spread,
                    **_tcs(
                        model=model,
                        config=config,
                        weight_gib=weight,
                        pool_gib=pool,
                        features=BUD_010,
                        e2e_s=None,
                    ),
                )
            )
    singles = [(row, T1_GRID_SIZES) for row in T1_SINGLES]
    singles += [(row, T1_ROUND3_SIZES) for row in T1_ROUND3_SINGLES]
    singles += [(row, T1_ROUND4_SIZES) for row in T1_ROUND4_SINGLES]
    singles += [(row, T1_ROUND5_SIZES) for row in T1_ROUND5_SINGLES]
    for (model, prompt, cold, tier, gpu), sizes in singles:
        config, weight, pool = sizes[model]
        base = model.split("/")[1].lower()
        genz = interpolated(GENZ_H100_MS[base])
        cached = (prompt - 1) // 256 * 256
        kib = KVGeometry.from_model(config, seq_length=prompt + 32).bytes_per_token_rank
        sessions = max(2.0, 3 * pool * GIB / (kib * prompt))
        out.append(
            Fixture(
                name=f"tcs-t1-{base}-{_length(prompt - 1)}-single",
                kind="single",
                tier="T1",
                source="TCS T1 accuracy grid, one request",
                recompute=genz,
                workload=_single(prompt, sessions=sessions),
                reload_ms=tier - gpu,
                recompute_ms=cold - gpu,
                genz_ms=genz(cached),
                **_tcs(
                    model=model,
                    config=config,
                    weight_gib=weight,
                    pool_gib=pool,
                    features=BUD_010,
                    e2e_s=None,
                ),
            )
        )
    return out


# ------------------------------------ TCS disk tier (T2): T1 only vs T1 + vLLM's fs tier
# (2026-10-02/03) vLLM 0.30 with spike per_copy_patch.py: T1 in register mode on the vGPU (tiering
# needs its shared region), every copy its own cudaMemcpyAsync. The node is RAM-limited by
# construction (T1 at the planner's size for twice the GPU pool in free host memory), so the working
# set spills to the fs tier on the node's ext4 VMware disk (O_DIRECT: 3.1 GB/s single stream, 5.0
# with 8 readers). Predicted blind at 5.0 GB/s with the idle break-even
# (spikes/accuracy/t2_predictions.json): 4/5 decided cells right, the miss Qwen3-8B 4K c16 (+14%
# with T2). Three more cells stalled the engine (FRD-023 S1); they have no throughput and aren't
# fixtures. Planner fixed after: staged T1 hop, per-copy T1 link, throughput rule.
T2_SIZES: Dict[str, Tuple[Dict[str, Any], float, float]] = {  # config, weights GiB, pool GiB
    "Qwen/Qwen3-8B": (QWEN3_8B, 15.27, 4.0),
    "Qwen/Qwen3-14B": (QWEN3_14B, 27.5, 6.0),
    "Qwen/Qwen2.5-72B-Instruct-AWQ": (QWEN2_5_72B_AWQ, 38.5, 6.0),
}
# What the T2 health gate reports for the class: a single-stream O_DIRECT sequential read.
TCS_DISK_GBPS = 3.1
T2_ROUND = [  # model, prefix, concurrency, prefixes, prompts, T1-only req/s, T1+T2 req/s, spread
    ("Qwen/Qwen2.5-72B-Instruct-AWQ", 4096, 4, 24, 192, 0.9600, 1.5750, 0.021),
    ("Qwen/Qwen2.5-72B-Instruct-AWQ", 16384, 2, 6, 48, 0.2400, 0.3050, 0.098),
    ("Qwen/Qwen2.5-72B-Instruct-AWQ", 16384, 8, 6, 48, 0.2400, 0.2850, 0.175),
    ("Qwen/Qwen3-8B", 4096, 4, 36, 288, 6.1550, 6.1300, 0.036),
    ("Qwen/Qwen3-8B", 4096, 16, 36, 288, 7.3950, 8.4250, 0.015),
]
T2_DISK_SINGLES = [  # model, prompt tokens, cold ms, disk-hit ms, GPU-hit ms (medians)
    ("Qwen/Qwen2.5-72B-Instruct-AWQ", 4097, 1036.9, 640.5, 32.6),
    ("Qwen/Qwen2.5-72B-Instruct-AWQ", 16385, 4579.3, 2713.2, 44.4),
    ("Qwen/Qwen3-14B", 32769, 2373.8, 1837.9, 46.9),
    ("Qwen/Qwen3-8B", 4097, 141.8, 219.1, 15.9),
    ("Qwen/Qwen3-8B", 16385, 649.5, 927.0, 27.9),
]
T1_PER_COPY_SINGLES = [  # model, prompt tokens, cold ms, T1-hit ms, GPU-hit ms (medians)
    ("Qwen/Qwen2.5-72B-Instruct-AWQ", 4097, 1021.3, 114.8, 32.0),
    ("Qwen/Qwen3-8B", 4097, 137.2, 63.1, 15.1),
    ("Qwen/Qwen3-8B", 16385, 625.0, 171.7, 26.1),
]


def _tcs_t2(*, model: str) -> Dict[str, Any]:
    config, weight_gib, pool_gib = T2_SIZES[model]
    node = {
        "name": "cscps3ee8ubu02",
        "kv_capabilities": TCS_NODE,
        # host memory the T1 sizing sees: twice the pool, as on the runs (T1_HOST_RAM_SHARE)
        "devices": [{"type": "cpu", "available_memory_gb": _gb(2 * pool_gib) + 6.5}],
    }
    cluster = {
        "kv_capability": {
            "storage_classes": [
                {
                    "name": "local-path",
                    "node_local": True,
                    "kv_read_gbps": TCS_DISK_GBPS,
                    "capacity_gib": 200.0,
                }
            ],
            "kv_storage": {"class": "local-path", "enabled": True},
        }
    }
    return dict(
        deployment=DeploymentSpec(
            model_config=config, input_tokens=0, output_tokens=0, concurrency=1, model_id=model
        ),
        engine=EngineSpec.of("vllm", UPSTREAM_030 + [T1_PER_COPY_FEATURE]),
        group=NodeGroup(
            device_type="cuda",
            weight_memory_gb=_gb(weight_gib),
            kv_cache_memory_gb=_gb(1.0),
            device_total_memory_gb=_pool_gib_device(pool_gib, weight_gib),
            device_model_key="H100XM-80C",
            device_generation="hopper",
            device_tflops=989.5,
            hardware={"Flops": 989.5, "Memory_size": 80, "Memory_BW": 3350, "ICN": 450},
        ),
        infra=InfraSnapshot(node=node, cluster=cluster),
    )


def _t2_fixtures() -> List[Fixture]:
    out: List[Fixture] = []
    for model, prefix, conc, prefixes, prompts, without, with_t2, spread in T2_ROUND:
        base = model.split("/")[1].lower()
        out.append(
            Fixture(
                name=f"tcs-t2-{base}-{_length(prefix)}-c{conc}",
                kind="load",
                tier="T2",
                source="TCS T2 round, H100 vGPU, fs tier on the node disk",
                recompute=interpolated(GENZ_H100_MS[base]),
                workload=_bench(
                    prefix=prefix,
                    suffix=256,
                    prefixes=prefixes,
                    concurrency=conc,
                    uses=prompts / prefixes,
                ),
                goodput_without=without,
                goodput_with=with_t2,
                noise=spread,
                **_tcs_t2(model=model),
            )
        )
    for table, tier in ((T2_DISK_SINGLES, "T2"), (T1_PER_COPY_SINGLES, "T1")):
        for model, prompt, cold, hit, gpu in table:
            config, _, pool = T2_SIZES[model]
            base = model.split("/")[1].lower()
            genz = interpolated(GENZ_H100_MS[base])
            kib = KVGeometry.from_model(config, seq_length=prompt + 32).bytes_per_token_rank
            sessions = max(2.0, 3 * pool * GIB / (kib * prompt))
            out.append(
                Fixture(
                    name=f"tcs-{tier.lower()}-{'disk' if tier == 'T2' else 'per-copy'}-{base}"
                    f"-{_length(prompt - 1)}-single",
                    kind="single",
                    tier=tier,
                    source="TCS T2 round, one request",
                    recompute=genz,
                    workload=_single(prompt, sessions=sessions),
                    reload_ms=hit - gpu,
                    recompute_ms=cold - gpu,
                    genz_ms=genz((prompt - 1) // 256 * 256),
                    **_tcs_t2(model=model),
                )
            )
    return out


def fixtures() -> List[Fixture]:
    """Every measured scenario, in the order the validation report lists them."""
    genz_8b = interpolated(GENZ_H100_MS["qwen3-8b"])
    genz_27b = interpolated(GENZ_H100_MS["qwen3.8-27b"])
    rtx = per_token(MEASURED_RTX3050_MS["qwen3-0.6b"])
    out: List[Fixture] = []

    def add(**kw: Any) -> None:
        out.append(Fixture(**kw))

    tcs8 = dict(model="Qwen/Qwen3-8B", config=QWEN3_8B, weight_gib=15.27, pool_gib=8.0)
    # vLLM reported the 27B's 8 GiB pool as 103,641 tokens (its hybrid allocator also holds state):
    # 103,641 x 64 KiB = 6.33 GiB of attention KV.
    tcs27 = dict(model="Qwen/Qwen3.8-27B", config=QWEN3_8_27B, weight_gib=51.0, pool_gib=6.33)
    s10 = "spike S10, TCS H100 vGPU, pool on the CPU node over TCP"

    # -- TCS, Mooncake pool, under load (oracle: goodput)
    add(
        name="tcs-8b-4k-load",
        kind="load",
        tier="T3",
        source=s10,
        recompute=genz_8b,
        workload=_bench(prefix=4096, suffix=256, prefixes=24, concurrency=8),
        goodput_without=11.83,
        goodput_with=2.13,
        **_tcs(**tcs8, features=UPSTREAM_030, e2e_s=0.197 + 32 * 0.0084, pool=TCS_POOL),
    )
    add(
        name="tcs-8b-16k-load",
        kind="load",
        tier="T3",
        source=s10,
        recompute=genz_8b,
        workload=_bench(prefix=16384, suffix=256, prefixes=8, concurrency=3, uses=6),
        goodput_without=2.12,
        goodput_with=0.48,
        **_tcs(**tcs8, features=UPSTREAM_030, e2e_s=0.760 + 32 * 0.009, pool=TCS_POOL),
    )
    add(
        name="tcs-27b-4k-load",
        kind="load",
        tier="T3",
        source=s10,
        recompute=genz_27b,
        workload=_bench(prefix=4096, suffix=256, prefixes=40, concurrency=8, uses=6),
        goodput_without=3.18,
        goodput_with=1.70,
        **_tcs(**tcs27, features=UPSTREAM_030, e2e_s=0.618 + 32 * 0.0224, pool=TCS_POOL),
    )
    add(
        name="tcs-27b-16k-load",
        kind="load",
        tier="T3",
        source=s10,
        recompute=genz_27b,
        workload=_bench(prefix=16384, suffix=256, prefixes=8, concurrency=3, uses=6),
        goodput_without=0.67,
        goodput_with=0.55,
        **_tcs(**tcs27, features=UPSTREAM_030, e2e_s=2.624 + 32 * 0.0216, pool=TCS_POOL),
    )

    # -- TCS, Mooncake pool, one request (component checks)
    for name, cfg, recompute, prompt, sessions, reload, measured_recompute, genz, nbytes in (
        ("tcs-8b-4k-single", tcs8, genz_8b, 4097, 30, 1321.6 - 17.8, 104.3 - 17.8, 99.2, 603979776),
        (
            "tcs-8b-16k-single",
            tcs8,
            genz_8b,
            16129,
            8,
            4562.8 - 26.1,
            484.2 - 26.1,
            461.5,
            2378170368,
        ),
        (
            "tcs-27b-4k-single",
            tcs27,
            genz_27b,
            4097,
            64,
            674.9 - 70.4,
            348.6 - 70.4,
            289.6,
            411041792,
        ),
        (
            "tcs-27b-16k-single",
            tcs27,
            genz_27b,
            16129,
            17,
            1242.6 - 83.9,
            1392.6 - 83.9,
            1337.2,
            1181745152,
        ),
    ):
        add(
            name=name,
            kind="single",
            tier="T3",
            source=s10,
            recompute=recompute,
            workload=_single(prompt, sessions=sessions),
            reload_ms=reload,
            recompute_ms=measured_recompute,
            genz_ms=genz,
            measured_bytes=nbytes,
            **_tcs(**cfg, features=UPSTREAM_030, e2e_s=None, pool=TCS_POOL),
        )

    # -- TCS, T1 in alloc mode (spike S2): a 32K hit, 155 ms vs 1,230 ms recompute. The GPU-hit
    # TTFT at 32K wasn't recorded; ~30 ms (26 ms at 16K) is subtracted.
    add(
        name="tcs-8b-32k-t1-single",
        kind="single",
        tier="T1",
        source="spike S2, TCS H100 vGPU, alloc-mode T1",
        recompute=genz_8b,
        workload=_single(32769, sessions=2),
        reload_ms=155.0 - 30.0,
        recompute_ms=1230.0 - 30.0,
        genz_ms=1152.1,
        notes="GPU-hit TTFT at 32K estimated (30 ms)",
        **_tcs(
            model="Qwen/Qwen3-8B",
            config=QWEN3_8B,
            weight_gib=15.27,
            pool_gib=2.0,
            features=BUD_010,
            e2e_s=None,
        ),
    )

    # -- accubits01, RTX 3050 Laptop (x4 link), Qwen3-0.6B
    acc = "accubits01 (spike lmcache tables and the Phase 1 bare-metal check)"
    add(
        name="accubits-0.6b-2k-t1-load",
        kind="load",
        tier="T1",
        source=acc,
        recompute=rtx,
        workload=_bench(prefix=2048, suffix=256, prefixes=12, concurrency=4),
        goodput_without=3.62,
        goodput_with=5.24,
        **_accubits(features=UPSTREAM_030, e2e_s=4 / 3.62),
    )
    add(
        name="accubits-0.6b-1k-t1-load",
        kind="load",
        tier="T1",
        source=acc,
        recompute=rtx,
        workload=_bench(prefix=1024, suffix=256, prefixes=16, concurrency=4),
        goodput_without=5.55,
        goodput_with=6.40,
        **_accubits(features=UPSTREAM_030, e2e_s=4 / 5.55),
    )
    add(
        name="accubits-0.6b-4k-t1-single",
        kind="single",
        tier="T1",
        source=acc,
        recompute=rtx,
        workload=_single(4097, sessions=6),
        reload_ms=130.0 - 23.5,
        recompute_ms=517.5 - 23.5,
        **_accubits(features=UPSTREAM_030, e2e_s=None),
    )
    add(
        name="accubits-0.6b-3.5k-t1-single",
        kind="single",
        tier="T1",
        source=acc,
        recompute=rtx,
        workload=_single(3585, sessions=7),
        reload_ms=147.5 - 28.5,
        recompute_ms=414.0 - 28.5,
        **_accubits(features=UPSTREAM_030, e2e_s=None),
    )
    # The pool on accubits ran on the same node, and no health gate measured its rate or floor: the
    # rate below is inferred from the single-request load (~1.9 GB/s effective at 64 KiB segments),
    # and the floor is the TCP default fitted on TCS.
    acc_pool = {"backend": "mooncake", "transport": "tcp", "read_gbps": 3.8, "capacity_gib": 6.0}
    pool_only = ["kv_events", "cache_salt", "mooncake_store"]
    add(
        name="accubits-0.6b-2k-pool-load",
        kind="load",
        tier="T3",
        source=acc,
        recompute=rtx,
        workload=_bench(prefix=2048, suffix=256, prefixes=12, concurrency=4),
        goodput_without=3.62,
        goodput_with=2.57,
        gate_measured=False,
        **_accubits(features=pool_only, e2e_s=4 / 3.62, pool=acc_pool),
    )
    add(
        name="accubits-0.6b-4k-pool-single",
        kind="single",
        tier="T3",
        source=acc,
        recompute=rtx,
        workload=_single(4097, sessions=6),
        reload_ms=269.0 - 23.5,
        recompute_ms=517.5 - 23.5,
        gate_measured=False,
        notes="pool rate inferred and floor defaulted: no health gate on accubits",
        **_accubits(features=pool_only, e2e_s=None, pool=acc_pool),
    )
    out.extend(_t1_grid_fixtures())
    out.extend(_tcs_cpu_fixtures())
    out.extend(_t2_fixtures())
    return out


@dataclass(frozen=True)
class GeometryCheck:
    """A measured byte count a tier moved, and the block size it moved at."""

    name: str
    config: Mapping[str, Any]
    prompt_tokens: int
    block_size: Optional[int]  # None: the planner's own block
    measured_bytes: int
    source: str = field(default="spike S10 (Mooncake load_get_total_bytes)")


def geometry_checks() -> List[GeometryCheck]:
    """Bytes the TCS pool reported per prefix load, at the default and two raised block sizes."""
    rows = []
    for block, four_k, sixteen_k in (
        (None, 411041792, 1181745152),
        (1568, 513802240, 1335885824),
        (3136, 822083584, 1644167168),
    ):
        label = "default" if block is None else str(block)
        rows.append(GeometryCheck(f"27b-4k-block-{label}", QWEN3_8_27B, 4097, block, four_k))
        rows.append(GeometryCheck(f"27b-16k-block-{label}", QWEN3_8_27B, 16129, block, sixteen_k))
    rows.append(GeometryCheck("8b-4k", QWEN3_8B, 4097, None, 603979776))
    rows.append(GeometryCheck("8b-16k", QWEN3_8B, 16129, None, 2378170368))
    return rows
