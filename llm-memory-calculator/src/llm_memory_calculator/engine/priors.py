"""Prior values for every grounding kind the vLLM engine layer reads (FRD-027 §9.1, the ladder's last level).

A prior is what a plan uses when nothing on any ladder is measured. Residuals are deliberately wide: a
prior-level prediction is reported ``ungrounded`` (FR-GND-2) and can never win a comparison, so an unknown
SKU, family or build gets engine defaults and the capacity rules only (NFR-5).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

GIB = 1024**3

#: kind -> (value, relative residual). Memory kinds carry an absolute residual in bytes.
PRIORS: Dict[str, Tuple[float, float]] = {
    "memory.overhead": (
        3.3 * GIB,
        1.5 * GIB,
    ),  # non-KV, non-weight memory inside the fraction at the reference settings
    "memory.capture": (
        6.5e-7 * GIB,
        0.5,
    ),  # bytes per captured sequence per hidden unit (CUDA graphs, buffers)
    "memory.chunk": (
        2.2e-6,
        0.6,
    ),  # bytes per (batched token x (hidden + intermediate)) away from 8192
    "memory.spec": (1.6 * GIB, 1.0 * GIB),
    "decode.kappa": (1.15, 0.45),  # multiplier on the decode roofline
    "decode.tau_layer": (0.025, 0.6),  # ms per layer per step
    "decode.tau_moe": (0.05, 0.8),  # ms per MoE layer per step
    "step.tau": (1.0, 0.5),  # ms per engine step
    "prefill.kappa": (1.6, 0.45),
    "prefill.tau_layer": (0.04, 0.6),
    "prefill.mix": (1.0, 0.45),
    "api.tau": (12.0, 0.6),
    "arrival.sync": (1.0, 0.3),
    "arrival.drain": (0.0, 0.5),  # rounds lost to the tail of a fixed-count run (0: none)
    "arrival.queue": (
        0.5,
        0.6,
    ),  # decode steps a steady arrival waits before its prefill runs  # fraction of a synchronized benchmark wave that survives (1 = full burst)  # ms of request handling around TTFT
    "precision.ratio": (1.0, 0.25),
    "spec.acceptance": (1.0, 0.5),  # accepted draft tokens per step, median request (TPOT)
    "spec.acceptance_mean": (
        0.85,
        0.5,
    ),  # accepted draft tokens per step, mean over requests (throughput)
    "spec.draft_tau": (0.35, 0.6),  # ms per draft token beyond its weight bytes
    "spec.mix": (
        1.0,
        0.4,
    ),  # other requests' prefill interference on TPOT with a speculative method, vs the base fit
    "lora.ratio": (1.02, 0.05),
    "precision.prefill": (
        1.0,
        0.3,
    ),  # prefill efficiency of a weight format relative to the base fit
    "spec.prefill": (
        1.1,
        0.3,
    ),  # prefill cost with a speculative method on (the draft also runs over the prompt)
    "boot.base": (90.0, 0.5),  # seconds: process start, weights load excluded
    "boot.capture": (0.08, 0.6),  # seconds per captured CUDA-graph size
    "boot.load_bandwidth": (2.0 * GIB, 0.6),  # bytes/s from the model store
    "stall.first_shape": (0.0, 0.5),  # p99 TTFT inflation without kernel warm-up, ms
}

#: Per-token acceptance probability for the acceptance prior, by (method, workload class).
ACCEPTANCE_PRIOR = {
    ("mtp", "real"): 0.82,
    ("mtp", "synthetic"): 0.55,
    ("eagle3", "real"): 0.45,
    ("eagle3", "synthetic"): 0.25,
    ("ngram", "real"): 0.15,
    ("ngram", "synthetic"): 0.45,
    ("draft", "real"): 0.5,
    ("draft", "synthetic"): 0.3,
}
SAMPLED_FACTOR = 0.75


def acceptance_prior(method: str, k: int, sampling: str, workload_class: str) -> float:
    p = ACCEPTANCE_PRIOR.get((method, workload_class), 0.4)
    if sampling == "sampled":
        p *= SAMPLED_FACTOR
    return float(sum(p**i for i in range(1, max(int(k), 0) + 1)))


def prior(kind: str) -> Tuple[float, float]:
    return PRIORS[kind]


def prior_detail(kind: str, detail: Dict[str, Any]) -> Tuple[float, float]:
    if kind == "arrival.sync":
        return (1.0 if detail.get("method") in (None, "none") else 0.5), PRIORS[kind][1]
    if kind in ("spec.acceptance", "spec.acceptance_mean"):
        value = acceptance_prior(
            str(detail.get("method")),
            int(detail.get("k") or 1),
            str(detail.get("sampling")),
            str(detail.get("workload")),
        )
        return (value if kind == "spec.acceptance" else 0.85 * value), PRIORS[kind][1]
    return PRIORS[kind]
