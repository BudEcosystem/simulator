"""Recompute vs reload: the break-even every offload tier must pass (ARCHITECTURE §10.3 rule 7).

    reload = (floor + bytes / bandwidth) / (1 - utilisation)   must be   < BREAK_EVEN_RATIO x
    recompute

* **recompute** is GenZ's prefill estimate (``estimate_prefill_performance``), the estimator behind
  budsim's TTFT, for the cached part of the prefix on this hardware and TP. On TCS's H100 it was
  4-17% low (spike S10). Without GenZ (an unknown model id or GPU), a FLOPs formula at
  ``PREFILL_MFU`` stands in, counting quadratic attention only on full-attention layers.
* **reload** is a fixed per-load cost plus the bytes over the tier's effective bandwidth. One load
  decides the break-even. The deployment's loads and write-through then only have to fit the link:
  under ``MAX_LINK_UTILISATION`` of it, a queued load costs that request TTFT (reported,
  M/M/1-style) but not throughput, because the GPU serves other requests meanwhile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from ..parameter_counter import UniversalParameterCounter
from .constants import BREAK_EVEN_RATIO, GB, MAX_LINK_UTILISATION, PREFILL_MFU


def _text_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    text = config.get("text_config")
    return text if isinstance(text, Mapping) else config


def _routed_experts(text: Mapping[str, Any]) -> Tuple[int, int]:
    """(routed experts per MoE layer, experts each token runs): Qwen/OLMoE/Mixtral/DeepSeek and
    ERNIE-4.5 (``moe_num_experts``, ``moe_k``) spellings."""
    experts = int(
        text.get("num_experts")
        or text.get("num_local_experts")
        or text.get("n_routed_experts")
        or text.get("moe_num_experts")
        or 0
    )
    top_k = int(
        text.get("num_experts_per_tok")
        or text.get("num_experts_per_token")
        or text.get("moe_k")
        or 0
    )
    return experts, top_k


def _moe_layers(text: Mapping[str, Any]) -> int:
    """Layers that hold routed experts: DeepSeek keeps its first ``first_k_dense_replace`` layers
    dense; ERNIE-4.5 routes from ``moe_layer_start_index`` to ``moe_layer_end_index``."""
    layers = int(text.get("num_hidden_layers") or 0)
    if text.get("moe_layer_start_index") is not None:
        start = int(text["moe_layer_start_index"])
        end = int(text.get("moe_layer_end_index", layers - 1))
        interval = max(int(text.get("moe_layer_interval") or 1), 1)
        return max(0, (min(end, layers - 1) - start) // interval + 1)
    return max(0, layers - int(text.get("first_k_dense_replace") or 0))


def is_moe(model_config: Mapping[str, Any]) -> bool:
    """Routed experts with a top-k: a mixture-of-experts model."""
    experts, top_k = _routed_experts(_text_config(model_config))
    return experts > 1 and top_k > 0


def active_parameters(model_config: Mapping[str, Any]) -> float:
    """Parameters one token runs through: a MoE model's routed experts count at top-k of E.

    Qwen3-30B-A3B has 30.5B parameters and runs ~3.3B per token; counting all of them made its
    recompute ~9x too slow, so every T1 looked like a win.
    """
    total = float(UniversalParameterCounter().count_parameters(dict(model_config)))
    text = _text_config(model_config)
    experts, top_k = _routed_experts(text)
    if experts <= 1 or top_k <= 0:
        return total
    hidden = int(text.get("hidden_size") or 0)
    width = int(text.get("moe_intermediate_size") or text.get("intermediate_size") or 0)
    routed = _moe_layers(text) * experts * 3 * hidden * width  # gated (up, gate, down) experts
    return max(total - routed * (1.0 - top_k / experts), 0.0)


def prefill_seconds(
    model_config: Mapping[str, Any],
    tokens: int,
    device_tflops: Optional[float],
    *,
    attention_layers: Optional[int] = None,
) -> Optional[float]:
    """Estimate one prefill of ``tokens`` on the whole model, or None without the device's FLOPs.

    Dense matmuls (2 x parameters per token) plus causal attention (2 x attention layers x tokens^2
    x hidden). The fallback when GenZ can't estimate the model.
    """
    if not device_tflops or device_tflops <= 0 or tokens <= 0:
        return None
    params = active_parameters(model_config)
    text = _text_config(model_config)
    layers = attention_layers or int(text.get("num_hidden_layers") or text.get("n_layer") or 0)
    hidden = int(text.get("hidden_size") or text.get("n_embd") or 0)
    flops = 2.0 * params * tokens + 2.0 * layers * tokens * tokens * hidden
    return flops / (device_tflops * 1e12 * PREFILL_MFU)


class RecomputeModel:
    """Prefill time for ``n`` prompt tokens on this deployment, and where the number came from.

    Order: a caller-supplied function (tests, recorded measurements); GenZ, when the model id and a
    GenZ hardware entry are known; the FLOPs formula, when the device's FLOPs are known; else None.
    """

    def __init__(
        self,
        *,
        model_config: Mapping[str, Any],
        model_id: Optional[str] = None,
        hardware: Optional[Mapping[str, Any]] = None,
        device_tflops: Optional[float] = None,
        tensor_parallel: int = 1,
        precision: str = "bf16",
        attention_layers: Optional[int] = None,
        fn: Optional[Callable[[int], Optional[float]]] = None,
    ) -> None:
        self._config = model_config
        self._model_id = model_id
        self._hardware = dict(hardware) if hardware else None
        self._tflops = device_tflops
        self._tp = max(int(tensor_parallel or 1), 1)
        self._bits = "bf16" if precision in ("bf16", "fp16", "float16", "bfloat16") else precision
        self._attention_layers = attention_layers
        self._fn = fn
        self._cache: Dict[int, Tuple[Optional[float], str]] = {}
        self._genz_failed = False

    def batched_ms(self, tokens: int) -> Optional[float]:
        """The GPU work recomputing ``tokens`` adds inside a busy batch: the prefill's FLOPs over
        the model's active parameters at ``PREFILL_MFU``. Unlike one request alone, it doesn't pay
        for
        streaming the weights (a MoE's experts), which the batch shares. None without device FLOPs.
        """
        if tokens <= 0 or not self._tflops:
            return None
        seconds = prefill_seconds(
            self._config,
            int(tokens),
            self._tflops * self._tp,
            attention_layers=self._attention_layers,
        )
        return seconds * 1e3 if seconds is not None else None

    def ms(self, tokens: int) -> Tuple[Optional[float], str]:
        """Return ``(milliseconds, source)``; ``None`` when nothing can estimate it."""
        tokens = int(tokens)
        if tokens <= 0:
            return 0.0, "nothing to recompute"
        if tokens not in self._cache:
            self._cache[tokens] = self._estimate(tokens)
        return self._cache[tokens]

    def _estimate(self, tokens: int) -> Tuple[Optional[float], str]:
        if self._fn is not None:
            value = self._fn(tokens)
            return (float(value), "given") if value is not None else (None, "not given")
        if self._model_id and self._hardware and not self._genz_failed:
            try:
                from ..performance_estimator import estimate_prefill_performance

                result = estimate_prefill_performance(
                    model=self._model_id,
                    batch_size=1,
                    input_tokens=tokens,
                    system_name=self._hardware,
                    bits=self._bits,
                    tensor_parallel=self._tp,
                )
                latency = result.get("Latency")
                if latency and latency > 0:
                    return float(latency), "GenZ"
            except Exception:  # noqa: BLE001 - any GenZ failure falls back to the formula
                self._genz_failed = True
        seconds = prefill_seconds(
            self._config,
            tokens,
            (self._tflops or 0) * self._tp if self._tflops else None,
            attention_layers=self._attention_layers,
        )
        if seconds is None:
            return None, "unknown (no GenZ model and no device FLOPs)"
        return seconds * 1e3, f"FLOPs formula at {PREFILL_MFU:.0%} MFU"


@dataclass(frozen=True)
class TierLink:
    """How a tier moves bytes: its effective bandwidth, fixed cost per load, and scope."""

    tier: str
    gbps: float  # effective GB/s for this model's loads
    floor_ms: float
    scope: str  # rank: per GPU (T1); total: the node's ranks share it (T2, T3)
    source: str
    efficiency: float = 1.0  # effective = measured x efficiency (connector or copy overhead)


@dataclass(frozen=True)
class BreakEven:
    """One tier's recompute-vs-reload verdict for the typical prefix."""

    tier: str
    cached_tokens: int
    bytes: float
    reload_ms: (
        float  # under the deployment's load (the TTFT a hit sees); inf when the link saturates
    )
    reload_idle_ms: float  # one load alone: what the break-even compares
    recompute_ms: Optional[float]
    recompute_source: str
    utilisation: float
    keep: Optional[bool]  # None: recompute unknown, so not checked
    # The measured bandwidth (the unit the health gate reports, before ``link.efficiency``) that
    # would pass at this load; None if no bandwidth would (the floor alone is over budget).
    needed_gbps: Optional[float]
    link: TierLink

    def describe(self) -> str:
        if self.utilisation >= MAX_LINK_UTILISATION:
            load = (
                f"; the link would be {self.utilisation:.0%} busy"
                f" (limit {MAX_LINK_UTILISATION:.0%})"
            )
        elif self.utilisation > 0.005:
            load = f"; {self.reload_ms:.0f} ms under load, {self.utilisation:.0%} busy"
        else:
            load = ""
        recompute = (
            f"{self.recompute_ms:.0f} ms ({self.recompute_source})"
            if self.recompute_ms is not None
            else self.recompute_source
        )
        return (
            f"{self.cached_tokens}-token prefix ({self.bytes / 1e6:.0f} MB) reloads in"
            f" {self.reload_idle_ms:.0f} ms at {self.link.gbps:g} GB/s ({self.link.source}){load}"
            f" vs recompute {recompute}"
        )


def break_even(
    *,
    tier: str,
    cached_tokens: int,
    nbytes: float,
    link: TierLink,
    recompute_ms: Optional[float],
    recompute_source: str,
    load_rate_per_s: float = 0.0,
    write_bytes_per_s: float = 0.0,
    staged_ms: float = 0.0,
) -> BreakEven:
    """Decide one tier for one prefix.

    ``load_rate_per_s`` is how many loads per second this tier serves on the link in question (the
    deployment's request rate times the share of requests served from the tier).
    ``write_bytes_per_s`` is the write-through traffic sharing the same link: every request writes
    the blocks it computed to a write-through tier (T2, the pool). Zero for T1, whose writes go the
    other way over PCIe.
    ``staged_ms`` is a second hop after this link: vLLM's tiering promotes a request's chunks into
    T1 first and loads them to the GPU only once all have arrived, so a disk or shared-fs hit also
    pays one T1 load (TCS, Qwen3-8B: 193 ms for a 4K disk hit, 55 ms of it the T1 load).
    """
    link_s = link.floor_ms / 1e3 + nbytes / (link.gbps * GB) if link.gbps > 0 else math.inf
    service_s = link_s + staged_ms / 1e3
    if link.gbps > 0:
        demand = (load_rate_per_s * nbytes + max(write_bytes_per_s, 0.0)) / (link.gbps * GB)
    else:
        demand = math.inf
    utilisation = max(0.0, demand)
    reload_s = link_s / (1.0 - utilisation) + staged_ms / 1e3 if utilisation < 1.0 else math.inf
    keep: Optional[bool] = None
    needed: Optional[float] = None
    if recompute_ms is not None:
        budget_s = BREAK_EVEN_RATIO * recompute_ms / 1e3
        keep = service_s < budget_s and utilisation < MAX_LINK_UTILISATION
        # floor + B/bw < T  and  (lam B + W)/bw < MAX_LINK_UTILISATION
        spare = budget_s - link.floor_ms / 1e3 - staged_ms / 1e3
        if spare > 0:
            traffic = load_rate_per_s * nbytes + max(write_bytes_per_s, 0.0)
            effective = max(nbytes / spare, traffic / MAX_LINK_UTILISATION)
            needed = effective / GB / max(link.efficiency, 1e-9)
    return BreakEven(
        tier=tier,
        cached_tokens=cached_tokens,
        bytes=nbytes,
        reload_ms=reload_s * 1e3 if math.isfinite(reload_s) else math.inf,
        reload_idle_ms=service_s * 1e3,
        recompute_ms=recompute_ms,
        recompute_source=recompute_source,
        utilisation=utilisation,
        keep=keep,
        needed_gbps=needed,
        link=link,
    )


def short_prefix(prefix_tokens: int, block_size: int) -> str:
    """Why a tier stores nothing: the connector moves whole blocks, and the prefix is under one."""
    return (
        f"nothing to store: the {int(prefix_tokens)}-token reusable prefix is shorter than one"
        f" {int(block_size)}-token block, and the connector stores whole blocks only"
    )


def connector_efficiency(segment_bytes: float, table: Any) -> float:
    """Share of the pool's measured rate vLLM's connector reaches for this segment size: the table's
    points, interpolated on log(segment bytes) and clamped at both ends."""
    points = sorted(table)
    if not points:
        return 1.0
    if segment_bytes <= points[0][0]:
        return points[0][1]
    if segment_bytes >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= segment_bytes <= x1:
            t = (math.log(segment_bytes) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return y0 + t * (y1 - y0)
    return points[-1][1]
