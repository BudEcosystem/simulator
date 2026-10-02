"""Prefill/decode disaggregation (FR-PD-*): proposed only when every gate passes and GenZ's
``DisaggregationAnalyzer`` favours it."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..parameter_counter import UniversalParameterCounter
from . import constants as c
from .context import PlanContext


def plan_pd(ctx: PlanContext, *, hardware_name: Optional[str] = None) -> Dict[str, Any]:
    """Return the ``pd`` block: ``{"enabled": False, ...}`` with the reason, or the split."""
    decisions = ctx.plan.decisions
    off: Dict[str, Any] = {"enabled": False, "role": None}

    def no(reason: str) -> Dict[str, Any]:
        decisions.append(f"P/D: no ({reason})")
        return off

    if c.PD_FEATURE not in ctx.features:
        return no(f"the engine doesn't list '{c.PD_FEATURE}'")
    try:
        params = UniversalParameterCounter().count_parameters(dict(ctx.model_config))
    except Exception:  # noqa: BLE001
        params = 0
    if params < c.PD_MIN_PARAMS:
        return no(f"{params / 1e9:.0f}B parameters, under {c.PD_MIN_PARAMS / 1e9:.0f}B")
    if ctx.workload.input_tokens <= c.PD_MIN_INPUT_TOKENS:
        return no(f"inputs of {ctx.workload.input_tokens} tokens, not over {c.PD_MIN_INPUT_TOKENS}")
    rate = ctx.request_rate_per_replica() * ctx.replicas
    if rate < c.PD_MIN_REQUEST_RATE:
        return no(f"{rate:.1f} req/s, under {c.PD_MIN_REQUEST_RATE:g}")
    if not (ctx.node.rdma_nics and ctx.node.gpudirect()):
        return no("no verified RDMA with GPUDirect on this node")
    if not ctx.model_id or not hardware_name:
        return no("GenZ's disaggregation analysis needs the model id and the GPU's GenZ name")
    try:
        from ..genz.serving.disaggregation import DisaggregationAnalyzer

        link_gbps = (ctx.node.rdma_link_gbps or 0.0) / 8.0  # Gbit/s -> GB/s
        result = DisaggregationAnalyzer(
            ctx.model_id, hardware_name
        ).compare_colocated_vs_disaggregated(
            total_instances=max(ctx.replicas, 2),
            input_tokens=ctx.workload.input_tokens,
            output_tokens=max(ctx.workload.output_tokens, 1),
            kv_transfer_bw_gbps=link_gbps,
        )
    except Exception as exc:  # noqa: BLE001 - an analysis failure means no P/D, never a failed plan
        return no(f"GenZ's disaggregation analysis failed ({type(exc).__name__})")
    speedup = float(result.get("speedup") or 0.0)
    if speedup <= 1.0:
        return no(f"GenZ predicts {speedup:.2f}x of colocated throughput")
    split = result.get("disaggregated") or {}
    decisions.append(f"P/D: yes, GenZ predicts {speedup:.2f}x colocated throughput")
    return {
        "enabled": True,
        "role": None,
        "prefill_instances": split.get("prefill_instances"),
        "decode_instances": split.get("decode_instances"),
    }
