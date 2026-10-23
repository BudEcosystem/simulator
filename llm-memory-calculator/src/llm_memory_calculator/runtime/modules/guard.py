"""engine.guard (FRD-027 §10.9) and the prediction report: no module, override or legacy path emits a flag the study
measured harmful, and the flags measured inert stay at the engine default; the spec records the predicted
performance of what it renders (FRD-025 §7.1 ``predicted``)."""

from __future__ import annotations

from typing import Any, Dict

from ..compose import Composition, CompositionError, resolve_settings
from ..context import Fragment, PlanContext
from .base import Module, register

NEVER = {
    "enforce-eager": "eager mode measured 0.18-0.52x",
    "no-async-scheduling": "async scheduling off measured 0.82-0.92x",
}
DEFAULT_ROWS = (
    "performance-mode",
    "compilation-config",
    "O",
    "optimization-level",
    "attention-backend",
    "moe-backend",
    "gdn-prefill-backend",
    "stream-interval",
    "api-server-count",
    "long-prefill-token-threshold",
    "mamba-cache-mode",
)
RELIED_ON = (
    "engine defaults relied on: attention backend auto, MoE backend auto, CUDA graphs on, async scheduling on, "
    "performance mode default, one API server"
)


@register
class EngineGuard(Module):
    name = "engine.guard"
    version = "v2"
    after = (
        "lora",
        "context.batching",
        "memory.gpu",
        "memory.host",
        "kv.tiers",
        "parsers.tool_calling",
        "test.fault_inject",
        "parsers.reasoning",
        "engine.base",
        "engine.legacy_args",
        "parallelism",
        "probes",
        "placement",
        "scheduler",
    )

    def plan(self, ctx: PlanContext) -> Fragment:
        f = self.frag()
        f.decisions.append(RELIED_ON)
        text_only = ctx.intent.text_only
        if (
            text_only
            and "multimodal" in ctx.model.attributes
            and ctx.record.maps("mm.text_only")
            and "language_model_only" in ctx.record.features
        ):
            f.set(
                "mm.text_only",
                True,
                self.name,
                "multimodal checkpoint served for text only: skip the vision tower",
            )
        return f

    def post_compose(self, ctx: PlanContext, comp: Composition) -> None:
        args, _ = resolve_settings(comp.settings, ctx.record)
        merged = {**{k: v for k, v in comp.args.items()}, **args}
        for key, why in NEVER.items():
            if key in merged:
                raise CompositionError(
                    f"engine.guard: {merged[key].module} set --{key}, which is never rendered ({why})"
                )
        for key in DEFAULT_ROWS:
            if key in merged:
                raise CompositionError(
                    f"engine.guard: {merged[key].module} set --{key}; it stays at the engine default (measured inert or worse)"
                )


@register
class PredictedReport(Module):
    name = "plan.predicted"
    version = "v2"
    after = ("engine.guard",)

    def post_compose(self, ctx: PlanContext, comp: Composition) -> None:
        settings = {k: v.value for k, v in comp.settings.items()}
        c = int(ctx.placement.concurrency if ctx.placement else ctx.intent.concurrency or 1)
        draft = (
            ctx.drafts.get(str(settings.get("spec.draft") or ""))
            if settings.get("spec.draft")
            else None
        )
        try:
            pred = ctx.predictor.predict(
                settings, ctx.workload(c), ctx.model, ctx.device, draft=draft
            )
            boot = ctx.predictor.boot_time(settings, ctx.model, ctx.device)
        except Exception as exc:  # noqa: BLE001 - the report never fails a plan
            comp.decisions.append(f"plan.predicted: not predicted ({exc})")
            return
        report: Dict[str, Any] = {
            "ttft_ms": [round(pred.ttft_ms.low, 1), round(pred.ttft_ms.high, 1)],
            "tpot_ms": [round(pred.tpot_ms.low, 2), round(pred.tpot_ms.high, 2)],
            "throughput_tok_s": [
                round(pred.throughput_tok_s.low, 1),
                round(pred.throughput_tok_s.high, 1),
            ],
            "boot_s": [round(boot.low, 1), round(boot.high, 1)],
            "point": {
                "ttft_ms": round(pred.ttft_ms.value, 1),
                "tpot_ms": round(pred.tpot_ms.value, 2),
                "throughput_tok_s": round(pred.throughput_tok_s.value, 1),
                "boot_s": round(boot.value, 1),
            },
            "levels": dict(pred.levels),
            "ungrounded": bool(pred.ungrounded),
            "concurrency": c,
            "notes": list(pred.notes),
        }
        penalty = ctx.cluster.get("_penalty_vs_dedicated")
        if penalty is not None:
            report["penalty_vs_dedicated"] = penalty
        comp.predicted = report

    def plan(self, ctx: PlanContext) -> Fragment:
        return self.frag()
