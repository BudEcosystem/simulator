"""engine.fallback (FRD-027 §10.10): the alternatives ladder every tuned plan carries.

1. the plan as chosen (the spec itself);
2. without speculative decoding (prefix caching restored);
3. the original weight format, KV and state dtype at the engine default;
4. ``context.max_tokens`` halved and ``batch.max_running`` at the record default;
5. stock-equivalent: record defaults for every setting the modules changed, utilization at the record default. Where
   the engine's own defaults cannot boot on the device (a HAMi slice is sized for the plan), the rung keeps the least
   it needs to: the record's batch default (profiling transient), then the largest context that fits.

Rungs the plan did not touch are omitted. Each rung is a merge patch on the engine arguments (``None`` removes an
argument) plus the rung's full settings, triggered by FRD-025's ``boot_failed`` check: the renderer observes the
engine's startup and renders the next rung; reaching rung 5 is reported as a prediction failure.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from ..compose import Composition, resolve_settings
from ..context import Fragment, PlanContext, Value
from .base import Module, register

SPEC = ("spec.method", "spec.tokens", "spec.draft")
NUMERICS = ("weights.format", "weights.online", "weights.variant", "kv.dtype", "state.dtype")
CAPACITY = (
    "batch.max_running",
    "batch.prefill_chunk_tokens",
    "context.max_tokens",
    "memory.device_fraction",
)
KEEP = (
    "parallel.tensor",
    "parallel.pipeline",
    "lora.max_adapters",
    "lora.max_rank",
    "mm.text_only",
)
BOOT_MARGIN = 3.0


@register
class EngineFallback(Module):
    name = "engine.fallback"
    version = "v2"
    after = ("engine.guard",)

    def plan(self, ctx: PlanContext) -> Fragment:
        return self.frag()

    def post_compose(self, ctx: PlanContext, comp: Composition) -> None:
        base = dict(comp.settings)
        rungs: List[Tuple[int, str, Dict[str, Value], bool]] = []
        current = dict(base)
        if any(k in current for k in SPEC):
            current = {k: v for k, v in current.items() if k not in SPEC}
            if (
                current.get("prefix_cache.enabled") is not None
                and current["prefix_cache.enabled"].module == "spec_decode"
            ):
                current.pop("prefix_cache.enabled")
            rungs.append(
                (2, "without speculative decoding (prefix caching restored)", dict(current), False)
            )
        if any(
            k in current and current[k].module in ("weights.precision", "kv.dtype", "state.dtype")
            for k in NUMERICS
        ):
            current = {
                k: v
                for k, v in current.items()
                if not (
                    k in NUMERICS and v.module in ("weights.precision", "kv.dtype", "state.dtype")
                )
            }
            rungs.append(
                (
                    3,
                    "original weight format; KV and state dtype at the engine default",
                    dict(current),
                    False,
                )
            )
        halved = dict(current)
        ctx_tokens = halved.get("context.max_tokens")
        if ctx_tokens is not None:
            need = int((ctx.intent.input_tokens or 0) + (ctx.intent.output_tokens or 0))
            value = max(
                int(ctx_tokens.value) // 2 // 16 * 16, min(int(ctx_tokens.value), max(need, 2048))
            )
            halved["context.max_tokens"] = Value(value, self.name, "rung 4: context halved")
        halved["batch.max_running"] = Value(
            ctx.record.planning_max_running(), self.name, "rung 4: record default batch"
        )
        if any(
            halved.get(k) != current.get(k) for k in ("context.max_tokens", "batch.max_running")
        ):
            rungs.append(
                (4, "max-model-len halved, max-num-seqs at the record default", halved, False)
            )
        stock = {k: v for k, v in base.items() if k in KEEP}
        stock["memory.device_fraction"] = Value(
            ctx.record.default_fraction(ctx.device), self.name, "rung 5: record default"
        )
        if not self._engine_default_boots(ctx, stock):
            stock["batch.max_running"] = Value(
                ctx.record.planning_max_running(),
                self.name,
                "rung 5: the engine's own default would not boot (profiling transient)",
            )
            lim = self._limits(ctx, stock)
            if lim is not None and not lim.boots and lim.max_bootable_context > 0:
                # on a slice one engine-default-length request may not fit (TCS 10-18: gpt-oss, 131,072 tokens on
                # 18,688 MiB); the pod was sized for the plan, so the cap that makes it fit is not a tuning choice
                stock["context.max_tokens"] = Value(
                    int(lim.max_bootable_context) // 16 * 16,
                    self.name,
                    "rung 5: the engine's default context does not fit the device; the largest that boots",
                )
        if stock != current:
            rungs.append(
                (5, "stock-equivalent: engine defaults for every tuned setting", stock, True)
            )
        base_args, _ = resolve_settings(base, ctx.record)
        for rung, text, settings, is_stock in rungs:
            args, _ = resolve_settings(settings, ctx.record)
            patch: Dict[str, Any] = {}
            for key in set(base_args) | set(args):
                old = base_args.get(key).value if key in base_args else None
                new = args.get(key).value if key in args else None
                if old != new:
                    patch[key] = new
            if "weights.variant" in base and "weights.variant" not in settings:
                patch["model"] = "$(BUD_MODEL_PATH)"
            values = {k: v.value for k, v in settings.items()}
            alt: Dict[str, Any] = {
                "when": {"check": "boot_failed"},
                "patch": {"containers.engine.args": dict(sorted(patch.items()))},
                "settings": values,
                "decision": f"engine.fallback: rung {rung}, {text}",
                "rung": rung,
                "stock_equivalent": is_stock,
                "predicted_margin_bytes": self._margin(ctx, values),
            }
            comp.alternatives.append(alt)
        base_values = {k: v.value for k, v in base.items()}
        budget = self._budget(ctx, base_values)
        check: Dict[str, Any] = {"kind": "boot_failed", "budget_s": budget, "rungs": len(rungs) + 1}
        margin = self._margin(ctx, base_values)
        if margin is not None:
            check["margin_bytes"] = margin  # rung 1's; each alternative carries its own
        comp.checks.append(check)
        comp.decisions.append(
            f"engine.fallback: {len(rungs)} fallback rung(s) "
            f"({', '.join(str(r[0]) for r in rungs) or 'none'}), boot budget {budget} s per rung"
        )

    def _engine_default_boots(self, ctx: PlanContext, stock: Dict[str, Value]) -> bool:
        lim = self._limits(ctx, stock)
        return bool(lim is not None and lim.boots)

    def _limits(self, ctx: PlanContext, stock: Dict[str, Value]) -> Any:
        settings = {k: v.value for k, v in stock.items()}
        try:
            return ctx.predictor.limits(settings, ctx.model, ctx.device)
        except Exception:  # noqa: BLE001
            return None

    def _margin(self, ctx: PlanContext, settings: Dict[str, Any]) -> Optional[int]:
        """The smallest predicted memory margin over the engine's boot checks for ``settings`` (bytes; < 0 = OOM)."""
        try:
            draft = (
                ctx.drafts.get(str(settings.get("spec.draft") or ""))
                if settings.get("spec.draft")
                else None
            )
            checks = ctx.predictor.limits(settings, ctx.model, ctx.device, draft=draft).checks
        except Exception:  # noqa: BLE001
            return None
        return int(min(c.margin_bytes for c in checks)) if checks else None

    def _budget(self, ctx: PlanContext, settings: Dict[str, Any]) -> int:
        try:
            boot = ctx.predictor.boot_time(settings, ctx.model, ctx.device)
            return int(math.ceil(max(boot.high * BOOT_MARGIN, 600)))
        except Exception:  # noqa: BLE001
            return 1800
