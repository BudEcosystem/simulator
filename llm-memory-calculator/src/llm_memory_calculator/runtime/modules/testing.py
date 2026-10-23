"""Test-only fault injection (TEST_CASES I-RENDER-01): force rung 1 of a plan to carry a setting the predictor would
never choose (e.g. ``memory.device_fraction = 0.99``) so the renderer's ``boot_failed`` ladder can be exercised on a
real cluster. Off unless selected (``versions["test.fault_inject"] = "on"``) and given ``ctx.cluster["fault_inject"]``;
it runs before ``engine.fallback``, so every lower rung is computed against the faulty plan and corrects it.
"""

from __future__ import annotations

from typing import Tuple

from ..context import Fragment, PlanContext
from .base import Module, register


@register
class FaultInject(Module):
    name = "test.fault_inject"
    version = "on"
    after = ("context.batching", "memory.gpu", "lora")
    overrides = ("memory.gpu", "context.batching", "lora")

    def applies(self, ctx: PlanContext) -> Tuple[bool, str]:
        return (bool(ctx.cluster.get("fault_inject")), "")

    def plan(self, ctx: PlanContext) -> Fragment:
        f = self.frag()
        for name, value in sorted(dict(ctx.cluster["fault_inject"]).items()):
            f.set(name, value, self.name, "FAULT INJECTED FOR TESTING (I-RENDER-01)")
        f.decisions.append(
            f"fault injected into rung 1 for testing: {dict(ctx.cluster['fault_inject'])}"
        )
        return f
