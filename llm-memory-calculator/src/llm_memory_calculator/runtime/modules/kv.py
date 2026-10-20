"""FRD-023 KV rendering as modules (FRD-025 §6.3 ``kv.*``): the T1 CPU tier, KV events, prompt-token details and the
KV dtype from the kv_plan. T2 (fs tier) is not ported: no engine lists ``tiering_fs`` (vLLM #49829), and a plan that
needs it renders through the legacy path (budsim emits no runtime spec for it)."""

from __future__ import annotations

from typing import Tuple

from ..context import Fragment, PlanContext
from . import legacy
from .base import Module, register
from .capacity import legacy_budget
from .pod import group

T1_CONNECTOR = "OffloadingConnector"
KV_EVENTS_PORT, KV_EVENTS_REPLAY_PORT = 5557, 5558
KV_EVENTS_CONFIG = {
    "enable_kv_cache_events": True,
    "publisher": "zmq",
    "endpoint": f"tcp://*:{KV_EVENTS_PORT}",
    "replay_endpoint": f"tcp://*:{KV_EVENTS_REPLAY_PORT}",
    "topic": "",
}
REGISTER_COMMAND = [
    "/bin/sh",
    "-c",
    'rm -f /dev/shm/vllm_offload_*.mmap; exec python3 -m vllm.entrypoints.openai.api_server "$@"',
    "vllm",
]


def serves_generation(ctx: PlanContext) -> bool:
    g = group(ctx)
    if g.get("is_embedding"):
        return False
    args = g.get("args") or {}
    return not (
        str(args.get("runner") or "").lower() in ("pooling",)
        or str(args.get("task") or "").lower()
        in ("embed", "embedding", "classify", "score", "reward")
    )


@register
class KVTiers(Module):
    name = "kv.tiers"
    version = "v1"
    after = ("spec_decode", "engine.base")

    def applies(self, ctx: PlanContext) -> Tuple[bool, str]:
        if not ctx.record.kv_features:
            return False, "the engine lists no KV features"
        return True, ""

    def plan(self, ctx: PlanContext) -> Fragment:
        f = self.frag()
        features = ctx.record.features
        f.arg(
            "enable-prompt-tokens-details",
            True,
            self.name,
            "per-request cached tokens (engine lists KV features)",
        )
        prefix_off = ctx.draft.setting("prefix_cache.enabled") is False
        b = legacy_budget(ctx) if ctx.device.device_class == "cuda" else None
        if b is not None and b.t1_bytes > 0 and not prefix_off:
            extra: dict = {"cpu_bytes_to_use": int(b.t1_bytes), "block_size": int(b.t1_block)}
            if b.t1_mode == "alloc":
                extra["host_memory"] = "alloc"
            f.arg(
                "kv-transfer-config",
                {
                    "kv_connector": T1_CONNECTOR,
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": extra,
                },
                self.name,
                f"T1 {b.t1_bytes // legacy.MIB} MiB, {b.t1_mode} mode",
            )
            if b.t1_mode == "register":
                f.engine["command"] = list(REGISTER_COMMAND)
        elif b is not None and b.t1_bytes > 0 and prefix_off:
            f.decisions.append(
                "T1 not rendered: prefix caching is off (speculative decoding), so no reuse to tier"
            )
        if "kv_events" in features and serves_generation(ctx):
            f.arg(
                "kv-events-config",
                dict(KV_EVENTS_CONFIG),
                self.name,
                "KV cache events (FRD-023 Phase 0)",
            )
            f.ports.extend(
                [
                    {"name": "kv-events", "port": KV_EVENTS_PORT, "protocol": "TCP"},
                    {"name": "kv-replay", "port": KV_EVENTS_REPLAY_PORT, "protocol": "TCP"},
                ]
            )
            f.podfield("labels.model.aibrix.ai/kv-events-enabled", "true", self.name)
        return f


@register
class KVDtype(Module):
    """v1: the kv_plan's dtype (FRD-023 FR-QUANT-2), rendered at the capacity stage as budcluster does today."""

    name = "kv.dtype"
    version = "v1"
    after = ("engine.base",)

    def applies(self, ctx: PlanContext) -> Tuple[bool, str]:
        if "kv_fp8" not in ctx.record.features or not ctx.kv_plan:
            return False, "no kv_plan dtype, or the engine doesn't list kv_fp8"
        return True, ""

    def plan(self, ctx: PlanContext) -> Fragment:
        f = self.frag()
        dtype, skip = legacy.plan_kv_dtype(
            {"kv_features": sorted(ctx.record.features)}, ctx.kv_plan
        )
        if dtype:
            f.set("kv.dtype", dtype, self.name, "the kv_plan (FRD-023)")
            if skip:
                f.arg(
                    "kv-cache-dtype-skip-layers",
                    skip,
                    self.name,
                    "sliding-window layers keep the model dtype",
                )
        return f
