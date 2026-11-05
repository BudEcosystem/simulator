"""FRD-023 KV rendering as modules (FRD-025 §6.3 ``kv.*``): the T1 CPU tier, the T3 Mooncake store connector, KV
events, prompt-token details and the KV dtype from the kv_plan. T2 (fs tier) is not ported: no engine lists
``tiering_fs`` (vLLM #49829), and a plan that needs it renders through the legacy path (budsim emits no runtime spec
for it). T3 renders exactly as budcluster's legacy path does (``engine_processors._apply_kv_plan``)."""

from __future__ import annotations

import json

from typing import Tuple

from ..context import Fragment, PlanContext
from . import legacy
from .base import Module, register
from .capacity import legacy_budget
from .pod import group

T1_CONNECTOR = "OffloadingConnector"
T3_CONNECTOR = "MooncakeStoreConnector"
MULTI_CONNECTOR = "MultiConnector"
# Mooncake's TCP transport overflows its queues on a single 4K-token prefix with its defaults (spike S10).
MC_TCP_ENV = {
    "MC_TCP_LANES_PER_PEER": "16",
    "MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER": "65535",
    "MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER": "65535",
    "MC_TCP_ADMISSION_TIMEOUT_MS": "30000",
}
SERVER = 'exec python3 -m vllm.entrypoints.openai.api_server "$@"'
CLEAR_SHM = "rm -f /dev/shm/vllm_offload_*.mmap; "
WRITE_MOONCAKE_CONFIG = 'printf \'%s\' "$MOONCAKE_CONFIG_JSON" > "$MOONCAKE_CONFIG_PATH" && '
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
        t1 = None
        if b is not None and b.t1_bytes > 0 and not prefix_off:
            extra: dict = {"cpu_bytes_to_use": int(b.t1_bytes), "block_size": int(b.t1_block)}
            if b.t1_mode == "alloc":
                extra["host_memory"] = "alloc"
            t1 = {"kv_connector": T1_CONNECTOR, "kv_role": "kv_both", "kv_connector_extra_config": extra}
        elif b is not None and b.t1_bytes > 0 and prefix_off:
            f.decisions.append(
                "T1 not rendered: prefix caching is off (speculative decoding), so no reuse to tier"
            )
        t3 = b.t3 if b is not None and not prefix_off else None
        if b is not None and b.t3 and prefix_off:
            f.decisions.append("T3 not rendered: prefix caching is off (speculative decoding), so no reuse to tier")
        config = t1
        if t3:
            store = {
                "kv_connector": T3_CONNECTOR,
                "kv_role": "kv_both",
                "kv_connector_extra_config": {"load_async": True, "cache_prefix": t3["cache_prefix"]},
            }
            config = (
                {
                    "kv_connector": MULTI_CONNECTOR,
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {"connectors": [t1, store]},
                }
                if t1
                else dict(store)
            )
            # vLLM's default "fail" fails the request on a failed load (FR-FAIL-1)
            config["kv_load_failure_policy"] = "recompute"
        if config:
            parts = []
            if t1:
                parts.append(f"T1 {b.t1_bytes // legacy.MIB} MiB, {b.t1_mode} mode")
            if t3:
                parts.append(f"T3 Mooncake pool at {t3['master']} ({t3['protocol']}), keys under {t3['cache_prefix']}")
            f.arg("kv-transfer-config", config, self.name, "; ".join(parts))
        if t3:
            f.arg("prefix-caching-hash-algo", "sha256_cbor", self.name, "a cache shared across processes (FR-RENDER-2)")
            f.envvar("MOONCAKE_CONFIG_PATH", legacy.T3_CONFIG_PATH, self.name, "the store client reads its config here")
            f.envvar(
                "MOONCAKE_CONFIG_JSON",
                json.dumps(legacy.mooncake_config(t3), separators=(",", ":")),
                self.name,
                "standalone-store against the cluster pool",
            )
            if t3["protocol"] == "tcp":
                for key, value in MC_TCP_ENV.items():
                    f.envvar(key, value, self.name, "Mooncake TCP limits (FR-T3-4)")
            f.envvar("PYTHONHASHSEED", "0", self.name, "block hashes equal across processes")
            f.podfield("labels.bud-ai-foundry/kv-namespace", t3["cache_prefix"], self.name, "the pool admits labelled pods")
        if t1 and b.t1_mode == "register" or t3:
            script = (CLEAR_SHM if t1 and b.t1_mode == "register" else "") + (WRITE_MOONCAKE_CONFIG if t3 else "")
            f.engine["command"] = ["/bin/sh", "-c", script + SERVER, "vllm"]
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
