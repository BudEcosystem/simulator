"""Engine container and pod modules (FRD-025 §6.3): engine.base, parallelism, probes, storage.model, the two
sidecars, placement, scheduler and autoscaling. v1 reproduces budcluster's chart and handler output for vLLM on CUDA.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Tuple

from ..context import Fragment, PlanContext
from .base import Module, register

MODEL_STORE_MOUNT = "/data/models-registry"
UNKNOWN_GPU = "Unknown NVIDIA GPU"
SIDECAR_PORT = 9090
AIBRIX_PORT = 8080


def k8s_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text or "").lower()).strip("-")


def group(ctx: PlanContext) -> Dict[str, Any]:
    return ctx.placement.group if ctx.placement else {}


@register
class EngineBase(Module):
    name = "engine.base"
    version = "v1"
    after = ()

    def applies(self, ctx: PlanContext) -> Tuple[bool, str]:
        if ctx.record.engine != "vllm":
            return False, f"engine {ctx.record.engine} renders through the legacy path"
        return True, ""

    def plan(self, ctx: PlanContext) -> Fragment:
        f = self.frag()
        runtime = ctx.record.runtime or {}
        f.engine["image"] = group(ctx).get("image") or ctx.record.images.get("cuda")
        f.engine["pull_policy"] = "Always"  # the chart's imagePullPolicy today
        f.engine["command"] = list((runtime.get("launch") or {}).get("single_node") or [])
        f.engine["trailing_args"] = list((runtime.get("args") or {}).get("trailing") or [])
        variant = ctx.draft.stage1.get("weights.precision", {}).get("path")
        if variant:
            f.arg("model", variant, self.name, "weights.precision chose a checkpoint variant")
        f.arg("trust-remote-code", True, self.name, "every vLLM pod loads model code (legacy rule)")
        for key, value in sorted((runtime.get("env_defaults") or {}).items()):
            f.envvar(key, value, self.name, "record env defaults")
        f.envvar("VLLM_TARGET_DEVICE", ctx.device.device_class, self.name, "device class")
        serve = ((runtime.get("ports") or {}).get("serve") or {}).get("container_port", 8000)
        f.ports.append({"name": "serve", "port": int(serve), "protocol": "TCP"})
        f.podfield("labels.engine_type", ctx.record.engine, self.name)
        f.podfield(
            "labels.concurrency",
            str(ctx.placement.concurrency if ctx.placement else ctx.intent.concurrency),
            self.name,
            "per-replica concurrency the plan was sized for",
        )
        name = group(ctx).get("name") or group(ctx).get("device_name") or ctx.device.sku
        f.podfield("labels.device_name", k8s_label(name), self.name)
        f.mounts.append({"volume": "shm", "path": "/dev/shm", "read_only": False})
        return f


@register
class LegacyEngineArgs(Module):
    """budsim's engine_ops arguments for the node group (block size, chunked prefill, prefix caching...), carried
    as they were chosen (FRD-025 M2: ported, not re-derived)."""

    name = "engine.legacy_args"
    version = "v1"
    after = ("engine.base",)
    # Renderer bindings ($(BUD_*)) and the keys a capacity module owns; budsim cannot know the namespace.
    SKIP = {
        "model",
        "served-model-name",
        "tensor-parallel-size",
        "pipeline-parallel-size",
        "max-num-seqs",
        "max-model-len",
        "max-num-batched-tokens",
        "gpu-memory-utilization",
        "disable-custom-all-reduce",
    }

    def plan(self, ctx: PlanContext) -> Fragment:
        f = self.frag()
        for key, value in (group(ctx).get("args") or {}).items():
            if key in self.SKIP or value is None:
                continue
            if key == "enable-prefix-caching":
                if value and ctx.draft.setting("prefix_cache.enabled") is None:
                    f.set("prefix_cache.enabled", True, self.name, "budsim engine_ops")
                continue  # False is dropped on cuda (not negatable there); a numerics decision wins
            f.arg(key, value, self.name, "budsim engine_ops")
        return f


@register
class Parallelism(Module):
    name = "parallelism"
    version = "v1"
    after = ("engine.base",)

    def plan(self, ctx: PlanContext) -> Fragment:
        f = self.frag()
        p = ctx.placement
        tp, pp = (p.tp, p.pp) if p else (1, 1)
        f.set("parallel.tensor", tp, self.name, "search result")
        if "pipeline-parallel-size" in (group(ctx).get("args") or {}) or pp > 1:
            f.set("parallel.pipeline", pp, self.name, "search result")
        if ctx.device.device_class == "cuda" and tp > 1:
            f.arg(
                "disable-custom-all-reduce",
                True,
                self.name,
                "TP > 1 on CUDA (custom all-reduce needs P2P)",
            )
            if (p.hardware_mode if p else "dedicated") != "shared":
                f.envvar("CUDA_DISABLE_CONTROL", "true", self.name, "dedicated TP > 1")
        f.resource("requests", "nvidia.com/gpu", tp, self.name, "one device per TP rank")
        f.resource("limits", "nvidia.com/gpu", tp, self.name, "one device per TP rank")
        if p:
            f.checks.append(
                {"kind": "device_count", "count": p.replicas * tp * pp, "advisory": True}
            )
        return f


@register
class Probes(Module):
    name = "probes"
    version = "v1"
    after = ("engine.base",)

    def plan(self, ctx: PlanContext) -> Fragment:
        f = self.frag()
        probes = (ctx.record.runtime or {}).get("probes") or {}
        health = (probes.get("health") or {}).get("path", "/health")
        st, rd, lv = (
            probes.get("startup") or {},
            probes.get("readiness") or {},
            probes.get("liveness") or {},
        )
        f.probes["liveness"] = {
            "http": health,
            "port": "serve",
            "initial_delay_s": lv.get("initial_delay_s", 120),
            "period_s": lv.get("period_s", 30),
            "failure_threshold": lv.get("failure_threshold", 20),
        }
        f.probes["readiness"] = {
            "http": health,
            "port": "serve",
            "initial_delay_s": rd.get("initial_delay_s", 60),
            "period_s": rd.get("period_s", 5),
        }
        f.probes["startup"] = {
            "http": health,
            "port": "serve",
            "initial_delay_s": st.get("initial_delay_s", 0),
            "period_s": st.get("period_s", 30),
            "failure_threshold": self.startup_threshold(ctx, st),
            "success_threshold": 1,
            "timeout_s": st.get("timeout_s", 1),
        }
        return f

    def startup_threshold(self, ctx: PlanContext, startup: Dict[str, Any]) -> int:
        return int(startup.get("failure_threshold", 200))


@register
class ProbesV2(Probes):
    """FR-ENG-7: the startup budget follows the predicted boot time (upper band, x3 margin), never below the record's."""

    version = "v2"

    def startup_threshold(self, ctx: PlanContext, startup: Dict[str, Any]) -> int:
        base = int(startup.get("failure_threshold", 200))
        settings = {k: v.value for k, v in ctx.draft.settings.items()}
        try:
            boot = ctx.predictor.boot_time(settings, ctx.model, ctx.device)
        except Exception:  # noqa: BLE001 - a prediction failure keeps the record's budget
            return base
        period = int(startup.get("period_s", 30)) or 30
        return max(base, int(math.ceil(3.0 * boot.high / period)))


@register
class StorageModel(Module):
    name = "storage.model"
    version = "v1"
    after = ("engine.base",)

    def plan(self, ctx: PlanContext) -> Fragment:
        f = self.frag()
        f.volumes.append({"name": "model-registry", "kind": "model_store"})
        f.mounts.append({"volume": "model-registry", "path": MODEL_STORE_MOUNT, "read_only": False})
        return f


@register
class SidecarMetrics(Module):
    name = "sidecar.metrics"
    version = "v1"
    after = ("engine.base",)

    def plan(self, ctx: PlanContext) -> Fragment:
        f = self.frag()
        f.containers.append(
            {
                "role": "sidecar",
                "name": "bud-runtime-sidecar",
                "image": "{cluster.runtime_sidecar_image}",
                "env": {
                    "ENGINE_METRICS_URL": "http://localhost:$(BUD_SERVE_PORT)/metrics",
                    "LISTEN_PORT": str(SIDECAR_PORT),
                    "SCRAPE_INTERVAL_SECONDS": "1",
                    "CALCULATION_WINDOW_SECONDS": "30",
                    "ENGINE_TYPE": ctx.record.engine,
                },
                "ports": [{"name": "sidecar-metrics", "port": SIDECAR_PORT, "protocol": "TCP"}],
                "probes": {
                    "liveness": {
                        "http": "/metrics",
                        "port": SIDECAR_PORT,
                        "initial_delay_s": 120,
                        "period_s": 30,
                    },
                    "readiness": {
                        "http": "/metrics",
                        "port": SIDECAR_PORT,
                        "initial_delay_s": 60,
                        "period_s": 10,
                    },
                },
            }
        )
        return f


@register
class SidecarAibrix(Module):
    name = "sidecar.aibrix"
    version = "v1"
    after = ("engine.base",)

    def plan(self, ctx: PlanContext) -> Fragment:
        f = self.frag()
        f.containers.append(
            {
                "role": "sidecar",
                "name": "aibrix-runtime",
                "image": "aibrix/runtime:v0.3.0",
                "command": ["aibrix_runtime", "--port", str(AIBRIX_PORT)],
                "env": {
                    "INFERENCE_ENGINE": "vllm",
                    "INFERENCE_ENGINE_ENDPOINT": "http://localhost:$(BUD_SERVE_PORT)",
                },
                "ports": [{"name": "aibrix", "port": AIBRIX_PORT, "protocol": "TCP"}],
                "probes": {
                    "liveness": {
                        "http": "/healthz",
                        "port": AIBRIX_PORT,
                        "initial_delay_s": 3,
                        "period_s": 2,
                    },
                    "readiness": {
                        "http": "/ready",
                        "port": AIBRIX_PORT,
                        "initial_delay_s": 5,
                        "period_s": 10,
                    },
                },
            }
        )
        return f


@register
class Placement(Module):
    name = "placement"
    version = "v1"
    after = ("engine.base",)

    def plan(self, ctx: PlanContext) -> Fragment:
        f = self.frag()
        g = group(ctx)
        raw = str(g.get("raw_name") or "").strip()
        if ctx.device.device_class == "cuda":
            if raw and raw != UNKNOWN_GPU:
                f.podfield(
                    "node_selector.nvidia.com/gpu.product",
                    raw,
                    self.name,
                    "the GPU product budsim sized for",
                )
            else:
                f.podfield(
                    "node_selector.nvidia.com/gpu.present",
                    "true",
                    self.name,
                    "no product name reported",
                )
        f.podfield(
            "affinity",
            {"replicas_share_node": True},
            self.name,
            "every replica on one node (capacity per node)",
        )
        target = ctx.placement.target_node if ctx.placement else None
        if target:
            f.podfield(
                "pin_to_node", target, self.name, "budsim sized the replicas against this node"
            )
            f.checks.append({"kind": "pin_node", "node": target})
        return f


@register
class Scheduler(Module):
    name = "scheduler"
    version = "v1"
    after = ("engine.base",)

    def plan(self, ctx: PlanContext) -> Fragment:
        f = self.frag()
        if ctx.device.device_class == "cuda":
            f.podfield(
                "scheduler",
                "hami-scheduler",
                self.name,
                "cuda pods schedule through HAMi (chart rule)",
            )
            f.podfield("runtime_class", "nvidia", self.name, "cuda pods")
        return f


@register
class Autoscaling(Module):
    name = "autoscaling"
    version = "v1"
    after = ("engine.base",)

    def plan(self, ctx: PlanContext) -> Fragment:
        f = self.frag()
        cfg = ctx.intent.autoscaling
        f.mutable["_autoscaling"] = dict(cfg) if cfg else {"enabled": False}
        replicas = ctx.placement.replicas if ctx.placement else 1
        f.mutable["replicas"] = {
            "min": 1,
            "max": max(int((cfg or {}).get("maxReplicas") or replicas), replicas),
        }
        return f
