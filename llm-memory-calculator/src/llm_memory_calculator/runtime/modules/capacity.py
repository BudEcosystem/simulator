"""Capacity modules (FRD-027 §10.2, §10.3, §10.8; FRD-025 §6.3).

v1 is budcluster's rule ported (memory_plan, engine_processors); v2 is FRD-027's:
- ``context.batching`` v2: ``batch.max_running = min(ceiling, max(record default, planned x 1.25))``;
  ``context.max_tokens = min(model maximum, intent ceiling, largest bootable)``; the chunk budget is the record
  default, with {8192, 16384} evaluated when the planned input is >= 8K tokens;
- ``memory.gpu`` v2: dedicated -> the record's fraction for the device class (FRD-023's T0 headroom folds in);
  shared -> the plan-sized HAMi slice, now predicted, with the penalty against dedicated reported;
- ``lora`` v2: only when adapters are asked for or attached.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple

from ..context import Fragment, PlanContext
from . import legacy
from .base import Module, register
from .pod import group

GIB = 2**30
MIB = 2**20
MAX_RUNNING_HEADROOM = 1.25
#: Shared mode: the slice holds the planned load's tokens (concurrency x one request) with this much headroom.
SLICE_KV_HEADROOM = 1.10
#: HAMi slices are rounded up to this granularity (MiB).
SLICE_STEP_MIB = 256


def shared_slice_mib(
    predictor: Any,
    model: Any,
    device: Any,
    settings: Dict[str, Any],
    *,
    concurrency: int,
    request_tokens: int,
    floor_mib: int = 0,
    cap_mib: Optional[int] = None,
    draft: Any = None,
) -> Tuple[int, bool]:
    """The smallest HAMi slice (MiB, >= ``floor_mib``) on which ``settings`` boot and hold the planned load.

    FRD-027 §10.3: the slice stays plan-sized but is planned and predicted with the engine's own terms -- weights,
    the overhead the settings bring (activation peak of the prefill chunk, graphs, sampler) and KV for
    ``concurrency`` requests of ``request_tokens`` (+10%). ``cap_mib`` (the free memory, else the card) bounds it.
    Returns (slice_mib, fits): fits is False when even the cap cannot hold the load (it then queues).
    """
    from ...device import HardwareMode

    card_mib = int(device.memory_visible_bytes // MIB)
    cap = min(int(cap_mib), card_mib) if cap_mib else card_mib
    need = int(
        math.ceil(max(int(concurrency), 1) * max(int(request_tokens), 1) * SLICE_KV_HEADROOM)
    )
    probe = {**settings, "context.max_tokens": max(int(request_tokens), 16)}
    mib = min(max(int(floor_mib), SLICE_STEP_MIB), cap)
    fraction = max(float(settings.get("memory.device_fraction") or 0.9), 0.5)
    for _ in range(16):
        dev = device.with_mode(HardwareMode(kind="shared", slice_bytes=mib * MIB))
        lim = predictor.limits(probe, model, dev, draft=draft)
        # the pool must hold the load plus the memory band (FR-GND-6: how wrong a boot prediction has been here)
        band = float(lim.memory_band_bytes or 0.0)
        grouped = bool(
            lim.block_bytes
            and lim.block_tokens
            and (
                getattr(lim, "sliding_groups", 0)
                or lim.state_bytes_per_sequence
                or (lim.attn_groups or 1) > 1
            )
        )
        if grouped:
            # a block covers one group of layers, and a sequence holds blocks in every group (full, sliding up to
            # the window, recurrent state): count the load in blocks, not in one group's tokens
            need_blocks = math.ceil(
                max(int(concurrency), 1) * lim.blocks_per_sequence(max(int(request_tokens), 1)) * SLICE_KV_HEADROOM
            )
            need_bytes = need_blocks * float(lim.block_bytes)
            holds = lim.num_blocks >= need_blocks
        else:
            if lim.block_bytes and lim.block_tokens:
                per_token = lim.block_bytes / lim.block_tokens
            else:
                per_token = float(model.kv_bytes_per_token())
            need_bytes = need * per_token
            holds = lim.kv_tokens >= need
        if lim.boots and holds and float(lim.kv_pool_bytes) - band >= need_bytes:
            return mib, True
        # grow by the larger deficit: the KV pool short of the load (negative when the weights don't fit), or the
        # worst failing boot check (e.g. the CUDA-graph profiling transient)
        pool_short = need_bytes + band - float(lim.kv_pool_bytes)
        worst = max([-float(c.margin_bytes) for c in lim.failures] + [0.0])
        grow = max(pool_short, worst, SLICE_STEP_MIB * MIB) / fraction
        new = min(cap, int(math.ceil((mib * MIB + grow) / (SLICE_STEP_MIB * MIB))) * SLICE_STEP_MIB)
        if new <= mib:
            return mib, False
        mib = new
    return mib, False


LONG_INPUT_TOKENS = 8192
CHUNK_CANDIDATES = (8192, 16384)
#: Revision 2's search (FRD-027 FR-SEL-2): the simulation prices each chunk size, so every input length is searched
R2_CHUNK_CANDIDATES = (2048, 4096, 8192, 16384)


def _profiled(pred: Any) -> bool:
    return any(str(n).startswith("predictor: step profile") for n in getattr(pred, "notes", ()) or ())


def choose_chunk(
    predictor: Any,
    settings: Mapping[str, Any],
    workload: Any,
    model: Any,
    device: Any,
    default: int,
    *,
    intent: Any = None,
    draft: Any = None,
) -> Tuple[int, str]:
    """Return the prefill chunk (``max-num-batched-tokens``) for a configuration, and why.

    With the fitted predictor only long inputs are compared: the record default against 16,384, switching when the
    throughput band separates above (FR-GND-2), as before.

    On the step-profile path (FRD-027 Revision 2) the simulation prices every candidate in ``R2_CHUNK_CANDIDATES`` for
    any input (FR-SEL-2). A candidate qualifies when P(every target the intent sets holds) reaches the confidence
    (FR-SEL-1):
    - when the default qualifies, or no target is set, another chunk wins only if it qualifies and its throughput band
      separates above the default's;
    - when the default does not, the qualifying candidate with the highest throughput wins, the closest to the
      default on a tie;
    - an ungrounded prediction gives no probability, so the throughput rule alone decides (NFR-5).
    """
    import math

    def predict(chunk: int):
        return predictor.predict(
            {**settings, "batch.prefill_chunk_tokens": int(chunk)}, workload, model, device, draft=draft
        )

    try:
        base = predict(default)
    except Exception:  # noqa: BLE001
        return default, "record default for the device class"
    r2 = _profiled(base)
    if not r2 and float(workload.input_tokens or 0) < LONG_INPUT_TOKENS:
        return default, "record default for the device class"
    floor = int(settings.get("batch.max_running") or 1)  # vLLM refuses a budget below max-num-seqs
    preds = {int(default): base}
    for chunk in R2_CHUNK_CANDIDATES if r2 else CHUNK_CANDIDATES:
        if chunk in preds or chunk < floor:
            continue
        try:
            preds[chunk] = predict(chunk)
        except Exception:  # noqa: BLE001
            continue
    others = sorted(set(preds) - {int(default)})
    pool = dict(preds)
    targets = {}
    if r2 and intent is not None:
        targets = {
            "ttft_ms": getattr(intent, "target_ttft_ms", None),
            "e2e_s": getattr(intent, "target_e2e_s", None),
            "throughput_per_user": getattr(intent, "target_throughput_per_user", None),
        }
        targets = {k: float(v) for k, v in targets.items() if v}
    if targets:
        from ...perf.rank import slo_confidence, slo_probability

        conf = slo_confidence()
        p = {
            k: slo_probability(v.to_dict(), concurrency=int(workload.concurrency), **targets) for k, v in preds.items()
        }
        if all(v is not None for v in p.values()):
            ok = [k for k in preds if p[k] >= conf]
            if p[int(default)] < conf and ok:
                best = max(
                    ok, key=lambda k: (preds[k].throughput_tok_s.value, -abs(math.log2(k / max(default, 1))))
                )
                return best, (
                    f"prefill chunk {best}: meets the targets with P={p[best]:.2f} >= {conf:g} where {default} has "
                    f"P={p[int(default)]:.2f}"
                )
            pool = {k: v for k, v in preds.items() if k in ok or k == int(default)}
    best = int(default)
    for chunk, pred in sorted(pool.items()):
        if chunk != best and pred.throughput_tok_s.separates_above(pool[best].throughput_tok_s):
            best = chunk
    if best == int(default):
        return default, f"prefill chunk {default}: bands overlap with {others}, engine default kept"
    return best, f"prefill chunk {best}: predicted throughput band separates above {default}"


class PlanError(ValueError):
    """The plan cannot be built for this placement; the message is the reason budsim reports."""


def legacy_budget(ctx: PlanContext) -> legacy.CudaBudget:
    cached = ctx.cluster.get("_legacy_budget")
    if cached is not None:
        return cached
    g = dict(group(ctx))
    g.setdefault(
        "labels",
        {
            "concurrency": str(
                ctx.placement.concurrency if ctx.placement else ctx.intent.concurrency
            )
        },
    )
    g["concurrency"] = ctx.placement.concurrency if ctx.placement else ctx.intent.concurrency
    g.setdefault("type", ctx.device.device_class)
    g["kv_features"] = sorted(ctx.record.kv_features)
    intent = {
        "input_tokens": ctx.intent.input_tokens,
        "output_tokens": ctx.intent.output_tokens,
        "model_max_context_length": ctx.intent.model_max_context_length,
    }
    try:
        budget = legacy.cuda_budget(g, intent, ctx.kv_plan)
    except legacy.SizingError as exc:
        raise PlanError(str(exc)) from exc
    ctx.cluster["_legacy_budget"] = budget
    return budget


def current_settings(ctx: PlanContext) -> Dict[str, Any]:
    return {k: v.value for k, v in ctx.draft.settings.items()}


# ====================================================================== context.batching


@register
class ContextBatching(Module):
    name = "context.batching"
    version = "v1"
    after = ("memory.gpu",)

    def plan(self, ctx: PlanContext) -> Fragment:
        f = self.frag()
        b = legacy_budget(ctx)
        f.set("batch.max_running", b.max_num_seqs, self.name, "= per-replica concurrency (legacy)")
        f.set(
            "context.max_tokens",
            b.max_model_len,
            self.name,
            f"(input + output) x 1.1 ({b.context_source})",
        )
        f.set(
            "batch.prefill_chunk_tokens",
            b.max_num_batched_tokens,
            self.name,
            "max(min(context, 8192), seqs)",
        )
        return f


def transient_cap(
    predictor: Any, settings: Dict[str, Any], seqs: int, model: Any, device: Any, draft: Any = None
) -> Optional[int]:
    """Return the most sequences at or below ``seqs`` whose CUDA-graph profiling transient fits, or None when ``seqs``
    already does.

    The engine's overhead grows with the sequences captured, so a ceiling read at fewer sequences can overstate the one
    at more: each cap is re-checked at its own value.
    """
    capped = None
    for _ in range(6):
        lim = predictor.limits({**settings, "batch.max_running": int(seqs)}, model, device, draft=draft)
        if lim.transient_ceiling is None or lim.transient_ceiling >= seqs:
            return capped
        seqs = capped = max(int(lim.transient_ceiling), 1)
    return capped


@register
class ContextBatchingV2(Module):
    name = "context.batching"
    version = "v2"
    after = ("memory.gpu",)

    def plan(self, ctx: PlanContext) -> Fragment:
        f = self.frag()
        rec = ctx.record
        planned = int(ctx.placement.concurrency if ctx.placement else ctx.intent.concurrency)
        settings = current_settings(ctx)
        # ceiling and bootable context under everything decided so far (numerics + memory.gpu)
        probe = dict(settings)
        probe.setdefault("batch.max_running", rec.planning_max_running())
        probe["context.max_tokens"] = 4096
        lim = ctx.predictor.limits(probe, ctx.model, ctx.device, draft=self._draft(ctx, settings))
        ceiling = lim.sequence_ceiling if lim.sequence_ceiling else 10**6
        want = max(rec.planning_max_running(), int(math.ceil(planned * MAX_RUNNING_HEADROOM)))
        seqs = min(ceiling, want)
        reason = f"max(record default {rec.planning_max_running()}, planned {planned} x 1.25)"
        if seqs < want:
            reason += f", capped by the sequence ceiling {ceiling}"
        transient = transient_cap(ctx.predictor, probe, int(seqs), ctx.model, ctx.device, self._draft(ctx, settings))
        if transient is not None:
            # vLLM's CUDA-graph profiling allocates min(max_num_seqs, capture) hybrid blocks at startup; above this
            # the engine OOMs before serving (gtech 10-20: FP8 27B on a 45 GB L20)
            ceiling = seqs = transient
            reason += f", capped by the CUDA-graph profiling transient at {seqs}"
            f.decisions.append(
                f"max_num_seqs {seqs}: more sequences would not fit the CUDA-graph profiling transient at startup"
            )
        if ceiling < planned:
            f.decisions.append(
                f"the sequence ceiling {ceiling} is below the planned per-replica concurrency {planned}: "
                "the search must place more replicas"
            )
        f.set("batch.max_running", int(seqs), self.name, reason)
        # context: model maximum, the user's ceiling, what boots
        probe["batch.max_running"] = int(seqs)
        probe["context.max_tokens"] = 4096
        lim = ctx.predictor.limits(probe, ctx.model, ctx.device, draft=self._draft(ctx, settings))
        model_max = int(ctx.intent.model_max_context_length or ctx.model.max_context)
        candidates = {"model maximum": model_max}
        if ctx.intent.max_context_length:
            candidates["requested maximum"] = int(ctx.intent.max_context_length)
        if lim.max_bootable_context:
            bootable = int(lim.max_bootable_context)
            band = float(lim.memory_band_bytes or 0.0)
            if band > 0 and lim.block_bytes and lim.block_tokens:
                # FR-GND-6: leave the memory band, so a pool that much smaller than predicted still boots
                reserve = int(math.ceil(band / (lim.block_bytes / lim.block_tokens)))
                candidates["largest bootable less the memory band"] = (
                    max(bootable - reserve, 16) // 16 * 16
                )
            else:
                candidates["largest bootable"] = bootable
        source, ctx_tokens = min(candidates.items(), key=lambda kv: kv[1])
        if source.startswith("largest bootable") and settings.get("spec.method"):
            source = (
                f"largest bootable with {settings['spec.method']} (shrunk for speculative decoding)"
            )
        refusals = [
            r
            for r in ctx.predictor.validate(
                {**settings, "context.max_tokens": ctx_tokens}, ctx.model
            )
            if r.setting == "context.max_tokens"
        ]
        if refusals:
            reason = refusals[0].reason
            ctx_tokens = (ctx_tokens - 1) // 16 * 16
            while ctx_tokens > 4096 and [
                r
                for r in ctx.predictor.validate(
                    {**settings, "context.max_tokens": ctx_tokens}, ctx.model
                )
                if r.setting == "context.max_tokens"
            ]:
                ctx_tokens = int(ctx_tokens * 0.95) // 16 * 16
            source = f"below the record's conflict ({reason})"
        need = int(
            (ctx.intent.input_tokens_max or ctx.intent.input_tokens or 0)
            + (ctx.intent.output_tokens_max or ctx.intent.output_tokens or 0)
        )
        if ctx_tokens < need:
            raise PlanError(
                f"the largest bootable context ({ctx_tokens}) is below one request of the workload ({need} tokens)"
            )
        f.set("context.max_tokens", int(ctx_tokens), self.name, source)
        # chunked prefill budget: the record default; evaluated against 16384 for long inputs, and over every
        # candidate on the step-profile path (FRD-027 FR-SEL-2)
        default = rec.default_chunk(ctx.device)
        chunk, why = choose_chunk(
            ctx.predictor,
            {**settings, "batch.max_running": int(seqs), "context.max_tokens": int(ctx_tokens)},
            ctx.workload(),
            ctx.model,
            ctx.device,
            default,
            intent=ctx.intent,
            draft=self._draft(ctx, settings),
        )
        if why.startswith("prefill chunk"):
            f.decisions.append(why)
        f.set(
            "batch.prefill_chunk_tokens",
            int(chunk),
            self.name,
            "record default for the device class" if chunk == default else why,
        )
        return f

    def _draft(self, ctx: PlanContext, settings: Dict[str, Any]):
        return (
            ctx.drafts.get(str(settings.get("spec.draft") or ""))
            if settings.get("spec.draft")
            else None
        )

# ====================================================================== memory.gpu


@register
class MemoryGpu(Module):
    name = "memory.gpu"
    version = "v1"
    after = (
        "engine.base",
        "parallelism",
        "kv.dtype",
        "spec_decode",
        "state.dtype",
        "weights.precision",
    )

    def applies(self, ctx: PlanContext) -> Tuple[bool, str]:
        if ctx.device.device_class != "cuda":
            return False, "not a cuda device"
        return True, ""

    def plan(self, ctx: PlanContext) -> Fragment:
        f = self.frag()
        b = legacy_budget(ctx)
        shared = (ctx.placement.hardware_mode if ctx.placement else "dedicated") == "shared"
        f.set(
            "memory.device_fraction",
            b.gpu_memory_utilization,
            self.name,
            "fraction of the slice" if shared else "demand / card, or the T0 headroom",
        )
        if shared:
            self._shared(
                ctx,
                f,
                b.gpu_memory_mib,
                b.gpu_memory_utilization,
                b.slice_fallback_mib,
                b.slice_fallback_util,
            )
        return f

    def _shared(
        self,
        ctx: PlanContext,
        f: Fragment,
        slice_mib: int,
        util: float,
        fallback_mib: Optional[int],
        fallback_util: Optional[float],
    ) -> None:
        tp = ctx.placement.tp if ctx.placement else 1
        f.resource("requests", "nvidia.com/gpumem", int(slice_mib), self.name, "HAMi slice")
        f.resource("limits", "nvidia.com/gpumem", int(slice_mib), self.name, "HAMi slice")
        f.envvar(
            "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS",
            "1",
            self.name,
            "shared cuda vLLM: graphs inside the slice",
        )
        pods = 1 if not ctx.placement else max(1, int(ctx.placement.replicas))
        f.checks.append(
            {"kind": "hami_slice", "slice_mib": int(slice_mib), "gpus": int(tp), "pods": pods}
        )
        if fallback_mib:
            f.alternatives.append(
                {
                    "when": {"check": "hami_slice", "failed": True},
                    "patch": {
                        "containers.engine.args": {"gpu-memory-utilization": fallback_util},
                        "containers.engine.resources.requests": {
                            "nvidia.com/gpumem": int(fallback_mib)
                        },
                        "containers.engine.resources.limits": {
                            "nvidia.com/gpumem": int(fallback_mib)
                        },
                    },
                    "settings": {"memory.device_fraction": fallback_util},
                    "decision": "memory.gpu: the grown slice didn't fit on live data; demand-sized slice",
                }
            )


@register
class MemoryGpuV2(MemoryGpu):
    version = "v2"

    def plan(self, ctx: PlanContext) -> Fragment:
        f = self.frag()
        shared = (ctx.placement.hardware_mode if ctx.placement else "dedicated") == "shared"
        rec = ctx.record
        if shared:
            from ...device import HardwareMode

            b = legacy_budget(ctx)  # FRD-023's plan-sized slice: the floor, and its fraction
            util = b.gpu_memory_utilization
            slice_mib, why = self._plan_sized_slice(ctx, util, int(b.gpu_memory_mib))
            f.set(
                "memory.device_fraction",
                util,
                self.name,
                f"fraction of the {slice_mib} MiB HAMi slice",
            )
            self._shared(ctx, f, slice_mib, util, b.slice_fallback_mib, b.slice_fallback_util)
            f.decisions.append(why)
            # FRD-027 §10.3: what is planned after this -- context, batching, the prediction, the fallback's margins --
            # is planned on the slice the pod gets, not on the card
            ctx.device = ctx.device.with_mode(
                HardwareMode(kind="shared", slice_bytes=slice_mib * MIB)
            )
            f.decisions.append(self._penalty(ctx, slice_mib, util))
            return f
        fraction = rec.default_fraction(ctx.device)
        reason = f"the record's fraction for {ctx.device.device_class} (not demand / card)"
        plan_target = (
            (ctx.kv_plan or {}).get("gpu_memory_utilization") if ctx.record.kv_features else None
        )
        if plan_target and float(plan_target) > fraction:
            fraction = min(float(plan_target), rec.max_fraction(ctx.device))
            reason = "FRD-023 T0 headroom (kv_plan), within the record's ceiling"
        f.set("memory.device_fraction", round(float(fraction), 2), self.name, reason)
        return f

    def _plan_sized_slice(self, ctx: PlanContext, util: float, floor_mib: int) -> Tuple[int, str]:
        rec = ctx.record
        conc = int(ctx.placement.concurrency if ctx.placement else ctx.intent.concurrency)
        request = int(
            (ctx.intent.input_tokens_max or ctx.intent.input_tokens or 0)
            + (ctx.intent.output_tokens_max or ctx.intent.output_tokens or 0)
        )
        settings = {
            **current_settings(ctx),
            "memory.device_fraction": util,
            # what context.batching v2 sets after this module
            "batch.max_running": max(
                rec.planning_max_running(), int(math.ceil(conc * MAX_RUNNING_HEADROOM))
            ),
            "batch.prefill_chunk_tokens": rec.default_chunk(ctx.device),
        }
        draft = (
            ctx.drafts.get(str(settings.get("spec.draft") or ""))
            if settings.get("spec.draft")
            else None
        )
        mib, fits = shared_slice_mib(
            ctx.predictor,
            ctx.model,
            ctx.device,
            settings,
            concurrency=conc,
            request_tokens=request,
            floor_mib=floor_mib,
            draft=draft,
        )
        if mib == floor_mib:
            why = f"memory.gpu: the FRD-023 slice ({mib} MiB) holds {conc} x {request} tokens at the planned settings"
        else:
            why = (
                f"memory.gpu: slice {floor_mib} -> {mib} MiB so {conc} x {request} tokens (+10%) fit at the planned "
                "settings (FRD-027 §10.3)"
            )
        if not fits:
            why += "; the card cannot hold that load in one slice: it queues"
        return mib, why

    def _penalty(self, ctx: PlanContext, slice_mib: int, util: float) -> str:
        from ...device import HardwareMode

        settings = current_settings(ctx)
        w = ctx.workload()
        try:
            shared_dev = ctx.device.with_mode(
                HardwareMode(kind="shared", slice_bytes=int(slice_mib) * MIB)
            )
            dedicated = ctx.device.with_mode(HardwareMode(kind="dedicated"))
            s = {
                **settings,
                "memory.device_fraction": util,
                "batch.max_running": ctx.record.planning_max_running(),
            }
            ps = ctx.predictor.predict(s, w, ctx.model, shared_dev)
            pd = ctx.predictor.predict(
                {**s, "memory.device_fraction": ctx.record.default_fraction(dedicated)},
                w,
                ctx.model,
                dedicated,
            )
            ratio = (
                ps.throughput_tok_s.value / pd.throughput_tok_s.value
                if pd.throughput_tok_s.value
                else 1.0
            )
            ctx.cluster["_penalty_vs_dedicated"] = round(ratio - 1.0, 3)
            return (
                f"shared slice {slice_mib} MiB: predicted {ratio:.2f}x the dedicated card's throughput at c{w.concurrency} "
                "(co-tenant contention unmodelled)"
            )
        except Exception as exc:  # noqa: BLE001
            return f"shared slice {slice_mib} MiB: penalty not predicted ({exc})"


# ====================================================================== memory.host


@register
class MemoryHost(Module):
    name = "memory.host"
    version = "v1"
    after = ("memory.gpu", "kv.tiers")

    def plan(self, ctx: PlanContext) -> Fragment:
        f = self.frag()
        b = legacy_budget(ctx)
        if b.t1_bytes > 0:
            f.resource(
                "requests",
                "memory_mib",
                int(b.engine_limit_mib),
                self.name,
                "host memory with a T1 CPU tier",
            )
            f.resource(
                "limits",
                "memory_mib",
                int(b.engine_limit_mib),
                self.name,
                "host memory with a T1 CPU tier",
            )
            pods = max(1, int(ctx.placement.replicas)) if ctx.placement else 1
            node = ctx.placement.target_node if ctx.placement else None
            f.checks.append(
                {
                    "kind": "node_memory",
                    "pods": pods,
                    "request_mib": int(b.engine_limit_mib),
                    "node": node,
                }
            )
        size = b.shm_mib if b.t1_mode == "register" else 2048
        f.volumes.append(
            {"name": "shm", "kind": "emptyDir", "medium": "Memory", "size_mib": int(size)}
        )
        return f


# ====================================================================== lora


@register
class Lora(Module):
    name = "lora"
    version = "v1"
    after = ("context.batching",)

    def applies(self, ctx: PlanContext) -> Tuple[bool, str]:
        if not group(ctx).get("supports_lora") or "lora" not in ctx.record.features:
            return False, "the engine record doesn't support LoRA for this model"
        return True, ""

    def plan(self, ctx: PlanContext) -> Fragment:
        f = self.frag()
        g = group(ctx)
        loras = int(g.get("max_loras") or 1)
        rank = int(g.get("max_lora_rank") or 64)
        self._render(ctx, f, loras, rank, "on whenever the record supports LoRA (legacy)")
        return f

    def _render(self, ctx: PlanContext, f: Fragment, loras: int, rank: int, why: str) -> None:
        f.set("lora.max_adapters", int(loras), self.name, why)
        f.set("lora.max_rank", int(rank), self.name, why)
        f.envvar("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "True", self.name, "adapters load at runtime")
        f.mutable["adapters"] = {"arg_limits": {"max-loras": int(loras)}}


@register
class LoraV2(Lora):
    version = "v2"

    def applies(self, ctx: PlanContext) -> Tuple[bool, str]:
        asked = int(ctx.intent.max_loras or 0)
        attached = len(ctx.intent.adapters or [])
        if not asked and not attached:
            return False, "no adapters asked for or attached: no LoRA slot (FRD-027 §10.8)"
        if "lora" not in ctx.record.features:
            return False, "the engine record doesn't list lora"
        return True, ""

    def plan(self, ctx: PlanContext) -> Fragment:
        f = self.frag()
        limit = int(ctx.record.limits.get("max_loras") or 8)
        max_rank = int(ctx.record.limits.get("max_lora_rank") or 64)
        loras = max(int(ctx.intent.max_loras or 0), len(ctx.intent.adapters or []))
        if loras > limit:
            f.decisions.append(f"{loras} adapter slots asked for; the record allows {limit}")
            loras = limit
        rank = min(int(ctx.intent.lora_rank or max_rank), max_rank)
        self._render(ctx, f, loras, rank, "asked for in the deployment intent")
        return f
