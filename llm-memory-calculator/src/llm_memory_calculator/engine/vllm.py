"""``VLLMEngine``: how one vLLM build turns model facts and settings into memory, limits and time (FRD-027 §8).

Allocator (vLLM V1, hybrid KV-cache manager):
- the engine gets ``memory.device_fraction`` of the usable memory (the HAMi slice in shared mode); from it the
  weights (at the served format, MTP head and draft included when loaded, vision tower unless text-only) and
  the non-KV overhead (activation peak at the chunk budget, CUDA-graph pool, sampler buffers, speculative
  buffers) are taken; the rest is the KV pool;
- a hybrid model's attention block is the smallest 16-aligned block whose page holds one recurrent-state page
  (state at ``state.dtype``, conv ring buffer lengthened by ``spec.tokens``): 784 / 1568 / 400 / 800 tokens on
  Qwen3.8-27B;
- layers are grouped into equal KV-cache groups (group size = the smaller layer count) sharing one block pool;
  a block is one page in every slot of a group; a recurrent group holds 1 + k blocks per running sequence;
- vLLM refuses ``max_num_seqs`` above the recurrent blocks, a ``max_model_len`` one request cannot fill, and
  OOMs when the V2 runner's CUDA-graph profiling transient (``min(max_num_seqs, 512)`` hybrid blocks) does not
  fit beside the weights.

Time (closed loop, ``m`` sequences running): a decode step costs the roofline of the bytes it streams (weights
once -- for MoE only the experts the batch routes to --, each sequence's KV and state read and written) or the
FLOPs, times a grounded efficiency per batch bucket, plus per-step, per-layer and per-MoE-layer host costs.
Other requests' prefills interleave with a sequence's decode (chunked prefill), so
``TPOT = t_decode + (m-1) x T_prefill_mixed / output_tokens`` and
``throughput = m x output / (t_decode x output + m x T_prefill_mixed)``. When ``batch.max_running`` (or the KV
pool) binds below the concurrency, the excess queues: by Little's law the response time is ``c / X`` and the
wait lands in TTFT. Speculative decoding divides the verify step (``1 + k`` tokens per sequence, plus the draft
steps) by ``1 + accepted`` tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

from ..grounding.store import Found, GroundingStore, bucket_of, default_store
from ..planner_env import planner_env
from .base import (
    BootCheck,
    Estimate,
    Limits,
    Prediction,
    Refusal,
    Settings,
    UnsupportedSetting,
    Workload,
)
from .priors import GIB, prior_detail
from .record import EngineRecord, conflicts_for

MIB = 1024**2
BLOCK_ALIGN = 16
DENSE_BLOCK = 16
TRANSIENT_SAFETY = 2 * GIB  # free memory the profiling transient must leave (study v0b-noprofile)
REF_CAPTURE = 256  # reference captured sizes for memory.overhead
REF_CHUNK = 8192

#: NFR-1 ceilings, relative.
CEILINGS = {
    "tpot": 0.15,
    "throughput": 0.15,
    "ttft": 0.20,
    "ttft_queueing": 0.30,
    "memory": 0.03,
    "boot": 0.25,
}


@dataclass
class Resolved:
    fraction: float
    max_running: int
    max_running_explicit: bool
    chunk: int
    context: int
    fmt: str
    online: bool
    kv_dtype: str
    state_dtype: str
    spec_method: Optional[str]
    spec_k: int
    tp: int
    pp: int
    lora_adapters: int
    lora_rank: int
    text_only: bool
    prefix_cache: bool

    def as_settings(self) -> Dict[str, Any]:
        out = {
            "memory.device_fraction": self.fraction,
            "batch.max_running": self.max_running,
            "batch.prefill_chunk_tokens": self.chunk,
            "context.max_tokens": self.context,
            "weights.format": self.fmt,
            "kv.dtype": self.kv_dtype,
            "state.dtype": self.state_dtype,
            "parallel.tensor": self.tp,
            "parallel.pipeline": self.pp,
            "prefix_cache.enabled": self.prefix_cache,
        }
        if self.spec_method:
            out["spec.method"] = self.spec_method
            out["spec.tokens"] = self.spec_k
        if self.lora_adapters:
            out["lora.max_adapters"] = self.lora_adapters
            out["lora.max_rank"] = self.lora_rank
        return out


@dataclass
class Geometry:
    block_tokens: int
    page_bytes: float  # one layer slot, one block
    group_size: int
    attn_groups: int
    rec_groups: int
    sliding_groups: int
    kv_per_token_layer: float
    hybrid: bool

    @property
    def block_bytes(self) -> float:
        return self.page_bytes * self.group_size


@dataclass
class Params:
    """Grounded parameters for one (device, record, model), with the levels and residuals used."""

    found: Dict[str, Found] = field(default_factory=dict)

    def v(self, name: str) -> float:
        return self.found[name].value

    def residual(self, *names: str) -> float:
        return max((self.found[n].residual for n in names if n in self.found), default=0.0)

    def levels(self, *names: str) -> Dict[str, str]:
        order = {
            "hardware": ("sku", "arch", "prior"),
            "engine": ("record", "minor", "family", "prior"),
            "model": ("family", "class", "pooled", "prior"),
        }
        out = {}
        for dim, ladder in order.items():
            worst = 0
            for n in names:
                if n in self.found:
                    worst = max(worst, ladder.index(self.found[n].levels[dim]))
            out[dim] = ladder[worst]
        return out


class VLLMEngine:
    """The v1 engine implementation (FR-ENG-5)."""

    def __init__(
        self,
        record: EngineRecord,
        store: Optional[GroundingStore] = None,
        profiles: Any = None,
        corrections: Any = None,
    ):
        self.record = record
        self._store = store
        # FRD-027 Revision 2: measured step profiles; a configuration with one is predicted from it (opt-in until the
        # §20.7 acceptance: ``profiles`` given, or PLANNER_PREDICTOR_VERSION=r2 for the package's profiles), and its
        # outputs are grounded by L5's corrections (``perf.residual``; the package's with r2)
        self._profiles = profiles
        self._corrections = corrections

    @property
    def store(self) -> GroundingStore:
        return self._store if self._store is not None else default_store()

    def _has_boot_entries(self) -> bool:
        try:
            return any(k[0] == "boot.time" for k in self.store._index)
        except Exception:  # noqa: BLE001
            return False

    # ================================================================== unary and pairwise checks

    def supports(self, setting: str) -> bool:
        return self.record.maps(setting)

    def check_supported(self, settings: Settings) -> None:
        """Raise :class:`UnsupportedSetting` for a setting the record doesn't map or a value whose feature the
        record doesn't list (the predictor refuses to predict it, so a module cannot pick it)."""
        for name, value in settings.items():
            if value is None:
                continue
            if name in ("weights.online", "weights.variant", "weights.format"):
                if name == "weights.online" and value and "online_fp8" not in self.record.features:
                    raise UnsupportedSetting(
                        f"{self.record.id}: online quantization is not verified on this build"
                    )
                continue
            if not self.record.maps(name):
                raise UnsupportedSetting(f"{self.record.id} does not support setting {name!r}")
            feature = self.record.feature_for(name, value)
            if feature and feature not in self.record.features:
                raise UnsupportedSetting(
                    f"{self.record.id}: {name}={value!r} needs feature {feature!r}"
                )

    def validate(self, settings: Settings, model: Any = None) -> List[Refusal]:
        """Pairwise refusals from the record's ``conflicts`` (FR-ENG-6)."""
        resolved: Dict[str, Any] = {}
        if model is not None:
            resolved["context.max_tokens"] = getattr(model, "max_context", None)
        resolved.setdefault("memory.device_fraction", self.record.default_fraction(None))
        out = []
        for conflict, setting in conflicts_for(self.record, settings, resolved, model):
            out.append(
                Refusal(
                    setting=setting,
                    reason=conflict.reason or conflict.id,
                    kind="conflict",
                    conflict=conflict.id,
                )
            )
        return out

    # ================================================================== resolution and geometry

    def resolve(self, settings: Settings, model: Any, device: Any) -> Resolved:
        s = dict(settings or {})
        explicit_seqs = s.get("batch.max_running")
        return Resolved(
            fraction=float(s.get("memory.device_fraction") or self.record.default_fraction(device)),
            max_running=int(explicit_seqs or self.record.engine_default_max_running(device)),
            max_running_explicit=explicit_seqs is not None,
            chunk=int(s.get("batch.prefill_chunk_tokens") or self.record.default_chunk(device)),
            context=int(s.get("context.max_tokens") or getattr(model, "max_context", 4096)),
            fmt=str(s.get("weights.format") or getattr(model, "native_format", "bf16")),
            online=bool(s.get("weights.online", False)),
            kv_dtype=str(s.get("kv.dtype") or "auto"),
            state_dtype=str(s.get("state.dtype") or "float32"),
            spec_method=s.get("spec.method"),
            spec_k=int(s.get("spec.tokens") or (1 if s.get("spec.method") else 0)),
            tp=int(s.get("parallel.tensor") or 1),
            pp=int(s.get("parallel.pipeline") or 1),
            lora_adapters=int(s.get("lora.max_adapters") or 0),
            lora_rank=int(s.get("lora.max_rank") or (16 if s.get("lora.max_adapters") else 0)),
            text_only=bool(s.get("mm.text_only", False)),
            prefix_cache=bool(s.get("prefix_cache.enabled", True)),
        )

    def geometry(self, r: Resolved, model: Any, draft: Any = None) -> Geometry:
        kv_dtype = "bf16" if r.kv_dtype == "auto" else r.kv_dtype
        extra_layers = 0
        if r.spec_method == "eagle3" and draft is not None:
            extra_layers = max(getattr(draft.layers, "full", 1), 1)
        attn = model.layers.full + model.layers.sliding
        per_layer = model._kv_bytes_per_token_layer(kv_dtype) / max(r.tp, 1)
        if model.layers.recurrent:
            k = r.spec_k if r.spec_method else 0
            state = model.state_page_bytes(r.state_dtype, k) / max(r.tp, 1)
            block = (
                max(BLOCK_ALIGN, BLOCK_ALIGN * math.ceil(state / (per_layer * BLOCK_ALIGN)))
                if per_layer
                else BLOCK_ALIGN
            )
            group = min(attn, model.layers.recurrent) if attn else model.layers.recurrent
            return Geometry(
                block_tokens=int(block),
                page_bytes=block * per_layer,
                group_size=max(group, 1),
                attn_groups=math.ceil(attn / group) if attn else 0,
                rec_groups=math.ceil(model.layers.recurrent / group),
                sliding_groups=0,
                kv_per_token_layer=per_layer,
                hybrid=True,
            )
        full, sliding = model.layers.full + extra_layers, model.layers.sliding
        if full and sliding:
            group = min(full, sliding)
            return Geometry(
                DENSE_BLOCK,
                DENSE_BLOCK * per_layer,
                group,
                math.ceil(full / group),
                0,
                math.ceil(sliding / group),
                per_layer,
                False,
            )
        layers = max(full + sliding, 1)
        return Geometry(DENSE_BLOCK, DENSE_BLOCK * per_layer, layers, 1, 0, 0, per_layer, False)

    # ================================================================== grounding

    def params(
        self,
        model: Any,
        device: Any,
        r: Resolved,
        workload: Optional[Workload] = None,
        *,
        running: int = 1,
        prefill_tokens: float = 1024,
        decode_running: Optional[int] = None,
        draft: Any = None,
    ) -> Params:
        store = self.store
        p = Params()
        fmt = r.fmt

        def get(name: str, kind: str, detail: Optional[Dict[str, Any]] = None) -> None:
            detail = detail or {}
            value, residual = prior_detail(kind, detail)
            p.found[name] = store.find(
                kind, detail, device, self.record, model, prior=value, prior_residual=residual
            )

        get("memory.overhead", "memory.overhead", {"format": fmt})
        if p.found["memory.overhead"].levels["model"] == "prior":
            get(
                "memory.overhead", "memory.overhead", {}
            )  # no entry for this format anywhere: any format
        get("memory.capture", "memory.capture")
        get("memory.chunk", "memory.chunk")
        if r.spec_method:
            get("memory.spec", "memory.spec", {"method": r.spec_method})
        get(
            "prefill.kappa",
            "prefill.kappa",
            {"tokens": prefill_bucket(prefill_tokens), "bucket": bucket_of(running)},
        )
        shape = workload.shape_for(r.prefix_cache) if workload is not None else "chat"
        get("arrival.sync", "arrival.sync", {"method": r.spec_method or "none"})
        # the TTFT terms per workload shape (the decode wave keeps half its burst, the long one none); TPOT and
        # throughput keep the pooled value they were grounded with; pooled when the shape is unmeasured
        get("arrival.sync.ttft", "arrival.sync", {"method": r.spec_method or "none", "shape": shape})
        if p.found["arrival.sync.ttft"].levels["model"] == "prior":
            p.found["arrival.sync.ttft"] = p.found["arrival.sync"]
        # per batch bucket: real text queues longer at c32 than at c8 (T-GND-21); pooled when the bucket is unmeasured
        get(
            "arrival.queue",
            "arrival.queue",
            {"bucket": bucket_of(decode_running if decode_running is not None else running)},
        )
        if p.found["arrival.queue"].levels["model"] == "prior":
            get("arrival.queue", "arrival.queue")
        if workload is not None:
            get("arrival.drain", "arrival.drain", {"workload": workload.workload_class})
        for name in (
            "decode.tau_layer",
            "decode.tau_moe",
            "step.tau",
            "prefill.tau_layer",
            "prefill.mix",
            "api.tau",
            "lora.ratio",
            "spec.draft_tau",
            "boot.base",
            "boot.capture",
            "boot.load_bandwidth",
            "stall.first_shape",
        ):
            get(name, name)
        bucket = bucket_of(decode_running if decode_running is not None else running)
        get("decode.kappa", "decode.kappa", {"bucket": bucket})
        if r.spec_method:
            # per batch bucket: a small speculative batch is memory-bound and hides part of a mixed-in prefill, a large
            # one does not (27B MTP: c16 carries ~half the base interference, c64 all of it); pooled when unmeasured
            get("spec.mix", "spec.mix", {"method": r.spec_method, "bucket": bucket})
            if p.found["spec.mix"].levels["model"] == "prior":
                get("spec.mix", "spec.mix", {"method": r.spec_method})
        if fmt not in ("bf16", "fp16"):
            get("precision.ratio", "precision.ratio", {"format": fmt, "bucket": bucket})
            get("precision.prefill", "precision.prefill", {"format": fmt})
        if r.spec_method:
            get("spec.prefill", "spec.prefill", {"method": r.spec_method})
            get("spec.prefill.ttft", "spec.prefill", {"method": r.spec_method, "shape": shape})
            if p.found["spec.prefill.ttft"].levels["model"] == "prior":
                p.found["spec.prefill.ttft"] = p.found["spec.prefill"]
        if r.spec_method and workload is not None:
            detail = {
                "method": r.spec_method,
                "k": r.spec_k,
                "sampling": workload.sampling,
                "workload": workload.workload_class,
            }
            # a draft model's acceptance is its own (8B, chat c1, k=3: the RedHat and AngelSlim drafts are 67% and -20%
            # off one pooled value); keyed by the draft first, the method's pooled entries when the draft is unmeasured
            dkey = _draft_id(draft) if r.spec_method in ("eagle3", "draft") else None
            tiers = ([{"draft": dkey}] if dkey else []) + [{}]
            for extra in tiers:
                get("spec.acceptance", "spec.acceptance", {**detail, **extra})
                if p.found["spec.acceptance"].levels["model"] != "prior":
                    break
            # the mean (throughput) per batch bucket: at c = 1 a run's mean request is its median one, while a
            # closed-loop run at c > 1 lasts as long as its slowest requests; pooled when the bucket is unmeasured
            for extra in [*({**t, "bucket": bucket} for t in tiers[:-1]), *tiers[:-1], {"bucket": bucket}, {}]:
                get("spec.acceptance_mean", "spec.acceptance_mean", {**detail, **extra})
                if p.found["spec.acceptance_mean"].levels["model"] != "prior":
                    break
            # this stratum's (method[, draft], shape, batch bucket) TPOT bias where it was measured (T-GND-19)
            stratum = {"method": r.spec_method, "shape": shape, "bucket": bucket}
            for extra in tiers:
                found = store.find(
                    "spec.scale", {**stratum, **extra}, device, self.record, model, prior=1.0, prior_residual=1.0
                )
                if found.levels["model"] != "prior":
                    p.found["spec.scale"] = found
                    break
        if workload is not None:
            # each quantity's band from its stratum's measured error, most specific stratum first, where this model's
            # own family measured it, on a build at least as specific as the decode terms it replaces (another family's
            # stratum would make a hold-out look grounded, NFR-5; another build's would stand in for this build's own
            # terms); the terms' residuals otherwise (T-GND-19)
            dkey = _draft_id(draft) if r.spec_method in ("eagle3", "draft") else None
            strata = band_strata(r, model, device, shape, bucket, dkey)
            ref = p.found["decode.kappa"].levels
            for q in BAND_QUANTITIES:
                for st in strata:
                    found = store.find(
                        f"band.{q}", st, device, self.record, model, prior=1.0, prior_residual=1.0
                    )
                    if found.levels["model"] == "prior":
                        continue
                    if found.levels["model"] == "family" and LEVEL_ORDER["engine"].index(
                        found.levels["engine"]
                    ) <= LEVEL_ORDER["engine"].index(ref["engine"]):
                        p.found[f"band.{q}"] = found
                    break
            # the stratum's measured replicate spread (FRD-027 §16, the A-05 amendment), from this family's own runs on
            # this build or its upstream minor: another family's noise would excuse a band the model earned on its own
            for q in NOISE_REPORTED:
                for st in strata:
                    found = store.find(
                        f"noise.{q}", st, device, self.record, model, prior=0.0, prior_residual=0.0
                    )
                    if found.levels["model"] == "prior":
                        continue
                    if found.levels["model"] == "family" and found.levels["engine"] in ("record", "minor"):
                        p.found[f"noise.{q}"] = found
                    break
        return p

    # ================================================================== memory and limits

    def _weights(self, r: Resolved, model: Any, draft: Any) -> float:
        fmt = r.fmt if (r.online or r.fmt != model.native_format) else "native"
        weights = model.weight_bytes(
            fmt, include_mtp=(r.spec_method == "mtp"), include_vision=not r.text_only
        )
        weights /= max(r.tp, 1)
        if r.spec_method == "eagle3" and draft is not None:
            weights += _eagle3_draft_bytes(draft) / max(r.tp, 1)
        if r.lora_adapters:
            weights += _lora_bytes(model, r.lora_adapters, r.lora_rank) / max(r.tp, 1)
        return float(weights)

    def _overhead(self, r: Resolved, model: Any, p: Params) -> float:
        captured = min(r.max_running, self.record.capture_max())
        o = p.v("memory.overhead")
        # sampler buffers: fp32 logits for every running sequence (physics, not fitted)
        o += 4.0 * model.vocab_size * (min(r.max_running, 1024) - REF_CAPTURE)
        o += p.v("memory.capture") * model.hidden_size * (captured - REF_CAPTURE)
        inter = int(getattr(model, "hidden_size", 4096) * 4)
        o += p.v("memory.chunk") * (r.chunk - REF_CHUNK) * (model.hidden_size + inter)
        if r.spec_method:
            o += p.v("memory.spec") * (1.0 + 0.1 * max(r.spec_k - 1, 0))
        return max(float(o), 0.25 * GIB)

    def limits(self, settings: Settings, model: Any, device: Any, *, draft: Any = None) -> Limits:
        r = self.resolve(settings, model, device)
        p = self.params(model, device, r)
        geo = self.geometry(r, model, draft)
        usable = float(device.usable_memory_bytes)
        budget = r.fraction * usable
        weights = self._weights(r, model, draft)
        overhead = self._overhead(r, model, p)
        pool = budget - weights - overhead
        num_blocks = max(int(pool // geo.block_bytes), 0) if geo.block_bytes > 0 else 0
        k = r.spec_k if r.spec_method else 0
        checks: List[BootCheck] = []
        # a sliding group holds at most the window's blocks plus the one it is filling (0: no window, unbounded)
        sliding_group_blocks = (
            math.ceil(model.layers.sliding_window / DENSE_BLOCK) + 1
            if (geo.sliding_groups and model.layers.sliding_window)
            else 0
        )

        # recurrent blocks one request holds beyond its attention blocks (vLLM 0.30 hybrid allocator:
        # 2 per recurrent group, 9 + k with MTP -- fitted to the s-mtp1..3 / s-mbt32k refusals)
        if geo.hybrid:
            rec_per_request = geo.rec_groups * (2 if not r.spec_method else 9 + k)
            # a running sequence holds 1 + k blocks in EACH recurrent group (the schedulers' own counts: the fork's
            # 27B runs at most 83 x 1280-token sequences on 416 blocks = 2 + 3 x 1; upstream MTP k=3 at most 24)
            state_per_seq = geo.rec_groups * (1 + k) * geo.block_bytes
            sequence_ceiling = num_blocks
        else:
            sliding_blocks = 0
            if geo.sliding_groups and model.layers.sliding_window:
                sliding_blocks = geo.sliding_groups * sliding_group_blocks
            rec_per_request = sliding_blocks
            state_per_seq = 0.0
            sequence_ceiling = max(num_blocks // max(geo.attn_groups, 1), 0)
        attn_blocks_available = max(num_blocks - rec_per_request, 0)
        if geo.hybrid:
            max_context = (attn_blocks_available // max(geo.attn_groups, 1)) * geo.block_tokens
        else:
            max_context = (attn_blocks_available // max(geo.attn_groups, 1)) * DENSE_BLOCK
        kv_tokens = (
            int(num_blocks * geo.block_tokens // max(geo.attn_groups + geo.sliding_groups, 1))
            if geo.hybrid
            else int(num_blocks * DENSE_BLOCK)
        )

        band = (
            self.store.memory_band_bytes(device, self.record, model)
            + p.found["memory.overhead"].residual
        )

        checks.append(
            BootCheck(
                "weights_fit",
                pool > 0,
                pool,
                (
                    "weights and the engine overhead fit the fraction"
                    if pool > 0
                    else "No available memory for the cache blocks"
                ),
            )
        )
        max_fraction = self.record.max_fraction(device)
        checks.append(
            BootCheck(
                "device_fraction",
                r.fraction <= max_fraction + 1e-9,
                (max_fraction - r.fraction) * usable,
                f"memory.device_fraction {r.fraction:.2f} vs the build's measured ceiling {max_fraction:.2f} on {device.sku}",
            )
        )
        need = (
            math.ceil(r.context / (geo.block_tokens if geo.hybrid else DENSE_BLOCK))
            * max(geo.attn_groups, 1)
            + rec_per_request
        )
        checks.append(
            BootCheck(
                "one_request_fits",
                need <= num_blocks,
                (num_blocks - need) * geo.block_bytes,
                f"context.max_tokens {r.context} needs {need} blocks; the pool holds {num_blocks} "
                f"(largest bootable {max_context})",
            )
        )
        transient = 0.0
        transient_ceiling: Optional[int] = None
        if geo.hybrid and self.record.runner == "v2":
            # sized with the model's own (fp32) state page: a bf16 state halves the steady block, not the profiling
            # allocation (TCS 10-18: 27B, bf16 state, 256 seqs on a 65,024 MiB slice OOMed in profile_cudagraph_memory
            # at 62.27 GiB allocated -- the halved estimate said 6.25 GiB, the fp32 page gives 12.25)
            profile_geo = self.geometry(replace(r, state_dtype="float32"), model, draft)
            transient = min(r.max_running, self.record.capture_max()) * profile_geo.block_bytes
            free = usable - weights - overhead - TRANSIENT_SAFETY
            if profile_geo.block_bytes > 0:
                # fewer sequences shrink the capture and the overhead, so this ceiling is safe below r.max_running too
                fits = max(int(free // profile_geo.block_bytes), 0)
                transient_ceiling = fits if fits < self.record.capture_max() else None
            checks.append(
                BootCheck(
                    "profiling_transient",
                    transient <= free,
                    free - transient,
                    f"CUDA-graph profiling allocates {transient / GIB:.1f} GiB "
                    f"(min(max_num_seqs, {self.record.capture_max()}) hybrid blocks); {free / GIB:.1f} GiB is free",
                )
            )
            checks.append(
                BootCheck(
                    "sequence_ceiling",
                    r.max_running <= sequence_ceiling,
                    (sequence_ceiling - r.max_running) * geo.block_bytes,
                    f"batch.max_running {r.max_running} vs {sequence_ceiling} recurrent-state blocks",
                )
            )
        notes = []
        if device.hardware_mode.kind == "shared":
            notes.append("contention: unmodelled")
        return Limits(
            budget_bytes=budget,
            weights_bytes=weights,
            overhead_bytes=overhead,
            kv_pool_bytes=max(pool, 0.0),
            block_tokens=geo.block_tokens,
            block_bytes=geo.block_bytes,
            num_blocks=num_blocks,
            kv_tokens=kv_tokens,
            sequence_ceiling=int(sequence_ceiling),
            state_bytes_per_sequence=float(state_per_seq),
            attn_groups=max(int(geo.attn_groups), 1),
            sliding_groups=int(geo.sliding_groups),
            sliding_group_blocks=int(sliding_group_blocks),
            max_bootable_context=int(max(max_context, 0)),
            profiling_transient_bytes=float(transient),
            transient_ceiling=transient_ceiling,
            checks=checks,
            memory_band_bytes=float(band),
            levels=p.levels("memory.overhead"),
            notes=notes,
            resolved=r.as_settings(),
        )

    # ================================================================== time

    def _compute_flops(self, device: Any, fmt: str) -> float:
        support = device.dtype_support.get(fmt)
        if fmt in ("fp8", "int8") and support == "native":
            return device.flops(fmt)
        return device.flops("bf16")

    def _decode_roof_ms(
        self,
        model: Any,
        device: Any,
        r: Resolved,
        m: float,
        ctx: float,
        tokens_per_seq: float = 1.0,
    ) -> float:
        fmt = r.fmt if (r.online or r.fmt != model.native_format) else "native"
        kv_dtype = "bf16" if r.kv_dtype == "auto" else r.kv_dtype
        tokens = max(m * tokens_per_seq, 1.0)
        weight = model.weight_bytes_streamed(
            fmt, batch=max(int(round(m)), 1), tokens=max(int(round(tokens)), 1)
        )
        if r.spec_method == "mtp":
            weight += model.mtp_weight_bytes(fmt) * 0  # the MTP head is costed in the draft steps
        kv = m * model.kv_bytes_for_context(int(ctx), kv_dtype)
        state = m * 2.0 * model.state_bytes_per_sequence(r.state_dtype)
        bytes_ = (weight + kv + state) / max(r.tp, 1)
        flops = (
            tokens
            * (model.linear_flops_per_token() + model.attention_flops_per_token(int(ctx)))
            / max(r.tp, 1)
        )
        return 1000.0 * max(
            bytes_ / device.hbm_bandwidth, flops / self._compute_flops(device, r.fmt)
        )

    def _host_ms(self, model: Any, p: Params, phase: str = "decode") -> float:
        layers = model.layers.total or 1
        if phase == "decode":
            return (
                p.v("step.tau")
                + p.v("decode.tau_layer") * layers
                + p.v("decode.tau_moe") * model.moe_layers
            )
        return (
            p.v("step.tau")
            + p.v("prefill.tau_layer") * layers
            + p.v("decode.tau_moe") * model.moe_layers
        )

    def decode_step_ms(
        self, model: Any, device: Any, r: Resolved, p: Params, m: float, ctx: float
    ) -> float:
        kappa = p.v("decode.kappa")
        if "precision.ratio" in p.found:
            kappa *= p.v("precision.ratio")
        if r.lora_adapters:
            kappa *= p.v("lora.ratio")
        tps = 1.0 + (r.spec_k if r.spec_method else 0)
        step = kappa * self._decode_roof_ms(model, device, r, m, ctx, tps) + self._host_ms(
            model, p, "decode"
        )
        if r.spec_method:
            step += r.spec_k * self._draft_step_ms(model, device, r, p, m)
        return step

    def _draft_step_ms(self, model: Any, device: Any, r: Resolved, p: Params, m: float) -> float:
        fmt = r.fmt if (r.online or r.fmt != model.native_format) else "native"
        if r.spec_method == "mtp":
            head = model.mtp_weight_bytes(fmt) + 2.0 * model.params_by_role.get("lm_head", 0)
        elif r.spec_method == "eagle3":
            head = 2.0 * (
                model.hidden_size * model.hidden_size * 12
            )  # one draft layer, draft-vocab head
        else:
            head = 0.0
        return 1000.0 * head / max(r.tp, 1) / device.hbm_bandwidth + p.v("spec.draft_tau")

    def prefill_ms(
        self,
        model: Any,
        device: Any,
        r: Resolved,
        p: Params,
        tokens: float,
        *,
        requests: int = 1,
        mixed: bool = False,
        ttft: bool = False,
    ) -> float:
        """Prefill ``tokens`` new tokens spread over ``requests`` requests, in chunks of ``batch.prefill_chunk_tokens``.

        ``mixed`` is the marginal cost when the chunks ride along decode steps (the weights are already streamed for
        the decode, so only the FLOPs count). ``ttft`` prices it with the TTFT terms of the workload's shape."""
        n = max(float(tokens), 0.0)
        if n <= 0:
            return 0.0
        fmt = r.fmt if (r.online or r.fmt != model.native_format) else "native"
        req = max(int(requests), 1)
        per_request = n / req
        chunk = max(r.chunk, 1)
        chunks = max(math.ceil(n / chunk), 1)
        per_chunk = n / chunks
        flops_rate = self._compute_flops(device, r.fmt)
        # linear FLOPs for every token; attention per request over its own length
        linear = model.linear_flops_per_token() * n
        attention = req * model.prefill_flops(int(per_request)) - model.linear_flops_per_token() * n
        flops = (linear + max(attention, 0.0)) / max(r.tp, 1)
        kappa = p.v("prefill.kappa")
        if "precision.prefill" in p.found:
            kappa *= p.v("precision.prefill")
        if "spec.prefill" in p.found:
            kappa *= p.v("spec.prefill.ttft" if ttft and "spec.prefill.ttft" in p.found else "spec.prefill")
        if mixed:
            return p.v("prefill.mix") * kappa * 1000.0 * flops / flops_rate
        weight = model.weight_bytes_streamed(fmt, tokens=int(per_chunk)) / max(r.tp, 1)
        roof = 1000.0 * max(chunks * weight / device.hbm_bandwidth, flops / flops_rate)
        return kappa * roof + chunks * self._host_ms(model, p, "prefill")

    def profiles(self) -> Any:
        if self._profiles is not None:
            return self._profiles
        if planner_env("PREDICTOR_VERSION").lower() == "r2":
            from ..perf.store import default_profiles

            return default_profiles()
        return None

    def corrections(self) -> Any:
        if self._corrections is not None:
            return self._corrections
        if self._profiles is None and planner_env("PREDICTOR_VERSION").lower() == "r2":
            from ..perf.residual import default_corrections

            return default_corrections()
        return None

    def predict(
        self, settings: Settings, workload: Workload, model: Any, device: Any, *, draft: Any = None
    ) -> Prediction:
        self.check_supported(settings)
        profiles = self.profiles()
        if profiles is not None:
            from ..perf.predict import find_profile
            from ..perf.predict import predict as profile_predict

            profile = find_profile(self, profiles, settings, model, device)
            if profile is not None:
                return profile_predict(
                    self, profile, settings, workload, model, device, draft=draft, corrections=self.corrections()
                )
        lim = self.limits(settings, model, device, draft=draft)
        r = self.resolve(settings, model, device)
        c = max(int(workload.concurrency), 1)
        inp, out = max(float(workload.input_tokens), 1.0), max(float(workload.output_tokens), 1.0)
        per_seq_tokens = inp + out

        def kv_capacity(tokens: float) -> int:
            """Sequences of ``tokens`` the KV pool holds at once: a sequence holds its blocks in every attention group,
            in every sliding group (up to the window) and, in a hybrid, 1 + k blocks per recurrent group."""
            if not (lim.kv_tokens and tokens):
                return c
            if (lim.state_bytes_per_sequence or lim.sliding_groups) and lim.block_bytes:
                return int(lim.num_blocks // max(lim.blocks_per_sequence(tokens), 1))
            return int(lim.kv_tokens // tokens)

        hard_cap = min(r.max_running, lim.sequence_ceiling or r.max_running)
        # running (throughput, TPOT): sequences held at their full length -- growth past the pool is preempted
        cap = max(min(hard_cap, max(kv_capacity(per_seq_tokens), 1)), 1)
        m = min(c, cap)
        queueing = c > m
        # the median request carries the queue wait only when the load cannot run: in a prefill-heavy load the admitted
        # requests hold nearly all their blocks from admission, so arrivals wait for completions; in a decode-heavy
        # one admission is cheap and the overflow is resolved by preempting growing requests, so arrivals still get a
        # first token at once -- unless over half the load cannot run (study: 27B decode c128 runs 83, median TTFT
        # 2.3 s; TCS shared 8B chat c64 runs ~33, median 3.0 s)
        median_waits = (inp / (inp + out) >= 0.5) or (c - m) / c >= 0.5
        hit = workload.prefix_hit_rate
        if hit is None:
            hit = 0.0
            if workload.prefix_tokens and r.prefix_cache:
                cached = (
                    int(workload.prefix_tokens) // max(lim.block_tokens, 1)
                ) * lim.block_tokens
                hit = min(cached / inp, 1.0)
        if not r.prefix_cache:
            hit = 0.0
        # a fixed-count benchmark of identical waves finds nothing cached in its first wave, so a measured run-average
        # hit rate is the later waves' hit spread over all of them: the median request is a warm wave, and the cycle
        # carries the cold wave once per run (27B prefix c16, 4 waves: 0.65 average hit, ~0.87 warm; mean TTFT
        # 1.1-1.6 s against medians of 0.4-1.0 s)
        waves = float(workload.requests) / m if (workload.requests and m > 1) else 1.0
        cold_share = 0.0
        if (
            workload.arrival_model == "wave"
            and m > 1
            and hit > 0
            and waves > 1.5
            and workload.prefix_hit_rate is not None
        ):
            # the warm waves cannot hit more than the shared prefix (block-aligned): at a run-average 0.76 over 4 waves
            # the uncapped warm hit saturated at 1.0 (study 27B prefix: 42 ms predicted, 905 ms measured)
            # and no request hits its whole prompt: vLLM caches full blocks only and always computes the last token, so
            # a request re-sent whole hits floor((input - 1) / block) blocks (gpt-oss repeat at c16: 1.0 had left the API
            # time alone, 12.4 ms against 37 ms measured; T-GND-23)
            block = max(lim.block_tokens, 1)
            ceiling = min((max(int(inp) - 1, 0) // block) * block / inp, 1.0)
            if workload.prefix_tokens:
                ceiling = min(ceiling, (int(workload.prefix_tokens) // block) * block / inp)
            hit = min(hit * waves / (waves - 1.0), ceiling)
            cold_share = 1.0 / waves
        new_tokens = inp * (1.0 - hit)
        # a benchmark of identical requests runs in synchronized waves; speculative decoding partly desynchronizes
        # them (arrival.sync is the fraction of the wave behaviour that survives, grounded per method)
        wave = workload.arrival_model == "wave" and m > 1
        p = self.params(
            model,
            device,
            r,
            workload,
            running=m if wave else 1,
            prefill_tokens=new_tokens,
            decode_running=m,
            draft=draft,
        )
        ctx = inp + out / 2.0
        step = self.decode_step_ms(model, device, r, p, m, ctx)
        tokens_per_step = 1.0
        if r.spec_method:
            tokens_per_step += max(p.v("spec.acceptance"), 0.0)
        t_tok = step / tokens_per_step
        t_tok_mean = (
            step / (1.0 + max(p.v("spec.acceptance_mean"), 0.0)) if r.spec_method else t_tok
        )
        api = p.v("api.tau")
        t_pref = self.prefill_ms(model, device, r, p, new_tokens)
        t_pref_ttft = self.prefill_ms(model, device, r, p, new_tokens, ttft=True)
        t_mix = self.prefill_ms(model, device, r, p, new_tokens, mixed=True)
        # what other requests' prefills add to a speculative request's TPOT, relative to a plain one (spec.mix, grounded
        # on the TPOT of the loads that carry them); the arrival rate and the cycle keep the plain cost
        mix = p.v("spec.mix") if r.spec_method else 1.0
        # steady arrivals: other requests' prefills interleave with this request's decode
        x_steady = m / (t_tok * out + m * t_mix)

        def steady_ttft(prefill: float) -> float:
            # a steady arrival waits for the step in flight and for prefills queued ahead of it (arrival.queue: decode
            # steps waited, grounded on real-text traffic)
            rho = min(x_steady * prefill, 0.95)
            queue_prefill = rho * prefill / (2.0 * (1.0 - rho)) if m > 1 else 0.0
            return prefill + (p.v("arrival.queue") * step if m > 1 else 0.0) + queue_prefill + api

        ttft_steady = steady_ttft(t_pref)
        tpot_steady = t_tok + (m - 1) * mix * t_mix / out
        ttft_m, tpot = ttft_steady, tpot_steady
        # the reported TTFT uses the shape's TTFT terms; TPOT and the cycle (throughput) keep the pooled ones
        ttft_shown = steady_ttft(t_pref_ttft)
        cycle_ttft = ttft_m
        if wave:

            def wave_terms(per_request: float, for_ttft: bool = False) -> Tuple[float, float]:
                """The wave's median first-token time and the prefill time left after it, for ``per_request`` new
                tokens per request."""
                burst = m * per_request
                chunk_tokens = min(float(r.chunk), burst)
                t_chunk = self.prefill_ms(
                    model,
                    device,
                    r,
                    p,
                    chunk_tokens,
                    requests=max(int(chunk_tokens // max(per_request, 1)), 1),
                    ttft=for_ttft,
                )
                n_chunks = max(math.ceil(burst / r.chunk), 1)

                def done(rank: int) -> float:
                    return math.ceil((rank + 1) * per_request / r.chunk) * t_chunk

                med = (done(m // 2 - 1) + done(m // 2)) / 2.0 if m % 2 == 0 else done(m // 2)
                return med, max(n_chunks * t_chunk - med, 0.0)

            median, overlap = wave_terms(new_tokens)
            sync = min(max(p.v("arrival.sync"), 0.0), 1.0)
            ttft_m = sync * (median + api) + (1.0 - sync) * ttft_steady
            tpot = (
                sync * (t_tok + mix * p.v("prefill.mix") * overlap / out)
                + (1.0 - sync) * tpot_steady
            )
            median_shown, _ = wave_terms(new_tokens, for_ttft=True)
            sync_shown = min(max(p.v("arrival.sync.ttft"), 0.0), 1.0)
            ttft_shown = sync_shown * (median_shown + api) + (1.0 - sync_shown) * ttft_shown
            cycle_ttft = ttft_m
            if cold_share:
                cold_median, _ = wave_terms(inp)
                cold_steady = ttft_steady + self.prefill_ms(model, device, r, p, inp) - t_pref
                cold_ttft = sync * (cold_median + api) + (1.0 - sync) * cold_steady
                cycle_ttft = ttft_m + cold_share * max(cold_ttft - ttft_m, 0.0)
        cycle = t_tok * out + m * t_mix
        # closed loop: each running request's cycle is its own TTFT plus its decode (Little's law); speculative
        # decoding uses the mean acceptance here, the median in TPOT
        tpot_mean = tpot + (t_tok_mean - t_tok)
        e2e_cycle = cycle_ttft + (out - 1) * tpot_mean
        throughput = m * out / e2e_cycle * 1000.0
        if workload.requests and m > 1 and workload.arrival_model == "steady":
            # a fixed-count run of steady, variable-length requests ends with its last one: the tail costs a share of
            # a round (arrival.drain, grounded on runs without speculation; real text at c8/c32 ran 14-24% below the
            # open-ended rate with its TPOT matched). Synthetic waves are fixed-length; the wave model covers them.
            rounds = max(float(workload.requests) / m, 1.0)
            throughput *= rounds / (rounds + max(p.v("arrival.drain"), 0.0))
        x = m / e2e_cycle
        self.last_components = {
            "running": m,
            "wave": wave,
            "decode_step_ms": step,
            "tokens_per_step": tokens_per_step,
            "t_tok_ms": t_tok,
            "api_ms": api,
            "cycle_ms": cycle,
            "tpot_ms": tpot,
            "ttft_m_ms": ttft_m,
            "kappa": p.v("decode.kappa"),
            "prefill_kappa": p.v("prefill.kappa"),
            "roof_ms": self._decode_roof_ms(
                model, device, r, m, ctx, 1.0 + (r.spec_k if r.spec_method else 0)
            ),
            "host_ms": self._host_ms(model, p, "decode"),
            "draft_ms": (
                (r.spec_k * self._draft_step_ms(model, device, r, p, m)) if r.spec_method else 0.0
            ),
        }
        ttft = ttft_shown
        e2e_m = ttft_m + (out - 1) * tpot
        if queueing and median_waits:
            response = c / x
            ttft += max(response - e2e_m, 0.0)
        # the reported TPOT carries the stratum's bias (other requests' prefill stalls); the cycle above does not
        if "spec.scale" in p.found:
            tpot = tpot * max(p.v("spec.scale"), 0.0)
        e2e = ttft + (out - 1) * tpot

        # bands come from fitted entries only; the structural host terms are absorbed by the fitted efficiencies
        dec_names = ["decode.kappa"]
        if "precision.ratio" in p.found:
            dec_names.append("precision.ratio")
        if r.spec_method:
            dec_names.append("spec.acceptance")
            if m > 1:
                dec_names.append("spec.mix")
        pref_names = ["prefill.kappa", "api.tau"] + (
            ["arrival.sync.ttft"] if wave else ["arrival.queue"] if m > 1 else []
        )
        pref_names += [n for n in ("precision.prefill", "spec.prefill.ttft") if n in p.found]
        r_tpot = p.residual(*dec_names, "prefill.mix") if m > 1 else p.residual(*dec_names)
        # a prefill error moves throughput by its share of the request's cycle (d thr / thr = d ttft / e2e), not 1:1
        # -- the wave's median-TTFT scatter that widens prefill.kappa at c > 1 does not reach the total
        prefill_share = min(ttft_m / e2e_cycle, 1.0) if e2e_cycle > 0 else 1.0
        r_thr = (
            max(r_tpot, p.residual("prefill.kappa") * prefill_share) if wave and m > 1 else r_tpot
        )
        r_ttft = p.residual(*pref_names, *(dec_names if queueing else []))
        if "band.tpot" in p.found:
            r_tpot = p.found["band.tpot"].residual
        if "band.throughput" in p.found:
            r_thr = p.found["band.throughput"].residual

        notes = list(lim.notes)
        widen = 1.25 if device.hardware_mode.kind == "shared" else 1.0
        # TP or PP > 1 is grounded only where a point at that parallelism was measured on this SKU: its cost is the
        # interconnect's, and every grounded point so far is TP=1 (FRD-027 §16, A-10)
        parallel_ungrounded = False
        if r.tp * r.pp > 1:
            found = self.store.find(
                "parallel.measured", {"tp": r.tp, "pp": r.pp}, device, self.record, model, prior=0.0
            )
            if found.levels["hardware"] != "sku" or found.levels["model"] == "prior":
                parallel_ungrounded = True
                notes.append(
                    f"parallelism: TP={r.tp} PP={r.pp} ungrounded (no point measured at this parallelism on "
                    f"{getattr(device, 'sku', None) or 'this device'})"
                )
        if not getattr(model, "recognised", True):
            notes.append("architecture: unrecognised")
        if r.spec_method and "kernel_warmup" not in self.record.features:
            notes.append("first-shape stalls: the build has no kernel warm-up (p99 TTFT)")
        ceiling_ttft = CEILINGS["ttft_queueing"] if queueing or m >= 64 else CEILINGS["ttft"]
        levels = p.levels(*dec_names, *pref_names)
        noise = {q: p.v(f"noise.{q}") if f"noise.{q}" in p.found else 0.0 for q in NOISE_REPORTED}

        def flagged(est: Estimate) -> Estimate:
            return replace(est, ungrounded=True) if parallel_ungrounded else est

        return Prediction(
            ttft_ms=flagged(Estimate.around(ttft, r_ttft * widen, ceiling=ceiling_ttft, noise=noise["ttft"])),
            tpot_ms=flagged(Estimate.around(tpot, r_tpot * widen, ceiling=CEILINGS["tpot"], noise=noise["tpot"])),
            throughput_tok_s=flagged(
                Estimate.around(throughput, r_thr * widen, ceiling=CEILINGS["throughput"], noise=noise["throughput"])
            ),
            e2e_s=flagged(Estimate.around(e2e / 1000.0, max(r_tpot, r_ttft) * widen)),
            running=int(m),
            queueing=bool(queueing),
            limits=lim,
            levels=levels,
            notes=notes,
        )

    # ================================================================== boot time

    def boot_time(
        self, settings: Settings, model: Any, device: Any, *, load_bandwidth: Optional[float] = None
    ) -> Estimate:
        """Predicted engine startup: load (bytes / store bandwidth), profile, CUDA-graph capture per size,
        and warm-up (FR-ENG-7)."""
        r = self.resolve(settings, model, device)
        p = self.params(model, device, r)
        kind = boot_kind(r, model)
        found = None
        if self._has_boot_entries():
            try:
                found = self.store.find(
                    "boot.time",
                    {"kind": kind},
                    device,
                    self.record,
                    model,
                    prior=None,
                    prior_residual=0.5,
                )
            except (
                KeyError
            ):  # nothing measured on this hardware ladder: the structural model below, from priors
                found = None
        if found is not None and found.levels["model"] != "prior":
            seconds = found.value + 4.0 * math.log2(max(r.context, 2048) / 2048.0) * 0.0
            return Estimate.around(seconds, found.residual, ceiling=CEILINGS["boot"])
        weights = self._weights(r, model, None)
        bandwidth = float(load_bandwidth or p.v("boot.load_bandwidth"))
        captured = _capture_sizes(min(r.max_running, self.record.capture_max()))
        seconds = (
            p.v("boot.base")
            + weights / bandwidth
            + p.v("boot.capture") * captured * (1 + (r.spec_k if r.spec_method else 0))
        )
        seconds += 4.0 * math.log2(max(r.context, 2048) / 2048.0)
        residual = p.residual("boot.base", "boot.capture", "boot.load_bandwidth")
        return Estimate.around(seconds, residual, ceiling=CEILINGS["boot"])


#: Prefill tokens PER REQUEST (a chunk of many short requests and one of a long request cost differently).
PREFILL_BUCKETS = (
    (0, 512, "<=512"),
    (513, 2048, "512-2K"),
    (2049, 6144, "2K-6K"),
    (6145, 10**12, ">6K"),
)


#: The quantities with a stratum band (T-GND-19). TTFT keeps its shape-keyed terms' bands: its stratum errors are
#: heavy-tailed, and on the few points a stratum holds the conformal quantile is their largest error (store 19.1 trial:
#: the 27B's speculative TTFT bands past twice the ceiling rose from 16% to 41%).
BAND_QUANTITIES = ("tpot", "throughput")
#: The quantities whose stratum replicate spread an estimate reports (``Estimate.noise``).
NOISE_REPORTED = ("ttft", "tpot", "throughput")
LEVEL_ORDER = {
    "hardware": ("sku", "arch", "prior"),
    "engine": ("record", "minor", "family", "prior"),
    "model": ("family", "class", "pooled", "prior"),
}


def band_strata(
    r: "Resolved", model: Any, device: Any, shape: str, bucket: str, draft_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """The strata a prediction's bands are calibrated in, most specific first: speculative method and draft, served
    format, LoRA, hardware mode, workload shape and batch bucket; then without the draft. A stratum without the format
    would pool configurations a deployment is not exchangeable with (the 27B's online-FP8 MTP on TCS ran 9% off a
    40-point bf16/FP8-checkpoint stratum of +-4%), so there is none: an unmeasured format keeps the terms' bands. The
    fitter and the engine both key them here, so a band is found exactly where it was measured."""
    # the served format and how it got there: a bf16 checkpoint, an fp8 one and fp8 quantized at load are each their
    # own stratum ("native" would pool the first two, whose errors differ)
    fmt = f"{r.fmt}-online" if r.online else str(r.fmt)
    base = {
        "spec": r.spec_method or "none",
        # k changes the acceptance and the verify cost: an MTP k=2 deployment is not exchangeable with k=1 and k=3 runs
        "k": int(r.spec_k) if r.spec_method else 0,
        "lora": "1" if r.lora_adapters else "0",
        "mode": device.hardware_mode.kind,
        "shape": shape,
        "bucket": bucket,
    }
    tiers = []
    if draft_id:
        tiers.append({**base, "format": fmt, "draft": draft_id})
    tiers.append({**base, "format": fmt})
    return tiers


def _draft_id(draft: Any) -> Optional[str]:
    """The grounding key of a draft's facts (``family`` is ``draft:<engine.draft_key>``), or None."""
    family = str(getattr(draft, "family", "") or "")
    return family.split(":", 1)[1] if family.startswith("draft:") else None


def prefill_bucket(tokens: float) -> str:
    t = float(tokens)
    for lo, hi, name in PREFILL_BUCKETS:
        if lo <= t <= hi:
            return name
    return ">6K"


def boot_kind(r: "Resolved", model: Any) -> str:
    """The boot-time class of a configuration: speculative decoding, a converted weight format and LoRA add start-up
    work (draft load and graphs; online quantization and new kernel shapes to compile; the adapter path's graphs and
    adapter warm-up -- the fork's 27B boots in 149 s stock and 239-258 s with LoRA)."""
    if r.spec_method:
        return "spec"
    if r.online or r.fmt != getattr(model, "native_format", r.fmt):
        return "precision"
    if r.lora_adapters:
        return "lora"
    return "base"


def _capture_sizes(max_size: int) -> int:
    """How many CUDA-graph batch sizes vLLM captures up to ``max_size`` (1, 2, 4, 8, then every 8, then 16)."""
    sizes = [s for s in (1, 2, 4) if s <= max_size]
    sizes += list(range(8, min(max_size, 256) + 1, 8))
    sizes += list(range(272, max_size + 1, 16))
    return len(sizes)


def _eagle3_draft_bytes(draft: Any) -> float:
    """EAGLE3 draft at bf16: its own embedding, one decoder layer, the 3h->h fusion and a draft-vocab head."""
    h = draft.hidden_size
    roles = dict(draft.params_by_role)
    lm_head = roles.pop("lm_head", 0)
    if getattr(draft, "shares_embedding", False):
        roles.pop("embedding", None)
    draft_vocab = int(getattr(draft, "draft_vocab", 0) or 0)
    if draft_vocab:
        lm_head = draft_vocab * h
    return 2.0 * (sum(roles.values()) + lm_head + 3 * h * h)


def _lora_bytes(model: Any, adapters: int, rank: int) -> float:
    """vLLM's LoRA slots: A and B for every linear layer, ``adapters`` x rank, bf16."""
    h = model.hidden_size
    per_layer_dims = 0
    for role, count in model.params_by_role.items():
        if role in ("attention", "dense_ffn", "shared_expert"):
            per_layer_dims += count / max(h, 1)  # sum of (out) dims per layer, times layers
    total_dims = (
        per_layer_dims + h * max(model.layers.total, 1) * 4
    )  # + the input side of each projection
    return 2.0 * adapters * rank * total_dims
