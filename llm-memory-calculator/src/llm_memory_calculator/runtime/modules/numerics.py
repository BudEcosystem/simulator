"""Stage-1 (numerics) modules (FRD-027 §10.4-10.7): weights.precision, spec_decode, state.dtype, kv.dtype.

They run once per plan before budsim's search, so the search sizes and predicts with their choices. Each offers a
numerics-changing setting only with a passing validation record (§12), only where the device can run it (FR-DEV-4)
and, for an automatic choice, only when the predicted gain's band separates from the default (FR-GND-2).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ...device import allows_dtype
from ...engine import Workload
from ...grounding.store import bucket_of
from ...kv.dtype import calibrated_kv_scales
from ...model import ModelFacts
from ..context import Fragment, PlanContext, PrecisionCandidate
from ..validation import default_validation
from .base import Module, register

INT4_FORMATS = {"int4", "awq", "gptq"}
STOCK_FRACTION_KEY = "memory.device_fraction"


def _validation(ctx: PlanContext):
    return ctx.validation or default_validation()


def _gate(ctx: PlanContext, setting: str):
    return _validation(ctx).gate(
        ctx.model.family,
        ctx.intent.model_id,
        ctx.intent.base_model_relation,
        ctx.record.id,
        setting,
    )


def _stock(ctx: PlanContext) -> Dict[str, Any]:
    """Capacity settings a stage-1 comparison predicts under (the search sets the real ones later)."""
    return {
        "batch.max_running": ctx.record.planning_max_running(),
        STOCK_FRACTION_KEY: ctx.record.default_fraction(ctx.device),
    }


def _settings(ctx: PlanContext) -> Dict[str, Any]:
    return {k: v.value for k, v in ctx.draft.settings.items()}


def effect_separates(
    ctx: PlanContext, candidate: Any, base: Any, kind: str, detail: Dict[str, Any]
) -> Tuple[bool, float]:
    """FR-GND-2 for a relative effect: the two predictions share every entry but the one that differs, so the
    candidate wins when its gain exceeds the residual of that entry (and neither prediction is ungrounded).
    """
    if not base.throughput_tok_s.value:
        return False, 0.0
    gain = candidate.throughput_tok_s.value / base.throughput_tok_s.value
    if candidate.throughput_tok_s.ungrounded or base.throughput_tok_s.ungrounded:
        return False, gain
    found = ctx.predictor.store.find(
        kind, detail, ctx.device, ctx.record, ctx.model, prior=1.0, prior_residual=1.0
    )
    return gain - 1.0 > found.residual, gain


def _grounded(ctx: PlanContext, fmt: str, concurrency: int) -> bool:
    found = ctx.predictor.store.find(
        "precision.ratio",
        {"format": fmt, "bucket": bucket_of(concurrency)},
        ctx.device,
        ctx.record,
        ctx.model,
        prior=1.0,
    )
    return found.levels["model"] != "prior" and found.levels["hardware"] != "prior"


# ====================================================================== weights.precision


@register
class WeightsPrecision(Module):
    name = "weights.precision"
    version = "v2"
    stage = "numerics"
    after = ()

    def applies(self, ctx: PlanContext) -> Tuple[bool, str]:
        if not ctx.model.recognised:
            return False, "architecture unrecognised: original weights (FR-MDL-5)"
        return True, ""

    def _candidates(self, ctx: PlanContext) -> List[PrecisionCandidate]:
        out = list(ctx.intent.precision_candidates)
        native = ctx.model.native_format
        if "online_fp8" in ctx.record.features and native in ("bf16", "fp16"):
            out.append(
                PrecisionCandidate(
                    id="online:fp8",
                    format="fp8",
                    path=None,
                    relation=ctx.intent.base_model_relation,
                )
            )
        return out

    def _facts(self, ctx: PlanContext, cand: PrecisionCandidate) -> ModelFacts:
        if cand.config:
            return ModelFacts.from_config(cand.config, family=ctx.model.family)
        return ctx.model

    def _eligible(
        self, ctx: PlanContext, cand: PrecisionCandidate, concurrency: int
    ) -> Tuple[bool, str]:
        online = cand.id.startswith("online:")
        setting = f"weights.online={cand.format}" if online else f"weights.format={cand.format}"
        path = allows_dtype(ctx.device, cand.format, record_features=ctx.record.features)
        if path is None:
            return (
                False,
                f"{ctx.device.sku} cannot run {cand.format} (no native path and no grounded fallback)",
            )
        verdict = _gate(ctx, setting)
        if not verdict.ok:
            return False, verdict.reason
        if cand.format in INT4_FORMATS:
            return (
                False,
                f"{cand.format}: no measured speed ratio on this SKU (offered only by explicit choice)",
            )
        if not _grounded(ctx, cand.format, concurrency):
            return False, f"{cand.format}: no grounded speed ratio for this SKU and family"
        return True, verdict.reason

    def plan(self, ctx: PlanContext) -> Fragment:
        f = self.frag()
        choice = ctx.intent.weight_precision or "auto"
        concurrency = int(ctx.intent.concurrency or 1)
        cands = self._candidates(ctx)
        chosen: Optional[PrecisionCandidate] = None
        effect: Optional[float] = None
        if choice == "original":
            f.decisions.append(f"original weights ({ctx.model.native_format}) by choice")
        elif choice != "auto":
            chosen = next((c for c in cands if c.id == choice), None)
            if chosen is None:
                f.decisions.append(
                    f"weight variant {choice!r} is not a candidate for this model; original kept"
                )
            else:
                ok, why = self._eligible(ctx, chosen, concurrency)
                if (
                    allows_dtype(ctx.device, chosen.format, record_features=ctx.record.features)
                    is None
                ):
                    f.decisions.append(
                        f"{chosen.id} refused: {ctx.device.sku} cannot run {chosen.format}; original kept"
                    )
                    chosen = None
                elif not ok:
                    f.decisions.append(f"{chosen.id} by explicit choice; warning: {why}")
        else:
            chosen, effect = self._auto(ctx, cands, concurrency, f)
        if chosen is None:
            self.output = {
                "format": ctx.model.native_format,
                "path": None,
                "online": False,
                "id": "original",
                "facts": ctx.model,
                "gain": None,
            }
            return f
        online = chosen.id.startswith("online:")
        f.set(
            "weights.format",
            chosen.format,
            self.name,
            f"{chosen.id}" + (f" (predicted {effect:.2f}x)" if effect else ""),
        )
        if online:
            f.set("weights.online", True, self.name, "online conversion of the original checkpoint")
        else:
            f.set("weights.variant", chosen.id, self.name, "registry variant served")
        if ctx.intent.base_model_relation in ("finetune", "merge"):
            f.decisions.append(
                "fine-tuned checkpoint: the family's quality evidence may not hold; run a quality check (budeval)"
            )
        self.output = {
            "format": chosen.format,
            "path": None if online else chosen.path,
            "online": online,
            "id": chosen.id,
            "facts": self._facts(ctx, chosen),
            "gain": effect,
        }
        return f

    def _auto(
        self, ctx: PlanContext, cands: List[PrecisionCandidate], concurrency: int, f: Fragment
    ):
        w = ctx.workload(concurrency)
        try:
            base = ctx.predictor.predict(_stock(ctx), w, ctx.model, ctx.device)
        except Exception as exc:  # noqa: BLE001
            f.decisions.append(f"original kept: baseline not predictable ({exc})")
            return None, None
        best, best_pred = None, base
        for cand in cands:
            ok, why = self._eligible(ctx, cand, concurrency)
            if not ok:
                f.decisions.append(f"{cand.id} not offered: {why}")
                continue
            settings = {**_stock(ctx), "weights.format": cand.format}
            if cand.id.startswith("online:"):
                settings["weights.online"] = True
            try:
                pred = ctx.predictor.predict(settings, w, self._facts(ctx, cand), ctx.device)
            except Exception as exc:  # noqa: BLE001
                f.decisions.append(f"{cand.id} not offered: {exc}")
                continue
            wins, gain = effect_separates(
                ctx,
                pred,
                best_pred,
                "precision.ratio",
                {"format": cand.format, "bucket": bucket_of(concurrency)},
            )
            if wins:
                best, best_pred = cand, pred
            else:
                f.decisions.append(
                    f"{cand.id}: predicted {gain:.2f}x vs {best.id if best else 'original'}, "
                    "within the effect's band, not chosen"
                )
        if best is None:
            f.decisions.append(
                f"original weights ({ctx.model.native_format}): no candidate's band separates"
            )
            return None, None
        gain = best_pred.throughput_tok_s.value / base.throughput_tok_s.value
        f.decisions.append(f"{best.id} chosen: predicted {gain:.2f}x throughput at c{concurrency}")
        return best, gain

    def stage1_output(self, ctx: PlanContext, frag: Fragment) -> Dict[str, Any]:
        return dict(getattr(self, "output", {}) or {})


# ====================================================================== spec_decode


@register
class SpecDecode(Module):
    name = "spec_decode"
    version = "v2"
    stage = "numerics"
    after = ("weights.precision",)

    FEATURE = {"mtp": "spec_mtp", "eagle3": "spec_eagle3", "ngram": "spec_ngram"}

    def applies(self, ctx: PlanContext) -> Tuple[bool, str]:
        if not ctx.intent.speculative.enabled:
            return False, ""
        if not ctx.model.recognised:
            return False, "architecture unrecognised (FR-MDL-5)"
        return True, ""

    def method(self, ctx: PlanContext) -> Optional[str]:
        spec = ctx.intent.speculative
        if spec.method:
            return spec.method
        if ctx.model.mtp_layers and ("spec_mtp" in ctx.record.features or not spec.draft_path):
            return "mtp"  # eligibility reports a record without spec_mtp
        if spec.draft_path:
            return "eagle3"
        return None

    def eligible(self, ctx: PlanContext, method: Optional[str]) -> Tuple[bool, str]:
        if method is None:
            return False, "no eligible method: no MTP head on this record, no validated draft"
        feature = self.FEATURE.get(method)
        if feature not in ctx.record.features:
            return False, f"the engine record {ctx.record.id} doesn't list {feature!r}"
        if method == "mtp" and not ctx.model.mtp_layers:
            return False, "the checkpoint has no MTP head"
        if method == "eagle3" and not ctx.intent.speculative.draft_path:
            return False, "EAGLE3 needs a draft model"
        if method != "ngram":
            verdict = _gate(ctx, f"spec.method={method}")
            if not verdict.ok:
                return False, verdict.reason
        return True, ""

    def default_k(self, ctx: PlanContext, method: str) -> Tuple[int, str]:
        sampling = ctx.intent.sampling_class
        c = int(ctx.intent.concurrency or 1)
        if method == "mtp":
            if sampling == "sampled":
                return 2, "MTP k=2 under sampled traffic (k=2 >= k=3 measured at T=1)"
            if c >= 64:
                return 1, "MTP k=1 at >= 64 concurrent (verify cost dominates)"
            if c <= 16:
                return 3, "MTP k=3 under greedy traffic at <= 16 concurrent"
            return 2, "MTP k=2 between 16 and 64 concurrent"
        if method == "eagle3":
            if sampling == "sampled":
                return (
                    3,
                    "EAGLE3 under sampled traffic is not recommended: 0.79-0.88x measured at T>=0.7",
                )
            return 3, "EAGLE3 k=3 under greedy traffic"
        return 3, "n-gram k=3"

    def plan(self, ctx: PlanContext) -> Fragment:
        f = self.frag()
        method = self.method(ctx)
        ok, why = self.eligible(ctx, method)
        if not ok:
            f.decisions.append(f"speculative decoding not rendered: {why}")
            self.output = {"method": None, "reason": why}
            return f
        user_k = ctx.intent.speculative.num_speculative_tokens
        k, why_k = (int(user_k), "the user's k") if user_k else self.default_k(ctx, method)
        limit = ctx.record.max_spec_tokens(method)
        if k > limit:
            f.decisions.append(f"k={k} clamped to the record's maximum {limit} for {method}")
            k = limit
        f.set("spec.method", method, self.name, "enabled in the deployment intent")
        f.set("spec.tokens", int(k), self.name, why_k)
        if method in ("eagle3", "draft"):
            f.set(
                "spec.draft",
                ctx.intent.speculative.draft_path,
                self.name,
                "the validated draft for this target",
            )
        if ctx.record.spec_prefix_cache_safe(ctx.model):
            f.decisions.append(
                "prefix caching stays on with speculative decoding: this build is verified for full-attention models "
                "(output-corruption scan, vLLM #53912)"
                if "spec_prefix_cache_safe" not in ctx.record.features
                else "prefix caching stays on with speculative decoding: this build is verified safe (vLLM #53912)"
            )
        else:
            f.set(
                "prefix_cache.enabled",
                False,
                self.name,
                "speculative decoding with prefix caching can corrupt output on this build (vLLM #53912)",
            )
            if (ctx.intent.workload_profile or "") == "agentic":
                f.decisions.append(
                    "agentic profile: the prefix-hit loss usually outweighs the speculative gain; "
                    "advice is off until the build is safe"
                )
        advice = self.advice(ctx, method)
        self.output = {"method": method, "k": k, "advice": advice}
        if advice:
            c = int(ctx.intent.concurrency or 1)
            row = advice.get(str(k), {}).get(str(min((1, 16, 64), key=lambda x: abs(x - c))), {})
            if row.get(ctx.intent.sampling_class):
                f.decisions.append(
                    f"{method} k={k}: predicted {row[ctx.intent.sampling_class]:.2f}x at the planned load "
                    f"({ctx.intent.sampling_class})"
                )
        return f

    def advice(self, ctx: PlanContext, method: str) -> Dict[str, Dict[str, Dict[str, float]]]:
        """``advice[k][concurrency][sampling] = predicted throughput ratio vs no speculation`` (FR-SPEC-5)."""
        table: Dict[str, Dict[str, Dict[str, float]]] = {}
        draft = ctx.drafts.get(ctx.intent.speculative.draft_path or "")
        base_settings = {**_stock(ctx), **_settings(ctx), "context.max_tokens": 32768}
        for k in range(1, ctx.record.max_spec_tokens(method) + 1):
            for c in (1, 16, 64):
                for sampling in ("greedy", "sampled"):
                    w = Workload(
                        input_tokens=float(ctx.intent.input_tokens),
                        output_tokens=float(ctx.intent.output_tokens),
                        concurrency=c,
                        sampling=sampling,
                        workload_class="real",
                        arrival="steady",
                    )
                    try:
                        base = ctx.predictor.predict(base_settings, w, ctx.model, ctx.device)
                        spec = {**base_settings, "spec.method": method, "spec.tokens": k}
                        if method in ("eagle3", "draft"):
                            spec["spec.draft"] = ctx.intent.speculative.draft_path
                        if not ctx.record.spec_prefix_cache_safe(ctx.model):
                            spec["prefix_cache.enabled"] = False
                        pred = ctx.predictor.predict(spec, w, ctx.model, ctx.device, draft=draft)
                    except Exception:  # noqa: BLE001
                        continue
                    table.setdefault(str(k), {}).setdefault(str(c), {})[sampling] = round(
                        pred.throughput_tok_s.value / base.throughput_tok_s.value, 3
                    )
        return table

    def stage1_output(self, ctx: PlanContext, frag: Fragment) -> Dict[str, Any]:
        return dict(getattr(self, "output", {}) or {})


# ====================================================================== state.dtype


@register
class StateDtype(Module):
    name = "state.dtype"
    version = "v2"
    stage = "numerics"
    after = ("spec_decode",)

    def applies(self, ctx: PlanContext) -> Tuple[bool, str]:
        if not ctx.model.layers.recurrent:
            return False, ""
        if "mamba_state_bf16" not in ctx.record.features:
            return False, f"the engine record {ctx.record.id} doesn't list mamba_state_bf16"
        verdict = _gate(ctx, "state.dtype=bfloat16")
        if not verdict.ok:
            return False, verdict.reason
        return True, ""

    def plan(self, ctx: PlanContext) -> Fragment:
        f = self.frag()
        f.set(
            "state.dtype",
            "bfloat16",
            self.name,
            "validated for the family: half the state, ~2x the sequence ceiling",
        )
        return f


# ====================================================================== kv.dtype (v2, numerics)


@register
class KVDtypeV2(Module):
    name = "kv.dtype"
    version = "v2"
    stage = "numerics"
    after = ("state.dtype",)

    def eligible(self, ctx: PlanContext) -> Tuple[bool, str]:
        if "kv_fp8" not in ctx.record.features:
            return False, f"the engine record {ctx.record.id} doesn't list kv_fp8"
        if ctx.model.layers.mla:
            return False, "MLA latents need vLLM's own fp8 format, not offered yet"
        cc = ctx.device.compute_capability or (0, 0)
        if ctx.device.device_class != "cuda" or tuple(cc) < (8, 0):
            return False, f"{ctx.device.sku} has no fp8 KV path"
        verdict = _gate(ctx, "kv.dtype=fp8")
        if not verdict.ok:
            return False, verdict.reason
        return True, ""

    def applies(self, ctx: PlanContext) -> Tuple[bool, str]:
        return (ctx.intent.enable_kv_fp8 is not False), (
            "fp8 KV off for this deployment" if ctx.intent.enable_kv_fp8 is False else ""
        )

    def plan(self, ctx: PlanContext) -> Fragment:
        f = self.frag()
        ok, why = self.eligible(ctx)
        requested = ctx.intent.enable_kv_fp8
        if not ok:
            if requested:
                f.decisions.append(f"fp8 KV asked for, not rendered: {why}")
            return f
        hybrid = bool(ctx.model.layers.recurrent)
        if requested is None:
            if hybrid and (ctx.intent.workload_profile or "") == "agentic":
                f.decisions.append(
                    "fp8 KV off by default on a hybrid with agentic traffic: it doubles the block "
                    "(784 -> 1568 tokens) and cut prefix hits to 0.87x in the study"
                )
                return f
            if not calibrated_kv_scales(ctx.model_config):
                f.decisions.append(
                    "fp8 KV not on by default: the checkpoint ships no calibrated KV scales (FRD-023 FR-QUANT-2)"
                )
                return f
        f.set(
            "kv.dtype",
            "fp8",
            self.name,
            "asked for at deploy time" if requested else "calibrated scales, validated",
        )
        if ctx.model.layers.sliding:
            f.arg(
                "kv-cache-dtype-skip-layers",
                "sliding_window",
                self.name,
                "sliding-window layers keep the model dtype",
            )
        return f
