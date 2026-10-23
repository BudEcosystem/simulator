"""The two-stage plan for one node group (FRD-027 §10.0), as pure functions budsim calls.

``numerics(...)`` runs stage 1 once per candidate device group, before budsim's search, and returns the model
profile the search sizes and predicts with (the chosen weight variant's facts, the stage-1 settings).
``spec(...)`` runs stage 2 on the placement the search chose and composes the runtime spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from ..device import DeviceFacts
from ..engine.record import EngineRecord
from ..model import ModelFacts
from ..predict import Predictor
from .compose import Composition, assemble, order, run_stage
from .context import Intent, Placement, PlanContext
from .modules import PLANNER, build
from .spec import RuntimeSpec


@dataclass
class Numerics:
    """Stage 1's result: the profile the search uses."""

    composition: Composition
    settings: Dict[str, Any]
    model: ModelFacts  # the facts of the checkpoint that will be served (a variant's, when chosen)
    model_path: Optional[str]
    decisions: List[str]
    stage1: Dict[str, Any] = field(default_factory=dict)
    versions: Dict[str, str] = field(default_factory=dict)

    def to_wire(self) -> Dict[str, Any]:
        """What budsim persists with a search result (``top_k_configs`` JSONB): no live objects."""
        stage1 = {}
        for name, out in self.stage1.items():
            stage1[name] = {k: v for k, v in (out or {}).items() if k != "facts"}
        return {
            "settings": dict(self.settings),
            "model_path": self.model_path,
            "decisions": list(self.decisions),
            "stage1": stage1,
            "module_versions": dict(self.versions),
        }


def context(
    intent: Intent,
    model: ModelFacts,
    model_config: Mapping[str, Any],
    record: EngineRecord,
    device: DeviceFacts,
    *,
    predictor: Optional[Predictor] = None,
    placement: Optional[Placement] = None,
    versions: Optional[Mapping[str, str]] = None,
    validation: Any = None,
    drafts: Optional[Mapping[str, ModelFacts]] = None,
    node: Optional[Mapping[str, Any]] = None,
    kv_plan: Optional[Mapping[str, Any]] = None,
    cluster: Optional[Mapping[str, Any]] = None,
) -> PlanContext:
    return PlanContext(
        stage="numerics",
        intent=intent,
        model=model,
        model_config=dict(model_config or {}),
        record=record,
        device=device,
        predictor=predictor or Predictor(record),
        placement=placement,
        node=dict(node or {}),
        kv_plan=dict(kv_plan) if kv_plan else None,
        module_versions=dict(versions or PLANNER),
        validation=validation,
        drafts=dict(drafts or {}),
        cluster=dict(cluster or {}),
    )


def numerics(ctx: PlanContext) -> Numerics:
    modules = build(ctx.module_versions)
    ctx.stage = "numerics"
    comp = run_stage("numerics", ctx, modules)
    precision = comp.stage1.get("weights.precision") or {}
    facts = precision.get("facts") or ctx.model
    return Numerics(
        composition=comp,
        settings={k: v.value for k, v in comp.settings.items()},
        model=facts,
        model_path=precision.get("path"),
        decisions=list(comp.decisions),
        stage1=dict(comp.stage1),
        versions=dict(ctx.module_versions),
    )


def spec(ctx: PlanContext, stage1: Optional[Numerics] = None) -> RuntimeSpec:
    """Stage 2 on ``ctx.placement`` after ``stage1`` (run here when not given), composed into the spec."""
    modules = build(ctx.module_versions)
    if stage1 is None:
        stage1 = numerics(ctx)
    ctx.model = stage1.model
    comp = stage1.composition
    ctx.stage = "capacity"
    comp = run_stage("capacity", ctx, modules, comp)
    for module in order([m for m in modules if m.stage == "capacity"]):
        post = getattr(module, "post_compose", None)
        if callable(post):
            ctx.draft = comp.draft()
            post(ctx, comp)
    return assemble(ctx, comp)
