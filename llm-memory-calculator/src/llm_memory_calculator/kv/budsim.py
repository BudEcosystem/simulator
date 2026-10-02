"""budsim's call site: plan the KV tiers of a node group budsim chose (IMPLEMENTATION_PLAN 1.2r).

budsim passes its ``NodeGroupConfiguration``, the simulation row and the live ``cluster_info``; this
reads the deployment's fields, the target node and the cluster from them and runs :func:`plan_kv`.
Nothing here raises: a plan that can't be made is None, and the deployment renders as it does today
(NFR-3).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Mapping, Optional, Sequence

from .facts import EngineSpec
from .group import NodeGroup
from .infra import InfraSnapshot
from .planner import plan_kv
from .workload import DeploymentSpec

logger = logging.getLogger(__name__)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def deployment_from_budsim(group: Any, row: Any, model_config: Mapping[str, Any]) -> DeploymentSpec:
    """The deployment's own fields as budsim holds them."""
    ttft = _num(_get(row, "target_ttft"))
    labels = _get(group, "labels") or {}
    e2e = _num(_get(group, "e2e_latency"))
    return DeploymentSpec(
        model_config=model_config,
        input_tokens=int(_get(row, "input_tokens", 0) or 0),
        output_tokens=int(_get(row, "output_tokens", 0) or 0),
        concurrency=int(_num(labels.get("concurrency")) or 1),
        model_id=_get(row, "model_name"),
        target_ttft_ms=ttft if ttft and ttft > 0 else None,
        tool_calling=bool(_get(row, "enable_tool_calling", False)),
        reasoning=bool(_get(row, "enable_reasoning", False)),
        is_embedding=bool(_get(group, "is_embedding")),
        e2e_latency_s=e2e if e2e and e2e > 0 else None,
    )


def plan_node_group(
    group: Any,
    row: Any,
    *,
    cluster_info: Optional[Sequence[Mapping[str, Any]]],
    model_config: Optional[Mapping[str, Any]] = None,
    load_model_config: Any = None,
) -> Optional[Dict[str, Any]]:
    """Plan one node group and return the ``kv_plan`` wire dict, or None when no plan can be made.

    ``model_config`` is loaded with ``load_model_config(model_name)`` only when the plan can use it
    (a vLLM engine with KV features, and not an embedding model).
    """
    try:
        engine = EngineSpec.of(str(_get(group, "engine_type") or ""), _get(group, "kv_features"))
        if model_config is None:
            usable = (
                engine.name == "vllm"
                and engine.kv_features
                and not bool(_get(group, "is_embedding"))
            )
            model_config = (
                load_model_config(_get(row, "model_name")) if usable and load_model_config else {}
            )
        deployment = deployment_from_budsim(group, row, model_config or {})
        infra = InfraSnapshot.from_cluster_info(
            cluster_info, _get(row, "cluster_id"), _get(group, "target_node_name")
        )
        plan = plan_kv(deployment, engine, NodeGroup.from_object(group, row), infra)
        if deployment.tool_calling is False and plan.workload_profile != "off":
            plan.decisions.insert(
                1, "tool calling: not known to budsim; the agentic profile isn't inferred"
            )
        return plan.to_wire()
    except Exception:  # noqa: BLE001 - a planner failure must never fail a deployment
        logger.exception("KV plan failed for node group %s", _get(group, "name", "?"))
        return None
