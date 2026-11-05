"""budsim's call site: plan the KV tiers of a node group budsim chose (IMPLEMENTATION_PLAN 1.2r).

budsim passes its ``NodeGroupConfiguration``, the simulation row and the live ``cluster_info``; this
reads the deployment's fields, the target node and the cluster from them and runs :func:`plan_kv`.
Nothing here raises: a plan that can't be made is None, and the deployment renders as it does today
(NFR-3).
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Dict, Mapping, Optional, Sequence

from .dtype import kv_fp8_support
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


def _arg(args: Mapping[str, Any], *names: str) -> Any:
    """An engine argument as budsim's ``args`` holds it, under dashes or underscores."""
    for name in names:
        for key in (name, name.replace("-", "_")):
            if key in args:
                return args[key]
    return None


def _truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() in ("1", "true", "yes", "on")


def prefix_caching_on(args: Mapping[str, Any]) -> bool:
    """Whether the engine runs with prefix caching (vLLM's default is on)."""
    if _truthy(_arg(args, "no-enable-prefix-caching")):
        return False
    enabled = _arg(args, "enable-prefix-caching")
    return not (enabled is False or str(enabled).strip().lower() in ("false", "0", "no", "off"))


#: Attention backends that keep vLLM's paged KV cache in one shape, so a block one writes another reads (FRD-028
#: FR-MIX-13; the RTX 3050's FLASH_ATTN and the GTX 1650's TRITON_ATTN shared pool blocks on accubits, 2026-11-03).
PAGED_NHD_BACKENDS = frozenset({"FLASH_ATTN", "TRITON_ATTN", "FLEX_ATTENTION"})
PAGED_NHD = "paged-nhd"
#: NVIDIA cards, Volta to Hopper, on which vLLM's own pick is FLASH_ATTN (Ampere and newer) or TRITON_ATTN (older):
#: both paged-nhd. Blackwell's pick isn't pinned down, so it keeps a namespace of its own.
_PAGED_NHD_CARDS = re.compile(
    r"\b(V100|T4|TITAN|QUADRO|GTX ?16\d\d|RTX ?[234]0\d\d|RTX ?A\d{4}|A(2|10|16|30|40|100|800|6000)|"
    r"L(4|20|40)S?|H(20|100|200|800))\b",
    re.IGNORECASE,
)


#: Cards without bf16 (before Ampere): vLLM computes a bf16 model in float16 there.
_NO_BF16_CARDS = re.compile(r"\b(V100|T4|TITAN|QUADRO|GTX ?\d{3,4}|RTX ?20\d\d|P\d{2,3})\b", re.IGNORECASE)


def kv_layout_family(backend: str) -> str:
    """The KV layout an attention backend writes: one name for every backend of the paged-nhd family, else its own."""
    return PAGED_NHD if str(backend).upper() in PAGED_NHD_BACKENDS else str(backend)


def layout_facts(group: Any, model_config: Mapping[str, Any]) -> Dict[str, Optional[str]]:
    """What changes a KV block's bytes on this node group, for the namespace (FR-ISO-3).

    The dtype resolved as the engine will compute in; the KV layout of the attention backend (FR-MIX-13): the pinned
    backend's family, else ``auto@paged-nhd`` on a card whose own pick is known to be in that family, else
    ``auto@<card>``; the speculative config.
    """
    from .planner import resolve_dtype

    args = _get(group, "args") or {}
    envs = _get(group, "envs") or {}
    backend = envs.get("VLLM_ATTENTION_BACKEND") or _arg(args, "attention-backend")
    card = _get(group, "raw_name") or _get(group, "device_model") or _get(group, "name") or ""
    speculative = _arg(args, "speculative-config")
    if isinstance(speculative, Mapping):
        speculative = json.dumps(speculative, sort_keys=True)
    return {
        "resolved_dtype": _card_dtype(resolve_dtype(_arg(args, "dtype"), model_config), card),
        "attention_backend": kv_layout_family(backend) if backend else _auto_layout(card),
        "speculative": str(speculative) if speculative else None,
    }


def _card_dtype(dtype: str, card: str) -> str:
    """The dtype the card computes ``dtype`` in: bf16 falls back to float16 on a card without it."""
    if dtype == "bfloat16" and _NO_BF16_CARDS.search(str(card).replace("-", " ")):
        return "float16"
    return dtype


def _auto_layout(card: str) -> str:
    """The layout of the backend vLLM picks on ``card`` by itself: the paged family where known, else the card's own."""
    return f"auto@{PAGED_NHD}" if _PAGED_NHD_CARDS.search(str(card).replace("-", " ")) else f"auto@{card}"


def engine_version(group: Any) -> Optional[str]:
    """The engine build for the namespace: its upstream vLLM base (or Bud version) and image."""
    version = _get(group, "upstream_vllm") or _get(group, "engine_version")
    image = _get(group, "image")
    parts = [str(p) for p in (version, image) if p]
    return "|".join(parts) or None


def deployment_from_budsim(
    group: Any,
    row: Any,
    model_config: Mapping[str, Any],
    *,
    tool_calling: Optional[bool] = None,
    reasoning: Optional[bool] = None,
    kv_fp8: Optional[bool] = None,
    kv_fp8_hybrid_validated: bool = False,
    spans_nodes: Optional[bool] = None,
    autoscale_max_replicas: Optional[int] = None,
    cross_node_need: Optional[bool] = None,
) -> DeploymentSpec:
    """The deployment's own fields as budsim holds them.

    ``tool_calling`` / ``reasoning`` are the deployment's settings when the caller knows them
    (budcluster sends them at deploy time, FR-PLAN-3); otherwise the simulation row's, which budapp
    doesn't fill.
    """
    ttft = _num(_get(row, "target_ttft"))
    labels = _get(group, "labels") or {}
    e2e = _num(_get(group, "e2e_latency"))
    spread = bool(_get(group, "spans_nodes")) if spans_nodes is None else bool(spans_nodes)
    peak = _num(autoscale_max_replicas)
    if peak is None:
        peak = _num(_get(group, "autoscale_max_replicas"))
    layout = layout_facts(group, model_config)
    return DeploymentSpec(
        model_config=model_config,
        input_tokens=int(_get(row, "input_tokens", 0) or 0),
        output_tokens=int(_get(row, "output_tokens", 0) or 0),
        concurrency=int(_num(labels.get("concurrency")) or 1),
        model_id=_get(row, "model_name"),
        target_ttft_ms=ttft if ttft and ttft > 0 else None,
        tool_calling=(
            bool(_get(row, "enable_tool_calling", False))
            if tool_calling is None
            else bool(tool_calling)
        ),
        reasoning=(
            bool(_get(row, "enable_reasoning", False)) if reasoning is None else bool(reasoning)
        ),
        is_embedding=bool(_get(group, "is_embedding")),
        e2e_latency_s=e2e if e2e and e2e > 0 else None,
        kv_fp8=kv_fp8,
        kv_fp8_hybrid_validated=bool(kv_fp8_hybrid_validated),
        revision=_get(row, "model_revision") or _get(row, "revision"),
        weight_quantization=_get(row, "quantization") or _get(group, "quantization"),
        spans_nodes=spread,
        autoscale_max_replicas=int(peak) if peak and peak > 0 else None,
        cross_node_need=bool(cross_node_need),
        prefix_caching=prefix_caching_on(_get(group, "args") or {}),
        **layout,
    )


def kv_fp8_option(
    engine_type: Optional[str],
    kv_features: Any,
    model_config: Mapping[str, Any],
    *,
    hybrid_validated: bool = False,
) -> Optional[Dict[str, Any]]:
    """The deployment form's fp8 KV option for one simulation result (``dtype.kv_fp8_support``), or
    None for an engine the planner doesn't plan (not vLLM)."""
    if str(engine_type or "") != "vllm":
        return None
    return kv_fp8_support(
        model_config or {}, frozenset(kv_features or ()), hybrid_validated=hybrid_validated
    )


def plan_node_group(
    group: Any,
    row: Any,
    *,
    cluster_info: Optional[Sequence[Mapping[str, Any]]],
    model_config: Optional[Mapping[str, Any]] = None,
    load_model_config: Any = None,
    tool_calling: Optional[bool] = None,
    reasoning: Optional[bool] = None,
    kv_fp8: Optional[bool] = None,
    kv_fp8_hybrid_validated: bool = False,
    spans_nodes: Optional[bool] = None,
    autoscale_max_replicas: Optional[int] = None,
    cross_node_need: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """Plan one node group and return the ``kv_plan`` wire dict, or None when no plan can be made.

    ``model_config`` is loaded with ``load_model_config(model_name)`` only when the plan can use it
    (a vLLM engine with KV features, and not an embedding model). ``tool_calling`` / ``reasoning``
    are the deployment's settings when known; None reads them from the row and notes that tool
    calling is unknown. ``kv_fp8`` is the deployment form's fp8 KV choice (None: not asked).
    ``spans_nodes`` (default: the group's field) says its replicas may land on several nodes (FR-PLACE-2);
    ``autoscale_max_replicas`` is the most the autoscaler runs (FR-PLAN-19). ``cross_node_need`` is set by a caller
    that knows the deployment spans nodes beyond this group: a mix whose groups share one KV namespace
    (FRD-028 FR-MIX-12).
    """
    try:
        engine = EngineSpec.of(
            str(_get(group, "engine_type") or ""),
            _get(group, "kv_features"),
            version=engine_version(group),
        )
        if model_config is None:
            usable = (
                engine.name == "vllm"
                and engine.kv_features
                and not bool(_get(group, "is_embedding"))
            )
            model_config = (
                load_model_config(_get(row, "model_name")) if usable and load_model_config else {}
            )
        deployment = deployment_from_budsim(
            group,
            row,
            model_config or {},
            tool_calling=tool_calling,
            reasoning=reasoning,
            kv_fp8=kv_fp8,
            kv_fp8_hybrid_validated=kv_fp8_hybrid_validated,
            spans_nodes=spans_nodes,
            autoscale_max_replicas=autoscale_max_replicas,
            cross_node_need=cross_node_need,
        )
        node_group = NodeGroup.from_object(group, row)
        target = _get(group, "target_node_name")
        infra = InfraSnapshot.from_cluster_info(
            cluster_info,
            _get(row, "cluster_id"),
            None if deployment.spans_nodes else target,
            spread_model_key=node_group.device_model_key if deployment.spans_nodes else None,
        )
        plan = plan_kv(deployment, engine, node_group, infra)
        if (
            tool_calling is None
            and deployment.tool_calling is False
            and plan.workload_profile != "off"
        ):
            plan.decisions.insert(
                1, "tool calling: not known to budsim; the agentic profile isn't inferred"
            )
        return plan.to_wire()
    except Exception:  # noqa: BLE001 - a planner failure must never fail a deployment
        logger.exception("KV plan failed for node group %s", _get(group, "name", "?"))
        return None
