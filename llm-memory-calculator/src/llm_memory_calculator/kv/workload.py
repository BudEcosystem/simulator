"""The deployment as the planner sees it, and the reuse it assumes (FR-PLAN-3).

Every input is a field the deployment already has. The reuse profile and its assumptions are
inferred from those fields; there is no KV option on a deployment.

**Working set.** Estimated analytically from per-profile reuse assumptions (``PROFILE_REUSE``):
sessions alive per concurrency slot times the reusable part of each prompt. GenZ's ``RadixCache``
can't model it yet (it stops inserting when full instead of evicting), so a trace simulation is a
later refinement. The assumptions are defaults to be measured, and measured hit rates don't feed
back yet (FRD §12).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .constants import INTERACTIVE_TTFT_MS, PREFIX_HEAVY_RATIO, PROFILE_REUSE, PROFILE_USES


@dataclass(frozen=True)
class DeploymentSpec:
    """A deployment's own fields (FR-PLAN-3), as budsim receives them from budapp."""

    model_config: Mapping[str, Any]
    input_tokens: int
    output_tokens: int
    concurrency: int
    model_id: Optional[str] = None  # the HF id, for GenZ and the KV namespace
    revision: Optional[str] = None  # weights checksum or revision, for the KV namespace
    weight_quantization: Optional[str] = None
    target_ttft_ms: Optional[float] = None
    tool_calling: bool = False
    reasoning: bool = False
    is_embedding: bool = False
    lora_adapters: int = 0
    # budsim's predicted end-to-end latency for one request, seconds; sets the request rate the
    # load model uses. Estimated from the prefill and DECODE_MS_PER_TOKEN when absent.
    e2e_latency_s: Optional[float] = None
    # Why the cluster tier would be needed at all (FR-T3-1): replicas on several nodes, other
    # deployments of the same model in the project, or durability across node loss.
    cross_node_need: bool = False
    autoscale_max_replicas: Optional[int] = None


@dataclass(frozen=True)
class KVWorkload:
    """The reuse the planner assumes, inferred from the deployment's own fields (FR-PLAN-3)."""

    profile: str  # off | agentic | chat | interactive | rag
    input_tokens: int
    output_tokens: int
    concurrency: int
    sessions_per_slot: float = 0.0
    reusable_share: float = 0.0
    reason: str = ""
    # Uses of a reusable prefix over its life; its first use misses. 0: unknown, no adjustment.
    uses_per_prefix: float = 0.0

    @classmethod
    def from_deployment(
        cls,
        *,
        input_tokens: int,
        output_tokens: int,
        concurrency: int,
        is_embedding: bool = False,
        enable_tool_calling: bool = False,
        target_ttft_ms: Optional[float] = None,
    ) -> "KVWorkload":
        """Infer the reuse profile from the deployment's own fields."""
        input_tokens = max(int(input_tokens or 0), 0)
        output_tokens = max(int(output_tokens or 0), 0)
        concurrency = max(int(concurrency or 1), 1)
        if is_embedding:
            profile, reason = "off", "embedding or pooling model: no reusable KV cache"
        elif enable_tool_calling:
            profile, reason = (
                "agentic",
                "tool calling on: tool definitions and history are a shared prefix",
            )
        elif output_tokens and input_tokens / output_tokens >= PREFIX_HEAVY_RATIO:
            profile, reason = "rag", f"input:output {input_tokens}:{output_tokens} is prefix-heavy"
        elif target_ttft_ms is not None and target_ttft_ms <= INTERACTIVE_TTFT_MS:
            profile, reason = "interactive", f"TTFT target {target_ttft_ms:g} ms"
        else:
            profile, reason = "chat", "default profile"
        sessions, share = PROFILE_REUSE.get(profile, (0.0, 0.0))
        return cls(
            profile,
            input_tokens,
            output_tokens,
            concurrency,
            sessions,
            share,
            reason,
            uses_per_prefix=PROFILE_USES.get(profile, 0.0),
        )

    @classmethod
    def from_spec(cls, spec: DeploymentSpec) -> "KVWorkload":
        return cls.from_deployment(
            input_tokens=spec.input_tokens,
            output_tokens=spec.output_tokens,
            concurrency=spec.concurrency,
            is_embedding=spec.is_embedding,
            enable_tool_calling=spec.tool_calling,
            target_ttft_ms=spec.target_ttft_ms,
        )

    @property
    def repeat_share(self) -> float:
        """Share of a prefix's uses that come after its first (the ones a cache can serve)."""
        return 1.0 - 1.0 / self.uses_per_prefix if self.uses_per_prefix >= 1 else 1.0

    @property
    def reusable_prefix_tokens(self) -> int:
        """Tokens of a typical prompt that another request can reuse (the p50 prefix)."""
        return int(self.input_tokens * self.reusable_share)

    @property
    def working_set_tokens(self) -> int:
        """Distinct reusable tokens one replica should keep to serve its reuse from cache."""
        return int(self.concurrency * self.sessions_per_slot * self.reusable_prefix_tokens)
