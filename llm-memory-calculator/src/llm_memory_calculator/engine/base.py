"""Layer 2 of the FRD-027 predictor: the engine-model contract (FR-ENG-5).

An ``EngineModel`` turns model facts, device facts and engine-neutral settings into what one engine build
will allocate, admit and run: ``limits`` (memory, sequence ceiling, bootable context, boot checks),
``predict`` (TTFT, TPOT, throughput with bands and the grounding levels used), ``supports`` (the unary
check), ``validate`` (the pairwise check over the record's ``conflicts``) and ``boot_time``.
"""

from __future__ import annotations

import math
import re

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol, Tuple

Settings = Mapping[str, Any]


class UnsupportedSetting(ValueError):
    """A setting the engine record does not map, or a value whose feature the record does not list."""


@dataclass(frozen=True)
class Estimate:
    """A point prediction with its error band."""

    value: float
    low: float
    high: float
    ungrounded: bool = False
    #: the stratum's measured replicate spread (0 when unmeasured), and whether it exceeds the ceiling: no prediction
    #: meets the ceiling on every run there, so A-05 grades the point on coverage only (FRD-027 §16). Reporting only:
    #: a band past twice the ceiling stays ungrounded either way (NFR-5).
    noise: float = 0.0
    noise_limited: bool = False

    @property
    def width(self) -> float:
        """Relative half-width of the band."""
        if not self.value:
            return 0.0
        return max(self.high - self.value, self.value - self.low) / abs(self.value)

    @classmethod
    def around(
        cls, value: float, rel: float, *, ceiling: Optional[float] = None, noise: float = 0.0
    ) -> "Estimate":
        rel = max(float(rel), 0.0)
        ungrounded = ceiling is not None and rel > 2.0 * ceiling
        noise = max(float(noise), 0.0)
        return cls(
            value=float(value),
            low=float(value) * (1 - min(rel, 0.95)),
            high=float(value) * (1 + rel),
            ungrounded=ungrounded,
            noise=noise,
            noise_limited=ceiling is not None and noise > ceiling,
        )

    def separates_above(self, other: "Estimate") -> bool:
        """True when this estimate's band lies entirely above ``other``'s (FR-GND-2 comparison)."""
        return not self.ungrounded and not other.ungrounded and self.low > other.high

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": round(self.value, 4),
            "low": round(self.low, 4),
            "high": round(self.high, 4),
            "ungrounded": self.ungrounded,
            "noise": round(self.noise, 4),
            "noise_limited": self.noise_limited,
        }


def draft_key(path: str) -> str:
    """A speculative draft's grounding identity: its hub org and repo, lowercased with separators collapsed. A plan names
    a draft by hub id (``RedHatAI/Qwen3-8B-speculator.eagle3``); the study staged drafts as
    ``/drafts/eagle3-RedHatAI_Qwen3-8B-speculator.eagle3``. Both give the same key."""
    parts = [p for p in str(path).strip().rstrip("/").split("/") if p]
    if not parts:
        return ""
    last = parts[-1]
    staged = last.lower().startswith("eagle3-") and "_" in last
    if staged and (len(parts) == 1 or parts[-2].lower() == "drafts"):
        ident = last[len("eagle3-"):]
    elif len(parts) >= 2:
        ident = "/".join(parts[-2:])
    else:
        ident = last
    return re.sub(r"[/_\-]+", "-", ident.lower()).strip("-")


@dataclass(frozen=True)
class Workload:
    """A text-generation workload (FRD-027 §10.1). Other tasks extend this shape later."""

    input_tokens: float
    output_tokens: float
    concurrency: int
    task: str = "chat"
    input_tokens_max: Optional[int] = None
    output_tokens_max: Optional[int] = None
    prefix_tokens: int = 0  # shared prefix length a request can hit in the cache
    prefix_hit_rate: Optional[float] = (
        None  # measured hit rate, when known (overrides prefix_tokens)
    )
    sampling: str = "sampled"  # greedy | sampled
    workload_class: str = "real"  # real | synthetic (random-token benchmark prompts)
    arrival: Optional[str] = (
        None  # steady (independent users) | wave (synchronized identical requests); None: by class
    )
    # a fixed-count benchmark of this many requests: the run lasts until its last request finishes, so its throughput
    # over the duration drains below the open-ended rate (arrival.drain); None for a plan's open-ended traffic
    requests: Optional[int] = None

    def digest_key(self) -> Tuple[Any, ...]:
        return (
            round(float(self.input_tokens), 1),
            round(float(self.output_tokens), 1),
            int(self.concurrency),
            self.task,
            self.input_tokens_max,
            self.output_tokens_max,
            self.prefix_tokens,
            None if self.prefix_hit_rate is None else round(self.prefix_hit_rate, 4),
            self.sampling,
            self.workload_class,
            self.arrival,
            self.requests,
        )

    @property
    def shape(self) -> str:
        """The load's shape, which keys the speculative TTFT terms: real text, a shared-prefix load, a long prompt, a
        decode-heavy load, or chat. Each desynchronizes a benchmark wave differently (27B MTP: the decode wave keeps half
        of its burst, the long one none of it)."""
        if self.workload_class == "real":
            return "real"
        if self.prefix_tokens or (self.prefix_hit_rate or 0.0) > 0.0:
            return "prefix"
        if self.input_tokens >= 4096:
            return "long"
        if self.output_tokens > self.input_tokens:
            return "decode"
        return "chat"

    def shape_for(self, prefix_cache: bool) -> str:
        """The shape on an engine whose prefix cache is on or off. A shared-prefix load cannot hit with the cache off (MTP
        plans turn it off, vLLM #53912), but it is not a random long load either: its repeated text decodes with higher
        speculative acceptance and the requests desynchronize (0.30 27B MTP k=2, 4,608 tokens c16: 671 ms against 1,031 ms
        for random prompts), so it is ``prefix-cold`` (T-GND-25)."""
        if prefix_cache or self.shape != "prefix":
            return self.shape
        return "prefix-cold"

    @property
    def arrival_model(self) -> str:
        """``wave`` for a benchmark of identical requests (they arrive and finish together, so each wave's
        prefills burst), ``steady`` for real traffic, whose requests desynchronize."""
        if self.arrival:
            return self.arrival
        return "wave" if self.workload_class == "synthetic" else "steady"


@dataclass(frozen=True)
class BootCheck:
    check: str
    ok: bool
    margin_bytes: float = 0.0
    message: str = ""


@dataclass(frozen=True)
class Refusal:
    setting: str
    reason: str
    kind: str = "conflict"  # conflict | unsupported | boot
    conflict: Optional[str] = None


@dataclass
class Limits:
    """What the engine will allocate and refuse for one configuration (FR-ENG-1)."""

    budget_bytes: float
    weights_bytes: float
    overhead_bytes: float
    kv_pool_bytes: float
    block_tokens: int
    block_bytes: float  # one block across every layer slot
    num_blocks: int
    kv_tokens: int
    sequence_ceiling: int
    state_bytes_per_sequence: float
    max_bootable_context: int
    profiling_transient_bytes: float
    checks: List[BootCheck] = field(default_factory=list)
    attn_groups: int = (
        1  # attention KV-cache groups; each running sequence holds its blocks in every one
    )
    # windowed KV-cache groups (vLLM's hybrid allocator: a stack of full and sliding layers); each running sequence
    # also holds its blocks in every one of these, up to ``sliding_group_blocks`` each (0: no window read, unbounded)
    sliding_groups: int = 0
    sliding_group_blocks: int = 0
    memory_band_bytes: float = 0.0
    # the most sequences whose CUDA-graph profiling allocation fits (None: the whole capture fits, or none is made)
    transient_ceiling: Optional[int] = None
    levels: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    resolved: Dict[str, Any] = field(
        default_factory=dict
    )  # the effective value of every setting, defaults filled

    def blocks_per_sequence(self, tokens: float) -> float:
        """Blocks one running sequence of ``tokens`` holds: its blocks in every attention group, in every windowed
        group (at most the window's), and its fixed recurrent state."""
        blocks = math.ceil(max(float(tokens), 0.0) / max(self.block_tokens, 1))
        per = max(self.attn_groups, 1) * blocks
        if self.sliding_groups:
            cap = self.sliding_group_blocks
            per += self.sliding_groups * (min(blocks, cap) if cap else blocks)
        if self.state_bytes_per_sequence and self.block_bytes:
            per += self.state_bytes_per_sequence / self.block_bytes
        return per

    @property
    def boots(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def failures(self) -> List[BootCheck]:
        return [c for c in self.checks if not c.ok]

    def margin(self, check: str) -> Optional[float]:
        for c in self.checks:
            if c.check == check:
                return c.margin_bytes
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kv_pool_bytes": int(self.kv_pool_bytes),
            "kv_tokens": int(self.kv_tokens),
            "block_tokens": self.block_tokens,
            "num_blocks": self.num_blocks,
            "sequence_ceiling": self.sequence_ceiling,
            "max_bootable_context": self.max_bootable_context,
            "profiling_transient_bytes": int(self.profiling_transient_bytes),
            "transient_ceiling": self.transient_ceiling,
            "weights_bytes": int(self.weights_bytes),
            "boots": self.boots,
            "checks": [
                {
                    "check": c.check,
                    "ok": c.ok,
                    "margin_bytes": int(c.margin_bytes),
                    "message": c.message,
                }
                for c in self.checks
            ],
            "memory_band_bytes": int(self.memory_band_bytes),
        }


@dataclass
class Prediction:
    ttft_ms: Estimate
    tpot_ms: Estimate
    throughput_tok_s: Estimate  # output tokens per second, whole replica
    e2e_s: Estimate
    running: int  # sequences decoded concurrently
    queueing: bool  # batch.max_running (or KV) binds below the concurrency
    limits: Limits
    levels: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def ungrounded(self) -> bool:
        return any(e.ungrounded for e in (self.ttft_ms, self.tpot_ms, self.throughput_tok_s))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ttft_ms": self.ttft_ms.to_dict(),
            "tpot_ms": self.tpot_ms.to_dict(),
            "throughput_tok_s": self.throughput_tok_s.to_dict(),
            "e2e_s": self.e2e_s.to_dict(),
            "running": self.running,
            "queueing": self.queueing,
            "levels": dict(self.levels),
            "notes": list(self.notes),
        }


class EngineModel(Protocol):
    record: Any

    def supports(self, setting: str) -> bool: ...

    def validate(self, settings: Settings, model: Any = None) -> List[Refusal]: ...

    def limits(
        self, settings: Settings, model: Any, device: Any, *, draft: Any = None
    ) -> Limits: ...

    def predict(
        self, settings: Settings, workload: Workload, model: Any, device: Any, *, draft: Any = None
    ) -> Prediction: ...

    def boot_time(self, settings: Settings, model: Any, device: Any) -> Estimate: ...
