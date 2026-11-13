"""Sizing one Bud-Decision-Runtime replica for a decisions workload (Bud spec 031 FRD §5.16.3, C10.1).

A call is ``Q`` questions over a state of ``S`` tokens, each question ``q`` tokens (its options included). The
runtime packs the rows of many calls into passes, so a replica is a queue in front of one throughput:

    work per call          W = the call's tokens (per family layout, ``call_work``)
    throughput             R_tok = eff x peak FLOP/s / FLOPs per token of the call's rows
    one call alone         t_1 = max(W / R_tok, t_pass_min)
    calls within target T  k = floor(0.8 x R_tok x T / W)     (Little's law on a saturated queue; 0.8 keeps it
                                                              out of its knee)

Measured on the RTX 3050: throughput is flat from 16 to 1,024 closed-loop clients and latency is all queue, so a call
with ``C`` calls in flight takes about ``C x W / R_tok``. A device on which one call already takes longer than ``T`` is
refused with the reason.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from . import calibration
from .families import BRANCH, ONE_SEQUENCE, DecisionFacts
from .precision import FLOAT32, PRECISIONS, WEIGHT_BYTES, choose, is_cpu

#: the queue's utilisation ceiling (FRD §5.16.3)
RHO_MAX = 0.8
#: the runtime's pass-size ceiling (``crates/sched`` ``Limits::max_pass_tokens``); a row longer than it raises it
MAX_PASS_TOKENS = 16_384
#: CPU compute threads beyond this scale badly (runtime PERFORMANCE.md §4.5): more load gets more replicas
MAX_CPU_THREADS = 16
#: activation memory over the analytic figure until the benchmark measures peak memory (runtime v0.1.1)
ACTIVATION_MARGIN = 1.5
#: the CUDA context, cuBLAS / Triton workspaces and the allocator's slack, until measured
DEFAULT_CUDA_CONTEXT_BYTES = 700 * 2**20
#: the Python process, torch, the tokenizer and its 4M-token caches on a CPU replica
DEFAULT_CPU_PROCESS_BYTES = 1536 * 2**20
#: Kev's state cache when the deployment names none (basal_engine.models.kev.STATE_CACHE_MB)
DEFAULT_STATE_CACHE_MB = 2048
#: the runtime's own pass-time ceiling for its automatic target (``max_auto_pass_ms``)
RUNTIME_MAX_AUTO_PASS_MS = 300.0


@dataclass
class DecisionPlan:
    precision: str
    precision_reason: str
    tokens_per_second: float
    single_call_ms: float
    call_latency_ms: float
    calls_per_replica: int
    threads: Optional[int]
    weight_bytes: int
    activation_bytes: int
    state_bytes: int
    overhead_bytes: int
    basis: str
    calibration: Optional[str]
    refused: Optional[str]
    call_tokens: int = 0
    row_tokens: int = 0
    max_pass_tokens: int = MAX_PASS_TOKENS
    target_call_latency_ms: Optional[float] = None
    limits: Dict[str, float] = field(default_factory=dict)

    @property
    def total_bytes(self) -> int:
        return int(self.weight_bytes + self.activation_bytes + self.state_bytes + self.overhead_bytes)

    @property
    def calls_per_second(self) -> float:
        if not self.call_tokens or self.tokens_per_second <= 0:
            return 0.0
        return self.tokens_per_second / self.call_tokens

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["total_bytes"] = self.total_bytes
        out["calls_per_second"] = self.calls_per_second
        return out


def call_work(facts: DecisionFacts, workload: Mapping[str, Any]) -> Tuple[int, float, int, Optional[str]]:
    """(tokens, FLOPs, longest row, refusal) of one call of ``workload`` on the family's layout."""
    q_count = max(int(workload.get("questions_per_call") or 1), 1)
    state = max(int(workload.get("state_tokens") or 0), 0)
    question = max(int(workload.get("question_tokens") or 1), 1)
    if facts.layout == ONE_SEQUENCE:
        row = state + q_count * question
        if row > facts.max_len:
            return 0, 0.0, row, (
                f"a {facts.family} call reads its {q_count} questions and the state as one sequence of {row:,} tokens, "
                f"over the family's {facts.max_len:,}-token limit: the runtime refuses it"
            )
        return row, row * facts.flops_per_token(row), row, None
    if facts.layout == BRANCH:
        row = state + question
        if row > facts.max_len:
            return 0, 0.0, row, f"a state of {state:,} tokens is over the {facts.family} family's limit"
        tokens = state + q_count * question
        flops = facts.linear_flops_per_token() * tokens
        flops += state * facts.attention_flops_per_token(state, context=state / 2.0)
        flops += q_count * question * facts.attention_flops_per_token(row, context=state + question / 2.0)
        return tokens, flops, row, None
    row = min(state + question, facts.max_len)
    tokens = q_count * row
    return tokens, tokens * facts.flops_per_token(row), row, None


def resolve_hardware(device: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """The device's hardware-table entry, from what budsim passes: ``hardware`` itself, else a match on its name."""
    given = device.get("hardware")
    if isinstance(given, Mapping) and given:
        return given
    if is_cpu(device):
        return None
    try:
        from ..hardware.configs import HARDWARE_CONFIGS
        from ..hardware.device_matcher import match_device
    except ImportError:  # pragma: no cover - the package always ships both
        return None
    name = device.get("raw_name") or device.get("name") or device.get("device_name") or device.get("product_name")
    if not name:
        return None
    info: Dict[str, Any] = {"raw_name": str(name)}
    mem = device.get("mem_per_GPU_in_GB") or device.get("mem_per_gpu_in_gb") or device.get("memory_gb")
    try:
        if mem:
            info["memory_mb"] = float(mem) * 1024
    except (TypeError, ValueError):
        pass
    for key in ("pci_vendor", "pci_device"):
        if device.get(key):
            info[key] = device[key]
    try:
        return match_device(info, HARDWARE_CONFIGS)
    except Exception:  # noqa: BLE001 - an unmatched name is an unknown device, never a failed plan
        return None


def cpu_threads(device: Mapping[str, Any], threads: Optional[int] = None) -> int:
    """Compute threads for one CPU replica: as asked, else min(16, the physical cores of one NUMA node)."""
    if threads:
        return max(int(threads), 1)
    physical = _int(device.get("physical_cores"))
    if not physical:
        logical = _int(device.get("cores") or device.get("cpu_cores"))
        physical = logical // 2 if logical and logical > 1 else logical
    numa = _int(device.get("numa_nodes")) or 1
    per_node = (physical // max(numa, 1)) if physical else MAX_CPU_THREADS
    return max(1, min(MAX_CPU_THREADS, per_node or 1))


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def admission_limits(target_call_latency_ms: Optional[float]) -> Dict[str, float]:
    """The runtime's ``[limits]`` for a call target T (D-36): queue no call longer than T, and keep passes to T / 2
    when that is under the runtime's own 300 ms ceiling."""
    if not target_call_latency_ms:
        return {}
    t = float(target_call_latency_ms)
    limits = {"interactive_queue_ms": round(t, 3)}
    if t / 2.0 < RUNTIME_MAX_AUTO_PASS_MS:
        limits["max_auto_pass_ms"] = round(t / 2.0, 3)
    return limits


def size(
    facts: DecisionFacts,
    device: Mapping[str, Any],
    *,
    workload: Mapping[str, Any],
    precision: Optional[str] = None,
    threads: Optional[int] = None,
    mode: str = "dedicated",
    state_cache_mb: Optional[float] = None,
    memory_budget_bytes: Optional[float] = None,
) -> DecisionPlan:
    """Size one replica of ``facts`` on ``device`` for ``workload`` (DM3's decisions block). Never raises: a failure
    is a plan with ``refused`` set and ``calls_per_replica`` 0.

    ``memory_budget_bytes`` is what the caller's memory gate leaves the replica's weights, activations and state.
    When the runtime's default pass ceiling does not fit in it, the plan lowers ``[limits] max_pass_tokens`` until it
    does, never below the longest row (the runtime raises its ceiling to that row itself), and refuses below that.
    """
    try:
        return _size(facts, device, workload=workload, precision=precision, threads=threads, mode=mode,
                     state_cache_mb=state_cache_mb, memory_budget_bytes=memory_budget_bytes)
    except Exception as exc:  # noqa: BLE001 - the planner reports a refusal, it never fails a simulation
        return DecisionPlan(
            precision=precision or FLOAT32, precision_reason="", tokens_per_second=0.0, single_call_ms=0.0,
            call_latency_ms=0.0, calls_per_replica=0, threads=None, weight_bytes=0, activation_bytes=0,
            state_bytes=0, overhead_bytes=0, basis=calibration.ESTIMATED, calibration=None,
            refused=f"the decision plan could not be computed: {exc}",
        )


def _size(facts, device, *, workload, precision, threads, mode, state_cache_mb, memory_budget_bytes) -> DecisionPlan:
    cpu = is_cpu(device)
    hardware = resolve_hardware(device)
    chosen, reason = choose(device, hardware)
    if precision:
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, got '{precision}'")
        if precision != chosen:
            reason = f"{precision} as asked ({chosen} is what the runtime would choose: {reason})"
        chosen = precision
    n_threads = cpu_threads(device, threads) if cpu else None
    target = workload.get("target_call_latency_ms")
    target = float(target) if target not in (None, "", 0) else None

    tokens, flops, row, refusal = call_work(facts, workload)
    entry, eff, basis = calibration.lookup(device, hardware, chosen, facts.family, facts.kind)
    peak = calibration.peak_flops(device, hardware, chosen, n_threads)

    memory = _memory(facts, cpu=cpu, precision=chosen, entry=entry, state_cache_mb=state_cache_mb, row=row)
    plan = DecisionPlan(
        precision=chosen, precision_reason=reason, tokens_per_second=0.0, single_call_ms=0.0, call_latency_ms=0.0,
        calls_per_replica=0, threads=n_threads, basis=basis, calibration=entry.key if entry else None, refused=None,
        call_tokens=int(tokens), row_tokens=int(row), target_call_latency_ms=target,
        limits=admission_limits(target), **memory,
    )
    if refusal:
        plan.refused = refusal
        return plan
    if memory_budget_bytes:
        fitted = _fit_pass_ceiling(plan, facts, row, float(memory_budget_bytes))
        if fitted:
            plan.refused = fitted
            return plan
    if peak is None or entry is None or eff <= 0:
        name = device.get("name") or device.get("device_name") or "this device"
        plan.refused = f"{name} is not in the hardware table, so its multiply rate is not known: it cannot be sized"
        return plan

    flops_per_token = flops / max(tokens, 1)
    r_tok = eff * peak / flops_per_token
    t_pass_min = entry.t_pass_min_ms
    t_1 = max(tokens / r_tok * 1000.0, t_pass_min)
    plan.tokens_per_second = round(r_tok, 1)
    plan.single_call_ms = round(t_1, 2)

    device_gb = _float(device.get("mem_per_GPU_in_GB") or device.get("mem_per_gpu_in_gb"))
    if not cpu and device_gb and plan.total_bytes > device_gb * 2**30:
        plan.refused = (
            f"one replica needs {plan.total_bytes / 2**30:.1f} GiB ({chosen}) and the GPU has {device_gb:.1f} GiB"
        )
        return plan
    if target is None:
        plan.refused = "the decisions workload names no target_call_latency_ms: a decision replica is sized to one"
        return plan
    if t_1 > target:
        plan.refused = (
            f"one call of {int(workload.get('questions_per_call') or 1)} questions over "
            f"{int(workload.get('state_tokens') or 0):,} state tokens takes {t_1:.0f} ms on this device ({chosen}), "
            f"over the {target:.0f} ms target"
        )
        plan.call_latency_ms = plan.single_call_ms
        return plan
    k = max(int(math.floor(RHO_MAX * r_tok * (target / 1000.0) / tokens)), 1)
    plan.calls_per_replica = k
    plan.call_latency_ms = round(max(t_1, k * tokens / r_tok * 1000.0), 2)
    return plan


def _fit_pass_ceiling(plan: DecisionPlan, facts: DecisionFacts, row: int, budget: float) -> Optional[str]:
    """Lower the plan's pass ceiling until weights, activations and state fit ``budget``; the refusal, or None."""
    used = plan.weight_bytes + plan.activation_bytes + plan.state_bytes
    if used <= budget:
        return None
    per_token = plan.activation_bytes / max(plan.max_pass_tokens, 1)
    floor = max(64, int(row), int(facts.max_len if facts.layout != BRANCH else row))
    room = budget - plan.weight_bytes - plan.state_bytes
    allowed = int(room // per_token) // 64 * 64 if per_token > 0 and room > 0 else 0
    if allowed < floor:
        need = plan.weight_bytes + plan.state_bytes + per_token * floor
        return (
            f"one replica needs {need / 2**30:.1f} GiB ({plan.precision}) even with passes of {floor:,} tokens, and "
            f"the device has {max(budget, 0.0) / 2**30:.1f} GiB free for it"
        )
    plan.max_pass_tokens = min(allowed, plan.max_pass_tokens)
    plan.activation_bytes = int(per_token * plan.max_pass_tokens)
    plan.limits = {**plan.limits, "max_pass_tokens": plan.max_pass_tokens}
    return None


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _memory(facts: DecisionFacts, *, cpu: bool, precision: str, entry, state_cache_mb, row: int) -> Dict[str, int]:
    """The replica's memory terms (FRD §5.16.3): weights, activations at the pass ceiling, Kev's state cache, and the
    process overhead. There is no KV term."""
    weights = facts.params_total * WEIGHT_BYTES[precision]
    if cpu and precision != FLOAT32:
        # oneDNN keeps a packed bfloat16 copy of each linear weight beside the one it was given
        weights += facts.linear_params * 2.0
    compute = 4.0 if precision == FLOAT32 else 2.0
    pass_tokens = max(MAX_PASS_TOKENS, int(row), int(facts.max_len if facts.layout != BRANCH else row))
    measured = entry.act_bytes_per_token if entry is not None else None
    per_token = float(measured) if measured else facts.activation_bytes_per_token(compute) * ACTIVATION_MARGIN
    activation = per_token * pass_tokens
    state = 0.0
    if facts.keeps_state:
        state = float(state_cache_mb or DEFAULT_STATE_CACHE_MB) * 2**20
    if cpu:
        overhead = DEFAULT_CPU_PROCESS_BYTES
    else:
        overhead = float(entry.context_bytes) if entry is not None and entry.context_bytes else DEFAULT_CUDA_CONTEXT_BYTES
    return {
        "weight_bytes": int(weights), "activation_bytes": int(activation), "state_bytes": int(state),
        "overhead_bytes": int(overhead), "max_pass_tokens": int(pass_tokens),
    }
