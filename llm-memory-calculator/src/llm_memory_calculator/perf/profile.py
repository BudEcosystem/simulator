"""FRD-027 Revision 2, L2/L3: a measured step profile (§20.2).

A profile holds what one engine build did on one device for one model, as ``perf.profiler`` measured it:
- decode-only step times over a (running sequences, context) grid;
- prefill cost per token, by prompt length and by how full the step's token budget was;
- the points the engine preempted, which measure recompute rather than a step.

Nothing here is fitted. Between grid points the step is interpolated piecewise-linearly in the batch and linearly in
the context (attention reads grow with it); a leave-one-out pass over the grid is the interpolation error the
prediction's band carries.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def _lerp(x: float, x0: float, y0: float, x1: float, y1: float) -> float:
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _piecewise(points: Sequence[Tuple[float, float]], x: float) -> float:
    """Linear interpolation over sorted ``points``, extrapolated linearly from the nearest two outside them."""
    if len(points) == 1:
        return points[0][1]
    xs = [p[0] for p in points]
    i = bisect.bisect_left(xs, x)
    if i < len(xs) and xs[i] == x:
        return points[i][1]
    if i == 0:
        a, b = points[0], points[1]
    elif i >= len(points):
        a, b = points[-2], points[-1]
    else:
        a, b = points[i - 1], points[i]
    return _lerp(x, a[0], a[1], b[0], b[1])


def _log_piecewise(points: Sequence[Tuple[float, float]], x: float) -> float:
    """Interpolation in log x, held flat outside the measured range (a per-token cost does not extrapolate)."""
    if len(points) == 1:
        return points[0][1]
    lx = math.log(max(x, 1e-9))
    pts = [(math.log(max(p[0], 1e-9)), p[1]) for p in points]
    if lx <= pts[0][0]:
        return pts[0][1]
    if lx >= pts[-1][0]:
        return pts[-1][1]
    return _piecewise(pts, lx)


@dataclass
class StepProfile:
    sku: str
    engine: str
    model: str
    weights: str = "random"
    settings: Dict[str, Any] = field(default_factory=dict)
    chunk: int = 8192
    kv_blocks: int = 0
    block_tokens: int = 0
    #: (running sequences, context) -> decode step ms
    decode: Dict[Tuple[int, int], float] = field(default_factory=dict)
    #: (prompt length, prompt tokens in the phase) -> prefill ms per token
    prefill: Dict[Tuple[int, int], float] = field(default_factory=dict)
    #: (b, prompt length) -> preemptions the engine reported at that point
    preempted: Dict[Tuple[int, int], float] = field(default_factory=dict)
    #: (b, prompt length) -> how much longer the shape's slowest step took on first use than warm (kernels compiled
    #: per shape on first use: the fork's fused MoE, R1 finding 5), ms
    cold_stall: Dict[Tuple[int, int], float] = field(default_factory=dict)
    #: the API path's round trip at one request in flight (measured TTFT at c1 less the engine's prefill), ms
    api_ms: float = 0.0
    #: the profile's measured error per quantity against real deployments (p90 |predicted / measured - 1|), set by
    #: validation; the prediction's band combines it with the interpolation error
    residual: Dict[str, float] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ building

    @classmethod
    def from_steps(
        cls, data: Mapping[str, Any], *, sku: str, engine: str, model: str, settings: Optional[Mapping[str, Any]] = None
    ) -> "StepProfile":
        """A profile from the profiler's ``bud-perf-steps/1`` output."""
        prof = cls(
            sku=sku,
            engine=engine,
            model=model,
            weights="real" if data.get("load_format") not in (None, "dummy") else "random",
            settings=dict(settings or {}),
            chunk=int(data.get("chunk") or 8192),
            kv_blocks=int(data.get("kv_blocks") or 0),
            block_tokens=int(data.get("block_tokens") or 0),
            provenance={k: data.get(k) for k in ("name", "vllm", "load_format", "max_model_len", "gpu_util", "extra")},
        )
        for p in data.get("points") or []:
            if p.get("skipped"):
                continue
            b, length = int(p["b"]), int(p["len"])
            if p.get("preemptions"):
                prof.preempted[(b, length)] = float(p["preemptions"])
                continue
            if p.get("decode_ms"):
                prof.decode[(b, int(p.get("decode_ctx") or length))] = float(p["decode_ms"])
            if p.get("warm_max_ms") is not None and p.get("steps"):
                stall = float(p["warm_max_ms"]) - max(float(s["ms"]) for s in p["steps"])
                if stall > 0:
                    prof.cold_stall[(b, length)] = round(stall, 1)
            # the prefill phase: every step before the last prompt's first token (they all had a prompt waiting)
            phase = [s["ms"] for s in p.get("steps") or [] if s.get("waiting", 0) > 0]
            if phase:
                prof.prefill[(length, b * length)] = sum(phase) / (b * length)
        return prof

    def cold_stall_ms(self) -> float:
        """The longest first-use stall the profiler saw at any shape (0 when it saw none, or did not record it)."""
        return max(self.cold_stall.values(), default=0.0)

    # ------------------------------------------------------------------ decode

    def _rows(self, table: Mapping[Tuple[int, int], float]) -> Dict[int, List[Tuple[float, float]]]:
        rows: Dict[int, List[Tuple[float, float]]] = {}
        for (b, ctx), ms in table.items():
            rows.setdefault(ctx, []).append((float(b), ms))
        return {ctx: sorted(pts) for ctx, pts in rows.items()}

    def _decode_from(self, rows: Mapping[int, List[Tuple[float, float]]], b: float, ctx: float) -> float:
        if not rows:
            raise ValueError("the profile has no decode points")
        covering = [c for c, pts in rows.items() if pts[0][0] <= b <= pts[-1][0]]
        candidates = sorted(covering or rows)
        below = [c for c in candidates if c <= ctx]
        above = [c for c in candidates if c >= ctx]
        if below and above:
            c0, c1 = max(below), min(above)
            v0, v1 = _piecewise(rows[c0], b), _piecewise(rows[c1], b)
            value = v0 if c0 == c1 else _lerp(ctx, c0, v0, c1, v1)
        else:
            side = below or above
            near = sorted(side, key=lambda c: abs(c - ctx))[:2]
            if len(near) == 2:
                c0, c1 = sorted(near)
                value = _lerp(ctx, c0, _piecewise(rows[c0], b), c1, _piecewise(rows[c1], b))
            else:
                value = _piecewise(rows[near[0]], b)
        return max(value, 1e-3)

    def _decode_rows(self) -> Dict[int, List[Tuple[float, float]]]:
        # the table is fixed once built; rebuilding it per step cost a quarter of a large simulation (NFR-9)
        key = (id(self.decode), len(self.decode))
        cached = self.__dict__.get("_decode_rows_cache")
        if cached is None or cached[0] != key:
            cached = (key, self._rows(self.decode))
            self.__dict__["_decode_rows_cache"] = cached
        return cached[1]

    def decode_ms(self, b: float, ctx: float) -> float:
        """The decode-only step with ``b`` running sequences at a mean context of ``ctx`` tokens."""
        return self._decode_from(self._decode_rows(), max(float(b), 1.0), max(float(ctx), 1.0))

    def interpolation_error(self) -> float:
        """p90 of the leave-one-out relative error over the decode grid (FR-PROF-1: part of the band)."""
        errs = []
        for key, ms in self.decode.items():
            rest = {k: v for k, v in self.decode.items() if k != key}
            if len(rest) < 2:
                continue
            pred = self._decode_from(self._rows(rest), float(key[0]), float(key[1]))
            errs.append(abs(pred / ms - 1.0))
        if not errs:
            return 0.0
        errs.sort()
        return errs[min(len(errs) - 1, max(0, math.ceil(0.9 * len(errs)) - 1))]

    # ------------------------------------------------------------------ prefill

    def _prefill_rows(self) -> List[Tuple[float, List[Tuple[float, float]], float]]:
        key = (id(self.prefill), len(self.prefill))
        cached = self.__dict__.get("_prefill_rows_cache")
        if cached is None or cached[0] != key:
            by_len: Dict[int, List[Tuple[float, float]]] = {}
            for (plen, total), rate in self.prefill.items():
                by_len.setdefault(plen, []).append((float(total), rate))
            rows = [(float(plen), sorted(pts), max(p[0] for p in pts)) for plen, pts in sorted(by_len.items())]
            cached = (key, rows)
            self.__dict__["_prefill_rows_cache"] = cached
        return cached[1]

    def _per_token(self, length: float, tokens: float) -> float:
        if not self.prefill:
            raise ValueError("the profile has no prefill points")
        # a chunk is at most the token budget: a longer phase is several full chunks at the same rate
        per_len = [(plen, _log_piecewise(pts, min(tokens, top))) for plen, pts, top in self._prefill_rows()]
        return _log_piecewise(per_len, length)

    def chunk_ms(self, tokens: float, request_len: float) -> float:
        """One prefill step of ``tokens`` prompt tokens from prompts of ``request_len`` tokens."""
        return self._per_token(request_len, tokens) * max(float(tokens), 0.0)

    def prefill_ms(self, tokens: float, request_len: float) -> float:
        """The whole prefill of ``tokens`` prompt tokens (several chunks when past the budget)."""
        return self._per_token(request_len, min(tokens, self.chunk)) * max(float(tokens), 0.0)

    # ------------------------------------------------------------------ storage

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "bud-perf-profile/1",
            "sku": self.sku,
            "engine": self.engine,
            "model": self.model,
            "weights": self.weights,
            "settings": self.settings,
            "chunk": self.chunk,
            "kv_blocks": self.kv_blocks,
            "block_tokens": self.block_tokens,
            "decode": [[b, c, ms] for (b, c), ms in sorted(self.decode.items())],
            "prefill": [[l, t, r] for (l, t), r in sorted(self.prefill.items())],
            "preempted": [[b, l, n] for (b, l), n in sorted(self.preempted.items())],
            "cold_stall": [[b, l, ms] for (b, l), ms in sorted(self.cold_stall.items())],
            "api_ms": self.api_ms,
            "residual": self.residual,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StepProfile":
        return cls(
            sku=data["sku"],
            engine=data["engine"],
            model=data["model"],
            weights=data.get("weights", "random"),
            settings=dict(data.get("settings") or {}),
            chunk=int(data.get("chunk") or 8192),
            kv_blocks=int(data.get("kv_blocks") or 0),
            block_tokens=int(data.get("block_tokens") or 0),
            decode={(int(b), int(c)): float(ms) for b, c, ms in data.get("decode") or []},
            prefill={(int(l), int(t)): float(r) for l, t, r in data.get("prefill") or []},
            preempted={(int(b), int(l)): float(n) for b, l, n in data.get("preempted") or []},
            cold_stall={(int(b), int(l)): float(ms) for b, l, ms in data.get("cold_stall") or []},
            api_ms=float(data.get("api_ms") or 0.0),
            residual={k: float(v) for k, v in (data.get("residual") or {}).items()},
            provenance=dict(data.get("provenance") or {}),
        )
