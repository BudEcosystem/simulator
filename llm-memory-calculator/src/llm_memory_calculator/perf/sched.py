"""FRD-027 Revision 2, L4: a simulation of vLLM's V1 scheduler (§20.2).

It runs a benchmark-shaped load (`vllm bench serve --max-concurrency c`: c clients, each sending its next request
when the previous one completes) step by step, with every step's time taken from a measured step profile:
- each request holds prompt + generated tokens, and a step schedules what is not yet in the KV cache: one token when
  decoding, more when prefilling or recomputing after a preemption;
- running requests are scheduled first, then waiting ones first-come-first-served, while the step's token budget
  (`max-num-batched-tokens`, decode tokens included), `max-num-seqs` and the free KV pages allow;
- a request is sampled once its tokens are all in the cache: its first token, or the next one;
- a running request that needs a page and finds none preempts the most recently admitted one, which frees its pages
  and recomputes everything later.

This replaces the fitted wave, queue and drain terms (`arrival.*`): waves, mixed prefill steps, queueing and
KV-limited admission are what the simulation does, not constants. Nothing is fitted here.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SimConfig:
    max_num_seqs: int
    chunk: int  # the step's token budget (max-num-batched-tokens)
    kv_blocks: int  # KV pages in the pool
    block_tokens: int  # tokens per page
    attn_groups: int = 1  # pages per block of tokens (a hybrid holds one per attention group)
    state_blocks: float = 0.0  # pages a sequence's recurrent state holds, whatever its length
    api_ms: float = 0.0  # what the API server adds to the first token and the end-to-end time
    #: a request reaches the engine this long after the client sends it (HTTP, tokenizing the prompt), and the client
    #: sees a completion this long after the engine's last step. Together they keep a benchmark's waves synchronized:
    #: while a finished request's successor is on its way, the rest of its wave decodes on and finishes too
    arrive_ms: float = 0.0
    return_ms: float = 0.0
    #: async scheduling: the engine schedules step k + 1 while step k runs, so a request that arrives during a step is
    #: first scheduled in the step after the next one
    async_scheduling: bool = False
    #: run a decode-only window (nothing waiting or on its way, the pages to grow into free) to its first completion in
    #: one go: the batch is fixed and every context grows by one per step, so it is exact (NFR-9: decode c128 took
    #: 441 ms step by step)
    fast_forward: bool = True


@dataclass
class _Req:
    idx: int
    prompt: int
    output: int
    submit: float = 0.0
    computed: int = 0
    generated: int = 0
    pages: int = 0
    first: Optional[float] = None
    done: Optional[float] = None
    preempted: int = 0
    active: bool = False  # in the running list (a flag: list membership per step made the loop quadratic)
    holds: int = 0  # tokens its pages can hold, so a step within its last page allocates nothing

    @property
    def num_tokens(self) -> int:
        return self.prompt + self.generated


@dataclass
class SimResult:
    ttft_samples: List[float]  # per request, in request order (API time included)
    tpot_samples: List[float]
    e2e_samples: List[float]
    duration_ms: float
    output_tokens: int
    preemptions: int
    max_running: int
    mean_running: float
    steps: int

    @property
    def ttft_ms(self) -> float:
        return float(statistics.median(self.ttft_samples))

    @property
    def tpot_ms(self) -> float:
        return float(statistics.median(self.tpot_samples)) if self.tpot_samples else 0.0

    @property
    def throughput_tok_s(self) -> float:
        return 1000.0 * self.output_tokens / self.duration_ms if self.duration_ms > 0 else 0.0

    def percentile(self, samples: Sequence[float], q: float) -> float:
        ordered = sorted(samples)
        return float(ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))])


def step_ms(
    model: Any, decode: Sequence[_Req], prefill: Sequence[Tuple[_Req, int]], ctx_sum: Optional[float] = None
) -> float:
    """A step's time from the profile. A decode-only step is the measured decode step at its batch and mean context. A
    step with prompt tokens costs its tokens at the prefill rate for its prompts' length, and never less than a
    decode step for its sequences: every step reads all the weights. ``ctx_sum`` is the decoding requests' summed
    context when the caller already has it."""
    d = len(decode)
    if ctx_sum is None:
        ctx_sum = sum(r.computed + 1 for r in decode)
    ctx = (ctx_sum / d) if d else 1.0
    tokens = sum(t for _, t in prefill)
    if not tokens:
        return model.decode_ms(d, ctx)
    length = sum(r.num_tokens * t for r, t in prefill) / tokens
    floor = model.decode_ms(max(d + len(prefill), 1), ctx if d else length)
    return max(model.chunk_ms(tokens + d, length), floor)


def _decode_window(running: Sequence[_Req], cfg: SimConfig, free: int, pages_for: Any) -> int:
    """Decode steps every running request can take before the first one finishes, if all are decoding and the free
    pages cover their growth; 0 otherwise."""
    window = None
    need = 0
    for r in running:
        if r.generated == 0 or r.num_tokens - r.computed != 1:
            return 0
        left = r.output - r.generated
        window = left if window is None else min(window, left)
    for r in running:
        need += max(pages_for(r.computed + window) - r.pages, 0)
    return window if need <= free else 0


def simulate(
    model: Any, cfg: SimConfig, lengths: Sequence[Tuple[int, int]], *, concurrency: int, max_steps: int = 2_000_000
) -> SimResult:
    """Run ``lengths`` (prompt, output) requests through ``concurrency`` closed-loop clients."""
    reqs = [_Req(i, int(p), max(int(o), 1)) for i, (p, o) in enumerate(lengths)]
    waiting: Deque[_Req] = deque()
    inflight: List[Tuple[float, _Req]] = []  # sent by a client, not yet at the engine: (arrival time, request)
    running: List[_Req] = []
    nxt = 0
    t = 0.0
    free = int(cfg.kv_blocks)
    state = int(math.ceil(cfg.state_blocks))
    preemptions = steps = max_running = 0
    running_sum = 0.0

    def submit(at: float) -> None:
        nonlocal nxt
        if nxt < len(reqs):
            reqs[nxt].submit = at
            inflight.append((at + cfg.arrive_ms, reqs[nxt]))
            nxt += 1

    def arrivals(now: float) -> None:
        for item in sorted(inflight, key=lambda a: (a[0], a[1].idx)):
            if item[0] <= now + 1e-9:
                inflight.remove(item)
                waiting.append(item[1])

    def pages_for(tokens: int) -> int:
        return cfg.attn_groups * math.ceil(tokens / max(cfg.block_tokens, 1)) + state

    def holds(pages: int) -> int:
        """The tokens ``pages`` hold: whole blocks across every attention group, after the state pages."""
        return ((pages - state) // max(cfg.attn_groups, 1)) * max(cfg.block_tokens, 1)

    def release(r: _Req) -> None:
        nonlocal free
        free += r.pages
        r.pages = 0
        r.holds = 0

    def preempt(victim: _Req) -> None:
        nonlocal preemptions
        running.remove(victim)
        victim.active = False
        release(victim)
        victim.computed = 0
        victim.preempted += 1
        preemptions += 1
        waiting.appendleft(victim)

    def grow(r: _Req, tokens: int) -> bool:
        """Pages for ``tokens`` of ``r`` in the cache, preempting the newest running requests; False if ``r`` itself
        had to go."""
        nonlocal free
        if r.pages and tokens <= r.holds:  # within its last page: nothing to allocate (most decode steps)
            return True
        delta = pages_for(tokens) - r.pages
        while delta > free:
            victim = running[-1]
            preempt(victim)
            if victim is r:
                return False
        if delta > 0:
            free -= delta
            r.pages += delta
        r.holds = holds(r.pages)
        return True

    for _ in range(min(concurrency, len(reqs))):
        submit(0.0)
    scheduled_at = 0.0  # when the step now starting was scheduled (async: while the previous one ran)
    while waiting or running or inflight:
        arrivals(scheduled_at if cfg.async_scheduling else t)
        if not waiting and not running:
            t = scheduled_at = min(a for a, _ in inflight)  # the engine idles until the next request arrives
            continue
        if cfg.fast_forward and not waiting and not inflight and len(running) > 1:
            window = _decode_window(running, cfg, free, pages_for)
            if window > 1:
                d = len(running)
                ctx0 = sum(r.computed + 1 for r in running) / d
                durations = [model.decode_ms(d, ctx0 + j) for j in range(window - 1)]
                for r in running:  # all but the window's last step, which the loop below runs (it may finish requests)
                    r.computed += window - 1
                    r.generated += window - 1
                    grow(r, r.computed + 1)
                t += sum(durations)
                scheduled_at = t - durations[-1]
                steps += window - 1
                running_sum += d * (window - 1)
                max_running = max(max_running, d)
        steps += 1
        if steps > max_steps:
            raise RuntimeError("the simulation did not finish")
        budget = int(cfg.chunk)
        decode: List[_Req] = []
        prefill: List[Tuple[_Req, int]] = []
        ctx_sum = 0
        for r in list(running):
            if budget <= 0:
                break
            if not r.active:
                continue
            want = r.prompt + r.generated - r.computed
            tok = want if want < budget else budget
            # inline grow's fast path: a step within the request's last page allocates nothing (most decode steps)
            if not (r.pages and r.computed + tok <= r.holds) and not grow(r, r.computed + tok):
                continue
            if want == 1 and r.generated > 0:
                decode.append(r)
                ctx_sum += r.computed + 1
            else:
                prefill.append((r, tok))
            budget -= tok
        while waiting and budget > 0 and len(running) < cfg.max_num_seqs:
            r = waiting[0]
            tok = min(r.num_tokens - r.computed, budget)
            if pages_for(r.computed + tok) - r.pages > free:
                break
            waiting.popleft()
            running.append(r)
            r.active = True
            grow(r, r.computed + tok)
            prefill.append((r, tok))
            budget -= tok
        if not decode and not prefill:
            raise RuntimeError("the KV pool cannot hold one request at the step's budget")
        max_running = max(max_running, len(running))
        running_sum += len(running)
        scheduled_at = t
        t += step_ms(model, decode, prefill, ctx_sum)
        finished = 0
        for r in decode:  # each decoding request is caught up after its one token: it samples the next
            r.computed += 1
            r.generated += 1
            if r.generated >= r.output:
                r.done = t
                r.active = False
                release(r)
                finished += 1
        for r, tok in prefill:
            r.computed += tok
            if r.computed >= r.prompt + r.generated:  # caught up: the step samples its next token
                r.generated += 1
                if r.first is None:
                    r.first = t
                if r.generated >= r.output:
                    r.done = t
                    r.active = False
                    release(r)
                    finished += 1
        if finished:
            running[:] = [r for r in running if r.active]
            for _ in range(finished):  # each client sends its next request once it sees the completion
                submit(t + cfg.return_ms)
    api = cfg.api_ms + cfg.return_ms
    ttft = [r.first - r.submit + api for r in reqs]
    tpot = [(r.done - r.first) / (r.output - 1) for r in reqs if r.output > 1]
    e2e = [r.done - r.submit + api for r in reqs]
    t = max(r.done for r in reqs) + cfg.return_ms
    return SimResult(
        ttft_samples=ttft,
        tpot_samples=tpot,
        e2e_samples=e2e,
        duration_ms=t,
        output_tokens=sum(r.output for r in reqs),
        preemptions=preemptions,
        max_running=max_running,
        mean_running=running_sum / max(steps, 1),
        steps=steps,
    )
