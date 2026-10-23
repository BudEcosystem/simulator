"""Per-request lengths of a load for the scheduler simulation (FRD-027 §20.2 L4).

Random-token benchmark prompts and outputs are fixed-length (the suite sets ignore_eos). Real text is not:
- prompts follow the Spec-Bench shape the study's real points use (median 224 characters, mean 1,202, p90 3,303),
  scaled to the workload's mean;
- outputs stop at end-of-sequence, so they are drawn from an exponential capped at the workload's maximum, whose mean
  matches the workload's mean output. This shape is an assumption until per-request output lengths are recorded.

Lengths are drawn at fixed quantiles in a fixed order, so the same workload always gives the same simulation.
"""

from __future__ import annotations

import json
import math
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Tuple

_SHAPE = Path(__file__).parent / "data" / "specbench_prompt_shape.json"
DEFAULT_OUTPUT_CAP = 512  # the study suite's real points (--sharegpt-output-len 512)


@lru_cache(maxsize=1)
def _prompt_ratios() -> Tuple[float, ...]:
    return tuple(json.loads(_SHAPE.read_text())["ratios"])


def _capped_exponential_scale(mean: float, cap: float) -> float:
    """The scale s with E[min(cap, Exp(s))] = s (1 - exp(-cap / s)) = mean."""
    lo, hi = 1e-6, max(mean, 1.0) * 1e4
    for _ in range(200):
        mid = (lo + hi) / 2
        if mid * (1 - math.exp(-cap / mid)) < mean:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def lengths_for(workload: Any, n: int) -> List[Tuple[int, int]]:
    inp = max(float(workload.input_tokens), 1.0)
    out = max(float(workload.output_tokens), 1.0)
    if getattr(workload, "workload_class", "real") == "synthetic":
        return [(int(round(inp)), int(round(out)))] * n
    ratios = _prompt_ratios()
    picks = [ratios[min(len(ratios) - 1, int((i + 0.5) / n * len(ratios)))] for i in range(n)]
    scale = sum(picks) / len(picks)
    prompts = [max(int(round(inp * r / scale)), 1) for r in picks]
    cap = float(getattr(workload, "output_tokens_max", None) or max(DEFAULT_OUTPUT_CAP, out))
    if out >= cap:
        outputs = [int(cap)] * n
    else:
        s = _capped_exponential_scale(out, cap)
        outputs = [max(1, min(int(cap), int(round(-s * math.log(1 - (i + 0.5) / n))))) for i in range(n)]
    random.Random(0).shuffle(prompts)
    random.Random(1).shuffle(outputs)
    return list(zip(prompts, outputs))
