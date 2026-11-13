#!/usr/bin/env python3
"""Write ``decision_runtime/calibration.json`` from Bud-Decision-Runtime's decision-benchmark results (Bud spec 031 D-45).

    python tools/decision_calibration.py --runtime 0.1.0 \\
        <runtime>/engine/bench/decision_bench/baselines/{rtx3050,xeon6767p-16t,gtx1650,i7-11370h-4t}-*-new.json

For each result file: the family (``model``), the machine (its GPU, or its CPU's features and the engine's
``threads=``), the precision the engine ran in (``auto`` is resolved by the same rule as the sizing), and the
saturated throughput of the ``typical`` load (16 closed-loop clients, 3 questions of 60 words). The efficiency is
tokens/s x FLOPs per token / the device's peak; ``t_pass_min_ms`` is the one-question latency at one client.

Peak memory (``memory``, recorded from runtime v0.1.1) becomes ``act_bytes_per_token`` and ``context_bytes`` when a
result carries it; without it the sizing keeps its analytic activation figure.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

from llm_memory_calculator.decision_runtime import calibration, families
from llm_memory_calculator.decision_runtime.families import BRANCH, ONE_SEQUENCE
from llm_memory_calculator.decision_runtime.plan import (
    MAX_PASS_TOKENS,
    call_work,
    resolve_hardware,
)
from llm_memory_calculator.decision_runtime.precision import choose

#: question tokens assumed per question of a branch (Kev) benchmark call, to split its tokens between state and questions
_BRANCH_QUESTION_TOKENS = 10


def _device(machine: Dict[str, Any], engine: str) -> Dict[str, Any]:
    gpu = str(machine.get("gpu") or "").split(",")[0].strip()
    if gpu and "device=cpu" not in engine:
        return {"type": "cuda", "name": gpu}
    features = [str(f).lower() for f in machine.get("cpu_features") or []]
    return {"type": "cpu", "name": machine.get("cpu"), "cpu_features": features}


def _threads(engine: str) -> Optional[int]:
    m = re.search(r"threads=(\d+)", engine)
    return int(m.group(1)) if m else None


def _precision(engine: str, device: Dict[str, Any], hardware) -> str:
    parts = engine.split()
    asked = parts[1] if len(parts) > 1 else "auto"
    return choose(device, hardware)[0] if asked == "auto" else asked


def _workload(facts, load: Dict[str, Any]) -> Dict[str, Any]:
    q = max(int(load.get("questions") or 1), 1)
    tokens = int(load["input_tokens"])
    if facts.layout == ONE_SEQUENCE:
        return {"questions_per_call": q, "state_tokens": tokens - q, "question_tokens": 1}
    if facts.layout == BRANCH:
        return {"questions_per_call": q, "state_tokens": max(tokens - q * _BRANCH_QUESTION_TOKENS, 1),
                "question_tokens": _BRANCH_QUESTION_TOKENS}
    row = max(tokens // q, 2)
    return {"questions_per_call": q, "state_tokens": row - 1, "question_tokens": 1}


def entry_from_result(result: Dict[str, Any], source: str, runtime: str, configs: Dict[str, Any]) -> Optional[tuple]:
    family = str(result.get("model") or "").lower()
    if family not in families.FAMILIES:
        return None
    throughput = (result.get("throughput") or {}).get("typical")
    latency = result.get("latency") or {}
    if not throughput or not latency.get("typical"):
        return None
    facts = configs[family]
    engine = str(result.get("engine") or "")
    device = _device(result.get("machine") or {}, engine)
    hardware = resolve_hardware(device)
    precision = _precision(engine, device, hardware)
    threads = _threads(engine) if device["type"] == "cpu" else None
    peak = calibration.peak_flops(device, hardware, precision, threads)
    if not peak:
        print(f"skip {source}: no peak for {device}", file=sys.stderr)
        return None
    tokens, flops, _, refused = call_work(facts, _workload(facts, latency["typical"]))
    if refused:
        print(f"skip {source}: {refused}", file=sys.stderr)
        return None
    eff = float(throughput["tokens_per_s"]) * (flops / tokens) / peak
    short = latency.get("short") or latency["typical"]
    entry: Dict[str, Any] = {
        "device_class": calibration.device_class(device, hardware),
        "device_type": calibration.device_type(device),
        "architecture": calibration.architecture(device, hardware),
        "precision": precision,
        "family": family,
        "kind": facts.kind,
        "eff": round(eff, 4),
        "t_pass_min_ms": float(short["p50_ms"]),
        "measured_tokens_per_s": float(throughput["tokens_per_s"]),
        "act_bytes_per_token": None,
        "context_bytes": None,
        "threads": threads,
        "runtime": runtime,
        "source": os.path.basename(source),
    }
    memory = result.get("memory") or {}
    peak_bytes = memory.get("peak_bytes")
    if peak_bytes:
        weights = memory.get("weight_bytes") or 0
        context = memory.get("context_bytes") or 0
        pass_tokens = memory.get("max_pass_tokens") or MAX_PASS_TOKENS
        entry["act_bytes_per_token"] = round(max(peak_bytes - weights - context, 0) / pass_tokens, 1)
        entry["context_bytes"] = context or None
    key = f"{entry['device_class']}/{precision}/{family}"
    return key, entry


def main(argv: Optional[List[str]] = None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results", nargs="+", help="decision_bench result files")
    ap.add_argument("--runtime", required=True, help="the runtime release the results were measured on")
    ap.add_argument("--configs", default=os.path.join(here, "..", "tests", "fixtures", "decision_runtime"),
                    help="a directory with one checkpoint's config files per family (laya/, julia/, gliner/, kev/ and "
                         "qwen2.5-0.5b/ as Kev's base)")
    ap.add_argument("--out", default=os.path.join(here, "..", "src", "llm_memory_calculator", "decision_runtime",
                                                  "calibration.json"))
    args = ap.parse_args(argv)
    configs = {
        "laya": families.facts(os.path.join(args.configs, "laya"), "laya"),
        "julia": families.facts(os.path.join(args.configs, "julia"), "julia"),
        "gliner": families.facts(os.path.join(args.configs, "gliner"), "gliner"),
        "kev": families.facts(os.path.join(args.configs, "kev"), "kev", base_dir=os.path.join(args.configs, "qwen2.5-0.5b")),
    }
    entries: Dict[str, Any] = {}
    for path in args.results:
        with open(path) as f:
            made = entry_from_result(json.load(f), path, args.runtime, configs)
        if made:
            key, entry = made
            entries[key] = entry
    out = {
        "about": "Bud-Decision-Runtime efficiency per device class / precision / family, from its decision benchmark "
                 "(tools/decision_calibration.py; Bud spec 031 D-45)",
        "entries": dict(sorted(entries.items())),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(f"wrote {len(entries)} entries to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
