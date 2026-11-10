"""The grid TC-SZ-01 replays: predictions of ungrouped workloads, captured before ``Workload.group_size`` existed.

Run ``python tests/planner/decisions_baseline.py`` to rewrite the fixture (only from a checkout without group support).
"""

import json
from pathlib import Path

from llm_memory_calculator.device import catalogue
from llm_memory_calculator.engine import EngineRecord, Workload, engine_for
from llm_memory_calculator.model import ModelFacts

HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parent / "fixtures" / "planner"
BASELINE = FIXTURES / "decisions" / "ungrouped_baseline.json"

MODELS = ("qwen3-8b", "qwen3.8-27b", "gpt-oss-20b")
DEVICES = ("h100xm-80c-vgpu", "l40s-48gb")
WORKLOADS = (
    dict(input_tokens=1024, output_tokens=256, concurrency=16),
    dict(input_tokens=4096, output_tokens=128, concurrency=64, arrival="wave", workload_class="synthetic"),
    dict(input_tokens=1072, output_tokens=1, concurrency=64, prefix_tokens=1024, arrival="wave"),
    dict(input_tokens=1072, output_tokens=1, concurrency=64, prefix_tokens=1024, prefix_hit_rate=0.6, requests=256,
         arrival="wave", workload_class="synthetic"),
    dict(input_tokens=2048, output_tokens=512, concurrency=8, prefix_tokens=512),
    dict(input_tokens=8240, output_tokens=1, concurrency=512, prefix_tokens=8192, arrival="wave"),
)


def engine():
    record = json.loads((FIXTURES / "records" / "vllm-0.30-upstream.json").read_text())
    return engine_for(EngineRecord.from_dict(record))


def model(name):
    return ModelFacts.from_config(json.loads((FIXTURES / "models" / f"{name}.json").read_text()))


def cases():
    for m in MODELS:
        for d in DEVICES:
            for i, w in enumerate(WORKLOADS):
                yield f"{m}|{d}|{i}", m, d, w


def predict(m, d, w, **extra):
    p = engine().predict({}, Workload(**w, **extra), model(m), catalogue.get(d))
    return {**p.to_dict(), "kv_tokens": p.limits.kv_tokens, "sequence_ceiling": p.limits.sequence_ceiling}


if __name__ == "__main__":
    BASELINE.write_text(json.dumps({key: predict(m, d, w) for key, m, d, w in cases()}, indent=1, sort_keys=True) + "\n")
    print(f"wrote {BASELINE}")
