"""Read the FRD-027 harness output (``study/results/<cfg>/``) into a grounding corpus (FR-GND-4).

One :class:`CorpusConfig` per engine configuration the harness ran: its engine-neutral settings (FRD-027
§5.1) derived from the vLLM arguments, the startup facts vLLM logged (weights loaded, KV pool, block size,
CUDA-graph estimate, boot outcome and its message), every measured suite point and the GSM8K quality runs.
Arguments the settings vocabulary does not cover are kept in ``unmodelled``; the corpus marks a configuration
``guarded`` when one of them is a flag ``engine.guard`` never emits (eager, async scheduling off, ...), so the
accuracy gate does not grade the predictor on configurations the planner cannot render.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

#: vLLM argument -> setting (FRD-027 §5.1). The inverse of the vLLM record's ``settings`` mapping.
ARG_SETTINGS = {
    "max-num-seqs": "batch.max_running",
    "max-num-batched-tokens": "batch.prefill_chunk_tokens",
    "max-model-len": "context.max_tokens",
    "gpu-memory-utilization": "memory.device_fraction",
    "kv-cache-dtype": "kv.dtype",
    "mamba-ssm-cache-dtype": "state.dtype",
    "max-loras": "lora.max_adapters",
    "max-lora-rank": "lora.max_rank",
    "tensor-parallel-size": "parallel.tensor",
    "pipeline-parallel-size": "parallel.pipeline",
}

#: Flags engine.guard never emits (FRD-027 §10.0 disposition "never") -- configurations using them are kept
#: for the record but not graded.
GUARDED_NEVER = {"enforce-eager", "no-async-scheduling"}
#: Flags left at the engine default ("default" rows): measured inert or worse; a configuration that sets one
#: is graded as if it were the default only when the study measured it inert.
GUARDED_DEFAULT = {
    "performance-mode",
    "compilation-config",
    "O3",
    "O",
    "attention-backend",
    "moe-backend",
    "gdn-prefill-backend",
    "stream-interval",
    "api-server-count",
    "long-prefill-token-threshold",
    "mamba-cache-mode",
    "language-model-only",
    "limit-mm-per-prompt",
    "cudagraph-capture-sizes",
    "hf-overrides",
}
#: Default-row flags the study measured as inert on throughput (screen-table.md): graded as default.
INERT = {
    "stream-interval",
    "api-server-count",
    "long-prefill-token-threshold",
    "mamba-cache-mode",
    "O3",
    "O",
    "gdn-prefill-backend",
    "compilation-config",
}

#: Harness checkpoint key -> (family, weight format of the checkpoint).
CHECKPOINTS = {
    "bf16": ("Qwen3.8-27B", "bf16"),
    "fp8": ("Qwen3.8-27B", "fp8"),
    "8b-bf16": ("Qwen3-8B", "bf16"),
    "8b-fp8": ("Qwen3-8B", "fp8"),
    "oss-mxfp4": ("gpt-oss-20b", "mxfp4"),
    "oss-bf16": ("gpt-oss-20b", "bf16"),
    "qmoe-bf16": ("Qwen1.5-MoE-A2.7B", "bf16"),  # round 9: the MoE class's second family
}
#: Harness image key -> engine record id.
RECORDS = {
    "upstream": "vllm@0.30.0",
    "bud": "bud-vllm@0.9.0-481e481be",
    "bud010": "bud-vllm@0.10.0",  # BudConnect dev's new build: upstream v0.31.0 re-tagged (round 7, TCS)
}
SKU = "h100xm-80c-vgpu"
VISIBLE_GIB = 79.67

#: Suite points whose prompts are random tokens (speculative acceptance differs from real text).
SYNTHETIC_POINTS = {"chat", "decode", "long", "xlong", "prefix"}
#: The suite's prefix point: 8 distinct shared prefixes of this many tokens, each followed by a 512-token suffix.
PREFIX_SHARED_TOKENS = 4096


@dataclass
class Point:
    point: str
    input_tokens: float
    output_tokens: float
    concurrency: int
    num_prompts: int
    completed: int
    failed: int
    ttft_ms: float  # median
    tpot_ms: float  # median
    output_tok_s: float
    sampling: str  # greedy | sampled
    workload_class: str  # synthetic | real
    prefix_hit_rate: float = 0.0
    preemptions: float = 0.0


@dataclass
class Startup:
    ready: bool
    model_load_gib: Optional[float] = None
    kv_gib: Optional[float] = None
    kv_tokens: Optional[int] = None
    block_tokens: Optional[int] = None
    max_model_len: Optional[int] = None
    cudagraph_gib: Optional[float] = None
    init_s: Optional[float] = None
    api_up_s: Optional[float] = None
    failure: Optional[str] = None  # oom | refused | error
    message: Optional[str] = None


@dataclass
class CorpusConfig:
    name: str
    family: str
    checkpoint: str  # weight format of the checkpoint the harness loaded
    record: str
    sku: str
    hardware_mode: str
    slice_mib: Optional[int]
    args: List[str]
    settings: Dict[str, Any]
    unmodelled: List[str]
    guarded: bool
    startup: Startup
    points: List[Point] = field(default_factory=list)
    quality: Dict[str, Any] = field(default_factory=dict)
    invalid_points: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CorpusConfig":
        data = dict(data)
        data["startup"] = Startup(**data["startup"])
        data["points"] = [Point(**p) for p in data.get("points") or []]
        return cls(**data)


def parse_args(args: Iterable[str]) -> Dict[str, Any]:
    """vLLM CLI arguments as ``{flag: value}`` (``True`` for bare flags)."""
    out: Dict[str, Any] = {}
    items = list(args or [])
    i = 0
    while i < len(items):
        token = str(items[i])
        if token.startswith("--") or (token.startswith("-O") and len(token) > 2):
            if token.startswith("-O") and not token.startswith("--"):
                out["O"] = token[2:]
                i += 1
                continue
            key = token[2:]
            if "=" in key:
                key, value = key.split("=", 1)
                out[key] = value
                i += 1
                continue
            if i + 1 < len(items) and not str(items[i + 1]).startswith("-"):
                out[key] = items[i + 1]
                i += 2
                continue
            out[key] = True
        i += 1
    return out


def _num(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    text = str(value)
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return value


def settings_from_args(
    args: Iterable[str], checkpoint_format: str
) -> Tuple[Dict[str, Any], List[str]]:
    """``(settings, unmodelled flags)`` for a vLLM argument list."""
    flags = parse_args(args)
    settings: Dict[str, Any] = {"weights.format": checkpoint_format}
    unmodelled: List[str] = []
    for key, value in flags.items():
        if key in ARG_SETTINGS:
            settings[ARG_SETTINGS[key]] = _num(value)
        elif key == "quantization":
            settings["weights.format"] = str(value)
            settings["weights.online"] = True
        elif key == "speculative-config":
            spec = json.loads(value) if isinstance(value, str) else dict(value)
            settings["spec.method"] = str(spec.get("method"))
            settings["spec.tokens"] = int(spec.get("num_speculative_tokens") or 0)
            if spec.get("model"):
                settings["spec.draft"] = str(spec["model"])
        elif key == "enable-lora":
            settings.setdefault("lora.max_adapters", 1)
        elif key in ("enable-prefix-caching", "no-enable-prefix-caching"):
            settings["prefix_cache.enabled"] = key == "enable-prefix-caching"
        else:
            unmodelled.append(key)
    if settings.get("kv.dtype") in ("auto", None):
        settings.pop("kv.dtype", None)
    if str(settings.get("state.dtype")) in ("float32", "auto"):
        settings.pop("state.dtype", None)
    return settings, sorted(unmodelled)


def _float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return None


_EQUIV = re.compile(r"equivalent to --gpu-memory-utilization=([0-9.]+)")
_CURRENT = re.compile(r"current --gpu-memory-utilization=([0-9.]+)")
_REFUSALS = (
    ("refused", re.compile(r"ValueError: (.*)")),
    ("oom", re.compile(r"(CUDA out of memory.*?)(?: Including|$)")),
    ("error", re.compile(r"RuntimeError: (The size of tensor.*)")),
)


def _startup(cfg_dir: Path) -> Startup:
    path = cfg_dir / "startup.json"
    if not path.exists():
        return Startup(ready=False, failure="error", message="no startup record")
    data = json.loads(path.read_text())
    st = Startup(
        ready=bool(data.get("ready")),
        model_load_gib=_float(data.get("model_load_gib")),
        kv_gib=_float(data.get("available_kv_gib")),
        kv_tokens=int(_float(data.get("kv_cache_tokens")) or 0) or None,
        block_tokens=int(_float(data.get("attn_block_size")) or 0) or None,
        max_model_len=int(_float(data.get("max_model_len")) or 0) or None,
        init_s=_float(data.get("init_engine_s")),
        api_up_s=_float(data.get("wait_s")),
    )
    log = cfg_dir / "engine.log"
    if log.exists():
        text = log.read_text(errors="replace")
        equiv, current = _EQUIV.search(text), _CURRENT.search(text)
        if equiv and current:
            st.cudagraph_gib = round(
                (float(current.group(1)) - float(equiv.group(1))) * VISIBLE_GIB, 3
            )
        if not st.ready:
            for kind, pattern in _REFUSALS:
                match = pattern.search(text)
                if match:
                    st.failure, st.message = kind, match.group(1).strip()[:400]
                    break
            else:
                st.failure, st.message = "error", "engine did not become ready"
    return st


def _points(cfg_dir: Path) -> List[Point]:
    path = cfg_dir / "summary.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("rc") not in (0, None) or not row.get("completed"):
            continue
        delta = row.get("engine_delta") or {}
        completed = int(row.get("completed") or 0)
        prompt_total = float(delta.get("vllm:prompt_tokens_total") or 0.0)
        gen_total = float(delta.get("vllm:generation_tokens_total") or 0.0)
        queries = float(delta.get("vllm:prefix_cache_queries_total") or 0.0)
        hits = float(delta.get("vllm:prefix_cache_hits_total") or 0.0)
        input_tokens = float(row.get("input_len") or 0) or (
            prompt_total / completed if completed else 0.0
        )
        output_tokens = (
            gen_total / completed if completed and gen_total else float(row.get("output_len") or 0)
        )
        point = str(row.get("point"))
        out.append(
            Point(
                point=point,
                input_tokens=round(input_tokens, 1),
                output_tokens=round(output_tokens, 1),
                concurrency=int(row.get("concurrency") or 1),
                num_prompts=int(row.get("num_prompts") or completed),
                completed=completed,
                failed=int(row.get("failed") or 0),
                ttft_ms=float(row.get("median_ttft_ms") or 0.0),
                tpot_ms=float(row.get("median_tpot_ms") or 0.0),
                output_tok_s=float(row.get("output_throughput") or 0.0),
                sampling="sampled" if point.endswith("S") else "greedy",
                workload_class="synthetic" if point in SYNTHETIC_POINTS else "real",
                prefix_hit_rate=round(hits / queries, 4) if queries else 0.0,
                preemptions=float(delta.get("vllm:num_preemptions_total") or 0.0),
            )
        )
    return out


def _quality(cfg_dir: Path) -> Dict[str, Any]:
    out = {}
    for name, key in (
        ("quality.json", "gsm8k_zeroshot"),
        ("quality-fewshot.json", "gsm8k_fewshot"),
    ):
        path = cfg_dir / name
        if path.exists():
            data = json.loads(path.read_text())
            out[key] = {
                "accuracy": data.get("accuracy"),
                "n": data.get("n"),
                "garbage": data.get("garbage", 0),
                "errors": data.get("errors", 0),
            }
    return out


def read_config(cfg_dir: Path) -> Optional[CorpusConfig]:
    path = cfg_dir / "config.json"
    if not path.exists():
        return None
    cfg = json.loads(path.read_text())
    checkpoint_key = cfg.get("model") or "bf16"
    family, fmt = CHECKPOINTS.get(checkpoint_key, ("Qwen3.8-27B", "bf16"))
    args = [str(a) for a in cfg.get("args") or []]
    settings, unmodelled = settings_from_args(args, fmt)
    gpumem = int(cfg.get("gpumem") or 0)
    flags = set(parse_args(args))
    guarded = bool(flags & GUARDED_NEVER) or bool((flags & GUARDED_DEFAULT) - INERT)
    return CorpusConfig(
        name=cfg.get("name") or cfg_dir.name,
        family=family,
        checkpoint=fmt,
        record=RECORDS.get(cfg.get("image_key") or "upstream", "vllm@0.30.0"),
        sku=SKU,
        hardware_mode="shared" if gpumem else "dedicated",
        slice_mib=gpumem or None,
        args=args,
        settings=settings,
        unmodelled=unmodelled,
        guarded=guarded,
        startup=_startup(cfg_dir),
        points=_points(cfg_dir),
        quality=_quality(cfg_dir),
        invalid_points=(cfg_dir / "INVALID").exists(),
    )


def read_results(results_dir: Path) -> List[CorpusConfig]:
    out = []
    for cfg_dir in sorted(Path(results_dir).iterdir()):
        if cfg_dir.is_dir():
            cfg = read_config(cfg_dir)
            if cfg is not None:
                out.append(cfg)
    return out


def write_corpus(configs: Iterable[CorpusConfig], out_dir: Path) -> List[Path]:
    """One JSON file per family, sorted, so the fixture diff is reviewable."""
    by_family: Dict[str, List[Dict[str, Any]]] = {}
    for cfg in configs:
        by_family.setdefault(cfg.family, []).append(cfg.to_dict())
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for family, items in sorted(by_family.items()):
        path = out_dir / f"{family}.json"
        path.write_text(
            json.dumps(sorted(items, key=lambda c: c["name"]), indent=1, sort_keys=True) + "\n"
        )
        paths.append(path)
    return paths


def load_corpus(corpus_dir: Path) -> List[CorpusConfig]:
    out = []
    for path in sorted(Path(corpus_dir).glob("*.json")):
        out.extend(CorpusConfig.from_dict(item) for item in json.loads(path.read_text()))
    return out


#: A point whose prefix-cache hit rate is at least this is a warm repeat: another workload than its cold run.
WARM_HIT = 0.5


def read_replicates(runs_dir: Path, manifest: Mapping[str, Optional[str]]) -> List[CorpusConfig]:
    """Acceptance runs the corpus holds as replicates (FRD-027 §16, the A-05 amendment).

    ``manifest`` maps each run directory to the run whose pod it shares (None for a run on its own pod). A run is named
    by its directory, since the repeats of one deployment share a config.json name; a later suite pass on an earlier
    run's pod is not another boot, so it carries none. A run whose checkpoint key the corpus does not know is refused:
    the reader would take it for the 27B. So is a warm repeat (most of its points served from the prefix cache): a
    second suite pass on one engine is not a first run (the study harness caveat), and its nearly fully cached points
    would widen the cold strata's bands."""
    out = []
    for run, first in sorted(manifest.items()):
        cfg_dir = Path(runs_dir) / run
        if not (cfg_dir / "config.json").exists():
            raise FileNotFoundError(cfg_dir / "config.json")
        key = json.loads((cfg_dir / "config.json").read_text()).get("model") or "bf16"
        if key not in CHECKPOINTS:
            raise ValueError(f"{run}: checkpoint {key!r} is not a grounded family")
        cfg = read_config(cfg_dir)
        assert cfg is not None
        loads = [p for p in cfg.points if p.point != "prefix"]
        if loads and sum(p.prefix_hit_rate >= WARM_HIT for p in loads) * 2 > len(loads):
            raise ValueError(f"{run}: a warm repeat (its points are served from the prefix cache), not a first run")
        cfg.name = run
        if first:
            cfg.startup.api_up_s = None
            cfg.startup.init_s = None
        out.append(cfg)
    return out


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Rebuild the fixture corpus: the study's results plus the acceptance runs a manifest lists as replicates."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--results", type=Path, required=True, help="study/results")
    parser.add_argument("--replicates", type=Path, help="the acceptance results directory")
    parser.add_argument("--manifest", type=Path, help="study/acceptance/replicates.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    configs = read_results(args.results)
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())["runs"]
        configs += read_replicates(args.replicates or args.manifest.parent / "results", manifest)
    names = [c.name for c in configs]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        parser.error(f"configuration names repeat: {duplicates}")
    for path in write_corpus(configs, args.out):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
