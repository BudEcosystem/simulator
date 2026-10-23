"""Validation records from the harness's quality runs (FRD-027 §12, FR-VAL-3; IMPLEMENTATION_PLAN 2.1).

For every configuration with GSM8K results, each numerics setting it changes relative to its family's baseline on
the same engine record becomes a record, judged with the §12 tolerance. A KV-dtype record on a hybrid is held at
``fail`` until a 128K needle-in-a-haystack run exists for that family and build (GSM8K alone is not sufficient,
FRD-027 §10.7).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..runtime.validation import ValidationRecord, judge
from .corpus import CorpusConfig, load_corpus

HYBRID_FAMILIES = {"Qwen3.8-27B"}
#: Builds measured after the study, with the run their evidence comes from (default: the 10-15 study).
STUDY_PROVENANCE = {"harness": "study-harness@2026-10-16", "date": "2026-10-16", "run_id": "study-2026-10-15"}
PROVENANCE = {
    "bud-vllm@0.10.0": {"harness": "round7@2026-10-18", "date": "2026-10-18", "run_id": "frd027-round7-tcs"},
}
#: A new build -> the build whose evidence it may inherit for kernels it declares unchanged. bud-vllm 0.10.0 is
#: upstream v0.31.0 re-tagged.
INHERIT = {"bud-vllm@0.10.0": "vllm@0.31.0"}
GARBAGE_RATE = 0.01
NUMERICS = ("weights.format", "weights.online", "spec.method", "state.dtype", "kv.dtype")


def _accuracy(cfg: CorpusConfig) -> Optional[Tuple[float, int, int]]:
    z = cfg.quality.get("gsm8k_zeroshot")
    if not z or z.get("accuracy") is None:
        return None
    return float(z["accuracy"]) * 100.0, int(z.get("garbage") or 0), int(z.get("n") or 1)


def _numerics(cfg: CorpusConfig) -> Dict[str, Any]:
    s = cfg.settings
    out = {}
    if s.get("weights.online"):
        out["weights.online"] = str(s.get("weights.format"))
    elif cfg.checkpoint not in ("bf16", "fp16") and cfg.family != "gpt-oss-20b":
        out["weights.format"] = cfg.checkpoint
    elif cfg.family == "gpt-oss-20b" and cfg.checkpoint != "mxfp4":
        out["weights.format"] = cfg.checkpoint
    for key in ("spec.method", "state.dtype", "kv.dtype"):
        if s.get(key):
            out[key] = s[key]
    return out


def seed_records(corpus: List[CorpusConfig]) -> List[ValidationRecord]:
    by_key: Dict[Tuple[str, str], List[CorpusConfig]] = {}
    for cfg in corpus:
        if _accuracy(cfg) is not None and not cfg.guarded:
            by_key.setdefault((cfg.family, cfg.record), []).append(cfg)
    out: Dict[Tuple[str, str, str], ValidationRecord] = {}

    def baseline_of(family: str, record: str):
        cfgs = by_key.get((family, record), [])
        sane = [c for c in cfgs if not _numerics(c) and _accuracy(c)[0] >= 50.0]
        if not sane:
            return None, None
        best = max(sane, key=lambda c: _accuracy(c)[0])
        return _accuracy(best), best

    for (family, record), cfgs in sorted(by_key.items()):
        base, base_cfg = baseline_of(family, record)
        if base is None:
            # no sane baseline on this build: compare with the family's upstream baseline (named in the evidence)
            base, base_cfg = baseline_of(family, "vllm@0.30.0")
        for cfg in sorted(cfgs, key=lambda c: c.name):
            numerics = _numerics(cfg)
            if not numerics:
                continue
            acc, garbage, n = _accuracy(cfg)
            if base is None:
                result, reason = ("fail", "no baseline on this build")
            else:
                bad = garbage / max(n, 1) >= GARBAGE_RATE
                result, reason = judge(base[0], acc, garbage=garbage if bad else 0)
            for key, value in numerics.items():
                setting = f"{key}={value}"
                rec = ValidationRecord(
                    target=f"family:{family}",
                    record=record,
                    setting=setting,
                    result=result,
                    evidence={
                        "gsm8k_zeroshot": round(acc, 2),
                        "baseline": round(base[0], 2) if base else None,
                        "garbage": garbage,
                        "n": n,
                        "config": cfg.name,
                        "baseline_config": base_cfg.name if base_cfg else None,
                        "gsm8k_fewshot": (cfg.quality.get("gsm8k_fewshot") or {}).get("accuracy"),
                    },
                    reason=reason,
                    provenance=dict(PROVENANCE.get(record, STUDY_PROVENANCE)),
                )
                if key == "kv.dtype" and family in HYBRID_FAMILIES and result == "pass":
                    rec.result = "fail"
                    rec.reason = (
                        "GSM8K within tolerance, but a hybrid needs the 128K needle-in-a-haystack record on this "
                        "build before fp8 KV is eligible (FRD-027 §10.7)"
                    )
                k = (rec.target, record, setting)
                prev = out.get(k)
                # a single failing combination does not fail a setting that passed alone; keep the single-factor run
                if prev is None or (len(_numerics(cfg)) < prev.evidence.get("_factors", 99)):
                    rec.evidence["_factors"] = len(numerics)
                    out[k] = rec
    records = sorted(out.values(), key=lambda r: (r.target, r.record, r.setting))
    for r in records:
        r.evidence.pop("_factors", None)
    return records


def write(
    corpus_dir: Path, out_path: Path, only_records: Optional[List[str]] = None
) -> List[ValidationRecord]:
    """Write the records seeded from the corpus; ``only_records`` keeps those engine records (one round's file)."""
    records = seed_records(load_corpus(corpus_dir))
    if only_records:
        records = [r for r in records if r.record in only_records]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "version": out_path.stem,
                "inherit": dict(INHERIT),
                "records": [r.to_dict() for r in records],
            },
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )
    return records


if __name__ == "__main__":
    import sys

    corpus, out = Path(sys.argv[1]), Path(sys.argv[2])
    for r in write(corpus, out, sys.argv[3:] or None):
        print(r.target, r.record, r.setting, r.result, r.reason)
