"""Build the seeded grounding store from a harness corpus (``python -m llm_memory_calculator.grounding.seed``).

Inputs are the reviewed corpus (one JSON per family, produced by :mod:`.corpus` from the harness output), the
model configs of every checkpoint the harness served, the drafts, and the engine records. The output is a
versioned folder under ``grounding/data/`` that the predictor loads; it is regenerated, never hand-edited
(IMPLEMENTATION_PLAN delivery rule 4).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from ..engine.base import draft_key
from ..engine.record import EngineRecord
from ..model import ModelFacts
from .corpus import load_corpus
from .fit import FitInputs, fit_store
from .store import DATA_DIR, GroundingStore

#: (family, checkpoint format) -> model config file name (models/<name>.json)
CHECKPOINT_CONFIGS = {
    ("Qwen3.8-27B", "bf16"): "qwen3.8-27b",
    ("Qwen3.8-27B", "fp8"): "qwen3.8-27b-fp8",
    ("Qwen3-8B", "bf16"): "qwen3-8b",
    ("Qwen3-8B", "fp8"): "qwen3-8b-fp8",
    ("gpt-oss-20b", "mxfp4"): "gpt-oss-20b",
    ("gpt-oss-20b", "bf16"): "gpt-oss-20b-bf16",
    ("Qwen1.5-MoE-A2.7B", "bf16"): "qwen1.5-moe-a2.7b",
}
#: draft path substring -> draft config file name
DRAFT_CONFIGS = {
    "RedHatAI_Qwen3-8B": "eagle3-qwen3-8b-redhatai",
    "AngelSlim_Qwen3-8B": "eagle3-qwen3-8b-angelslim",
    "Tengyunw_qwen3_8b": "eagle3-qwen3-8b-tengyunw",
    "RedHatAI_gpt-oss-20b": "eagle3-gpt-oss-20b-redhatai",
    "zhuyksir_EAGLE3-gpt-oss-20b": "eagle3-gpt-oss-20b-zhuyksir",
}
#: draft path substring -> the draft's hub id (its grounding identity, engine.draft_key)
DRAFT_HUB_IDS = {
    "RedHatAI_Qwen3-8B": "RedHatAI/Qwen3-8B-speculator.eagle3",
    "AngelSlim_Qwen3-8B": "AngelSlim/Qwen3-8B_eagle3",
    "Tengyunw_qwen3_8b": "Tengyunw/qwen3_8b_eagle3",
    "RedHatAI_gpt-oss-20b": "RedHatAI/gpt-oss-20b-speculator.eagle3",
    "zhuyksir_EAGLE3-gpt-oss-20b": "zhuyksir/EAGLE3-gpt-oss-20b-bf16",
}
RECORD_FILES = {
    "vllm@0.30.0": "vllm-0.30-upstream",
    "bud-vllm@0.9.0-481e481be": "bud-vllm-0.9.0-481e481be",
    "bud-vllm@0.10.0": "bud-vllm-0.10.0",
}


def build_inputs(
    corpus_dir: Path,
    models_dir: Path,
    records_dir: Path,
    *,
    version: str,
    exclude_families: Iterable[str] = (),
) -> FitInputs:
    corpus = load_corpus(corpus_dir)
    models: Dict[Tuple[str, str], ModelFacts] = {}
    for (family, fmt), name in CHECKPOINT_CONFIGS.items():
        path = models_dir / f"{name}.json"
        if path.exists():
            models[(family, fmt)] = ModelFacts.from_config(
                json.loads(path.read_text()), family=family
            )
    drafts = {}
    for needle, name in DRAFT_CONFIGS.items():
        path = models_dir / f"{name}.json"
        if path.exists():
            drafts[needle] = ModelFacts.from_config(
                json.loads(path.read_text()), family=f"draft:{draft_key(DRAFT_HUB_IDS[needle])}"
            )
    records = {}
    for record_id, name in RECORD_FILES.items():
        path = records_dir / f"{name}.json"
        if path.exists():
            records[record_id] = EngineRecord.from_dict(json.loads(path.read_text()))
    return FitInputs(
        corpus=corpus,
        records=records,
        models=models,
        drafts=drafts,
        exclude_families=exclude_families,
        version=version,
    )


def seed(
    corpus_dir: Path,
    models_dir: Path,
    records_dir: Path,
    *,
    version: str,
    out_root: Optional[Path] = None,
    exclude_families: Iterable[str] = (),
) -> GroundingStore:
    store = fit_store(
        build_inputs(
            corpus_dir, models_dir, records_dir, version=version, exclude_families=exclude_families
        )
    )
    if out_root is not False:  # noqa: E712 - False means "don't write"
        store.save(out_root or DATA_DIR)
    return store


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--out", type=Path, default=DATA_DIR)
    args = parser.parse_args(list(argv) if argv is not None else None)
    store = seed(args.corpus, args.models, args.records, version=args.version, out_root=args.out)
    print(
        f"wrote {len(store.entries)} entries and {len(store.outcomes)} boot outcomes to {args.out / args.version}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
