"""FRD-027 Revision 2: replay measured deployments through the profile path, for L5 and the gates (§20.2 L5, §20.6).

Every usable point of a corpus configuration that has a step profile is predicted the way a plan predicts it:
- the engine's limits;
- the profile's steps through the scheduler simulation;
- the configuration's own settings and workload.

The replay applies no correction, so its residuals are what L1–L4 alone get wrong: NFR-6's grade, and L5's input.
Configurations without a profile (another kernel class, a HAMi slice) are skipped: the fitted predictor serves them.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .residual import QUANTITIES, Residual, cross_validated_errors, fit, load_bucket

#: the study suite's real-text points ran with --sharegpt-output-len 512
REAL_OUTPUT_CAP = 512


def replay(inputs: Any, profiles: Any, *, store: Any = None, families: Optional[Iterable[str]] = None) -> List[Residual]:
    """The residuals of every profiled corpus point (uncorrected)."""
    from ..engine import engine_for
    from ..grounding.fit import device_for, model_for, predict_settings, usable_point, workload_for
    from .predict import find_profile, predict

    wanted = set(families) if families is not None else None
    out: List[Residual] = []
    for cfg in inputs.corpus:
        if wanted is not None and cfg.family not in wanted:
            continue
        record = inputs.records.get(cfg.record)
        model = model_for(inputs, cfg)
        if record is None or model is None or cfg.guarded:
            continue
        engine = engine_for(record, store, profiles)
        settings = predict_settings(cfg)
        device = device_for(cfg)
        try:
            profile = find_profile(engine, profiles, settings, model, device)
        except Exception:  # noqa: BLE001 - a configuration the engine refuses has nothing to replay
            continue
        if profile is None:
            continue
        for p in cfg.points:
            if not usable_point(cfg, p) or p.point == "prefix" or (p.prefix_hit_rate or 0.0) >= 0.5:
                continue
            w = workload_for(p)
            if w.workload_class != "synthetic":
                w = dataclasses.replace(w, output_tokens_max=REAL_OUTPUT_CAP)
            try:
                pred = predict(engine, profile, settings, w, model, device)
            except Exception:  # noqa: BLE001
                continue
            bucket = load_bucket(p.concurrency, p.workload_class)
            measured = {"tpot": p.tpot_ms, "throughput": p.output_tok_s, "ttft": p.ttft_ms}
            predicted = {"tpot": pred.tpot_ms.value, "throughput": pred.throughput_tok_s.value, "ttft": pred.ttft_ms.value}
            for q in QUANTITIES:
                if measured[q] and measured[q] > 0 and predicted[q] > 0:
                    out.append(
                        Residual(
                            sku=profile.sku,
                            engine=profile.engine,
                            family=profile.model,
                            quantity=q,
                            bucket=bucket,
                            predicted=float(predicted[q]),
                            measured=float(measured[q]),
                            config=cfg.name,
                            queueing=bool(pred.queueing),
                        )
                    )
    return out


#: flags a configuration's arguments may carry that do not change engine performance, so the planner can still
#: render it: logging, config loading, and API-side parsing
SERVING_ONLY_FLAGS = frozenset(
    {
        "trust-remote-code",
        "uvicorn-log-level",
        "disable-log-requests",
        "disable-log-stats",
        "served-model-name",
        "enable-auto-tool-choice",
        "reasoning-parser",
        "tool-call-parser",
        "enable-prompt-tokens-details",
    }
)


def _plannable(cfg: Any) -> bool:
    """A configuration the planner could have rendered: its performance-relevant flags are all settings."""
    s = cfg.settings
    return (
        not cfg.guarded
        and not s.get("lora.max_adapters")
        and cfg.hardware_mode == "dedicated"
        and set(cfg.unmodelled or ()) <= SERVING_ONLY_FLAGS
    )


def choice_replay(
    inputs: Any, profiles: Any = None, *, store: Any = None, corrections: Any = None, families: Optional[Iterable[str]] = None
) -> List[dict]:
    """A-11 replayed on the corpus: for every load point measured under several plannable configurations of one
    deployment (family, checkpoint, build, SKU), the configuration the planner's rule picks and how its measured
    throughput compares with the best measured.

    The rule is FR-GND-2's: the highest predicted throughput wins only when its band separates above the default
    configuration's (the one with no settings beyond the checkpoint); otherwise the default stays.
    """
    from collections import defaultdict

    from ..engine import engine_for
    from ..grounding.fit import device_for, draft_for, model_for, predict_settings, usable_point, workload_for

    wanted = set(families) if families is not None else None
    groups: Dict[tuple, List[tuple]] = defaultdict(list)
    for cfg in inputs.corpus:
        if (wanted is not None and cfg.family not in wanted) or not _plannable(cfg):
            continue
        record, model = inputs.records.get(cfg.record), model_for(inputs, cfg)
        if record is None or model is None:
            continue
        for p in cfg.points:
            if not usable_point(cfg, p) or p.point == "prefix" or (p.prefix_hit_rate or 0.0) >= 0.5:
                continue
            key = (cfg.family, cfg.checkpoint, cfg.record, cfg.sku, p.point, p.concurrency, p.workload_class,
                   round(p.input_tokens), round(p.output_tokens))
            groups[key].append((cfg, p, record, model))
    out: List[dict] = []
    for key, members in sorted(groups.items(), key=lambda kv: kv[0]):
        by_cfg = {}
        for cfg, p, record, model in members:
            by_cfg.setdefault(cfg.name, (cfg, p, record, model))  # one point per configuration and load
        if len(by_cfg) < 2:
            continue
        preds = {}
        for name, (cfg, p, record, model) in by_cfg.items():
            engine = engine_for(record, store, profiles, corrections)
            w = workload_for(p)
            if w.workload_class != "synthetic":
                w = dataclasses.replace(w, output_tokens_max=REAL_OUTPUT_CAP)
            try:
                preds[name] = engine.predict(predict_settings(cfg), w, model, device_for(cfg), draft=draft_for(inputs, cfg))
            except Exception:  # noqa: BLE001 - the engine refuses it: the planner would not pick it
                continue
        if len(preds) < 2:
            continue
        defaults = [n for n in preds if not predict_settings(by_cfg[n][0])]
        default = sorted(defaults)[0] if defaults else None
        best_pred = max(preds, key=lambda n: preds[n].throughput_tok_s.value)
        chosen = best_pred
        if default is not None and not preds[best_pred].throughput_tok_s.separates_above(preds[default].throughput_tok_s):
            chosen = default
        measured = {n: by_cfg[n][1].output_tok_s for n in preds}
        best_measured = max(measured, key=measured.get)
        out.append(
            {
                "family": key[0],
                "checkpoint": key[1],
                "record": key[2],
                "point": key[4],
                "concurrency": key[5],
                "class": key[6],
                "candidates": len(preds),
                "chosen": chosen,
                "best": best_measured,
                "default": default,
                "ratio": measured[chosen] / measured[best_measured],
                "default_ratio": (measured[default] / measured[best_measured]) if default else None,
            }
        )
    return out


def grade(residuals: Sequence[Residual]) -> List[dict]:
    """NFR-6 (before correction) and NFR-7 (after a correction fitted without the point's configuration), per family,
    quantity, workload class and queueing: MAPE and p90."""
    after, before, gaps = cross_validated_errors(residuals)
    rows = []
    for key in sorted(after):
        family, quantity, cls, queueing = key

        def stats(v: Sequence[float]) -> dict:
            ordered = sorted(v)
            return {
                "mape": statistics.fmean(v),
                "p90": ordered[min(len(ordered) - 1, max(0, math.ceil(0.9 * len(ordered)) - 1))],
            }

        rows.append(
            {
                "family": family,
                "quantity": quantity,
                "class": cls,
                "queueing": queueing,
                "n": len(after[key]),
                "gaps": gaps.get(key, 0),
                "before": stats(before[key]),
                "after": stats(after[key]),
            }
        )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Replay a corpus through the package's (or a directory's) profiles; fit L5's corrections; print the grade."""
    from ..grounding.seed import build_inputs
    from .store import ProfileStore

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, action="append", required=True, help="a corpus directory (repeatable)")
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--profiles", type=Path, help="a directory of profiles (default: the package's)")
    ap.add_argument("--family", action="append", help="replay only these families")
    ap.add_argument("--out", type=Path, help="write the corrections here (bud-perf-corrections/1)")
    ap.add_argument("--residuals", type=Path, help="write the replayed residuals here (JSON lines)")
    a = ap.parse_args(argv)
    profiles = ProfileStore.load(a.profiles)
    residuals: List[Residual] = []
    for corpus in a.corpus:
        inputs = build_inputs(corpus, a.models, a.records, version="replay")
        residuals += replay(inputs, profiles, families=a.family)
    if a.residuals:
        a.residuals.write_text("".join(json.dumps(dataclasses.asdict(r)) + "\n" for r in residuals))
    for row in grade(residuals):
        print(
            f"{row['family'][:20]:20} {row['quantity']:10} {row['class']:9} {'queue' if row['queueing'] else '':5} "
            f"n {row['n']:4}  before MAPE {row['before']['mape']:6.1%} p90 {row['before']['p90']:6.1%}  "
            f"after MAPE {row['after']['mape']:6.1%} p90 {row['after']['p90']:6.1%}  gaps {row['gaps']}"
        )
    if a.out:
        a.out.write_text(json.dumps(fit(residuals).to_dict(), indent=1))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
