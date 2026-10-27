#!/usr/bin/env python3
"""KV / memory figures for every model config the simulator ships or this machine caches.

Run it against two source trees (before and after a change) and diff the two JSON files
with ``--diff``: every figure that moved must be explained by a layer-kind correction, and
a dense model must not move at all. The import root is whatever ``PYTHONPATH`` puts first,
so the same script measures both trees.

    PYTHONPATH=<tree>/llm-memory-calculator/src python kv_corpus.py --out after.json \
        --db BudSimulator/data/prepopulated.db --fixtures tests/fixtures/hybrid_configs \
        --hf-cache ~/.cache/huggingface/hub --models /home/bud/models --genz-data <scratch>

    python kv_corpus.py --diff before.json after.json --report diff.md

Corpus: every row of the BudSimulator database, every hybrid fixture, every config.json in
the HuggingFace cache and the model folders that names a decoder geometry, every GGUF in the
model folders (read through bud_component's GGUF reader), and every built-in GenZ model.
Model folders and caches are only read.
"""

import argparse
import glob
import json
import os
import sqlite3
import sys
import traceback
import warnings

SEQS = (4096, 32768)


def _is_llm_config(cfg):
    if not isinstance(cfg, dict):
        return False
    tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    n = tc.get("num_hidden_layers") or tc.get("n_layers") or tc.get("num_layers")
    h = tc.get("hidden_size") or tc.get("d_model") or tc.get("n_embd")
    return bool(n and h)


def collect(args):
    corpus = []
    if args.db and os.path.exists(args.db):
        con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
        for model_id, cfg_json in con.execute("select model_id, config_json from models"):
            try:
                cfg = json.loads(cfg_json)
            except Exception:
                continue
            corpus.append(("db:" + model_id, cfg))
        con.close()
    for root in args.fixtures or []:
        for path in sorted(glob.glob(os.path.join(root, "*.json"))):
            try:
                cfg = json.load(open(path))
            except Exception:
                continue
            corpus.append(("fixture:" + os.path.basename(path), cfg))
    seen = set()
    for root in [args.hf_cache, args.models]:
        if not root or not os.path.isdir(root):
            continue
        for path in sorted(glob.glob(os.path.join(root, "**", "config.json"), recursive=True)):
            real = os.path.realpath(path)
            if real in seen:
                continue
            seen.add(real)
            try:
                cfg = json.load(open(path))
            except Exception:
                continue
            if _is_llm_config(cfg):
                rel = os.path.relpath(path, root)
                corpus.append(("config:" + rel, cfg))
    return corpus


def collect_ggufs(args):
    out = []
    if not args.models or not os.path.isdir(args.models):
        return out
    seen = set()
    for path in sorted(glob.glob(os.path.join(args.models, "**", "*.gguf"), recursive=True)):
        real = os.path.realpath(path)
        base = os.path.basename(path).lower()
        if not os.path.isfile(real):  # dangling link into another machine's layout
            continue
        if real in seen or "mmproj" in base or os.path.getsize(real) < 10 * 1024 * 1024:
            continue
        seen.add(real)
        out.append(path)
    return out


def _gb(x):
    return None if x is None else round(float(x), 9)


def measure_config(name, cfg, genz_data):
    from llm_memory_calculator import calculate_memory
    from llm_memory_calculator.calculator import ModelMemoryCalculator

    row = {"name": name}
    try:
        calc = ModelMemoryCalculator()
        calc.model_type = calc.detect_model_type(cfg)
        row["model_type"] = calc.model_type
        row["attention_type"] = calc.detect_attention_type(cfg)
        for seq in SEQS:
            row[f"kv_gb@{seq}"] = _gb(calc.calculate_kv_cache(cfg, 1, seq, "bf16"))
        bd = calc.kv_cache_breakdown(cfg, 1, 4096, "bf16")
        row["kv_full_layers"] = bd.get("num_full_layers")
        row["kv_sliding_layers"] = bd.get("num_sliding_layers")
        row["kv_free_layers"] = bd.get("num_no_kv_layers")
        row["kv_source"] = bd.get("source")
        row["state_gb"] = _gb(calc.calculate_state_memory(cfg, 1, "bf16"))
    except Exception as e:  # recorded, not fatal: a crash is itself a figure to diff
        row["calc_error"] = f"{type(e).__name__}: {e}"
    try:
        rep = calculate_memory(cfg, batch_size=1, seq_length=4096, precision="bf16")
        row["report_kv_gb"] = _gb(rep.kv_cache_bytes / 1e9)
        row["report_state_gb"] = _gb(rep.state_memory_bytes / 1e9)
        row["report_weight_gb"] = _gb(rep.weight_memory_bytes / 1e9)
        row["report_params"] = int(rep.parameter_count)
        row["report_total_gb"] = _gb(rep.total_memory_bytes / 1e9)
    except Exception as e:
        row["report_error"] = f"{type(e).__name__}: {e}"
    row.update(measure_genz(cfg, genz_data))
    return row


def measure_genz(cfg_or_name, genz_data):
    """GenZ's own view: per-layer kinds, its KV size, and its decode graph's KV bytes."""
    out = {}
    try:
        from llm_memory_calculator.genz.Models.get_language_model import (
            create_full_decode_model,
            get_configs,
        )
        from llm_memory_calculator.genz.analyse_model import get_model_df, get_summary_table

        mc = get_configs(cfg_or_name)
        kinds = {}
        reps = mc.num_decoder_layers / max(len(mc.layer_type), 1)
        for lt in mc.layer_type:
            kinds[lt[0]] = kinds.get(lt[0], 0) + reps
        out["genz_layer_kinds"] = {k: int(v) for k, v in sorted(kinds.items())}
        out["genz_kv_size"] = int(mc.get_kv_size())
        path = create_full_decode_model(
            name=mc, input_sequence_length=4096, output_gen_tokens=0, data_path=genz_data
        )
        df = get_model_df(path, batch_size=1, data_path=genz_data, model_characterstics=True)
        st = get_summary_table(df, model_characterstics=True)
        out["genz_decode_kv_mb"] = round(float(st["KV Cache (MB)"].values[0]), 6)
        out["genz_decode_weights_mb"] = round(float(st["Total Weights (MB)"].values[0]), 6)
    except Exception as e:
        out["genz_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return out


def measure_gguf(path, repo_root, genz_data):
    sys.path.insert(0, repo_root)
    row = {"name": "gguf:" + os.path.basename(path)}
    try:
        import bud_component

        cfg, _, _ = bud_component.gguf_to_hf_config(path)
        row["gguf_config_keys"] = sorted(k for k in cfg if k != "name")
        row.update(
            {k: v for k, v in measure_config(row["name"], cfg, genz_data).items() if k != "name"}
        )
    except Exception as e:
        row["gguf_error"] = f"{type(e).__name__}: {e}"
    finally:
        sys.path.pop(0)
    return row


def run(args):
    warnings.simplefilter("ignore")
    os.makedirs(os.path.join(args.genz_data, "model"), exist_ok=True)
    os.makedirs(os.path.join(args.genz_data, "sparsity"), exist_ok=True)
    rows = []
    for name, cfg in collect(args):
        rows.append(measure_config(name, cfg, args.genz_data))
    for path in collect_ggufs(args):
        rows.append(measure_gguf(path, args.repo_root, args.genz_data))
    try:
        from llm_memory_calculator.genz.Models.default_models import MODEL_DICT

        for model_name in sorted(MODEL_DICT.list_models()):
            row = {"name": "genz:" + model_name}
            row.update(measure_genz(model_name, args.genz_data))
            rows.append(row)
    except Exception:
        traceback.print_exc()
    json.dump(rows, open(args.out, "w"), indent=1, sort_keys=True)
    print(f"{len(rows)} rows -> {args.out}")


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def diff(before_path, after_path, report):
    before = {r["name"]: r for r in json.load(open(before_path))}
    after = {r["name"]: r for r in json.load(open(after_path))}
    changed, unchanged = [], 0
    for name in sorted(set(before) | set(after)):
        b, a = before.get(name, {}), after.get(name, {})
        moves = []
        for key in sorted(set(b) | set(a)):
            if key == "name":
                continue
            bv, av = b.get(key), a.get(key)
            if bv == av:
                continue
            if _num(bv) and _num(av) and abs(bv - av) <= 1e-9 * max(1.0, abs(bv)):
                continue
            moves.append((key, bv, av))
        if moves:
            changed.append((name, a.get("model_type") or b.get("model_type"), moves))
        else:
            unchanged += 1
    lines = [
        "# KV / memory corpus diff",
        "",
        f"before: `{os.path.basename(before_path)}`  after: `{os.path.basename(after_path)}`",
        "",
        f"{len(set(before) | set(after))} configs measured; {unchanged} unchanged; "
        f"{len(changed)} changed.",
        "",
    ]
    for name, mtype, moves in changed:
        lines.append(f"## {name} ({mtype})")
        lines.append("")
        lines.append("| figure | before | after |")
        lines.append("|---|---|---|")
        for key, bv, av in moves:
            lines.append(f"| {key} | {json.dumps(bv)} | {json.dumps(av)} |")
        lines.append("")
    text = "\n".join(lines) + "\n"
    if report:
        open(report, "w").write(text)
    print(text[:4000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--db")
    ap.add_argument("--fixtures", nargs="*")
    ap.add_argument("--hf-cache")
    ap.add_argument("--models")
    ap.add_argument("--repo-root", default=os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    ap.add_argument("--genz-data")
    ap.add_argument("--diff", nargs=2)
    ap.add_argument("--report")
    args = ap.parse_args()
    if args.diff:
        diff(args.diff[0], args.diff[1], args.report)
    else:
        run(args)


if __name__ == "__main__":
    main()
