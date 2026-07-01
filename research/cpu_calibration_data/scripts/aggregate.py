#!/usr/bin/env python3
"""Aggregate all bench result JSONs into a CSV + markdown summary.

Reads /home/nived-bud/qwen3_cpu_bench/results/*.json (vllm bench serve output)
plus *.timeout markers, and emits:
  - results/summary.csv      (one row per cell)
  - results/SUMMARY.md       (pivot tables + best-config-per-workload)
"""
import csv
import json
from pathlib import Path

RESULTS = Path("/home/nived-bud/qwen3_cpu_bench/results")

FIELDS = [
    "config", "io", "conc", "num_prompts", "completed", "duration_s",
    "request_throughput", "output_throughput", "total_token_throughput",
    "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
    "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
    "mean_itl_ms", "median_itl_ms",
    "mean_e2el_ms", "median_e2el_ms",
    "status",
]


def parse_meta(d: dict):
    """vllm stores --metadata as a dict under 'metadata' or flattens it."""
    meta = d.get("metadata") or {}
    if isinstance(meta, list):
        m = {}
        for item in meta:
            if isinstance(item, str) and "=" in item:
                k, v = item.split("=", 1)
                m[k] = v
        meta = m
    return meta


def load_rows():
    rows = []
    for p in sorted(RESULTS.glob("*.json")):
        if p.name in ("summary.csv",):
            continue
        if "__" not in p.stem:   # skip non-grid files (e.g. prefix demo)
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        meta = parse_meta(d)
        # fall back to filename: config__io__cN.json
        stem = p.stem
        fn_config = fn_io = fn_conc = None
        if "__" in stem:
            parts = stem.split("__")
            if len(parts) == 3:
                fn_config, fn_io, c = parts
                fn_conc = c.lstrip("c")
        row = {
            "config": meta.get("config", fn_config),
            "io": meta.get("io", fn_io),
            "conc": meta.get("conc", fn_conc),
            "num_prompts": d.get("num_prompts"),
            "completed": d.get("completed"),
            "duration_s": round(d.get("duration", 0), 1),
            "request_throughput": round(d.get("request_throughput", 0), 4),
            "output_throughput": round(d.get("output_throughput", 0), 2),
            "total_token_throughput": round(d.get("total_token_throughput", 0), 2),
            "mean_ttft_ms": round(d.get("mean_ttft_ms", 0), 1),
            "median_ttft_ms": round(d.get("median_ttft_ms", 0), 1),
            "p99_ttft_ms": round(d.get("p99_ttft_ms", 0), 1),
            "mean_tpot_ms": round(d.get("mean_tpot_ms", 0), 2),
            "median_tpot_ms": round(d.get("median_tpot_ms", 0), 2),
            "p99_tpot_ms": round(d.get("p99_tpot_ms", 0), 2),
            "mean_itl_ms": round(d.get("mean_itl_ms", 0), 2),
            "median_itl_ms": round(d.get("median_itl_ms", 0), 2),
            "mean_e2el_ms": round(d.get("mean_e2el_ms", 0), 1),
            "median_e2el_ms": round(d.get("median_e2el_ms", 0), 1),
            "status": "ok",
        }
        rows.append(row)
    for p in sorted(RESULTS.glob("*.timeout")):
        d = json.loads(p.read_text())
        rows.append({"config": d["config"], "io": d["io"], "conc": d["conc"],
                     "status": "timeout"})
    return rows


def write_csv(rows):
    out = RESULTS / "summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    return out


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def write_md(rows):
    md = ["# Qwen3-8B CPU benchmark — summary", ""]
    ok = [r for r in rows if r.get("status") == "ok"]
    md.append(f"Total cells: {len(rows)}  |  ok: {len(ok)}  |  "
              f"timeout: {sum(1 for r in rows if r.get('status')=='timeout')}")
    md.append("")

    configs = sorted({r["config"] for r in ok})
    ios = ["200_300", "512_512", "1024_1024", "4096_1024", "8192_2048"]
    concs = ["1", "20", "50", "100"]

    # Output throughput pivot per config
    for cfg in configs:
        md.append(f"## {cfg} — output tok/s (rows=io, cols=concurrency)")
        md.append("")
        md.append("| io \\ conc | " + " | ".join(concs) + " |")
        md.append("|" + "---|" * (len(concs) + 1))
        for io in ios:
            cells = []
            for c in concs:
                m = [r for r in ok if r["config"] == cfg and str(r["io"]) == io
                     and str(r["conc"]) == c]
                cells.append(f"{m[0]['output_throughput']:.1f}" if m else "·")
            md.append(f"| {io} | " + " | ".join(cells) + " |")
        md.append("")

    # Best config per workload (by output throughput)
    md.append("## Best config per workload (max output tok/s)")
    md.append("")
    md.append("| io | conc | best config | output tok/s | mean TTFT ms | mean TPOT ms |")
    md.append("|---|---|---|---|---|---|")
    for io in ios:
        for c in concs:
            cand = [r for r in ok if str(r["io"]) == io and str(r["conc"]) == c]
            if not cand:
                continue
            best = max(cand, key=lambda r: fnum(r["output_throughput"]) or 0)
            md.append(f"| {io} | {c} | **{best['config']}** | "
                      f"{best['output_throughput']} | {best['mean_ttft_ms']} | "
                      f"{best['mean_tpot_ms']} |")
    md.append("")
    (RESULTS / "SUMMARY.md").write_text("\n".join(md))
    return RESULTS / "SUMMARY.md"


if __name__ == "__main__":
    rows = load_rows()
    c = write_csv(rows)
    m = write_md(rows)
    print(f"wrote {c}")
    print(f"wrote {m}")
    print(f"{len(rows)} cells")
