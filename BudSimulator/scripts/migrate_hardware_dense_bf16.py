#!/usr/bin/env python3
"""Idempotent migration: correct sparse/FP32/mis-scaled FLOPS in the hardware DB to DENSE bf16 per-chip.

Part of the accuracy remediation (F2/C6/C7). The prepopulated DB stored several FLOPS values that were
the 2:4-SPARSE peak, the FP32 peak, or pod/board-scaled numbers, not the dense bf16 per-chip tensor
throughput that LLM GEMMs achieve. This rewrites them to the datasheet dense bf16 value. Each target is
cited inline. Idempotent: `UPDATE ... WHERE flops <> ?` so re-runs change 0 rows.

Intentionally NOT touched (review FIX-1): Intel Data Center GPU Max (MAX1100, MAX1550) and PonteVecchio_GPU
— their dense bf16 XMX peak is not confidently citable to a single datasheet line, so we do not guess
(no magic numbers). They remain real_values=0 (estimated) pending a sourced value.

Usage:
    python BudSimulator/scripts/migrate_hardware_dense_bf16.py [--dry-run]
"""
import argparse
import os
import sqlite3
import sys

# name -> (dense bf16 TFLOPS per chip, citation). DENSE, never 2:4-sparse / FP32 / pod-scaled.
DENSE_BF16 = {
    "H100_GPU":      (989.5, "NVIDIA H100 SXM datasheet dense BF16 989.5 (1979 = 2:4 sparse)"),
    "H200_GPU":      (989.5, "H200 = H100 compute die, dense BF16 989.5"),
    "GH200_GPU":     (989.5, "Grace-Hopper = one H100 SXM die, dense BF16 989.5 (1979 = sparse)"),
    "H100_SXM_GPU":  (989.5, "NVIDIA H100 SXM5 dense BF16 989.5 (1979 = sparse)"),
    "H800_GPU":      (989.5, "H800 = export-capped H100 SXM die; GEMM units uncapped → dense BF16 989.5 (NVLink capped, not compute)"),
    "B100":          (1750.0, "NVIDIA Blackwell B100 dense BF16 1750 (3500 = 2:4 sparse)"),
    "GB200":         (2250.0, "GB200 per-chip = one B200 dense BF16 2250 (4500 = sparse / 2-die)"),
    "MI300X_GPU":    (1307.0, "AMD MI300X CDNA3 dense BF16 1307.4 (1600 = marketing/sparse)"),
    "RTX4090_GPU":   (330.3, "NVIDIA Ada RTX 4090 dense BF16 330.3 = 165.2 FP16-tensor ×2 (661 = 2:4 sparse)"),
    "TPU_v2":        (46.0, "Google TPU v2 per-chip BF16 46 (180 = pod/board-scaled)"),
    "TPU_v3":        (123.0, "Google TPU v3 per-chip BF16 123 (420 = pod/board-scaled)"),
    "V100_GPU":      (125.0, "NVIDIA V100 FP16 tensor 125 (157 mixes FP32-derived)"),
}


def _resolve_db_path() -> str:
    env = os.environ.get("BUDSIM_HARDWARE_DB") or os.environ.get("LLMMC_HARDWARE_DB")
    if env and os.path.exists(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "..", "data", "prepopulated.db")
    return os.path.normpath(cand)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print pre/post without writing")
    ap.add_argument("--db", default=None, help="path to hardware DB")
    args = ap.parse_args()

    db_path = args.db or _resolve_db_path()
    if not os.path.exists(db_path):
        print(f"ERROR: DB not found at {db_path}", file=sys.stderr)
        return 2
    print(f"DB: {db_path}{'  (DRY RUN)' if args.dry_run else ''}")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    changed = 0
    for name, (target, cite) in DENSE_BF16.items():
        row = cur.execute("SELECT flops FROM hardware WHERE name = ?", (name,)).fetchone()
        if row is None:
            print(f"  - {name:16} ABSENT (skip)")
            continue
        current = row[0]
        if abs(current - target) < 1e-6:
            print(f"  = {name:16} {current} (already dense, no-op)")
            continue
        print(f"  ~ {name:16} {current} -> {target}   [{cite}]")
        if not args.dry_run:
            # idempotent: only writes when the value actually differs
            cur.execute("UPDATE hardware SET flops = ? WHERE name = ? AND flops <> ?",
                        (target, name, target))
            changed += cur.rowcount
    # sanity: CHECK(flops>0) holds for all targets (all are > 0)
    if not args.dry_run:
        conn.commit()
    print(f"{'Would change' if args.dry_run else 'Changed'} {changed} rows.")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
