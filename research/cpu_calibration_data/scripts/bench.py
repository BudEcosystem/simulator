#!/usr/bin/env python3
"""Qwen3-8B CPU benchmark driver for vLLM 0.24.0+cpu.

For each server CONFIG we launch one `vllm serve` process, wait until it is
healthy, then run every requested workload cell (concurrency x input/output)
against it with `vllm bench serve`, saving each cell's JSON result. Resumable:
a cell whose result JSON already exists is skipped.

Run from /home/nived-bud/qwen3_cpu_bench (NOT the vLLM source tree, which
shadows the installed wheel).
"""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BENCH_DIR = Path("/home/nived-bud/qwen3_cpu_bench")
RESULTS = Path(os.environ.get("BENCH_RESULTS", str(BENCH_DIR / "results")))
LOGS = BENCH_DIR / "logs"
VENV_BIN = Path("/home/nived-bud/vllm/.venv/bin")
VLLM = str(VENV_BIN / "vllm")
MODEL = os.environ.get("BENCH_MODEL", "Qwen/Qwen3-8B")
PORT = 8100
IOMP = "/home/nived-bud/vllm/.venv/lib/libiomp5.so"
CC_BIN = "/home/nived-bud/micromamba/envs/cc/bin"  # conda-forge gcc/g++ 15.2
MAX_MODEL_LEN = 11264  # >= 8192+2048

# ---- workload grid -------------------------------------------------------
# (input_len, output_len)
IO = {
    "200_300": (200, 300),
    "512_512": (512, 512),
    "1024_1024": (1024, 1024),
    "4096_1024": (4096, 1024),
    "8192_2048": (8192, 2048),
}
CONC = [1, 20, 50, 100]


def num_prompts_for(conc: int, out_len: int) -> int:
    """Pick prompt count given ~10 tok/s/stream on this CPU: ~one wave at the
    target concurrency (two for cheap/short outputs), fewer when decode is long."""
    if conc == 1:
        return 3 if out_len <= 1024 else 2
    if out_len <= 512:
        return conc * 2
    return conc  # one wave for heavy outputs


# ---- server configs ------------------------------------------------------
# Each config: extra `vllm serve` args + extra env. Base args/env added below.
CONFIGS = {
    "c0_baseline": dict(args=[], env={}),
    "c1_tp2_numa": dict(
        args=["--tensor-parallel-size", "2",
              "--distributed-executor-backend", "mp"],
        env={"VLLM_CPU_KVCACHE_SPACE": "60"},
    ),
    "c2_prefix_cache": dict(args=["--enable-prefix-caching"], env={}),
    "c3_chunked2k": dict(
        args=["--enable-chunked-prefill", "--max-num-batched-tokens", "2048"],
        env={},
    ),
    "c4_socket0_bind": dict(
        args=[],
        # bind all OMP threads to physical cores of NUMA node 0 only
        env={"VLLM_CPU_OMP_THREADS_BIND": "0-63"},
    ),
    "c5_combo": dict(
        args=["--tensor-parallel-size", "2",
              "--distributed-executor-backend", "mp",
              "--enable-prefix-caching"],
        env={"VLLM_CPU_KVCACHE_SPACE": "60"},
    ),
    # chunked prefill ON TOP of the winning TP=2 config (does it cut the
    # 25s TTFT seen at long-prefill high-concurrency cells?)
    # NOTE: chunked prefill is ON BY DEFAULT (max_num_batched_tokens=4096), so
    # this is really the 2048 point of a max-num-batched-tokens sweep.
    "c6_tp2_chunked": dict(
        args=["--tensor-parallel-size", "2",
              "--distributed-executor-backend", "mp",
              "--enable-chunked-prefill",
              "--max-num-batched-tokens", "2048"],
        env={"VLLM_CPU_KVCACHE_SPACE": "60"},
    ),
    # ---- round 2: remaining CPU-relevant levers ----
    # TP=4: 4 ranks (2 per NUMA node). auto-bind caps at 1 node/worker, so we
    # bind manually: 4 groups of 32 physical cores (2 groups per socket).
    "c7_tp4": dict(
        args=["--tensor-parallel-size", "4",
              "--distributed-executor-backend", "mp"],
        env={"VLLM_CPU_KVCACHE_SPACE": "30",
             "VLLM_CPU_OMP_THREADS_BIND": "0-31|32-63|64-95|96-127"},
    ),
    # max-num-batched-tokens sweep (the real "chunking" knob): 1024 and
    # 11264 (>= max prefill => effectively NO chunking) bracket c6(2048)/c1(4096).
    "c8_tp2_bt1024": dict(
        args=["--tensor-parallel-size", "2",
              "--distributed-executor-backend", "mp",
              "--max-num-batched-tokens", "1024"],
        env={"VLLM_CPU_KVCACHE_SPACE": "60"},
    ),
    "c9_tp2_nochunk": dict(
        args=["--tensor-parallel-size", "2",
              "--distributed-executor-backend", "mp",
              "--max-num-batched-tokens", "11264"],
        env={"VLLM_CPU_KVCACHE_SPACE": "60"},
    ),
    # int8 KV cache — shrinks KV to relieve the KV-pressured long-seq cells.
    "c10_tp2_int8kv": dict(
        args=["--tensor-parallel-size", "2",
              "--distributed-executor-backend", "mp",
              "--kv-cache-dtype", "fp8_e5m2"],
        env={"VLLM_CPU_KVCACHE_SPACE": "60"},
    ),
}


def base_env(cfg_env: dict) -> dict:
    env = os.environ.copy()
    # conda g++ for torch-inductor (CPU backend requires a C++ compiler)
    env["PATH"] = CC_BIN + ":" + env.get("PATH", "")
    env["CC"] = f"{CC_BIN}/gcc"
    env["CXX"] = f"{CC_BIN}/g++"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["TORCHINDUCTOR_CACHE_DIR"] = "/home/nived-bud/qwen3_cpu_bench/.inductor_cache"
    env["LD_PRELOAD"] = f"{IOMP}:" + env.get("LD_PRELOAD", "")
    env["VLLM_CPU_KVCACHE_SPACE"] = "110"   # GB host RAM for KV (per rank)
    env["VLLM_LOGGING_LEVEL"] = "INFO"
    env["HF_HUB_OFFLINE"] = "1"
    env.update(cfg_env)
    return env


def base_server_args(cfg_args: list) -> list:
    args = [
        VLLM, "serve", MODEL,
        "--dtype", "bfloat16",
        "--port", str(PORT),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--trust-remote-code",
    ]
    return args + cfg_args


def port_free() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", PORT)) != 0


def wait_health(timeout=600) -> bool:
    url = f"http://127.0.0.1:{PORT}/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def start_server(config_name: str):
    cfg = CONFIGS[config_name]
    env = base_env(cfg["env"])
    args = base_server_args(cfg["args"])
    log_path = LOGS / f"server_{config_name}.log"
    log = open(log_path, "w")
    log.write(f"# cmd: {' '.join(args)}\n# extra_env: {cfg['env']}\n")
    log.flush()
    print(f"[{config_name}] launching server -> {log_path}")
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT,
                            env=env, cwd=str(BENCH_DIR),
                            preexec_fn=os.setsid)
    return proc, log


def stop_server(proc, log):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        pass
    try:
        proc.wait(timeout=60)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    try:
        log.close()
    except Exception:
        pass
    # wait for port to free
    for _ in range(30):
        if port_free():
            break
        time.sleep(2)


def result_path(config_name: str, io_key: str, conc: int) -> Path:
    return RESULTS / f"{config_name}__{io_key}__c{conc}.json"


def run_cell(config_name: str, io_key: str, conc: int, timeout: int) -> str:
    in_len, out_len = IO[io_key]
    n = num_prompts_for(conc, out_len)
    out_file = result_path(config_name, io_key, conc)
    if out_file.exists():
        print(f"  [skip] {out_file.name} exists")
        return "skip"
    cmd = [
        VLLM, "bench", "serve",
        "--backend", "vllm",
        "--model", MODEL,
        "--port", str(PORT),
        "--dataset-name", "random",
        "--random-input-len", str(in_len),
        "--random-output-len", str(out_len),
        "--num-prompts", str(n),
        "--max-concurrency", str(conc),
        "--ignore-eos",
        "--seed", "1234",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--save-result",
        "--result-dir", str(RESULTS),
        "--result-filename", out_file.name,
        "--metadata", f"config={config_name}", f"io={io_key}", f"conc={conc}",
    ]
    env = base_env(CONFIGS[config_name]["env"])
    cell_log = LOGS / f"cell_{config_name}__{io_key}__c{conc}.log"
    print(f"  [run] {io_key} conc={conc} n={n} (timeout {timeout}s)")
    t0 = time.time()
    try:
        with open(cell_log, "w") as cl:
            subprocess.run(cmd, env=env, cwd=str(BENCH_DIR), check=True,
                          stdout=cl, stderr=subprocess.STDOUT, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"    -> TIMEOUT after {timeout}s")
        out_file.with_suffix(".timeout").write_text(
            json.dumps({"status": "timeout", "timeout_s": timeout,
                        "config": config_name, "io": io_key, "conc": conc}))
        return "timeout"
    except subprocess.CalledProcessError as e:
        print(f"    -> FAILED rc={e.returncode} (see {cell_log.name})")
        return "fail"
    dt = time.time() - t0
    print(f"    -> done in {dt:.0f}s")
    return "ok"


def cell_timeout(conc: int, out_len: int) -> int:
    # wall-clock budget; heavy decode cells get more, capped at 40 min so a
    # single infeasible cell can't stall the whole run (timeout => recorded).
    base = 600
    factor = (out_len / 300.0) * (1 + conc / 50.0)
    return int(min(2400, max(base, base * factor)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--io", nargs="*", default=list(IO),
                    help="subset of io keys")
    ap.add_argument("--conc", nargs="*", type=int, default=CONC)
    ap.add_argument("--max-cell-timeout", type=int, default=None)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)

    proc, log = start_server(args.config)
    try:
        if not wait_health():
            print(f"[{args.config}] server failed to become healthy; abort.")
            tail = (LOGS / f"server_{args.config}.log").read_text()[-3000:]
            print(tail)
            return 2
        print(f"[{args.config}] server healthy.")
        for io_key in args.io:
            for conc in args.conc:
                _, out_len = IO[io_key]
                to = args.max_cell_timeout or cell_timeout(conc, out_len)
                run_cell(args.config, io_key, conc, to)
    finally:
        print(f"[{args.config}] stopping server.")
        stop_server(proc, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
