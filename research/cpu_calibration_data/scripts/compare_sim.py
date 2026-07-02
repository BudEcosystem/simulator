#!/usr/bin/env python3
"""Compare BudSimulator predictions vs our real vLLM CPU runs, per workload cell."""
import json, os, subprocess

SIM_PY = "/home/nived-bud/simulator/.venv_sim/bin/python"
SIM = "/home/nived-bud/simulator/bud_component.py"
SIM_CWD = "/home/nived-bud/simulator"
RES = "/home/nived-bud/qwen3_cpu_bench/results"
MODEL = "Qwen/Qwen3-8B"
# verify BOTH socket counts on the SAME 1-socket profile: TP=1 (1 rank) vs real baseline,
# TP=2 (2 ranks, models the inter-socket all-reduce) vs real TP=2 run.
HW = "Xeon6767P_1socket_CPU"

IO = {"200_300": (200, 300), "512_512": (512, 512), "1024_1024": (1024, 1024),
      "4096_1024": (4096, 1024), "8192_2048": (8192, 2048)}
CONC = [1, 20, 50, 100]


def sim(inp, out, batch, tp):
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    r = subprocess.run(
        [SIM_PY, SIM, "--hf-id", MODEL, "--hardware", HW, "--tensor-parallel", str(tp),
         "--batch", str(batch), "--seq", str(inp + out),
         "--input-tokens", str(inp), "--output-tokens", str(out)],
        cwd=SIM_CWD, env=env, capture_output=True, text=True, timeout=120)
    d = json.loads(r.stdout.strip().splitlines()[-1])
    s = d.get("slo", {})
    return s.get("ttft_ms"), s.get("tpot_ms"), s.get("throughput_tok_s")


def real(cfg, io, c, key):
    f = f"{RES}/{cfg}__{io}__c{c}.json"
    if not os.path.exists(f):
        return None
    return json.load(open(f)).get(key)


rows = []
for io, (inp, outp) in IO.items():
    for c in CONC:
        try:
            p1_ttft, p1_tpot, p1_thr = sim(inp, outp, c, 1)   # 1 socket, 1 rank
            p2_ttft, p2_tpot, p2_thr = sim(inp, outp, c, 2)   # 2 sockets, 2 ranks (+all-reduce)
        except Exception as e:
            print(f"sim fail {io} c{c}: {e}"); continue
        rows.append(dict(
            io=io, c=c,
            # 1-socket sim vs real TP=1 (c0_baseline)
            p1_thr=p1_thr, p1_ttft=p1_ttft, p1_tpot=p1_tpot,
            r0_thr=real("c0_baseline", io, c, "output_throughput"),
            r0_tpot=real("c0_baseline", io, c, "mean_tpot_ms"),
            # 2-socket sim vs real TP=2 (c1_tp2_numa)
            p2_thr=p2_thr, p2_ttft=p2_ttft, p2_tpot=p2_tpot,
            r1_thr=real("c1_tp2_numa", io, c, "output_throughput"),
            r1_tpot=real("c1_tp2_numa", io, c, "mean_tpot_ms"),
        ))

json.dump(rows, open(f"{RES}/sim_compare.json", "w"), indent=1)


def rr(a, b):
    return f"{a/b:.2f}x" if (a and b) else "-"


print(f"\n{'cell':>16} | {'TP=1 (1 socket) tok/s':^26} | {'TP=2 (2 sockets) tok/s':^26}")
print(f"{'':>16} | {'sim':>8} {'real':>8} {'sim/real':>8} | {'sim':>8} {'real':>8} {'sim/real':>8}")
for r in rows:
    print(f"{r['io']+' c'+str(r['c']):>16} | "
          f"{(r['p1_thr'] or 0):8.1f} {(r['r0_thr'] or 0):8.1f} {rr(r['p1_thr'], r['r0_thr']):>8} | "
          f"{(r['p2_thr'] or 0):8.1f} {(r['r1_thr'] or 0):8.1f} {rr(r['p2_thr'], r['r1_thr']):>8}")

import statistics as st
def geo(key_p, key_r):
    v = [r[key_p]/r[key_r] for r in rows if r.get(key_p) and r.get(key_r)]
    return st.geometric_mean(v), st.median([abs(x-1) for x in v]), min(v), max(v), len(v)
g1 = geo('p1_thr', 'r0_thr'); g2 = geo('p2_thr', 'r1_thr')
print(f"\nThroughput sim/real:")
print(f"  TP=1 (1-socket profile vs baseline): geomean {g1[0]:.2f}x  median|err| {g1[1]*100:.0f}%  range[{g1[2]:.2f},{g1[3]:.2f}]  n={g1[4]}")
print(f"  TP=2 (2-socket profile vs TP=2 run): geomean {g2[0]:.2f}x  median|err| {g2[1]*100:.0f}%  range[{g2[2]:.2f},{g2[3]:.2f}]  n={g2[4]}")
