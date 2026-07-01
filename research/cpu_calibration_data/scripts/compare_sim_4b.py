#!/usr/bin/env python3
"""Out-of-sample validation: simulator (UNCHANGED constants, calibrated on 8B) vs real Qwen3-4B."""
import json, os, subprocess, statistics as st

SIM_PY = "/home/nived-bud/simulator/.venv_sim/bin/python"
SIM = "/home/nived-bud/simulator/bud_component.py"
SIM_CWD = "/home/nived-bud/simulator"
RES = "/home/nived-bud/qwen3_cpu_bench/results_4b"
HW = "Xeon6767P_1socket_CPU"
MODEL = "Qwen/Qwen3-4B"
CELLS = [("512_512", 512, 512), ("1024_1024", 1024, 1024), ("8192_2048", 8192, 2048)]
CONC = [1, 50]


def sim(inp, out, batch, tp):
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    r = subprocess.run(
        [SIM_PY, SIM, "--hf-id", MODEL, "--hardware", HW, "--tensor-parallel", str(tp),
         "--batch", str(batch), "--seq", str(inp + out),
         "--input-tokens", str(inp), "--output-tokens", str(out)],
        cwd=SIM_CWD, env=env, capture_output=True, text=True, timeout=120)
    return json.loads(r.stdout.strip().splitlines()[-1]).get("slo", {}).get("throughput_tok_s")


def real(cfg, io, c):
    f = f"{RES}/{cfg}__{io}__c{c}.json"
    return json.load(open(f))["output_throughput"] if os.path.exists(f) else None


rows = []
print(f"{'cell':>16} | {'TP=1 sim':>9} {'real':>7} {'r':>6} | {'TP=2 sim':>9} {'real':>7} {'r':>6}")
for io, inp, outp in CELLS:
    for c in CONC:
        p1, p2 = sim(inp, outp, c, 1), sim(inp, outp, c, 2)
        r1, r2 = real("c0_baseline", io, c), real("c1_tp2_numa", io, c)
        rows.append((io, c, p1, r1, p2, r2))
        def rr(a, b): return a / b if (a and b) else None
        print(f"{io+' c'+str(c):>16} | {(p1 or 0):9.1f} {(r1 or 0):7.1f} "
              f"{(rr(p1,r1) or 0):6.2f} | {(p2 or 0):9.1f} {(r2 or 0):7.1f} {(rr(p2,r2) or 0):6.2f}")

v1 = [p/r for _, _, p, r, _, _ in rows if p and r]
v2 = [p/r for _, _, _, _, p, r in rows if p and r]
print(f"\n4B out-of-sample (constants fit on 8B, unchanged):")
print(f"  TP=1: geomean {st.geometric_mean(v1):.2f}x  median|err| {st.median([abs(x-1) for x in v1])*100:.0f}%")
print(f"  TP=2: geomean {st.geometric_mean(v2):.2f}x  median|err| {st.median([abs(x-1) for x in v2])*100:.0f}%")
json.dump([dict(io=r[0], c=r[1], p1=r[2], r1=r[3], p2=r[4], r2=r[5]) for r in rows],
          open(f"{RES}/sim_compare_4b.json", "w"), indent=1)
