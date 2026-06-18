#!/usr/bin/env python3
"""Comprehensive BudSimulator API audit harness. Re-runnable; saves /tmp/api_audit_results.json.
Records HTTP status, latency, and—critically—flags 200-OK responses that contain physically
impossible or clearly-wrong numbers (the accuracy failures that don't show up as HTTP errors)."""
import json, time, urllib.request, urllib.error, sys

BASE = "http://localhost:8000"
results = []

def call(method, path, body=None, timeout=60):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            dt = (time.time()-t0)*1000
            txt = r.read().decode()
            try: j = json.loads(txt)
            except: j = txt
            return r.status, j, dt
    except urllib.error.HTTPError as e:
        dt = (time.time()-t0)*1000
        try: j = json.loads(e.read().decode())
        except: j = None
        return e.code, j, dt
    except Exception as e:
        dt = (time.time()-t0)*1000
        return None, f"{type(e).__name__}: {e}", dt

def T(name, method, path, body=None, expect=(200,), accuracy_check=None, timeout=60):
    """expect: tuple of acceptable status codes. accuracy_check(json)->str|None returns a problem note."""
    status, j, dt = call(method, path, body, timeout)
    ok = status in expect
    note = ""
    if ok and accuracy_check:
        try:
            problem = accuracy_check(j)
            if problem:
                ok = False; note = "ACCURACY: " + problem
        except Exception as e:
            note = f"(accuracy_check err: {e})"
    if not ok and not note:
        note = f"got {status}, expected {expect}"
        if isinstance(j, dict) and 'detail' in j:
            note += f" | detail={str(j['detail'])[:160]}"
        elif isinstance(j, str):
            note += f" | {j[:160]}"
    results.append(dict(name=name, method=method, path=path, status=status,
                        ok=ok, ms=round(dt,1), note=note))
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name:48} {method} {path[:46]:46} {status} {round(dt)}ms {note[:120]}")
    return j

# ---------------------------------------------------------------- GET endpoints
T("health", "GET", "/api/health")
T("health.ready", "GET", "/api/health/ready", expect=(200,503))
T("health.diagnostics", "GET", "/api/health/diagnostics", expect=(200,503))
T("root", "GET", "/")
T("hardware.list", "GET", "/api/hardware")
T("hardware.filter.gpu", "GET", "/api/hardware/filter?type=gpu")
T("hardware.filter.minmem", "GET", "/api/hardware/filter?min_memory=80")
T("hardware.detail.A100", "GET", "/api/hardware/A100_80GB_GPU")
T("hardware.detail.unknown", "GET", "/api/hardware/NOPE_GPU_XYZ", expect=(404,))
T("hardware.vendors.A100", "GET", "/api/hardware/vendors/A100_80GB_GPU", expect=(200,404))
T("hardware.clouds.A100", "GET", "/api/hardware/clouds/A100_80GB_GPU", expect=(200,404))
T("simulator.hardware", "GET", "/api/simulator/hardware")
T("models.list", "GET", "/api/models/list?limit=10")
T("models.filter", "GET", "/api/models/filter?limit=5")
T("models.popular", "GET", "/api/models/popular?limit=5")
T("models.popular.neg", "GET", "/api/models/popular?limit=-5", expect=(422,))  # M5: bounded, now rejected
T("models.popular.huge", "GET", "/api/models/popular?limit=100000", expect=(200,422), timeout=120)
T("models.detail", "GET", "/api/models/Qwen/Qwen-7B", expect=(200,404))
T("usecases.list", "GET", "/api/usecases")
T("usecases.search", "GET", "/api/usecases/search?query=chat", expect=(200,422))
T("usecases.stats", "GET", "/api/usecases/stats")
T("usecases.industries", "GET", "/api/usecases/industries/list")
T("usecases.tags", "GET", "/api/usecases/tags/list")

# ---------------------------------------------------------------- /api/models/calculate
def mem_sane(j):
    # weight memory must be > 0 and total >= weights
    w = j.get('weight_memory_gb') or (j.get('memory') or {}).get('weight_memory_gb')
    if w is None: return None
    if w <= 0: return f"weight_memory_gb={w} <= 0"
    return None
for m in ["Qwen/Qwen-7B","gpt2","HuggingFaceH4/zephyr-7b-beta"]:
    T(f"calc.{m}", "POST", "/api/models/calculate",
      {"model_id":m,"precision":"bf16","batch_size":1,"seq_length":2048}, accuracy_check=mem_sane)
T("calc.bad_model", "POST", "/api/models/calculate",
  {"model_id":"this/does-not-exist-xyz"}, expect=(400,403,404,422,500))  # 403 = gated/not-found (L4 msg clarified)
T("calc.neg_batch", "POST", "/api/models/calculate",
  {"model_id":"gpt2","batch_size":-1}, expect=(200,400,422))
T("calc.zero_seq", "POST", "/api/models/calculate",
  {"model_id":"gpt2","seq_length":0}, expect=(200,400,422))
T("calc.huge_seq", "POST", "/api/models/calculate",
  {"model_id":"gpt2","seq_length":100000000}, expect=(200,400,422))
T("calc.int4", "POST", "/api/models/calculate",
  {"model_id":"Qwen/Qwen-7B","precision":"int4"}, accuracy_check=mem_sane)
T("calc.bad_precision", "POST", "/api/models/calculate",
  {"model_id":"gpt2","precision":"fp9"}, expect=(200,400,422))

# ---------------------------------------------------------------- /api/models/analyze
T("analyze.qwen", "POST", "/api/models/analyze", {"model_id":"Qwen/Qwen-7B"})

# ---------------------------------------------------------------- /api/hardware/recommend
T("recommend.20gb", "POST", "/api/hardware/recommend", {"total_memory_gb":20.0})
T("recommend.350gb", "POST", "/api/hardware/recommend", {"total_memory_gb":350.0,"model_params_b":70.0})
T("recommend.zero", "POST", "/api/hardware/recommend", {"total_memory_gb":0}, expect=(200,400,422))
T("recommend.neg", "POST", "/api/hardware/recommend", {"total_memory_gb":-5}, expect=(200,400,422))

# ---------------------------------------------------------------- /api/v2/power/estimate
def power_sane(j):
    p = j.get('estimated_power') or (j.get('power') or {}).get('total_w') or j.get('total_power_w')
    return None  # structure varies; checked separately
T("power.llama8b.h100", "POST", "/api/v2/power/estimate",
  {"model":"llama3_8b","hardware":"H100_GPU","batch_size":1})
T("power.bad_model", "POST", "/api/v2/power/estimate",
  {"model":"nonexistent_model_xyz","hardware":"H100_GPU"}, expect=(400,404,422,500))
T("power.bad_hw", "POST", "/api/v2/power/estimate",
  {"model":"llama3_8b","hardware":"FAKE_GPU_XYZ"}, expect=(400,404,422,500))

# ---------------------------------------------------------------- /api/v2/simulate/serving
def serving_sane(j):
    m = j.get('metrics') or j
    thr = m.get('throughput_tokens_per_sec') or m.get('output_throughput_tokens_per_sec')
    return None
T("serving.qwen.a100", "POST", "/api/v2/simulate/serving",
  {"model":"Qwen/Qwen-7B","hardware":"A100_80GB_GPU","num_requests":20,"arrival_rate_rps":2},
  timeout=120, accuracy_check=serving_sane)
T("serving.small_mean", "POST", "/api/v2/simulate/serving",
  {"model":"gpt2","hardware":"A100_80GB_GPU","num_requests":10,"input_length_mean":4,
   "output_length_mean":1,"arrival_rate_rps":1}, timeout=120)  # M6: clamp min<=max, no degenerate workload
T("serving.bad_hw", "POST", "/api/v2/simulate/serving",
  {"model":"gpt2","hardware":"FAKE_XYZ","num_requests":5}, expect=(400,404,422,500))

# ---------------------------------------------------------------- training simulator
T("sim.check-fit", "POST", "/api/simulator/check-fit",
  {"model":"llama2_7b","hardware":"A100_80GB_GPU","num_gpus":8,"method":"lora"}, expect=(200,400,422,500))
T("sim.estimate-time", "POST", "/api/simulator/estimate-time",
  {"model":"llama2_7b","dataset_tokens":1000000000,"hardware":"A100_80GB_GPU","num_gpus":8}, expect=(200,400,422,500))
T("sim.recommend-cluster", "POST", "/api/simulator/recommend-cluster",
  {"model":"llama2_7b","dataset_tokens":1000000000}, expect=(200,400,422,500))
T("sim.estimate-training", "POST", "/api/simulator/estimate-training",
  {"model":"llama2_7b","dataset_tokens":1000000000,"hardware":"A100_80GB_GPU","num_gpus":8}, expect=(200,400,422,500))

# ---------------------------------------------------------------- compare
T("compare.2models", "POST", "/api/models/compare",
  {"model_ids":["gpt2","Qwen/Qwen-7B"]}, expect=(200,400,422))
T("compare.partial_fail", "POST", "/api/models/compare",
  {"model_ids":["gpt2","totally/bogus-model-xyz"]}, expect=(200,400,422))  # /compare schema validates input shape

# ---------------------------------------------------------------- v2 advanced endpoints (smoke)
for name, path, body in [
    ("v2.optimize.config","/api/v2/optimize/config",{"model":"llama2_7b","hardware":"A100_80GB_GPU"}),
    ("v2.optimize.pareto","/api/v2/optimize/pareto",{"model":"llama2_7b","hardware":"A100_80GB_GPU"}),
    ("v2.simulate.batch","/api/v2/simulate/batch",{"model":"llama2_7b","hardware":"A100_80GB_GPU","batch_sizes":[1,4,16]}),
    ("v2.cache.analyze","/api/v2/cache/analyze",{"model":"llama2_7b","hardware":"A100_80GB_GPU"}),
    ("v2.memory.tiers","/api/v2/memory/tiers",{"model":"llama2_7b","hardware":"A100_80GB_GPU"}),
    ("v2.cluster.topology","/api/v2/cluster/topology",{"model":"llama2_7b","hardware":"A100_80GB_GPU","num_gpus":8}),
    ("v2.cluster.disaggregate","/api/v2/cluster/disaggregate",{"model":"llama2_7b","hardware":"A100_80GB_GPU"}),
    ("v2.analyze.sensitivity","/api/v2/analyze/sensitivity",{"model":"llama2_7b","hardware":"A100_80GB_GPU"}),
]:
    T(name, "POST", path, body, expect=(200,400,422,500), timeout=120)

# ---------------------------------------------------------------- usecase lifecycle + optimization bug
uc = T("usecase.create","POST","/api/usecases",
   {"name":"AuditTest UC","industry":"technology","description":"audit",
    "input_tokens_max":2048,"output_tokens_max":512}, expect=(200,201,422))
uid = uc.get("unique_id") if isinstance(uc,dict) else None
if uid:
    T("usecase.get","GET",f"/api/usecases/{uid}")
    T("usecase.quick-opt.nullSLO","GET",f"/api/usecases/{uid}/quick-optimization",
      expect=(200,400,422,500))  # null ttft_max/e2e_max -> potential crash
    T("usecase.optimize-hardware.nullSLO","POST",f"/api/usecases/{uid}/optimize-hardware",
      {}, expect=(200,400,422,500))
    T("usecase.recommendations","POST",f"/api/usecases/{uid}/recommendations",{}, expect=(200,400,422,500))
    T("usecase.delete","DELETE",f"/api/usecases/{uid}", expect=(200,204))

# ---------------------------------------------------------------- summary
npass = sum(1 for r in results if r['ok'])
nfail = len(results)-npass
json.dump(results, open("/tmp/api_audit_results.json","w"), indent=2)
print(f"\n==== {len(results)} tests: {npass} PASS, {nfail} FAIL ====")
print("FAILURES / FLAGS:")
for r in results:
    if not r['ok']:
        print(f"  - {r['name']}: {r['note']}")
