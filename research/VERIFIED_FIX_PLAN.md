# Verified Bug Fix Plan — BudSimulator

**Date**: 2026-03-01
**Process**: 57 issues reported → 3 verification agents read every line of code + web research → 25 confirmed bugs, 5 false positives, 3 partially correct (adjusted), 24 medium/low deferred

---

## VERIFICATION SUMMARY

| Original Report | Verified | False Positive | Partially Correct |
|-----------------|----------|----------------|-------------------|
| 10 Critical | 7 confirmed | C2 (decode double-count), C10 (checkpointing sqrt) | C6 (GH200 Flops — inconsistency not 2x) |
| 15 High | 12 confirmed | H10 (TPUv5e BW correct at 820) | H8 (DPO — effective batch size, not 2x activations) |
| 20 Medium | 17 confirmed | B100 Flops (1750 correct), GH200 BW (4900 correct) | M18 (INT4 — 0.5 not 0.25, but not 1.0 either) |

**False positives caught (5)**:
1. **C2**: `decode_lat * output_tokens` is correct — decode_moddeling returns per-token TPOT, not total
2. **C10**: `sqrt(L)` checkpointing memory is a reasonable approximation for practical implementations
3. **H10**: TPUv5e bandwidth IS 820 GB/s (not 1600 — that's TPU v5p)
4. **B100 Flops**: 1750 TFLOPS is correct per NVIDIA spec
5. **GH200 Memory_BW**: 4900 GB/s is correct for HBM3e variant

---

## PHASE 0: HARDWARE CONFIG FIXES (No code logic changes, just data)

**Effort**: 1 day | **Risk**: Minimal | **Dependencies**: None

### Fix 0A: GH200 Flops Consistency
**File**: `hardware/configs.py:184`
**Current**: `'Flops': 1979` (sparse BF16)
**Fix**: `'Flops': 989` (dense BF16, matching H100 convention)
**Rationale**: H100 uses 989 (dense). GH200 uses same GPU die. Must use same convention. LLM inference frameworks do NOT reliably exploit 2:4 sparsity.
**Validation**: `assert HARDWARE_CONFIGS['GH200_GPU']['Flops'] == HARDWARE_CONFIGS['H100_GPU']['Flops']`

### Fix 0B: H100 SXM Memory Bandwidth
**File**: `hardware/configs.py:151`
**Current**: `'Memory_BW': 3400`
**Fix**: `'Memory_BW': 3350`
**Source**: NVIDIA H100 datasheet — 3.35 TB/s
**Validation**: Cross-reference nvidia.com/en-us/data-center/h100/

### Fix 0C: Add H200 Configuration
**File**: `hardware/configs.py`
**Add**:
```python
'H200_GPU': {
    'name': 'H200_GPU',
    'Flops': 989,          # Same Hopper die as H100
    'Memory_size': 141,    # 141 GB HBM3e
    'Memory_BW': 4800,     # 4.8 TB/s
    'ICN': 450,            # NVLink 900 GB/s bidirectional
    'num_chips': 1,
    'architecture': 'HOPPER',
    'real_values': True,
    'cost': {'tdp_watts': 700},
}
```
**Source**: NVIDIA H200 datasheet
**Validation**: `assert 'H200_GPU' in HARDWARE_CONFIGS`

### Fix 0D: Add Standalone B200 Configuration
**File**: `hardware/configs.py`
**Add**:
```python
'B200_GPU': {
    'name': 'B200_GPU',
    'Flops': 2250,         # 2.25 PFLOPS BF16 dense
    'Memory_size': 192,    # 192 GB HBM3e
    'Memory_BW': 8000,     # 8 TB/s
    'ICN': 900,            # NVLink 5, 1.8 TB/s bidirectional
    'num_chips': 1,
    'architecture': 'BLACKWELL',
    'real_values': True,
    'cost': {'tdp_watts': 1000},
}
```
**Source**: NVIDIA Blackwell Architecture whitepaper
**Validation**: `assert 'B200_GPU' in HARDWARE_CONFIGS`

### Fix 0E: Add H100 PCIe Variant
**File**: `hardware/configs.py`
**Add**:
```python
'H100_PCIe_GPU': {
    'name': 'H100_PCIe_GPU',
    'Flops': 756,          # 756 TFLOPS BF16 dense
    'Memory_size': 80,     # 80 GB HBM3
    'Memory_BW': 2000,     # 2 TB/s
    'ICN': 300,            # PCIe Gen5 + P2P
    'num_chips': 1,
    'architecture': 'HOPPER',
    'real_values': True,
    'cost': {'tdp_watts': 350},
}
```

### Fix 0F: MI300X Interconnect Bandwidth
**File**: `hardware/configs.py:713`
**Current**: `'ICN': 400`
**Fix**: `'ICN': 896`
**Source**: AMD MI300X datasheet — 7 Infinity Fabric links × 128 GB/s each = 896 GB/s aggregate

**Regression test for Phase 0**: Run full existing test suite — no logic changes, only data. All tests should pass unchanged.

---

## PHASE 1: CRITICAL BUGS (Correctness — wrong results)

**Effort**: 3-4 days | **Risk**: Medium (formula changes affect test assertions) | **Dependencies**: Phase 0

### Fix 1A: GQA Weight Memory in Simulator
**File**: `genz/serving/simulator.py:411`
**Current**:
```python
attn_params = 4 * hidden * n_heads * head_dim * layers
```
**Fix**:
```python
num_kv_heads = getattr(model_config, 'num_key_value_heads', n_heads)
attn_params = (
    2 * hidden * n_heads * head_dim +       # Q and O projections
    2 * hidden * num_kv_heads * head_dim     # K and V projections
) * layers
```
**Math**: For GQA, Q/O use `n_heads` but K/V use `num_kv_heads`. Standard MHA has `num_kv_heads == n_heads`, so this is backward compatible.
**Verification**: For Llama-3-8B (n_heads=32, kv_heads=8, hidden=4096, head_dim=128, layers=32):
- Old: `4 * 4096 * 32 * 128 * 32 = 2,147,483,648` (2.15B params for attention)
- New: `(2*4096*32*128 + 2*4096*8*128) * 32 = 1,342,177,280` (1.34B params) — 37% reduction
**Regression test**: `assert _estimate_weight_bytes(llama3_8b_config) < _estimate_weight_bytes_old(llama3_8b_config)` for GQA models; `==` for MHA models

### Fix 1B: Attention TP*EP Over-Sharding (MoE)
**File**: `genz/Models/attention.py` — ALL 9 attention functions
**Current**: `tp = parallelism_config.tensor_parallel * parallelism_config.expert_parallel` (lines 11, 45, 72, 100, 147, 199, 211, 232, 281)
**Fix**: `tp = parallelism_config.tensor_parallel` in ALL attention functions
**Math**: Expert parallelism splits MoE FFN/expert layers across EP ranks. Attention layers are dense and only split by TP. Reference: AMD ROCm docs — "Dense layers benefit from TP, MoE layers call for EP."
**Verification**: With TP=4, EP=2, attention heads should be divided by 4 (not 8). For model with 32 heads, each GPU gets 8 heads (not 4).
**Regression test**: Test with EP=1 (should be unchanged) and EP=2 (should now give different attention timing vs old code). Assert attention FLOPs with EP=2 equal attention FLOPs with EP=1.

### Fix 1C: Decode Memory Check Missing EP Division
**File**: `genz/LLM_inference/llm_decode.py:77`
**Current**: `if per_chip_memory < total_memory_req/pipeline_parallel:`
**Fix**:
```python
memory_parallelism = pipeline_parallel * expert_parallel
if per_chip_memory < total_memory_req / memory_parallelism:
```
Also update the offload path similarly (line 79).
**Math**: Matches prefill path (llm_prefill.py:60-61) which already has this fix.
**Regression test**: Test MoE model with EP=2 — decode should NOT trigger offload when prefill doesn't.

### Fix 1D: ZeRO-1 Support in Basic Training Calculator
**File**: `training/calculator.py:384-386`
**Current**:
```python
if deepspeed_stage in ('zero2', 'zero3'):
    optimizer_memory /= data_parallel
```
**Fix**:
```python
if deepspeed_stage in ('zero1', 'zero2', 'zero3'):
    optimizer_memory /= data_parallel
```
**Math**: ZeRO-1 (Rajbhandari et al., SC'20) partitions optimizer states (m, v for Adam) across DP ranks. Gradients and parameters remain replicated.
**Verification**: For 7B model with AdamW (12 bytes/param optimizer), DP=8:
- No ZeRO: 84 GB optimizer
- ZeRO-1: 84/8 = 10.5 GB optimizer
- Currently: 84 GB (bug — same as no ZeRO)
**Regression test**: `assert calc.calculate("zero1", dp=8).optimizer_memory_gb < calc.calculate(None, dp=8).optimizer_memory_gb`

### Fix 1E: LoRA Params Inconsistency in Advanced Calculator
**File**: `training/advanced_calculator.py:437-450`
**Current**: 4 attention targets + 2 MLP targets with `// 2` bug
**Fix**: Align with basic calculator's 7-target formula:
```python
kv_dim = hidden_size * num_kv_heads // num_heads
attn_params_per_layer = (
    2 * lora_rank * (hidden_size + hidden_size) +      # q_proj
    2 * lora_rank * (hidden_size + kv_dim) +           # k_proj
    2 * lora_rank * (hidden_size + kv_dim) +           # v_proj
    2 * lora_rank * (hidden_size + hidden_size)        # o_proj
)
mlp_params_per_layer = (
    2 * lora_rank * (hidden_size + intermediate_size) * 3  # gate, up, down
)
trainable_params = num_layers * (attn_params_per_layer + mlp_params_per_layer)
```
**Math**: LoRA (Hu et al., 2021) adds low-rank matrices A (d×r) and B (r×k) to each target, giving 2*r*(d+k) trainable params per target. SwiGLU FFN has 3 projections (gate, up, down), not 2.
**Verification**: For Llama-3.1-8B (r=16): basic calc = ~82.9M, advanced calc should now also = ~82.9M (was ~35.7M — 57% underestimate).
**Regression test**: `assert abs(basic_calc(llama8b, r=16) - advanced_calc(llama8b, r=16)) / basic_calc(llama8b, r=16) < 0.05`

### Fix 1F: release_onchip_mem max→min
**File**: `genz/system.py:177`
**Current**: `self.on_chip_mem_left_size = max(self.on_chip_mem_size, data_sz + self.on_chip_mem_left_size)`
**Fix**: `self.on_chip_mem_left_size = min(self.on_chip_mem_size, data_sz + self.on_chip_mem_left_size)`
**Math**: After releasing memory, available memory increases but cannot exceed total capacity.
**Regression test**: `assert system.on_chip_mem_left_size <= system.on_chip_mem_size` after any release operation.

---

## PHASE 2: HIGH BUGS (Significant inaccuracy)

**Effort**: 4-5 days | **Risk**: Medium | **Dependencies**: Phase 1

### Fix 2A: Token Throughput — Output Only
**File**: `genz/serving/simulator.py:275`
**Current**: `total_tokens_generated += req.input_tokens + req.tokens_generated`
**Fix**:
```python
total_output_tokens += req.tokens_generated
total_input_tokens += req.input_tokens  # Track separately
```
Update throughput calculation to use `total_output_tokens` for the primary metric.
**Regression test**: With 512 input + 128 output tokens, assert throughput = output_tokens / duration (not 640 / duration).

### Fix 2B: Power Model — Full 7-Component Tracking in Sim Loop
**File**: `genz/serving/simulator.py:278-281`
**Fix**: After each batch completion, call `estimate_from_simulation_result()` instead of only `add_accelerator_active_energy()`:
```python
if power_model and batch:
    latency_ms = (current_time_ns - last_batch_end_ns) / NS_PER_MS
    compute_util = batch.size / max_batch_size  # Rough utilization estimate
    mem_bytes = _estimate_batch_memory_bytes(batch, model_config)
    power_model.estimate_from_simulation_result(
        latency_ms=latency_ms,
        compute_util=compute_util,
        mem_util=min(1.0, mem_bytes / total_hbm_bytes),
        data_read_bytes=mem_bytes,
        num_accel=num_accel,
        pue=pue,
    )
```
**Regression test**: Assert `power_summary["breakdown_j"]` has entries for accelerator, dram, host_cpu, cooling (not just accelerator).

### Fix 2C: Eviction → Request Preemption
**File**: `genz/serving/memory_model.py:155-190` and `genz/serving/batch_scheduler.py:118-126`
**Fix**:
1. `evict_blocks()` returns `(evicted_count, evicted_request_ids: List[int])`
2. Scheduler removes evicted requests from `_running` and re-adds to `_pending`
3. Evicted requests reset their decode state for re-prefill
**Math**: Matches vLLM preemption model — when blocks are evicted, the request must be re-prefilled from scratch (or swapped back, but swap is not yet implemented).
**Regression test**: Create memory-pressure scenario, trigger eviction, assert evicted request is in `_pending` (not `_running`).

### Fix 2D: First Token During Prefill Completion
**File**: `genz/serving/batch_scheduler.py:167-170`
**Current**:
```python
for req in batch.prefill_requests:
    req.set_decoding(current_time_ns)
    self._running.append(req)
```
**Fix**:
```python
for req in batch.prefill_requests:
    req.set_decoding(current_time_ns)
    req.record_token(current_time_ns)  # First token from prefill logits
    if req.is_complete():
        completed.append(req)
    else:
        self._running.append(req)
```
**Math**: The prefill forward pass computes logits for all input positions; the last position's logit produces the first output token. This is the standard behavior in all serving frameworks.
**Regression test**: Assert TTFT ≈ prefill_latency (not prefill + decode_latency). Assert single-token-output requests complete after prefill without decode.

### Fix 2E: Pipeline Parallel Latency — Uncomment and Fix
**Files**: `genz/LLM_inference/llm_prefill.py:99-107`, `genz/LLM_inference/llm_decode.py:163-170`
**Fix**: Uncomment the PP latency formula with correct 1F1B model:
```python
if pipeline_parallel > 1:
    stage_latency = prefill_latency / pipeline_parallel  # Per-stage (already computed for PP layers)
    # 1F1B schedule: bubble = (PP-1) * stage_latency
    total_latency = prefill_latency + (pipeline_parallel - 1) * stage_latency
    thrpt = 1000 * batch_size / total_latency
else:
    thrpt = 1000 * batch_size / prefill_latency
```
**Math**: For 1F1B pipeline scheduling (Narayanan et al., 2021), total latency = pipeline fill time + steady state. Fill time = (PP-1) × stage_latency. Total = (PP-1) × stage_latency + M × stage_latency ≈ full_latency + (PP-1) × stage_latency for M=PP microbatches.
**Note**: The GenZ model already divides layers by PP in the operator graph, so `prefill_latency` is already the per-stage compute time × PP (full pipeline traverse). The bubble overhead is `(PP-1) * (prefill_latency / PP)`.
**Regression test**: PP=4 latency > PP=1 latency / 4 (bubble overhead exists). PP=1 unchanged.

### Fix 2F: Decode KV Weighting — 50/50 Average
**File**: `genz/LLM_inference/llm_decode.py:129`
**Current**: `decode_latency = initial_latency * 0.8 + final_latency * 0.2`
**Fix**: `decode_latency = (initial_latency + final_latency) / 2.0`
**Math**: KV cache grows linearly from S_input to S_input + S_output. Per-token latency is approximately linear in context length (attention is O(n), memory reads are O(n)). The time-averaged value of a linear function from a to b over [0,1] is (a+b)/2 — the trapezoidal rule, which is exact for linear functions.
**Regression test**: With initial_latency=10ms, final_latency=20ms, assert decode_latency=15ms (not 12ms as with 80/20).

### Fix 2G: Reference Model Memory — Remove 0.8 Factor
**File**: `training/advanced_calculator.py:314`
**Current**: `reference_memory = weight_memory * 0.8`
**Fix**: `reference_memory = weight_memory`
**Math**: `model.eval()` in PyTorch changes BN/dropout behavior but NOT weight storage. Weights occupy identical memory in eval vs train mode.
**Regression test**: Assert reference_memory == weight_memory for DPO stage.

### Fix 2H: DPO Activation — Double Effective Batch Size
**File**: `training/advanced_calculator.py:298-306`
**Fix**: When `stage_config.uses_pairwise_data` is True, double the effective batch size for activation memory:
```python
effective_batch = batch_size * 2 if stage_config.uses_pairwise_data else batch_size
activation_memory = self._calculate_activation_memory(
    config, effective_batch, seq_length, precision_bytes,
    gradient_checkpointing, tensor_parallel,
)
```
**Math**: DPO concatenates chosen + rejected into one batch (TRL DPOTrainer implementation), so effective batch size = 2 × user batch size for activation memory.
**Regression test**: DPO activation memory ≈ 2 × SFT activation memory (same batch size, same model).

### Fix 2I: FP8 in Basic Training Calculator
**File**: `training/calculator.py:37-45`
**Fix**: Add `'fp8': 1` to PRECISION_BYTES dict.
**Regression test**: `assert calc.PRECISION_BYTES['fp8'] == 1`

### Fix 2J: FP8 Hardware Compatibility Check
**File**: `genz/system.py:17` + precision handling
**Fix**: Add validation when FP8 is requested:
```python
# In System class or compute_time calculation
FP8_CAPABLE_ARCHS = {'HOPPER', 'BLACKWELL', 'CDNA3'}  # H100+, MI300X+
if precision == 'fp8' and self.architecture not in FP8_CAPABLE_ARCHS:
    warnings.warn(f"FP8 not natively supported on {self.architecture}. Using BF16 performance.")
    effective_multiplier = compute_multiplier['bf16']  # Fall back to BF16
```
**Math**: FP8 tensor cores only exist on Hopper+ (NVIDIA) and CDNA3+ (AMD). Pre-Hopper GPUs should not get 2x FP8 speedup.
**Regression test**: A100 with FP8 should warn and use BF16 performance. H100 with FP8 should use 0.5x multiplier.

---

## PHASE 3: MEDIUM BUGS (Accuracy improvements)

**Effort**: 5-7 days | **Risk**: Low-Medium | **Dependencies**: Phases 1-2

### Fix 3A: Recomputation FLOPs for Gradient Checkpointing
**File**: `training_modeling.py:1188-1196`
**Fix**: For standard full checkpointing: `recompute_flops = total_forward` (1.0x, not sqrt(L)/L ≈ 0.18x)
**Math**: Standard implementations (PyTorch `gradient_checkpointing_enable()`, Megatron-LM) checkpoint every layer and recompute all forward activations during backward. Total training FLOPs = 1F + 1F_recompute + 2B = 4F (vs 3F without checkpointing). MFU formula at line 500-558 already uses factor=4, confirming this.

### Fix 3B: Pipeline Bubble — Use 1F1B Formula
**File**: `training/distributed.py:337-339`
**Fix**:
```python
if self.pipeline_parallel > 1:
    pp = self.pipeline_parallel
    m = getattr(self, 'gradient_accumulation_steps', 4 * pp)  # microbatches
    bubble_fraction = (pp - 1) / (pp + m - 1)
    overhead += bubble_fraction
```
**Math**: Narayanan et al. 2021 (Megatron-LM): 1F1B bubble = (p-1)/(p+m-1). With interleaved schedule: (p-1)/(p*v+m-1) where v = virtual pipeline stages.

### Fix 3C: Adafactor Memory — Factorized Statistics
**File**: `training/optimizers.py:137-150`
**Fix**: Change `total_bytes_per_param` from 4.0 to 1.0
**Math**: Shazeer & Stern 2018: Adafactor stores row factors (m values) + column factors (n values) for an m×n matrix, which is O(m+n) vs Adam's O(m×n). For large square matrices this approaches 0; overall ~1 byte/param is a good approximation.

### Fix 3D: Sequence Parallelism for Activation Memory
**File**: `training/calculator.py:390-457`
**Fix**: Add `sequence_parallel` parameter. When True, divide non_tp_coeff by T:
```python
if sequence_parallel:
    per_layer_bytes = s * b * h * ((non_tp_coeff + tp_coeff + score_coeff) / T)
else:
    per_layer_bytes = s * b * h * (non_tp_coeff + tp_coeff / T + score_coeff / T)
```
**Math**: Korthikanti et al. 2022: SP shards LayerNorm/dropout/residual activations along sequence dimension across TP ranks. All terms divided by T.

### Fix 3E: DoRA Magnitude Vector
**File**: `training/calculator.py:238+`
**Fix**: When method == 'dora', add output_dim per target module per layer to trainable params.
**Math**: Liu et al. 2024 (ICML): DoRA decomposes W = m * (V/||V||). Magnitude vector m has dim = output_dim.

### Fix 3F: Context Parallelism → Activation Memory Reduction
**File**: `training/distributed.py:289+` and `training/calculator.py`
**Fix**: Divide activation memory (specifically the sequence-length term) by context_parallel.
**Math**: Ring attention distributes sequence across CP ranks. Each rank holds seq_len/CP tokens.

### Fix 3G: GaLore — Remove Arbitrary 50% Cap
**File**: `training/optimizers.py:72-76`
**Fix**: `effective_params = min(effective_params, trainable_params)` (cap at 100% instead of 50%)

### Fix 3H: INT4 Compute Multiplier — W4A16
**File**: `genz/system.py:10`
**Fix**: Change `'int4': 0.25` to `'int4': 0.5` (reflecting realistic W4A16 speedup with optimized Marlin kernels, not theoretical W4A4).
**Alternative**: Add separate entries `'w4a16': 0.5, 'w4a4': 0.25`.

---

## PHASE 4: SERVING SIMULATION DESIGN FIXES

**Effort**: 3-5 days | **Risk**: Medium (structural changes) | **Dependencies**: Phase 2

### Fix 4A: Batch-In-Flight → Support PP>1 Pipelining
**File**: `genz/serving/simulator.py:286-292`
**Change**: Replace `batch_in_flight: bool` with `batches_in_flight: int` counter. Allow up to `pipeline_parallel` concurrent batches.
**Impact**: Mostly affects PP>1 simulations (less common for inference). Can be deferred if PP>1 is not a priority.

### Fix 4B: Sweep Copies All WorkloadConfig Fields
**File**: `genz/serving/simulator.py:179-187`
**Fix**: Use `dataclasses.replace()` instead of manual field copying:
```python
sweep_config = dataclasses.replace(workload_config, arrival_rate=rate)
```

---

## REGRESSION TEST STRATEGY

### Pre-Fix Baseline
Before any changes, capture baseline test results:
```bash
pytest llm-memory-calculator/tests/ -v --tb=short > baseline_results.txt 2>&1
```

### Per-Phase Verification
After each phase, run:
```bash
# All serving tests
pytest llm-memory-calculator/tests/serving/ -v --tb=short

# All training tests
pytest llm-memory-calculator/tests/training/ -v --tb=short

# Core tests
pytest llm-memory-calculator/tests/unit/ -v --tb=short

# API tests
pytest BudSimulator/tests/test_serving_api.py -v --tb=short

# Full suite
pytest llm-memory-calculator/tests/ -v --tb=short
```

### Accuracy Validation Points
After all fixes, verify against published benchmarks:

| Metric | Hardware | Model | Expected | Source |
|--------|----------|-------|----------|--------|
| Idle power | A100 | - | ~50W | nvidia-smi measurements |
| Active power | H100 | - | ~490W | 0.70 × 700W TDP |
| Prefill TFLOPS | H100 | Llama-3-8B | ~800 TFLOPS | MLPerf Inference v5.1 |
| Decode TPOT | H100 | Llama-3-8B BS=1 | ~10ms | vLLM benchmarks |
| QLoRA 7B memory | A100 80GB | Llama-2-7B | fits 24GB GPU | Dettmers et al. 2023 |
| ZeRO-1 savings | 8xA100 | 7B AdamW | ~7x optimizer reduction | Rajbhandari et al. 2020 |
| Llama-3 training TPS | 8xH100 | 70B | ~900 TPS (378 TFLOPS) | Meta Llama 3 report |

### New Tests to Add
For each confirmed bug fix, add a specific regression test:
1. `test_gqa_weight_memory_reduction` — GQA vs MHA weight estimation
2. `test_zero1_optimizer_sharding` — ZeRO-1 optimizer memory < no sharding
3. `test_lora_calculators_consistent` — basic vs advanced within 5%
4. `test_reference_model_full_memory` — reference_memory == weight_memory
5. `test_dpo_activation_doubled` — DPO activation ≈ 2× SFT
6. `test_fp8_basic_calculator` — FP8 = 1 byte
7. `test_attention_ep_independent` — EP>1 doesn't change attention FLOPs
8. `test_decode_ep_division` — MoE decode memory check matches prefill
9. `test_onchip_mem_capped` — released memory ≤ total memory
10. `test_output_token_throughput` — throughput uses output tokens only
11. `test_power_all_components` — power breakdown has >1 component
12. `test_eviction_preempts_request` — evicted requests move to pending
13. `test_first_token_from_prefill` — TTFT ≈ prefill time
14. `test_pp_latency_has_bubble` — PP>1 latency > PP=1/PP
15. `test_decode_kv_weighting_5050` — average latency = (initial+final)/2
16. `test_pipeline_bubble_1f1b` — bubble varies with PP and microbatches
17. `test_adafactor_memory_factorized` — ~1 byte/param not 4
18. `test_sequence_parallel_activation` — SP divides all activation terms by TP
19. `test_fp8_a100_warns` — FP8 on A100 produces warning
20. `test_int4_compute_multiplier` — INT4 = 0.5x not 0.25x

---

## EXECUTION ORDER

```
Phase 0 (Day 1)          → Hardware config data fixes
    ↓
Phase 1 (Days 2-4)       → 6 critical bug fixes (parallel where independent)
    ↓                      1A,1B,1C can be parallel (different files)
    ↓                      1D,1E can be parallel (different files)
    ↓                      1F independent
    ↓
Phase 2 (Days 5-9)       → 10 high bug fixes (many parallel)
    ↓                      2A,2B,2C,2D can be parallel (different areas of simulator.py + memory_model.py)
    ↓                      2E,2F can be parallel (different files in LLM_inference/)
    ↓                      2G,2H,2I,2J can be parallel (training files)
    ↓
Phase 3 (Days 10-16)     → 8 medium bug fixes (all parallelizable)
    ↓                      3A-3H touch different files, fully parallel
    ↓
Phase 4 (Days 17-19)     → 2 design fixes (optional, lower priority)
    ↓
Test Suite Update         → Add 20 new regression tests
    ↓
Final Validation          → Full test suite + accuracy benchmarks
```

**Total: ~19 working days for all phases, or ~12 days with parallelization**

---

## RISK MITIGATION

1. **Test before and after each fix** — capture baseline, apply fix, verify delta
2. **Fix tests alongside code** — update assertions that check old (wrong) values
3. **Incremental commits** — one commit per fix, easy to revert
4. **Phase gates** — all tests must pass before moving to next phase
5. **Accuracy spot-checks** — verify calibrated values against published benchmarks at each phase gate
