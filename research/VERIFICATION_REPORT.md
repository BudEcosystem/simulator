# Post-Implementation Verification Report

**Date**: 2026-03-01
**Process**: 4 parallel verification agents read every modified line of code, traced execution paths, and cross-referenced with published papers and official hardware datasheets.

---

## EXECUTIVE SUMMARY

Of the 25 bug fixes applied, verification found:
- **20 CONFIRMED CORRECT** — no issues
- **3 NEEDS ADJUSTMENT** — correct direction but has secondary issues
- **2 NEW ISSUES DISCOVERED** — inconsistencies between code paths that were not fixed
- **1 FALSE POSITIVE** from initial verification (PP latency — actually correct)

No false fixes were introduced (all 25 changes improved accuracy over the pre-fix state).

---

## CONFIRMED CORRECT (19 fixes — no action needed)

| # | Fix | Verification |
|---|-----|-------------|
| 1 | GQA weight estimation (simulator.py) | Q+O use n_heads, K+V use num_kv_heads. MHA backward-compatible. |
| 2 | Output-only token throughput (simulator.py) | Matches vLLM/TGI standard metric definition |
| 3 | Sweep config via dataclasses.replace() | Shallow copy sufficient; all fields preserved |
| 4 | First token from prefill (batch_scheduler.py) | Order correct: set_decoding → record_token. max_output=1 completes correctly |
| 5 | Decode KV 50/50 weighting (llm_decode.py) | Trapezoidal rule exact for linear KV growth |
| 6 | release_onchip_mem min() (system.py) | Critical fix prevents exceeding physical memory |
| 7 | Attention tp=tensor_parallel only (attention.py, 9 functions) | EP only applies to MoE FFN, not attention. Confirmed for DeepSeek MLA too |
| 8 | Decode EP memory division (llm_decode.py) | Now matches prefill path. Correct for MoE |
| 9 | H100 Memory_BW=3350 | NVIDIA datasheet: 3.35 TB/s confirmed |
| 10 | H200 specs (141GB, 4800 GB/s) | All values confirmed per NVIDIA datasheet |
| 11 | B200 specs (2250 TFLOPS, 192GB, 8000 GB/s) | All values confirmed. BF16 dense correct |
| 12 | MI300X ICN=896 | 7 × 128 GB/s confirmed per AMD Hot Chips 2024 |
| 13 | INT4 compute_multiplier=0.5 | Correct for W4A16 (dominant INT4 pattern) |
| 14 | FP8 warning | Accurate text, appropriate trigger point |
| 15 | ZeRO-1 optimizer sharding (calculator.py) | Matches Rajbhandari et al. SC'20 exactly |
| 16 | FP8=1 byte in PRECISION_BYTES | 8 bits = 1 byte, consistent across all calculators |
| 17 | Context parallel activation (/= CP) | Ring attention correctly divides by CP |
| 18 | DoRA magnitude vector | All 7 output_dims correct per Liu et al. 2024 |
| 19 | LoRA params in both calculators | Basic and advanced now match. All 7 targets, GQA-aware kv_dim |
| 20 | Reference model = full weight_memory | model.eval() doesn't change weight storage |
| 21 | DPO activation 2× batch | uses_pairwise_data=True only for DPO/ORPO, not KTO |
| 22 | GaLore cap at 100% | Correct: can't need more optimizer states than full params |
| 23 | Pipeline bubble 1F1B formula (distributed.py) | (pp-1)/(pp+m-1) with m=4*pp. Reasonable default |
| 24 | Recompute FLOPs = total_forward | Standard checkpointing recomputes full forward. Correct |
| 25 | H100_PCIe (756 TFLOPS, 2000 GB/s) | Values match NVIDIA PCIe datasheet |

---

## NEEDS ADJUSTMENT (3 issues — correct direction, secondary concerns)

### ADJ-1: Hopper Flops Convention Inconsistency (LOW PRIORITY)

**Files**: `hardware/configs.py` — H100, H100_PCIe, H200, GH200 entries
**Issue**: All Hopper GPUs use TF32 TFLOPS (989/756), while B200 uses BF16 TFLOPS (2250). Since `compute_multiplier['bf16'] = 1.0` implies the Flops field IS the BF16 rate, Blackwell appears ~2.3× faster than Hopper in BF16 workloads when the true ratio is ~1.14×.
**Impact**: Cross-generation comparisons (H100 vs B200) produce incorrect relative performance. Same-generation comparisons (H100 vs H100) are unaffected.
**Research**: NVIDIA datasheets confirm BF16 dense = 1,979 TFLOPS for H100 SXM, 1,513 for H100 PCIe.
**Options**:
1. Change all Hopper entries to BF16 dense (H100=1979, H100_PCIe=1513, H200=1979, GH200=1979)
2. Change B200 to TF32 (B200=1125) — but B200 TF32 figure not widely published
3. Add a `flops_convention` field and let roofline analysis use the right multiplier

**Recommended**: Option 1 (change Hopper to BF16 dense), since BF16 is the dominant LLM precision.

---

### ADJ-2: Sequence Parallel Double-Division of TP Terms (LOW PRIORITY)

**File**: `training/calculator.py:458-461`
**Issue**: The activation formula computes:
```python
per_layer_bytes = s * b * h * (non_tp_coeff + tp_coeff / T + score_coeff / T)
```
Then when SP is enabled: `total_bytes /= tensor_parallel`

This divides ALL terms by T, including `tp_coeff/T` and `score_coeff/T` which are already divided. Result: TP terms are divided by T² instead of T.

**Impact**: Overestimates SP memory savings. For TP=4: non_tp terms correctly /4, but TP terms become /16 instead of /4. Since `non_tp_coeff` (10) dominates `tp_coeff/T` (~3.5) and `score_coeff/T` (~0.5), the error is ~25% underestimate of activation memory when SP=True.

**Research**: Korthikanti et al. 2022 (Megatron-LM): "With SP, all activation terms can be distributed along the sequence dimension across TP ranks." This means all terms get /T, but the base formula should not pre-divide TP terms.

**Fix**: Rewrite to separate TP and non-TP terms:
```python
if sequence_parallel and tensor_parallel > 1:
    per_layer_bytes = s * b * h * (non_tp_coeff + tp_coeff + score_coeff) / T
else:
    per_layer_bytes = s * b * h * (non_tp_coeff + tp_coeff / T + score_coeff / T)
```
**Note**: Default is `sequence_parallel=False`, so this only affects explicit SP usage.

---

### ADJ-3: Adafactor bytes_per_param Slightly Conservative (VERY LOW PRIORITY)

**File**: `training/optimizers.py:144`
**Issue**: `total_bytes_per_param = 1.0` is higher than the theoretical factorized cost (~0.3-0.5 bytes/param for large matrices). But for practical purposes with 1D tensors, momentum, and scaling, 1.0 is a reasonable upper bound.
**Impact**: Adafactor memory estimates ~2-3× higher than theoretical minimum, but still ~8× less than AdamW. Acceptable for a simulator.
**Note**: The GenZ-side `training_modeling.py:826` uses `'adafactor': 4` which is significantly worse (4× overestimate). This should be updated separately.

---

## NEW ISSUES DISCOVERED (3 — require fixes)

### ~~NEW-1: PP Latency Double-Division~~ — FALSE POSITIVE

**Files**: `llm_prefill.py:99-103`, `llm_decode.py:163-168`
**Investigation**: `create_full_prefill_model()` wraps `layers_per_stage` in `Repeat(PP-1)` + last stage outside.
`get_summary_table()` applies repeat multipliers, so `prefill_latency = (PP-1)*stage_latency + last_stage_latency ≈ PP * stage_latency`.
Therefore `stage_latency = prefill_latency / PP` is the FIRST division (extracting per-stage from full-pipeline total), NOT a double-division.

The formula `total_latency = prefill_latency + (PP-1)*stage_latency = (2PP-1)*stage_latency` correctly models 1F1B with M=PP micro-batches (batch_size is divided by PP at line 21).

**Verdict**: CONFIRMED CORRECT. No fix needed.

---

### NEW-2: GenZ-Side LoRA Formula Not Updated (CONFIRMED BUG)

**Files**: `training_modeling.py:3993-3996`, `training_parallelization.py:122-123`
**Code** (still broken):
```python
attn_lora_params = 4 * 2 * lora_rank * hidden_size * num_layers
ffn_lora_params = 2 * 2 * lora_rank * (hidden_size + intermediate_size) // 2 * num_layers
```
**Issues**:
1. Attention: `4 * 2 * r * h` assumes all 4 projections are h×h. Wrong for GQA (K,V should use kv_dim)
2. FFN: Only 2 targets (up, down), misses gate_proj. SwiGLU has 3 MLP projections
3. `// 2` integer division of the sum makes no mathematical sense

**Impact**: This causes `test_auto_parallelism_validation.py` to skip with "Memory ratio for lora is 1.05" — LoRA appears to use 105% of full fine-tuning memory through the GenZ code path. The calculator.py/advanced_calculator.py fixes are correct but this parallel code path was not updated.

**Fix**: Update both functions to match the corrected formula:
```python
kv_dim = hidden_size * num_kv_heads // num_heads
attn_lora_params = (
    2 * lora_rank * (hidden_size + hidden_size) +     # q_proj
    2 * lora_rank * (hidden_size + kv_dim) +           # k_proj
    2 * lora_rank * (hidden_size + kv_dim) +           # v_proj
    2 * lora_rank * (hidden_size + hidden_size)        # o_proj
) * num_layers
ffn_lora_params = 3 * 2 * lora_rank * (hidden_size + intermediate_size) * num_layers
```

---

### NEW-3: ZeRO-1 Missing in optimizers.py:calculate_optimizer_memory (CONFIRMED BUG)

**File**: `training/optimizers.py:577`
**Code** (still broken):
```python
if deepspeed_stage in ('zero2', 'zero3'):
    optimizer_memory /= data_parallel
```
**Issue**: `calculator.py` was fixed to include `'zero1'`, but the parallel code path in `optimizers.py` was not. The `advanced_calculator.py` calls `optimizers.py:calculate_optimizer_memory`, so the advanced calculator does NOT shard optimizer memory for ZeRO-1.
**Impact**: Advanced calculator's ZeRO-1 memory estimates are 8× too high (for DP=8).
**Fix**: Change to `if deepspeed_stage in ('zero1', 'zero2', 'zero3'):`

---

## SECONDARY FINDINGS (design observations, not blocking)

### Eviction Partial State (LOW PRIORITY)
- `evict_blocks()` can partially evict a request's blocks. Partially-evicted requests remain in `_running` with corrupted KV cache.
- Evicted-and-requeued requests don't reset `tokens_generated`, causing double-counted tokens after re-prefill.
- **Recommendation**: Either evict entire requests only, or reset decode state on re-queue.

### Power Model Timing (LOW PRIORITY)
- `batch_latency_ns` uses wall-clock gap between batch completions, not actual compute time. This includes idle time in the "active energy" calculation.
- `batch_util` proxy (batch_size / max_batch_size) doesn't account for token count variation.
- **Recommendation**: Use scheduler's estimated batch latency for power calculation instead of wall-clock gaps.

---

## OVERALL ASSESSMENT

The 25 bug fixes are fundamentally correct and significantly improve accuracy. The 3 new issues found are real but lower-severity — they affect specific code paths (GenZ-side LoRA, advanced calculator ZeRO-1, and PP>1 latency). The core calculator and serving simulation fixes are solid.

**Priority order for remaining fixes**:
1. **NEW-2** (GenZ LoRA formula) — causes test skip, affects LoRA memory estimates through GenZ path
2. **NEW-3** (ZeRO-1 in optimizers.py) — affects advanced calculator ZeRO-1 estimates
3. **ADJ-2** (SP double-division) — only affects explicit SP=True usage
4. **ADJ-1** (Hopper Flops convention) — affects cross-generation GPU comparisons

---

## OPTIMAL FIX PLAN

### Fix A: GenZ-Side LoRA Formula (CONFIRMED BUG — HIGH PRIORITY)

**File 1**: `llm-memory-calculator/src/llm_memory_calculator/genz/LLM_training/training_modeling.py:3986-3998`

The function `_calculate_trainable_params` receives `hidden_size`, `intermediate_size`, `num_layers`, `lora_rank` but is missing `num_attention_heads` and `num_key_value_heads` for GQA-aware calculation.

**Current broken code** (line 3993-3996):
```python
attn_lora_params = 4 * 2 * lora_rank * hidden_size * num_layers
ffn_lora_params = 2 * 2 * lora_rank * (hidden_size + intermediate_size) // 2 * num_layers
```

**Bugs**: (1) Assumes all 4 attention projections are h×h (wrong for GQA K/V), (2) Only 2 FFN targets instead of 3, (3) `//2` integer division makes no mathematical sense.

**Fix**: Add `num_attention_heads` and `num_key_value_heads` parameters to the function, then:
```python
num_kv_heads = num_key_value_heads if num_key_value_heads else num_attention_heads
kv_dim = hidden_size * num_kv_heads // num_attention_heads
attn_lora_params = (
    2 * lora_rank * (hidden_size + hidden_size) +      # q_proj
    2 * lora_rank * (hidden_size + kv_dim) +           # k_proj
    2 * lora_rank * (hidden_size + kv_dim) +           # v_proj
    2 * lora_rank * (hidden_size + hidden_size)        # o_proj
) * num_layers
ffn_lora_params = (
    2 * lora_rank * (hidden_size + intermediate_size) +  # gate_proj
    2 * lora_rank * (hidden_size + intermediate_size) +  # up_proj
    2 * lora_rank * (hidden_size + intermediate_size)    # down_proj
) * num_layers
```
**Also update all callers** to pass the new parameters.

**File 2**: `llm-memory-calculator/src/llm_memory_calculator/genz/LLM_training/training_parallelization.py:120-124`

Same formula with same bugs. Additionally has hardcoded `lora_rank = 16`.

**Fix**: Extract `num_kv_heads` from `model_config` and apply same corrected formula.

**Regression test**: After fix, `test_auto_parallelism_validation.py` LoRA memory ratio should fall within [0.05, 0.4] instead of skipping at 1.05.

---

### Fix B: ZeRO-1 in optimizers.py (CONFIRMED BUG — MEDIUM PRIORITY)

**File**: `llm-memory-calculator/src/llm_memory_calculator/training/optimizers.py:577`

**Current broken code**:
```python
if deepspeed_stage in ('zero2', 'zero3'):
    memory_gb /= data_parallel
```

**Fix**:
```python
if deepspeed_stage in ('zero1', 'zero2', 'zero3'):
    memory_gb /= data_parallel
```

**Impact**: `advanced_calculator.py` calls this function, so ZeRO-1 through the advanced calculator path currently does NOT shard optimizer memory. Basic calculator (calculator.py) was already fixed.

**Regression test**: Advanced calculator with ZeRO-1 DP=8 should show ~8× optimizer memory reduction.

---

### Fix C: Sequence Parallel Double-Division (LOW PRIORITY)

**File**: `llm-memory-calculator/src/llm_memory_calculator/training/calculator.py:458-461`

**Current code**:
```python
per_layer_bytes = s * b * h * (non_tp_coeff + tp_coeff / T + score_coeff / T)
...
if sequence_parallel and tensor_parallel > 1:
    total_bytes /= tensor_parallel  # Divides ALL terms by T, but tp/score already divided
```

**Fix**: When SP is enabled, all terms should be divided by T exactly once:
```python
if sequence_parallel and tensor_parallel > 1:
    per_layer_bytes = s * b * h * (non_tp_coeff + tp_coeff + score_coeff) / T
else:
    per_layer_bytes = s * b * h * (non_tp_coeff + (tp_coeff + score_coeff) / T)
```
Then remove the later `total_bytes /= tensor_parallel` when SP is on.

**Impact**: Only affects `sequence_parallel=True` (default is False). Overestimates SP savings by ~25%.

---

### Fix D: Hopper Flops Convention (LOW PRIORITY — DATA ONLY)

**File**: `llm-memory-calculator/src/llm_memory_calculator/hardware/configs.py`

All Hopper GPUs use TF32 TFLOPS while B200 uses BF16 TFLOPS. Since `compute_multiplier['bf16'] = 1.0`, the Flops field is treated as the BF16 rate.

**Option 1** (recommended): Update Hopper entries to BF16 dense:
- H100_GPU: 989 → 1979
- H100_PCIe_GPU: 756 → 1513
- H200_GPU: 989 → 1979
- GH200_GPU: 989 → 1979

**Impact**: Cross-generation comparisons (H100 vs B200) will be accurate. All same-generation comparisons unchanged.

---

## VERIFICATION CRITERIA

For each fix, confirm:
1. All 865 existing tests still pass (0 regressions)
2. Previously-skipping tests now pass (especially `test_auto_parallelism_validation.py` for Fix A)
3. Basic and advanced calculators produce consistent results
4. New regression tests cover the exact bug condition
