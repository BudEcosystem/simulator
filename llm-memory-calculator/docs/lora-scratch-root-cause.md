# LoRA "prefill scratch" on vLLM CPU — root cause analysis

**TL;DR.** The gigabytes we have been calling "LoRA prefill scratch" are not model
physics. They are an implementation artifact of vLLM's **CPU** LoRA path: the
reference ops materialize **one copy of the stacked LoRA weight matrix per token**.
The term is therefore computable from first principles — `T × max_lora_rank ×
module out_features × dtype`, times a small live-concurrency factor — and it does
not exist on CUDA at all. Architecture enters only through module dimensions and
through which modules the engine wraps; it is not a fundamental per-architecture
constant. The best fix is a ~20-line vLLM patch, not a bigger budget.

---

## 1. The mechanism, from source

`vllm/lora/ops/torch_ops/lora_ops.py` (used by `PunicaWrapperCPU`; CUDA uses
`triton_ops` instead — verified from the imports):

```python
def bgmv_expand(inputs, lora_b_weights, output_tensor, lora_indices_tensor, ...):
    selected_loras = lora_b_weights[lora_indices_tensor].to(dtype=output_tensor.dtype)
    ...
    outputs = torch.einsum("bi, boi -> bo", inputs, selected_loras)
```

`lora_indices_tensor` is per-token, shape `(T,)` (SGMV explodes it with
`repeat_interleave`). `lora_b_weights` is the **stacked slot tensor**
`(max_loras, 1, out_features, max_lora_rank)`. Fancy-indexing a `(T,)` tensor into
it materializes:

> **`(T, out_features, max_lora_rank)` — one full copy of the LoRA B matrix per
> token** — per wrapped module, per layer, per forward pass. The shrink side does
> the same with the A matrix, in fp32.

Three properties follow, each experimentally confirmed below:

1. It scales with the **configured** `max_lora_rank` (the stacked slot's rank),
   even though warmup dummy adapters are capped at rank 8
   (`lora_model_runner_mixin.py:104`) — the copy is of the slot, not the adapter.
2. It scales with `T` (`max_num_batched_tokens`) and with the wrapped modules'
   `out_features` / `in_features` — i.e. `intermediate_size` dominates (gate/up
   slices are the largest).
3. It is **CPU-only**. `punica_gpu.py` imports Triton grouped-GEMM kernels that
   never materialize per-token copies.

An all-zeros index tensor still copies (`weights[zeros(T)]` gathers slot 0, T
times), which is why the cost appears at warmup even though the profile run
activates zero adapters.

## 2. Experimental confirmation (Qwen3-0.6B, T=4273, control-subtracted)

| probe | change vs baseline | peak | scratch | verdict |
|---|---|---:|---:|---|
| baseline | — (rank 64) | 11.08 GiB | 6.10 GiB | — |
| **p-rank32** | `--max-lora-rank=32` | 8.48 GiB | 3.50 GiB | **linear in configured rank** (fit: `0.081 GiB/rank + 0.90 GiB`) — kills the "warmup is rank-8" objection and confirms the stacked-slot copy |
| **p-malloc** | `MALLOC_TRIM_THRESHOLD_=0`, fixed mmap threshold | 11.17 GiB | ~same | **not glibc retention — the peak is live memory** |
| **p-loras2** | `--max-loras=2` | 11.55 GiB | 6.57 GiB | **+7.7%, not the 2x the old formula predicts** — indexing selects one slot per token, `max_loras` only grows the (small) slot storage |
| **p-omp4** | `OMP_NUM_THREADS=4` | 11.08 GiB | 6.10 GiB | **identical to baseline** — no per-thread scratch; within-CPU the term is core-count-insensitive |

Cross-model check — measured scratch as a multiple of one live copy of the
largest wrapped slice (`T × intermediate_size × rank × 2 B`):

| model | T | live-copy unit | measured | implied concurrency `C` |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 4273 | 1.68 GB | 6.55 GB | 3.9 |
| Qwen3-4B | 4273 | 5.32 GB | 25.84 GB | 4.9 |
| Qwen3-4B | 2048 | 2.55 GB | 16.60 GB | 6.5 |
| Llama-3.1-8B | 4273 | 7.84 GB | 21.70 GB | 2.8 |
| Gemma-4-12B | 2048 | 4.03 GB | 9.27 GB | 2.3 |

`C` = how many such copies are live at the peak instant (einsum transpose
temporaries, per-module overlap). It spans **2.3–6.5** — an engine scheduling
detail, not model physics — where the previous "per-architecture multiplier"
spanned 100× with no explanation. `C = 7` is a conservative envelope covering
every measurement.

Sanity check at the deployment that triggered this analysis: Qwen3-4B, T=9583,
rank 64 → `7 × 9583 × 9728 × 64 × 2 B = 83.5 GB` — the shipped table predicted
84.0 GB. **The number was right; the model behind it was wrong.** Now we know
which knobs actually move it.

## 3. Answers to the question asked

**Is it a model-architecture problem?** Only indirectly. Architecture determines
(a) the wrapped modules' dimensions (`intermediate_size`, qkv widths — all
readable from `config.json`) and (b) *whether the engine's warmup exercises the
path at all*. The 100× per-architecture spread in the fitted table decomposes
entirely into those two plus `C`.

**Is it a hardware problem?** Yes, in the sharpest sense: **the term only exists
on the CPU backend.** CUDA's Triton kernels do grouped GEMM without per-token
copies. (HPU has its own path — unverified.) Within CPU, the allocator is ruled
out (p-malloc); thread-count is ruled out too (p-omp4: identical peak at 4 vs 12 OMP threads).

## 4. The correct calculation

For vLLM CPU, LoRA enabled, per TP rank:

```
scratch_bytes ≈ C × T × R × S_max × 2      (expand side, bf16)
              + c_base                      (~1 GB observed; shrink + indexing overhead)

T     = max_num_batched_tokens
R     = configured max_lora_rank   (NOT the adapter's actual rank)
S_max = largest wrapped slice out_features (= intermediate_size for gated FFN)
C     = live-concurrency factor; measured 2.3–6.5, use 7 as a budget envelope
```

For CUDA/HPU: **zero** (pending an HPU check). `max_loras` does not appear (confirmed: 2 adapters cost +7.7%, not 2x).

## 5. Defects this exposes, beyond the formula

1. **budsim applies the term without device gating** (`heuristic.py` adds
   `max_loras` to `calc_kwargs` unconditionally). On CUDA the phantom 20+ GiB is
   charged against *VRAM* totals — it can reject valid GPU configs or force
   unnecessary TP.
2. **The hybrid anomaly is a landmine, not a discount.** Qwen3.5-4B measured
   ~0.21 GiB — consistent with warmup *not exercising* the punica path for
   GDN-hybrid models (the profile run activates `num_active_loras=0`, and the
   hybrid's execution path appears to skip the copies). If a **real** adapter
   request at serving time does take the copy path, a pod budgeted from the
   near-zero measurement OOMs on its first big LoRA prefill — after passing
   warmup. Needs a serving-time test with a real adapter before trusting the
   0.034 multiplier.
3. **The right fix is upstream.** For `max_loras=1` (our default),
   `bgmv_expand` is one matmul with one B matrix — no `(T, out, r)` gather is
   needed; grouped-by-adapter matmul handles the general case. A ~20-line
   `torch_ops` patch collapses ~24 GiB of scratch to megabytes and makes the
   entire budgeting question moot on CPU.

## 6. Recommended actions

| action | effect |
|---|---|
| Patch vLLM CPU `torch_ops` (group-by-adapter matmul) | eliminates the term; the real fix |
| Until then: replace the per-arch multiplier table with the mechanistic formula (§4) | predictable across unmeasured architectures; no more 100× table |
| Gate the term on `device ∈ {cpu, cpu_high}` in budsim | stops charging phantom GiBs to CUDA plans |
| Serving-time LoRA test on a hybrid model | closes the landmine in (2) |
| Consider lowering `max_lora_rank` default | scratch is linear in it; rank 32 halves the term |
