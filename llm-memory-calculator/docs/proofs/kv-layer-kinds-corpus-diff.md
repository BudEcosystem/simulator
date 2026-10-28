# Proof: KV layer kinds -- before and after

Branch `fix/hybrid-kv-layer-kinds`, measured against **origin/main `c17ef89`** (what this branch
lands on). Generated on the shared GB10 box, 27-28 Oct 2026, with `scripts/kv_corpus.py`; no GPU
used. Four rounds: the fix itself; the corrections an independent review asked for (the
chunked graph, the state's bytes, MoE placement, unplaced windows, depth from a layer list,
SambaY, the GGUF reader's Mamba-1 / RoPE / gating / tying, and this proof's own coverage); and the
second review's (Zamba's shared MLP and its unknown names, transformers before 4.53, the FRD-027
facts' allowlist and `mistral` rule, RWKV / xLSTM state bytes in the graph, Llama 4's chunk in
the graph, optional test dependencies); and the third review's (the serving path's missing
recurrent state, GPT-Neo's layout under `attention_types`, per-layer lists shorter than the
depth, Kimi's one-indexing without `kda_layers`, and the two window policies stated).

## 1. The measured case

Qwen3.5-4B on the GB10 (llama.cpp and mistral.rs servers, 27 Oct 2026): the paged pool is held by
the 8 `full_attention` layers only, 129 MiB at 4,096 tokens ("Allocating 129 MB for
PagedAttention"); the 24 `linear_attention` (Gated DeltaNet) layers hold a fixed state.

| Path | origin/main | This branch | Truth |
|---|---|---|---|
| calculator, HF config (`calculate_memory`, engine estimate) | 128 MiB (already right) | 128 MiB | 129 MiB measured (128 MiB + page rounding) |
| `bud_component.py --gguf` Qwen3.5-2B (6 of 24 layers attend) | 192 MiB (24 layers) | **48 MiB** (6 layers) | llama.cpp allocates 6 layers |
| GenZ decode graph KV, Qwen3.5-4B, 4,096 tokens | 512 MiB (32 layers) | 128 MiB + 49.1 MiB fixed state | 128 MiB KV; state 49.1 MiB (fp32 matrix, as the calculator) |
| GenZ chunked graph KV (512-token chunk + one 4k decode) | 576 MiB (32 layers) | 242 MiB | 8 layers + state |
| GenZ `get_kv_size` / disaggregation transfer (x 2 B) | 65,536 elem/token | 16,384 | 8 x 2 x 4 x 256 = 16,384 |
| serving `MemoryModel.bytes_per_token_kv` | 131,072 | 32,768 | 32,768 |
| serving per-request state (`MemoryModel.state_bytes_per_sequence`) | none | 51,511,296 B | `calculate_state_memory`: 51.5 MB |
| serving: 1,024-token requests admitted into 8 GiB | 64 (round-3 head: 256) | **100** | 8 GiB / (32 MiB KV + 51.5 MB state) = 100 |
| disaggregation transfer, 512 + 128 tokens | 83,886,080 B (round-3 head: 20,971,520) | 72,482,816 B | 20,971,520 KV + 51,511,296 state |
| engine estimate, state term | 0 | 51.5 MB in the total | 24 x (32x128x128 x 4 B + 24,576 x 2 B) |

## 2. Red first

`tests/test_layer_kinds_by_category.py`, 100 cases.

* Round 1 (38 cases) against the merge base `1cc4f1e`: 25 failed, 13 passed (the 13 are HF-config
  calculator cases upstream had already fixed).
* Round 2 (13 cases added for the review) against the round-1 head `51cc203`: 12 failed, 1 passed
  (`test_a_dense_layer_of_an_moe_model_graphs_under_expert_parallelism` guards the new per-layer
  MoE graph against expert parallelism; it could not fail before MoE was placed per layer):
  - test_the_chunked_graph_charges_kv_only_on_the_attention_layers (512 MiB growth, not 128)
  - test_the_chunked_graph_bounds_windowed_layers_by_their_window (288 MiB, not 144)
  - test_the_genz_recurrent_state_is_priced_like_the_calculator (153.1 MiB, not 177.1)
  - test_the_genz_graph_places_moe_only_on_the_layers_the_config_says (32 MoE, not 16)
  - test_the_genz_jamba_weights_match_the_parameter_count
  - test_a_window_without_a_stated_layout_follows_the_config_class
  - test_a_window_on_an_unknown_model_type_is_charged_in_full_and_said_so
  - test_depth_comes_from_the_per_layer_list_when_the_config_omits_it
  - test_a_nemotron_pattern_reads_its_moe_slots
  - test_a_mamba1_gguf_is_sized_as_mamba1
  - test_the_qwen35_gguf_parameter_count_matches_its_tensors
  - test_sambay_kv_is_never_below_its_one_shared_full_cache
* Round 3 (9 cases added for the second review) against the round-2 head `6815b6a`: all 9
  failed. One assertion was added after the first fix, and checked red against a copy of the tree
  with the fix's other half reverted (a `qwen35` config's state priced 0):
  - test_zamba2_runs_its_shared_mlp_on_every_hybrid_layer (no FFN on any of the 54 layers)
  - test_zamba2_facts_count_the_shared_mlp_once_per_block (no `dense_ffn` role at all)
  - test_an_unknown_name_in_a_zamba_layer_list_is_charged_full_kv_and_said_so (charged as Mamba,
    no warning)
  - test_a_transformers_release_without_layer_layouts_places_no_window (Gemma 2 clamped to the
    window on every layer)
  - test_model_facts_read_layer_kinds_for_a_model_type_the_recogniser_lacks (`qwen35` GGUF and
    MiniMax-Text-01: KV on every layer)
  - test_model_facts_slide_a_uniform_window_by_the_table_not_by_family_name (Phi-3: 32 full)
  - test_the_genz_state_of_every_mechanism_is_priced_like_the_calculator[rwkv] (0.75 MiB, not
    0.9375) and [xlstm] (128.1255, not 128.2510)
  - test_the_genz_graph_bounds_chunked_layers_by_their_chunk (6,144 MiB at 32k, not 2,688)
* Round 4 (40 cases added for the third review) against the round-3 head `2e30f66`: all 40
  failed (39 in the first run; the fortieth,
  test_gpt_neo_is_not_taken_for_a_hybrid_by_its_attention_layers_key, was written once the
  corpus showed GPT-Neo's phantom state, and failed before its fix). Two `BudSimulator/tests/test_serving_api.py` cases were added
  beside them (the memory-tier API had no state field):
  - test_the_serving_memory_model_charges_each_request_its_fixed_state x 7 (Qwen3.5-4B,
    granite-4.0-h-small, Nemotron-H-8B, Falcon-H1-34B, LFM2-2.6B, mamba-2.8b, Mamba-Codestral:
    no state term; Qwen3.5 admitted 256 requests, mamba-2.8b without bound)
  - test_the_serving_state_is_split_by_tensor_parallelism, test_a_dense_model_carries_no_serving_state,
    test_the_serving_state_is_freed_with_its_request,
    test_a_pure_ssm_spills_without_dividing_by_its_zero_kv (ZeroDivisionError on the round-3 head,
    where a pure SSM's KV is 0), test_the_scheduler_fails_a_request_whose_state_alone_exceeds_every_tier
  - test_serving_concurrency_counts_the_fixed_state x 14 (cluster and disaggregation `B_eff`; a
    pure SSM's 0 footprint answered 1)
  - test_the_disaggregated_transfer_carries_the_state x 7 (a pure SSM moved 0 bytes)
  - test_gpt_neo_alternates_its_window_by_attention_types (24 sliding, not 12 + 12; facts 50.3 MB
    at 2,048 tokens, truth 226.5 MB), test_gpt_neo_attention_types_alone_state_the_layout,
    test_the_calculator_charges_gpt_neo_its_global_and_local_kv (0 bytes: `num_heads` unread)
  - test_a_per_layer_list_shorter_than_the_depth_charges_the_rest_full_and_says_so (16 of 32
    layers free, no warning), test_a_short_hybrid_list_keeps_its_stated_kinds_and_charges_the_rest_full
  - test_kimi_layer_lists_are_one_indexed_even_without_kda_layers (6 full layers, not 7)
* In a clean venv with neither `transformers` nor `gguf` (the calculator's own dependencies), four
  older cases failed instead of skipping; they now skip, and every other case passes there.
* This branch: 100 passed (the two GGUF-file cases skip where `gguf` is absent).
* Two older SambaY tests in `tests/test_hybrid_layer_plan.py` pinned the all-windowed reading
  ("every layer sliding", an "OVER-estimate"). That reading under-counts at long context, so they
  now pin the upper bound -- one full layer, the window elsewhere, never below the published
  design -- and keep their original guard: the window is never discarded (the 195x blow-up).

## 3. The whole suites

| Suite | origin/main `c17ef89` | This branch |
|---|---|---|
| `llm-memory-calculator/tests` (pytest) | 2,091 passed, 14 skipped, 1 failed | 2,191 passed, 14 skipped, 1 failed |
| `BudSimulator/tests` (pytest) | 24 passed | 26 passed |

(Round 3 ended at 2,151 / 24; round 4 adds the 40 category cases and the two memory-tier API
cases. One existing assertion was widened: `tests/serving/test_memory_model.py::
TestMemorySnapshot::test_snapshot_has_all_fields` pins the snapshot's keys, and the snapshot now
reports `state_bytes` beside `kv_cache_bytes`.)

The one failure is the same test on both sides and predates this work:
`tests/planner/test_runtime_spec.py::test_committed_schemas_are_current` compares the committed
FRD-027 JSON schemas with ones regenerated by the installed pydantic, which differ on this
machine's pydantic (2.13).

## 4. Corpus: every model config the simulator ships or this machine caches

193 configs measured: the 26 rows of `BudSimulator/data/prepopulated.db` that carry a config, the
29 fixtures under `tests/fixtures` (round 4 added GPT-Neo-1.3B's hub config), the 44 decoder configs in the HuggingFace cache and model
folders, 5 GGUFs read through `bud_component.gguf_to_hf_config` (beside each file's own tensor
count), and the 89 built-in GenZ models. The database's other **63 rows have `config_json = '{}'`**
and are listed in the JSON (`_empty_db_rows`), not measured: they would only measure the
calculator's defaults. Both trees were measured with the same script, the same committed database
and the same fixtures.

Figures per config: the calculator's KV at 4,096 and 32,768 tokens, its layer breakdown, the
recurrent state, `calculate_memory`'s weights / params / total; the engine estimate
(`estimate_memory_for_engine`, llama.cpp and generic, 4k and 32k: KV, state, total, params); the
FRD-027 `ModelFacts` layer counts; GenZ's attention and FFN kinds, `get_kv_size` (the
disaggregation KV transfer is this x 2 B per token), the decode graph's KV and weights, the
chunked graph's KV; and the serving `MemoryModel`'s bytes per token, its per-request state, the
1,024-token requests that fit 8 GiB, and the disaggregation transfer for 512 + 128 tokens (KV x
2 B, plus the state where the tree counts it). Round 4 added the last three figures and re-measured
origin/main, the round-3 head and this branch with the same script.

Reproduce (the first `PYTHONPATH` entry is the tree measured):

    PYTHONPATH=<tree>/llm-memory-calculator/src python llm-memory-calculator/scripts/kv_corpus.py \
        --out after.json --db BudSimulator/data/prepopulated.db \
        --fixtures llm-memory-calculator/tests/fixtures/hybrid_configs llm-memory-calculator/tests/fixtures \
        --hf-cache ~/.cache/huggingface/hub --models /home/bud/models --repo-root <tree> --genz-data <scratch>
    python llm-memory-calculator/scripts/kv_corpus.py --diff before.json after.json --report diff.md

**Result: 134 unchanged, 59 changed** (193 configs; the empty-row marker is unchanged). No uniform dense
transformer's KV moved in any output. No config that measured before fails now; three that failed
now measure (Jamba, Kimi Linear, DeepSeek-V2-Lite in GenZ). Every change, by cause:

* **Recurrent layers no longer charged a KV cache** (Qwen3.5-4B/-Base and two Qwen3.5
  derivatives, Qwen3.6-27B x2, Qwen3-Next, LFM2 / LFM2.5-Audio, MiniMax-Text-01, Kimi Linear, the
  Qwen3.5-2B GGUF): GenZ decode / chunked KV, `get_kv_size`, the serving bytes per token and the
  disaggregation transfer fall to the attention layers; GenZ weights rise slightly because each
  mixer is charged its own parameters instead of an attention block's. In the calculator only the
  GGUF moves (24 -> 6 KV layers); every HF-config hybrid was already right there.
* **Mamba hybrids regain their attention layers in GenZ** (Jamba, Bamba, granite-4.0-h,
  Nemotron-H x2, Zamba2, Falcon-H1, Hymba, the built-in Jamba models): KV rises from "all Mamba"
  to the attention layers plus state; weights fall to the real parameter counts (Nemotron-H-8B
  29,104 -> 14,427 MiB = 7.6B params; Jamba now builds: 97,850 MiB = 51.3B params, real 51.6B).
* **The recurrent state is counted**: the engine estimate's total now includes the state term for
  every hybrid and SSM (falcon-mamba-7b +294 MB, Bamba-9B +123 MB, Qwen3.5-4B +51.5 MB); without it
  removing the KV would have turned the over-count into an under-count. Pure SSMs' `get_kv_size`
  and serving bytes fall to 0 (they hold no KV). Since round 4 the serving path counts it too:
  each request is charged its state beside its KV blocks, in the admission check, `B_eff` and the
  prefill-to-decode transfer (below).
* **MLA read as its latent** in the serving model and the chunked graph (DeepSeek-V3 / V3.2:
  1,748,992 -> 70,272 B/token = 61 x 576 x 2 B; DeepSeek-V2 1,228,800 -> 69,120; DeepSeek-V2-Lite
  221,184 -> 31,104; Kimi Linear 248,832 -> 8,064; the built-in DeepSeek-V3-Base 27,328 -> 70,272,
  which was low).
* **Per-layer attention read in the chunked graph and the KV counters** (Nemotron-51B, DeciLM:
  54 of 80 layers attend, with 1-8 KV heads each; 18 are linear replacements and 8 no-ops):
  `get_kv_size` 131,840 -> 94,976 elements per token, serving 327,680 -> 189,952 B/token, chunked
  KV 1,440 -> 834.75 MiB. Its decode graph, which already read the per-layer configs, is
  unchanged at 742 MiB -- the counters now agree with it.
* **Windowed layers bounded in GenZ** (Gemma-3 27B decode 1,984 -> 736 MiB; SmallThinker;
  gpt-oss chunked 216 -> 123 MiB) and in the FRD-027 facts (Gemma-3 27B [62 full] -> [10 full, 52
  sliding, window 1,024]).
* **MoE placed per layer in GenZ** (Jamba's dense even layers; the first `first_k_dense_replace`
  layers of DeepSeek-V3 / V3.2 / V2-Lite / deepseek-moe-16b / Ling-flash / Kimi Linear): weights
  fall by the experts those layers do not have (DeepSeek-V3 1,335,494 -> 1,273,240 MiB = 667.6B).
* **The GGUF reader states what the file states**: RoPE, a gated FFN, tied embeddings. Parameter
  counts against the files' own tensors: Qwen3.5-2B 1.901B -> 1.856B (truth 1.882B), gpt-oss-20b
  1.99B -> 20.908B (truth 20.915B; the MoE geometry is the merged accuracy-remediation commits),
  SmolVLM-500M 369.8M -> 409.25M (exact), nomic-embed-text-v1.5 110.0M -> 136.7M (truth 136.7M).
  The calculator's KV moved for exactly two configs: this GGUF and SambaY below.
* **SambaY (Phi-4-mini-flash)**: KV is now an upper bound (15 windowed layers + one full shared
  cache) instead of 32 windowed layers, which under-counted at long context. With the default
  engine capabilities -- no heterogeneous per-layer KV layout, the measured vLLM behaviour -- the
  windowed layers are charged the full sequence, so the default figure is 0.336 GB at 4k and
  2.68 GB at 32k (was 0.084 GB at both); with `heterogeneous_kv_layout` it is 15 x 512 + the
  context. Its parameter count rises 4.33B -> 4.67B (real 3.85B): the 16 Mamba mixers are now
  sized as mixers, from fallback geometry (the config states none; flagged as an ESTIMATE). The
  residual over-count is older and separate: the config states no RoPE key, so the parameter
  counter assumes a learned 262,144-row position table (671M); it is not changed here.
* **Unplaced windows** (`stated_layer_types`): no config in this corpus carries a window without a
  layout on a model type transformers does not know, and none takes its layout from a config class
  -- so this rule moves no corpus figure; its tests are synthetic (Gemma-2 shape, unknown type).
  Likewise the round-3 rule that a transformers release before 4.53 places no window (this
  machine runs 5.12); checked by hand in a venv with 4.50.3: Gemma 2's hub config goes from the
  window on all 42 layers (1.41 GB at 32k, flat) to full KV with the warning (11.27 GB).

Round 3 (the second review) moved 11 configs against the round-2 head, each by one of these:

* **Zamba2 runs its shared MLP on its 9 `hybrid` layers** in GenZ: FFN kinds `no_op` x 54 ->
  `Dense` x 9 + `no_op` x 45, decode weights 4,714.5 -> 5,614.5 MiB. (On origin/main the graph had
  54 Dense FFNs.) The parameter count is unchanged: the block's weights are still counted once per
  `num_mem_blocks`. The FRD-027 facts now count them the same way (they counted none).
* **The FRD-027 facts read layer kinds from the table for every model type** (the recogniser's
  allowlist only flags the type): the Qwen3.5-2B GGUF [24 full] -> [6 full, 18 recurrent];
  MiniMax-Text-01 [80 full] -> [10 full, 70 recurrent]; SmallThinker [52 full] -> [13 full, 39
  sliding, window 4,096]; five ModernBERT encoder configs (Julia-1, Laya) take their `layer_types`
  (e.g. [28 full] -> [10 full, 18 sliding]; their `local_attention` window is not a window key the
  table reads, so the sliding layers keep the full context -- an over-estimate).
* **The facts slide a uniform window by the table, not by `model_type == "mistral"`**: Mimi
  [8 full] -> [8 sliding, window 250] and Voxtral-Mini-4B-Realtime [26 full] -> [26 sliding,
  window 8,192] -- what the calculator already charged for both.
* RWKV / xLSTM state bytes and Llama 4's chunk in the graph move no corpus config (none of those
  shapes is in it); their tests are synthetic.

Round 4 (the third review) moved 33 configs against the round-3 head, each by one of these, and
nothing else (`--diff` of the round-3 head's corpus against this branch's):

* **The serving memory model charges each request its recurrent state** (32 configs, every one
  with a recurrent layer; no dense model moved). `serving_state_bytes` equals the calculator's
  `state_gb` for every config the calculator measures, and the 1,024-token requests 8 GiB admits
  fall back to what KV + state allow. origin/main -> round-3 head -> this branch, against
  truth = 8 GiB / (KV + state):

  | config | origin/main | round-3 head | this branch | state per request |
  |---|---|---|---|---|
  | Qwen3.5-4B (and 5 derivatives) | 64 | 256 | **100** | 51.5 MB |
  | granite-4.0-h-small | 51 | 512 | **50** | 152.8 MB |
  | Nemotron-H-8B | 39 | 512 | **72** | 102.1 MB |
  | NVIDIA-Nemotron-Nano-9B-v2 | 36 | 512 | **53** | 143.5 MB |
  | Bamba-9B-v2 | 64 | 682 | **63** | 123.1 MB |
  | Qwen3-Next-80B | 85 | 341 | **83** | 77.3 MB |
  | Qwen3.6-27B (x2) | 32 | 128 | **38** | 153.9 MB |
  | MiniMax-Text-01 | 25 | 204 | **25** | 293.6 MB |
  | Kimi Linear | 33 | 1,040 | **166** | 43.4 MB |
  | Jamba-v0.1 (and the 2 built-in Jamba 1.5 / v0.1) | 64 | 512 | **261** | 16.1 MB |
  | built-in AI21-Jamba-1.5-Large | 28 | 227 | **78** | 72.3 MB |
  | Zamba2-2.7B | 15 | 91 | **51** | 72.5 MB |
  | Falcon-H1-34B (parallel) | 56 | 56 | **18** | 304.2 MB |
  | Hymba-1.5B (parallel) | 204 | 204 | **174** | 7.2 MB |
  | Phi-4-mini-flash (SambaY) | 51 | 102 | **95** | 5.7 MB |
  | LFM2-2.6B / LFM2.5-Audio x2 | 136 / 1,024 | 512 / 2,730 | **503 / 2,717** | 0.27 / 0.015 MB |
  | Qwen3.5-2B GGUF | 170 | 682 | **267** | 19.5 MB |
  | mamba-2.8b-hf (fixture and built-in) | 12 | unbounded | **374** | 22.9 MB |
  | Mamba-Codestral-7B | 16 | unbounded | **31** | 272.4 MB |
  | falcon-mamba-7b (database) | 8 | unbounded | **29** | 293.6 MB |
  | built-in falcon-mamba-7b / mamba-130m | 8 / 113 | unbounded | **234 / 3,328** | 36.7 / 2.6 MB |

  (origin/main charged every recurrent layer a full KV cache and no state; the round-3 head
  dropped that KV and added nothing.) The disaggregation transfer for 512 + 128 tokens moves the
  state with the KV: Qwen3.5-4B 20,971,520 -> 72,482,816 B, mamba-2.8b 0 -> 22,937,600 B.
* **GPT-Neo-1.3B reads its layout from `attention_types`**: [24 sliding] (round 3; origin/main
  [24 full]) -> [12 full, 12 sliding, window 256] in the facts and the calculator's breakdown
  (`layer_plan:attention_types`). Its head count is read from `num_heads`, so the calculator
  charges KV at all: 0 -> 0.805 GB at 4k, 6.44 GB at 32k (24 layers x the context: the stack is
  mixed, so the window is withheld by default; 12 x context + 12 x 256 with
  `heterogeneous_kv_layout`). Its `attention_layers` key no longer makes it a "hybrid" in
  `detect_model_type` (which now reads the head count and depth through the normalizer:
  `decoder-only`), so the phantom Mamba state guessed for it (1.7 MB on origin/main, 3.4 MB on
  the round-3 head) is gone: GPT-Neo has no recurrent layer. `calculate_memory`'s total also
  re-estimates its activations as a decoder's (+0.16 GB).
* A per-layer list shorter than the depth, and Kimi lists without `kda_layers`, move no corpus
  config (no shipped config has either shape); their tests are synthetic.

Found while checking round 4, not changed by it (present on origin/main): the database's
falcon-mamba-7b states `expand: 16` beside `intermediate_size: 8192`; transformers'
`FalconMambaConfig` keeps the stated 8,192 as the mixer width, but `mechanism_dims.mamba_dims`
derives `expand x hidden` = 65,536, so its parameters (54.5B, on origin/main too) and state
(293.6 MB; the built-in GenZ model with the real width: 36.7 MB) are 8x high. An over-estimate,
left for a separate fix.

The full per-config table (before = origin/main, after = this branch) follows.

## config:chaos-pressure/gpt-oss-20b/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| genz_chunked_kv_mb | 216.0 | 123.0 |

## config:models--Cloudflare--clef-flash/snapshots/17f0b0ad64efb65d273590632833508766b2aae6/config.json (multimodal)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 83886080 | 72482816 |
| eng_None_32768_state_mb | 0.0 | 51.511296 |
| eng_None_32768_total_mb | 21954.286067 | 22005.797363 |
| eng_None_4096_state_mb | 0.0 | 51.511296 |
| eng_None_4096_total_mb | 20652.374106 | 20703.885402 |
| eng_llamacpp_32768_state_mb | 0.0 | 51.511296 |
| eng_llamacpp_32768_total_mb | 21954.286067 | 22005.797363 |
| eng_llamacpp_4096_state_mb | 0.0 | 51.511296 |
| eng_llamacpp_4096_total_mb | 20652.374106 | 20703.885402 |
| genz_chunked_kv_mb | 576.0 | 242.25 |
| genz_decode_kv_mb | 512.0 | 177.125 |
| genz_decode_weights_mb | 13716.0 | 14881.6875 |
| genz_kv_size | 65536 | 16384 |
| genz_layer_kinds | {"MHA-global": 32} | {"MHA-global": 8, "Recurrent": 24} |
| serving_bytes_per_token | 131072 | 32768 |
| serving_requests_8gib_1k | 64 | 100 |
| serving_state_bytes | 0 | 51511296 |

## config:models--LiquidAI--LFM2.5-Audio-1.5B-JP/snapshots/6c34b4d590f80563f8cb2939c2ebd7686d952394/audio_detokenizer/config.json (hybrid)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 5242880 | 1981440 |
| eng_None_32768_state_mb | 0.0 | 0.01536 |
| eng_None_32768_total_mb | 325.305754 | 325.321114 |
| eng_None_4096_state_mb | 0.0 | 0.01536 |
| eng_None_4096_total_mb | 244.775117 | 244.790477 |
| eng_llamacpp_32768_state_mb | 0.0 | 0.01536 |
| eng_llamacpp_32768_total_mb | 325.305754 | 325.321114 |
| eng_llamacpp_4096_state_mb | 0.0 | 0.01536 |
| eng_llamacpp_4096_total_mb | 244.775117 | 244.790477 |
| genz_chunked_kv_mb | 36.0 | 1.617188 |
| genz_decode_kv_mb | 20.087891 | 0.102539 |
| genz_decode_weights_mb | 128.0 | 130.514648 |
| genz_kv_size | 4096 | 1536 |
| genz_layer_kinds | {"MHA-global": 5, "MHA-local": 3} | {"MHA-local": 3, "Recurrent": 5} |
| serving_bytes_per_token | 8192 | 3072 |
| serving_requests_8gib_1k | 1024 | 2717 |
| serving_state_bytes | 0 | 15360 |

## config:models--LiquidAI--LFM2.5-Audio-1.5B/snapshots/c362a0625dfe45aa588dce5f0ada28a7e5707628/audio_detokenizer/config.json (hybrid)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 5242880 | 1981440 |
| eng_None_32768_state_mb | 0.0 | 0.01536 |
| eng_None_32768_total_mb | 325.305754 | 325.321114 |
| eng_None_4096_state_mb | 0.0 | 0.01536 |
| eng_None_4096_total_mb | 244.775117 | 244.790477 |
| eng_llamacpp_32768_state_mb | 0.0 | 0.01536 |
| eng_llamacpp_32768_total_mb | 325.305754 | 325.321114 |
| eng_llamacpp_4096_state_mb | 0.0 | 0.01536 |
| eng_llamacpp_4096_total_mb | 244.775117 | 244.790477 |
| genz_chunked_kv_mb | 36.0 | 1.617188 |
| genz_decode_kv_mb | 20.087891 | 0.102539 |
| genz_decode_weights_mb | 128.0 | 130.514648 |
| genz_kv_size | 4096 | 1536 |
| genz_layer_kinds | {"MHA-global": 5, "MHA-local": 3} | {"MHA-local": 3, "Recurrent": 5} |
| serving_bytes_per_token | 8192 | 3072 |
| serving_requests_8gib_1k | 1024 | 2717 |
| serving_state_bytes | 0 | 15360 |

## config:models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b/config.json (multimodal)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 83886080 | 72482816 |
| eng_None_32768_state_mb | 0.0 | 51.511296 |
| eng_None_32768_total_mb | 11478.217318 | 11529.728614 |
| eng_None_4096_state_mb | 0.0 | 51.511296 |
| eng_None_4096_total_mb | 10282.001818 | 10333.513114 |
| eng_llamacpp_32768_state_mb | 0.0 | 51.511296 |
| eng_llamacpp_32768_total_mb | 11478.217318 | 11529.728614 |
| eng_llamacpp_4096_state_mb | 0.0 | 51.511296 |
| eng_llamacpp_4096_total_mb | 10282.001818 | 10333.513114 |
| genz_chunked_kv_mb | 576.0 | 242.25 |
| genz_decode_kv_mb | 512.0 | 177.125 |
| genz_decode_weights_mb | 7132.5 | 7861.523438 |
| genz_kv_size | 65536 | 16384 |
| genz_layer_kinds | {"MHA-global": 32} | {"MHA-global": 8, "Recurrent": 24} |
| serving_bytes_per_token | 131072 | 32768 |
| serving_requests_8gib_1k | 64 | 100 |
| serving_state_bytes | 0 | 51511296 |

## config:models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json (multimodal)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 83886080 | 72482816 |
| eng_None_32768_state_mb | 0.0 | 51.511296 |
| eng_None_32768_total_mb | 11478.217318 | 11529.728614 |
| eng_None_4096_state_mb | 0.0 | 51.511296 |
| eng_None_4096_total_mb | 10282.001818 | 10333.513114 |
| eng_llamacpp_32768_state_mb | 0.0 | 51.511296 |
| eng_llamacpp_32768_total_mb | 11478.217318 | 11529.728614 |
| eng_llamacpp_4096_state_mb | 0.0 | 51.511296 |
| eng_llamacpp_4096_total_mb | 10282.001818 | 10333.513114 |
| genz_chunked_kv_mb | 576.0 | 242.25 |
| genz_decode_kv_mb | 512.0 | 177.125 |
| genz_decode_weights_mb | 7132.5 | 7861.523438 |
| genz_kv_size | 65536 | 16384 |
| genz_layer_kinds | {"MHA-global": 32} | {"MHA-global": 8, "Recurrent": 24} |
| serving_bytes_per_token | 131072 | 32768 |
| serving_requests_8gib_1k | 64 | 100 |
| serving_state_bytes | 0 | 51511296 |

## config:models--SupersonicLabs--Julia-1/snapshots/a85b127321d580d65176c89ced8273f305745d85/encoder/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| facts_full_attention | true | false |
| facts_layers | [22, 0, 0, null] | [8, 14, 0, null] |

## config:models--convaiinnovations--laya-multilingual/snapshots/e4e9ddf21a7b1903b7acffd8814ad4307bf63a67/encoder/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| facts_full_attention | true | false |
| facts_layers | [22, 0, 0, null] | [8, 14, 0, null] |

## config:models--convaiinnovations--laya-typed-decisions/snapshots/1a793eb568e6718f15941d08f85432581df534e3/encoder/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| facts_full_attention | true | false |
| facts_layers | [28, 0, 0, null] | [10, 18, 0, null] |

## config:models--convaiinnovations--laya/snapshots/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851/encoder/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| facts_full_attention | true | false |
| facts_layers | [28, 0, 0, null] | [10, 18, 0, null] |

## config:models--convaiinnovations--laya/snapshots/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851/typed-decisions/encoder/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| facts_full_attention | true | false |
| facts_layers | [28, 0, 0, null] | [10, 18, 0, null] |

## config:models--internlm--Intern-Decision-4B/snapshots/0e5e6aa7d6d750e2b1504ba11a8136cb58aeb3cd/config.json (multimodal)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 83886080 | 72482816 |
| eng_None_32768_state_mb | 0.0 | 51.511296 |
| eng_None_32768_total_mb | 11478.217318 | 11529.728614 |
| eng_None_4096_state_mb | 0.0 | 51.511296 |
| eng_None_4096_total_mb | 10282.001818 | 10333.513114 |
| eng_llamacpp_32768_state_mb | 0.0 | 51.511296 |
| eng_llamacpp_32768_total_mb | 11478.217318 | 11529.728614 |
| eng_llamacpp_4096_state_mb | 0.0 | 51.511296 |
| eng_llamacpp_4096_total_mb | 10282.001818 | 10333.513114 |
| genz_chunked_kv_mb | 576.0 | 242.25 |
| genz_decode_kv_mb | 512.0 | 177.125 |
| genz_decode_weights_mb | 7132.5 | 7861.523438 |
| genz_kv_size | 65536 | 16384 |
| genz_layer_kinds | {"MHA-global": 32} | {"MHA-global": 8, "Recurrent": 24} |
| serving_bytes_per_token | 131072 | 32768 |
| serving_requests_8gib_1k | 64 | 100 |
| serving_state_bytes | 0 | 51511296 |

## config:models--kyutai--mimi/snapshots/89091b3e466eb6a9d11e537bf26b144f194978f7/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| facts_full_attention | true | false |
| facts_layers | [8, 0, 0, null] | [0, 8, 0, 250] |

## config:models--mistralai--Voxtral-Mini-4B-Realtime-2602/snapshots/2769294da9567371363522aac9bbcfdd19447add/config.json (audio-llm)

| figure | before | after |
|---|---|---|
| facts_full_attention | true | false |
| facts_layers | [26, 0, 0, null] | [0, 26, 0, 8192] |

## config:models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| genz_chunked_kv_mb | 324.0 | 184.5 |

## config:models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| genz_chunked_kv_mb | 216.0 | 123.0 |

## db:deepseek-ai/DeepSeek-V3-Base (decoder-only)

| figure | before | after |
|---|---|---|
| genz_chunked_kv_mb | 7686.0 | 171.5625 |
| genz_decode_weights_mb | 1335494.375 | 1273239.875 |
| genz_ffn_kinds | {"MoE": 61} | {"Dense": 3, "MoE": 58} |
| serving_bytes_per_token | 1748992 | 70272 |
| serving_requests_8gib_1k | 4 | 119 |

## db:deepseek-ai/deepseek-moe-16b-base (decoder-only)

| figure | before | after |
|---|---|---|
| genz_decode_weights_mb | 30871.0 | 29943.0 |
| genz_ffn_kinds | {"MoE": 28} | {"Dense": 1, "MoE": 27} |

## db:tiiuae/falcon-mamba-7b (state-space)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 671088640 | 293601280 |
| eng_None_32768_state_mb | 0.0 | 293.60128 |
| eng_None_32768_total_mb | 109466.73705 | 109760.33833 |
| eng_None_4096_state_mb | 0.0 | 293.60128 |
| eng_None_4096_total_mb | 109184.879821 | 109478.481101 |
| eng_llamacpp_32768_state_mb | 0.0 | 293.60128 |
| eng_llamacpp_32768_total_mb | 109466.73705 | 109760.33833 |
| eng_llamacpp_4096_state_mb | 0.0 | 293.60128 |
| eng_llamacpp_4096_total_mb | 109184.879821 | 109478.481101 |
| genz_kv_size | 524288 | 0 |
| serving_bytes_per_token | 1048576 | 0 |
| serving_requests_8gib_1k | 8 | 29 |
| serving_state_bytes | 0 | 293601280 |

## fixture:EleutherAI_gpt-neo-1.3B.json (decoder-only)

| figure | before | after |
|---|---|---|
| eng_None_32768_kv_mb | 0.0 | 6442.450944 |
| eng_None_32768_params | 706906112 | 1310885888 |
| eng_None_32768_total_mb | 1695.669453 | 9346.079949 |
| eng_None_4096_kv_mb | 0.0 | 805.306368 |
| eng_None_4096_params | 706906112 | 1310885888 |
| eng_None_4096_total_mb | 1554.740838 | 3568.006758 |
| eng_llamacpp_32768_kv_mb | 0.0 | 6442.450944 |
| eng_llamacpp_32768_params | 706906112 | 1310885888 |
| eng_llamacpp_32768_total_mb | 1695.669453 | 9346.079949 |
| eng_llamacpp_4096_kv_mb | 0.0 | 805.306368 |
| eng_llamacpp_4096_params | 706906112 | 1310885888 |
| eng_llamacpp_4096_total_mb | 1554.740838 | 3568.006758 |
| facts_full_attention | true | false |
| facts_layers | [24, 0, 0, null] | [12, 12, 0, 256] |
| kv_full_layers | 0 | 12 |
| kv_gb@32768 | 0.0 | 6.442450944 |
| kv_gb@4096 | 0.0 | 0.805306368 |
| kv_sliding_layers | 24 | 12 |
| kv_source | "uniform" | "layer_plan:attention_types" |
| model_type | "hybrid" | "decoder-only" |
| report_kv_gb | 0.0 | 0.805306368 |
| report_params | 706906112 | 1310885888 |
| report_state_gb | 0.00172032 | 0.0 |
| report_total_gb | 1.584990945 | 3.757253755 |
| report_weight_gb | 1.413812224 | 2.621771776 |
| state_gb | 0.00172032 | 0.0 |

## fixture:LiquidAI_LFM2-2.6B.json (hybrid)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 39321600 | 10756096 |
| eng_None_32768_state_mb | 0.0 | 0.270336 |
| eng_None_32768_total_mb | 6219.245158 | 6219.515494 |
| eng_None_4096_state_mb | 0.0 | 0.270336 |
| eng_None_4096_total_mb | 5477.692211 | 5477.962547 |
| eng_llamacpp_32768_state_mb | 0.0 | 0.270336 |
| eng_llamacpp_32768_total_mb | 6219.245158 | 6219.515494 |
| eng_llamacpp_4096_state_mb | 0.0 | 0.270336 |
| eng_llamacpp_4096_total_mb | 5477.692211 | 5477.962547 |
| genz_chunked_kv_mb | 270.0 | 72.515625 |
| genz_decode_kv_mb | 240.0 | 64.257812 |
| genz_decode_weights_mb | 3376.0 | 3640.257812 |
| genz_kv_size | 30720 | 8192 |
| genz_layer_kinds | {"MHA-global": 30} | {"MHA-global": 8, "Recurrent": 22} |
| serving_bytes_per_token | 61440 | 16384 |
| serving_requests_8gib_1k | 136 | 503 |
| serving_state_bytes | 0 | 270336 |

## fixture:MiniMaxAI_MiniMax-Text-01.json (hybrid)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 209715200 | 319815680 |
| eng_None_32768_state_mb | 0.0 | 293.60128 |
| eng_None_32768_total_mb | 917067.869389 | 917361.470669 |
| eng_None_4096_state_mb | 0.0 | 293.60128 |
| eng_None_4096_total_mb | 915349.88247 | 915643.48375 |
| eng_llamacpp_32768_state_mb | 0.0 | 293.60128 |
| eng_llamacpp_32768_total_mb | 917067.869389 | 917361.470669 |
| eng_llamacpp_4096_state_mb | 0.0 | 293.60128 |
| eng_llamacpp_4096_total_mb | 915349.88247 | 915643.48375 |
| facts_full_attention | true | false |
| facts_layers | [80, 0, 0, null] | [10, 0, 70, null] |
| genz_chunked_kv_mb | 1440.0 | 740.0 |
| genz_decode_kv_mb | 1280.0 | 440.0 |
| genz_decode_weights_mb | 849094.5 | 867574.5 |
| genz_kv_size | 163840 | 20480 |
| genz_layer_kinds | {"MHA-global": 80} | {"MHA-global": 10, "Recurrent": 70} |
| serving_bytes_per_token | 327680 | 40960 |
| serving_state_bytes | 0 | 293601280 |

## fixture:PowerInfer_SmallThinker-21BA3B-Instruct.json (decoder-only)

| figure | before | after |
|---|---|---|
| facts_full_attention | true | false |
| facts_layers | [52, 0, 0, null] | [13, 39, 0, 4096] |
| genz_layer_kinds | {"MHA-global": 52} | {"MHA-global": 13, "MHA-local": 39} |

## fixture:Qwen_Qwen3-Next-80B-A3B-Instruct.json (hybrid)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 62914560 | 92995584 |
| eng_None_32768_state_mb | 0.0 | 77.266944 |
| eng_None_32768_total_mb | 160595.413299 | 160672.680243 |
| eng_None_4096_state_mb | 0.0 | 77.266944 |
| eng_None_4096_total_mb | 159729.708954 | 159806.975898 |
| eng_llamacpp_32768_state_mb | 0.0 | 77.266944 |
| eng_llamacpp_32768_total_mb | 160595.413299 | 160672.680243 |
| eng_llamacpp_4096_state_mb | 0.0 | 77.266944 |
| eng_llamacpp_4096_total_mb | 159729.708954 | 159806.975898 |
| genz_chunked_kv_mb | 432.0 | 255.375 |
| genz_decode_kv_mb | 384.0 | 169.6875 |
| genz_decode_weights_mb | 149873.5 | 150892.890625 |
| genz_kv_size | 49152 | 12288 |
| genz_layer_kinds | {"MHA-global": 48} | {"MHA-global": 12, "Recurrent": 36} |
| serving_bytes_per_token | 98304 | 24576 |
| serving_requests_8gib_1k | 85 | 83 |
| serving_state_bytes | 0 | 77266944 |

## fixture:Qwen_Qwen3.5-4B.json (multimodal)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 83886080 | 72482816 |
| eng_None_32768_state_mb | 0.0 | 51.511296 |
| eng_None_32768_total_mb | 11478.217318 | 11529.728614 |
| eng_None_4096_state_mb | 0.0 | 51.511296 |
| eng_None_4096_total_mb | 10282.001818 | 10333.513114 |
| eng_llamacpp_32768_state_mb | 0.0 | 51.511296 |
| eng_llamacpp_32768_total_mb | 11478.217318 | 11529.728614 |
| eng_llamacpp_4096_state_mb | 0.0 | 51.511296 |
| eng_llamacpp_4096_total_mb | 10282.001818 | 10333.513114 |
| genz_chunked_kv_mb | 576.0 | 242.25 |
| genz_decode_kv_mb | 512.0 | 177.125 |
| genz_decode_weights_mb | 7132.5 | 7861.523438 |
| genz_kv_size | 65536 | 16384 |
| genz_layer_kinds | {"MHA-global": 32} | {"MHA-global": 8, "Recurrent": 24} |
| serving_bytes_per_token | 131072 | 32768 |
| serving_requests_8gib_1k | 64 | 100 |
| serving_state_bytes | 0 | 51511296 |

## fixture:Qwen_Qwen3.6-27B.json (multimodal)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 167772160 | 195887104 |
| eng_None_32768_state_mb | 0.0 | 153.944064 |
| eng_None_32768_total_mb | 58973.596557 | 59127.540621 |
| eng_None_4096_state_mb | 0.0 | 153.944064 |
| eng_None_4096_total_mb | 56601.298214 | 56755.242278 |
| eng_llamacpp_32768_state_mb | 0.0 | 153.944064 |
| eng_llamacpp_32768_total_mb | 58973.596557 | 59127.540621 |
| eng_llamacpp_4096_state_mb | 0.0 | 153.944064 |
| eng_llamacpp_4096_total_mb | 56601.298214 | 56755.242278 |
| genz_chunked_kv_mb | 1152.0 | 581.625 |
| genz_decode_kv_mb | 1024.0 | 402.8125 |
| genz_decode_weights_mb | 44025.0 | 47914.21875 |
| genz_kv_size | 131072 | 32768 |
| genz_layer_kinds | {"MHA-global": 64} | {"MHA-global": 16, "Recurrent": 48} |
| serving_bytes_per_token | 262144 | 65536 |
| serving_requests_8gib_1k | 32 | 38 |
| serving_state_bytes | 0 | 153944064 |

## fixture:Zyphra_Zamba2-2.7B.json (hybrid)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 353894400 | 131461632 |
| eng_None_32768_state_mb | 0.0 | 72.479232 |
| eng_None_32768_total_mb | 8264.260416 | 8336.739648 |
| eng_None_4096_state_mb | 0.0 | 72.479232 |
| eng_None_4096_total_mb | 5445.688128 | 5518.16736 |
| eng_llamacpp_32768_state_mb | 0.0 | 72.479232 |
| eng_llamacpp_32768_total_mb | 8264.260416 | 8336.739648 |
| eng_llamacpp_4096_state_mb | 0.0 | 72.479232 |
| eng_llamacpp_4096_total_mb | 5445.688128 | 5518.16736 |
| genz_chunked_kv_mb | 2430.0 | 543.243164 |
| genz_decode_kv_mb | 35.859375 | 429.121582 |
| genz_decode_weights_mb | 9842.5 | 5614.521484 |
| genz_ffn_kinds | {"Dense": 54} | {"Dense": 9, "no_op": 45} |
| genz_kv_size | 276480 | 46080 |
| genz_layer_kinds | {"Mamba": 54} | {"MHA-global": 9, "Recurrent": 45} |
| serving_bytes_per_token | 552960 | 92160 |
| serving_requests_8gib_1k | 15 | 51 |
| serving_state_bytes | 0 | 72479232 |

## fixture:ai21labs_Jamba-v0.1.json (hybrid)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 83886080 | 26542080 |
| eng_None_32768_state_mb | 0.0 | 16.05632 |
| eng_None_32768_total_mb | 107731.409306 | 107747.465626 |
| eng_None_4096_state_mb | 0.0 | 16.05632 |
| eng_None_4096_total_mb | 106577.136845 | 106593.193165 |
| eng_llamacpp_32768_state_mb | 0.0 | 16.05632 |
| eng_llamacpp_32768_total_mb | 107731.409306 | 107747.465626 |
| eng_llamacpp_4096_state_mb | 0.0 | 16.05632 |
| eng_llamacpp_4096_total_mb | 106577.136845 | 106593.193165 |
| genz_chunked_kv_mb | 576.0 | 102.625 |
| genz_decode_kv_mb | null | 79.3125 |
| genz_decode_weights_mb | null | 97850.0625 |
| genz_error | "ValueError: More than 1 unique layers not supported. Work in progress" | null |
| genz_kv_size | 65536 | 8192 |
| genz_layer_kinds | {"Mamba": 32} | {"MHA-global": 4, "Recurrent": 28} |
| serving_bytes_per_token | 131072 | 16384 |
| serving_requests_8gib_1k | 64 | 261 |
| serving_state_bytes | 0 | 16056320 |

## fixture:deepseek-ai_DeepSeek-V3.2-Exp.json (decoder-only)

| figure | before | after |
|---|---|---|
| genz_chunked_kv_mb | 7686.0 | 171.5625 |
| genz_decode_weights_mb | 1335494.375 | 1273239.875 |
| genz_ffn_kinds | {"MoE": 61} | {"Dense": 3, "MoE": 58} |
| serving_bytes_per_token | 1748992 | 70272 |
| serving_requests_8gib_1k | 4 | 119 |

## fixture:deepseek_v2_lite_chat_config.json (decoder-only)

| figure | before | after |
|---|---|---|
| genz_chunked_kv_mb | 972.0 | 75.9375 |
| genz_decode_kv_mb | null | 67.5 |
| genz_decode_weights_mb | null | 28699.5 |
| genz_error | "ValueError: cannot convert float NaN to integer" | null |
| genz_ffn_kinds | {"MoE": 27} | {"Dense": 1, "MoE": 26} |
| serving_bytes_per_token | 221184 | 31104 |
| serving_requests_8gib_1k | 37 | 269 |

## fixture:ibm-ai-platform_Bamba-9B-v2.json (hybrid)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 83886080 | 130969088 |
| eng_None_32768_state_mb | 0.0 | 123.104768 |
| eng_None_32768_total_mb | 20768.962816 | 20892.067584 |
| eng_None_4096_state_mb | 0.0 | 123.104768 |
| eng_None_4096_total_mb | 20013.988096 | 20137.092864 |
| eng_llamacpp_32768_state_mb | 0.0 | 123.104768 |
| eng_llamacpp_32768_total_mb | 20768.962816 | 20892.067584 |
| eng_llamacpp_4096_state_mb | 0.0 | 123.104768 |
| eng_llamacpp_4096_total_mb | 20013.988096 | 20137.092864 |
| genz_chunked_kv_mb | 576.0 | 288.803711 |
| genz_decode_kv_mb | 66.0 | 165.401855 |
| genz_decode_weights_mb | 18282.0 | 17651.945312 |
| genz_kv_size | 65536 | 6144 |
| genz_layer_kinds | {"Mamba": 32} | {"MHA-global": 3, "Recurrent": 29} |
| serving_bytes_per_token | 131072 | 12288 |
| serving_requests_8gib_1k | 64 | 63 |
| serving_state_bytes | 0 | 123104768 |

## fixture:ibm-granite_granite-4.0-h-small.json (hybrid)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 104857600 | 163305472 |
| eng_None_32768_state_mb | 0.0 | 152.819712 |
| eng_None_32768_total_mb | 63985.84279 | 64138.662502 |
| eng_None_4096_state_mb | 0.0 | 152.819712 |
| eng_None_4096_total_mb | 63244.289843 | 63397.109555 |
| eng_llamacpp_32768_state_mb | 0.0 | 152.819712 |
| eng_llamacpp_32768_total_mb | 63985.84279 | 64138.662502 |
| eng_llamacpp_4096_state_mb | 0.0 | 152.819712 |
| eng_llamacpp_4096_total_mb | 63244.289843 | 63397.109555 |
| genz_chunked_kv_mb | 720.0 | 363.480469 |
| genz_decode_kv_mb | 82.5 | 209.740234 |
| genz_decode_weights_mb | 60806.5 | 59990.15625 |
| genz_kv_size | 81920 | 8192 |
| genz_layer_kinds | {"Mamba": 40} | {"MHA-global": 4, "Recurrent": 36} |
| serving_bytes_per_token | 163840 | 16384 |
| serving_requests_8gib_1k | 51 | 50 |
| serving_state_bytes | 0 | 152819712 |

## fixture:inclusionAI_Ling-flash-2.0.json (decoder-only)

| figure | before | after |
|---|---|---|
| genz_decode_weights_mb | 200204.0 | 194274.0 |
| genz_ffn_kinds | {"MoE": 32} | {"Dense": 1, "MoE": 31} |

## fixture:microsoft_Phi-4-mini-flash-reasoning.json (hybrid)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 104857600 | 58163200 |
| eng_None_32768_kv_mb | 83.88608 | 2684.35456 |
| eng_None_32768_params | 4329313280 | 4674606080 |
| eng_None_32768_state_mb | 0.0 | 5.7344 |
| eng_None_32768_total_mb | 9296.160768 | 12592.949248 |
| eng_None_4096_kv_mb | 83.88608 | 335.54432 |
| eng_None_4096_params | 4329313280 | 4674606080 |
| eng_None_4096_state_mb | 0.0 | 5.7344 |
| eng_None_4096_total_mb | 9019.336704 | 9967.314944 |
| eng_llamacpp_32768_kv_mb | 83.88608 | 2684.35456 |
| eng_llamacpp_32768_params | 4329313280 | 4674606080 |
| eng_llamacpp_32768_state_mb | 0.0 | 5.7344 |
| eng_llamacpp_32768_total_mb | 9296.160768 | 12592.949248 |
| eng_llamacpp_4096_kv_mb | 83.88608 | 335.54432 |
| eng_llamacpp_4096_params | 4329313280 | 4674606080 |
| eng_llamacpp_4096_state_mb | 0.0 | 5.7344 |
| eng_llamacpp_4096_total_mb | 9019.336704 | 9967.314944 |
| facts_layers | [0, 32, 0, 512] | [1, 15, 16, 512] |
| genz_chunked_kv_mb | 720.0 | 108.4375 |
| genz_decode_kv_mb | 640.0 | 62.96875 |
| genz_decode_weights_mb | 6976.875 | 7635.46875 |
| genz_kv_size | 81920 | 40960 |
| genz_layer_kinds | {"MHA-global": 32} | {"MHA-global": 1, "MHA-local": 15, "Recurrent": 16} |
| kv_free_layers | 0 | 16 |
| kv_full_layers | 0 | 1 |
| kv_gb@32768 | 0.08388608 | 2.68435456 |
| kv_gb@4096 | 0.08388608 | 0.33554432 |
| kv_sliding_layers | 32 | 15 |
| report_kv_gb | 0.08388608 | 0.33554432 |
| report_params | 4329313280 | 4674606080 |
| report_state_gb | 0.0 | 0.0057344 |
| report_total_gb | 9.091478733 | 10.090935501 |
| report_weight_gb | 8.65862656 | 9.34921216 |
| serving_bytes_per_token | 163840 | 81920 |
| serving_requests_8gib_1k | 51 | 95 |
| serving_state_bytes | 0 | 5734400 |
| state_gb | 0.0 | 0.0057344 |

## fixture:mistralai_Mamba-Codestral-7B-v0.1.json (state-space)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 335544320 | 272367616 |
| eng_None_32768_state_mb | 0.0 | 272.367616 |
| eng_None_32768_total_mb | 15136.10281 | 15408.470426 |
| eng_None_4096_state_mb | 0.0 | 272.367616 |
| eng_None_4096_total_mb | 14854.245581 | 15126.613197 |
| eng_llamacpp_32768_state_mb | 0.0 | 272.367616 |
| eng_llamacpp_32768_total_mb | 15136.10281 | 15408.470426 |
| eng_llamacpp_4096_state_mb | 0.0 | 272.367616 |
| eng_llamacpp_4096_total_mb | 14854.245581 | 15126.613197 |
| genz_kv_size | 262144 | 0 |
| serving_bytes_per_token | 524288 | 0 |
| serving_requests_8gib_1k | 16 | 31 |
| serving_state_bytes | 0 | 272367616 |

## fixture:moonshotai_Kimi-Linear-48B-A3B-Instruct.json (hybrid)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 19906560 | 48578560 |
| eng_None_32768_state_mb | 0.0 | 43.4176 |
| eng_None_32768_total_mb | 98400.864307 | 98444.281907 |
| eng_None_4096_state_mb | 0.0 | 43.4176 |
| eng_None_4096_total_mb | 97920.511642 | 97963.929242 |
| eng_llamacpp_32768_state_mb | 0.0 | 43.4176 |
| eng_llamacpp_32768_total_mb | 98400.864307 | 98444.281907 |
| eng_llamacpp_4096_state_mb | 0.0 | 43.4176 |
| eng_llamacpp_4096_total_mb | 97920.511642 | 97963.929242 |
| genz_chunked_kv_mb | 1093.5 | 102.5 |
| genz_decode_kv_mb | null | 58.90625 |
| genz_decode_weights_mb | null | 92620.068359 |
| genz_error | "ValueError: cannot convert float NaN to integer" | null |
| genz_ffn_kinds | {"MoE": 27} | {"Dense": 1, "MoE": 26} |
| genz_kv_size | 15552 | 4032 |
| genz_layer_kinds | {"MHA-global": 27} | {"MHA-global": 7, "Recurrent": 20} |
| serving_bytes_per_token | 248832 | 8064 |
| serving_requests_8gib_1k | 33 | 166 |
| serving_state_bytes | 0 | 43417600 |

## fixture:nvidia_Hymba-1.5B-Base.json (hybrid)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 26214400 | 33382400 |
| eng_None_32768_state_mb | 0.0 | 7.168 |
| eng_None_32768_total_mb | 4875.103206 | 4882.271206 |
| eng_None_4096_state_mb | 0.0 | 7.168 |
| eng_None_4096_total_mb | 3545.299123 | 3552.467123 |
| eng_llamacpp_32768_state_mb | 0.0 | 7.168 |
| eng_llamacpp_32768_total_mb | 4875.103206 | 4882.271206 |
| eng_llamacpp_4096_state_mb | 0.0 | 7.168 |
| eng_llamacpp_4096_total_mb | 3545.299123 | 3552.467123 |
| genz_chunked_kv_mb | 180.0 | 84.921875 |
| genz_decode_kv_mb | 3.90625 | 58.085938 |
| genz_decode_weights_mb | 2692.971802 | 3072.463989 |
| genz_layer_kinds | {"Mamba": 32} | {"MHA-global": 3, "MHA-local": 29} |
| serving_requests_8gib_1k | 204 | 174 |
| serving_state_bytes | 0 | 7168000 |

## fixture:nvidia_NVIDIA-Nemotron-Nano-9B-v2.json (hybrid)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 146800640 | 154034176 |
| eng_None_32768_state_mb | 0.0 | 143.548416 |
| eng_None_32768_total_mb | 20370.048768 | 20513.597184 |
| eng_None_4096_state_mb | 0.0 | 143.548416 |
| eng_None_4096_total_mb | 19459.8848 | 19603.433216 |
| eng_llamacpp_32768_state_mb | 0.0 | 143.548416 |
| eng_llamacpp_32768_total_mb | 20370.048768 | 20513.597184 |
| eng_llamacpp_4096_state_mb | 0.0 | 143.548416 |
| eng_llamacpp_4096_total_mb | 19459.8848 | 19603.433216 |
| genz_chunked_kv_mb | 1008.0 | 345.796875 |
| genz_decode_kv_mb | 63.164062 | 200.898438 |
| genz_decode_weights_mb | 22925.0 | 15832.672119 |
| genz_ffn_kinds | {"Dense": 56} | {"Dense": 25, "no_op": 31} |
| genz_kv_size | 114688 | 8192 |
| genz_layer_kinds | {"Mamba": 56} | {"MHA-global": 4, "Recurrent": 27, "no_op": 25} |
| serving_bytes_per_token | 229376 | 16384 |
| serving_requests_8gib_1k | 36 | 53 |
| serving_state_bytes | 0 | 143548416 |

## fixture:nvidia_Nemotron-H-8B-Base-8K.json (hybrid)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 136314880 | 112623616 |
| eng_None_32768_state_mb | 0.0 | 102.137856 |
| eng_None_32768_total_mb | 17894.134989 | 17996.272845 |
| eng_None_4096_state_mb | 0.0 | 102.137856 |
| eng_None_4096_total_mb | 16880.791142 | 16982.928998 |
| eng_llamacpp_32768_state_mb | 0.0 | 102.137856 |
| eng_llamacpp_32768_total_mb | 17894.134989 | 17996.272845 |
| eng_llamacpp_4096_state_mb | 0.0 | 102.137856 |
| eng_llamacpp_4096_total_mb | 16880.791142 | 16982.928998 |
| genz_chunked_kv_mb | 936.0 | 266.8125 |
| genz_decode_kv_mb | 107.25 | 161.40625 |
| genz_decode_weights_mb | 29104.0 | 14426.8125 |
| genz_ffn_kinds | {"Dense": 52} | {"Dense": 24, "no_op": 28} |
| genz_kv_size | 106496 | 8192 |
| genz_layer_kinds | {"Mamba": 52} | {"MHA-global": 4, "Recurrent": 24, "no_op": 24} |
| serving_bytes_per_token | 212992 | 16384 |
| serving_requests_8gib_1k | 39 | 72 |
| serving_state_bytes | 0 | 102137856 |

## fixture:openai_gpt-oss-120b.json (decoder-only)

| figure | before | after |
|---|---|---|
| genz_chunked_kv_mb | 324.0 | 184.5 |

## fixture:qwen3_8_27b_config.json (multimodal)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 167772160 | 195887104 |
| eng_None_32768_state_mb | 0.0 | 153.944064 |
| eng_None_32768_total_mb | 58973.596557 | 59127.540621 |
| eng_None_4096_state_mb | 0.0 | 153.944064 |
| eng_None_4096_total_mb | 56601.298214 | 56755.242278 |
| eng_llamacpp_32768_state_mb | 0.0 | 153.944064 |
| eng_llamacpp_32768_total_mb | 58973.596557 | 59127.540621 |
| eng_llamacpp_4096_state_mb | 0.0 | 153.944064 |
| eng_llamacpp_4096_total_mb | 56601.298214 | 56755.242278 |
| genz_chunked_kv_mb | 1152.0 | 581.625 |
| genz_decode_kv_mb | 1024.0 | 402.8125 |
| genz_decode_weights_mb | 44025.0 | 47914.21875 |
| genz_kv_size | 131072 | 32768 |
| genz_layer_kinds | {"MHA-global": 64} | {"MHA-global": 16, "Recurrent": 48} |
| serving_bytes_per_token | 262144 | 65536 |
| serving_requests_8gib_1k | 32 | 38 |
| serving_state_bytes | 0 | 153944064 |

## fixture:state-spaces_mamba-2.8b-hf.json (state-space)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 419430400 | 22937600 |
| eng_None_32768_state_mb | 0.0 | 22.9376 |
| eng_None_32768_total_mb | 5890.000896 | 5912.938496 |
| eng_None_4096_state_mb | 0.0 | 22.9376 |
| eng_None_4096_total_mb | 5713.840128 | 5736.777728 |
| eng_llamacpp_32768_state_mb | 0.0 | 22.9376 |
| eng_llamacpp_32768_total_mb | 5890.000896 | 5912.938496 |
| eng_llamacpp_4096_state_mb | 0.0 | 22.9376 |
| eng_llamacpp_4096_total_mb | 5713.840128 | 5736.777728 |
| genz_kv_size | 327680 | 0 |
| serving_bytes_per_token | 655360 | 0 |
| serving_requests_8gib_1k | 12 | 374 |
| serving_state_bytes | 0 | 22937600 |

## fixture:tiiuae_Falcon-H1-34B-Instruct.json (hybrid)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 94371840 | 398573568 |
| eng_None_32768_state_mb | 0.0 | 304.201728 |
| eng_None_32768_total_mb | 73265.91703 | 73570.118758 |
| eng_None_4096_state_mb | 0.0 | 304.201728 |
| eng_None_4096_total_mb | 68464.277811 | 68768.479539 |
| eng_llamacpp_32768_state_mb | 0.0 | 304.201728 |
| eng_llamacpp_32768_total_mb | 73265.91703 | 73570.118758 |
| eng_llamacpp_4096_state_mb | 0.0 | 304.201728 |
| eng_llamacpp_4096_total_mb | 68464.277811 | 68768.479539 |
| genz_chunked_kv_mb | 648.0 | 1228.21875 |
| genz_decode_kv_mb | 365.625 | 866.109375 |
| genz_decode_weights_mb | 71130.0 | 61616.71875 |
| genz_layer_kinds | {"Mamba": 72} | {"MHA-global": 72} |
| serving_requests_8gib_1k | 56 | 18 |
| serving_state_bytes | 0 | 304201728 |

## fixture:unsloth_gemma-3-27b-it.json (multimodal)

| figure | before | after |
|---|---|---|
| facts_full_attention | true | false |
| facts_layers | [62, 0, 0, null] | [10, 52, 0, 1024] |
| genz_chunked_kv_mb | 2232.0 | 984.0 |
| genz_decode_kv_mb | 1984.0 | 736.0 |
| genz_layer_kinds | {"MHA-global": 62} | {"MHA-global": 10, "MHA-local": 52} |

## genz:ai21labs/AI21-Jamba-1.5-Large (None)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 188743680 | 95846400 |
| genz_kv_size | 147456 | 18432 |
| genz_layer_kinds | {"Mamba": 72} | {"MHA-global": 9, "Mamba": 63} |
| serving_bytes_per_token | 294912 | 36864 |
| serving_requests_8gib_1k | 28 | 78 |
| serving_state_bytes | 0 | 72253440 |

## genz:ai21labs/AI21-Jamba-1.5-Mini (None)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 83886080 | 26542080 |
| genz_kv_size | 65536 | 8192 |
| genz_layer_kinds | {"Mamba": 32} | {"MHA-global": 4, "Mamba": 28} |
| serving_bytes_per_token | 131072 | 16384 |
| serving_requests_8gib_1k | 64 | 261 |
| serving_state_bytes | 0 | 16056320 |

## genz:ai21labs/Jamba-v0.1 (None)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 83886080 | 26542080 |
| genz_kv_size | 65536 | 8192 |
| genz_layer_kinds | {"Mamba": 32} | {"MHA-global": 4, "Mamba": 28} |
| serving_bytes_per_token | 131072 | 16384 |
| serving_requests_8gib_1k | 64 | 261 |
| serving_state_bytes | 0 | 16056320 |

## genz:deepseek-ai/DeepSeek-V2 (None)

| figure | before | after |
|---|---|---|
| genz_chunked_kv_mb | 5400.0 | 168.75 |
| serving_bytes_per_token | 1228800 | 69120 |
| serving_requests_8gib_1k | 6 | 121 |

## genz:deepseek-ai/DeepSeek-V3-Base (None)

| figure | before | after |
|---|---|---|
| genz_chunked_kv_mb | 120.09375 | 171.5625 |
| serving_bytes_per_token | 27328 | 70272 |
| serving_requests_8gib_1k | 306 | 119 |

## genz:google/gemma-3-27B (None)

| figure | before | after |
|---|---|---|
| genz_chunked_kv_mb | 2232.0 | 1008.0 |

## genz:google/gemma-3-4B (None)

| figure | before | after |
|---|---|---|
| genz_chunked_kv_mb | 382.5 | 172.5 |

## genz:nvidia/Llama-3_1-Nemotron-51B-Instruct (None)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 168755200 | 121569280 |
| genz_chunked_kv_mb | 1440.0 | 834.75 |
| genz_kv_size | 131840 | 94976 |
| serving_bytes_per_token | 327680 | 189952 |
| serving_requests_8gib_1k | 25 | 44 |

## genz:state-spaces/mamba-130m-hf (None)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 47185920 | 2580480 |
| genz_kv_size | 36864 | 0 |
| serving_bytes_per_token | 73728 | 0 |
| serving_requests_8gib_1k | 113 | 3328 |
| serving_state_bytes | 0 | 2580480 |

## genz:state-spaces/mamba-2.8b-hf (None)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 419430400 | 22937600 |
| genz_kv_size | 327680 | 0 |
| serving_bytes_per_token | 655360 | 0 |
| serving_requests_8gib_1k | 12 | 374 |
| serving_state_bytes | 0 | 22937600 |

## genz:tiiuae/falcon-mamba-7b (None)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 671088640 | 36700160 |
| genz_kv_size | 524288 | 0 |
| serving_bytes_per_token | 1048576 | 0 |
| serving_requests_8gib_1k | 8 | 234 |
| serving_state_bytes | 0 | 36700160 |

## gguf:Qwen3.5-2B-Q4_K_M.gguf (hybrid)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 31457280 | 27402240 |
| eng_None_32768_kv_mb | 1610.612736 | 402.653184 |
| eng_None_32768_params | 1901268992 | 1855758390 |
| eng_None_32768_state_mb | 0.0 | 19.53792 |
| eng_None_32768_total_mb | 5775.538586 | 4496.09575 |
| eng_None_4096_kv_mb | 201.326592 | 50.331648 |
| eng_None_4096_params | 1901268992 | 1855758390 |
| eng_None_4096_state_mb | 0.0 | 19.53792 |
| eng_None_4096_total_mb | 4185.058509 | 3962.580281 |
| eng_llamacpp_32768_kv_mb | 1610.612736 | 402.653184 |
| eng_llamacpp_32768_params | 1901268992 | 1855758390 |
| eng_llamacpp_32768_state_mb | 0.0 | 19.53792 |
| eng_llamacpp_32768_total_mb | 5775.538586 | 4496.09575 |
| eng_llamacpp_4096_kv_mb | 201.326592 | 50.331648 |
| eng_llamacpp_4096_params | 1901268992 | 1855758390 |
| eng_llamacpp_4096_state_mb | 0.0 | 19.53792 |
| eng_llamacpp_4096_total_mb | 4185.058509 | 3962.580281 |
| facts_full_attention | true | false |
| facts_layers | [24, 0, 0, null] | [6, 0, 18, null] |
| genz_chunked_kv_mb | 216.0 | 91.265625 |
| genz_decode_kv_mb | 192.0 | 66.632812 |
| genz_decode_weights_mb | 2602.0 | 3539.265625 |
| genz_kv_size | 24576 | 6144 |
| genz_layer_kinds | {"MHA-global": 24} | {"MHA-global": 6, "Recurrent": 18} |
| gguf_config_keys | ["head_dim", "hidden_size", "intermediate_size", "max_position_embeddings", "model_type", "num_attention_heads", "num_hidden_layers", "num_key_value_heads", "vocab_size"] | ["head_dim", "hidden_act", "hidden_size", "intermediate_size", "layer_types", "max_position_embeddings", "model_type", "num_attention_heads", "num_hidden_layers", "num_key_value_heads", "rope_theta", "ssm_conv_kernel", "ssm_inner_size", "ssm_n_groups", "ssm_state_size", "tie_word_embeddings", "time_step_rank", "vocab_size"] |
| kv_free_layers | 0 | 18 |
| kv_full_layers | 24 | 6 |
| kv_gb@32768 | 1.610612736 | 0.402653184 |
| kv_gb@4096 | 0.201326592 | 0.050331648 |
| kv_source | "uniform" | "layer_plan:layer_types" |
| model_type | "decoder-only" | "hybrid" |
| report_kv_gb | 0.201326592 | 0.050331648 |
| report_params | 1901268992 | 1855758390 |
| report_state_gb | 0.0 | 0.01953792 |
| report_total_gb | 4.261562614 | 4.012792981 |
| report_weight_gb | 3.802537984 | 3.71151678 |
| serving_bytes_per_token | 49152 | 12288 |
| serving_requests_8gib_1k | 170 | 267 |
| serving_state_bytes | 0 | 19537920 |
| state_gb | 0.0 | 0.01953792 |

## gguf:SmolVLM-500M-Instruct-Q8_0.gguf (decoder-only)

| figure | before | after |
|---|---|---|
| eng_None_32768_params | 369808320 | 409252800 |
| eng_None_32768_total_mb | 2239.08032 | 2317.96928 |
| eng_None_4096_params | 369808320 | 409252800 |
| eng_None_4096_total_mb | 986.032 | 1064.92096 |
| eng_llamacpp_32768_params | 369808320 | 409252800 |
| eng_llamacpp_32768_total_mb | 2239.08032 | 2317.96928 |
| eng_llamacpp_4096_params | 369808320 | 409252800 |
| eng_llamacpp_4096_total_mb | 986.032 | 1064.92096 |
| gguf_config_keys | ["head_dim", "hidden_size", "intermediate_size", "max_position_embeddings", "model_type", "num_attention_heads", "num_hidden_layers", "num_key_value_heads", "vocab_size"] | ["head_dim", "hidden_act", "hidden_size", "intermediate_size", "max_position_embeddings", "model_type", "num_attention_heads", "num_hidden_layers", "num_key_value_heads", "rope_theta", "tie_word_embeddings", "vocab_size"] |
| report_params | 369808320 | 409252800 |
| report_total_gb | 1.035315072 | 1.114204032 |
| report_weight_gb | 0.73961664 | 0.8185056 |

## gguf:gpt-oss-20b-MXFP4.gguf (decoder-only)

| figure | before | after |
|---|---|---|
| eng_None_32768_params | 1992044160 | 20908068480 |
| eng_None_32768_total_mb | 5945.764301 | 44049.60384 |
| eng_None_4096_params | 1992044160 | 20908068480 |
| eng_None_4096_total_mb | 4360.946534 | 42328.890624 |
| eng_llamacpp_32768_params | 1992044160 | 20908068480 |
| eng_llamacpp_32768_total_mb | 5945.764301 | 44049.60384 |
| eng_llamacpp_4096_params | 1992044160 | 20908068480 |
| eng_llamacpp_4096_total_mb | 4360.946534 | 42328.890624 |
| genz_decode_weights_mb | 3078.984375 | 38773.828125 |
| genz_ffn_kinds | {"Dense": 24} | {"MoE": 24} |
| gguf_config_keys | ["head_dim", "hidden_size", "intermediate_size", "max_position_embeddings", "model_type", "num_attention_heads", "num_hidden_layers", "num_key_value_heads", "vocab_size"] | ["head_dim", "hidden_act", "hidden_size", "intermediate_size", "max_position_embeddings", "model_type", "moe_intermediate_size", "n_routed_experts", "num_attention_heads", "num_experts", "num_experts_per_tok", "num_hidden_layers", "num_key_value_heads", "rope_theta", "tie_word_embeddings", "vocab_size"] |
| report_params | 1992044160 | 20908068480 |
| report_total_gb | 4.436318177 | 42.431441357 |
| report_weight_gb | 3.98408832 | 41.81613696 |

## gguf:nomic-embed-text-v1.5.Q8_0.gguf (decoder-only)

| figure | before | after |
|---|---|---|
| eng_None_32768_params | 109986816 | 136725504 |
| eng_None_32768_total_mb | 1594.027622 | 1647.504998 |
| eng_None_4096_params | 109986816 | 136725504 |
| eng_None_4096_total_mb | 454.015795 | 507.493171 |
| eng_llamacpp_32768_params | 109986816 | 136725504 |
| eng_llamacpp_32768_total_mb | 1594.027622 | 1647.504998 |
| eng_llamacpp_4096_params | 109986816 | 136725504 |
| eng_llamacpp_4096_total_mb | 454.015795 | 507.493171 |
| genz_decode_weights_mb | 206.709961 | 260.709961 |
| gguf_config_keys | ["hidden_size", "intermediate_size", "max_position_embeddings", "model_type", "num_attention_heads", "num_hidden_layers", "num_key_value_heads", "vocab_size"] | ["hidden_act", "hidden_size", "intermediate_size", "max_position_embeddings", "model_type", "num_attention_heads", "num_hidden_layers", "num_key_value_heads", "rope_theta", "tie_word_embeddings", "vocab_size"] |
| report_params | 109986816 | 136725504 |
| report_total_gb | 0.500824228 | 0.554301604 |
| report_weight_gb | 0.219973632 | 0.273451008 |

