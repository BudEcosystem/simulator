# Proof: KV layer kinds -- before and after

Branch `fix/hybrid-kv-layer-kinds`, measured against **origin/main `c17ef89`** (what this branch
lands on). Generated on the shared GB10 box, 27-28 Oct 2026, with `scripts/kv_corpus.py`; no GPU
used. Six rounds: the fix itself; the corrections an independent review asked for (the
chunked graph, the state's bytes, MoE placement, unplaced windows, depth from a layer list,
SambaY, the GGUF reader's Mamba-1 / RoPE / gating / tying, and this proof's own coverage); and the
second review's (Zamba's shared MLP and its unknown names, transformers before 4.53, the FRD-027
facts' allowlist and `mistral` rule, RWKV / xLSTM state bytes in the graph, Llama 4's chunk in
the graph, optional test dependencies); and the third review's (the serving path's missing
recurrent state, GPT-Neo's layout under `attention_types`, per-layer lists shorter than the
depth, Kimi's one-indexing without `kda_layers`, and the two window policies stated); and the
fourth review's (the attention head width read under its other spellings -- Zamba2's KV was half
its size everywhere --, an SSM-state config with attention heads and no layout, a family name
overriding stated attention heads, and `max_window_layers` where a config class states no layout);
and the fifth review's (a dense layer of an MoE model graphed at the experts' width, sliding layers
whose window the table does not read, and the prefix cache's price of a token); and the sixth
review's (the vLLM planner's concurrency on a stack of full and sliding layers, GenZ's depth for a
config that states it only as a per-layer list, a GGUF's 0-KV layers taken for FFN-only slots,
the database converter's window, and the README's statement of the window); and the seventh
review's (an encoder-decoder's decoder read under its own spellings by every reader, the
converter's recurrent state and MLA latent, a GGUF window pattern shorter than the stack, and
this proof's headline); and the eighth review's (an encoder-decoder's cross-attention cache
charged by every reader of a request's KV, a whole encoder-decoder nested under `text_config`,
ProphetNet's decoder heads, the encoder in the GenZ prefill graph, and this proof's coverage:
every reader of the layer kinds, every fixture folder, synthetic edge shapes).

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
* Round 5 (32 cases added for the fourth review) against the round-4 head `9942738`: 9 failed,
  16 passed, 7 skipped. The passing 16 are guards: 15 fixtures of the new per-fixture head-width
  cross-check that were already right (the 7 skips are fixtures whose model type the installed
  transformers cannot build, so there is no class to check against), and the Mistral case below.
  - test_every_kv_reader_sizes_a_layer_by_the_configs_own_attention_head[Zyphra_Zamba2-2.7B]
    (builds each fixture's own transformers config class and checks the calculator, the breakdown,
    the facts, GenZ `get_kv_size` and the serving `MemoryModel` against its head width: Zamba2's
    calculator gave 92,160 B a token against 184,320)
  - test_zamba2_kv_is_its_nine_shared_attention_invocations_at_head_dim_160 (92,160; 51
    requests in 8 GiB, not 32)
  - test_zamba_v1_saved_by_its_config_class_is_sized_at_its_attention_width (193,024 against
    386,048, for `ZambaConfig().to_dict()` and for the same config with the derived keys left out)
  - test_the_head_width_spellings_cover_every_transformers_alias (no such table)
  - test_zamba2_attention_weights_take_the_attention_width (26,214,400 per block against
    91,750,400: Linear(5120, 5120) x 3 + Linear(5120, 2560))
  - test_an_unknown_hybrid_without_a_layout_is_charged_attention_and_state_everywhere (calculator
    24 full layers and no state; GenZ pure Mamba; serving 0 B a token)
  - test_genz_graphs_no_mamba_layer_the_table_does_not_name (a state key made every layer Mamba)
  - test_a_family_name_does_not_make_a_config_with_attention_heads_attention_free (`mamba` with
    16 heads: 0 KV)
  - test_max_window_layers_places_the_window_when_the_class_states_no_layout (4.53's
    Qwen2MoeConfig, simulated: 24 sliding layers, not 21 full + 3 sliding)
  - guard: test_a_uniform_window_without_max_window_layers_still_slides_every_layer (Mistral)
* Round 6 (7 cases added for the fifth review) against the round-5 head `18d38ae`: all 7 failed.
  - test_a_dense_layer_of_an_moe_model_is_graphed_at_the_width_its_config_names x 3
    (Llama 4 `intermediate_size_mlp` with `interleave_moe_layer_step` 2, MiniMax-M3
    `dense_intermediate_size`, Cohere2-MoE `prefix_dense_intermediate_size`: every dense layer
    graphed 256 wide, not 1,024)
  - test_sliding_layers_whose_window_is_not_read_are_charged_in_full_everywhere (no warning; the
    table kept 16 sliding layers with no window, which the facts reported and the vLLM planner
    charged nothing)
  - test_the_prefix_cache_prices_a_token_like_the_serving_memory_model x 3 (Qwen3.5-4B 131,072
    against 32,768 B; Nemotron-H-8B 212,992 against 16,384; DeepSeek-V3.2 1,748,992 against
    MLA's 70,272)
* Round 7 (23 cases added for the sixth review) against the round-6 head `8d74c2b`: 22 failed,
  one is a guard.
  - test_the_vllm_planner_admits_no_more_sequences_than_every_kv_group_holds x 8 (Gemma-3-27b and
    Gemma-2-9B's hub config, vllm-0.30-upstream and bud-vllm-0.10.0, 1,024 and 4,096 input tokens:
    160 running against the allocator's 27, 246 against 123, ...)
  - test_the_profile_scheduler_charges_every_sliding_group_too (the step scheduler ran more
    sequences than the allocator holds at their prompt's length)
  - test_a_shared_slice_holds_the_load_in_every_kv_group (a slice for 16 x 4,352 tokens held 8)
  - test_genz_and_serving_take_the_depth_from_the_per_layer_list x 2 (Nemotron-H-8B saved by
    transformers 5.12: GenZ 32 layers, serving 0 B a token; with a nested block, a ValueError)
  - test_genz_never_graphs_attention_layers_as_mamba (no such function: the decision was inline)
  - test_a_gguf_layer_without_kv_or_a_mixer_keeps_the_stack_s_ffns x 3 and
    test_genz_graphs_every_ffn_of_a_deci_gguf (Deci: 2 of 8 FFNs graphed)
  - guard: test_a_gguf_whose_kv_layers_run_no_ffn_still_reads_ffn_only_slots (Nemotron-H's GGUF)
  - BudSimulator `tests/test_model_converter_layer_kinds.py` x 5 (SmallThinker's
    `sliding_window_size`, Llama 4's `attention_chunk_size` flat and under `text_config`, Gemma 3
    under `text_config`: window None; the saved Nemotron-H: a ValueError)
* Round 8 (83 cases added for the seventh review) against the round-7 head `de8f1be`:
  - `tests/test_encoder_decoder_stack.py` x 47 (nine encoder-decoder shapes from
    `tests/fixtures/encoder_decoder`: Dia, Whisper large-v3-turbo, Moonshine and an asymmetric
    Moonshine, cohere_asr, flan-t5-xl, an asymmetric T5, a T5 saved with `architectures: null`,
    BART; each through `encoder_decoder_stacks`, GenZ, `calculate_kv_cache` / `kv_cache_breakdown`,
    the facts and `calculate_memory`'s decoder getters, plus a decoder-only guard over every hybrid
    fixture and the view's stability): the file does not import on `de8f1be` (the reader does
    not exist). Measured there by hand: `calculate_kv_cache` 0 for BART, Dia and Whisper, Whisper's
    breakdown 32 layers; GenZ a ValueError for T5, BART and Whisper; `calculate_memory`'s decoder
    4 layers of 20 heads for T5 and Dia; the facts 16 heads for flan-t5-xl's 32.
  - test_a_gguf_window_pattern_shorter_than_the_stack_leaves_the_rest_full x 2: the per-layer KV
    array case failed (IndexError); the scalar case is a guard.
  - BudSimulator `tests/test_model_converter_layer_kinds.py` x 34: 11 failed, 23 guards.
    test_a_database_model_holds_the_kv_and_state_its_hf_config_holds over the 26 hybrid fixtures
    failed on Mamba-Codestral (state 0.29% low), DeepSeek-V3.2 and Kimi Linear (no MLA latent);
    test_a_database_encoder_decoder_is_graphed_with_its_decoder_s_own_depth failed on all 8 (Dia
    9 decoder layers, flan-t5-xl 12, ...; BART and Whisper: the loader's ValueError).
* Round 9 (cases added for the eighth review) against the round-8 head `2b26ea0` (a `git archive`
  of it with this round's test files copied in):
  - `tests/test_encoder_decoder_cross_attention.py` x 49: 48 failed, 1 guard passed
    (`test_the_prefix_cache_prices_a_cached_token_by_the_self_attention`: a cached prefix token
    holds the self-attention only, which round 8 already priced). Of the 48, the codec guard
    (`test_two_nested_stacks_that_never_meet_hold_no_cross_cache`, added after the first fix
    over-charged Qwen3-TTS's speech tokenizer, below) fails there only because
    `cross_attention_cache` does not exist. Eleven encoder-decoder shapes (Whisper-turbo, Dia,
    cohere_asr, flan-t5-xl, BART, Moonshine, BLIP-2 and InstructBLIP with flan-T5, Florence-2,
    ProphetNet, a TrOCR VisionEncoderDecoder) through the table, `calculate_kv_cache`, GenZ's
    `get_kv_size` / `get_cross_kv_bytes`, the serving admission (Whisper-turbo 1,024 tokens:
    20,971,520 B, truth 51,691,520), the transfer, the facts, the vLLM planner (Dia: 1,024
    running), `KVGeometry` (32 KV layers for Whisper-turbo), `calculate_memory`'s cross width
    (Dia: half), the GenZ decode graph's KV (80 MiB, truth 109.3) and the prefill graph's weights
    (Whisper-turbo 403 MiB of 1,542). On the round-8 head ProphetNet's `calculate_kv_cache` is 0
    and BLIP-2's 0, GenZ graphs BLIP-2's Q-Former (12 layers, 18,432 elements a token).
  - `tests/test_encoder_decoder_stack.py`: three assertions widened to self + cross (the GenZ
    `get_kv_size` where the encoder's length is unstated, `calculate_kv_cache`, the breakdown's
    `num_cross_layers`); 15 of its cases fail on the round-8 head for that reason.
  - BudSimulator `tests/test_model_converter_layer_kinds.py`: three nested cases (BLIP-2,
    InstructBLIP, Florence-2) and a cross-cache case for all 11 shapes added, the KV assertion
    widened to self + cross: 53 cases, 19 failed on the round-8 head (BLIP-2 / InstructBLIP: 12
    decoder layers of 768, `get_kv_size` 18,432; Florence-2: 32 default layers; no fixed bytes
    anywhere), 34 passed.
* This branch: 157 cases, 150 passed and 7 skipped (the GGUF-file cases skip where `gguf` is absent; the cross-check
  skips the 7 fixtures transformers cannot build). The category file also runs in a venv with
  neither `transformers` nor `gguf` (149 passed, 33 skipped across it and
  `tests/test_hybrid_layer_plan.py`), with transformers 4.50.0 (159 passed, 23 skipped) and with
  4.53.0 (119 passed, 17 skipped; that venv lacks GenZ's `seaborn`, and its 46 failures are all
  that import). By hand under 4.53.0: the reviewer's `qwen2moe_swa_on` config (24 layers,
  `max_window_layers` 21) now reads 21 full + 3 sliding, `src = max_window_layers`, matching
  4.53's `modeling_qwen2_moe.py`.
* Two older SambaY tests in `tests/test_hybrid_layer_plan.py` pinned the all-windowed reading
  ("every layer sliding", an "OVER-estimate"). That reading under-counts at long context, so they
  now pin the upper bound -- one full layer, the window elsewhere, never below the published
  design -- and keep their original guard: the window is never discarded (the 195x blow-up).

## 3. The whole suites

| Suite | origin/main `c17ef89` | This branch |
|---|---|---|
| `llm-memory-calculator/tests` (pytest) | 2,091 passed, 14 skipped, 1 failed | 2,339 passed, 21 skipped, 1 failed |
| `BudSimulator/tests` (pytest) | 24 passed | 79 passed |

(Round 9 adds the 49 cases of `tests/test_encoder_decoder_cross_attention.py` and 14 BudSimulator
converter cases (three nested encoder-decoders, and the cross cache of all eleven shapes); round 8
ended at 2,290 passed, 21 skipped / 65. Round 8 adds the 49 calculator cases and the 34 BudSimulator converter cases above; round 7
ended at 2,241 passed, 21 skipped / 31. Round 3 ended at 2,151 / 24; round 4 at 2,191 passed, 14 skipped / 26; round 5 adds the 32
category cases above, 7 of which skip -- fixtures transformers cannot build. Round 6 adds the 7 category cases for the fifth review; round 7 the 18 category cases and the 5
BudSimulator converter cases for the sixth (no existing test changed: the planner's gpt-oss
calibrations hold with each sequence's sliding blocks counted). Round 4 added the 40
category cases and the two memory-tier API cases. One existing assertion was widened: `tests/serving/test_memory_model.py::
TestMemorySnapshot::test_snapshot_has_all_fields` pins the snapshot's keys, and the snapshot now
reports `state_bytes` beside `kv_cache_bytes`.)

The one failure is the same test on both sides and predates this work:
`tests/planner/test_runtime_spec.py::test_committed_schemas_are_current` compares the committed
FRD-027 JSON schemas with ones regenerated by the installed pydantic, which differ on this
machine's pydantic (2.13).

## 4. Corpus: every model config the simulator ships or this machine caches

257 configs measured (round 9): the 26 rows of `BudSimulator/data/prepopulated.db` that carry a
config; the 55 model configs under `tests/fixtures`, read recursively (the 26 hybrid configs, the
3 at its root, from round 9 the 6 encoder-decoders of `encoder_decoder/`, the 18 of
`planner/models/` and the 2 of `measured_checkpoints/`; the planner's engine records beside them
are not model configs and are skipped); the 48 configs in the HuggingFace cache and model folders
that name a decoder's or an encoder-decoder's geometry (from round 9 also those that state the
decoder only under its own spelling or in a nested block: Dia, Qwen3-TTS's two speech
tokenizers, VibeVoice); 5 GGUFs read through `bud_component.gguf_to_hf_config` (beside each
file's own tensor count); the 89 built-in GenZ models; and, from round 9, 34 synthetic shapes
(`kv_corpus.synthetic_configs`: depth stated only in a per-layer list, flat and nested; nested
decoders -- VisionEncoderDecoder with TrOCR and with GPT-2, EncoderDecoderModel, MusicGen,
BLIP-2 / InstructBLIP with flan-T5, Florence-2; ProphetNet, T5 / mT5 / Marian / Whisper / NLLB-MoE
spellings; RWKV, xLSTM; Llama 4 Scout and Maverick with interleaved MoE; GGUF-derived window
patterns, short and full; GPT-2, MPT, ChatGLM, Falcon, a dense Llama, Mistral's window, Gemma 2's
hub config, an unknown layer name, an audio-LLM, DeepSeek-V2-Lite's MLA). The database's other
**63 rows have `config_json = '{}'`** and are listed in the JSON (`_empty_db_rows`), not
measured: they would only measure the calculator's defaults. Every tree was measured with the same
script, the same committed database and the same fixtures.

Figures per config: the calculator's KV at 4,096 and 32,768 tokens, its layer breakdown (with,
from round 9, its cross-attention layers), the recurrent state, `calculate_memory`'s weights /
params / total; the engine estimate (`estimate_memory_for_engine`, llama.cpp and generic, 4k and
32k: KV, state, total, params); the FRD-027 `ModelFacts` layer counts; GenZ's attention and FFN
kinds, `get_kv_size` (the disaggregation KV transfer is this x 2 B per token), the decode graph's
KV and weights, the chunked graph's KV; and the serving `MemoryModel`'s bytes per token, its
per-request fixed bytes (state, and from round 9 the cross-attention cache), the 1,024-token
requests that fit 8 GiB, and the disaggregation transfer for 512 + 128 tokens (KV x 2 B, plus the
fixed bytes where the tree counts them). Round 4 added the last three figures. Round 8 added
BudSimulator's converter (`ModelConverter.hf_to_genz`, the database path; `--repo-root` names the
tree): its encoder and decoder depths, `get_kv_size` and fixed bytes. Round 9 added every other
reader of the layer kinds: the GenZ prefill graph's weights and KV; the prefix-cache analyser's
bytes a cached token; the KV planner's `KVGeometry` (bytes a token, KV layers, recurrent layers,
state a layer, block, cross bytes a sequence) at 4k and 32k; the vLLM planner (`engine/vllm.py`,
`vllm-0.30-upstream` on `h100xm-80c-vgpu`: `limits` -- blocks, KV tokens, sequence ceiling,
attention and sliding groups, fixed bytes a sequence, boots -- and the sequences `predict` runs for
1,024 clients at 1k and 8k prompts + 256); its profile scheduler (`perf.predict.sim_config` +
`perf.sched.simulate`, 400 requests of 1,024 + 256 at a flat 10 ms step: the most running); and
its shared-slice sizing (`runtime.modules.capacity.shared_slice_mib`, 16 x 4,352 tokens).
origin/main, the round-8 head and this branch were all re-measured with round 9's script.

Reproduce (the first `PYTHONPATH` entry is the tree measured):

    PYTHONPATH=<tree>/llm-memory-calculator/src python llm-memory-calculator/scripts/kv_corpus.py \
        --out after.json --db BudSimulator/data/prepopulated.db \
        --fixtures llm-memory-calculator/tests/fixtures \
        --hf-cache ~/.cache/huggingface/hub --models /home/bud/models --repo-root <tree> --genz-data <scratch>
    python llm-memory-calculator/scripts/kv_corpus.py --diff before.json after.json --report diff.md

**Result: 258 rows -- 257 configs and the empty-row marker -- 144 unchanged, 114 changed** against
origin/main, over every figure. Of the 223 shipped, cached and built-in configs, 86 changed and 137
did not; of the 34 synthetic shapes, 28 changed and 6 did not (a dense Llama with and without
`is_encoder_decoder: false`, Mistral's window, Falcon-40B, an audio-LLM, and GPT-2: GenZ reads
no GPT-2 spelling on either tree). Every config whose figures moved moved in a figure measured
before round 9 too: no change is confined to the planners' and the prefill graph's new figures
(the prefix bytes, `KVGeometry`, the vLLM planner, its scheduler and slice sizing follow the KV,
state and cross cache already in the table). Against the round-8 head, 31 changed and 227 did not:
the encoder-decoders, their nested and synthetic shapes, and three `KVGeometry` depths (round 9,
below). No uniform dense decoder-only transformer's KV moved in any output. No config that
measured before fails now (on either base); 14 configs that fail somewhere on origin/main now
measure everywhere (Jamba, Kimi Linear, DeepSeek-V2-Lite, GPT-Neo, Whisper large-v3-turbo x2,
BART, T5 in GenZ; the synthetic list-depth Nemotron-H x2 and Qwen3.5 and DeepSeek-V2-Lite; from
round 9 Florence-2 and the TrOCR shape, which fail on the round-8 head too). Every change, by
cause (rounds 1-7; rounds 8 and 9 below):

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
  table reads, so the sliding layers keep the full context -- an over-estimate). Round 6 puts them
  back to [28 full]: sliding layers with no window the table reads are full layers in the table
  itself, because the vLLM planner charged a sliding group with no window nothing.
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
  | Zamba2-2.7B | 15 | 91 | 51 (round 5: **32**) | 72.5 MB |
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
  dropped that KV and added nothing. Zamba2's round-4 figure, 51, inherited a KV per token half
  its true size; round 5 below corrects it to 32 = 8 GiB / (1,024 x 184,320 B + 72.5 MB).) The disaggregation transfer for 512 + 128 tokens moves the
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

Round 5 (the fourth review) moved exactly one config against the round-4 head (`--diff` of the
round-4 head's corpus against this branch's: 193 unchanged, 1 changed):

* **Zamba2-2.7B's KV is sized at its own attention head, 160 wide** (`attention_head_dim`; its
  hub config also states Megatron's `kv_channels: 80`, and hidden_size / heads is 80 too).
  transformers 5.12's `Zamba2Attention` builds k_proj / v_proj as Linear(5120, 32 x 160), so the
  truth is 9 hybrid layers x 2 x 32 x 160 x 2 B = 184,320 B a token. origin/main -> round-4 head
  -> this branch:

  | figure | origin/main | round-4 head | this branch |
  |---|---|---|---|
  | calculator KV at 4,096 / 32,768 tokens | 0.377 / 3.02 GB | 0.377 / 3.02 GB | **0.755 / 6.04 GB** |
  | GenZ `get_kv_size` (elements a token) | 276,480 (54 layers) | 46,080 | **92,160** |
  | serving bytes a token | 552,960 (54 layers at 80) | 92,160 | **184,320** |
  | 1,024-token requests in 8 GiB | 15 | 51 | **32** |
  | disaggregation transfer, 512 + 128 tokens | 353,894,400 B | 131,461,632 B | **190,444,032 B** |
  | GenZ decode KV / weights | 35.9 / 9,842.5 MiB | 429.1 / 5,614.5 MiB | **789.1 / 6,064.5 MiB** |
  | `calculate_memory` parameters | 2.446B | 2.446B | **2.577B** (real 2.689B) |

  The calculator was already 2x low on origin/main; the serving path was over-counted there (all
  54 layers charged KV) and the round-4 head turned that into an under-count. The parameter count
  rises because the shared attention's Q / K / V read its 5,120-wide input
  (`attention_hidden_size`) at 160-wide heads: per block 26.2M -> 91.75M, matching transformers'
  shapes; `UniversalParameterCounter` without weight tying: 2.528B -> 2.659B (real 2.689B).
* **The other three rules move no corpus config.** No shipped or cached config states an SSM
  state and attention heads without a layout key other than Falcon-H1 (which was already read as
  a parallel hybrid; it now carries the note), none is named `mamba*` while stating attention
  heads, and none states `max_window_layers` with the window switched on (Qwen2 / Qwen2.5 / Qwen3
  ship `use_sliding_window: false`). Their tests are synthetic.
* While writing this a first version of the head-width reader descended into `text_config` and
  read `d_model` when the caller had its own hidden size: it moved Whisper (GenZ, which models it
  from default geometry) and a composite Higgs-Audio config. The reader now reads only the config
  the caller passes, with the caller's own head count and hidden size as the last resort, so no
  config without a stated width moves.

Round 6 (the fifth review) moved exactly five configs against the round-5 head (`--diff` of the
reviewer's corpus of `18d38ae` against this branch's: 189 unchanged, 5 changed):

* **Sliding layers whose window the table cannot read are full layers** (`layer_plan._sliding_bound`,
  in the table, so every reader agrees). The five ModernBERT-family encoder configs (Julia-1, the
  four Laya configs) state `layer_types` with `sliding_attention` and their window as
  `local_attention`, which is not a window key. The facts go back to origin/main's
  [22 / 28 full, 0 sliding, no window] (from [8, 14, 0, None] / [10, 18, 0, None]); the
  calculator's breakdown, which on origin/main reported those layers as sliding while charging
  them in full, now reports them full (`kv_full_layers` 8 -> 22, `kv_sliding_layers` 14 -> 0). No
  KV byte figure moves except the vLLM planner's, which charged the sliding groups zero blocks: on
  Julia-1's encoder (vllm-0.30-upstream, h100xm-80c-vgpu) the round-5 head admitted 6,023,920 KV
  tokens against origin/main's 2,190,512; it now admits origin/main's figure again (the
  reviewer's `vllm_probe.py`). transformers 5.12's only `local_attention` families (modernbert,
  modernvbert, pe_audio) are encoders, so no decoder in the corpus moves.
* **A dense layer of an MoE model is graphed at the width the config names for it**
  (`layer_plan.dense_ffn_width`: Llama 4's `intermediate_size_mlp`, MiniMax-M3's
  `dense_intermediate_size`, Cohere2-MoE's `prefix_dense_intermediate_size`). No corpus config
  states one of these keys with dense layers (the built-in `genz:meta-llama/Llama-4-*` models do
  not go through this reader), so it moves no corpus figure. The reviewer's Llama-4-Maverick text
  config (48 layers, `interleave_moe_layer_step` 2): GenZ decode weights origin/main 1,482,353 MiB
  (every layer MoE, ~2x high) -> round-5 head 750,803 MiB (393.6B, the 24 dense layers at 8,192)
  -> **756,563 MiB = 396.7B** (at 16,384: +3.02B). The remaining ~3B against Maverick's ~400B is
  Llama 4's always-on shared expert, which GenZ graphs for no family without a shared-expert count
  key; it is equally missing on origin/main and is left for a separate fix (see below).
* **The prefix cache prices a cached token by `ModelConfig.get_kv_size`**, as the serving memory
  model does (`PrefixCacheAnalyzer`, BudSimulator `/cache/analyze`): Qwen3.5-4B 131,072 -> 32,768
  B, Nemotron-H-8B 212,992 -> 16,384, DeepSeek-V3.2 (MLA) 1,748,992 -> 70,272. The corpus script
  does not measure the prefix cache; the serving bytes-per-token figure it does measure is the same
  number.

Found while checking round 6, not changed by it (present on origin/main): GenZ graphs a shared
expert only when the config states `n_shared_experts` / `num_shared_experts`, so Llama 4's
always-on shared expert and Qwen2-MoE's `shared_expert_intermediate_size` (no count key) are not
graphed -- an under-count of weights, not KV; and with a count but no width key (DeepSeek), the
shared expert defaults to `intermediate_size` rather than `moe_intermediate_size` x the count --
an over-count. The calculator's parameter counter handles a flat `llama4_text` config (no
`text_config`) as every layer MoE (783.6B for the Maverick text config above). Left for a
separate fix.

Round 7 (the sixth review) moved exactly one config against the round-6 head (`--diff` of the
round-6 head's corpus against this branch's: 193 unchanged, 1 changed); against origin/main the
corpus is still 135 unchanged, 59 changed:

* **GPT-Neo-1.3B reaches the GenZ serving path.** Its hub config states its depth as `num_layers`
  and carries a nested block (`task_specific_params`), so the decoder-block detection, which
  required `num_hidden_layers`, refused it (a ValueError, on origin/main too). It now reads the
  table's depth: `genz_layer_kinds` {MHA-global 12, MHA-local 12}, `get_kv_size` 98,304, serving
  196,608 B a token (24 x 2 x 2,048 x 2: the serving model charges the windowed layers in full, an
  over-estimate) and 42 requests of 1,024 tokens in 8 GiB. Its decode and chunked graphs still
  fail, later: the config states `intermediate_size: null` (GPT-Neo's MLP is 4 x hidden), which
  GenZ's FFN does not read -- as on origin/main, left for a separate fix.
* **The vLLM planner's concurrency counts every KV-cache group** (`Limits.blocks_per_sequence`, in
  `predict`, the profile scheduler and `shared_slice_mib`). The corpus does not run the planner; by
  hand (`vllm-0.30-upstream`, h100xm-80c-vgpu, concurrency 1,024), running sequences against
  what vLLM's hybrid allocator holds (`num_blocks` / the sequence's blocks in every group):

  | model, tokens | origin/main | round-6 head | this branch | allocator |
  |---|---|---|---|---|
  | Gemma-3-27b, 1,024 + 256 | 25 | 160 | **27** | 27 |
  | Gemma-3-27b, 4,096 + 256 | 7 | 47 | **19** | 19 |
  | Gemma-2-9B hub config, 1,024 + 256 | 123 | 246 | **123** | 123 |
  | Gemma-2-9B hub config, 4,096 + 256 | 36 | 72 | **37** | 37 |
  | gpt-oss-120b, 1,024 + 256 | 204 | 204 | **183** | 183 |
  | gpt-oss-120b, 4,096 + 256 | 60 | 60 | **58** | 58 |

  bud-vllm-0.10.0 agrees within 2 (Gemma-3-27b 28 / 20; gpt-oss-120b 182 / 57). gpt-oss reached
  this path on origin/main already; each sequence's 9 blocks in its sliding group (a 128-token
  window) were not counted there either. Origin/main read the Gemma stacks as all full, so its
  figure is one group of all their layers; the round-6 head counted their 10- and 21-layer groups
  as if a sequence held blocks in only one. A dense stack, and a hybrid's state blocks, are
  counted as before.
* **GenZ and BudSimulator's converter take the depth from the table** (`layer_plan.depth`) and the
  converter its window and Mamba decision from the same function as the HF loader
  (`get_language_model.graph_layer_kinds`), with the decoder read from under `text_config`. No
  corpus config states its depth only as a list (the saved Nemotron-H shape is synthetic in the
  tests: `get_kv_size` 0 -> 8,192 elements, serving 0 -> 16,384 B a token, 4 attention layers of
  52), and the corpus does not run the converter. By hand: SmallThinker-21BA3B's 39 sliding
  layers graphed with window None -> 4,096; a Llama-4 layout (9 chunked + 3 full,
  `attention_chunk_size` 8,192) None -> 8,192.
* **A GGUF's 0-KV layer with no recurrent mixer is a `no_op` layer**, not an FFN-only slot, unless
  the stack's KV layers run no FFN (Nemotron-H). No GGUF in the corpus states per-layer KV heads
  without a mixer; the reviewer's Deci file (`gguf.GGUFWriter`, 8 layers) went from 2 graphed FFNs
  to 8, with 6 attention layers and 2 `no_op` ones. An FFN no-op on an attention layer
  (`feed_forward_length` 0) is still graphed with an FFN: an over-count.
* The README now states which readers apply the window: no figure moves.

Round 8 (the seventh review, of `de8f1be`) moved exactly the six encoder-decoder configs in the
figures measured before (`--diff` of the round-7 head's corpus against this branch's, both
re-measured with this round's script: 188 unchanged, 6 changed; the round-7 head's re-measured
figures are identical to its committed corpus) and eleven configs in the converter's figures,
which the corpus measures from this round on (`conv_layers` [encoder, decoder], `conv_kv_size`,
`conv_state_bytes`; origin/main's converter prices no state, recorded 0):

* **An encoder-decoder's decoder is read under its own spellings** (`layer_plan.decoder_view`,
  `encoder_decoder_stacks`), by `calculate_kv_cache` / `kv_cache_breakdown`, `calculate_memory`'s
  encoder-decoder getters (and so the engine estimate), the FRD-027 facts, the GenZ loader and the
  converter (docs/LAYER_KINDS.md, "Encoder-decoder models"). The six in the corpus, all hub
  configs in the HuggingFace cache:

  | config | figure | round-7 head | this branch | truth (decoder self-attention) |
  |---|---|---|---|---|
  | whisper-large-v3-turbo (32 enc / 4 dec, 20 heads of 64) | `calculate_kv_cache` @ 4k | 0 | 0.0839 GB | 2 x 4 x 20 x 64 x 4,096 x 2 B = 0.0839 GB |
  | | breakdown / facts layers | 32 / 32 | 4 / 4 | 4 |
  | | GenZ `get_kv_size`, serving B a token | ValueError | 10,240 / 20,480 | 10,240 |
  | | converter `get_kv_size`, layers | 32,768, [32, 4] | 10,240, [32, 4] | |
  | whisper-base (6 / 6, 8 heads, `d_model` 512) | `calculate_kv_cache` @ 4k | 0 | 0.0503 GB | 6,144 x 4,096 x 2 B |
  | | GenZ / converter `get_kv_size` | 49,152 / 49,152 (7B defaults) | 6,144 / 6,144 | 6,144 |
  | | GenZ decode weights | 2,205 MiB | 86.6 MiB | (the decoder and embeddings) |
  | whisper-tiny.en (4 / 4, 6 heads, 384) | `calculate_kv_cache` @ 4k | 0 | 0.0252 GB | 3,072 x 4,096 x 2 B |
  | | GenZ / converter `get_kv_size` | 32,768 / 32,768 | 3,072 / 3,072 | 3,072 |
  | moonshine-base (8 / 8, `decoder_num_*`) | converter `get_kv_size`, layers | 3,328, [4, 4] | 6,656, [8, 8] | 6,656 |
  | | `calculate_memory` KV @ 4k | 0.0745 GB | 0.0945 GB | + the encoder term, below |
  | moonshine-tiny (6 / 6) | converter `get_kv_size`, layers | 1,728, [3, 3] | 3,456, [6, 6] | 3,456 |
  | cohere-transcribe (48-layer `encoder_config`, 8-layer top-level decoder) | converter `get_kv_size`, layers | 8,192, [4, 4] | 16,384, [48, 8] | 16,384 |
  | | `calculate_memory` KV @ 4k, params | 0.183 GB, 576.6M | 0.564 GB, 580.8M | |

  `calculate_kv_cache`, the facts and GenZ already read moonshine and cohere right (their
  decoder sits under the decoder-only names); the converter halved them. `calculate_memory`'s
  encoder-decoder path charges decoder self-attention, cross-attention over the encoder's
  `max_source_positions` (1,500 when unstated) and an encoder self-attention term of the same
  shape (an over-estimate it always had: an encoder keeps no cache while the decoder runs). It
  now takes the encoder's depth and width from the encoder's own key or block, so moonshine
  (8 encoder layers, read as 0) gains that term (+0.020 GB), and cohere gains it for its 48
  encoder layers (+0.368 GB) with its cross-attention at the encoder's 1,280 width rather than
  the decoder's 1,024 (+0.012 GB of KV, +4.2M cross-attention K/V parameters = 8 x 2 x 1,024 x 256).
  Whisper's path read `decoder_layers` / `decoder_attention_heads` already and does not move.
  Outside the corpus (tests/fixtures/encoder_decoder, `tests/test_encoder_decoder_stack.py`):
  flan-t5-xl (T5's `num_layers` / `num_decoder_layers` / `num_heads` / `d_kv`) -- converter 12
  decoder layers and 49,152 elements -> 24 and 98,304; `calculate_memory`'s decoder 4 layers of 20
  heads (Whisper defaults) -> 24 of 32, its decoder KV 0.134 -> 0.805 GB at 4k; facts 16 heads ->
  32; GenZ ValueError -> 98,304. Dia-1.6B-0626 (nested `encoder_config` / `decoder_config`) --
  converter 9 decoder layers, 9,216 -> 18 and 18,432; `calculate_kv_cache` 0 -> 18,432 a token;
  `calculate_memory`'s decoder 4 layers of 20 -> 18 of 4 KV heads. BART (transformers 5.12's
  `to_dict`, no `num_hidden_layers`): `calculate_kv_cache` 0 -> 24,576 a token, GenZ ValueError ->
  24,576.
* **The converter keeps the all-recurrent stack's state and the MLA latent** the HF loader passes
  (`recurrent_state`, `kv_lora_rank`, `q_lora_rank`, `qk_rope_head_dim`, `qk_nope_head_dim`,
  `v_head_dim`). Mamba-Codestral-7B `get_state_bytes(2)` 271,581,184 -> 272,367,616 B (Mamba-2's
  B/C groups in the conv buffer: 2 x 8 x 128 x 3 x 2 B x 64 layers = 786,432 B); converter
  `get_kv_size` DeepSeek-V3-Base (a database row) and DeepSeek-V3.2 874,496 -> 35,136 (61 x (512
  + 64)), DeepSeek-V2-Lite 110,592 -> 15,552, Kimi Linear 32,256 -> 4,032 -- each the HF loader's
  figure. A test now holds the converter to the loader's depth, KV and state on every hybrid
  fixture; the reviewer's `conv.py` over the corpus finds no config left where the two read the
  config and differ in depth, window, head width, KV heads, KV or state (9 differ only in how a
  uniform stack is labelled; 11 the loader refuses and the converter reads as it stands).
* **A GGUF window pattern shorter than the stack** windows the layers it reaches, the rest full,
  beside a per-layer KV array as without one (it raised IndexError). No GGUF in the corpus has
  one (llama.cpp writes `n_layer` entries).
* **The converter against origin/main**, now in the table below: 40 configs move, every one the
  HF loader's figure on this branch. Rounds 1-7 moved 31 (Kimi Linear and Mamba-Codestral moved
  again in round 8, which moved 9 more): the hybrids' recurrent layers hold no KV and their state
  is charged (Qwen3.5 / 3.6 and their derivatives, Qwen3-Next, LFM2, Kimi Linear,
  MiniMax-Text-01, Jamba, Bamba, granite-4.0-h, Nemotron-H / Nano, Zamba2, Hymba, Falcon-H1,
  SambaY, the Qwen3.5-2B GGUF; the pure SSMs mamba-2.8b, Mamba-Codestral and the database's
  falcon-mamba-7b: KV 0, state 293.6 MB), the decoder read from under `text_config` (Gemma-3-27b:
  62 layers and its window, not the 32-layer 7B default; MOSS-Transcribe, higgs-audio,
  Voxtral-Realtime) and GPT-Neo's depth (24, not 32). On every corpus config GenZ reads, the
  converter's `conv_kv_size` equals `genz_kv_size` and its `conv_state_bytes` equals
  `serving_state_bytes`.
* **The headline** above counted 193 configs (134 unchanged) where round 7's line counted 194 rows
  with the marker (135): the same measurement on two bases. It now states one.

Found while checking round 8, not changed by it (present on origin/main): a GPT-2-style config
(`n_layer`, `n_head`, `n_embd`) is refused by GenZ and read by the converter as a 7B default, and
the decoder's cross-attention cache is charged by `calculate_memory` only, at the encoder's width
(docs/LAYER_KINDS.md, "What is conservative"). Round 9 fixes the second.

Round 9 (the eighth review, of `2b26ea0`) moved 31 configs against the round-8 head (`--diff` of
the round-8 head's corpus against this branch's, both measured with this round's script: 227
unchanged), every one an encoder-decoder (shipped, cached or synthetic) or one of the three
`KVGeometry` depth readings below; no decoder-only config's KV, state or weights moved:

* **The cross-attention cache, by every reader.** `layer_plan.cross_attention_cache` reads each
  decoder layer's K / V over the encoder's output (the cross-attention's own width, the encoder's
  stated length), and the calculator, `calculate_memory` and the engine estimate, GenZ's
  `ModelConfig` (serving admission, `B_eff`, the transfer, the prefix cache), the FRD-027 facts
  and the vLLM planner (`Limits.state_bytes_per_sequence`, so `predict`, `sim_config` and
  `shared_slice_mib`), `KVGeometry`, the GenZ graphs and the converter charge it
  (docs/LAYER_KINDS.md, "Encoder-decoder models"). Round 8 had read the decoder right and so
  removed origin/main's accidental over-count (or its refusal), leaving the cross cache out:
  these figures were below the truth. origin/main -> round-8 head -> this branch:

  | config | figure | origin/main | round-8 head | this branch | truth |
  |---|---|---|---|---|---|
  | whisper-tiny.en | 1,024-token requests in 8 GiB | 128 | 1,365 | **553** | 8 GiB / (6.29 MB self + 4 x 2 x 1,500 x 384 x 2 B = 9.22 MB) = 553 |
  | whisper-base | the same | 85 | 682 | **276** | 8 GiB / (12.58 + 18.43 MB) = 276 |
  | whisper-large-v3-turbo | the same | refused | 409 | **166** | 8 GiB / (20.97 + 30.72 MB) = 166 |
  | | GenZ decode KV at 4,096 tokens | refused | 80.0 MiB | **109.3 MiB** | 80 + 29.3 |
  | | `KVGeometry` KV layers, cross bytes | 32, -- | 32, -- | **4, 30,720,000** | 4 decoder layers |
  | Dia-1.6B-0626 | vLLM running, 1,024 + 256 tokens (vllm-0.30, h100xm-80c-vgpu) | 58 | 1,024 | **357** | pool / (47 MB self + 151 MB cross) |
  | | 1,024-token requests in 8 GiB | 227 | 227 | **45** | 8 GiB / (37.7 + 151.0 MB) = 45 |
  | | `calculate_kv_cache` at 4k | 0 | 0.151 GB | **0.302 GB** | 0.151 self + 0.151 cross (16 KV heads of 128 over 1,024) |
  | cohere-transcribe | vLLM running, 1,024 + 256 | 1,024 | 1,024 | **356** | cross over the parakeet encoder's 5,000 positions: 164 MB a request |
  | flan-t5-xl geometry | vLLM running, 1,024 + 256 | 273 | 276 | **65** | cross over the request's 4,096-token context (no stated length) |
  | | `calculate_kv_cache` at 4k | 0 | 0.805 GB | **1.611 GB** | self + cross over the sequence |

  Where the config states no encoder length (T5, BART, Marian, Moonshine, ProphetNet, BLIP-2's
  flan-T5, Florence-2, MusicGen), the calculator, GenZ and the serving model charge the cache over
  the sequence (`get_kv_size`, the prefix cache's bytes a token and the GenZ graphs double:
  flan-t5-xl 98,304 -> 196,608 elements), and the facts and the vLLM planner over the request's
  context. `calculate_memory` now prices the cross term at the cross-attention's own width over
  the same tokens: Dia 0.335 -> 0.376 GB at 4k (16 x 128 wide, not the 1,024-wide encoder);
  flan-t5-xl 1.395 -> 1.906 GB and BART 0.349 -> 0.476 GB (the whole 4,096-token sequence, not
  Whisper's 1,500 frames); cohere-transcribe 0.564 -> 0.667 GB (its encoder block's 5,000
  positions). Whisper's report does not move (1,500 frames at 1,280 wide either way).
* **The GenZ prefill graph runs the encoder** (`encoder_stack`) and each decoder layer's K / V
  projection of it; every graph reads the cross cache. Prefill weights origin/main -> round 8 ->
  this branch: whisper-large-v3-turbo refused -> 403 -> **1,653 MiB** (809M parameters = 1,542
  MiB, plus GenZ's own graphed tied classifier, 127 MiB); whisper-base 2,610 -> 137 -> **185 MiB**
  (72.6M = 139 MiB + 51 MiB classifier); BART-large defaults refused -> 484 -> **868 MiB** (410M =
  782 MiB + 98 MiB). Decode weights rise only by the cross-attention's query and output
  projections (whisper-base 86.6 -> 92.6 MiB): a decode step streams the decoder, without the
  input-embedding table, as for every model -- a step's weights, not the resident model's
  (docs/LAYER_KINDS.md).
* **A whole encoder-decoder under `text_config`** (BLIP-2 / InstructBLIP with flan-T5,
  Florence-2; synthetic, in the shape transformers 5.12's config classes write): `decoder_view`
  views it in place and the GenZ / converter search views every candidate block. BLIP-2 with
  flan-t5-xl: converter [0, 12] layers, 18,432 elements a token (the Q-Former; origin/main
  [16, 16], 131,072) -> **[24, 24], 196,608** (98,304 self + 98,304 cross a token); GenZ the same
  (round 8 and origin/main: the Q-Former, 18,432); `calculate_kv_cache` 0 -> 1.611 GB at 4k.
  Florence-2: converter [0, 32] at the 7B default (origin/main [16, 16]) -> [12, 12]; GenZ refused
  -> 49,152; calculator 0 -> 0.403 GB.
* **ProphetNet's decoder heads** (`num_decoder_attention_heads`): `calculate_kv_cache` 0 (and 0
  on origin/main) -> 0.403 GB at 4k (12 x 2 x 16 x 64 a token, self + cross), `KVGeometry` 0 ->
  98,304 B a token. Its `calculate_memory` parameters move by the head count the decoder getters
  now read (640.7M -> 640.9M).
* **Only a decoder that attends to the encoder holds a cross cache**: the config says
  `is_encoder_decoder`, states a decoder stack of its own spelling, or its decoder block says
  `add_cross_attention` / states a cross geometry. Qwen3-TTS's two speech tokenizers (an audio
  codec: an encoder and a decoder nested side by side, never meeting) were read as encoder-decoders
  in round 8 for their depths; the first version of this round charged them a 262 MB cross cache
  over 8,000 positions. They hold none (and GenZ runs no encoder for them).
* **Depth and nested blocks through the table.** `KVGeometry` takes its KV layers from
  `layer_plan.depth` of the decoder view (MPT's `n_layers`, ChatGLM's `num_layers`, Qwen3-TTS's
  codecs: 1 -> 32 / 28 / 8; only the per-layer segment size moves); a nested decoder block that
  spells its depth `decoder_layers` (TrOCR in a VisionEncoderDecoder) is a stack block (converter
  [12, 32] at the 7B default -> [12, 12]; GenZ refused -> 24,576 + 577 image tokens of cross
  cache), and MusicGen's `text_encoder` is its encoder (converter [0, 24] -> [12, 24]). Their
  `calculate_memory` parameter counts move with the encoder-decoder getters' depths (TrOCR
  synthetic 803M -> 885M, MusicGen 1,105M -> 1,080M): those getters size an encoder-decoder from
  Whisper-shaped defaults where a key is missing, an estimate outside this work.

Found while checking round 9, not changed by it (present on origin/main): GenZ reads no GPT-2
spelling (`n_layer` / `n_embd` / `n_head`): a GPT-2 or StarCoder config, and a VisionEncoderDecoder
with a GPT-2 decoder, are graphed from the 7B defaults or refused, as on origin/main (a dense
dialect, not a layer kind: left for a separate fix); Gemma 3n's per-layer `intermediate_size` list
fails GenZ and the vLLM planner (a TypeError on origin/main too). For an audio encoder that states
no output length (Moonshine) the sequence is not a bound on its cross cache (docs/LAYER_KINDS.md,
"What is conservative").

The full per-config table (before = origin/main, after = this branch) follows.

## config:chaos-pressure/gpt-oss-20b/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| genz_chunked_kv_mb | 216.0 | 123.0 |
| slice_mib_16x4352 | [20736, true] | [20992, true] |
| vllm_limits | [155360, 2485760, 155360, 1, 0, 0.0, true] | [155360, 2485760, 155360, 1, 1, 0.0, true] |
| vllm_running_8192 | 294 | 289 |

## config:models--Cloudflare--clef-flash/snapshots/17f0b0ad64efb65d273590632833508766b2aae6/config.json (multimodal)

| figure | before | after |
|---|---|---|
| conv_kv_size | 262144 | 16384 |
| conv_state_bytes | 0 | 51511296 |
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
| genz_prefill_kv_mb | 512.0 | 177.125 |
| genz_prefill_weights_mb | 15656.0 | 16821.6875 |
| prefix_kv_bytes_per_token | 131072 | 32768 |
| serving_bytes_per_token | 131072 | 32768 |
| serving_requests_8gib_1k | 64 | 100 |
| serving_state_bytes | 0 | 51511296 |

## config:models--LiquidAI--LFM2.5-Audio-1.5B-JP/snapshots/6c34b4d590f80563f8cb2939c2ebd7686d952394/audio_detokenizer/config.json (hybrid)

| figure | before | after |
|---|---|---|
| conv_kv_size | 4096 | 1536 |
| conv_state_bytes | 0 | 15360 |
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
| genz_prefill_kv_mb | 20.087891 | 0.102539 |
| genz_prefill_weights_mb | 192.0 | 194.514648 |
| prefix_kv_bytes_per_token | 8192 | 3072 |
| serving_bytes_per_token | 8192 | 3072 |
| serving_requests_8gib_1k | 1024 | 2717 |
| serving_state_bytes | 0 | 15360 |

## config:models--LiquidAI--LFM2.5-Audio-1.5B/snapshots/c362a0625dfe45aa588dce5f0ada28a7e5707628/audio_detokenizer/config.json (hybrid)

| figure | before | after |
|---|---|---|
| conv_kv_size | 4096 | 1536 |
| conv_state_bytes | 0 | 15360 |
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
| genz_prefill_kv_mb | 20.087891 | 0.102539 |
| genz_prefill_weights_mb | 192.0 | 194.514648 |
| prefix_kv_bytes_per_token | 8192 | 3072 |
| serving_bytes_per_token | 8192 | 3072 |
| serving_requests_8gib_1k | 1024 | 2717 |
| serving_state_bytes | 0 | 15360 |

## config:models--OpenMOSS-Team--MOSS-Transcribe-Diarize/snapshots/704aa4a9c304e8520be88901e0d1960158ef5b15/config.json (audio-llm)

| figure | before | after |
|---|---|---|
| conv_kv_size | 262144 | 57344 |
| conv_layers | [0, 32] | [0, 28] |

## config:models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/snapshots/5d83992436eae1d760afd27aff78a71d676296fc/speech_tokenizer/config.json (unknown)

| figure | before | after |
|---|---|---|
| conv_kv_size | 262144 | 16384 |
| conv_layers | [0, 32] | [8, 8] |
| eng_None_32768_kv_mb | 0.0 | 1073.741824 |
| eng_None_32768_total_mb | 469.858406 | 1543.60023 |
| eng_None_4096_kv_mb | 0.0 | 134.217728 |
| eng_None_4096_total_mb | 386.811187 | 521.028915 |
| eng_llamacpp_32768_kv_mb | 0.0 | 1073.741824 |
| eng_llamacpp_32768_total_mb | 469.858406 | 1543.60023 |
| eng_llamacpp_4096_kv_mb | 0.0 | 134.217728 |
| eng_llamacpp_4096_total_mb | 386.811187 | 521.028915 |
| facts_layers | [32, 0, 0, null] | [8, 0, 0, null] |
| kv_full_layers | 0 | 8 |
| kv_gb@32768 | 0.0 | 1.073741824 |
| kv_gb@4096 | 0.0 | 0.134217728 |
| kv_source | "uniform" | "_layer_metadata" |
| kvgeo_32768 | [0.0, 1, 0, 0.0, 16, 0.0] | [32768.0, 8, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [0.0, 1, 0, 0.0, 16, 0.0] | [32768.0, 8, 0, 0.0, 16, 0.0] |
| report_kv_gb | 0.0 | 0.134217728 |
| report_total_gb | 0.403420631 | 0.564481905 |
| sim_max_running | 104 | 400 |
| slice_mib_16x4352 | [65792, true] | [8192, true] |
| vllm_limits | [6750, 108000, 6750, 1, 0, 0.0, true] | [142753, 2284048, 142753, 1, 0, 0.0, true] |
| vllm_running_1024 | 84 | 1024 |
| vllm_running_8192 | 12 | 270 |

## config:models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5/speech_tokenizer/config.json (unknown)

| figure | before | after |
|---|---|---|
| conv_kv_size | 262144 | 16384 |
| conv_layers | [0, 32] | [8, 8] |
| eng_None_32768_kv_mb | 0.0 | 1073.741824 |
| eng_None_32768_total_mb | 469.858406 | 1543.60023 |
| eng_None_4096_kv_mb | 0.0 | 134.217728 |
| eng_None_4096_total_mb | 386.811187 | 521.028915 |
| eng_llamacpp_32768_kv_mb | 0.0 | 1073.741824 |
| eng_llamacpp_32768_total_mb | 469.858406 | 1543.60023 |
| eng_llamacpp_4096_kv_mb | 0.0 | 134.217728 |
| eng_llamacpp_4096_total_mb | 386.811187 | 521.028915 |
| facts_layers | [32, 0, 0, null] | [8, 0, 0, null] |
| kv_full_layers | 0 | 8 |
| kv_gb@32768 | 0.0 | 1.073741824 |
| kv_gb@4096 | 0.0 | 0.134217728 |
| kv_source | "uniform" | "_layer_metadata" |
| kvgeo_32768 | [0.0, 1, 0, 0.0, 16, 0.0] | [32768.0, 8, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [0.0, 1, 0, 0.0, 16, 0.0] | [32768.0, 8, 0, 0.0, 16, 0.0] |
| report_kv_gb | 0.0 | 0.134217728 |
| report_total_gb | 0.403420631 | 0.564481905 |
| sim_max_running | 104 | 400 |
| slice_mib_16x4352 | [65792, true] | [8192, true] |
| vllm_limits | [6750, 108000, 6750, 1, 0, 0.0, true] | [142753, 2284048, 142753, 1, 0, 0.0, true] |
| vllm_running_1024 | 84 | 1024 |
| vllm_running_8192 | 12 | 270 |

## config:models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b/config.json (multimodal)

| figure | before | after |
|---|---|---|
| conv_kv_size | 262144 | 16384 |
| conv_state_bytes | 0 | 51511296 |
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
| genz_prefill_kv_mb | 512.0 | 177.125 |
| genz_prefill_weights_mb | 8345.0 | 9074.023438 |
| prefix_kv_bytes_per_token | 131072 | 32768 |
| serving_bytes_per_token | 131072 | 32768 |
| serving_requests_8gib_1k | 64 | 100 |
| serving_state_bytes | 0 | 51511296 |

## config:models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json (multimodal)

| figure | before | after |
|---|---|---|
| conv_kv_size | 262144 | 16384 |
| conv_state_bytes | 0 | 51511296 |
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
| genz_prefill_kv_mb | 512.0 | 177.125 |
| genz_prefill_weights_mb | 8345.0 | 9074.023438 |
| prefix_kv_bytes_per_token | 131072 | 32768 |
| serving_bytes_per_token | 131072 | 32768 |
| serving_requests_8gib_1k | 64 | 100 |
| serving_state_bytes | 0 | 51511296 |

## config:models--SupersonicLabs--Julia-1/snapshots/a85b127321d580d65176c89ced8273f305745d85/encoder/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| kv_full_layers | 8 | 22 |
| kv_sliding_layers | 14 | 0 |
| kvgeo_32768 | [33792.0, 8, 0, 0.0, 16, 0.0] | [33792.0, 22, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [33792.0, 8, 0, 0.0, 16, 0.0] | [33792.0, 22, 0, 0.0, 16, 0.0] |

## config:models--bosonai--higgs-audio-v3-tts-4b/snapshots/239f63fb7b02b1aa085f98d9efae5e35cc5523e8/config.json (unknown)

| figure | before | after |
|---|---|---|
| conv_kv_size | 262144 | 73728 |
| conv_layers | [0, 32] | [0, 36] |

## config:models--convaiinnovations--laya-multilingual/snapshots/e4e9ddf21a7b1903b7acffd8814ad4307bf63a67/encoder/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| kv_full_layers | 8 | 22 |
| kv_sliding_layers | 14 | 0 |
| kvgeo_32768 | [67584.0, 8, 0, 0.0, 16, 0.0] | [67584.0, 22, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [67584.0, 8, 0, 0.0, 16, 0.0] | [67584.0, 22, 0, 0.0, 16, 0.0] |

## config:models--convaiinnovations--laya-typed-decisions/snapshots/1a793eb568e6718f15941d08f85432581df534e3/encoder/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| kv_full_layers | 10 | 28 |
| kv_sliding_layers | 18 | 0 |
| kvgeo_32768 | [114688.0, 10, 0, 0.0, 16, 0.0] | [114688.0, 28, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [114688.0, 10, 0, 0.0, 16, 0.0] | [114688.0, 28, 0, 0.0, 16, 0.0] |

## config:models--convaiinnovations--laya/snapshots/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851/encoder/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| kv_full_layers | 10 | 28 |
| kv_sliding_layers | 18 | 0 |
| kvgeo_32768 | [114688.0, 10, 0, 0.0, 16, 0.0] | [114688.0, 28, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [114688.0, 10, 0, 0.0, 16, 0.0] | [114688.0, 28, 0, 0.0, 16, 0.0] |

## config:models--convaiinnovations--laya/snapshots/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851/typed-decisions/encoder/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| kv_full_layers | 10 | 28 |
| kv_sliding_layers | 18 | 0 |
| kvgeo_32768 | [114688.0, 10, 0, 0.0, 16, 0.0] | [114688.0, 28, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [114688.0, 10, 0, 0.0, 16, 0.0] | [114688.0, 28, 0, 0.0, 16, 0.0] |

## config:models--internlm--Intern-Decision-4B/snapshots/0e5e6aa7d6d750e2b1504ba11a8136cb58aeb3cd/config.json (multimodal)

| figure | before | after |
|---|---|---|
| conv_kv_size | 262144 | 16384 |
| conv_state_bytes | 0 | 51511296 |
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
| genz_prefill_kv_mb | 512.0 | 177.125 |
| genz_prefill_weights_mb | 8345.0 | 9074.023438 |
| prefix_kv_bytes_per_token | 131072 | 32768 |
| serving_bytes_per_token | 131072 | 32768 |
| serving_requests_8gib_1k | 64 | 100 |
| serving_state_bytes | 0 | 51511296 |

## config:models--kyutai--mimi/snapshots/89091b3e466eb6a9d11e537bf26b144f194978f7/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| facts_full_attention | true | false |
| facts_layers | [8, 0, 0, null] | [0, 8, 0, 250] |

## config:models--microsoft--VibeVoice-1.5B/snapshots/c00898d257e6b46004e3e2866a47534085fb685a/config.json (unknown)

| figure | before | after |
|---|---|---|
| conv_kv_size | 262144 | 14336 |
| conv_layers | [0, 32] | [0, 28] |

## config:models--mistralai--Voxtral-Mini-4B-Realtime-2602/snapshots/2769294da9567371363522aac9bbcfdd19447add/config.json (audio-llm)

| figure | before | after |
|---|---|---|
| conv_kv_size | 196608 | 53248 |
| conv_layers | [0, 32] | [0, 26] |
| facts_full_attention | true | false |
| facts_layers | [26, 0, 0, null] | [0, 26, 0, 8192] |

## config:models--nari-labs--Dia-1.6B-0626/snapshots/ef2795fcc29c5abe6ffc91fd33808588b49bbc66/config.json (encoder-decoder)

| figure | before | after |
|---|---|---|
| conv_kv_size | 131072 | 18432 |
| conv_layers | [16, 16] | [12, 18] |
| conv_state_bytes | 0 | 150994944 |
| disagg_transfer_bytes_640 | 23592960 | 174587904 |
| eng_None_32768_kv_mb | 701.80864 | 1432.682496 |
| eng_None_32768_params | 808394240 | 1763025664 |
| eng_None_32768_total_mb | 2484.691558 | 5124.828262 |
| eng_None_4096_kv_mb | 114.60608 | 375.717888 |
| eng_None_4096_params | 808394240 | 1763025664 |
| eng_None_4096_total_mb | 1814.441779 | 3984.816435 |
| eng_llamacpp_32768_kv_mb | 701.80864 | 1432.682496 |
| eng_llamacpp_32768_params | 808394240 | 1763025664 |
| eng_llamacpp_32768_total_mb | 2484.691558 | 5124.828262 |
| eng_llamacpp_4096_kv_mb | 114.60608 | 375.717888 |
| eng_llamacpp_4096_params | 808394240 | 1763025664 |
| eng_llamacpp_4096_total_mb | 1814.441779 | 3984.816435 |
| facts_layers | [32, 0, 0, null] | [18, 0, 0, null] |
| genz_chunked_kv_mb | 162.0 | 450.0 |
| genz_decode_kv_mb | 144.0 | 288.0 |
| genz_decode_weights_mb | 2092.015625 | 2380.015625 |
| genz_prefill_kv_mb | 144.0 | 384.0 |
| genz_prefill_weights_mb | 2096.03125 | 3008.03125 |
| kv_cross_layers | 0 | 18 |
| kv_full_layers | 0 | 18 |
| kv_gb@32768 | 0.0 | 1.358954496 |
| kv_gb@4096 | 0.0 | 0.301989888 |
| kvgeo_32768 | [0.0, 1, 0, 0.0, 16, 0.0] | [36864.0, 18, 0, 0.0, 16, 150994944.0] |
| kvgeo_4096 | [0.0, 1, 0, 0.0, 16, 0.0] | [36864.0, 18, 0, 0.0, 16, 150994944.0] |
| report_kv_gb | 0.11460608 | 0.375717888 |
| report_params | 808394240 | 1763025664 |
| report_total_gb | 1.856430039 | 4.078566257 |
| report_weight_gb | 1.61678848 | 3.526051328 |
| serving_requests_8gib_1k | 227 | 45 |
| serving_state_bytes | 0 | 150994944 |
| sim_max_running | 72 | 372 |
| slice_mib_16x4352 | [81582, false] | [15360, true] |
| vllm_limits | [4702, 75232, 4702, 1, 0, 0.0, true] | [120000, 1920000, 120000, 1, 0, 150994944.0, true] |
| vllm_running_1024 | 58 | 357 |
| vllm_running_8192 | 8 | 153 |

## config:models--onnx-community--cohere-transcribe-03-2026-ONNX/snapshots/31b1c6211c9000d76b077ddd23b74c9090badeba/config.json (encoder-decoder)

| figure | before | after |
|---|---|---|
| conv_kv_size | 8192 | 16384 |
| conv_layers | [4, 4] | [48, 8] |
| conv_state_bytes | 0 | 163840000 |
| disagg_transfer_bytes_640 | 20971520 | 184811520 |
| eng_None_32768_kv_mb | 1122.893824 | 1606.221824 |
| eng_None_32768_params | 576643072 | 580837376 |
| eng_None_32768_total_mb | 2417.108582 | 2908.82519 |
| eng_None_4096_kv_mb | 183.369728 | 666.697728 |
| eng_None_4096_params | 576643072 | 580837376 |
| eng_None_4096_total_mb | 1407.120179 | 1898.836787 |
| eng_llamacpp_32768_kv_mb | 1122.893824 | 1606.221824 |
| eng_llamacpp_32768_params | 576643072 | 580837376 |
| eng_llamacpp_32768_total_mb | 2417.108582 | 2908.82519 |
| eng_llamacpp_4096_kv_mb | 183.369728 | 666.697728 |
| eng_llamacpp_4096_params | 576643072 | 580837376 |
| eng_llamacpp_4096_total_mb | 1407.120179 | 1898.836787 |
| genz_chunked_kv_mb | 144.0 | 456.5 |
| genz_decode_kv_mb | 128.0 | 284.25 |
| genz_decode_weights_mb | 224.0 | 256.0 |
| genz_prefill_kv_mb | 128.0 | 1456.125 |
| genz_prefill_weights_mb | 256.0 | 2128.0 |
| kv_cross_layers | 0 | 8 |
| kv_gb@32768 | 1.073741824 | 1.237581824 |
| kv_gb@4096 | 0.134217728 | 0.298057728 |
| kvgeo_32768 | [32768.0, 8, 0, 0.0, 16, 0.0] | [32768.0, 8, 0, 0.0, 16, 163840000.0] |
| kvgeo_4096 | [32768.0, 8, 0, 0.0, 16, 0.0] | [65536.0, 8, 0, 0.0, 16, 163840000.0] |
| report_kv_gb | 0.183369728 | 0.666697728 |
| report_params | 576643072 | 580837376 |
| report_total_gb | 1.459883786 | 2.048726794 |
| report_weight_gb | 1.153286144 | 1.161674752 |
| serving_requests_8gib_1k | 256 | 43 |
| serving_state_bytes | 0 | 163840000 |
| sim_max_running | 400 | 368 |
| slice_mib_16x4352 | [8704, true] | [12800, true] |
| vllm_limits | [141755, 2268080, 141755, 1, 0, 0.0, true] | [139835, 2237360, 139835, 1, 0, 163840000.0, true] |
| vllm_running_1024 | 1024 | 356 |
| vllm_running_8192 | 268 | 166 |

## config:models--onnx-community--moonshine-base-ONNX/snapshots/b1e9b6aae3c3c7298f10c3798393fdf38e8fbbad/config.json (encoder-decoder)

| figure | before | after |
|---|---|---|
| conv_kv_size | 3328 | 13312 |
| conv_layers | [4, 4] | [8, 8] |
| disagg_transfer_bytes_640 | 8519680 | 17039360 |
| eng_None_32768_kv_mb | 456.175616 | 892.383232 |
| eng_None_32768_total_mb | 753.298125 | 1189.505741 |
| eng_None_4096_kv_mb | 74.493952 | 129.019904 |
| eng_None_4096_total_mb | 326.63255 | 381.158502 |
| eng_llamacpp_32768_kv_mb | 456.175616 | 892.383232 |
| eng_llamacpp_32768_total_mb | 753.298125 | 1189.505741 |
| eng_llamacpp_4096_kv_mb | 74.493952 | 129.019904 |
| eng_llamacpp_4096_total_mb | 326.63255 | 381.158502 |
| genz_chunked_kv_mb | 58.5 | 117.0 |
| genz_decode_kv_mb | 52.0 | 104.0 |
| genz_decode_weights_mb | 57.6875 | 62.96875 |
| genz_kv_size | 6656 | 13312 |
| genz_prefill_kv_mb | 52.0 | 156.0 |
| genz_prefill_weights_mb | 83.6875 | 125.9375 |
| kv_cross_layers | 0 | 8 |
| kv_gb@32768 | 0.436207616 | 0.872415232 |
| kv_gb@4096 | 0.054525952 | 0.109051904 |
| kvgeo_32768 | [13312.0, 8, 0, 0.0, 16, 0.0] | [26624.0, 8, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [13312.0, 8, 0, 0.0, 16, 0.0] | [26624.0, 8, 0, 0.0, 16, 0.0] |
| prefix_kv_bytes_per_token | 13312 | 26624 |
| report_kv_gb | 0.074493952 | 0.129019904 |
| report_total_gb | 0.351430523 | 0.416861665 |
| serving_bytes_per_token | 13312 | 26624 |
| serving_requests_8gib_1k | 630 | 315 |
| slice_mib_16x4352 | [6656, true] | [7680, true] |
| vllm_limits | [351373, 5621968, 351373, 1, 0, 0.0, true] | [351373, 5621968, 351373, 1, 0, 6815744.0, true] |
| vllm_running_8192 | 665 | 627 |

## config:models--onnx-community--moonshine-tiny-ONNX/snapshots/a6da1241cd305dcd64eab1edbd615f2bb9aabb95/config.json (encoder-decoder)

| figure | before | after |
|---|---|---|
| conv_kv_size | 1728 | 6912 |
| conv_layers | [3, 3] | [6, 6] |
| disagg_transfer_bytes_640 | 4423680 | 8847360 |
| eng_None_32768_kv_mb | 236.860416 | 463.352832 |
| eng_None_32768_total_mb | 399.278822 | 625.771238 |
| eng_None_4096_kv_mb | 38.679552 | 66.991104 |
| eng_None_4096_total_mb | 169.955251 | 198.266803 |
| eng_llamacpp_32768_kv_mb | 236.860416 | 463.352832 |
| eng_llamacpp_32768_total_mb | 399.278822 | 625.771238 |
| eng_llamacpp_4096_kv_mb | 38.679552 | 66.991104 |
| eng_llamacpp_4096_total_mb | 169.955251 | 198.266803 |
| genz_chunked_kv_mb | 30.375 | 60.75 |
| genz_decode_kv_mb | 27.0 | 54.0 |
| genz_decode_weights_mb | 29.390625 | 31.289062 |
| genz_kv_size | 3456 | 6912 |
| genz_prefill_kv_mb | 27.0 | 81.0 |
| genz_prefill_weights_mb | 47.390625 | 62.578125 |
| kv_cross_layers | 0 | 6 |
| kv_gb@32768 | 0.226492416 | 0.452984832 |
| kv_gb@4096 | 0.028311552 | 0.056623104 |
| kvgeo_32768 | [6912.0, 6, 0, 0.0, 16, 0.0] | [13824.0, 6, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [6912.0, 6, 0, 0.0, 16, 0.0] | [13824.0, 6, 0, 0.0, 16, 0.0] |
| prefix_kv_bytes_per_token | 6912 | 13824 |
| report_kv_gb | 0.038679552 | 0.066991104 |
| report_total_gb | 0.184591703 | 0.218565565 |
| serving_bytes_per_token | 6912 | 13824 |
| serving_requests_8gib_1k | 1213 | 606 |
| slice_mib_16x4352 | [6144, true] | [6656, true] |
| vllm_limits | [677592, 10841472, 677592, 1, 0, 0.0, true] | [677592, 10841472, 677592, 1, 0, 3538944.0, true] |

## config:models--onnx-community--whisper-base/snapshots/1846881b6b3a3024392c1eea3ad983695bc23925/config.json (encoder-decoder)

| figure | before | after |
|---|---|---|
| conv_kv_size | 49152 | 6144 |
| conv_state_bytes | 0 | 18432000 |
| disagg_transfer_bytes_640 | 62914560 | 26296320 |
| genz_chunked_kv_mb | 432.0 | 89.15625 |
| genz_decode_kv_mb | 384.0 | 65.578125 |
| genz_decode_weights_mb | 2205.195312 | 92.649414 |
| genz_kv_size | 49152 | 6144 |
| genz_prefill_kv_mb | 384.0 | 83.15625 |
| genz_prefill_weights_mb | 2610.390625 | 185.298828 |
| kv_cross_layers | 0 | 6 |
| kv_gb@32768 | 0.0 | 0.421085184 |
| kv_gb@4096 | 0.0 | 0.068763648 |
| kvgeo_32768 | [0.0, 6, 0, 0.0, 16, 0.0] | [12288.0, 6, 0, 0.0, 16, 18432000.0] |
| kvgeo_4096 | [0.0, 6, 0, 0.0, 16, 0.0] | [12288.0, 6, 0, 0.0, 16, 18432000.0] |
| prefix_kv_bytes_per_token | 98304 | 12288 |
| serving_bytes_per_token | 98304 | 12288 |
| serving_requests_8gib_1k | 85 | 276 |
| serving_state_bytes | 0 | 18432000 |
| slice_mib_16x4352 | [6656, true] | [6912, true] |
| vllm_limits | [382126, 6114016, 382126, 1, 0, 0.0, true] | [382126, 6114016, 382126, 1, 0, 18432000.0, true] |
| vllm_running_8192 | 723 | 614 |

## config:models--onnx-community--whisper-large-v3-turbo/snapshots/360ebcde2559d60bb474678be3c1de9ef347d01a/config.json (encoder-decoder)

| figure | before | after |
|---|---|---|
| conv_kv_size | 32768 | 10240 |
| conv_state_bytes | 0 | 30720000 |
| disagg_transfer_bytes_640 | null | 43827200 |
| facts_layers | [32, 0, 0, null] | [4, 0, 0, null] |
| genz_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'apply_spec_augment', 'architectures', 'attention_dropout'," | null |
| genz_chunked_kv_mb | null | 148.59375 |
| genz_decode_kv_mb | null | 109.296875 |
| genz_decode_weights_mb | null | 301.625977 |
| genz_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'apply_spec_augment', 'architectures', 'attention_dropout'," | null |
| genz_ffn_kinds | null | {"Dense": 4} |
| genz_kv_size | null | 10240 |
| genz_layer_kinds | null | {"MHA-global": 4} |
| genz_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'apply_spec_augment', 'architectures', 'attention_dropout'," | null |
| genz_prefill_kv_mb | null | 343.671875 |
| genz_prefill_weights_mb | null | 1653.251953 |
| kv_cross_layers | 0 | 4 |
| kv_full_layers | 32 | 4 |
| kv_gb@32768 | 0.0 | 0.70180864 |
| kv_gb@4096 | 0.0 | 0.11460608 |
| kvgeo_32768 | [0.0, 32, 0, 0.0, 16, 0.0] | [20480.0, 4, 0, 0.0, 16, 30720000.0] |
| kvgeo_4096 | [0.0, 32, 0, 0.0, 16, 0.0] | [20480.0, 4, 0, 0.0, 16, 30720000.0] |
| prefix_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'apply_spec_augment" | null |
| prefix_kv_bytes_per_token | null | 20480 |
| serving_bytes_per_token | null | 20480 |
| serving_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'apply_spec_augment', 'architectures', 'attention_dropout'," | null |
| serving_requests_8gib_1k | null | 166 |
| serving_state_bytes | null | 30720000 |
| slice_mib_16x4352 | [22272, true] | [9472, true] |
| vllm_limits | [27344, 437504, 27344, 1, 0, 0.0, true] | [224354, 3589664, 224354, 1, 0, 30720000.0, true] |
| vllm_running_1024 | 341 | 1024 |
| vllm_running_8192 | 51 | 360 |

## config:models--onnx-community--whisper-tiny.en/snapshots/2575352d61be1bf7225cf8f8b268a4678025fc58/config.json (encoder-decoder)

| figure | before | after |
|---|---|---|
| conv_kv_size | 32768 | 3072 |
| conv_state_bytes | 0 | 9216000 |
| disagg_transfer_bytes_640 | 41943040 | 13148160 |
| genz_chunked_kv_mb | 288.0 | 44.578125 |
| genz_decode_kv_mb | 256.0 | 32.789062 |
| genz_decode_weights_mb | 1605.1875 | 53.736328 |
| genz_kv_size | 32768 | 3072 |
| genz_prefill_kv_mb | 256.0 | 41.578125 |
| genz_prefill_weights_mb | 2010.375 | 107.472656 |
| kv_cross_layers | 0 | 4 |
| kv_gb@32768 | 0.0 | 0.210542592 |
| kv_gb@4096 | 0.0 | 0.034381824 |
| kvgeo_32768 | [0.0, 4, 0, 0.0, 16, 0.0] | [6144.0, 4, 0, 0.0, 16, 9216000.0] |
| kvgeo_4096 | [0.0, 4, 0, 0.0, 16, 0.0] | [6144.0, 4, 0, 0.0, 16, 9216000.0] |
| prefix_kv_bytes_per_token | 65536 | 6144 |
| serving_bytes_per_token | 65536 | 6144 |
| serving_requests_8gib_1k | 128 | 553 |
| serving_state_bytes | 0 | 9216000 |
| slice_mib_16x4352 | [5888, true] | [6144, true] |
| vllm_limits | [765220, 12243520, 765220, 1, 0, 0.0, true] | [765220, 12243520, 765220, 1, 0, 9216000.0, true] |

## config:models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| genz_chunked_kv_mb | 324.0 | 184.5 |
| sim_max_running | 250 | 220 |
| vllm_limits | [16363, 261808, 16363, 1, 0, 0.0, true] | [16363, 261808, 16363, 1, 1, 0.0, true] |
| vllm_running_1024 | 204 | 183 |

## config:models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| genz_chunked_kv_mb | 216.0 | 123.0 |
| slice_mib_16x4352 | [20736, true] | [20992, true] |
| vllm_limits | [155360, 2485760, 155360, 1, 0, 0.0, true] | [155360, 2485760, 155360, 1, 1, 0.0, true] |
| vllm_running_8192 | 294 | 289 |

## db:deepseek-ai/DeepSeek-V3-Base (decoder-only)

| figure | before | after |
|---|---|---|
| conv_kv_size | 874496 | 35136 |
| genz_chunked_kv_mb | 7686.0 | 171.5625 |
| genz_decode_weights_mb | 1335494.375 | 1273239.875 |
| genz_ffn_kinds | {"MoE": 61} | {"Dense": 3, "MoE": 58} |
| genz_prefill_weights_mb | 1337261.875 | 1275007.375 |
| prefix_kv_bytes_per_token | 1748992 | 70272 |
| serving_bytes_per_token | 1748992 | 70272 |
| serving_requests_8gib_1k | 4 | 119 |

## db:deepseek-ai/deepseek-moe-16b-base (decoder-only)

| figure | before | after |
|---|---|---|
| genz_decode_weights_mb | 30871.0 | 29943.0 |
| genz_ffn_kinds | {"MoE": 28} | {"Dense": 1, "MoE": 27} |
| genz_prefill_weights_mb | 31271.0 | 30343.0 |

## db:tiiuae/falcon-mamba-7b (state-space)

| figure | before | after |
|---|---|---|
| conv_kv_size | 524288 | 0 |
| conv_state_bytes | 0 | 293601280 |
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
| prefix_kv_bytes_per_token | 1048576 | 0 |
| serving_bytes_per_token | 1048576 | 0 |
| serving_requests_8gib_1k | 8 | 29 |
| serving_state_bytes | 0 | 293601280 |

## fixture:EleutherAI_gpt-neo-1.3B.json (decoder-only)

| figure | before | after |
|---|---|---|
| conv_kv_size | 131072 | 98304 |
| conv_layers | [0, 32] | [0, 24] |
| disagg_transfer_bytes_640 | null | 125829120 |
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
| genz_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['activation_function', 'architectures', 'attention_dropout', 'attention_layers', 'attention_types', 'bos_token_id', 'embe" | "TypeError: unsupported operand type(s) for /: 'NoneType' and 'int'" |
| genz_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['activation_function', 'architectures', 'attention_dropout', 'attention_layers', 'attention_types', 'bos_token_id', 'embe" | "TypeError: unsupported operand type(s) for /: 'NoneType' and 'int'" |
| genz_ffn_kinds | null | {"Dense": 24} |
| genz_kv_size | null | 98304 |
| genz_layer_kinds | null | {"MHA-global": 12, "MHA-local": 12} |
| genz_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['activation_function', 'architectures', 'attention_dropout', 'attention_layers', 'attention_types', 'bos_token_id', 'embe" | "TypeError: unsupported operand type(s) for /: 'NoneType' and 'int'" |
| kv_full_layers | 0 | 12 |
| kv_gb@32768 | 0.0 | 6.442450944 |
| kv_gb@4096 | 0.0 | 0.805306368 |
| kv_sliding_layers | 24 | 12 |
| kv_source | "uniform" | "layer_plan:attention_types" |
| kvgeo_32768 | [0.0, 1, 0, 0.0, 16, 0.0] | [196608.0, 12, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [0.0, 1, 0, 0.0, 16, 0.0] | [196608.0, 12, 0, 0.0, 16, 0.0] |
| model_type | "hybrid" | "decoder-only" |
| prefix_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['activation_function', 'architectures', 'attention_dropout', 'attention_layers', " | null |
| prefix_kv_bytes_per_token | null | 196608 |
| report_kv_gb | 0.0 | 0.805306368 |
| report_params | 706906112 | 1310885888 |
| report_state_gb | 0.00172032 | 0.0 |
| report_total_gb | 1.584990945 | 3.757253755 |
| report_weight_gb | 1.413812224 | 2.621771776 |
| serving_bytes_per_token | null | 196608 |
| serving_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['activation_function', 'architectures', 'attention_dropout', 'attention_layers', 'attention_types', 'bos_token_id', 'embe" | null |
| serving_requests_8gib_1k | null | 42 |
| serving_state_bytes | null | 0 |
| sim_max_running | 342 | 400 |
| slice_mib_16x4352 | [25088, true] | [17664, true] |
| state_gb | 0.00172032 | 0.0 |
| vllm_limits | [22571, 361136, 22571, 1, 0, 0.0, true] | [45143, 722288, 45143, 1, 1, 0.0, true] |
| vllm_running_1024 | 282 | 465 |
| vllm_running_8192 | 42 | 82 |

## fixture:LiquidAI_LFM2-2.6B.json (hybrid)

| figure | before | after |
|---|---|---|
| conv_kv_size | 30720 | 8192 |
| conv_state_bytes | 0 | 270336 |
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
| genz_prefill_kv_mb | 240.0 | 64.257812 |
| genz_prefill_weights_mb | 3632.0 | 3896.257812 |
| prefix_kv_bytes_per_token | 61440 | 16384 |
| serving_bytes_per_token | 61440 | 16384 |
| serving_requests_8gib_1k | 136 | 503 |
| serving_state_bytes | 0 | 270336 |

## fixture:MiniMaxAI_MiniMax-Text-01.json (hybrid)

| figure | before | after |
|---|---|---|
| conv_kv_size | 163840 | 20480 |
| conv_state_bytes | 0 | 293601280 |
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
| genz_prefill_kv_mb | 1280.0 | 440.0 |
| genz_prefill_weights_mb | 851439.0 | 869919.0 |
| prefix_kv_bytes_per_token | 327680 | 40960 |
| serving_bytes_per_token | 327680 | 40960 |
| serving_state_bytes | 0 | 293601280 |
| vllm_limits | [0, 0, 0, 1, 0, 0.0, false] | [0, 0, 0, 1, 0, 293601280.0, false] |

## fixture:PowerInfer_SmallThinker-21BA3B-Instruct.json (decoder-only)

| figure | before | after |
|---|---|---|
| facts_full_attention | true | false |
| facts_layers | [52, 0, 0, null] | [13, 39, 0, 4096] |
| genz_layer_kinds | {"MHA-global": 52} | {"MHA-global": 13, "MHA-local": 39} |
| slice_mib_16x4352 | [26368, true] | [26112, true] |
| vllm_limits | [36572, 585152, 36572, 1, 0, 0.0, true] | [146290, 2340640, 146290, 1, 3, 0.0, true] |
| vllm_running_8192 | 69 | 112 |

## fixture:Qwen_Qwen3-Next-80B-A3B-Instruct.json (hybrid)

| figure | before | after |
|---|---|---|
| conv_kv_size | 49152 | 12288 |
| conv_state_bytes | 0 | 77266944 |
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
| genz_prefill_kv_mb | 384.0 | 169.6875 |
| genz_prefill_weights_mb | 150467.0 | 151486.390625 |
| prefix_kv_bytes_per_token | 98304 | 24576 |
| serving_bytes_per_token | 98304 | 24576 |
| serving_requests_8gib_1k | 85 | 83 |
| serving_state_bytes | 0 | 77266944 |

## fixture:Qwen_Qwen3.5-4B.json (multimodal)

| figure | before | after |
|---|---|---|
| conv_kv_size | 262144 | 16384 |
| conv_state_bytes | 0 | 51511296 |
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
| genz_prefill_kv_mb | 512.0 | 177.125 |
| genz_prefill_weights_mb | 8345.0 | 9074.023438 |
| prefix_kv_bytes_per_token | 131072 | 32768 |
| serving_bytes_per_token | 131072 | 32768 |
| serving_requests_8gib_1k | 64 | 100 |
| serving_state_bytes | 0 | 51511296 |

## fixture:Qwen_Qwen3.6-27B.json (multimodal)

| figure | before | after |
|---|---|---|
| conv_kv_size | 262144 | 32768 |
| conv_layers | [0, 32] | [0, 64] |
| conv_state_bytes | 0 | 153944064 |
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
| genz_prefill_kv_mb | 1024.0 | 402.8125 |
| genz_prefill_weights_mb | 46450.0 | 50339.21875 |
| prefix_kv_bytes_per_token | 262144 | 65536 |
| serving_bytes_per_token | 262144 | 65536 |
| serving_requests_8gib_1k | 32 | 38 |
| serving_state_bytes | 0 | 153944064 |

## fixture:Zyphra_Zamba2-2.7B.json (hybrid)

| figure | before | after |
|---|---|---|
| conv_kv_size | 276480 | 92160 |
| conv_state_bytes | 0 | 72479232 |
| disagg_transfer_bytes_640 | 353894400 | 190444032 |
| eng_None_32768_kv_mb | 3019.89888 | 6039.79776 |
| eng_None_32768_params | 2446020000 | 2577092000 |
| eng_None_32768_state_mb | 0.0 | 72.479232 |
| eng_None_32768_total_mb | 8264.260416 | 11820.10912 |
| eng_None_4096_kv_mb | 377.48736 | 754.97472 |
| eng_None_4096_params | 2446020000 | 2577092000 |
| eng_None_4096_state_mb | 0.0 | 72.479232 |
| eng_None_4096_total_mb | 5445.688128 | 6258.462016 |
| eng_llamacpp_32768_kv_mb | 3019.89888 | 6039.79776 |
| eng_llamacpp_32768_params | 2446020000 | 2577092000 |
| eng_llamacpp_32768_state_mb | 0.0 | 72.479232 |
| eng_llamacpp_32768_total_mb | 8264.260416 | 11820.10912 |
| eng_llamacpp_4096_kv_mb | 377.48736 | 754.97472 |
| eng_llamacpp_4096_params | 2446020000 | 2577092000 |
| eng_llamacpp_4096_state_mb | 0.0 | 72.479232 |
| eng_llamacpp_4096_total_mb | 5445.688128 | 6258.462016 |
| genz_chunked_kv_mb | 2430.0 | 948.243164 |
| genz_decode_kv_mb | 35.859375 | 789.121582 |
| genz_decode_weights_mb | 9842.5 | 6064.521484 |
| genz_ffn_kinds | {"Dense": 54} | {"Dense": 9, "no_op": 45} |
| genz_kv_size | 276480 | 92160 |
| genz_layer_kinds | {"Mamba": 54} | {"MHA-global": 9, "Recurrent": 45} |
| genz_prefill_kv_mb | 35.859375 | 789.121582 |
| genz_prefill_weights_mb | 9998.75 | 6220.771484 |
| kv_gb@32768 | 3.01989888 | 6.03979776 |
| kv_gb@4096 | 0.37748736 | 0.75497472 |
| kvgeo_32768 | [92160.0, 9, 54, 1342208.0, 144, 0.0] | [184320.0, 9, 54, 1342208.0, 80, 0.0] |
| kvgeo_4096 | [92160.0, 9, 54, 1342208.0, 144, 0.0] | [184320.0, 9, 54, 1342208.0, 80, 0.0] |
| prefix_kv_bytes_per_token | 552960 | 184320 |
| report_kv_gb | 0.37748736 | 0.75497472 |
| report_params | 2446020000 | 2577092000 |
| report_total_gb | 5.643392832 | 6.479317619 |
| report_weight_gb | 4.89204 | 5.154184 |
| serving_bytes_per_token | 552960 | 184320 |
| serving_requests_8gib_1k | 15 | 32 |
| serving_state_bytes | 0 | 72479232 |
| sim_max_running | 376 | 240 |
| slice_mib_16x4352 | [16896, true] | [27136, true] |
| vllm_limits | [5261, 757584, 5261, 1, 0, 79626240.0, true] | [4682, 374560, 4682, 1, 0, 88473600.0, true] |
| vllm_running_1024 | 350 | 212 |
| vllm_running_8192 | 80 | 41 |

## fixture:ai21labs_Jamba-v0.1.json (hybrid)

| figure | before | after |
|---|---|---|
| conv_kv_size | 65536 | 8192 |
| conv_state_bytes | 0 | 16056320 |
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
| genz_prefill_error | "ValueError: More than 1 unique layers not supported. Work in progress" | null |
| genz_prefill_kv_mb | null | 79.3125 |
| genz_prefill_weights_mb | null | 98362.0625 |
| prefix_kv_bytes_per_token | 131072 | 16384 |
| serving_bytes_per_token | 131072 | 16384 |
| serving_requests_8gib_1k | 64 | 261 |
| serving_state_bytes | 0 | 16056320 |

## fixture:bart_transformers-5.12-default.json (encoder-decoder)

| figure | before | after |
|---|---|---|
| conv_kv_size | 98304 | 49152 |
| disagg_transfer_bytes_640 | null | 62914560 |
| eng_None_32768_kv_mb | 1758.068736 | 3294.953472 |
| eng_None_32768_total_mb | 2718.961254 | 4255.84599 |
| eng_None_4096_kv_mb | 348.782592 | 476.381184 |
| eng_None_4096_total_mb | 1239.210803 | 1366.809395 |
| eng_llamacpp_32768_kv_mb | 1758.068736 | 3294.953472 |
| eng_llamacpp_32768_total_mb | 2718.961254 | 4255.84599 |
| eng_llamacpp_4096_kv_mb | 348.782592 | 476.381184 |
| eng_llamacpp_4096_total_mb | 1239.210803 | 1366.809395 |
| genz_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'architectures', 'attention_dropout', 'bos_token_id', 'chun" | null |
| genz_chunked_kv_mb | null | 432.0 |
| genz_decode_kv_mb | null | 384.0 |
| genz_decode_weights_mb | null | 434.173828 |
| genz_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'architectures', 'attention_dropout', 'bos_token_id', 'chun" | null |
| genz_ffn_kinds | null | {"Dense": 12} |
| genz_kv_size | null | 49152 |
| genz_layer_kinds | null | {"MHA-global": 12} |
| genz_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'architectures', 'attention_dropout', 'bos_token_id', 'chun" | null |
| genz_prefill_kv_mb | null | 576.0 |
| genz_prefill_weights_mb | null | 868.347656 |
| kv_cross_layers | 0 | 12 |
| kv_full_layers | 0 | 12 |
| kv_gb@32768 | 0.0 | 3.221225472 |
| kv_gb@4096 | 0.0 | 0.402653184 |
| kvgeo_32768 | [0.0, 1, 0, 0.0, 16, 0.0] | [98304.0, 12, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [0.0, 1, 0, 0.0, 16, 0.0] | [98304.0, 12, 0, 0.0, 16, 0.0] |
| prefix_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'architectures', 'a" | null |
| prefix_kv_bytes_per_token | null | 98304 |
| report_kv_gb | 0.348782592 | 0.476381184 |
| report_total_gb | 1.325056983 | 1.478175293 |
| serving_bytes_per_token | null | 98304 |
| serving_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'architectures', 'attention_dropout', 'bos_token_id', 'chun" | null |
| serving_requests_8gib_1k | null | 85 |
| serving_state_bytes | null | 0 |
| slice_mib_16x4352 | [10240, true] | [14080, true] |
| vllm_limits | [94617, 1513872, 94617, 1, 0, 0.0, true] | [94617, 1513872, 94617, 1, 0, 50331648.0, true] |
| vllm_running_1024 | 1024 | 657 |
| vllm_running_8192 | 179 | 159 |

## fixture:deepseek-ai_DeepSeek-V3.2-Exp.json (decoder-only)

| figure | before | after |
|---|---|---|
| conv_kv_size | 874496 | 35136 |
| genz_chunked_kv_mb | 7686.0 | 171.5625 |
| genz_decode_weights_mb | 1335494.375 | 1273239.875 |
| genz_ffn_kinds | {"MoE": 61} | {"Dense": 3, "MoE": 58} |
| genz_prefill_weights_mb | 1337261.875 | 1275007.375 |
| prefix_kv_bytes_per_token | 1748992 | 70272 |
| serving_bytes_per_token | 1748992 | 70272 |
| serving_requests_8gib_1k | 4 | 119 |

## fixture:deepseek-v3.json (decoder-only)

| figure | before | after |
|---|---|---|
| conv_kv_size | 874496 | 35136 |
| genz_chunked_kv_mb | 7686.0 | 171.5625 |
| genz_decode_weights_mb | 1335494.375 | 1273239.875 |
| genz_ffn_kinds | {"MoE": 61} | {"Dense": 3, "MoE": 58} |
| genz_prefill_weights_mb | 1337261.875 | 1275007.375 |
| prefix_kv_bytes_per_token | 1748992 | 70272 |
| serving_bytes_per_token | 1748992 | 70272 |
| serving_requests_8gib_1k | 4 | 119 |

## fixture:deepseek_v2_lite_chat_config.json (decoder-only)

| figure | before | after |
|---|---|---|
| conv_kv_size | 110592 | 15552 |
| genz_chunked_kv_mb | 972.0 | 75.9375 |
| genz_decode_kv_mb | null | 67.5 |
| genz_decode_weights_mb | null | 28699.5 |
| genz_error | "ValueError: cannot convert float NaN to integer" | null |
| genz_ffn_kinds | {"MoE": 27} | {"Dense": 1, "MoE": 26} |
| genz_prefill_error | "ValueError: cannot convert float NaN to integer" | null |
| genz_prefill_kv_mb | null | 67.5 |
| genz_prefill_weights_mb | null | 29099.5 |
| prefix_kv_bytes_per_token | 221184 | 31104 |
| serving_bytes_per_token | 221184 | 31104 |
| serving_requests_8gib_1k | 37 | 269 |

## fixture:gpt-oss-20b-bf16.json (decoder-only)

| figure | before | after |
|---|---|---|
| genz_chunked_kv_mb | 216.0 | 123.0 |
| slice_mib_16x4352 | [49152, true] | [49408, true] |
| vllm_limits | [85423, 1366768, 85423, 1, 0, 0.0, true] | [85423, 1366768, 85423, 1, 1, 0.0, true] |
| vllm_running_1024 | 1024 | 959 |
| vllm_running_8192 | 161 | 159 |

## fixture:gpt-oss-20b.json (decoder-only)

| figure | before | after |
|---|---|---|
| genz_chunked_kv_mb | 216.0 | 123.0 |
| slice_mib_16x4352 | [20736, true] | [20992, true] |
| vllm_limits | [155360, 2485760, 155360, 1, 0, 0.0, true] | [155360, 2485760, 155360, 1, 1, 0.0, true] |
| vllm_running_8192 | 294 | 289 |

## fixture:ibm-ai-platform_Bamba-9B-v2.json (hybrid)

| figure | before | after |
|---|---|---|
| conv_kv_size | 65536 | 6144 |
| conv_state_bytes | 0 | 123104768 |
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
| genz_prefill_kv_mb | 66.0 | 165.401855 |
| genz_prefill_weights_mb | 19284.0 | 18653.945312 |
| prefix_kv_bytes_per_token | 131072 | 12288 |
| serving_bytes_per_token | 131072 | 12288 |
| serving_requests_8gib_1k | 64 | 63 |
| serving_state_bytes | 0 | 123104768 |

## fixture:ibm-granite_granite-4.0-h-small.json (hybrid)

| figure | before | after |
|---|---|---|
| conv_kv_size | 81920 | 8192 |
| conv_state_bytes | 0 | 152819712 |
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
| genz_prefill_kv_mb | 82.5 | 209.740234 |
| genz_prefill_weights_mb | 61590.5 | 60774.15625 |
| prefix_kv_bytes_per_token | 163840 | 16384 |
| serving_bytes_per_token | 163840 | 16384 |
| serving_requests_8gib_1k | 51 | 50 |
| serving_state_bytes | 0 | 152819712 |

## fixture:inclusionAI_Ling-flash-2.0.json (decoder-only)

| figure | before | after |
|---|---|---|
| genz_decode_weights_mb | 200204.0 | 194274.0 |
| genz_ffn_kinds | {"MoE": 32} | {"Dense": 1, "MoE": 31} |
| genz_prefill_weights_mb | 201432.0 | 195502.0 |

## fixture:measured_checkpoints/openai_gpt-oss-20b/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| genz_chunked_kv_mb | 216.0 | 123.0 |
| slice_mib_16x4352 | [20736, true] | [20992, true] |
| vllm_limits | [155360, 2485760, 155360, 1, 0, 0.0, true] | [155360, 2485760, 155360, 1, 1, 0.0, true] |
| vllm_running_8192 | 294 | 289 |

## fixture:microsoft_Phi-4-mini-flash-reasoning.json (hybrid)

| figure | before | after |
|---|---|---|
| conv_kv_size | 81920 | 40960 |
| conv_state_bytes | 0 | 5734400 |
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
| genz_prefill_kv_mb | 640.0 | 62.96875 |
| genz_prefill_weights_mb | 7953.75 | 8612.34375 |
| kv_free_layers | 0 | 16 |
| kv_full_layers | 0 | 1 |
| kv_gb@32768 | 0.08388608 | 2.68435456 |
| kv_gb@4096 | 0.08388608 | 0.33554432 |
| kv_sliding_layers | 32 | 15 |
| kvgeo_32768 | [0.0, 32, 0, 0.0, 16, 0.0] | [81920.0, 1, 16, 358400.0, 16, 0.0] |
| kvgeo_4096 | [0.0, 32, 0, 0.0, 16, 0.0] | [81920.0, 1, 16, 358400.0, 16, 0.0] |
| prefix_kv_bytes_per_token | 163840 | 81920 |
| report_kv_gb | 0.08388608 | 0.33554432 |
| report_params | 4329313280 | 4674606080 |
| report_state_gb | 0.0 | 0.0057344 |
| report_total_gb | 9.091478733 | 10.090935501 |
| report_weight_gb | 8.65862656 | 9.34921216 |
| serving_bytes_per_token | 163840 | 81920 |
| serving_requests_8gib_1k | 51 | 95 |
| serving_state_bytes | 0 | 5734400 |
| sim_max_running | 389 | 400 |
| slice_mib_16x4352 | [24832, true] | [19200, true] |
| state_gb | 0.0 | 0.0057344 |
| vllm_limits | [25739, 411824, 25739, 1, 0, 0.0, true] | [10138, 811040, 10138, 1, 0, 6553600.0, true] |
| vllm_running_1024 | 321 | 596 |
| vllm_running_8192 | 48 | 94 |

## fixture:mistralai_Mamba-Codestral-7B-v0.1.json (state-space)

| figure | before | after |
|---|---|---|
| conv_kv_size | 262144 | 0 |
| conv_state_bytes | 0 | 272367616 |
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
| prefix_kv_bytes_per_token | 524288 | 0 |
| serving_bytes_per_token | 524288 | 0 |
| serving_requests_8gib_1k | 16 | 31 |
| serving_state_bytes | 0 | 272367616 |

## fixture:moonshotai_Kimi-Linear-48B-A3B-Instruct.json (hybrid)

| figure | before | after |
|---|---|---|
| conv_kv_size | 124416 | 4032 |
| conv_state_bytes | 0 | 43417600 |
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
| genz_prefill_error | "ValueError: cannot convert float NaN to integer" | null |
| genz_prefill_kv_mb | null | 58.90625 |
| genz_prefill_weights_mb | null | 93340.068359 |
| prefix_kv_bytes_per_token | 248832 | 8064 |
| serving_bytes_per_token | 248832 | 8064 |
| serving_requests_8gib_1k | 33 | 166 |
| serving_state_bytes | 0 | 43417600 |

## fixture:nari-labs_Dia-1.6B-0626.json (encoder-decoder)

| figure | before | after |
|---|---|---|
| conv_kv_size | 131072 | 18432 |
| conv_layers | [16, 16] | [12, 18] |
| conv_state_bytes | 0 | 150994944 |
| disagg_transfer_bytes_640 | 23592960 | 174587904 |
| eng_None_32768_kv_mb | 701.80864 | 1432.682496 |
| eng_None_32768_params | 808394240 | 1763025664 |
| eng_None_32768_total_mb | 2484.691558 | 5124.828262 |
| eng_None_4096_kv_mb | 114.60608 | 375.717888 |
| eng_None_4096_params | 808394240 | 1763025664 |
| eng_None_4096_total_mb | 1814.441779 | 3984.816435 |
| eng_llamacpp_32768_kv_mb | 701.80864 | 1432.682496 |
| eng_llamacpp_32768_params | 808394240 | 1763025664 |
| eng_llamacpp_32768_total_mb | 2484.691558 | 5124.828262 |
| eng_llamacpp_4096_kv_mb | 114.60608 | 375.717888 |
| eng_llamacpp_4096_params | 808394240 | 1763025664 |
| eng_llamacpp_4096_total_mb | 1814.441779 | 3984.816435 |
| facts_layers | [32, 0, 0, null] | [18, 0, 0, null] |
| genz_chunked_kv_mb | 162.0 | 450.0 |
| genz_decode_kv_mb | 144.0 | 288.0 |
| genz_decode_weights_mb | 2092.015625 | 2380.015625 |
| genz_prefill_kv_mb | 144.0 | 384.0 |
| genz_prefill_weights_mb | 2096.03125 | 3008.03125 |
| kv_cross_layers | 0 | 18 |
| kv_full_layers | 0 | 18 |
| kv_gb@32768 | 0.0 | 1.358954496 |
| kv_gb@4096 | 0.0 | 0.301989888 |
| kvgeo_32768 | [0.0, 1, 0, 0.0, 16, 0.0] | [36864.0, 18, 0, 0.0, 16, 150994944.0] |
| kvgeo_4096 | [0.0, 1, 0, 0.0, 16, 0.0] | [36864.0, 18, 0, 0.0, 16, 150994944.0] |
| report_kv_gb | 0.11460608 | 0.375717888 |
| report_params | 808394240 | 1763025664 |
| report_total_gb | 1.856430039 | 4.078566257 |
| report_weight_gb | 1.61678848 | 3.526051328 |
| serving_requests_8gib_1k | 227 | 45 |
| serving_state_bytes | 0 | 150994944 |
| sim_max_running | 72 | 372 |
| slice_mib_16x4352 | [81582, false] | [15360, true] |
| vllm_limits | [4702, 75232, 4702, 1, 0, 0.0, true] | [120000, 1920000, 120000, 1, 0, 150994944.0, true] |
| vllm_running_1024 | 58 | 357 |
| vllm_running_8192 | 8 | 153 |

## fixture:nvidia_Hymba-1.5B-Base.json (hybrid)

| figure | before | after |
|---|---|---|
| conv_state_bytes | 0 | 7168000 |
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
| genz_prefill_kv_mb | 3.90625 | 58.085938 |
| genz_prefill_weights_mb | 2790.631104 | 3170.123291 |
| serving_requests_8gib_1k | 204 | 174 |
| serving_state_bytes | 0 | 7168000 |

## fixture:nvidia_NVIDIA-Nemotron-Nano-9B-v2.json (hybrid)

| figure | before | after |
|---|---|---|
| conv_kv_size | 114688 | 8192 |
| conv_state_bytes | 0 | 143548416 |
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
| genz_prefill_kv_mb | 63.164062 | 200.898438 |
| genz_prefill_weights_mb | 24045.0 | 16952.672119 |
| prefix_kv_bytes_per_token | 229376 | 16384 |
| serving_bytes_per_token | 229376 | 16384 |
| serving_requests_8gib_1k | 36 | 53 |
| serving_state_bytes | 0 | 143548416 |

## fixture:nvidia_Nemotron-H-8B-Base-8K.json (hybrid)

| figure | before | after |
|---|---|---|
| conv_kv_size | 106496 | 8192 |
| conv_state_bytes | 0 | 102137856 |
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
| genz_prefill_kv_mb | 107.25 | 161.40625 |
| genz_prefill_weights_mb | 30128.0 | 15450.8125 |
| prefix_kv_bytes_per_token | 212992 | 16384 |
| serving_bytes_per_token | 212992 | 16384 |
| serving_requests_8gib_1k | 39 | 72 |
| serving_state_bytes | 0 | 102137856 |

## fixture:onnx-community_cohere-transcribe-03-2026-ONNX.json (encoder-decoder)

| figure | before | after |
|---|---|---|
| conv_kv_size | 8192 | 16384 |
| conv_layers | [4, 4] | [48, 8] |
| conv_state_bytes | 0 | 163840000 |
| disagg_transfer_bytes_640 | 20971520 | 184811520 |
| eng_None_32768_kv_mb | 1122.893824 | 1606.221824 |
| eng_None_32768_params | 576643072 | 580837376 |
| eng_None_32768_total_mb | 2417.108582 | 2908.82519 |
| eng_None_4096_kv_mb | 183.369728 | 666.697728 |
| eng_None_4096_params | 576643072 | 580837376 |
| eng_None_4096_total_mb | 1407.120179 | 1898.836787 |
| eng_llamacpp_32768_kv_mb | 1122.893824 | 1606.221824 |
| eng_llamacpp_32768_params | 576643072 | 580837376 |
| eng_llamacpp_32768_total_mb | 2417.108582 | 2908.82519 |
| eng_llamacpp_4096_kv_mb | 183.369728 | 666.697728 |
| eng_llamacpp_4096_params | 576643072 | 580837376 |
| eng_llamacpp_4096_total_mb | 1407.120179 | 1898.836787 |
| genz_chunked_kv_mb | 144.0 | 456.5 |
| genz_decode_kv_mb | 128.0 | 284.25 |
| genz_decode_weights_mb | 224.0 | 256.0 |
| genz_prefill_kv_mb | 128.0 | 1456.125 |
| genz_prefill_weights_mb | 256.0 | 2128.0 |
| kv_cross_layers | 0 | 8 |
| kv_gb@32768 | 1.073741824 | 1.237581824 |
| kv_gb@4096 | 0.134217728 | 0.298057728 |
| kvgeo_32768 | [32768.0, 8, 0, 0.0, 16, 0.0] | [32768.0, 8, 0, 0.0, 16, 163840000.0] |
| kvgeo_4096 | [32768.0, 8, 0, 0.0, 16, 0.0] | [65536.0, 8, 0, 0.0, 16, 163840000.0] |
| report_kv_gb | 0.183369728 | 0.666697728 |
| report_params | 576643072 | 580837376 |
| report_total_gb | 1.459883786 | 2.048726794 |
| report_weight_gb | 1.153286144 | 1.161674752 |
| serving_requests_8gib_1k | 256 | 43 |
| serving_state_bytes | 0 | 163840000 |
| sim_max_running | 400 | 368 |
| slice_mib_16x4352 | [8704, true] | [12800, true] |
| vllm_limits | [141755, 2268080, 141755, 1, 0, 0.0, true] | [139835, 2237360, 139835, 1, 0, 163840000.0, true] |
| vllm_running_1024 | 1024 | 356 |
| vllm_running_8192 | 268 | 166 |

## fixture:onnx-community_moonshine-base-ONNX.json (encoder-decoder)

| figure | before | after |
|---|---|---|
| conv_kv_size | 3328 | 13312 |
| conv_layers | [4, 4] | [8, 8] |
| disagg_transfer_bytes_640 | 8519680 | 17039360 |
| eng_None_32768_kv_mb | 456.175616 | 892.383232 |
| eng_None_32768_total_mb | 753.298125 | 1189.505741 |
| eng_None_4096_kv_mb | 74.493952 | 129.019904 |
| eng_None_4096_total_mb | 326.63255 | 381.158502 |
| eng_llamacpp_32768_kv_mb | 456.175616 | 892.383232 |
| eng_llamacpp_32768_total_mb | 753.298125 | 1189.505741 |
| eng_llamacpp_4096_kv_mb | 74.493952 | 129.019904 |
| eng_llamacpp_4096_total_mb | 326.63255 | 381.158502 |
| genz_chunked_kv_mb | 58.5 | 117.0 |
| genz_decode_kv_mb | 52.0 | 104.0 |
| genz_decode_weights_mb | 57.6875 | 62.96875 |
| genz_kv_size | 6656 | 13312 |
| genz_prefill_kv_mb | 52.0 | 156.0 |
| genz_prefill_weights_mb | 83.6875 | 125.9375 |
| kv_cross_layers | 0 | 8 |
| kv_gb@32768 | 0.436207616 | 0.872415232 |
| kv_gb@4096 | 0.054525952 | 0.109051904 |
| kvgeo_32768 | [13312.0, 8, 0, 0.0, 16, 0.0] | [26624.0, 8, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [13312.0, 8, 0, 0.0, 16, 0.0] | [26624.0, 8, 0, 0.0, 16, 0.0] |
| prefix_kv_bytes_per_token | 13312 | 26624 |
| report_kv_gb | 0.074493952 | 0.129019904 |
| report_total_gb | 0.351430523 | 0.416861665 |
| serving_bytes_per_token | 13312 | 26624 |
| serving_requests_8gib_1k | 630 | 315 |
| slice_mib_16x4352 | [6656, true] | [7680, true] |
| vllm_limits | [351373, 5621968, 351373, 1, 0, 0.0, true] | [351373, 5621968, 351373, 1, 0, 6815744.0, true] |
| vllm_running_8192 | 665 | 627 |

## fixture:onnx-community_whisper-large-v3-turbo.json (encoder-decoder)

| figure | before | after |
|---|---|---|
| conv_kv_size | 32768 | 10240 |
| conv_state_bytes | 0 | 30720000 |
| disagg_transfer_bytes_640 | null | 43827200 |
| facts_layers | [32, 0, 0, null] | [4, 0, 0, null] |
| genz_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'apply_spec_augment', 'architectures', 'attention_dropout'," | null |
| genz_chunked_kv_mb | null | 148.59375 |
| genz_decode_kv_mb | null | 109.296875 |
| genz_decode_weights_mb | null | 301.625977 |
| genz_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'apply_spec_augment', 'architectures', 'attention_dropout'," | null |
| genz_ffn_kinds | null | {"Dense": 4} |
| genz_kv_size | null | 10240 |
| genz_layer_kinds | null | {"MHA-global": 4} |
| genz_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'apply_spec_augment', 'architectures', 'attention_dropout'," | null |
| genz_prefill_kv_mb | null | 343.671875 |
| genz_prefill_weights_mb | null | 1653.251953 |
| kv_cross_layers | 0 | 4 |
| kv_full_layers | 32 | 4 |
| kv_gb@32768 | 0.0 | 0.70180864 |
| kv_gb@4096 | 0.0 | 0.11460608 |
| kvgeo_32768 | [0.0, 32, 0, 0.0, 16, 0.0] | [20480.0, 4, 0, 0.0, 16, 30720000.0] |
| kvgeo_4096 | [0.0, 32, 0, 0.0, 16, 0.0] | [20480.0, 4, 0, 0.0, 16, 30720000.0] |
| prefix_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'apply_spec_augment" | null |
| prefix_kv_bytes_per_token | null | 20480 |
| serving_bytes_per_token | null | 20480 |
| serving_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'apply_spec_augment', 'architectures', 'attention_dropout'," | null |
| serving_requests_8gib_1k | null | 166 |
| serving_state_bytes | null | 30720000 |
| slice_mib_16x4352 | [22272, true] | [9472, true] |
| vllm_limits | [27344, 437504, 27344, 1, 0, 0.0, true] | [224354, 3589664, 224354, 1, 0, 30720000.0, true] |
| vllm_running_1024 | 341 | 1024 |
| vllm_running_8192 | 51 | 360 |

## fixture:openai_gpt-oss-120b.json (decoder-only)

| figure | before | after |
|---|---|---|
| genz_chunked_kv_mb | 324.0 | 184.5 |
| sim_max_running | 250 | 220 |
| vllm_limits | [16363, 261808, 16363, 1, 0, 0.0, true] | [16363, 261808, 16363, 1, 1, 0.0, true] |
| vllm_running_1024 | 204 | 183 |

## fixture:qwen3.8-27b-fp8.json (multimodal)

| figure | before | after |
|---|---|---|
| conv_kv_size | 262144 | 32768 |
| conv_layers | [0, 32] | [0, 64] |
| conv_state_bytes | 0 | 153944064 |
| disagg_transfer_bytes_640 | 167772160 | 195887104 |
| eng_None_32768_state_mb | 0.0 | 153.944064 |
| eng_None_32768_total_mb | 34685.072013 | 34839.016077 |
| eng_None_4096_state_mb | 0.0 | 153.944064 |
| eng_None_4096_total_mb | 32312.77367 | 32466.717734 |
| eng_llamacpp_32768_state_mb | 0.0 | 153.944064 |
| eng_llamacpp_32768_total_mb | 34685.072013 | 34839.016077 |
| eng_llamacpp_4096_state_mb | 0.0 | 153.944064 |
| eng_llamacpp_4096_total_mb | 32312.77367 | 32466.717734 |
| genz_chunked_kv_mb | 1152.0 | 581.625 |
| genz_decode_kv_mb | 1024.0 | 402.8125 |
| genz_decode_weights_mb | 44025.0 | 47914.21875 |
| genz_kv_size | 131072 | 32768 |
| genz_layer_kinds | {"MHA-global": 64} | {"MHA-global": 16, "Recurrent": 48} |
| genz_prefill_kv_mb | 1024.0 | 402.8125 |
| genz_prefill_weights_mb | 46450.0 | 50339.21875 |
| prefix_kv_bytes_per_token | 262144 | 65536 |
| serving_bytes_per_token | 262144 | 65536 |
| serving_requests_8gib_1k | 32 | 38 |
| serving_state_bytes | 0 | 153944064 |

## fixture:qwen3.8-27b-w4a16.json (multimodal)

| figure | before | after |
|---|---|---|
| conv_kv_size | 262144 | 32768 |
| conv_layers | [0, 32] | [0, 64] |
| conv_state_bytes | 0 | 153944064 |
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
| genz_prefill_kv_mb | 1024.0 | 402.8125 |
| genz_prefill_weights_mb | 46450.0 | 50339.21875 |
| prefix_kv_bytes_per_token | 262144 | 65536 |
| serving_bytes_per_token | 262144 | 65536 |
| serving_requests_8gib_1k | 32 | 38 |
| serving_state_bytes | 0 | 153944064 |

## fixture:qwen3.8-27b.json (multimodal)

| figure | before | after |
|---|---|---|
| conv_kv_size | 262144 | 32768 |
| conv_layers | [0, 32] | [0, 64] |
| conv_state_bytes | 0 | 153944064 |
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
| genz_prefill_kv_mb | 1024.0 | 402.8125 |
| genz_prefill_weights_mb | 46450.0 | 50339.21875 |
| prefix_kv_bytes_per_token | 262144 | 65536 |
| serving_bytes_per_token | 262144 | 65536 |
| serving_requests_8gib_1k | 32 | 38 |
| serving_state_bytes | 0 | 153944064 |

## fixture:qwen3_8_27b_config.json (multimodal)

| figure | before | after |
|---|---|---|
| conv_kv_size | 262144 | 32768 |
| conv_layers | [0, 32] | [0, 64] |
| conv_state_bytes | 0 | 153944064 |
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
| genz_prefill_kv_mb | 1024.0 | 402.8125 |
| genz_prefill_weights_mb | 46450.0 | 50339.21875 |
| prefix_kv_bytes_per_token | 262144 | 65536 |
| serving_bytes_per_token | 262144 | 65536 |
| serving_requests_8gib_1k | 32 | 38 |
| serving_state_bytes | 0 | 153944064 |

## fixture:state-spaces_mamba-2.8b-hf.json (state-space)

| figure | before | after |
|---|---|---|
| conv_kv_size | 327680 | 0 |
| conv_state_bytes | 0 | 22937600 |
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
| prefix_kv_bytes_per_token | 655360 | 0 |
| serving_bytes_per_token | 655360 | 0 |
| serving_requests_8gib_1k | 12 | 374 |
| serving_state_bytes | 0 | 22937600 |
| slice_mib_16x4352 | [62208, true] | [62976, true] |

## fixture:t5_flan-xl-geometry.json (encoder-decoder)

| figure | before | after |
|---|---|---|
| conv_kv_size | 131072 | 196608 |
| conv_layers | [16, 16] | [24, 24] |
| disagg_transfer_bytes_640 | null | 251658240 |
| eng_None_32768_kv_mb | 1118.69952 | 13179.813888 |
| eng_None_32768_params | 1911775232 | 3002679296 |
| eng_None_32768_total_mb | 5264.372531 | 19507.295027 |
| eng_None_4096_kv_mb | 182.84544 | 1905.524736 |
| eng_None_4096_params | 1911775232 | 3002679296 |
| eng_None_4096_total_mb | 4167.457178 | 8071.944602 |
| eng_llamacpp_32768_kv_mb | 1118.69952 | 13179.813888 |
| eng_llamacpp_32768_params | 1911775232 | 3002679296 |
| eng_llamacpp_32768_total_mb | 5264.372531 | 19507.295027 |
| eng_llamacpp_4096_kv_mb | 182.84544 | 1905.524736 |
| eng_llamacpp_4096_params | 1911775232 | 3002679296 |
| eng_llamacpp_4096_total_mb | 4167.457178 | 8071.944602 |
| genz_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'architectures', 'chunk_size_feed_forward', 'classifier_dropout', 'd_ff', 'd_kv', 'd_model', 'dense_act_" | null |
| genz_chunked_kv_mb | null | 1728.0 |
| genz_decode_kv_mb | null | 1536.0 |
| genz_decode_weights_mb | null | 2237.5 |
| genz_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'architectures', 'chunk_size_feed_forward', 'classifier_dropout', 'd_ff', 'd_kv', 'd_model', 'dense_act_" | null |
| genz_ffn_kinds | null | {"Dense": 24} |
| genz_kv_size | null | 196608 |
| genz_layer_kinds | null | {"MHA-global": 24} |
| genz_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'architectures', 'chunk_size_feed_forward', 'classifier_dropout', 'd_ff', 'd_kv', 'd_model', 'dense_act_" | null |
| genz_prefill_kv_mb | null | 2304.0 |
| genz_prefill_weights_mb | null | 4475.0 |
| kv_cross_layers | 0 | 24 |
| kv_gb@32768 | 0.0 | 12.884901888 |
| kv_gb@4096 | 0.0 | 1.610612736 |
| kvgeo_32768 | [0.0, 1, 0, 0.0, 16, 0.0] | [393216.0, 24, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [0.0, 1, 0, 0.0, 16, 0.0] | [393216.0, 24, 0, 0.0, 16, 0.0] |
| prefix_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'architectures', 'chunk_size_feed_forward', 'classifier_dropout'" | null |
| prefix_kv_bytes_per_token | null | 393216 |
| report_kv_gb | 0.18284544 | 1.905524736 |
| report_params | 1911775232 | 3002679296 |
| report_total_gb | 4.24007852 | 8.489101804 |
| report_weight_gb | 3.823550464 | 6.005358592 |
| serving_bytes_per_token | null | 393216 |
| serving_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'architectures', 'chunk_size_feed_forward', 'classifier_dropout', 'd_ff', 'd_kv', 'd_model', 'dense_act_" | null |
| serving_requests_8gib_1k | null | 21 |
| serving_state_bytes | null | 0 |
| sim_max_running | 331 | 69 |
| slice_mib_16x4352 | [27904, true] | [42496, true] |
| vllm_limits | [21856, 349696, 21856, 1, 0, 0.0, true] | [22144, 354304, 22144, 1, 0, 805306368.0, true] |
| vllm_running_1024 | 273 | 65 |
| vllm_running_8192 | 41 | 28 |

## fixture:tiiuae_Falcon-H1-34B-Instruct.json (hybrid)

| figure | before | after |
|---|---|---|
| conv_state_bytes | 0 | 304201728 |
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
| genz_prefill_kv_mb | 365.625 | 866.109375 |
| genz_prefill_weights_mb | 73680.0 | 64166.71875 |
| serving_requests_8gib_1k | 56 | 18 |
| serving_state_bytes | 0 | 304201728 |

## fixture:unknown-arch.json (decoder-only)

| figure | before | after |
|---|---|---|
| eng_None_32768_kv_mb | 0.0 | 3758.096384 |
| eng_None_32768_total_mb | 5326.41792 | 9084.514304 |
| eng_None_4096_kv_mb | 0.0 | 469.762048 |
| eng_None_4096_total_mb | 5074.75968 | 5544.521728 |
| eng_llamacpp_32768_kv_mb | 0.0 | 3758.096384 |
| eng_llamacpp_32768_total_mb | 5326.41792 | 9084.514304 |
| eng_llamacpp_4096_kv_mb | 0.0 | 469.762048 |
| eng_llamacpp_4096_total_mb | 5074.75968 | 5544.521728 |
| kv_free_layers | 28 | 0 |
| kv_full_layers | 0 | 28 |
| kv_gb@32768 | 0.0 | 3.758096384 |
| kv_gb@4096 | 0.0 | 0.469762048 |
| kv_source | "_layer_metadata" | "layer_plan:layer_types" |
| kvgeo_32768 | [0.0, 28, 0, 0.0, 16, 0.0] | [114688.0, 28, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [0.0, 28, 0, 0.0, 16, 0.0] | [114688.0, 28, 0, 0.0, 16, 0.0] |
| report_kv_gb | 0.0 | 0.469762048 |
| report_total_gb | 5.125091328 | 5.688805786 |

## fixture:unsloth_gemma-3-27b-it.json (multimodal)

| figure | before | after |
|---|---|---|
| conv_kv_size | 262144 | 253952 |
| conv_layers | [0, 32] | [0, 62] |
| facts_full_attention | true | false |
| facts_layers | [62, 0, 0, null] | [10, 52, 0, 1024] |
| genz_chunked_kv_mb | 2232.0 | 984.0 |
| genz_decode_kv_mb | 1984.0 | 736.0 |
| genz_layer_kinds | {"MHA-global": 62} | {"MHA-global": 10, "MHA-local": 52} |
| genz_prefill_kv_mb | 1984.0 | 736.0 |
| sim_max_running | 32 | 28 |
| slice_mib_16x4352 | [81582, false] | [80896, true] |
| vllm_limits | [2067, 33072, 2067, 1, 0, 0.0, false] | [12818, 205088, 12818, 1, 6, 0.0, true] |
| vllm_running_1024 | 25 | 27 |
| vllm_running_8192 | 3 | 13 |

## fixture:whisper-large-v3.json (encoder-decoder)

| figure | before | after |
|---|---|---|
| conv_kv_size | 262144 | 81920 |
| conv_state_bytes | 0 | 245760000 |
| disagg_transfer_bytes_640 | 335544320 | 350617600 |
| genz_chunked_kv_mb | 2304.0 | 1188.75 |
| genz_decode_kv_mb | 2048.0 | 874.375 |
| genz_decode_weights_mb | 10005.203125 | 1526.625977 |
| genz_kv_size | 262144 | 81920 |
| genz_prefill_kv_mb | 2048.0 | 1108.75 |
| genz_prefill_weights_mb | 10410.40625 | 3053.251953 |
| kv_cross_layers | 0 | 32 |
| kv_full_layers | 0 | 32 |
| kv_gb@32768 | 0.0 | 5.61446912 |
| kv_gb@4096 | 0.0 | 0.91684864 |
| kvgeo_32768 | [0.0, 1, 0, 0.0, 16, 0.0] | [163840.0, 32, 0, 0.0, 16, 245760000.0] |
| kvgeo_4096 | [0.0, 1, 0, 0.0, 16, 0.0] | [163840.0, 32, 0, 0.0, 16, 245760000.0] |
| prefix_kv_bytes_per_token | 524288 | 163840 |
| serving_bytes_per_token | 524288 | 163840 |
| serving_requests_8gib_1k | 16 | 20 |
| serving_state_bytes | 0 | 245760000 |
| sim_max_running | 400 | 173 |
| slice_mib_16x4352 | [65536, true] | [70144, true] |
| vllm_limits | [27594, 441504, 27594, 1, 0, 0.0, true] | [27594, 441504, 27594, 1, 0, 245760000.0, true] |
| vllm_running_1024 | 344 | 158 |
| vllm_running_8192 | 52 | 44 |

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
| conv_kv_size | 24576 | 6144 |
| conv_state_bytes | 0 | 19537920 |
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
| genz_prefill_kv_mb | 192.0 | 66.632812 |
| genz_prefill_weights_mb | 3572.0 | 4509.265625 |
| gguf_config_keys | ["head_dim", "hidden_size", "intermediate_size", "max_position_embeddings", "model_type", "num_attention_heads", "num_hidden_layers", "num_key_value_heads", "vocab_size"] | ["head_dim", "hidden_act", "hidden_size", "intermediate_size", "layer_types", "max_position_embeddings", "model_type", "num_attention_heads", "num_hidden_layers", "num_key_value_heads", "rope_theta", "ssm_conv_kernel", "ssm_inner_size", "ssm_n_groups", "ssm_state_size", "tie_word_embeddings", "time_step_rank", "vocab_size"] |
| kv_free_layers | 0 | 18 |
| kv_full_layers | 24 | 6 |
| kv_gb@32768 | 1.610612736 | 0.402653184 |
| kv_gb@4096 | 0.201326592 | 0.050331648 |
| kv_source | "uniform" | "layer_plan:layer_types" |
| kvgeo_32768 | [49152.0, 24, 0, 0.0, 16, 0.0] | [12288.0, 6, 18, 1085440.0, 544, 0.0] |
| kvgeo_4096 | [49152.0, 24, 0, 0.0, 16, 0.0] | [12288.0, 6, 18, 1085440.0, 544, 0.0] |
| model_type | "decoder-only" | "hybrid" |
| prefix_kv_bytes_per_token | 49152 | 12288 |
| report_kv_gb | 0.201326592 | 0.050331648 |
| report_params | 1901268992 | 1855758390 |
| report_state_gb | 0.0 | 0.01953792 |
| report_total_gb | 4.261562614 | 4.012792981 |
| report_weight_gb | 3.802537984 | 3.71151678 |
| serving_bytes_per_token | 49152 | 12288 |
| serving_requests_8gib_1k | 170 | 267 |
| serving_state_bytes | 0 | 19537920 |
| slice_mib_16x4352 | [14080, true] | [12032, true] |
| state_gb | 0.0 | 0.01953792 |
| vllm_limits | [88602, 1417632, 88602, 1, 0, 0.0, true] | [10493, 5708192, 10493, 1, 0, 20054016.0, true] |
| vllm_running_8192 | 167 | 552 |

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
| genz_prefill_weights_mb | 4183.59375 | 39878.4375 |
| gguf_config_keys | ["head_dim", "hidden_size", "intermediate_size", "max_position_embeddings", "model_type", "num_attention_heads", "num_hidden_layers", "num_key_value_heads", "vocab_size"] | ["head_dim", "hidden_act", "hidden_size", "intermediate_size", "max_position_embeddings", "model_type", "moe_intermediate_size", "n_routed_experts", "num_attention_heads", "num_experts", "num_experts_per_tok", "num_hidden_layers", "num_key_value_heads", "rope_theta", "tie_word_embeddings", "vocab_size"] |
| report_params | 1992044160 | 20908068480 |
| report_total_gb | 4.436318177 | 42.431441357 |
| report_weight_gb | 3.98408832 | 41.81613696 |
| slice_mib_16x4352 | [14336, true] | [52736, true] |
| vllm_limits | [88043, 1408688, 88043, 1, 0, 0.0, true] | [40956, 655296, 40956, 1, 0, 0.0, true] |
| vllm_running_1024 | 1024 | 511 |
| vllm_running_8192 | 166 | 77 |

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
| genz_prefill_weights_mb | 251.419922 | 305.419922 |
| gguf_config_keys | ["hidden_size", "intermediate_size", "max_position_embeddings", "model_type", "num_attention_heads", "num_hidden_layers", "num_key_value_heads", "vocab_size"] | ["hidden_act", "hidden_size", "intermediate_size", "max_position_embeddings", "model_type", "num_attention_heads", "num_hidden_layers", "num_key_value_heads", "rope_theta", "tie_word_embeddings", "vocab_size"] |
| report_params | 109986816 | 136725504 |
| report_total_gb | 0.500824228 | 0.554301604 |
| report_weight_gb | 0.219973632 | 0.273451008 |
| vllm_limits | [126489, 2023824, 126489, 1, 0, 0.0, true] | [126569, 2025104, 126569, 1, 0, 0.0, true] |

## synth:blip2-flan-t5-xl (multimodal)

| figure | before | after |
|---|---|---|
| conv_kv_size | 131072 | 196608 |
| conv_layers | [16, 16] | [24, 24] |
| disagg_transfer_bytes_640 | 23592960 | 251658240 |
| eng_None_32768_kv_mb | 0.0 | 12884.901888 |
| eng_None_32768_params | 1663986176 | 1538943488 |
| eng_None_32768_total_mb | 3656.150426 | 16290.966938 |
| eng_None_4096_kv_mb | 0.0 | 1610.612736 |
| eng_None_4096_params | 1663986176 | 1538943488 |
| eng_None_4096_total_mb | 3495.089152 | 4855.616512 |
| eng_llamacpp_32768_kv_mb | 0.0 | 12884.901888 |
| eng_llamacpp_32768_params | 1663986176 | 1538943488 |
| eng_llamacpp_32768_total_mb | 3656.150426 | 16290.966938 |
| eng_llamacpp_4096_kv_mb | 0.0 | 1610.612736 |
| eng_llamacpp_4096_params | 1663986176 | 1538943488 |
| eng_llamacpp_4096_total_mb | 3495.089152 | 4855.616512 |
| genz_chunked_kv_mb | 162.0 | 1728.0 |
| genz_decode_kv_mb | 144.0 | 1536.0 |
| genz_decode_weights_mb | 208.875 | 2237.5 |
| genz_ffn_kinds | {"Dense": 12} | {"Dense": 24} |
| genz_kv_size | 18432 | 196608 |
| genz_layer_kinds | {"MHA-global": 12} | {"MHA-global": 24} |
| genz_prefill_kv_mb | 144.0 | 2304.0 |
| genz_prefill_weights_mb | 255.75 | 4475.0 |
| kv_cross_layers | 0 | 24 |
| kv_gb@32768 | 0.0 | 12.884901888 |
| kv_gb@4096 | 0.0 | 1.610612736 |
| kvgeo_32768 | [0.0, 1, 0, 0.0, 16, 0.0] | [393216.0, 24, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [0.0, 1, 0, 0.0, 16, 0.0] | [393216.0, 24, 0, 0.0, 16, 0.0] |
| prefix_kv_bytes_per_token | 36864 | 393216 |
| report_kv_gb | 0.0 | 1.610612736 |
| report_params | 1663986176 | 1538943488 |
| report_total_gb | 3.528512512 | 5.211162419 |
| report_weight_gb | 3.327972352 | 3.077886976 |
| serving_bytes_per_token | 36864 | 393216 |
| serving_requests_8gib_1k | 227 | 21 |
| sim_max_running | 322 | 67 |
| slice_mib_16x4352 | [30208, true] | [44800, true] |
| vllm_limits | [21221, 339536, 21221, 1, 0, 0.0, true] | [21508, 344128, 21508, 1, 0, 805306368.0, true] |
| vllm_running_1024 | 265 | 64 |
| vllm_running_8192 | 40 | 27 |

## synth:chatglm3 (decoder-only)

| figure | before | after |
|---|---|---|
| conv_kv_size | 262144 | 229376 |
| conv_layers | [0, 32] | [0, 28] |
| disagg_transfer_bytes_640 | 335544320 | 293601280 |
| eng_None_32768_kv_mb | 6442.450944 | 15032.385536 |
| eng_None_32768_params | 2357735424 | 5226901504 |
| eng_None_32768_total_mb | 11938.062336 | 26266.329088 |
| eng_None_4096_kv_mb | 805.306368 | 1879.048192 |
| eng_None_4096_params | 2357735424 | 5226901504 |
| eng_None_4096_total_mb | 5910.847488 | 12722.921472 |
| eng_llamacpp_32768_kv_mb | 6442.450944 | 15032.385536 |
| eng_llamacpp_32768_params | 2357735424 | 5226901504 |
| eng_llamacpp_32768_total_mb | 11938.062336 | 26266.329088 |
| eng_llamacpp_4096_kv_mb | 805.306368 | 1879.048192 |
| eng_llamacpp_4096_params | 2357735424 | 5226901504 |
| eng_llamacpp_4096_total_mb | 5910.847488 | 12722.921472 |
| genz_chunked_kv_mb | 2304.0 | 2016.0 |
| genz_decode_kv_mb | 2048.0 | 1792.0 |
| genz_decode_weights_mb | 9850.0 | 8650.0 |
| genz_ffn_kinds | {"Dense": 32} | {"Dense": 28} |
| genz_kv_size | 262144 | 229376 |
| genz_layer_kinds | {"MHA-global": 32} | {"MHA-global": 28} |
| genz_prefill_kv_mb | 2048.0 | 1792.0 |
| genz_prefill_weights_mb | 10100.0 | 8900.0 |
| kv_gb@32768 | 6.442450944 | 15.032385536 |
| kv_gb@4096 | 0.805306368 | 1.879048192 |
| kvgeo_32768 | [458752.0, 1, 0, 0.0, 16, 0.0] | [458752.0, 28, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [458752.0, 1, 0, 0.0, 16, 0.0] | [458752.0, 28, 0, 0.0, 16, 0.0] |
| model_type | "unknown" | "decoder-only" |
| prefix_kv_bytes_per_token | 524288 | 458752 |
| report_kv_gb | 0.805306368 | 1.879048192 |
| report_params | 2357735424 | 5226901504 |
| report_total_gb | 6.149922816 | 13.176745165 |
| report_weight_gb | 4.715470848 | 10.453803008 |
| serving_bytes_per_token | 524288 | 458752 |
| serving_requests_8gib_1k | 16 | 18 |

## synth:deepseek-v2-lite-mla (decoder-only)

| figure | before | after |
|---|---|---|
| conv_kv_size | 110592 | 15552 |
| genz_chunked_kv_mb | 972.0 | 75.9375 |
| genz_decode_kv_mb | null | 67.5 |
| genz_decode_weights_mb | null | 28699.5 |
| genz_error | "ValueError: cannot convert float NaN to integer" | null |
| genz_ffn_kinds | {"MoE": 27} | {"Dense": 1, "MoE": 26} |
| genz_prefill_error | "ValueError: cannot convert float NaN to integer" | null |
| genz_prefill_kv_mb | null | 67.5 |
| genz_prefill_weights_mb | null | 29099.5 |
| prefix_kv_bytes_per_token | 221184 | 31104 |
| serving_bytes_per_token | 221184 | 31104 |
| serving_requests_8gib_1k | 37 | 269 |

## synth:enc-dec-llama-dec (encoder-decoder)

| figure | before | after |
|---|---|---|
| conv_kv_size | 131072 | 32768 |
| conv_layers | [16, 16] | [24, 16] |
| disagg_transfer_bytes_640 | 20971520 | 41943040 |
| eng_None_32768_kv_mb | 701.80864 | 2294.939648 |
| eng_None_32768_params | 808394240 | 1649767168 |
| eng_None_32768_total_mb | 2484.691558 | 5760.568422 |
| eng_None_4096_kv_mb | 114.60608 | 415.891456 |
| eng_None_4096_params | 808394240 | 1649767168 |
| eng_None_4096_total_mb | 1814.441779 | 3798.473011 |
| eng_llamacpp_32768_kv_mb | 701.80864 | 2294.939648 |
| eng_llamacpp_32768_params | 808394240 | 1649767168 |
| eng_llamacpp_32768_total_mb | 2484.691558 | 5760.568422 |
| eng_llamacpp_4096_kv_mb | 114.60608 | 415.891456 |
| eng_llamacpp_4096_params | 808394240 | 1649767168 |
| eng_llamacpp_4096_total_mb | 1814.441779 | 3798.473011 |
| facts_layers | [32, 0, 0, null] | [16, 0, 0, null] |
| genz_chunked_kv_mb | 144.0 | 288.0 |
| genz_decode_kv_mb | 128.0 | 256.0 |
| genz_decode_weights_mb | 1981.0 | 2237.0 |
| genz_kv_size | 16384 | 32768 |
| genz_prefill_kv_mb | 128.0 | 640.0 |
| genz_prefill_weights_mb | 2106.0 | 3162.0 |
| kv_cross_layers | 0 | 16 |
| kv_full_layers | 0 | 16 |
| kv_gb@32768 | 0.0 | 2.147483648 |
| kv_gb@4096 | 0.0 | 0.268435456 |
| kvgeo_32768 | [0.0, 1, 0, 0.0, 16, 0.0] | [65536.0, 16, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [0.0, 1, 0, 0.0, 16, 0.0] | [65536.0, 16, 0, 0.0, 16, 0.0] |
| prefix_kv_bytes_per_token | 32768 | 65536 |
| report_kv_gb | 0.11460608 | 0.415891456 |
| report_params | 808394240 | 1649767168 |
| report_total_gb | 1.856430039 | 3.900257546 |
| report_weight_gb | 1.61678848 | 3.299534336 |
| serving_bytes_per_token | 32768 | 65536 |
| serving_requests_8gib_1k | 256 | 128 |
| sim_max_running | 72 | 400 |
| slice_mib_16x4352 | [81582, false] | [16128, true] |
| vllm_limits | [4702, 75232, 4702, 1, 0, 0.0, true] | [133835, 2141360, 133835, 1, 0, 134217728.0, true] |
| vllm_running_1024 | 58 | 398 |
| vllm_running_8192 | 8 | 170 |

## synth:florence2 (multimodal)

| figure | before | after |
|---|---|---|
| conv_kv_size | 131072 | 49152 |
| conv_layers | [16, 16] | [12, 12] |
| disagg_transfer_bytes_640 | null | 62914560 |
| eng_None_32768_kv_mb | 0.0 | 3221.225472 |
| eng_None_32768_total_mb | 804.21417 | 4025.439642 |
| eng_None_4096_kv_mb | 0.0 | 402.653184 |
| eng_None_4096_total_mb | 693.484544 | 1096.137728 |
| eng_llamacpp_32768_kv_mb | 0.0 | 3221.225472 |
| eng_llamacpp_32768_total_mb | 804.21417 | 4025.439642 |
| eng_llamacpp_4096_kv_mb | 0.0 | 402.653184 |
| eng_llamacpp_4096_total_mb | 693.484544 | 1096.137728 |
| genz_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['is_encoder_decoder', 'model_type', 'text_config', 'vision_config']. Looked for ['num_hidden_layers', 'hidden_size'] at t" | null |
| genz_chunked_kv_mb | null | 432.0 |
| genz_decode_kv_mb | null | 384.0 |
| genz_decode_weights_mb | null | 436.173828 |
| genz_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['is_encoder_decoder', 'model_type', 'text_config', 'vision_config']. Looked for ['num_hidden_layers', 'hidden_size'] at t" | null |
| genz_ffn_kinds | null | {"Dense": 12} |
| genz_kv_size | null | 49152 |
| genz_layer_kinds | null | {"MHA-global": 12} |
| genz_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['is_encoder_decoder', 'model_type', 'text_config', 'vision_config']. Looked for ['num_hidden_layers', 'hidden_size'] at t" | null |
| genz_prefill_kv_mb | null | 576.0 |
| genz_prefill_weights_mb | null | 872.347656 |
| kv_cross_layers | 0 | 12 |
| kv_full_layers | 0 | 12 |
| kv_gb@32768 | 0.0 | 3.221225472 |
| kv_gb@4096 | 0.0 | 0.402653184 |
| kvgeo_32768 | [0.0, 1, 0, 0.0, 16, 0.0] | [98304.0, 12, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [0.0, 1, 0, 0.0, 16, 0.0] | [98304.0, 12, 0, 0.0, 16, 0.0] |
| prefix_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['is_encoder_decoder', 'model_type', 'text_config', 'vision_config']. Looked for [" | null |
| prefix_kv_bytes_per_token | null | 98304 |
| report_kv_gb | 0.0 | 0.402653184 |
| report_total_gb | 0.716136243 | 1.199320064 |
| serving_bytes_per_token | null | 98304 |
| serving_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['is_encoder_decoder', 'model_type', 'text_config', 'vision_config']. Looked for ['num_hidden_layers', 'hidden_size'] at t" | null |
| serving_requests_8gib_1k | null | 85 |
| serving_state_bytes | null | 0 |
| slice_mib_16x4352 | [10496, true] | [14336, true] |
| vllm_limits | [93844, 1501504, 93844, 1, 0, 0.0, true] | [93844, 1501504, 93844, 1, 0, 50331648.0, true] |
| vllm_running_1024 | 1024 | 651 |
| vllm_running_8192 | 177 | 158 |

## synth:gemma2-9b-hub (decoder-only)

| figure | before | after |
|---|---|---|
| eng_None_32768_kv_mb | 1409.286144 | 11274.289152 |
| eng_None_32768_total_mb | 20726.532915 | 30591.535923 |
| eng_llamacpp_32768_kv_mb | 1409.286144 | 11274.289152 |
| eng_llamacpp_32768_total_mb | 20726.532915 | 30591.535923 |
| facts_full_attention | true | false |
| facts_layers | [42, 0, 0, null] | [21, 21, 0, 4096] |
| genz_layer_kinds | {"MHA-global": 42} | {"MHA-global": 21, "MHA-local": 21} |
| kv_full_layers | 0 | 21 |
| kv_gb@32768 | 1.409286144 | 11.274289152 |
| kv_sliding_layers | 42 | 21 |
| kv_source | "uniform" | "_layer_metadata" |
| kvgeo_32768 | [0.0, 42, 0, 0.0, 16, 0.0] | [344064.0, 42, 0, 0.0, 16, 0.0] |
| slice_mib_16x4352 | [52992, true] | [52224, true] |
| vllm_limits | [9855, 157680, 9855, 1, 0, 0.0, true] | [19711, 315376, 19711, 1, 1, 0.0, true] |
| vllm_running_8192 | 18 | 25 |

## synth:gguf-swa-full-pattern (decoder-only)

| figure | before | after |
|---|---|---|
| facts_full_attention | true | false |
| facts_layers | [34, 0, 0, null] | [5, 29, 0, 1024] |
| genz_chunked_kv_mb | 612.0 | 264.0 |
| genz_decode_kv_mb | 544.0 | 196.0 |
| genz_layer_kinds | {"MHA-global": 34} | {"MHA-global": 5, "MHA-local": 29} |
| genz_prefill_kv_mb | 544.0 | 196.0 |
| kv_full_layers | 34 | 5 |
| kv_sliding_layers | 0 | 29 |
| kv_source | "uniform" | "layer_plan:layer_types" |
| kvgeo_32768 | [139264.0, 34, 0, 0.0, 16, 0.0] | [139264.0, 5, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [139264.0, 34, 0, 0.0, 16, 0.0] | [139264.0, 5, 0, 0.0, 16, 0.0] |
| slice_mib_16x4352 | [24832, true] | [17664, true] |
| vllm_limits | [29394, 470304, 29394, 1, 0, 0.0, true] | [199885, 3198160, 199885, 1, 6, 0.0, true] |
| vllm_running_1024 | 367 | 425 |
| vllm_running_8192 | 55 | 217 |

## synth:gguf-swa-short-pattern (decoder-only)

| figure | before | after |
|---|---|---|
| facts_full_attention | true | false |
| facts_layers | [34, 0, 0, null] | [29, 5, 0, 1024] |
| genz_chunked_kv_mb | 612.0 | 552.0 |
| genz_decode_kv_mb | 544.0 | 484.0 |
| genz_layer_kinds | {"MHA-global": 34} | {"MHA-global": 29, "MHA-local": 5} |
| genz_prefill_kv_mb | 544.0 | 484.0 |
| kv_full_layers | 34 | 29 |
| kv_sliding_layers | 0 | 5 |
| kv_source | "uniform" | "layer_plan:layer_types" |
| kvgeo_32768 | [139264.0, 34, 0, 0.0, 16, 0.0] | [139264.0, 29, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [139264.0, 34, 0, 0.0, 16, 0.0] | [139264.0, 29, 0, 0.0, 16, 0.0] |
| slice_mib_16x4352 | [24832, true] | [24064, true] |
| vllm_limits | [29394, 470304, 29394, 1, 0, 0.0, true] | [199885, 3198160, 33314, 6, 1, 0.0, true] |
| vllm_running_1024 | 367 | 366 |
| vllm_running_8192 | 55 | 61 |

## synth:instructblip-flan-t5-xl (multimodal)

| figure | before | after |
|---|---|---|
| conv_kv_size | 131072 | 196608 |
| conv_layers | [16, 16] | [24, 24] |
| disagg_transfer_bytes_640 | 23592960 | 251658240 |
| eng_None_32768_kv_mb | 0.0 | 12884.901888 |
| eng_None_32768_params | 1663986176 | 1538943488 |
| eng_None_32768_total_mb | 3656.150426 | 16290.966938 |
| eng_None_4096_kv_mb | 0.0 | 1610.612736 |
| eng_None_4096_params | 1663986176 | 1538943488 |
| eng_None_4096_total_mb | 3495.089152 | 4855.616512 |
| eng_llamacpp_32768_kv_mb | 0.0 | 12884.901888 |
| eng_llamacpp_32768_params | 1663986176 | 1538943488 |
| eng_llamacpp_32768_total_mb | 3656.150426 | 16290.966938 |
| eng_llamacpp_4096_kv_mb | 0.0 | 1610.612736 |
| eng_llamacpp_4096_params | 1663986176 | 1538943488 |
| eng_llamacpp_4096_total_mb | 3495.089152 | 4855.616512 |
| genz_chunked_kv_mb | 162.0 | 1728.0 |
| genz_decode_kv_mb | 144.0 | 1536.0 |
| genz_decode_weights_mb | 208.875 | 2237.5 |
| genz_ffn_kinds | {"Dense": 12} | {"Dense": 24} |
| genz_kv_size | 18432 | 196608 |
| genz_layer_kinds | {"MHA-global": 12} | {"MHA-global": 24} |
| genz_prefill_kv_mb | 144.0 | 2304.0 |
| genz_prefill_weights_mb | 255.75 | 4475.0 |
| kv_cross_layers | 0 | 24 |
| kv_gb@32768 | 0.0 | 12.884901888 |
| kv_gb@4096 | 0.0 | 1.610612736 |
| kvgeo_32768 | [0.0, 1, 0, 0.0, 16, 0.0] | [393216.0, 24, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [0.0, 1, 0, 0.0, 16, 0.0] | [393216.0, 24, 0, 0.0, 16, 0.0] |
| prefix_kv_bytes_per_token | 36864 | 393216 |
| report_kv_gb | 0.0 | 1.610612736 |
| report_params | 1663986176 | 1538943488 |
| report_total_gb | 3.528512512 | 5.211162419 |
| report_weight_gb | 3.327972352 | 3.077886976 |
| serving_bytes_per_token | 36864 | 393216 |
| serving_requests_8gib_1k | 227 | 21 |
| sim_max_running | 322 | 67 |
| slice_mib_16x4352 | [30208, true] | [44800, true] |
| vllm_limits | [21221, 339536, 21221, 1, 0, 0.0, true] | [21508, 344128, 21508, 1, 0, 805306368.0, true] |
| vllm_running_1024 | 265 | 64 |
| vllm_running_8192 | 40 | 27 |

## synth:llama4-maverick-text (unknown)

| figure | before | after |
|---|---|---|
| conv_kv_size | 262144 | 98304 |
| conv_layers | [0, 32] | [0, 48] |
| genz_decode_weights_mb | 990833.125 | 506963.125 |
| genz_ffn_kinds | {"MoE": 48} | {"Dense": 24, "MoE": 24} |
| genz_prefill_weights_mb | 992806.25 | 508936.25 |

## synth:llama4-scout-text (unknown)

| figure | before | after |
|---|---|---|
| conv_kv_size | 262144 | 98304 |
| conv_layers | [0, 32] | [0, 48] |

## synth:marian-noflag (encoder-decoder)

| figure | before | after |
|---|---|---|
| conv_kv_size | 262144 | 12288 |
| conv_layers | [0, 32] | [6, 6] |
| disagg_transfer_bytes_640 | 335544320 | 15728640 |
| eng_None_32768_kv_mb | 439.517184 | 823.738368 |
| eng_None_32768_total_mb | 701.818266 | 1086.03945 |
| eng_None_4096_kv_mb | 87.195648 | 119.095296 |
| eng_None_4096_total_mb | 294.131917 | 326.031565 |
| eng_llamacpp_32768_kv_mb | 439.517184 | 823.738368 |
| eng_llamacpp_32768_total_mb | 701.818266 | 1086.03945 |
| eng_llamacpp_4096_kv_mb | 87.195648 | 119.095296 |
| eng_llamacpp_4096_total_mb | 294.131917 | 326.031565 |
| genz_chunked_kv_mb | 2304.0 | 108.0 |
| genz_decode_kv_mb | 2048.0 | 96.0 |
| genz_decode_weights_mb | 10053.914062 | 98.739258 |
| genz_ffn_kinds | {"Dense": 32} | {"Dense": 6} |
| genz_kv_size | 262144 | 12288 |
| genz_layer_kinds | {"MHA-global": 32} | {"MHA-global": 6} |
| genz_prefill_kv_mb | 2048.0 | 144.0 |
| genz_prefill_weights_mb | 10507.828125 | 197.478516 |
| kv_cross_layers | 0 | 6 |
| kv_full_layers | 0 | 6 |
| kv_gb@32768 | 0.0 | 0.805306368 |
| kv_gb@4096 | 0.0 | 0.100663296 |
| kvgeo_32768 | [0.0, 1, 0, 0.0, 16, 0.0] | [24576.0, 6, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [0.0, 1, 0, 0.0, 16, 0.0] | [24576.0, 6, 0, 0.0, 16, 0.0] |
| prefix_kv_bytes_per_token | 524288 | 24576 |
| report_kv_gb | 0.087195648 | 0.119095296 |
| report_total_gb | 0.323719209 | 0.361998787 |
| serving_bytes_per_token | 524288 | 24576 |
| serving_requests_8gib_1k | 16 | 341 |
| slice_mib_16x4352 | [6656, true] | [7680, true] |
| vllm_limits | [380038, 6080608, 380038, 1, 0, 0.0, true] | [380038, 6080608, 380038, 1, 0, 50331648.0, true] |
| vllm_running_8192 | 719 | 484 |

## synth:mpt-7b (unknown)

| figure | before | after |
|---|---|---|
| eng_None_32768_kv_mb | 0.0 | 17179.869184 |
| eng_None_32768_params | 6647455744 | 6649552896 |
| eng_None_32768_total_mb | 14180.748493 | 31364.811981 |
| eng_None_4096_kv_mb | 0.0 | 2147.483648 |
| eng_None_4096_params | 6647455744 | 6649552896 |
| eng_None_4096_total_mb | 13737.82999 | 15889.507942 |
| eng_llamacpp_32768_kv_mb | 0.0 | 17179.869184 |
| eng_llamacpp_32768_params | 6647455744 | 6649552896 |
| eng_llamacpp_32768_total_mb | 14180.748493 | 31364.811981 |
| eng_llamacpp_4096_kv_mb | 0.0 | 2147.483648 |
| eng_llamacpp_4096_params | 6647455744 | 6649552896 |
| eng_llamacpp_4096_total_mb | 13737.82999 | 15889.507942 |
| kv_gb@32768 | 0.0 | 17.179869184 |
| kv_gb@4096 | 0.0 | 2.147483648 |
| kvgeo_32768 | [0.0, 1, 0, 0.0, 16, 0.0] | [524288.0, 32, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [0.0, 1, 0, 0.0, 16, 0.0] | [524288.0, 32, 0, 0.0, 16, 0.0] |
| report_kv_gb | 0.0 | 2.147483648 |
| report_params | 6647455744 | 6649552896 |
| report_total_gb | 13.826413691 | 16.407588372 |
| report_weight_gb | 13.294911488 | 13.299105792 |

## synth:mt5-noflag (unknown)

| figure | before | after |
|---|---|---|
| conv_kv_size | 262144 | 12288 |
| conv_layers | [0, 32] | [8, 8] |
| disagg_transfer_bytes_640 | 335544320 | 15728640 |
| eng_None_32768_kv_mb | 0.0 | 805.306368 |
| eng_None_32768_params | 153052160 | 142754816 |
| eng_None_32768_total_mb | 376.568627 | 1161.280307 |
| eng_None_4096_kv_mb | 0.0 | 100.663296 |
| eng_None_4096_params | 153052160 | 142754816 |
| eng_None_4096_total_mb | 341.336474 | 421.405082 |
| eng_llamacpp_32768_kv_mb | 0.0 | 805.306368 |
| eng_llamacpp_32768_params | 153052160 | 142754816 |
| eng_llamacpp_32768_total_mb | 376.568627 | 1161.280307 |
| eng_llamacpp_4096_kv_mb | 0.0 | 100.663296 |
| eng_llamacpp_4096_params | 153052160 | 142754816 |
| eng_llamacpp_4096_total_mb | 341.336474 | 421.405082 |
| genz_chunked_kv_mb | 2304.0 | 108.0 |
| genz_decode_kv_mb | 2048.0 | 96.0 |
| genz_decode_weights_mb | 11554.0 | 278.25 |
| genz_ffn_kinds | {"Dense": 32} | {"Dense": 8} |
| genz_kv_size | 262144 | 12288 |
| genz_layer_kinds | {"MHA-global": 32} | {"MHA-global": 8} |
| genz_prefill_kv_mb | 2048.0 | 144.0 |
| genz_prefill_weights_mb | 13508.0 | 556.5 |
| kv_cross_layers | 0 | 8 |
| kv_gb@32768 | 0.0 | 0.805306368 |
| kv_gb@4096 | 0.0 | 0.100663296 |
| kvgeo_32768 | [0.0, 1, 0, 0.0, 16, 0.0] | [24576.0, 8, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [0.0, 1, 0, 0.0, 16, 0.0] | [24576.0, 8, 0, 0.0, 16, 0.0] |
| prefix_kv_bytes_per_token | 524288 | 24576 |
| report_kv_gb | 0.0 | 0.100663296 |
| report_params | 153052160 | 142754816 |
| report_total_gb | 0.348382904 | 0.448584172 |
| report_weight_gb | 0.30610432 | 0.285509632 |
| serving_bytes_per_token | 524288 | 24576 |
| serving_requests_8gib_1k | 16 | 341 |
| slice_mib_16x4352 | [7424, true] | [8192, true] |
| vllm_limits | [281214, 4499424, 281214, 1, 0, 0.0, true] | [375102, 6001632, 375102, 1, 0, 50331648.0, true] |
| vllm_running_8192 | 532 | 478 |

## synth:musicgen-small (encoder-decoder)

| figure | before | after |
|---|---|---|
| conv_kv_size | 131072 | 98304 |
| conv_layers | [16, 16] | [12, 24] |
| disagg_transfer_bytes_640 | 62914560 | 125829120 |
| eng_None_32768_kv_mb | 701.80864 | 6497.746944 |
| eng_None_32768_params | 808394240 | 1080272640 |
| eng_None_32768_total_mb | 2484.691558 | 8824.386662 |
| eng_None_4096_kv_mb | 114.60608 | 860.602368 |
| eng_None_4096_params | 808394240 | 1080272640 |
| eng_None_4096_total_mb | 1814.441779 | 3104.194867 |
| eng_llamacpp_32768_kv_mb | 701.80864 | 6497.746944 |
| eng_llamacpp_32768_params | 808394240 | 1080272640 |
| eng_llamacpp_32768_total_mb | 2484.691558 | 8824.386662 |
| eng_llamacpp_4096_kv_mb | 114.60608 | 860.602368 |
| eng_llamacpp_4096_params | 808394240 | 1080272640 |
| eng_llamacpp_4096_total_mb | 1814.441779 | 3104.194867 |
| facts_layers | [32, 0, 0, null] | [24, 0, 0, null] |
| genz_chunked_kv_mb | 432.0 | 864.0 |
| genz_decode_kv_mb | 384.0 | 768.0 |
| genz_decode_weights_mb | 580.0 | 676.0 |
| genz_kv_size | 49152 | 98304 |
| genz_prefill_kv_mb | 384.0 | 912.0 |
| genz_prefill_weights_mb | 584.0 | 914.0 |
| kv_cross_layers | 0 | 24 |
| kv_full_layers | 0 | 24 |
| kv_gb@32768 | 0.0 | 6.442450944 |
| kv_gb@4096 | 0.0 | 0.805306368 |
| kvgeo_32768 | [0.0, 1, 0, 0.0, 16, 0.0] | [196608.0, 24, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [0.0, 1, 0, 0.0, 16, 0.0] | [196608.0, 24, 0, 0.0, 16, 0.0] |
| prefix_kv_bytes_per_token | 98304 | 196608 |
| report_kv_gb | 0.11460608 | 0.860602368 |
| report_params | 808394240 | 1080272640 |
| report_total_gb | 1.856430039 | 3.294460785 |
| report_weight_gb | 1.61678848 | 2.16054528 |
| serving_bytes_per_token | 98304 | 196608 |
| serving_requests_8gib_1k | 85 | 42 |
| sim_max_running | 72 | 242 |
| slice_mib_16x4352 | [81582, false] | [22528, true] |
| vllm_limits | [4702, 75232, 4702, 1, 0, 0.0, true] | [46826, 749216, 46826, 1, 0, 201326592.0, true] |
| vllm_running_1024 | 58 | 225 |
| vllm_running_8192 | 8 | 71 |

## synth:nemotronh-list-depth (hybrid)

| figure | before | after |
|---|---|---|
| conv_kv_size | 65536 | 8192 |
| conv_layers | [0, 32] | [0, 52] |
| conv_state_bytes | 0 | 102137856 |
| disagg_transfer_bytes_640 | null | 112623616 |
| eng_None_32768_kv_mb | 805.306368 | 536.870912 |
| eng_None_32768_params | 2183850336 | 8135050240 |
| eng_None_32768_state_mb | 0.0 | 102.137856 |
| eng_None_32768_total_mb | 6260.170637 | 17996.272845 |
| eng_None_4096_kv_mb | 100.663296 | 67.108864 |
| eng_None_4096_params | 2183850336 | 8135050240 |
| eng_None_4096_state_mb | 0.0 | 102.137856 |
| eng_None_4096_total_mb | 5011.945766 | 16982.928998 |
| eng_llamacpp_32768_kv_mb | 805.306368 | 536.870912 |
| eng_llamacpp_32768_params | 2183850336 | 8135050240 |
| eng_llamacpp_32768_state_mb | 0.0 | 102.137856 |
| eng_llamacpp_32768_total_mb | 6260.170637 | 17996.272845 |
| eng_llamacpp_4096_kv_mb | 100.663296 | 67.108864 |
| eng_llamacpp_4096_params | 2183850336 | 8135050240 |
| eng_llamacpp_4096_state_mb | 0.0 | 102.137856 |
| eng_llamacpp_4096_total_mb | 5011.945766 | 16982.928998 |
| facts_full_attention | true | false |
| facts_layers | [32, 0, 0, null] | [4, 0, 24, null] |
| genz_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'attention_bias', 'attention_dropout', 'attention_head_dim', 'auto_map', 'bos_token_id', 'chunk_size', '" | null |
| genz_chunked_kv_mb | null | 266.8125 |
| genz_decode_kv_mb | null | 161.40625 |
| genz_decode_weights_mb | null | 14426.8125 |
| genz_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'attention_bias', 'attention_dropout', 'attention_head_dim', 'auto_map', 'bos_token_id', 'chunk_size', '" | null |
| genz_ffn_kinds | null | {"Dense": 24, "no_op": 28} |
| genz_kv_size | null | 8192 |
| genz_layer_kinds | null | {"MHA-global": 4, "Recurrent": 24, "no_op": 24} |
| genz_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'attention_bias', 'attention_dropout', 'attention_head_dim', 'auto_map', 'bos_token_id', 'chunk_size', '" | null |
| genz_prefill_kv_mb | null | 161.40625 |
| genz_prefill_weights_mb | null | 15450.8125 |
| kv_free_layers | 0 | 48 |
| kv_full_layers | 0 | 4 |
| kv_gb@32768 | 0.805306368 | 0.536870912 |
| kv_gb@4096 | 0.100663296 | 0.067108864 |
| kv_source | "uniform" | "layer_plan:layers_block_type" |
| kvgeo_32768 | [0.0, 1, 0, 0.0, 16, 0.0] | [16384.0, 4, 24, 4255744.0, 1040, 0.0] |
| kvgeo_4096 | [0.0, 1, 0, 0.0, 16, 0.0] | [16384.0, 4, 24, 4255744.0, 1040, 0.0] |
| prefix_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'attention_bias', 'attention_dropout', 'attention_head_dim', 'au" | null |
| prefix_kv_bytes_per_token | null | 16384 |
| report_kv_gb | 0.100663296 | 0.067108864 |
| report_params | 2183850336 | 8135050240 |
| report_state_gb | 0.025534464 | 0.102137856 |
| report_total_gb | 5.171436142 | 17.125494702 |
| report_weight_gb | 4.367700672 | 16.27010048 |
| serving_bytes_per_token | null | 16384 |
| serving_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'attention_bias', 'attention_dropout', 'attention_head_dim', 'auto_map', 'bos_token_id', 'chunk_size', '" | null |
| serving_requests_8gib_1k | null | 72 |
| serving_state_bytes | null | 102137856 |
| sim_max_running | 384 | 400 |
| slice_mib_16x4352 | [37376, true] | [32512, true] |
| state_gb | 0.025534464 | 0.102137856 |
| vllm_limits | [25363, 405808, 25363, 1, 0, 0.0, true] | [3176, 3303040, 3176, 1, 0, 102236160.0, true] |
| vllm_running_1024 | 317 | 397 |
| vllm_running_8192 | 48 | 211 |

## synth:nemotronh-list-depth-nested (hybrid)

| figure | before | after |
|---|---|---|
| conv_kv_size | 262144 | 8192 |
| conv_layers | [0, 32] | [0, 52] |
| conv_state_bytes | 0 | 102137856 |
| disagg_transfer_bytes_640 | null | 112623616 |
| eng_None_32768_kv_mb | 1610.612736 | 536.870912 |
| eng_None_32768_params | 123570432 | 247087176 |
| eng_None_32768_state_mb | 0.0 | 102.137856 |
| eng_None_32768_total_mb | 2944.917197 | 2220.346717 |
| eng_None_4096_kv_mb | 201.326592 | 67.108864 |
| eng_None_4096_params | 123570432 | 247087176 |
| eng_None_4096_state_mb | 0.0 | 102.137856 |
| eng_None_4096_total_mb | 992.049254 | 1207.00287 |
| eng_llamacpp_32768_kv_mb | 1610.612736 | 536.870912 |
| eng_llamacpp_32768_params | 123570432 | 247087176 |
| eng_llamacpp_32768_state_mb | 0.0 | 102.137856 |
| eng_llamacpp_32768_total_mb | 2944.917197 | 2220.346717 |
| eng_llamacpp_4096_kv_mb | 201.326592 | 67.108864 |
| eng_llamacpp_4096_params | 123570432 | 247087176 |
| eng_llamacpp_4096_state_mb | 0.0 | 102.137856 |
| eng_llamacpp_4096_total_mb | 992.049254 | 1207.00287 |
| facts_full_attention | true | false |
| facts_layers | [32, 0, 0, null] | [4, 0, 24, null] |
| genz_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['model_type', 'text_config']. Looked for ['num_hidden_layers', 'hidden_size'] at the top level and in nested blocks. Refu" | null |
| genz_chunked_kv_mb | null | 266.8125 |
| genz_decode_kv_mb | null | 161.40625 |
| genz_decode_weights_mb | null | 14426.8125 |
| genz_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['model_type', 'text_config']. Looked for ['num_hidden_layers', 'hidden_size'] at the top level and in nested blocks. Refu" | null |
| genz_ffn_kinds | null | {"Dense": 24, "no_op": 28} |
| genz_kv_size | null | 8192 |
| genz_layer_kinds | null | {"MHA-global": 4, "Recurrent": 24, "no_op": 24} |
| genz_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['model_type', 'text_config']. Looked for ['num_hidden_layers', 'hidden_size'] at the top level and in nested blocks. Refu" | null |
| genz_prefill_kv_mb | null | 161.40625 |
| genz_prefill_weights_mb | null | 15450.8125 |
| kv_free_layers | 0 | 48 |
| kv_full_layers | 0 | 4 |
| kv_gb@32768 | 1.610612736 | 0.536870912 |
| kv_gb@4096 | 0.201326592 | 0.067108864 |
| kv_source | "uniform" | "layer_plan:layers_block_type" |
| kvgeo_32768 | [0.0, 1, 0, 0.0, 16, 0.0] | [16384.0, 4, 24, 4255744.0, 1040, 0.0] |
| kvgeo_4096 | [0.0, 1, 0, 0.0, 16, 0.0] | [16384.0, 4, 24, 4255744.0, 1040, 0.0] |
| model_type | "unknown" | "hybrid" |
| prefix_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['model_type', 'text_config']. Looked for ['num_hidden_layers', 'hidden_size'] at " | null |
| prefix_kv_bytes_per_token | null | 16384 |
| report_kv_gb | 0.201326592 | 0.067108864 |
| report_params | 123570432 | 247087176 |
| report_state_gb | 0.0 | 0.102137856 |
| report_total_gb | 1.141030932 | 1.349568574 |
| report_weight_gb | 0.247140864 | 0.494174352 |
| serving_bytes_per_token | null | 16384 |
| serving_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['model_type', 'text_config']. Looked for ['num_hidden_layers', 'hidden_size'] at the top level and in nested blocks. Refu" | null |
| serving_requests_8gib_1k | null | 72 |
| serving_state_bytes | null | 102137856 |
| sim_max_running | 384 | 400 |
| slice_mib_16x4352 | [37376, true] | [32512, true] |
| state_gb | 0.0 | 0.102137856 |
| vllm_limits | [25363, 405808, 25363, 1, 0, 0.0, true] | [3176, 3303040, 3176, 1, 0, 102236160.0, true] |
| vllm_running_1024 | 317 | 397 |
| vllm_running_8192 | 48 | 211 |

## synth:nllb-moe (encoder-decoder)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 335544320 | 251658240 |
| eng_None_32768_kv_mb | 7032.274944 | 13179.813888 |
| eng_None_32768_total_mb | 14196.983398 | 20344.522342 |
| eng_None_4096_kv_mb | 1395.130368 | 1905.524736 |
| eng_None_4096_total_mb | 8338.379571 | 8848.773939 |
| eng_llamacpp_32768_kv_mb | 7032.274944 | 13179.813888 |
| eng_llamacpp_32768_total_mb | 14196.983398 | 20344.522342 |
| eng_llamacpp_4096_kv_mb | 1395.130368 | 1905.524736 |
| eng_llamacpp_4096_total_mb | 8338.379571 | 8848.773939 |
| genz_chunked_kv_mb | 2304.0 | 1728.0 |
| genz_decode_kv_mb | 2048.0 | 1536.0 |
| genz_decode_weights_mb | 710641.609375 | 198772.804688 |
| genz_ffn_kinds | {"MoE": 32} | {"MoE": 24} |
| genz_kv_size | 262144 | 196608 |
| genz_layer_kinds | {"MHA-global": 32} | {"MHA-global": 24} |
| genz_prefill_kv_mb | 2048.0 | 2304.0 |
| genz_prefill_weights_mb | 712643.21875 | 202461.609375 |
| kv_cross_layers | 0 | 24 |
| kv_full_layers | 0 | 24 |
| kv_gb@32768 | 0.0 | 12.884901888 |
| kv_gb@4096 | 0.0 | 1.610612736 |
| kvgeo_32768 | [0.0, 1, 0, 0.0, 16, 0.0] | [393216.0, 24, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [0.0, 1, 0, 0.0, 16, 0.0] | [393216.0, 24, 0, 0.0, 16, 0.0] |
| prefix_kv_bytes_per_token | 524288 | 393216 |
| report_kv_gb | 1.395130368 | 1.905524736 |
| report_total_gb | 8.665537495 | 9.278010737 |
| serving_bytes_per_token | 524288 | 393216 |
| serving_requests_8gib_1k | 16 | 21 |
| vllm_limits | [0, 0, 0, 1, 0, 0.0, false] | [0, 0, 0, 1, 0, 805306368.0, false] |

## synth:prophetnet (encoder-decoder)

| figure | before | after |
|---|---|---|
| conv_kv_size | 32768 | 49152 |
| conv_layers | [16, 16] | [12, 12] |
| disagg_transfer_bytes_640 | 83886080 | 62914560 |
| eng_None_32768_kv_mb | 559.34976 | 3294.953472 |
| eng_None_32768_params | 506632192 | 640940032 |
| eng_None_32768_total_mb | 1794.073395 | 4798.292787 |
| eng_None_4096_kv_mb | 91.42272 | 476.381184 |
| eng_None_4096_params | 506632192 | 640940032 |
| eng_None_4096_total_mb | 1215.41673 | 1868.990874 |
| eng_llamacpp_32768_kv_mb | 559.34976 | 3294.953472 |
| eng_llamacpp_32768_params | 506632192 | 640940032 |
| eng_llamacpp_32768_total_mb | 1794.073395 | 4798.292787 |
| eng_llamacpp_4096_kv_mb | 91.42272 | 476.381184 |
| eng_llamacpp_4096_params | 506632192 | 640940032 |
| eng_llamacpp_4096_total_mb | 1215.41673 | 1868.990874 |
| facts_layers | [32, 0, 0, null] | [12, 0, 0, null] |
| genz_chunked_kv_mb | 576.0 | 432.0 |
| genz_decode_kv_mb | 512.0 | 384.0 |
| genz_decode_weights_mb | 1691.613281 | 395.613281 |
| genz_ffn_kinds | {"Dense": 32} | {"Dense": 12} |
| genz_kv_size | 65536 | 49152 |
| genz_layer_kinds | {"MHA-global": 32} | {"MHA-global": 12} |
| genz_prefill_kv_mb | 512.0 | 576.0 |
| genz_prefill_weights_mb | 1751.226562 | 791.226562 |
| kv_cross_layers | 0 | 12 |
| kv_full_layers | 0 | 12 |
| kv_gb@32768 | 0.0 | 3.221225472 |
| kv_gb@4096 | 0.0 | 0.402653184 |
| kvgeo_32768 | [0.0, 1, 0, 0.0, 16, 0.0] | [98304.0, 12, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [0.0, 1, 0, 0.0, 16, 0.0] | [98304.0, 12, 0, 0.0, 16, 0.0] |
| prefix_kv_bytes_per_token | 131072 | 98304 |
| report_kv_gb | 0.09142272 | 0.476381184 |
| report_params | 506632192 | 640940032 |
| report_total_gb | 1.257843999 | 1.988409836 |
| report_weight_gb | 1.013264384 | 1.281880064 |
| serving_bytes_per_token | 131072 | 98304 |
| serving_requests_8gib_1k | 64 | 85 |
| slice_mib_16x4352 | [18432, true] | [14336, true] |
| vllm_limits | [34622, 553952, 34622, 1, 0, 0.0, true] | [94033, 1504528, 94033, 1, 0, 25165824.0, true] |
| vllm_running_1024 | 432 | 839 |
| vllm_running_8192 | 65 | 167 |

## synth:qwen35-list-depth (multimodal)

| figure | before | after |
|---|---|---|
| conv_kv_size | 262144 | 16384 |
| conv_state_bytes | 0 | 51511296 |
| disagg_transfer_bytes_640 | null | 72482816 |
| eng_None_32768_params | 2622843904 | 4944920064 |
| eng_None_32768_state_mb | 0.0 | 51.511296 |
| eng_None_32768_total_mb | 6833.655398 | 11529.319014 |
| eng_None_4096_params | 2622843904 | 4944920064 |
| eng_None_4096_state_mb | 0.0 | 51.511296 |
| eng_None_4096_total_mb | 5637.439898 | 10333.103514 |
| eng_llamacpp_32768_params | 2622843904 | 4944920064 |
| eng_llamacpp_32768_state_mb | 0.0 | 51.511296 |
| eng_llamacpp_32768_total_mb | 6833.655398 | 11529.319014 |
| eng_llamacpp_4096_params | 2622843904 | 4944920064 |
| eng_llamacpp_4096_state_mb | 0.0 | 51.511296 |
| eng_llamacpp_4096_total_mb | 5637.439898 | 10333.103514 |
| facts_full_attention | true | false |
| facts_layers | [32, 0, 0, null] | [8, 0, 24, null] |
| genz_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'image_token_id', 'model_type', 'text_config', 'tie_word_embeddings', 'transformers_version', 'video_tok" | null |
| genz_chunked_kv_mb | null | 242.25 |
| genz_decode_kv_mb | null | 177.125 |
| genz_decode_weights_mb | null | 7861.523438 |
| genz_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'image_token_id', 'model_type', 'text_config', 'tie_word_embeddings', 'transformers_version', 'video_tok" | null |
| genz_ffn_kinds | null | {"Dense": 32} |
| genz_kv_size | null | 16384 |
| genz_layer_kinds | null | {"MHA-global": 8, "Recurrent": 24} |
| genz_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'image_token_id', 'model_type', 'text_config', 'tie_word_embeddings', 'transformers_version', 'video_tok" | null |
| genz_prefill_kv_mb | null | 177.125 |
| genz_prefill_weights_mb | null | 9074.023438 |
| kv_free_layers | 0 | 24 |
| kv_source | "_layer_metadata" | "layer_plan:layer_types" |
| kvgeo_32768 | [32768.0, 1, 0, 0.0, 16, 0.0] | [32768.0, 8, 24, 2146304.0, 528, 0.0] |
| kvgeo_4096 | [32768.0, 1, 0, 0.0, 16, 0.0] | [32768.0, 8, 24, 2146304.0, 528, 0.0] |
| prefix_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'image_token_id', 'model_type', 'text_config', 'tie_word_embeddi" | null |
| prefix_kv_bytes_per_token | null | 32768 |
| report_params | 2622843904 | 4944920064 |
| report_state_gb | 0.0 | 0.051511296 |
| report_total_gb | 5.715790316 | 10.421756191 |
| report_weight_gb | 5.245687808 | 9.889840128 |
| serving_bytes_per_token | null | 32768 |
| serving_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'image_token_id', 'model_type', 'text_config', 'tie_word_embeddings', 'transformers_version', 'video_tok" | null |
| serving_requests_8gib_1k | null | 100 |
| serving_state_bytes | null | 51511296 |
| slice_mib_16x4352 | [23808, true] | [20480, true] |
| state_gb | 0.0 | 0.051511296 |
| vllm_limits | [31388, 502208, 31388, 1, 0, 0.0, true] | [3767, 1988976, 3767, 1, 0, 51904512.0, true] |
| vllm_running_1024 | 392 | 627 |
| vllm_running_8192 | 59 | 198 |

## synth:rwkv (state-space)

| figure | before | after |
|---|---|---|
| conv_kv_size | 98304 | 0 |
| conv_state_bytes | 0 | 983040 |
| disagg_transfer_bytes_640 | 125829120 | 983040 |
| eng_None_32768_params | 1309554688 | 1311373312 |
| eng_None_32768_state_mb | 0.0 | 0.98304 |
| eng_None_32768_total_mb | 3062.027878 | 3066.648166 |
| eng_None_4096_params | 1309554688 | 1311373312 |
| eng_None_4096_state_mb | 0.0 | 0.98304 |
| eng_None_4096_total_mb | 2840.568627 | 2845.188915 |
| eng_llamacpp_32768_params | 1309554688 | 1311373312 |
| eng_llamacpp_32768_state_mb | 0.0 | 0.98304 |
| eng_llamacpp_32768_total_mb | 3062.027878 | 3066.648166 |
| eng_llamacpp_4096_params | 1309554688 | 1311373312 |
| eng_llamacpp_4096_state_mb | 0.0 | 0.98304 |
| eng_llamacpp_4096_total_mb | 2840.568627 | 2845.188915 |
| facts_full_attention | true | false |
| facts_layers | [24, 0, 0, null] | [0, 0, 24, null] |
| genz_chunked_kv_mb | 864.0 | 1.875 |
| genz_decode_kv_mb | 768.0 | 0.9375 |
| genz_decode_weights_mb | 2500.394531 | 2500.863281 |
| genz_kv_size | 98304 | 0 |
| genz_layer_kinds | {"MHA-global": 24} | {"Recurrent": 24} |
| genz_prefill_kv_mb | 768.0 | 0.9375 |
| genz_prefill_weights_mb | 2696.789062 | 2697.257812 |
| kv_free_layers | 0 | 24 |
| kv_full_layers | 24 | 0 |
| kv_source | "uniform" | "layer_plan:rwkv_stack" |
| kvgeo_32768 | [0.0, 24, 0, 0.0, 16, 0.0] | [0.0, 1, 24, 40960.0, 16, 0.0] |
| kvgeo_4096 | [0.0, 24, 0, 0.0, 16, 0.0] | [0.0, 1, 24, 40960.0, 16, 0.0] |
| prefix_kv_bytes_per_token | 196608 | 0 |
| report_params | 1309554688 | 1311373312 |
| report_state_gb | 0.00688128 | 0.00098304 |
| report_total_gb | 2.893118013 | 2.889677373 |
| report_weight_gb | 2.619109376 | 2.622746624 |
| serving_bytes_per_token | 196608 | 0 |
| serving_requests_8gib_1k | 42 | 8738 |
| serving_state_bytes | 0 | 983040 |
| sim_max_running | 342 | 335 |
| slice_mib_16x4352 | [25088, true] | [25600, true] |
| state_gb | 0.00688128 | 0.00098304 |
| vllm_limits | [22571, 361136, 22571, 1, 0, 0.0, true] | [22433, 358928, 22433, 1, 0, 3145728.0, true] |
| vllm_running_1024 | 282 | 276 |

## synth:t5-asym (encoder-decoder)

| figure | before | after |
|---|---|---|
| conv_kv_size | 131072 | 49152 |
| conv_layers | [16, 16] | [24, 12] |
| disagg_transfer_bytes_640 | 335544320 | 62914560 |
| eng_None_32768_kv_mb | 559.34976 | 3368.681472 |
| eng_None_32768_params | 497725440 | 611061760 |
| eng_None_32768_total_mb | 1725.928243 | 4761.932595 |
| eng_None_4096_kv_mb | 91.42272 | 550.109184 |
| eng_None_4096_params | 497725440 | 611061760 |
| eng_None_4096_total_mb | 1172.437402 | 1857.796506 |
| eng_llamacpp_32768_kv_mb | 559.34976 | 3368.681472 |
| eng_llamacpp_32768_params | 497725440 | 611061760 |
| eng_llamacpp_32768_total_mb | 1725.928243 | 4761.932595 |
| eng_llamacpp_4096_kv_mb | 91.42272 | 550.109184 |
| eng_llamacpp_4096_params | 497725440 | 611061760 |
| eng_llamacpp_4096_total_mb | 1172.437402 | 1857.796506 |
| facts_layers | [24, 0, 0, null] | [12, 0, 0, null] |
| genz_chunked_kv_mb | 2304.0 | 432.0 |
| genz_decode_kv_mb | 2048.0 | 384.0 |
| genz_decode_weights_mb | 9851.0 | 338.75 |
| genz_ffn_kinds | {"Dense": 32} | {"Dense": 12} |
| genz_kv_size | 262144 | 49152 |
| genz_layer_kinds | {"MHA-global": 32} | {"MHA-global": 12} |
| genz_prefill_kv_mb | 2048.0 | 768.0 |
| genz_prefill_weights_mb | 10102.0 | 905.5 |
| kv_cross_layers | 0 | 12 |
| kv_full_layers | 24 | 12 |
| kv_gb@32768 | 0.0 | 3.221225472 |
| kv_gb@4096 | 0.0 | 0.402653184 |
| kvgeo_32768 | [0.0, 1, 0, 0.0, 16, 0.0] | [98304.0, 12, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [0.0, 1, 0, 0.0, 16, 0.0] | [98304.0, 12, 0, 0.0, 16, 0.0] |
| prefix_kv_bytes_per_token | 524288 | 98304 |
| report_kv_gb | 0.09142272 | 0.550109184 |
| report_params | 497725440 | 611061760 |
| report_total_gb | 1.209831506 | 1.986927903 |
| report_weight_gb | 0.99545088 | 1.22212352 |
| serving_bytes_per_token | 524288 | 98304 |
| serving_requests_8gib_1k | 16 | 85 |
| sim_max_running | 400 | 293 |
| slice_mib_16x4352 | [15104, true] | [14336, true] |
| vllm_limits | [46813, 749008, 46813, 1, 0, 0.0, true] | [94387, 1510192, 94387, 1, 0, 201326592.0, true] |
| vllm_running_1024 | 585 | 280 |
| vllm_running_8192 | 88 | 120 |

## synth:t5-small (encoder-decoder)

| figure | before | after |
|---|---|---|
| conv_kv_size | 131072 | 12288 |
| conv_layers | [16, 16] | [6, 6] |
| disagg_transfer_bytes_640 | 335544320 | 15728640 |
| eng_None_32768_kv_mb | 274.432 | 823.738368 |
| eng_None_32768_params | 135813120 | 144303104 |
| eng_None_32768_total_mb | 656.787866 | 1223.074202 |
| eng_None_4096_kv_mb | 45.056 | 119.095296 |
| eng_None_4096_params | 135813120 | 144303104 |
| eng_None_4096_total_mb | 372.047053 | 463.066317 |
| eng_llamacpp_32768_kv_mb | 274.432 | 823.738368 |
| eng_llamacpp_32768_params | 135813120 | 144303104 |
| eng_llamacpp_32768_total_mb | 656.787866 | 1223.074202 |
| eng_llamacpp_4096_kv_mb | 45.056 | 119.095296 |
| eng_llamacpp_4096_params | 135813120 | 144303104 |
| eng_llamacpp_4096_total_mb | 372.047053 | 463.066317 |
| genz_chunked_kv_mb | 2304.0 | 108.0 |
| genz_decode_kv_mb | 2048.0 | 96.0 |
| genz_decode_weights_mb | 9851.0 | 73.375 |
| genz_ffn_kinds | {"Dense": 32} | {"Dense": 6} |
| genz_kv_size | 262144 | 12288 |
| genz_layer_kinds | {"MHA-global": 32} | {"MHA-global": 6} |
| genz_prefill_kv_mb | 2048.0 | 144.0 |
| genz_prefill_weights_mb | 10102.0 | 146.75 |
| kv_cross_layers | 0 | 6 |
| kv_gb@32768 | 0.0 | 0.805306368 |
| kv_gb@4096 | 0.0 | 0.100663296 |
| kvgeo_32768 | [0.0, 1, 0, 0.0, 16, 0.0] | [24576.0, 6, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [0.0, 1, 0, 0.0, 16, 0.0] | [24576.0, 6, 0, 0.0, 16, 0.0] |
| prefix_kv_bytes_per_token | 524288 | 24576 |
| report_kv_gb | 0.045056 | 0.119095296 |
| report_params | 135813120 | 144303104 |
| report_total_gb | 0.393206415 | 0.499033539 |
| report_weight_gb | 0.27162624 | 0.288606208 |
| serving_bytes_per_token | 524288 | 24576 |
| serving_requests_8gib_1k | 16 | 341 |
| slice_mib_16x4352 | [6656, true] | [7424, true] |
| vllm_limits | [382640, 6122240, 382640, 1, 0, 0.0, true] | [382640, 6122240, 382640, 1, 0, 50331648.0, true] |
| vllm_running_8192 | 724 | 488 |

## synth:unknown-layer-name (decoder-only)

| figure | before | after |
|---|---|---|
| eng_None_32768_kv_mb | 134.217728 | 268.435456 |
| eng_None_32768_total_mb | 1060.603494 | 1194.821222 |
| eng_None_4096_kv_mb | 16.777216 | 33.554432 |
| eng_None_4096_total_mb | 721.703731 | 738.480947 |
| eng_llamacpp_32768_kv_mb | 134.217728 | 268.435456 |
| eng_llamacpp_32768_total_mb | 1060.603494 | 1194.821222 |
| eng_llamacpp_4096_kv_mb | 16.777216 | 33.554432 |
| eng_llamacpp_4096_total_mb | 721.703731 | 738.480947 |
| kv_free_layers | 2 | 0 |
| kv_full_layers | 2 | 4 |
| kv_gb@32768 | 0.134217728 | 0.268435456 |
| kv_gb@4096 | 0.016777216 | 0.033554432 |
| kv_source | "_layer_metadata" | "layer_plan:layer_types" |
| kvgeo_32768 | [4096.0, 4, 0, 0.0, 16, 0.0] | [8192.0, 4, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [4096.0, 4, 0, 0.0, 16, 0.0] | [8192.0, 4, 0, 0.0, 16, 0.0] |
| report_kv_gb | 0.016777216 | 0.033554432 |
| report_total_gb | 0.769351025 | 0.789483684 |

## synth:vision-enc-dec (encoder-decoder)

| figure | before | after |
|---|---|---|
| conv_kv_size | 131072 | 24576 |
| conv_layers | [16, 16] | [12, 12] |
| conv_state_bytes | 0 | 28360704 |
| disagg_transfer_bytes_640 | null | 59817984 |
| eng_None_32768_kv_mb | 701.80864 | 1694.26944 |
| eng_None_32768_params | 808394240 | 885200640 |
| eng_None_32768_total_mb | 2484.691558 | 3630.765158 |
| eng_None_4096_kv_mb | 114.60608 | 284.983296 |
| eng_None_4096_params | 808394240 | 885200640 |
| eng_None_4096_total_mb | 1814.441779 | 2138.431795 |
| eng_llamacpp_32768_kv_mb | 701.80864 | 1694.26944 |
| eng_llamacpp_32768_params | 808394240 | 885200640 |
| eng_llamacpp_32768_total_mb | 2484.691558 | 3630.765158 |
| eng_llamacpp_4096_kv_mb | 114.60608 | 284.983296 |
| eng_llamacpp_4096_params | 808394240 | 885200640 |
| eng_llamacpp_4096_total_mb | 1814.441779 | 2138.431795 |
| facts_layers | [32, 0, 0, null] | [12, 0, 0, null] |
| genz_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['decoder', 'encoder', 'is_encoder_decoder', 'model_type']. Looked for ['num_hidden_layers', 'hidden_size'] at the top lev" | null |
| genz_chunked_kv_mb | null | 270.09375 |
| genz_decode_kv_mb | null | 219.046875 |
| genz_decode_weights_mb | null | 434.173828 |
| genz_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['decoder', 'encoder', 'is_encoder_decoder', 'model_type']. Looked for ['num_hidden_layers', 'hidden_size'] at the top lev" | null |
| genz_ffn_kinds | null | {"Dense": 12} |
| genz_kv_size | null | 24576 |
| genz_layer_kinds | null | {"MHA-global": 12} |
| genz_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['decoder', 'encoder', 'is_encoder_decoder', 'model_type']. Looked for ['num_hidden_layers', 'hidden_size'] at the top lev" | null |
| genz_prefill_kv_mb | null | 239.332031 |
| genz_prefill_weights_mb | null | 730.347656 |
| kv_cross_layers | 0 | 12 |
| kv_full_layers | 0 | 12 |
| kv_gb@32768 | 0.0 | 1.63897344 |
| kv_gb@4096 | 0.0 | 0.229687296 |
| kvgeo_32768 | [0.0, 1, 0, 0.0, 16, 0.0] | [49152.0, 12, 0, 0.0, 16, 28360704.0] |
| kvgeo_4096 | [0.0, 1, 0, 0.0, 16, 0.0] | [49152.0, 12, 0, 0.0, 16, 28360704.0] |
| prefix_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['decoder', 'encoder', 'is_encoder_decoder', 'model_type']. Looked for ['num_hidde" | null |
| prefix_kv_bytes_per_token | null | 49152 |
| report_kv_gb | 0.11460608 | 0.284983296 |
| report_params | 808394240 | 885200640 |
| report_total_gb | 1.856430039 | 2.213573898 |
| report_weight_gb | 1.61678848 | 1.77040128 |
| serving_bytes_per_token | null | 49152 |
| serving_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['decoder', 'encoder', 'is_encoder_decoder', 'model_type']. Looked for ['num_hidden_layers', 'hidden_size'] at the top lev" | null |
| serving_requests_8gib_1k | null | 109 |
| serving_state_bytes | null | 28360704 |
| sim_max_running | 72 | 400 |
| slice_mib_16x4352 | [81582, false] | [11008, true] |
| vllm_limits | [4702, 75232, 4702, 1, 0, 0.0, true] | [93853, 1501648, 93853, 1, 0, 28360704.0, true] |
| vllm_running_1024 | 58 | 808 |
| vllm_running_8192 | 8 | 166 |

## synth:vision-enc-dec-gpt2 (encoder-decoder)

| figure | before | after |
|---|---|---|
| conv_kv_size | 131072 | 262144 |
| conv_layers | [16, 16] | [12, 32] |
| eng_None_32768_kv_mb | 701.80864 | 744.81664 |
| eng_None_32768_params | 808394240 | 803151360 |
| eng_None_32768_total_mb | 2484.691558 | 2517.213798 |
| eng_None_4096_kv_mb | 114.60608 | 157.61408 |
| eng_None_4096_params | 808394240 | 803151360 |
| eng_None_4096_total_mb | 1814.441779 | 1846.964019 |
| eng_llamacpp_32768_kv_mb | 701.80864 | 744.81664 |
| eng_llamacpp_32768_params | 808394240 | 803151360 |
| eng_llamacpp_32768_total_mb | 2484.691558 | 2517.213798 |
| eng_llamacpp_4096_kv_mb | 114.60608 | 157.61408 |
| eng_llamacpp_4096_params | 808394240 | 803151360 |
| eng_llamacpp_4096_total_mb | 1814.441779 | 1846.964019 |
| report_kv_gb | 0.11460608 | 0.15761408 |
| report_params | 808394240 | 803151360 |
| report_total_gb | 1.856430039 | 1.896632279 |
| report_weight_gb | 1.61678848 | 1.60630272 |
| sim_max_running | 72 | 87 |
| slice_mib_16x4352 | [81582, false] | [76032, true] |
| vllm_limits | [4702, 75232, 4702, 1, 0, 0.0, true] | [5662, 90592, 5662, 1, 0, 0.0, true] |
| vllm_running_1024 | 58 | 70 |
| vllm_running_8192 | 8 | 10 |

## synth:whisper-large-v3 (encoder-decoder)

| figure | before | after |
|---|---|---|
| conv_kv_size | 262144 | 81920 |
| conv_state_bytes | 0 | 245760000 |
| disagg_transfer_bytes_640 | 335544320 | 350617600 |
| genz_chunked_kv_mb | 2304.0 | 1188.75 |
| genz_decode_kv_mb | 2048.0 | 874.375 |
| genz_decode_weights_mb | 10005.203125 | 1526.625977 |
| genz_kv_size | 262144 | 81920 |
| genz_prefill_kv_mb | 2048.0 | 1108.75 |
| genz_prefill_weights_mb | 10410.40625 | 3053.251953 |
| kv_cross_layers | 0 | 32 |
| kv_gb@32768 | 0.0 | 5.61446912 |
| kv_gb@4096 | 0.0 | 0.91684864 |
| kvgeo_32768 | [0.0, 32, 0, 0.0, 16, 0.0] | [163840.0, 32, 0, 0.0, 16, 245760000.0] |
| kvgeo_4096 | [0.0, 32, 0, 0.0, 16, 0.0] | [163840.0, 32, 0, 0.0, 16, 245760000.0] |
| prefix_kv_bytes_per_token | 524288 | 163840 |
| serving_bytes_per_token | 524288 | 163840 |
| serving_requests_8gib_1k | 16 | 20 |
| serving_state_bytes | 0 | 245760000 |
| sim_max_running | 400 | 171 |
| slice_mib_16x4352 | [22272, true] | [26624, true] |
| vllm_limits | [27344, 437504, 27344, 1, 0, 0.0, true] | [27344, 437504, 27344, 1, 0, 245760000.0, true] |
| vllm_running_1024 | 341 | 157 |
| vllm_running_8192 | 51 | 43 |

## synth:xlstm (state-space)

| figure | before | after |
|---|---|---|
| conv_kv_size | 262144 | 0 |
| conv_state_bytes | 0 | 134480896 |
| disagg_transfer_bytes_640 | 335544320 | 134480896 |
| eng_None_32768_params | 6646931456 | 6651125760 |
| eng_None_32768_state_mb | 0.0 | 134.480896 |
| eng_None_32768_total_mb | 14179.699917 | 14322.569421 |
| eng_None_4096_params | 6646931456 | 6651125760 |
| eng_None_4096_state_mb | 0.0 | 134.480896 |
| eng_None_4096_total_mb | 13736.781414 | 13879.650918 |
| eng_llamacpp_32768_params | 6646931456 | 6651125760 |
| eng_llamacpp_32768_state_mb | 0.0 | 134.480896 |
| eng_llamacpp_32768_total_mb | 14179.699917 | 14322.569421 |
| eng_llamacpp_4096_params | 6646931456 | 6651125760 |
| eng_llamacpp_4096_state_mb | 0.0 | 134.480896 |
| eng_llamacpp_4096_total_mb | 13736.781414 | 13879.650918 |
| facts_full_attention | true | false |
| facts_layers | [32, 0, 0, null] | [0, 0, 32, null] |
| genz_chunked_kv_mb | 2304.0 | 256.501953 |
| genz_decode_kv_mb | 2048.0 | 128.250977 |
| genz_decode_weights_mb | 9993.0 | 9997.0 |
| genz_kv_size | 262144 | 0 |
| genz_layer_kinds | {"MHA-global": 32} | {"Recurrent": 32} |
| genz_prefill_kv_mb | 2048.0 | 128.250977 |
| genz_prefill_weights_mb | 10386.0 | 10390.0 |
| kv_free_layers | 0 | 32 |
| kv_full_layers | 32 | 0 |
| kv_source | "uniform" | "layer_plan:mlstm_stack" |
| kvgeo_32768 | [0.0, 32, 0, 0.0, 16, 0.0] | [0.0, 1, 32, 4202528.0, 16, 0.0] |
| kvgeo_4096 | [0.0, 32, 0, 0.0, 16, 0.0] | [0.0, 1, 32, 4202528.0, 16, 0.0] |
| model_type | "unknown" | "state-space" |
| prefix_kv_bytes_per_token | 524288 | 0 |
| report_params | 6646931456 | 6651125760 |
| report_state_gb | 0.0 | 0.134480896 |
| report_total_gb | 13.825365115 | 13.995130798 |
| report_weight_gb | 13.293862912 | 13.30225152 |
| serving_bytes_per_token | 524288 | 0 |
| serving_requests_8gib_1k | 16 | 63 |
| serving_state_bytes | 0 | 134480896 |
| sim_max_running | 103 | 75 |
| slice_mib_16x4352 | [66304, true] | [70912, true] |
| state_gb | 0.0 | 0.134480896 |
| vllm_limits | [6707, 107312, 6707, 1, 0, 0.0, true] | [378, 102816, 378, 1, 0, 142606336.0, false] |
| vllm_running_1024 | 83 | 63 |
| vllm_running_8192 | 12 | 11 |

