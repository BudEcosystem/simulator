# Proof: KV layer kinds -- before and after

Branch `fix/hybrid-kv-layer-kinds`, measured against **origin/main `c17ef89`** (what this branch
lands on). Generated on the shared GB10 box, 27-29 Oct 2026, with `scripts/kv_corpus.py`; no GPU
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
every reader of the layer kinds, every fixture folder, synthetic edge shapes); and the ninth
review's (GenZ's runtime modelling, which raised on every encoder-decoder; the decode fit check's
resident weights; a decoder-only stack's own cross-attention layers and Pix2Struct's image
encoder in every reader; a nested flan-T5's parameter count and gated FFN; and this proof's
coverage of the runtime modelling); and the tenth review's (the configs as the hub ships them:
Pix2Struct's and BLIP's decoder beside its image tower under a stale `is_encoder_decoder: false`,
the facts' head count under `num_heads`, MusicGen's cross-attention hidden by its decoder block's
own flags in the calculator, a BART-family decoder block's depth and a [height, width] staged
image tower in Donut and Nougat, and Mllama's one-image assumption stated); and the eleventh
review's (**the gate**: a committed test that fails on any figure below origin/main's -- or a
capacity above it -- that an allowlist does not name with both figures and a ground truth, and
the readers it then found below their truth: GenZ's missing shared experts, MLA cache, V3.2
indexer and attention output gate; the FRD-027 facts' encoder at the decoder's width, its missing
Q-Former and its four-spelling expert count; a flat mT5's shape in the parameter counter; and the
weights of towers, codebooks, learned positions, conformers and wrappers; section 5); and the
twelfth review's (the gate's own ground truths and reach: truths of 0 KV for ProphetNet and
MusicGen, a gate that passed only with transformers 5.12 installed, model-card totals as truths,
figures lost or new on the branch, the chunked graph's truth, and no GGUF in the committed
corpus; section 6); and the thirteenth review's (the serving path's weights -- a ServingSimulator's
weight load and the cluster's and disaggregation's B_eff -- counted from the decoder fields alone,
which the gate did not measure; per-token truths of 0 for a stack whose every layer is windowed; a
missing module of any kind accepted as "not measured"; a nested decoder spelled the GPT-2 way;
section 7).

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
* Round 10 (the ninth review, of `66a5c40`): `tests/test_cross_attention_every_reader.py`, 37
  cases, and 2 BudSimulator converter cases. Against the round-9 head (its `src`, these tests):
  32 failed, 5 passed; both converter cases failed. The 5 that pass there are guards: Mllama's
  calculator figure (already right) and GenZ's runtime modelling for Mllama, which round 9 graphed
  as 40 self-attention layers and so did not reach the new operators; and Mllama's decode weights,
  which hold no encoder. The 32:
  - `test_runtime_modelling_runs_every_cross_attention_shape` x 15 (prefill / decode / chunked x
    Whisper-turbo, Dia, flan-T5-xl, BART, Pix2Struct): `Layer Name:Enc QKV` / `Cross Q not found
    in the breakdown function`.
  - `test_the_decode_graph_weighs_the_resident_model` x 2 (Whisper large-v3 1,526.6 MiB against
    the prefill graph's 2,926.6 without its input table; flan-T5-xl) and
    `test_a_decode_step_that_does_not_fit_is_refused` (a 3 GB device passed the decode check).
  - `test_the_table_reads_the_cross_cache_wherever_the_config_puts_it` x 2 (Mllama: none;
    Pix2Struct: `tokens` None), `test_the_calculator_charges_self_and_cross[pix2struct]`
    (75,497,472 B at 1k, truth 188,743,680), `test_genz_charges_self_and_cross` x 2 and
    `test_model_facts_charge_self_and_cross` x 2 (Mllama 167,772,160 B, truth 268,566,528),
    `test_genz_graphs_an_interleaved_cross_layer_without_a_self_cache`.
  - `test_a_nested_flan_t5_is_counted_with_its_own_heads_and_both_stacks` (1,595,173,376, truth
    3,876,648,448), `test_the_attention_head_count_is_read_with_the_head_width` (12 heads),
    `test_a_stated_gated_ffn_is_three_matrices_everywhere` x 2 (4,475 MiB, a 2-matrix FFN),
    `test_a_config_written_by_to_dict_is_counted_not_defaulted` x 2 (MllamaConfig() and
    Pix2StructConfig() counted 123,532,032: `architectures: null` raised in the counter, which
    fell back to a default -- on origin/main too; added after the first corpus pass showed it).
  - BudSimulator `test_a_database_model_holds_its_cross_attention_wherever_the_config_puts_it` x 2
    (Mllama 167,772,160 B, Pix2Struct 75,497,472 B).
* Round 11 (the tenth review, of `a4615e0`): `tests/test_cross_attention_hub_shapes.py`, 32 cases
  over five hub configs saved as `tests/fixtures/cross_attention/*_hub.json` (google/deplot,
  Salesforce/blip-image-captioning-base, facebook/musicgen-small, naver-clova-ix/donut-base,
  facebook/nougat-small) plus a walk over every config fixture and Mllama's one-image note.
  Against the round-10 head (its `src`, these tests, `-o pythonpath=`): 27 failed, 5 passed. The
  5 are guards: MusicGen's table reading (raw config), its facts and its GenZ figure (the readers
  that already read the raw config), and the decoder view of BLIP and deplot (no cross cache on
  either side). The 27:
  - `test_the_table_reads_the_hub_shape` x 4: deplot and BLIP none (a `text_config` with a stale
    `is_encoder_decoder: false`); Donut and Nougat 12 decoder layers (the `mbart` block's
    `num_hidden_layers`) and `tokens` None ([height, width] `image_size`).
  - `test_the_decoder_view_keeps_the_cross_cache` x 3 (MusicGen: None after the view; Donut,
    Nougat: 12 encoder layers lifted from the `mbart` block) and
    `test_every_fixture_keeps_its_cross_cache_through_the_decoder_view` (MusicGen's fixture).
  - `test_the_calculator_charges_self_and_cross` x 5 and `..._breakdown_shows_the_cross_layers`
    x 5 (deplot 0 B, truth 188,743,680 at 1k; MusicGen 100,663,296, truth 201,326,592; Donut
    33,554,432, truth 95,420,416).
  - `test_model_facts_charge_self_and_cross` x 4 (deplot 18,432 B a token, truth 36,864: 6
    default heads; Donut and Nougat 49,152 B, 12 layers), `test_genz_charges_self_and_cross` x 4.
  - `test_interleaved_cross_layers_say_they_hold_one_image` (no note).
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
| `llm-memory-calculator/tests` (pytest) | 2,091 passed, 14 skipped, 1 failed | 2,548 passed, 21 skipped, 1 failed |
| `BudSimulator/tests` (pytest) | 24 passed | 86 passed |

(Round 14 adds the 45 calculator cases and the 5 BudSimulator cases of section 7, and updates one
pinned count to its checkpoint (`test_hybrid_parameter_count.py::test_kda_gates_are_low_rank_not_dense`:
KDA's `b_proj`); the one failure is the schema test below, which fails on origin/main too. The
suite ran as `tests --deselect tests/test_kv_gate.py` (2,539 passed, 21 skipped, 1 failed) beside
`tests/test_kv_gate.py` on its own (9 passed), 28 min. Round 13 ended at 2,503 passed, 21
skipped, 1 failed / 81. Round 13 adds the 28 cases of section 6 and the GGUF-fixture case; the one failure is the
schema test below, which fails on origin/main too. Round 12 ended at 2,475 passed, 21 skipped / 81.
Round 12 adds the 66 cases of `tests/test_genz_weights_against_transformers.py`,
`tests/test_weights_around_the_decoder.py`, `tests/test_gate_found_readers.py` and the gate,
`tests/test_kv_gate.py`; one existing assertion was tightened and widened by the Q-Former the
counter now counts (`test_cross_attention_every_reader.py::test_a_nested_flan_t5_is_counted_with_its_own_heads_and_both_stacks`:
from 0.95-1.01x to 1.00-1.02x of transformers' count). Its run took 30 min. The schema test that
failed on both sides before passed in this run (`test_committed_schemas_are_current`; nothing in
round 12 touches the schemas). Round 11 ended at 2,408 passed, 21 skipped, 1 failed / 81.

(Round 11 adds the 32 cases of `tests/test_cross_attention_hub_shapes.py`; no existing test changed;
its run took 30 min. Round 10 ended at 2,376 passed, 21 skipped / 81. Round 10 adds the 37 cases of
`tests/test_cross_attention_every_reader.py` and the 2 BudSimulator
converter cases above; no existing test changed. Its run took 53 min on the shared box, 27 of them
in `tests/planner/test_grounding.py`'s `test_family_hold_out` and `test_fit_is_deterministic`, whose
profile -- the grounding store's lookups -- is the same on the round-9 head. Round 9 ended at
2,339 passed, 21 skipped / 79. Round 9 adds the 49 cases of `tests/test_encoder_decoder_cross_attention.py` and 14 BudSimulator
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

**Result (round 12): 262 rows -- 260 configs, the empty-row marker and the not-a-language-model
marker -- 121 unchanged, 141 changed** against origin/main; **0 gate failures** over all of them
(section 5). Round 12 adds five figures to every row (the facts' weight, KV a token, KV at 4k and
state bytes; the converter's graph weights) and stops measuring eight configs that state no token
vocabulary (two TTS speech tokenizers, Mimi, w2v-BERT and three VieNeu-TTS codecs: not language
models; listed in `_not_language_models`). Against the round-11 head measured by the same script,
102 configs changed and 160 did not; no uniform dense decoder's KV moved (only the MLA latent and
Hunyuan's, which the facts and GenZ could not read before).

**Result (round 11): 269 rows -- 268 configs and the empty-row marker -- 144 unchanged, 125
changed** against origin/main, over every figure (round 11 adds the five `cross_attention/*_hub.json`
fixtures, all of which move; against the round-10 head those five changed and the other 264 rows
did not, below. Round 10 ended at 264 rows, 144 / 120; round 10 adds the runtime modelling to every
row, the three `cross_attention/` fixtures and three synthetic shapes; round 9 ended at 258 rows,
144 / 114). Of the 226 shipped, cached and built-in configs, 89 changed and 137 did not; of the 37
synthetic shapes, 31 changed and 6 did not (a dense Llama with and without
`is_encoder_decoder: false`, Mistral's window, Falcon-40B, an audio-LLM, and GPT-2: GenZ reads
no GPT-2 spelling on either tree). Every config whose figures moved moved in a figure measured
before round 9 too: no change is confined to the planners' and the prefill graph's new figures
(the prefix bytes, `KVGeometry`, the vLLM planner, its scheduler and slice sizing follow the KV,
state and cross cache already in the table). Against the round-9 head, 33 changed and 231 did
not (round 10, below). Round 9 against the round-8 head: 31 changed and 227 did not:
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
  fall by the experts those layers do not have. That alone left DeepSeek-V3 **below** its
  checkpoint (1,273,240 MiB decode, 667.6B, against transformers' 671.03B for V3-Base): the graph
  had no shared expert, which origin/main's extra MoE layers had hidden. Round 11 graphs it (and
  V3.2's indexer): DeepSeek-V3-Base prefill 1,279,879 MiB against transformers' 1,279,881 (its
  norm vectors are the difference; section 5).
* **The GGUF reader states what the file states**: RoPE, a gated FFN, tied embeddings. Parameter
  counts against the files' own tensors: Qwen3.5-2B 1.901B -> 1.856B (truth 1.882B; round 12
  reads its Gated DeltaNet as one, and its attention's output gate: **1.8819B**), gpt-oss-20b
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

Round 10 (the ninth review, of `66a5c40`) moved 33 configs against the round-9 head (`--diff` of
the round-9 head's corpus against this branch's, both measured with round 10's script) -- the
encoder-decoders, the new cross-attention fixtures and synthetic shapes -- and nothing else; no
decoder-only config without cross-attention layers moved in any figure, and no figure that
measured before fails now. Round 10's script adds GenZ's runtime modelling to every row
(`runtime_prefill` / `runtime_decode` / `runtime_chunked`: latency ms, weights MB, KV MB, or the
error; `decode_fit_refused`: the decode check on a device of 0.6 x the prefill graph's weights),
reads the disaggregation transfer from `DisaggregationAnalyzer._estimate_kv_bytes` itself, and
adds three synthetic shapes and the `cross_attention/` fixtures (transformers 5.12's own
MllamaConfig, Pix2StructConfig and Blip2Config with flan-T5); 264 rows in all.

* **GenZ's runtime modelling answers for every encoder-decoder.** On the round-9 head it raised for
  all 30 encoder-decoder rows in prefill, decode and chunked (90 runs, `Layer Name:Enc QKV` /
  `Cross Q not found in the breakdown function`); now none does. Against origin/main, no run that
  answered there fails here, and 53 runs that failed there answer (the encoder-decoders origin/main
  could not graph, and the hybrids earlier rounds fixed). The runs that still fail are the same on
  origin/main: models too large for the 200 GB device, GenZ's built-in Jamba graphs ("More than 1
  unique layers"), GPT-Neo's and Hunyuan's per-layer lists, w2v-BERT, a VisionEncoderDecoder with
  a GPT-2 decoder (no GPT-2 spelling in GenZ).
* **The decode and chunked graphs hold the encoder** (and the cross K / V projections), at no step
  time, so the fit checks weigh the resident model. Decode weights round 9 -> this branch (prefill
  graph, which has the input table): whisper-large-v3 1,526.6 -> **2,926.6 MiB** (3,053.3; origin/main
  10,005.2, every layer of both stacks graphed as one decoder); whisper-large-v3-turbo 301.6 ->
  1,526.6 (1,653.3); whisper-base 92.6 -> 134.6 (185.3; origin/main 2,205.2); flan-T5-xl 2,237.5
  -> 5,309.5 (5,435.0); Dia 2,380.0 -> 3,004.0 (3,008.0); cohere-transcribe 256.0 -> 2,096.0
  (2,128.0). The difference left to the prefill graph is the input embedding table, which no GenZ
  decode graph carries for any model (LAYER_KINDS.md). `decode_fit_refused` is True for every
  encoder-decoder (round 9: the decode run raised before the check).
* **A decoder-only stack's own cross-attention layers** (MllamaConfig(), and two synthetic
  shapes): GenZ layer kinds 40 x `MHA-global` -> 32 `MHA-global` + 8 `Cross`; converter and GenZ
  `get_kv_size` 81,920 -> 65,536 elements a token with 134,348,800 B fixed a request (origin/main's
  converter 262,144, every layer full, no fixed); the facts [40 full] -> [32 full] + the cross
  cache; serving 1,024-token requests in 8 GiB 51 -> **31** (8 GiB / (134.2 + 134.3 MB));
  vLLM running at 1k 258 -> 179 (origin/main 258), at 8k 39 -> 43 (the cross cache no longer
  grows past the 4,100 vision tokens); `KVGeometry` 163,840 -> 131,072 B a token plus 134,348,800
  B a sequence. `calculate_kv_cache` does not move (0.671 GB at 4k): the calculator already priced
  them.
* **Pix2Struct's image encoder** (`vision_config.seq_len`, 4,096): the cross cache is fixed at
  the stated length, not charged over the sequence. Pix2StructConfig(): converter / GenZ 36,864 ->
  18,432 elements a token + 150,994,944 B a request (at 1k: 75,497,472 -> **188,743,680 B**, the
  truth); serving requests in 8 GiB 113 -> 45; the calculator at 32k 2.416 -> 1.359 GB (the cross
  cache over 4,096, not 32,768); its decode graph holds the encoder (226.6 -> 379.6 MiB). The
  synthetic Pix2Struct-large shape moves the same way, and its encoder is graphed and counted at
  the image encoder's own width (1,536), not the decoder's.
* **Parameter counts**: BLIP-2 with flan-T5-xl 1,595,173,376 (origin/main 1,720,216,064) ->
  **3,835,230,720** (truth 3,876,648,448: both T5 stacks, the cross-attention, 32 heads); the
  synthetic BLIP-2 / InstructBLIP shapes (no `feed_forward_proj`) 1,538,943,488 -> 3,275,684,352,
  Florence-2 290,113,024 -> 491,687,424; MllamaConfig() 123,532,032 (a fallback on every tree)
  -> 10,943,653,376 (truth 10,642,941,475), Pix2StructConfig() 123,532,032 -> 257,280,000 (truth
  282,285,696). The engine estimate's parameters and totals follow. flan-T5-xl's GenZ prefill
  weights 4,475 -> **5,435 MiB** (2,783,959,040 parameters + GenZ's tied classifier = 5,435.5).
* **A stated encoder length is a fixed cost in `marginal_bytes_per_token`** (found while fixing:
  a cross group whose stated length was at least the sequence counted as growing, and
  `KVGeometry` then charged it twice -- per token and per sequence): cohere-transcribe's 5,000
  positions at 4k, `KVGeometry` 65,536 -> 32,768 B a token (the self-attention), and Mllama's
  vision tokens likewise.

Round 11 (the tenth review, of `a4615e0`) moved 5 configs against the round-10 head (`--diff` of
the round-10 head's corpus against this branch's, both measured with the same script, database and
fixtures) -- exactly the five hub fixtures it adds -- and nothing else: no fixture, cached, shipped,
GGUF, built-in or synthetic config moved in any figure. Against origin/main, round 10 -> round 11:

* **Pix2Struct and BLIP as the hub ships them.** google/deplot: calculator 0 -> 0.302 GB at 4k
  (origin/main 0); the facts 18,432 B a token (6 default heads), the converter and GenZ 49,152 elements (12 self + 12 cross over the sequence) -> 18,432 elements (36,864 B)
  a token + 150,994,944 B a request; serving requests of 1k in 8 GiB 85 -> **45** (truth 45);
  vLLM running at 8k 477 -> 160 (origin/main 238, which charged 6 heads of 128 per token and no
  cross); `KVGeometry` 0 -> 36,864 B a token; parameters 247,842,816 -> 257,280,000 (the
  cross-attention). BLIP (blip-image-captioning-base): the cross cache over 577 image tokens is
  now charged, 21,270,528 B a request (calculator 0.151 -> 0.172 GB at 4k; serving 227 -> 145;
  vLLM at 8k 239 -> 223); its parameters 194,795,520 -> 223,107,072 (the 12 cross-attention
  blocks; origin/main 123,532,032, a fallback).
* **MusicGen's cross-attention in the calculator.** musicgen-small: `kv_cross_layers` 0 -> 24,
  `calculate_kv_cache` 0.403 -> **0.805 GB** at 4k (origin/main 0.0), `KVGeometry` 98,304 ->
  196,608 B a token -- now the figure the facts, the vLLM planner, GenZ and the serving model
  already charged. The facts count its text encoder and cross-attention weights (the decoder view
  no longer says `is_encoder_decoder: false`): vLLM blocks 47,146 -> 46,826, running at 1k 226 ->
  225.
* **Donut and Nougat.** The decoder is 4 layers (`decoder_layers`), not 12, in the facts,
  `KVGeometry`, GenZ and the converter (GenZ `get_kv_size` 49,152 -> 8,192 elements), and its
  cross cache is over the Swin's 4,800 / 588 tokens, not the sequence: donut-base at 4k 0.268 ->
  **0.146 GB**, serving 85 -> 90, vLLM at 8k 151 -> 342; nougat-small 0.268 -> 0.077 GB, 85 ->
  325. origin/main charged both the 32-layer defaults (0 GB in the calculator). Parameters
  944,602,880 -> 810,360,576 (origin/main 874,782,720).
* **Mllama** does not move: the readers already charged one image's 6,404 tower tokens; the plan
  now says so in a note, and LAYER_KINDS.md lists it as not a bound for multi-image requests.

Found while checking round 11 and fixed in round 12 (section 5): a staged image
tower's encoder depth was its stage count (Swin's `num_layers`, 4, not `sum(depths)`, 20) at the
last stage's width; the facts sized an encoder-decoder's encoder at the decoder's width;
Pix2Struct's always-gated FFNs were graphed and counted as two matrices; the counter had no
Q-Former term.

Found while checking round 10, not changed by it (present on origin/main): the decode graph has no
input embedding table for any model (its fit check is short by `vocab x hidden`, 1,002 MiB on
Llama-3-8B).

Found while checking round 9, not changed by it (present on origin/main): GenZ reads no GPT-2
spelling (`n_layer` / `n_embd` / `n_head`): a GPT-2 or StarCoder config, and a VisionEncoderDecoder
with a GPT-2 decoder, are graphed from the 7B defaults or refused, as on origin/main (a dense
dialect, not a layer kind: left for a separate fix); Gemma 3n's per-layer `intermediate_size` list
fails GenZ and the vLLM planner (a TypeError on origin/main too). For an audio encoder that states
no output length (Moonshine) the sequence is not a bound on its cross cache (docs/LAYER_KINDS.md,
"What is conservative").

## 5. Round 12: the gate, and what it found

The eleventh review found no committed check behind the "no regression" claim: 108 configs moved
in the unsafe direction against origin/main -- a memory figure down, or a capacity figure (vLLM's
blocks, tokens and running sequences, the serving path's requests, the scheduler's peak) up --
with only grouped causes, and some readers had gone from above their ground truth to below it.

**The gate.** `tests/test_kv_gate.py` measures every config the repository ships (216 rows:
prepopulated.db's, every fixture, the synthetic shapes, GenZ's built-in models) through every
reader, against `tests/fixtures/kv_gate/origin-main.json` (the same harness on `c17ef89`). Each
unsafe move must be named in `tests/fixtures/kv_gate/allowlist.json` with its config, reader,
origin/main's and the branch's figures, its cause, and a ground truth the branch's figure is not
below; a capacity rise names the memory figures it is sized from, each held to its own truth.
`kv_corpus.py --gate` runs the same check over the machine's HF cache, model folders and GGUFs:

    python llm-memory-calculator/scripts/kv_corpus.py --gate main.json branch.json \
        --allowlist llm-memory-calculator/tests/fixtures/kv_gate/allowlist.json

Over the whole corpus: **953 unsafe moves in 107 configs, every one allowlisted and at or above
its ground truth; 990 entries** (with the 37 memory figures the capacity rises are held by);
**0 failures**. Against the round-11 head's source the committed test fails with 217.

**The ground truth.** `scripts/kv_truth.py` builds transformers 5.12's own model for each config on
the meta device: every parameter (the language model's weight matrices for GenZ's graphs, which
graph no norm or bias vector -- held with a slack of 5 parts in 100,000 -- and less the input
embedding a decode step gathers from for the decode graph), and per decoder layer the K and V its
self-attention caches (its own projections, or a fused QKV less its queries, or MLA's latent) with
the kinds and windows `DynamicCache(config=...)` builds. A figure's truth is its reader's workload
on that profile (KV at the reader's tokens, a per-token rate on the full layers, weights). Where
transformers cannot build the config as written, `tests/fixtures/kv_gate/truths.json` states the
truth and its source: the config built with a key its class takes (BLIP-2's `blip-2`, MiniMax's
`layer_types` from `attn_type_list`, Mllama with its pad token unset), the checkpoint's or the
GGUF's own tensors, or the published or config arithmetic it cites (ChatGLM3-6B 6,243,584,000;
deepseek-moe-16b; SmallThinker; the hub cards of Kimi-Linear, Ling-flash, Phi-4-mini-flash,
Hymba).

**What the gate found, and the fix** (each red first: `tests/test_genz_weights_against_transformers.py`,
`tests/test_weights_around_the_decoder.py`, `tests/test_gate_found_readers.py`: 62 cases, 41 of which
fail on the round-11 head):

| figure | origin/main | round-11 head | round 12 | ground truth |
|---|---|---|---|---|
| DeepSeek-V3 (planner fixture, V3.2) GenZ prefill weights, MiB | 1,337,262 | 1,275,007 | **1,281,503.5** | 1,281,505.4 (transformers; vectors the rest) |
| DeepSeek-V3-Base (database) GenZ prefill, MiB | 1,337,262 | 1,275,007 | **1,279,879** | 1,279,881 |
| granite-4.0-h-small GenZ prefill, MiB | 61,590.5 | 60,774.2 | **62,214** (untied head counted twice) | 61,430.6 |
| Llama-4 Maverick (synthetic) GenZ prefill, MiB | 992,806 | 508,936 | **764,296** | 764,297 |
| Qwen1.5-MoE-A2.7B GenZ prefill, MiB | -- | 25,720.6 | **27,304.6** | 27,305.2 |
| Qwen3-Next-80B GenZ prefill, MiB | -- | 151,486 | **151,966.4** | 151,966.8 |
| DeepSeek-V3 GenZ decode / chunked KV at 4k, MiB | 7,686 (chunked) | 152.5 / 171.6 | **274.5 / 308.8** | 274.5 (61 x 576 latent) |
| donut-base facts weights / 2, M | 17,442 | 193.1 | **217.0** | 201.85 |
| nougat-small facts weights / 2, M | 17,442 | 236.6 | **260.4** | 247.38 |
| BLIP-2 flan-t5-xl facts weights / 2, M | 4,276.7 | 3,824.6 | **3,881.6** | 3,876.65 |
| cohere-transcribe facts weights / 2, M | 302 | 805 | **2,641.6** | 2,048.9 |
| mT5 (flat, no flag) parameters, M | 153.1 | 142.8 | **247.5** | 172.1 |
| BLIP-2 / DePlot parameters, M | 1,720.2 / 123.5 | 3,835.2 / 257.3 | **3,942.6 / 282.4** | 3,876.6 / 282.3 |
| SmallThinker-21B facts weights / 2, M | 5,958 | 5,958 | **21,506.6** | 21,506.3 (arithmetic) |
| Llama-4 Scout / Maverick (synthetic) parameters, B | 0.152 | 0.152 | **107.77 / 400.71** | 107.77 / 400.71 |
| ChatGLM3 GenZ prefill, MiB | 10,100 | 8,900 | **13,588** | 11,908 |
| Qwen3.5-2B GGUF parameters, B | 1.901 | 1.856 | **1.8819** | 1.8818 (its tensors) |
| Nougat / Donut GenZ prefill, MiB | (failed) | 323.3 / 352.7 | **471.4 / 496.9** | 471.4 / 384.6 |
| Mllama (transformers default) facts weights / 2, M | 10,412 | 10,412 | **10,643.1** | 10,642.9 |

The causes, each read from the config as stated, through the one reader every caller uses:

* **Shared experts** (`layer_plan.shared_expert_width`): DeepSeek's and Kimi's `n_shared_experts`
  x the expert width, Qwen MoE's `shared_expert_intermediate_size` (one expert, not four),
  Granite 4's `shared_intermediate_size`, Ling's `moe_shared_expert_intermediate_size`, Llama 4's
  and Hunyuan's class-built one. GenZ's prefill and decode graphs never had them (only the
  chunked graph, at DeepSeek's dense 18,432 width instead of 2,048); the built-in DeepSeek V2 /
  V3 and Qwen1.5-MoE stated those wrong widths too. The facts and the parameter counter read the
  same reader. Routed experts: `layer_plan.routed_expert_count` / `routed_expert_width` for
  GenZ, the facts and the converter (which had read DeepSeek's `n_routed_experts` as a dense
  model).
* **MLA's cache is its latent** in every GenZ graph (`attention.mla_latent_op`); V3.2's indexer
  weights (`attention.mla_indexer_ops`). Its indexer key cache stays out of every reader, as on
  origin/main (LAYER_KINDS.md, "What errs low, measured").
* **An attention output gate** (`layer_plan.attention_output_gate`: Qwen3-Next, Qwen3.5 and
  their GGUF architectures), in GenZ, the facts and the counter.
* **Gating and tying from the config class** (`layer_plan.ffn_activation`,
  `config_class_default`): a config that leaves `hidden_act` or `tie_word_embeddings` out takes
  its class's default; a family table (`GATED_FFN_MODEL_TYPES`, the counter's, now shared with
  GenZ) for families whose activation never says so (SmallThinker, ChatGLM, Pix2Struct's
  `wi_0` / `wi_1`). This moves dense synthetic and database rows' weights **up**, toward their
  truth (Llama-3-8B with no tie flag 7.54B -> 8.06B, real 8.03B; GLM-4-9B 7.0B -> 9.7B, real 9.4B).
* **Weights beside the decoder** (`stack_params`): an encoder at its own geometry -- a
  hierarchical tower's stages (`layer_plan.encoder_stages`; its depth 20, not Swin's 4 stages, in
  every reader), a conformer's convolution module, a tower's patch / frame embedding and
  positions, a nested encoder's own token table, a composite's `enc_to_dec_proj` -- and an adapter
  stack (BLIP-2's Q-Former); a tower's class defaults where its config is silent (Mllama's global
  layers, output width and tile tables); an audio decoder's codebooks (`decoder_streams`: Dia,
  MusicGen); a learned position table (`learned_positions`: MBart, Whisper, GPT-2, BERT-style).
* **One shape for every reader**: a flat config that states two stacks is an encoder-decoder to
  `detect_model_type` too (mT5); a wrapper around a language model alone is counted from its
  `text_config` (the nested Nemotron-H, Llama 4); ChatGLM's `ffn_hidden_size` and
  `padded_vocab_size` (`layer_plan.ffn_width` / `vocab_size`) in GenZ and the converter; a qwen35
  GGUF's `ssm.*` keys read as the Gated DeltaNet they describe (`gguf_layers`); GenZ's built-in
  Gemma-3-4B's head width (256).

The proof's own statements are corrected where they had called a figure below its truth correct
(section 4: DeepSeek-V3's "667.6B"; LAYER_KINDS.md's "every unknown errs high", which holds for
layer kinds and is now stated beside the weights that err low, each measured). What stays below
its truth without having moved below origin/main is listed in LAYER_KINDS.md, "What errs low,
measured".

## 6. Round 13: the gate's own truths, and what it did not reach

The twelfth review checked the gate itself. Each finding, and what was done (red first:
`tests/test_kv_truth.py`, `tests/test_config_class_defaults.py`, the new cases of
`tests/test_kv_gate.py` and `tests/test_gate_found_readers.py`: 28 cases, 23 of which fail on
the round-12 head; the 5 that pass there pin what did not change -- the prefill graphs' truths, a
recurrent mixer's 0, the hand truths' sources, a key no reader asks for. Beside them
`test_the_gguf_fixtures_are_read_through_the_gguf_reader` runs on the committed corpus):

* **Truths of 0 KV for two encoder-decoders that cache** (ProphetNet: its `key_proj` /
  `value_proj` were not in the truth's projection names; MusicGen: the decoder's "layer list" was
  its 4 `lm_heads`). Both read now: ProphetNet 12 x 2 x 1,024 = 24,576 elements a token, MusicGen
  24 x 2 x 1,024 = 49,152 (`synth:prophetnet` facts 49,152 B a token = its truth;
  MusicGen's 98,304 = its truth). For the whole category: a softmax attention whose projections
  the truth does not name leaves the width undetermined (no truth, so the entry fails), never 0; a
  layer list is a list of blocks; and a layer the class builds as a convolution where the cache
  expects attention is undetermined too -- which found the same fault in two more configs,
  LFM2.5-Audio's detokenizers (transformers' Lfm2 reads only `full_attention` and built their
  `sliding_attention` layers as convolutions): their truth is now the checkpoint's own K / V
  tensors (3 layers x 512, window 30). No truth of 0 is left but where the model holds nothing of
  that kind (Mamba, RWKV, xLSTM and nomic-bert KV; GPT-Neo, Hunyuan and text encoders' fixed state;
  the per-token rate of an all-windowed stack: Voxtral-Realtime, the detokenizers).
* **The gate passed only with transformers 5.12 installed.** The readers took a config class's
  defaults from the installed transformers only; without it, or with 4.50, Mllama's tower lost its
  global layers and RWKV / Llama 4 their untied heads (weights below origin/main's), and GenZ's
  graphs read other activations and position tables (the "allowlist says" mismatches). The
  defaults now ship with the package (`config_class_defaults.json`: transformers 5.12's 642
  config classes, every key `layer_plan.CLASS_DEFAULT_KEYS` names; `scripts/config_class_defaults.py`
  rewrites it, and `test_the_snapshot_is_the_installed_transformers_5_12` checks it wherever 5.12
  is installed). Not one figure moved with transformers 5.12 installed; without transformers the
  Mllama synthetic's parameters are 1,370,625,536 (were 1,137,317,376), as with it. The gate now
  passes in all three environments -- 5.12, 4.50 and none (a clean venv without transformers,
  torch, gguf or lxml): **0 failures** each. A reader whose third-party dependency is not installed
  is reported as not measured there, by module and figure count (no transformers: the GGUF
  reader's `gguf` 248 figures, the converter's `requests` 376; 4.50: `gguf` 248, `lxml` 376); any
  other error, or a missing module of this repository, fails.
* **Model-card totals as truths.** Ling-flash-2.0, Kimi-Linear, Hymba and Phi-4-mini-flash were
  held to their cards' 100B / 48B / 1.52B / 3.85B. Each is now its checkpoint's own tensors, read
  from every safetensors shard's header on the hub (`kv_truth.py --checkpoint`, no weights
  downloaded): 102,889,705,216 (equal to the config arithmetic), 49,122,681,728, 1,522,797,824
  and 3,852,562,944; the same reader gave Llama-3.1-Nemotron-51B's 51,501,015,040.
  `test_a_hand_written_parameter_count_is_arithmetic_or_tensors_not_a_model_card` rejects a card.
  Every branch figure is at or above the new truths (Ling's GenZ decode graph: 195,018.0 MiB
  against 195,017.97).
* **Figures lost or new on the branch were never held.** The gate now also fails on a figure the
  branch reports and origin/main did not (held to its ground truth like a drop: 357 figures, from
  readers that raised on origin/main) and on a figure origin/main reported and the branch does not
  (its entry names the error the branch records, and the gate checks it is that error: 6 figures,
  Llama-4 Scout's runtime prefill and chunked graphs, whose weights -- 205,553.75 MiB, the
  model's own 107.8B parameters at 2 B -- do not fit the 200 GB device, which origin/main's
  152M-parameter reading did). The truths gained what those figures need: a sequence's recurrent
  state from transformers' own mixer modules (the depthwise convolution's ring buffer and the
  Gated DeltaNet / Mamba-2 / Mamba-1 matrix, at 2 B an element: a lower bound under the readers'
  fp32 matrix) and the cross-attention K / V each encoder token adds, times a fixed-length
  encoder's token count written with its arithmetic (`truths.json` `encoder_tokens`: Nougat's
  588 and Donut's 4,800 Swin tokens, Pix2Struct's `seq_len`, Mllama's 4 x 1,025, a ViT's 577,
  Whisper's 1,500; 0 where the encoder reads the request's own input). Every cross-attention
  state figure the review checked by hand equals its truth exactly (Whisper-large-v3-turbo
  30,720,000; Mllama 134,348,800; Donut 78,643,200; Nougat 9,633,792). What it found: the FRD-027
  facts read a query / key norm only as `qk_norm`, so Hunyuan-A13B's (`use_qk_norm`) weights sat
  12,288 B below transformers' count once its facts stopped raising -- now read under every
  spelling (`layer_plan.QK_NORM_KEYS`), which moves Ling-flash-2.0's and MiniMax-M2's facts up by
  their QK norms too; and Nougat's decode graph sits 7.0 MiB under "matrices less the token
  embedding" because it also gathers its learned position table -- the decode truth now leaves
  out every table a step gathers one row from (as the graph models it), and the graph is at it.
* **The chunked graph's truth.** `runtime_chunked[1]` was held to the decode graph's truth (less
  the input embedding) though the chunked graph holds that embedding; it is now held to the
  prefill graph's (deepseek-moe-16b's database row: 31,234 MiB against a truth of 31,234, where it had been held
  to 30,834; the 400 MiB is its 102,400 x 2,048 embedding).
* **No GGUF in the committed corpus.** `tests/fixtures/gguf` holds metadata-only copies of the
  four real GGUFs (`scripts/gguf_metadata_fixture.py`: every metadata key and tensor name
  verbatim, one-element placeholder tensors, the tokenizer arrays replaced by their length as
  `{arch}.vocab_size`; 11-42 KB each), which resolve to configs identical to the source files'.
  The committed gate now runs the GGUF reader on Qwen3.5-2B's hybrid, gpt-oss-20b's window
  pattern, SmolVLM's Llama and nomic-embed's encoder, against origin/main's reading of the same
  files (`origin-main.json` gained their 4 rows, measured on `c17ef89` with the same harness).

**The gate now.** The committed corpus is 220 rows (216 and the 4 GGUF fixtures); the whole
corpus 266. Over the whole corpus: **992 unsafe moves, 357 new figures and 6 lost figures in 111
configs, every one allowlisted and at or above its ground truth; 1,394 entries; 0 failures.**
Against round 12's allowlist the new gate fails the committed corpus with 385 and the whole
corpus with 402. Round 13 moves only the facts' weights of the three QK-norm models (up) and adds
the four GGUF fixture rows: against the round-12 head's full run, 259 of 266 rows are unchanged.

**The suites** (section 3): `llm-memory-calculator/tests` 2,503 passed, 21 skipped, 1 failed (the
schema test that fails on origin/main too); `BudSimulator/tests` 81 passed. The suite rewrites
`BudSimulator/data/prepopulated.db`; it is restored from git after every run.

## 7. Round 14: the serving path's weights, the gate's per-token truths and its environment

The thirteenth review found a reader the gate did not measure and three weaknesses in the gate.
Each finding, and what was done (red first: `tests/test_serving_weights_and_gpt2_spellings.py`,
the new cases of `tests/test_kv_truth.py` and `tests/test_kv_gate.py`, and BudSimulator's
`test_a_database_model_loads_the_weights_its_hf_config_loads`: 50 cases, 45 of which fail on the round-13 head `db3d88d`; the 5 that pass there pin what did not change -- the decoder formula a stand-in object keeps, a windowed layer's 0 once the context passes its window, and three counts the head already held over their truth (Falcon-H1's facts, GenZ's built-in Mamba-130m and Mamba-2.8b). One existing assertion was updated to the checkpoint: `tests/test_hybrid_parameter_count.py::test_kda_gates_are_low_rank_not_dense` pinned `kda_params` without KDA's `b_proj`):

* **The serving path's weights** (must-fix). `genz.serving.memory_model.estimate_weight_bytes` --
  the weight a `ServingSimulator` run loads into its device tier, and what the cluster's and
  disaggregation's `B_eff` take off the device -- counted the GenZ `ModelConfig`'s decoder fields:
  `num_decoder_layers` x (attention + 3 x hidden x intermediate) + one embedding. Once the loader
  and the converter gave an encoder-decoder's `ModelConfig` its own decoder (round 7 on), and the
  converter a multimodal model's language model, that was the decoder alone, below origin/main
  (which read a generic 7B default there) and below the model: Whisper large-v3 1.81 GB of 3.09,
  flan-T5-xl 2.45 of 5.57, NLLB-MoE 4.27 of 109, Qwen3.5-4B (converter) 7.48 of 9.08; and the
  cluster admitted more requests than the device holds (Whisper large-v3, A100 40 GB, 4k: 44 where
  the memory holds 43). The figure is now the model's own parameter count
  (`memory_model.model_parameter_count`): a `ModelConfig` read from a config carries it
  (`source_config`, set by the HF loader and by BudSimulator's converter), and is counted by the
  FRD-027 facts -- every stack, tower, adapter and expert, the count the vLLM planner reserves.
  For the whole category the facts gained what they had missed: an audio encoder beside a
  decoder-only language model (`audio_config` / `audio_encoder_config`, read at its own geometry
  by `stack_params.tower_view`, which now reads a Whisper-shaped tower's `encoder_attention_heads`,
  with its convolutional front end and a two-matrix projector: Voxtral, MOSS-Transcribe,
  Qwen2-Audio), and each recurrent layer counted by its own mechanism's count (`mixer_params`)
  instead of one copy of a Mamba-2 formula without its convolution's bias (Mamba-Codestral had sat
  393,216 parameters below transformers', Kimi-Linear 137,675,680 below its checkpoint -- its KDA
  layers counted as Mamba-2, and KDA's own count lacked the per-head beta projection `b_proj`
  [32, 2304]; Bamba-9B-v2,
  Falcon-H1-34B and Kimi-Linear are now their truths to the parameter). A `ModelConfig`
  built by hand (GenZ's built-in dictionary) is counted from its fields with its routed and
  shared experts on the layers it places them, each Mamba layer as its Mamba mixer and its LM
  head: Mixtral-8x7B 46,702,792,704 (it was one expert a layer: 14.2 GB on a 40 GB card that took
  53 requests), Llama-2-7b 6,738,415,616 -- both transformers' counts to the parameter -- and
  Falcon-Mamba 7B, whose built-in vocabulary is now the hub config's 65,024, 7,272,927,232 against
  its checkpoint's 7,272,665,088. Whisper large-v3 now loads 3.62 GB (the facts' 1,811M parameters: 17 % over transformers' 1,543M) and its cluster admits 42 at 4k on a 40 GB A100, under the 43 its memory holds.
  The harness now measures the figure (`serving_weight_bytes` through a `ServingSimulator`,
  `conv_serving_weight_bytes` on the converter's `ModelConfig`) and both concurrencies
  (`cluster_b_eff_4096`, `disagg_b_eff_4096`, uncapped by the scheduler's batch), for every row;
  origin/main's figures for them were measured with the same harness on `c17ef89` and added to
  `origin-main.json`. The gate holds the weights to every parameter of the model at 2 B and a
  concurrency to the weights, the KV a token and the fixed bytes a request it is sized from. Where
  a checkpoint holds modules no config key describes, the entry names them (`unstated`, with
  their source) and the figure may sit exactly that far below: Voxtral-Mini-4B-Realtime's
  AdaRmsNorms (5,111,808 parameters) and higgs-audio-v3's codec, second token table and codebook
  heads (632,382,441) -- see LAYER_KINDS.md, "What errs low, measured".
* **Per-token truths of 0 for a stack whose every layer is windowed** (should-fix). A per-token
  figure with no stated context is now held to the K + V a token adds on every layer that caches
  it (full, and windowed while the token is inside the window), and the KV planner's rate at a
  stated length to what that length's last token adds. Voxtral-Realtime's `conv_kv_size` is now
  held to 26 x 2,048 = 53,248 (its figure), and the LFM2.5-Audio detokenizers' per-token figures
  to 3 x 512 elements (3,072 B), which they equal. With the new rule, 77 memory entries keep a truth of 0, every one for a model that holds none of that kind (Mamba, RWKV, xLSTM and nomic-bert KV; a text encoder-decoder's fixed bytes, whose cross cache is carried by the token).
* **A reader's missing module accepted as "not measured" whatever it was** (should-fix). Only the
  modules a reader is declared to need may be missing (`kv_corpus.READER_DEPENDENCIES`: the GGUF
  reader's `gguf`; the converter's `requests`, `lxml`, `json_repair`); any other missing module --
  transformers or torch above all -- fails. The gate tests (`test_kv_gate.py`, `test_config_class_defaults.py`, `test_kv_truth.py`, `test_gate_found_readers.py` and the new file) pass in four environments, 0 failures each: transformers 5.12 (every figure measured); transformers 4.50.3 without torch, with the readers' declared modules (80 passed, 9 skipped -- the truth builds, which need torch -- every figure measured); neither transformers nor torch, with `gguf`, `requests`, `lxml` and `json_repair` (76 passed, 13 skipped, every figure measured); and a bare venv without those either (75 passed, 14 skipped; the GGUF reader's 264 figures and the converter's 501 reported as not measured, by `gguf` and `requests`). In round 13 the first three were not run with the readers' modules installed, so the GGUF and converter figures went unmeasured outside 5.12; the README's "with or without transformers" now says which modules each reader needs.
* **A nested GPT-2 decoder** (should-fix). The decoder-only names a config class aliases
  (`layer_plan.DECODER_NAME_ALIASES`: GPT2Config's `attribute_map`, which a test checks against
  GPT-2, GPT-J, CodeGen, GPT-BigCode, CTRL, OpenAI-GPT and BLOOM) are read by `decoder_view`, for a
  nested decoder block and a flat config alike, and `depth` reads `n_layer`. The ViT-GPT2 shape now
  holds 6 x 1,536 K + V a token and a cross cache over the ViT's 197 tokens in every reader (it held
  0, and GenZ, the serving path and the prefix cache raised); a flat GPT-2 config is graphed at its
  own 12 x 768 (GenZ graphed the generic 32 x 4,096) and the KV planner charges its 36,864 B a token
  (it charged 0).

**The gate now.** Over the whole corpus (264 rows; the committed corpus is 219 of them): **1,207 unsafe moves, 394 new figures and 6 lost figures in 112 configs, every one allowlisted and at or above its ground truth -- two of them by the named unstated modules above; 1,707 entries; 0 failures.** Against round 13's allowlist the new corpus fails with 288. The serving path's new figures account for most of the growth: 54 configs whose cluster and disaggregation `B_eff` rise (their KV, fixed bytes or weights now read as the model holds them), 44 converter and 17 loader weight figures below origin/main's (origin/main read a generic 7B or the decoder fields), and 19 weight figures origin/main could not produce (its loader raised). origin/main's figures for the new readers were measured with the same harness on `c17ef89` (`origin-main.json`), the truths by `scripts/kv_truth.py --allowlist-from` with `truths.json`'s new entries: the whole checkpoints of MOSS-Transcribe, higgs-audio-v3 and VibeVoice for their serving weights (the language model alone had been their truth), GenZ's built-in Mamba models' checkpoints, the LFM2.5-Audio detokenizers' own checkpoint, Whisper's 1,500-frame encoder and the text encoders' 0, Mllama's one tile of 1,025 tokens, and the config arithmetic of four recurrent states (Zamba2, Phi-4-mini-flash, Qwen3.5-2B's GGUF, the detokenizers). `scripts/kv_truth.py --checkpoint` now reads one copy of a repo's weights (its index's shards, else `model.safetensors`): it had summed Voxtral-Realtime's `consolidated.safetensors` with its transformers shards, twice the model; the five checkpoint truths of round 13 are unchanged under it.

**The suites** (section 3): `llm-memory-calculator/tests` 2,548 passed, 21 skipped, 1 failed (the
schema test that fails on origin/main too); `BudSimulator/tests` 86 passed. The suite rewrites
`BudSimulator/data/prepopulated.db`; it is restored from git after every run.

The full per-config table (before = origin/main, after = this branch) follows. 266 configs measured; 3 unchanged; 263 changed.

## config:chaos-pressure/gpt-oss-20b/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 195 | 5 |
| conv_serving_weight_bytes | 3626680320 | 41829514368 |
| disagg_b_eff_4096 | 195 | 5 |
| genz_chunked_kv_mb | 216.0 | 123.0 |
| runtime_chunked | [9.364153, 13110.469, 36.0] | [9.358584, 13110.469, 27.0] |
| serving_weight_bytes | 3626680320 | 41829514368 |
| slice_mib_16x4352 | [20736, true] | [20992, true] |
| vllm_limits | [155360, 2485760, 155360, 1, 0, 0.0, true] | [155360, 2485760, 155360, 1, 1, 0.0, true] |
| vllm_running_8192 | 294 | 289 |

## config:models--Cloudflare--clef-flash/snapshots/<rev>/config.json (multimodal)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 53 | 129 |
| conv_kv_size | 262144 | 16384 |
| conv_prefill_weights_mb | 12852.0 | 17077.6875 |
| conv_serving_weight_bytes | 13214154752 | 18819627488 |
| conv_state_bytes | 0 | 51511296 |
| disagg_b_eff_4096 | 53 | 129 |
| disagg_transfer_bytes_640 | 83886080 | 72482816 |
| eng_None_32768_params | 10077427968 | 10211645696 |
| eng_None_32768_state_mb | 0.0 | 51.511296 |
| eng_None_32768_total_mb | 21954.286067 | 22274.232819 |
| eng_None_4096_params | 10077427968 | 10211645696 |
| eng_None_4096_state_mb | 0.0 | 51.511296 |
| eng_None_4096_total_mb | 20652.374106 | 20972.320858 |
| eng_llamacpp_32768_params | 10077427968 | 10211645696 |
| eng_llamacpp_32768_state_mb | 0.0 | 51.511296 |
| eng_llamacpp_32768_total_mb | 21954.286067 | 22274.232819 |
| eng_llamacpp_4096_params | 10077427968 | 10211645696 |
| eng_llamacpp_4096_state_mb | 0.0 | 51.511296 |
| eng_llamacpp_4096_total_mb | 20652.374106 | 20972.320858 |
| genz_chunked_kv_mb | 576.0 | 242.25 |
| genz_decode_kv_mb | 512.0 | 177.125 |
| genz_decode_weights_mb | 13716.0 | 15137.6875 |
| genz_kv_size | 65536 | 16384 |
| genz_layer_kinds | {"MHA-global": 32} | {"MHA-global": 8, "Recurrent": 24} |
| genz_prefill_kv_mb | 512.0 | 177.125 |
| genz_prefill_weights_mb | 15656.0 | 17077.6875 |
| prefix_kv_bytes_per_token | 131072 | 32768 |
| report_params | 10077427968 | 10211645696 |
| report_total_gb | 20.813691295 | 21.082126751 |
| report_weight_gb | 20.154855936 | 20.423291392 |
| runtime_chunked | [14.202194, 15656.0, 96.0] | [15.638544, 17077.688, 122.25] |
| runtime_decode | [9.92986, 13716.0, 64.5] | [10.742435, 15137.688, 212.625] |
| runtime_prefill | [30.217538, 15656.0, 64.0] | [32.915873, 17077.688, 65.125] |
| serving_bytes_per_token | 131072 | 32768 |
| serving_requests_8gib_1k | 64 | 100 |
| serving_state_bytes | 0 | 51511296 |
| serving_weight_bytes | 14382268416 | 18819627488 |

## config:models--FunAudioLLM--Fun-CosyVoice3-0.5B-2512/snapshots/<rev>/CosyVoice-BlankEN/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 987922432 | 988010240 |
| serving_weight_bytes | 987922432 | 988010240 |

## config:models--HuggingFaceTB--SmolLM2-135M/snapshots/<rev>/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 268959744 | 269030016 |
| serving_weight_bytes | 268959744 | 269030016 |

## config:models--IndexTeam--IndexTTS-2/snapshots/<rev>/qwen0.6bemo4-merge/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 1191968768 | 1192099840 |
| serving_weight_bytes | 1191968768 | 1192099840 |

## config:models--LiquidAI--LFM2.5-Audio-1.5B-JP/snapshots/<rev>/audio_detokenizer/config.json (hybrid)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 1275 | 3390 |
| conv_kv_size | 4096 | 1536 |
| conv_prefill_weights_mb | 218.0 | 220.514648 |
| conv_serving_weight_bytes | 161480704 | 231243776 |
| conv_state_bytes | 0 | 15360 |
| disagg_b_eff_4096 | 1275 | 3390 |
| disagg_transfer_bytes_640 | 5242880 | 1981440 |
| eng_None_32768_state_mb | 0.0 | 0.01536 |
| eng_None_32768_total_mb | 325.305754 | 325.321114 |
| eng_None_4096_state_mb | 0.0 | 0.01536 |
| eng_None_4096_total_mb | 244.775117 | 244.790477 |
| eng_llamacpp_32768_state_mb | 0.0 | 0.01536 |
| eng_llamacpp_32768_total_mb | 325.305754 | 325.321114 |
| eng_llamacpp_4096_state_mb | 0.0 | 0.01536 |
| eng_llamacpp_4096_total_mb | 244.775117 | 244.790477 |
| facts_weight_bytes | 237925856 | 231243776 |
| genz_chunked_kv_mb | 36.0 | 1.617188 |
| genz_decode_kv_mb | 20.087891 | 0.102539 |
| genz_decode_weights_mb | 128.0 | 156.514648 |
| genz_kv_size | 4096 | 1536 |
| genz_layer_kinds | {"MHA-global": 5, "MHA-local": 3} | {"MHA-local": 3, "Recurrent": 5} |
| genz_prefill_kv_mb | 20.087891 | 0.102539 |
| genz_prefill_weights_mb | 192.0 | 220.514648 |
| prefix_kv_bytes_per_token | 8192 | 3072 |
| runtime_chunked | [0.186761, 192.0, 6.0] | [0.210063, 220.515, 0.867] |
| runtime_decode | [1.210186, 128.0, 2.619] | [1.185996, 156.515, 0.158] |
| runtime_prefill | [1.605842, 192.0, 2.588] | [1.601351, 220.515, 0.103] |
| serving_bytes_per_token | 8192 | 3072 |
| serving_requests_8gib_1k | 1024 | 2717 |
| serving_state_bytes | 0 | 15360 |
| serving_weight_bytes | 161480704 | 231243776 |
| vllm_limits | [1519413, 24310608, 1519413, 1, 0, 98304.0, true] | [1519549, 24312784, 1519549, 1, 0, 98304.0, true] |

## config:models--LiquidAI--LFM2.5-Audio-1.5B/snapshots/<rev>/audio_detokenizer/config.json (hybrid)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 1275 | 3390 |
| conv_kv_size | 4096 | 1536 |
| conv_prefill_weights_mb | 218.0 | 220.514648 |
| conv_serving_weight_bytes | 161480704 | 231243776 |
| conv_state_bytes | 0 | 15360 |
| disagg_b_eff_4096 | 1275 | 3390 |
| disagg_transfer_bytes_640 | 5242880 | 1981440 |
| eng_None_32768_state_mb | 0.0 | 0.01536 |
| eng_None_32768_total_mb | 325.305754 | 325.321114 |
| eng_None_4096_state_mb | 0.0 | 0.01536 |
| eng_None_4096_total_mb | 244.775117 | 244.790477 |
| eng_llamacpp_32768_state_mb | 0.0 | 0.01536 |
| eng_llamacpp_32768_total_mb | 325.305754 | 325.321114 |
| eng_llamacpp_4096_state_mb | 0.0 | 0.01536 |
| eng_llamacpp_4096_total_mb | 244.775117 | 244.790477 |
| facts_weight_bytes | 237925856 | 231243776 |
| genz_chunked_kv_mb | 36.0 | 1.617188 |
| genz_decode_kv_mb | 20.087891 | 0.102539 |
| genz_decode_weights_mb | 128.0 | 156.514648 |
| genz_kv_size | 4096 | 1536 |
| genz_layer_kinds | {"MHA-global": 5, "MHA-local": 3} | {"MHA-local": 3, "Recurrent": 5} |
| genz_prefill_kv_mb | 20.087891 | 0.102539 |
| genz_prefill_weights_mb | 192.0 | 220.514648 |
| prefix_kv_bytes_per_token | 8192 | 3072 |
| runtime_chunked | [0.186761, 192.0, 6.0] | [0.210063, 220.515, 0.867] |
| runtime_decode | [1.210186, 128.0, 2.619] | [1.185996, 156.515, 0.158] |
| runtime_prefill | [1.605842, 192.0, 2.588] | [1.601351, 220.515, 0.103] |
| serving_bytes_per_token | 8192 | 3072 |
| serving_requests_8gib_1k | 1024 | 2717 |
| serving_state_bytes | 0 | 15360 |
| serving_weight_bytes | 161480704 | 231243776 |
| vllm_limits | [1519413, 24310608, 1519413, 1, 0, 98304.0, true] | [1519549, 24312784, 1519549, 1, 0, 98304.0, true] |

## config:models--OpenMOSS-Team--MOSS-Transcribe-Diarize/snapshots/<rev>/config.json (audio-llm)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 88 | 87 |
| conv_kv_size | 262144 | 57344 |
| conv_layers | [0, 32] | [0, 28] |
| conv_prefill_weights_mb | 12852.0 | 1433.5 |
| conv_serving_weight_bytes | 13214154752 | 1817071616 |
| disagg_b_eff_4096 | 88 | 87 |
| facts_weight_bytes | 1192099840 | 1817071616 |
| serving_weight_bytes | 1191968768 | 1817071616 |
| slice_mib_16x4352 | [14592, true] | [15104, true] |
| vllm_limits | [40225, 643600, 40225, 1, 0, 0.0, true] | [39884, 638144, 39884, 1, 0, 0.0, true] |
| vllm_running_1024 | 502 | 498 |
| vllm_running_8192 | 76 | 75 |

## config:models--Qwen--Qwen2-0.5B/snapshots/<rev>/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 987922432 | 988010240 |
| serving_weight_bytes | 987922432 | 988010240 |

## config:models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/<rev>/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 987922432 | 988010240 |
| serving_weight_bytes | 987922432 | 988010240 |

## config:models--Qwen--Qwen2.5-0.5B/snapshots/<rev>/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 987922432 | 988010240 |
| serving_weight_bytes | 987922432 | 988010240 |

## config:models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/<rev>/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 3087138816 | 3087313920 |
| serving_weight_bytes | 3087138816 | 3087313920 |

## config:models--Qwen--Qwen2.5-3B/snapshots/<rev>/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 6171394048 | 6171693056 |
| serving_weight_bytes | 6171394048 | 6171693056 |

## config:models--Qwen--Qwen2.5-7B-Instruct/snapshots/<rev>/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 122 | 118 |
| conv_serving_weight_bytes | 14140571648 | 15230974976 |
| disagg_b_eff_4096 | 122 | 118 |
| serving_weight_bytes | 14140571648 | 15230974976 |

## config:models--Qwen--Qwen3-Embedding-0.6B/snapshots/<rev>/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 1191421952 | 1191553024 |
| serving_weight_bytes | 1191421952 | 1191553024 |

## config:models--Qwen--Qwen3.5-4B-Base/snapshots/<rev>/config.json (multimodal)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 66 | 182 |
| conv_kv_size | 262144 | 16384 |
| conv_prefill_weights_mb | 12852.0 | 9234.023438 |
| conv_serving_weight_bytes | 13214154752 | 9078531072 |
| conv_state_bytes | 0 | 51511296 |
| disagg_b_eff_4096 | 66 | 182 |
| disagg_transfer_bytes_640 | 83886080 | 72482816 |
| eng_None_32768_params | 4945124864 | 5029010944 |
| eng_None_32768_state_mb | 0.0 | 51.511296 |
| eng_None_32768_total_mb | 11478.217318 | 11697.500774 |
| eng_None_4096_params | 4945124864 | 5029010944 |
| eng_None_4096_state_mb | 0.0 | 51.511296 |
| eng_None_4096_total_mb | 10282.001818 | 10501.285274 |
| eng_llamacpp_32768_params | 4945124864 | 5029010944 |
| eng_llamacpp_32768_state_mb | 0.0 | 51.511296 |
| eng_llamacpp_32768_total_mb | 11478.217318 | 11697.500774 |
| eng_llamacpp_4096_params | 4945124864 | 5029010944 |
| eng_llamacpp_4096_state_mb | 0.0 | 51.511296 |
| eng_llamacpp_4096_total_mb | 10282.001818 | 10501.285274 |
| genz_chunked_kv_mb | 576.0 | 242.25 |
| genz_decode_kv_mb | 512.0 | 177.125 |
| genz_decode_weights_mb | 7132.5 | 8021.523438 |
| genz_kv_size | 65536 | 16384 |
| genz_layer_kinds | {"MHA-global": 32} | {"MHA-global": 8, "Recurrent": 24} |
| genz_prefill_kv_mb | 512.0 | 177.125 |
| genz_prefill_weights_mb | 8345.0 | 9234.023438 |
| prefix_kv_bytes_per_token | 131072 | 32768 |
| report_params | 4945124864 | 5029010944 |
| report_total_gb | 10.422165791 | 10.589937951 |
| report_weight_gb | 9.890249728 | 10.058021888 |
| runtime_chunked | [7.637411, 8345.0, 96.0] | [8.59528, 9234.023, 122.25] |
| runtime_decode | [5.911349, 7132.5, 64.5] | [6.398831, 8021.523, 212.625] |
| runtime_prefill | [17.133994, 8345.0, 64.0] | [18.879089, 9234.023, 65.125] |
| serving_bytes_per_token | 131072 | 32768 |
| serving_requests_8gib_1k | 64 | 100 |
| serving_state_bytes | 0 | 51511296 |
| serving_weight_bytes | 7478968320 | 9078531072 |

## config:models--Qwen--Qwen3.5-4B/snapshots/<rev>/config.json (multimodal)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 66 | 182 |
| conv_kv_size | 262144 | 16384 |
| conv_prefill_weights_mb | 12852.0 | 9234.023438 |
| conv_serving_weight_bytes | 13214154752 | 9078531072 |
| conv_state_bytes | 0 | 51511296 |
| disagg_b_eff_4096 | 66 | 182 |
| disagg_transfer_bytes_640 | 83886080 | 72482816 |
| eng_None_32768_params | 4945124864 | 5029010944 |
| eng_None_32768_state_mb | 0.0 | 51.511296 |
| eng_None_32768_total_mb | 11478.217318 | 11697.500774 |
| eng_None_4096_params | 4945124864 | 5029010944 |
| eng_None_4096_state_mb | 0.0 | 51.511296 |
| eng_None_4096_total_mb | 10282.001818 | 10501.285274 |
| eng_llamacpp_32768_params | 4945124864 | 5029010944 |
| eng_llamacpp_32768_state_mb | 0.0 | 51.511296 |
| eng_llamacpp_32768_total_mb | 11478.217318 | 11697.500774 |
| eng_llamacpp_4096_params | 4945124864 | 5029010944 |
| eng_llamacpp_4096_state_mb | 0.0 | 51.511296 |
| eng_llamacpp_4096_total_mb | 10282.001818 | 10501.285274 |
| genz_chunked_kv_mb | 576.0 | 242.25 |
| genz_decode_kv_mb | 512.0 | 177.125 |
| genz_decode_weights_mb | 7132.5 | 8021.523438 |
| genz_kv_size | 65536 | 16384 |
| genz_layer_kinds | {"MHA-global": 32} | {"MHA-global": 8, "Recurrent": 24} |
| genz_prefill_kv_mb | 512.0 | 177.125 |
| genz_prefill_weights_mb | 8345.0 | 9234.023438 |
| prefix_kv_bytes_per_token | 131072 | 32768 |
| report_params | 4945124864 | 5029010944 |
| report_total_gb | 10.422165791 | 10.589937951 |
| report_weight_gb | 9.890249728 | 10.058021888 |
| runtime_chunked | [7.637411, 8345.0, 96.0] | [8.59528, 9234.023, 122.25] |
| runtime_decode | [5.911349, 7132.5, 64.5] | [6.398831, 8021.523, 212.625] |
| runtime_prefill | [17.133994, 8345.0, 64.0] | [18.879089, 9234.023, 65.125] |
| serving_bytes_per_token | 131072 | 32768 |
| serving_requests_8gib_1k | 64 | 100 |
| serving_state_bytes | 0 | 51511296 |
| serving_weight_bytes | 7478968320 | 9078531072 |

## config:models--SupersonicLabs--Julia-1/snapshots/<rev>/encoder/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 280952832 | 280987392 |
| kv_full_layers | 8 | 22 |
| kv_sliding_layers | 14 | 0 |
| kvgeo_32768 | [33792.0, 8, 0, 0.0, 16, 0.0] | [33792.0, 22, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [33792.0, 8, 0, 0.0, 16, 0.0] | [33792.0, 22, 0, 0.0, 16, 0.0] |
| serving_weight_bytes | 280952832 | 280987392 |

## config:models--answerdotai--ModernBERT-base/snapshots/<rev>/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| conv_prefill_weights_mb | 357.9375 | 369.9375 |
| conv_serving_weight_bytes | 297959424 | 298028544 |
| genz_prefill_weights_mb | 320.8125 | 332.8125 |
| runtime_chunked | [0.336617, 320.812, 49.5] | [0.347354, 332.812, 49.5] |
| runtime_prefill | [2.302303, 320.812, 33.0] | [2.323778, 332.812, 33.0] |
| serving_weight_bytes | 297959424 | 298028544 |

## config:models--bosonai--higgs-audio-v2-generation-3B-base/snapshots/<rev>/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 77 | 76 |
| conv_prefill_weights_mb | 6879.0 | 17400.0 |
| conv_serving_weight_bytes | 6425149440 | 7213504512 |
| disagg_b_eff_4096 | 77 | 76 |
| genz_decode_weights_mb | 6127.5 | 11388.0 |
| genz_prefill_weights_mb | 6879.0 | 17400.0 |
| runtime_chunked | [6.290686, 6879.0, 84.0] | [15.723104, 17400.0, 84.0] |
| runtime_decode | [5.227983, 6127.5, 56.438] | [8.442918, 11388.0, 56.438] |
| runtime_prefill | [14.342742, 6879.0, 56.0] | [33.170805, 17400.0, 56.0] |
| serving_weight_bytes | 6425149440 | 7213504512 |

## config:models--bosonai--higgs-audio-v3-tts-4b/snapshots/<rev>/config.json (unknown)

| figure | before | after |
|---|---|---|
| conv_kv_size | 262144 | 73728 |
| conv_layers | [0, 32] | [0, 36] |
| conv_prefill_weights_mb | 12852.0 | 8413.75 |
| conv_serving_weight_bytes | 13214154752 | 8044936192 |
| eng_None_32768_params | 151881984 | 4106531840 |
| eng_None_32768_total_mb | 5669.117645 | 13578.417357 |
| eng_None_4096_params | 151881984 | 4106531840 |
| eng_None_4096_total_mb | 1174.501478 | 9083.80119 |
| eng_llamacpp_32768_params | 151881984 | 4106531840 |
| eng_llamacpp_32768_total_mb | 5669.117645 | 13578.417357 |
| eng_llamacpp_4096_params | 151881984 | 4106531840 |
| eng_llamacpp_4096_total_mb | 1174.501478 | 9083.80119 |
| report_params | 151881984 | 4106531840 |
| report_total_gb | 1.34864898 | 9.257948692 |
| report_weight_gb | 0.303763968 | 8.21306368 |
| serving_weight_bytes | 8044544000 | 8044936192 |

## config:models--convaiinnovations--laya-multilingual/snapshots/<rev>/encoder/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 613810176 | 613879296 |
| kv_full_layers | 8 | 22 |
| kv_sliding_layers | 14 | 0 |
| kvgeo_32768 | [67584.0, 8, 0, 0.0, 16, 0.0] | [67584.0, 22, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [67584.0, 8, 0, 0.0, 16, 0.0] | [67584.0, 22, 0, 0.0, 16, 0.0] |
| serving_weight_bytes | 613810176 | 613879296 |

## config:models--convaiinnovations--laya-typed-decisions/snapshots/<rev>/encoder/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 789446656 | 789563392 |
| kv_full_layers | 10 | 28 |
| kv_sliding_layers | 18 | 0 |
| kvgeo_32768 | [114688.0, 10, 0, 0.0, 16, 0.0] | [114688.0, 28, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [114688.0, 10, 0, 0.0, 16, 0.0] | [114688.0, 28, 0, 0.0, 16, 0.0] |
| serving_weight_bytes | 789446656 | 789563392 |

## config:models--convaiinnovations--laya/snapshots/<rev>/encoder/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 789446656 | 789563392 |
| kv_full_layers | 10 | 28 |
| kv_sliding_layers | 18 | 0 |
| kvgeo_32768 | [114688.0, 10, 0, 0.0, 16, 0.0] | [114688.0, 28, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [114688.0, 10, 0, 0.0, 16, 0.0] | [114688.0, 28, 0, 0.0, 16, 0.0] |
| serving_weight_bytes | 789446656 | 789563392 |

## config:models--convaiinnovations--laya/snapshots/<rev>/typed-decisions/encoder/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 789446656 | 789563392 |
| kv_full_layers | 10 | 28 |
| kv_sliding_layers | 18 | 0 |
| kvgeo_32768 | [114688.0, 10, 0, 0.0, 16, 0.0] | [114688.0, 28, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [114688.0, 10, 0, 0.0, 16, 0.0] | [114688.0, 28, 0, 0.0, 16, 0.0] |
| serving_weight_bytes | 789446656 | 789563392 |

## config:models--facebook--wav2vec2-base/snapshots/<rev>/config.json (encoder-decoder)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 226541568 | 226629120 |
| serving_weight_bytes | 226541568 | 226629120 |

## config:models--fastino--GLiNER2.5-Decide/snapshots/<rev>/encoder_config/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 104 | 103 |
| conv_prefill_weights_mb | 1268.042969 | 1269.042969 |
| conv_serving_weight_bytes | 1067472896 | 1329739776 |
| disagg_b_eff_4096 | 104 | 103 |
| genz_prefill_weights_mb | 1076.042969 | 1077.042969 |
| runtime_chunked | [1.030875, 1076.043, 72.0] | [1.03177, 1077.043, 72.0] |
| runtime_prefill | [3.740289, 1076.043, 48.0] | [3.742079, 1077.043, 48.0] |
| serving_weight_bytes | 1067472896 | 1329739776 |

## config:models--internlm--Intern-Decision-4B/snapshots/<rev>/config.json (multimodal)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 66 | 182 |
| conv_kv_size | 262144 | 16384 |
| conv_prefill_weights_mb | 12852.0 | 9234.023438 |
| conv_serving_weight_bytes | 13214154752 | 9078531072 |
| conv_state_bytes | 0 | 51511296 |
| disagg_b_eff_4096 | 66 | 182 |
| disagg_transfer_bytes_640 | 83886080 | 72482816 |
| eng_None_32768_params | 4945124864 | 5029010944 |
| eng_None_32768_state_mb | 0.0 | 51.511296 |
| eng_None_32768_total_mb | 11478.217318 | 11697.500774 |
| eng_None_4096_params | 4945124864 | 5029010944 |
| eng_None_4096_state_mb | 0.0 | 51.511296 |
| eng_None_4096_total_mb | 10282.001818 | 10501.285274 |
| eng_llamacpp_32768_params | 4945124864 | 5029010944 |
| eng_llamacpp_32768_state_mb | 0.0 | 51.511296 |
| eng_llamacpp_32768_total_mb | 11478.217318 | 11697.500774 |
| eng_llamacpp_4096_params | 4945124864 | 5029010944 |
| eng_llamacpp_4096_state_mb | 0.0 | 51.511296 |
| eng_llamacpp_4096_total_mb | 10282.001818 | 10501.285274 |
| genz_chunked_kv_mb | 576.0 | 242.25 |
| genz_decode_kv_mb | 512.0 | 177.125 |
| genz_decode_weights_mb | 7132.5 | 8021.523438 |
| genz_kv_size | 65536 | 16384 |
| genz_layer_kinds | {"MHA-global": 32} | {"MHA-global": 8, "Recurrent": 24} |
| genz_prefill_kv_mb | 512.0 | 177.125 |
| genz_prefill_weights_mb | 8345.0 | 9234.023438 |
| prefix_kv_bytes_per_token | 131072 | 32768 |
| report_params | 4945124864 | 5029010944 |
| report_total_gb | 10.422165791 | 10.589937951 |
| report_weight_gb | 9.890249728 | 10.058021888 |
| runtime_chunked | [7.637411, 8345.0, 96.0] | [8.59528, 9234.023, 122.25] |
| runtime_decode | [5.911349, 7132.5, 64.5] | [6.398831, 8021.523, 212.625] |
| runtime_prefill | [17.133994, 8345.0, 64.0] | [18.879089, 9234.023, 65.125] |
| serving_bytes_per_token | 131072 | 32768 |
| serving_requests_8gib_1k | 64 | 100 |
| serving_state_bytes | 0 | 51511296 |
| serving_weight_bytes | 7478968320 | 9078531072 |

## config:models--llm-jp--llm-jp-3-150m/snapshots/<rev>/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 424 | 423 |
| conv_serving_weight_bytes | 202637312 | 304636928 |
| disagg_b_eff_4096 | 424 | 423 |
| serving_weight_bytes | 202637312 | 304636928 |

## config:models--microsoft--VibeVoice-1.5B/snapshots/<rev>/config.json (unknown)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 339 | 214 |
| conv_kv_size | 262144 | 14336 |
| conv_layers | [0, 32] | [0, 28] |
| conv_prefill_weights_mb | 12852.0 | 3389.25 |
| conv_serving_weight_bytes | 13214154752 | 17704689664 |
| disagg_b_eff_4096 | 339 | 214 |
| serving_weight_bytes | 3087138816 | 17704689664 |

## config:models--mistralai--Voxtral-Mini-4B-Realtime-2602/snapshots/<rev>/config.json (audio-llm)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 82 | 78 |
| conv_kv_size | 196608 | 53248 |
| conv_layers | [0, 32] | [0, 26] |
| conv_prefill_weights_mb | 8871.0 | 7308.0 |
| conv_serving_weight_bytes | 9105309696 | 8849772544 |
| disagg_b_eff_4096 | 82 | 78 |
| facts_full_attention | true | false |
| facts_layers | [26, 0, 0, null] | [0, 26, 0, 8192] |
| facts_weight_bytes | 6858012672 | 8849772544 |
| serving_weight_bytes | 6857687040 | 8849772544 |
| slice_mib_16x4352 | [21248, true] | [23296, true] |
| vllm_limits | [39525, 632400, 39525, 1, 0, 0.0, true] | [38356, 613696, 38356, 1, 0, 0.0, true] |
| vllm_running_1024 | 494 | 479 |
| vllm_running_8192 | 74 | 72 |

## config:models--nari-labs--Dia-1.6B-0626/snapshots/<rev>/config.json (encoder-decoder)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 269 | 131 |
| conv_kv_size | 131072 | 18432 |
| conv_layers | [16, 16] | [12, 18] |
| conv_prefill_weights_mb | 6676.0 | 3072.78125 |
| conv_serving_weight_bytes | 6738149376 | 3305693184 |
| conv_state_bytes | 0 | 150994944 |
| disagg_b_eff_4096 | 269 | 131 |
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
| facts_kv_bytes_4096 | 2147483648 | 301989888 |
| facts_kv_bytes_per_token | 524288 | 36864 |
| facts_layers | [32, 0, 0, null] | [18, 0, 0, null] |
| facts_weight_bytes | 34884558848 | 3305693184 |
| genz_chunked_kv_mb | 162.0 | 450.0 |
| genz_decode_kv_mb | 144.0 | 288.0 |
| genz_decode_weights_mb | 2092.015625 | 3036.640625 |
| genz_prefill_kv_mb | 144.0 | 384.0 |
| genz_prefill_weights_mb | 2096.03125 | 3072.78125 |
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
| runtime_chunked | [1.926192, 2096.031, 27.0] | [2.459324, 3072.781, 315.0] |
| runtime_decode | [2.57908, 2092.016, 18.141] | [3.405099, 3036.641, 594.141] |
| runtime_prefill | [5.375457, 2096.031, 18.0] | [9.655511, 3072.781, 258.0] |
| serving_requests_8gib_1k | 227 | 45 |
| serving_state_bytes | 0 | 150994944 |
| serving_weight_bytes | 2193637376 | 3305693184 |
| sim_max_running | 72 | 376 |
| slice_mib_16x4352 | [81582, false] | [14848, true] |
| vllm_limits | [4702, 75232, 4702, 1, 0, 0.0, true] | [121194, 1939104, 121194, 1, 0, 150994944.0, true] |
| vllm_running_1024 | 58 | 360 |
| vllm_running_8192 | 8 | 154 |

## config:models--onnx-community--cohere-transcribe-03-2026-ONNX/snapshots/<rev>/config.json (encoder-decoder)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 317 | 126 |
| conv_kv_size | 8192 | 16384 |
| conv_layers | [4, 4] | [48, 8] |
| conv_prefill_weights_mb | 192.0 | 2804.317383 |
| conv_serving_weight_bytes | 167772160 | 5283183616 |
| conv_state_bytes | 0 | 163840000 |
| disagg_b_eff_4096 | 317 | 126 |
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
| facts_kv_bytes_4096 | 134217728 | 298057728 |
| facts_weight_bytes | 604080128 | 5283183616 |
| genz_chunked_kv_mb | 144.0 | 456.5 |
| genz_decode_kv_mb | 128.0 | 284.25 |
| genz_decode_weights_mb | 224.0 | 2770.317383 |
| genz_prefill_kv_mb | 128.0 | 1456.125 |
| genz_prefill_weights_mb | 256.0 | 2804.317383 |
| kv_cross_layers | 0 | 8 |
| kv_gb@32768 | 1.073741824 | 1.237581824 |
| kv_gb@4096 | 0.134217728 | 0.298057728 |
| kvgeo_32768 | [32768.0, 8, 0, 0.0, 16, 0.0] | [32768.0, 8, 0, 0.0, 16, 163840000.0] |
| kvgeo_4096 | [32768.0, 8, 0, 0.0, 16, 0.0] | [32768.0, 8, 0, 0.0, 16, 163840000.0] |
| report_kv_gb | 0.183369728 | 0.666697728 |
| report_params | 576643072 | 580837376 |
| report_total_gb | 1.459883786 | 2.048726794 |
| report_weight_gb | 1.153286144 | 1.161674752 |
| runtime_chunked | [0.250475, 256.0, 24.0] | [0.574352, 2804.317, 336.5] |
| runtime_decode | [1.27734, 224.0, 16.125] | [2.183912, 2770.317, 641.125] |
| runtime_prefill | [1.732001, 256.0, 16.0] | [66.95409, 2804.317, 1344.125] |
| serving_requests_8gib_1k | 256 | 43 |
| serving_state_bytes | 0 | 163840000 |
| serving_weight_bytes | 301989888 | 5283183616 |
| sim_max_running | 400 | 350 |
| slice_mib_16x4352 | [8704, true] | [16640, true] |
| vllm_limits | [141755, 2268080, 141755, 1, 0, 0.0, true] | [132830, 2125280, 132830, 1, 0, 163840000.0, true] |
| vllm_running_1024 | 1024 | 338 |
| vllm_running_8192 | 268 | 158 |

## config:models--onnx-community--moonshine-base-ONNX/snapshots/<rev>/config.json (encoder-decoder)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 786 | 392 |
| conv_kv_size | 3328 | 13312 |
| conv_layers | [4, 4] | [8, 8] |
| conv_prefill_weights_mb | 73.125 | 147.0625 |
| conv_serving_weight_bytes | 49414144 | 154308544 |
| disagg_b_eff_4096 | 786 | 392 |
| disagg_transfer_bytes_640 | 8519680 | 17039360 |
| eng_None_32768_kv_mb | 456.175616 | 892.383232 |
| eng_None_32768_total_mb | 753.298125 | 1189.505741 |
| eng_None_4096_kv_mb | 74.493952 | 129.019904 |
| eng_None_4096_total_mb | 326.63255 | 381.158502 |
| eng_llamacpp_32768_kv_mb | 456.175616 | 892.383232 |
| eng_llamacpp_32768_total_mb | 753.298125 | 1189.505741 |
| eng_llamacpp_4096_kv_mb | 74.493952 | 129.019904 |
| eng_llamacpp_4096_total_mb | 326.63255 | 381.158502 |
| facts_kv_bytes_4096 | 54525952 | 109051904 |
| facts_weight_bytes | 143144768 | 154308544 |
| genz_chunked_kv_mb | 58.5 | 117.0 |
| genz_decode_kv_mb | 52.0 | 104.0 |
| genz_decode_weights_mb | 57.6875 | 121.0625 |
| genz_kv_size | 6656 | 13312 |
| genz_prefill_kv_mb | 52.0 | 156.0 |
| genz_prefill_weights_mb | 83.6875 | 147.0625 |
| kv_cross_layers | 0 | 8 |
| kv_gb@32768 | 0.436207616 | 0.872415232 |
| kv_gb@4096 | 0.054525952 | 0.109051904 |
| kvgeo_32768 | [13312.0, 8, 0, 0.0, 16, 0.0] | [26624.0, 8, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [13312.0, 8, 0, 0.0, 16, 0.0] | [26624.0, 8, 0, 0.0, 16, 0.0] |
| prefix_kv_bytes_per_token | 13312 | 26624 |
| report_kv_gb | 0.074493952 | 0.129019904 |
| report_total_gb | 0.351430523 | 0.416861665 |
| runtime_chunked | [0.092924, 83.688, 9.75] | [0.117825, 147.062, 19.5] |
| runtime_decode | [1.169651, 57.688, 6.551] | [1.340291, 121.062, 32.551] |
| runtime_prefill | [1.412893, 83.688, 6.5] | [1.992774, 147.062, 19.5] |
| serving_bytes_per_token | 13312 | 26624 |
| serving_requests_8gib_1k | 630 | 315 |
| serving_weight_bytes | 71565312 | 154308544 |
| slice_mib_16x4352 | [6656, true] | [7936, true] |
| vllm_limits | [351373, 5621968, 351373, 1, 0, 0.0, true] | [351321, 5621136, 351321, 1, 0, 6815744.0, true] |
| vllm_running_8192 | 665 | 627 |

## config:models--onnx-community--moonshine-tiny-ONNX/snapshots/<rev>/config.json (encoder-decoder)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 1515 | 757 |
| conv_kv_size | 1728 | 6912 |
| conv_layers | [3, 3] | [6, 6] |
| conv_prefill_weights_mb | 43.59375 | 70.171875 |
| conv_serving_weight_bytes | 26836992 | 73634112 |
| disagg_b_eff_4096 | 1515 | 757 |
| disagg_transfer_bytes_640 | 4423680 | 8847360 |
| eng_None_32768_kv_mb | 236.860416 | 463.352832 |
| eng_None_32768_total_mb | 399.278822 | 625.771238 |
| eng_None_4096_kv_mb | 38.679552 | 66.991104 |
| eng_None_4096_total_mb | 169.955251 | 198.266803 |
| eng_llamacpp_32768_kv_mb | 236.860416 | 463.352832 |
| eng_llamacpp_32768_total_mb | 399.278822 | 625.771238 |
| eng_llamacpp_4096_kv_mb | 38.679552 | 66.991104 |
| eng_llamacpp_4096_total_mb | 169.955251 | 198.266803 |
| facts_kv_bytes_4096 | 28311552 | 56623104 |
| facts_weight_bytes | 69606720 | 73634112 |
| genz_chunked_kv_mb | 30.375 | 60.75 |
| genz_decode_kv_mb | 27.0 | 54.0 |
| genz_decode_weights_mb | 29.390625 | 52.171875 |
| genz_kv_size | 3456 | 6912 |
| genz_prefill_kv_mb | 27.0 | 81.0 |
| genz_prefill_weights_mb | 47.390625 | 70.171875 |
| kv_cross_layers | 0 | 6 |
| kv_gb@32768 | 0.226492416 | 0.452984832 |
| kv_gb@4096 | 0.028311552 | 0.056623104 |
| kvgeo_32768 | [6912.0, 6, 0, 0.0, 16, 0.0] | [13824.0, 6, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [6912.0, 6, 0, 0.0, 16, 0.0] | [13824.0, 6, 0, 0.0, 16, 0.0] |
| prefix_kv_bytes_per_token | 6912 | 13824 |
| report_kv_gb | 0.038679552 | 0.066991104 |
| report_total_gb | 0.184591703 | 0.218565565 |
| runtime_chunked | [0.05987, 47.391, 5.062] | [0.071507, 70.172, 10.125] |
| runtime_decode | [1.118321, 29.391, 3.401] | [1.238616, 52.172, 16.901] |
| runtime_prefill | [1.285231, 47.391, 3.375] | [1.67448, 70.172, 10.125] |
| serving_bytes_per_token | 6912 | 13824 |
| serving_requests_8gib_1k | 1213 | 606 |
| serving_weight_bytes | 34799616 | 73634112 |
| slice_mib_16x4352 | [6144, true] | [6656, true] |
| vllm_limits | [677592, 10841472, 677592, 1, 0, 0.0, true] | [677555, 10840880, 677555, 1, 0, 3538944.0, true] |

## config:models--onnx-community--whisper-base/snapshots/<rev>/config.json (encoder-decoder)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 99 | 621 |
| conv_kv_size | 49152 | 6144 |
| conv_prefill_weights_mb | 3126.390625 | 212.9375 |
| conv_serving_weight_bytes | 2853380096 | 206978048 |
| conv_state_bytes | 0 | 18432000 |
| disagg_b_eff_4096 | 99 | 621 |
| disagg_transfer_bytes_640 | 62914560 | 26296320 |
| facts_kv_bytes_4096 | 50331648 | 68763648 |
| facts_weight_bytes | 206896128 | 206978048 |
| genz_chunked_kv_mb | 432.0 | 89.15625 |
| genz_decode_kv_mb | 384.0 | 65.578125 |
| genz_decode_weights_mb | 2205.195312 | 137.850586 |
| genz_kv_size | 49152 | 6144 |
| genz_prefill_kv_mb | 384.0 | 83.15625 |
| genz_prefill_weights_mb | 2610.390625 | 188.9375 |
| kv_cross_layers | 0 | 6 |
| kv_gb@32768 | 0.0 | 0.421085184 |
| kv_gb@4096 | 0.0 | 0.068763648 |
| kvgeo_32768 | [0.0, 6, 0, 0.0, 16, 0.0] | [12288.0, 6, 0, 0.0, 16, 18432000.0] |
| kvgeo_4096 | [0.0, 6, 0, 0.0, 16, 0.0] | [12288.0, 6, 0, 0.0, 16, 18432000.0] |
| prefix_kv_bytes_per_token | 98304 | 12288 |
| runtime_chunked | [2.405326, 2610.391, 72.0] | [0.170088, 188.938, 44.156] |
| runtime_decode | [2.475791, 2205.195, 48.375] | [1.311799, 137.851, 76.359] |
| runtime_prefill | [5.933091, 2610.391, 48.0] | [2.136997, 188.938, 41.156] |
| serving_bytes_per_token | 98304 | 12288 |
| serving_requests_8gib_1k | 85 | 276 |
| serving_state_bytes | 0 | 18432000 |
| serving_weight_bytes | 2853380096 | 206978048 |
| slice_mib_16x4352 | [6656, true] | [6912, true] |
| vllm_limits | [382126, 6114016, 382126, 1, 0, 0.0, true] | [382126, 6114016, 382126, 1, 0, 18432000.0, true] |
| vllm_running_8192 | 723 | 614 |

## config:models--onnx-community--whisper-large-v3-turbo/snapshots/<rev>/config.json (encoder-decoder)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 1 | 359 |
| conv_kv_size | 32768 | 10240 |
| conv_prefill_weights_mb | 2354.40625 | 2118.325195 |
| conv_serving_weight_bytes | 2043887616 | 1787082240 |
| conv_state_bytes | 0 | 30720000 |
| decode_fit_refused | null | true |
| disagg_b_eff_4096 | 1 | 359 |
| disagg_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'apply_spec_augment', 'architectures', 'attention_dropout'," | null |
| disagg_transfer_bytes_640 | null | 43827200 |
| facts_kv_bytes_4096 | 671088640 | 114606080 |
| facts_kv_bytes_per_token | 163840 | 20480 |
| facts_layers | [32, 0, 0, null] | [4, 0, 0, null] |
| facts_weight_bytes | 3621163520 | 1787082240 |
| genz_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'apply_spec_augment', 'architectures', 'attention_dropout'," | null |
| genz_chunked_kv_mb | null | 148.59375 |
| genz_decode_kv_mb | null | 109.296875 |
| genz_decode_weights_mb | null | 1540.605469 |
| genz_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'apply_spec_augment', 'architectures', 'attention_dropout'," | null |
| genz_ffn_kinds | null | {"Dense": 4} |
| genz_kv_size | null | 10240 |
| genz_layer_kinds | null | {"MHA-global": 4} |
| genz_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'apply_spec_augment', 'architectures', 'attention_dropout'," | null |
| genz_prefill_kv_mb | null | 343.671875 |
| genz_prefill_weights_mb | null | 1668.325195 |
| kv_cross_layers | 0 | 4 |
| kv_full_layers | 32 | 4 |
| kv_gb@32768 | 0.0 | 0.70180864 |
| kv_gb@4096 | 0.0 | 0.11460608 |
| kvgeo_32768 | [0.0, 32, 0, 0.0, 16, 0.0] | [20480.0, 4, 0, 0.0, 16, 30720000.0] |
| kvgeo_4096 | [0.0, 32, 0, 0.0, 16, 0.0] | [20480.0, 4, 0, 0.0, 16, 30720000.0] |
| prefix_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'apply_spec_augment" | null |
| prefix_kv_bytes_per_token | null | 20480 |
| runtime_chunked | null | [0.44293, 1668.325, 73.594] |
| runtime_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'apply_spec_augment" | null |
| runtime_decode | null | [1.626689, 1540.605, 127.266] |
| runtime_decode_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'apply_spec_augment" | null |
| runtime_prefill | null | [10.684733, 1668.325, 273.672] |
| runtime_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'apply_spec_augment" | null |
| serving_bytes_per_token | null | 20480 |
| serving_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'apply_spec_augment', 'architectures', 'attention_dropout'," | null |
| serving_requests_8gib_1k | null | 166 |
| serving_state_bytes | null | 30720000 |
| serving_weight_bytes | null | 1787082240 |
| serving_weight_bytes_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'apply_spec_augment', 'architectures', 'attention_dropout'," | null |
| slice_mib_16x4352 | [22272, true] | [9472, true] |
| vllm_limits | [27344, 437504, 27344, 1, 0, 0.0, true] | [224350, 3589600, 224350, 1, 0, 30720000.0, true] |
| vllm_running_1024 | 341 | 1024 |
| vllm_running_8192 | 51 | 360 |

## config:models--onnx-community--whisper-tiny.en/snapshots/<rev>/config.json (encoder-decoder)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 152 | 1245 |
| conv_kv_size | 32768 | 3072 |
| conv_prefill_weights_mb | 2354.375 | 118.92041 |
| conv_serving_weight_bytes | 2043871232 | 117460224 |
| conv_state_bytes | 0 | 9216000 |
| disagg_b_eff_4096 | 152 | 1245 |
| disagg_transfer_bytes_640 | 41943040 | 13148160 |
| facts_kv_bytes_4096 | 25165824 | 34381824 |
| facts_weight_bytes | 117418752 | 117460224 |
| genz_chunked_kv_mb | 288.0 | 44.578125 |
| genz_decode_kv_mb | 256.0 | 32.789062 |
| genz_decode_weights_mb | 1605.1875 | 71.605957 |
| genz_kv_size | 32768 | 3072 |
| genz_prefill_kv_mb | 256.0 | 41.578125 |
| genz_prefill_weights_mb | 2010.375 | 109.92041 |
| kv_cross_layers | 0 | 4 |
| kv_gb@32768 | 0.0 | 0.210542592 |
| kv_gb@4096 | 0.0 | 0.034381824 |
| kvgeo_32768 | [0.0, 4, 0, 0.0, 16, 0.0] | [6144.0, 4, 0, 0.0, 16, 9216000.0] |
| kvgeo_4096 | [0.0, 4, 0, 0.0, 16, 0.0] | [6144.0, 4, 0, 0.0, 16, 9216000.0] |
| prefix_kv_bytes_per_token | 65536 | 6144 |
| runtime_chunked | [1.845717, 2010.375, 48.0] | [0.11256, 109.92, 22.078] |
| runtime_decode | [2.067047, 1605.188, 32.25] | [1.196544, 71.606, 38.18] |
| runtime_prefill | [4.775449, 2010.375, 32.0] | [1.671071, 109.92, 20.578] |
| serving_bytes_per_token | 65536 | 6144 |
| serving_requests_8gib_1k | 128 | 553 |
| serving_state_bytes | 0 | 9216000 |
| serving_weight_bytes | 2043871232 | 117460224 |
| slice_mib_16x4352 | [5888, true] | [6144, true] |
| vllm_limits | [765220, 12243520, 765220, 1, 0, 0.0, true] | [765219, 12243504, 765219, 1, 0, 9216000.0, true] |

## config:models--openai--gpt-oss-120b/snapshots/<rev>/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 126 | 1 |
| conv_serving_weight_bytes | 4860887040 | 233658313344 |
| disagg_b_eff_4096 | 126 | 1 |
| genz_chunked_kv_mb | 324.0 | 184.5 |
| runtime_chunked | [39.655232, 62149.219, 54.0] | [39.646878, 62149.219, 40.5] |
| serving_weight_bytes | 4860887040 | 233658313344 |
| sim_max_running | 250 | 220 |
| vllm_limits | [16363, 261808, 16363, 1, 0, 0.0, true] | [16363, 261808, 16363, 1, 1, 0.0, true] |
| vllm_running_1024 | 204 | 183 |

## config:models--openai--gpt-oss-20b/snapshots/<rev>/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 195 | 5 |
| conv_serving_weight_bytes | 3626680320 | 41829514368 |
| disagg_b_eff_4096 | 195 | 5 |
| genz_chunked_kv_mb | 216.0 | 123.0 |
| runtime_chunked | [9.364153, 13110.469, 36.0] | [9.358584, 13110.469, 27.0] |
| serving_weight_bytes | 3626680320 | 41829514368 |
| slice_mib_16x4352 | [20736, true] | [20992, true] |
| vllm_limits | [155360, 2485760, 155360, 1, 0, 0.0, true] | [155360, 2485760, 155360, 1, 1, 0.0, true] |
| vllm_running_8192 | 294 | 289 |

## db:Qwen/Qwen-14B (decoder-only)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 43605032960 | 45162997760 |
| serving_weight_bytes | 43605032960 | 45162997760 |

## db:Qwen/Qwen-1_8B (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 47 | 46 |
| conv_serving_weight_bytes | 4674027520 | 5296558080 |
| disagg_b_eff_4096 | 47 | 46 |
| serving_weight_bytes | 4674027520 | 5296558080 |

## db:Qwen/Qwen-72B (decoder-only)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 238714617856 | 241208672256 |
| serving_weight_bytes | 238714617856 | 241208672256 |

## db:Qwen/Qwen-7B (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 9 | 8 |
| conv_serving_weight_bytes | 22853713920 | 24098906112 |
| disagg_b_eff_4096 | 9 | 8 |
| serving_weight_bytes | 22853713920 | 24098906112 |

## db:Qwen/Qwen1.5-0.5B (decoder-only)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 927727616 | 927827968 |
| serving_weight_bytes | 927727616 | 927827968 |

## db:Qwen/Qwen1.5-1.8B (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 49 | 48 |
| conv_serving_weight_bytes | 3050831872 | 3673362432 |
| disagg_b_eff_4096 | 49 | 48 |
| serving_weight_bytes | 3050831872 | 3673362432 |

## db:Qwen/Qwen1.5-110B (decoder-only)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 219924135936 | 222418190336 |
| serving_weight_bytes | 219924135936 | 222418190336 |

## db:Qwen/Qwen1.5-14B (decoder-only)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 26775388160 | 28333352960 |
| serving_weight_bytes | 26775388160 | 28333352960 |

## db:Qwen/Qwen1.5-32B (decoder-only)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 63465062400 | 65023518720 |
| serving_weight_bytes | 63465062400 | 65023518720 |

## db:Qwen/Qwen1.5-4B (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 21 | 20 |
| conv_serving_weight_bytes | 7121797120 | 7900124160 |
| disagg_b_eff_4096 | 21 | 20 |
| serving_weight_bytes | 7121797120 | 7900124160 |

## db:Qwen/Qwen1.5-72B (decoder-only)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 142077853696 | 144571908096 |
| serving_weight_bytes | 142077853696 | 144571908096 |

## db:Qwen/Qwen1.5-7B (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 13 | 12 |
| conv_serving_weight_bytes | 14196670464 | 15441862656 |
| disagg_b_eff_4096 | 13 | 12 |
| serving_weight_bytes | 14196670464 | 15441862656 |

## db:Qwen/Qwen1.5-MoE-A2.7B (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 49 | 17 |
| conv_prefill_weights_mb | 25720.625 | 27304.625 |
| conv_serving_weight_bytes | 3088580608 | 28631178048 |
| disagg_b_eff_4096 | 49 | 17 |
| eng_None_32768_params | 13796380672 | 14626852864 |
| eng_None_32768_total_mb | 34377.467494 | 36038.411878 |
| eng_None_4096_params | 13796380672 | 14626852864 |
| eng_None_4096_total_mb | 28569.195315 | 30230.139699 |
| eng_llamacpp_32768_params | 13796380672 | 14626852864 |
| eng_llamacpp_32768_total_mb | 34377.467494 | 36038.411878 |
| eng_llamacpp_4096_params | 13796380672 | 14626852864 |
| eng_llamacpp_4096_total_mb | 28569.195315 | 30230.139699 |
| genz_decode_weights_mb | 25127.125 | 26711.125 |
| genz_prefill_weights_mb | 25720.625 | 27304.625 |
| report_params | 13796380672 | 14626852864 |
| report_total_gb | 28.764482109 | 30.425426493 |
| report_weight_gb | 27.592761344 | 29.253705728 |
| runtime_chunked | [16.520671, 25720.625, 144.0] | [17.943546, 27304.625, 144.0] |
| runtime_decode | [6.05469, 25127.125, 96.75] | [7.117747, 26711.125, 96.75] |
| runtime_prefill | [20.272026, 25720.625, 96.0] | [23.346704, 27304.625, 96.0] |
| serving_weight_bytes | 3088580608 | 28631178048 |

## db:Qwen/Qwen2-0.5B (decoder-only)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 987922432 | 988010240 |
| serving_weight_bytes | 987922432 | 988010240 |

## db:Qwen/Qwen2.5-1.5B (decoder-only)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 3087138816 | 3087313920 |
| serving_weight_bytes | 3087138816 | 3087313920 |

## db:Qwen/Qwen2.5-14B (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 18 | 16 |
| conv_serving_weight_bytes | 27981250560 | 29539379200 |
| disagg_b_eff_4096 | 18 | 16 |
| serving_weight_bytes | 27981250560 | 29539379200 |

## db:Qwen/Qwen2.5-32B (decoder-only)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 63968378880 | 65526835200 |
| serving_weight_bytes | 63968378880 | 65526835200 |

## db:Qwen/Qwen2.5-3B (decoder-only)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 6171394048 | 6171693056 |
| serving_weight_bytes | 6171394048 | 6171693056 |

## db:Qwen/Qwen2.5-72B (decoder-only)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 142916714496 | 145410768896 |
| serving_weight_bytes | 142916714496 | 145410768896 |

## db:Qwen/Qwen2.5-7B (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 122 | 118 |
| conv_serving_weight_bytes | 14140571648 | 15230974976 |
| disagg_b_eff_4096 | 122 | 118 |
| serving_weight_bytes | 14140571648 | 15230974976 |

## db:THUDM/glm-4-9b-chat (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 283 | 280 |
| conv_serving_weight_bytes | 19136512000 | 19399319552 |
| disagg_b_eff_4096 | 283 | 280 |
| eng_None_32768_params | 7015636992 | 9699991552 |
| eng_None_32768_total_mb | 15588.199629 | 20956.908749 |
| eng_None_4096_params | 7015636992 | 9699991552 |
| eng_None_4096_total_mb | 14558.078566 | 19926.787686 |
| eng_llamacpp_32768_params | 7015636992 | 9699991552 |
| eng_llamacpp_32768_total_mb | 15588.199629 | 20956.908749 |
| eng_llamacpp_4096_params | 7015636992 | 9699991552 |
| eng_llamacpp_4096_total_mb | 14558.078566 | 19926.787686 |
| genz_decode_weights_mb | 13130.0 | 18250.0 |
| genz_prefill_weights_mb | 13380.0 | 18500.0 |
| report_params | 7015636992 | 9699991552 |
| report_total_gb | 14.663439483 | 20.032148603 |
| report_weight_gb | 14.031273984 | 19.399983104 |
| runtime_chunked | [12.146839, 13380.0, 15.0] | [16.746033, 18500.0, 15.0] |
| runtime_decode | [9.664202, 13130.0, 10.078] | [12.789541, 18250.0, 10.078] |
| runtime_prefill | [26.442664, 13380.0, 10.0] | [35.605261, 18500.0, 10.0] |
| serving_weight_bytes | 19136512000 | 19399319552 |

## db:deepseek-ai/DeepSeek-V3-Base (decoder-only)

| figure | before | after |
|---|---|---|
| conv_kv_size | 874496 | 35136 |
| conv_prefill_weights_mb | 73563.0 | 1279879.375 |
| conv_serving_weight_bytes | 75283038208 | 1342052838400 |
| disagg_transfer_bytes_640 | 1119354880 | 44974080 |
| genz_chunked_kv_mb | 7686.0 | 308.8125 |
| genz_decode_kv_mb | 152.5 | 274.5 |
| genz_decode_weights_mb | 1335494.375 | 1278111.875 |
| genz_ffn_kinds | {"MoE": 61} | {"Dense": 3, "MoE": 58} |
| genz_prefill_kv_mb | 152.5 | 274.5 |
| genz_prefill_weights_mb | 1337261.875 | 1279879.375 |
| prefix_kv_bytes_per_token | 1748992 | 70272 |
| runtime_chunked_error | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 678.35400390625 GB, KV Cache:1.2509765625. \n System:{'Flops': 400, 'Memory_size': 200, " | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 626.7662963867188 GB, KV Cache:0.050262451171875. \n System:{'Flops': 400, 'Memory_size'" |
| runtime_decode_error | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 653.0641479492188 GB, KV Cache:0.027923583984375. \n System:{'Flops': 400, 'Memory_size'" | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 625.0402221679688 GB, KV Cache:0.050262451171875. \n System:{'Flops': 400, 'Memory_size'" |
| runtime_prefill_error | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 654.7902221679688 GB, KV Cache:0.01861572265625. \n System:{'Flops': 400, 'Memory_size':" | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 626.7662963867188 GB, KV Cache:0.03350830078125. \n System:{'Flops': 400, 'Memory_size':" |
| serving_bytes_per_token | 1748992 | 70272 |
| serving_requests_8gib_1k | 4 | 119 |
| serving_weight_bytes | 75283038208 | 1342052838400 |

## db:deepseek-ai/deepseek-moe-16b-base (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 40 | 10 |
| conv_prefill_weights_mb | 5287.0 | 31234.0 |
| conv_serving_weight_bytes | 5124390912 | 32751459712 |
| disagg_b_eff_4096 | 40 | 10 |
| genz_decode_weights_mb | 30871.0 | 30834.0 |
| genz_ffn_kinds | {"MoE": 28} | {"Dense": 1, "MoE": 27} |
| genz_prefill_weights_mb | 31271.0 | 31234.0 |
| runtime_chunked | [26.401452, 38453.0, 168.0] | [20.212593, 31234.0, 168.0] |
| runtime_decode | [8.410507, 30871.0, 112.875] | [8.923105, 30834.0, 112.875] |
| runtime_prefill | [23.707989, 31271.0, 112.0] | [25.13573, 31234.0, 112.0] |
| serving_weight_bytes | 5124390912 | 32751459712 |

## db:intfloat/multilingual-e5-large-instruct (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 103 | 102 |
| conv_prefill_weights_mb | 1744.570312 | 1745.574219 |
| conv_serving_weight_bytes | 1317310464 | 1829414912 |
| disagg_b_eff_4096 | 103 | 102 |
| genz_prefill_weights_mb | 1552.570312 | 1553.574219 |
| runtime_chunked | [1.458098, 1552.57, 72.0] | [1.458996, 1553.574, 72.0] |
| runtime_prefill | [4.593068, 1552.57, 48.0] | [4.594865, 1553.574, 48.0] |
| serving_weight_bytes | 1317310464 | 1829414912 |

## db:tiiuae/falcon-7b-instruct (decoder-only)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 21734424576 | 22325953408 |
| serving_weight_bytes | 21734424576 | 22325953408 |

## db:tiiuae/falcon-mamba-7b (state-space)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 4 | 1 |
| conv_kv_size | 524288 | 0 |
| conv_serving_weight_bytes | 22007513088 | 108901965824 |
| conv_state_bytes | 0 | 293601280 |
| disagg_b_eff_4096 | 4 | 1 |
| disagg_transfer_bytes_640 | 671088640 | 293601280 |
| eng_None_32768_state_mb | 0.0 | 293.60128 |
| eng_None_32768_total_mb | 109466.73705 | 109760.33833 |
| eng_None_4096_state_mb | 0.0 | 293.60128 |
| eng_None_4096_total_mb | 109184.879821 | 109478.481101 |
| eng_llamacpp_32768_state_mb | 0.0 | 293.60128 |
| eng_llamacpp_32768_total_mb | 109466.73705 | 109760.33833 |
| eng_llamacpp_4096_state_mb | 0.0 | 293.60128 |
| eng_llamacpp_4096_total_mb | 109184.879821 | 109478.481101 |
| facts_weight_bytes | 104741625856 | 108901965824 |
| genz_kv_size | 524288 | 0 |
| prefix_kv_bytes_per_token | 1048576 | 0 |
| serving_bytes_per_token | 1048576 | 0 |
| serving_requests_8gib_1k | 8 | 29 |
| serving_state_bytes | 0 | 293601280 |
| serving_weight_bytes | 22007513088 | 108901965824 |

## fixture:EleutherAI_gpt-neo-1.3B.json (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 1 | 48 |
| conv_kv_size | 131072 | 98304 |
| conv_layers | [0, 32] | [0, 24] |
| conv_prefill_weights_mb | null | 4264.632812 |
| conv_serving_weight_bytes | null | 3633131520 |
| conv_serving_weight_bytes_error | "TypeError: unsupported operand type(s) for *: 'int' and 'NoneType'" | null |
| conv_weights_error | "TypeError: unsupported operand type(s) for /: 'NoneType' and 'int'" | null |
| decode_fit_refused | null | true |
| disagg_b_eff_4096 | 1 | 48 |
| disagg_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['activation_function', 'architectures', 'attention_dropout', 'attention_layers', 'attention_types', 'bos_token_id', 'embe" | null |
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
| facts_kv_bytes_4096 | 805306368 | 427819008 |
| facts_layers | [24, 0, 0, null] | [12, 12, 0, 256] |
| genz_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['activation_function', 'architectures', 'attention_dropout', 'attention_layers', 'attention_types', 'bos_token_id', 'embe" | null |
| genz_chunked_kv_mb | null | 504.0 |
| genz_decode_kv_mb | null | 408.0 |
| genz_decode_weights_mb | null | 3028.316406 |
| genz_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['activation_function', 'architectures', 'attention_dropout', 'attention_layers', 'attention_types', 'bos_token_id', 'embe" | null |
| genz_ffn_kinds | null | {"Dense": 24} |
| genz_kv_size | null | 98304 |
| genz_layer_kinds | null | {"MHA-global": 12, "MHA-local": 12} |
| genz_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['activation_function', 'architectures', 'attention_dropout', 'attention_layers', 'attention_types', 'bos_token_id', 'embe" | null |
| genz_prefill_kv_mb | null | 408.0 |
| genz_prefill_weights_mb | null | 3232.632812 |
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
| runtime_chunked | null | [3.018745, 3232.633, 120.0] |
| runtime_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['activation_function', 'architectures', 'attention_dropout', 'attention_layers', " | null |
| runtime_decode | null | [3.281436, 3028.316, 72.75] |
| runtime_decode_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['activation_function', 'architectures', 'attention_dropout', 'attention_layers', " | null |
| runtime_prefill | null | [7.656623, 3232.633, 72.0] |
| runtime_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['activation_function', 'architectures', 'attention_dropout', 'attention_layers', " | null |
| serving_bytes_per_token | null | 196608 |
| serving_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['activation_function', 'architectures', 'attention_dropout', 'attention_layers', 'attention_types', 'bos_token_id', 'embe" | null |
| serving_requests_8gib_1k | null | 42 |
| serving_state_bytes | null | 0 |
| serving_weight_bytes | null | 3633131520 |
| serving_weight_bytes_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['activation_function', 'architectures', 'attention_dropout', 'attention_layers', 'attention_types', 'bos_token_id', 'embe" | null |
| sim_max_running | 342 | 400 |
| slice_mib_16x4352 | [25088, true] | [17664, true] |
| state_gb | 0.00172032 | 0.0 |
| vllm_limits | [22571, 361136, 22571, 1, 0, 0.0, true] | [45143, 722288, 45143, 1, 1, 0.0, true] |
| vllm_running_1024 | 282 | 465 |
| vllm_running_8192 | 42 | 82 |

## fixture:LiquidAI_LFM2-2.6B.json (hybrid)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 151 | 557 |
| conv_kv_size | 30720 | 8192 |
| conv_prefill_weights_mb | 4892.0 | 5156.257812 |
| conv_serving_weight_bytes | 4861198336 | 5406978048 |
| conv_state_bytes | 0 | 270336 |
| disagg_b_eff_4096 | 151 | 557 |
| disagg_transfer_bytes_640 | 39321600 | 10756096 |
| eng_None_32768_state_mb | 0.0 | 0.270336 |
| eng_None_32768_total_mb | 6219.245158 | 6219.515494 |
| eng_None_4096_state_mb | 0.0 | 0.270336 |
| eng_None_4096_total_mb | 5477.692211 | 5477.962547 |
| eng_llamacpp_32768_state_mb | 0.0 | 0.270336 |
| eng_llamacpp_32768_total_mb | 6219.245158 | 6219.515494 |
| eng_llamacpp_4096_state_mb | 0.0 | 0.270336 |
| eng_llamacpp_4096_total_mb | 5477.692211 | 5477.962547 |
| facts_weight_bytes | 5805596928 | 5406978048 |
| genz_chunked_kv_mb | 270.0 | 72.515625 |
| genz_decode_kv_mb | 240.0 | 64.257812 |
| genz_decode_weights_mb | 3376.0 | 4900.257812 |
| genz_kv_size | 30720 | 8192 |
| genz_layer_kinds | {"MHA-global": 30} | {"MHA-global": 8, "Recurrent": 22} |
| genz_prefill_kv_mb | 240.0 | 64.257812 |
| genz_prefill_weights_mb | 3632.0 | 5156.257812 |
| prefix_kv_bytes_per_token | 61440 | 16384 |
| runtime_chunked | [3.342165, 3632.0, 45.0] | [4.692953, 5156.258, 12.516] |
| runtime_decode | [3.563216, 3376.0, 30.234] | [4.303897, 4900.258, 9.094] |
| runtime_prefill | [8.566374, 3632.0, 30.0] | [11.039983, 5156.258, 8.258] |
| serving_bytes_per_token | 61440 | 16384 |
| serving_requests_8gib_1k | 136 | 503 |
| serving_state_bytes | 0 | 270336 |
| serving_weight_bytes | 4861198336 | 5406978048 |
| slice_mib_16x4352 | [12288, true] | [12032, true] |
| vllm_limits | [263394, 4214304, 263394, 1, 0, 786432.0, true] | [264915, 4238640, 264915, 1, 0, 786432.0, true] |
| vllm_running_8192 | 496 | 498 |

## fixture:MiniMaxAI_MiniMax-M2.json (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 33 | 1 |
| conv_serving_weight_bytes | 8445493248 | 457378672640 |
| disagg_b_eff_4096 | 33 | 1 |
| facts_weight_bytes | 230023192576 | 230023224320 |
| serving_weight_bytes | 8445493248 | 457378672640 |

## fixture:MiniMaxAI_MiniMax-Text-01.json (hybrid)

| figure | before | after |
|---|---|---|
| conv_kv_size | 163840 | 20480 |
| conv_prefill_weights_mb | 851439.0 | 869919.0 |
| conv_serving_weight_bytes | 47756869632 | 912178168832 |
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
| facts_kv_bytes_4096 | 1342177280 | 167772160 |
| facts_kv_bytes_per_token | 327680 | 40960 |
| facts_layers | [80, 0, 0, null] | [10, 0, 70, null] |
| facts_state_bytes | 0 | 293601280 |
| facts_weight_bytes | 892800484352 | 912178168832 |
| genz_chunked_kv_mb | 1440.0 | 740.0 |
| genz_decode_kv_mb | 1280.0 | 440.0 |
| genz_decode_weights_mb | 849094.5 | 867574.5 |
| genz_kv_size | 163840 | 20480 |
| genz_layer_kinds | {"MHA-global": 80} | {"MHA-global": 10, "Recurrent": 70} |
| genz_prefill_kv_mb | 1280.0 | 440.0 |
| genz_prefill_weights_mb | 851439.0 | 869919.0 |
| prefix_kv_bytes_per_token | 327680 | 40960 |
| runtime_chunked_error | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 831.4833984375 GB, KV Cache:0.234375. \n System:{'Flops': 400, 'Memory_size': 200, 'Memo" | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 849.5302734375 GB, KV Cache:0.576171875. \n System:{'Flops': 400, 'Memory_size': 200, 'M" |
| runtime_decode_error | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 829.19384765625 GB, KV Cache:0.234375. \n System:{'Flops': 400, 'Memory_size': 200, 'Mem" | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 847.24072265625 GB, KV Cache:1.123046875. \n System:{'Flops': 400, 'Memory_size': 200, '" |
| runtime_prefill_error | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 831.4833984375 GB, KV Cache:0.15625. \n System:{'Flops': 400, 'Memory_size': 200, 'Memor" | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 849.5302734375 GB, KV Cache:0.29296875. \n System:{'Flops': 400, 'Memory_size': 200, 'Me" |
| serving_bytes_per_token | 327680 | 40960 |
| serving_state_bytes | 0 | 293601280 |
| serving_weight_bytes | 47756869632 | 912178168832 |
| vllm_limits | [0, 0, 0, 1, 0, 0.0, false] | [0, 0, 0, 1, 0, 293601280.0, false] |

## fixture:PowerInfer_SmallThinker-21BA3B-Instruct.json (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 71 | 1 |
| conv_prefill_weights_mb | 11948.75 | 41020.0 |
| conv_serving_weight_bytes | 11751260160 | 43013131776 |
| disagg_b_eff_4096 | 71 | 1 |
| facts_full_attention | true | false |
| facts_layers | [52, 0, 0, null] | [13, 39, 0, 4096] |
| facts_weight_bytes | 11916293120 | 43013131776 |
| genz_decode_weights_mb | 8411.875 | 40278.125 |
| genz_ffn_kinds | {"Dense": 52} | {"MoE": 52} |
| genz_layer_kinds | {"MHA-global": 52} | {"MHA-global": 13, "MHA-local": 39} |
| genz_prefill_weights_mb | 9153.75 | 41020.0 |
| runtime_chunked | [8.403656, 9153.75, 78.0] | [26.506247, 41020.0, 78.0] |
| runtime_decode | [7.004426, 8411.875, 52.406] | [11.413993, 40278.125, 52.406] |
| runtime_prefill | [19.303921, 9153.75, 52.0] | [32.946381, 41020.0, 52.0] |
| serving_weight_bytes | 11751260160 | 43013131776 |
| sim_max_running | 400 | 279 |
| slice_mib_16x4352 | [26368, true] | [58368, true] |
| vllm_limits | [36572, 585152, 36572, 1, 0, 0.0, true] | [73290, 1172640, 73290, 1, 3, 0.0, true] |
| vllm_running_1024 | 457 | 229 |
| vllm_running_8192 | 69 | 56 |

## fixture:Qwen_Qwen3-Next-80B-A3B-Instruct.json (hybrid)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 93 | 1 |
| conv_kv_size | 49152 | 12288 |
| conv_prefill_weights_mb | 150467.0 | 151966.390625 |
| conv_serving_weight_bytes | 5454168064 | 159348635136 |
| conv_state_bytes | 0 | 77266944 |
| disagg_b_eff_4096 | 93 | 1 |
| disagg_transfer_bytes_640 | 62914560 | 92995584 |
| eng_None_32768_params | 79733992192 | 79985650432 |
| eng_None_32768_state_mb | 0.0 | 77.266944 |
| eng_None_32768_total_mb | 160595.413299 | 161175.996723 |
| eng_None_4096_params | 79733992192 | 79985650432 |
| eng_None_4096_state_mb | 0.0 | 77.266944 |
| eng_None_4096_total_mb | 159729.708954 | 160310.292378 |
| eng_llamacpp_32768_params | 79733992192 | 79985650432 |
| eng_llamacpp_32768_state_mb | 0.0 | 77.266944 |
| eng_llamacpp_32768_total_mb | 160595.413299 | 161175.996723 |
| eng_llamacpp_4096_params | 79733992192 | 79985650432 |
| eng_llamacpp_4096_state_mb | 0.0 | 77.266944 |
| eng_llamacpp_4096_total_mb | 159729.708954 | 160310.292378 |
| facts_weight_bytes | 159147308544 | 159348635136 |
| genz_chunked_kv_mb | 432.0 | 255.375 |
| genz_decode_kv_mb | 384.0 | 169.6875 |
| genz_decode_weights_mb | 149873.5 | 151372.890625 |
| genz_kv_size | 49152 | 12288 |
| genz_layer_kinds | {"MHA-global": 48} | {"MHA-global": 12, "Recurrent": 36} |
| genz_prefill_kv_mb | 384.0 | 169.6875 |
| genz_prefill_weights_mb | 150467.0 | 151966.390625 |
| prefix_kv_bytes_per_token | 98304 | 24576 |
| report_params | 79733992192 | 79985650432 |
| report_total_gb | 159.8747742 | 160.37809068 |
| report_weight_gb | 159.467984384 | 159.971300864 |
| runtime_chunked | [93.106947, 150467.0, 72.0] | [94.206186, 151966.391, 165.375] |
| runtime_decode | [10.423923, 149873.5, 48.375] | [11.472062, 151372.891, 306.844] |
| runtime_prefill | [98.864405, 150467.0, 48.0] | [102.259445, 151966.391, 85.688] |
| serving_bytes_per_token | 98304 | 24576 |
| serving_requests_8gib_1k | 85 | 83 |
| serving_state_bytes | 0 | 77266944 |
| serving_weight_bytes | 5454168064 | 159348635136 |

## fixture:Qwen_Qwen3.5-4B.json (multimodal)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 66 | 182 |
| conv_kv_size | 262144 | 16384 |
| conv_prefill_weights_mb | 12852.0 | 9234.023438 |
| conv_serving_weight_bytes | 13214154752 | 9078531072 |
| conv_state_bytes | 0 | 51511296 |
| disagg_b_eff_4096 | 66 | 182 |
| disagg_transfer_bytes_640 | 83886080 | 72482816 |
| eng_None_32768_params | 4945124864 | 5029010944 |
| eng_None_32768_state_mb | 0.0 | 51.511296 |
| eng_None_32768_total_mb | 11478.217318 | 11697.500774 |
| eng_None_4096_params | 4945124864 | 5029010944 |
| eng_None_4096_state_mb | 0.0 | 51.511296 |
| eng_None_4096_total_mb | 10282.001818 | 10501.285274 |
| eng_llamacpp_32768_params | 4945124864 | 5029010944 |
| eng_llamacpp_32768_state_mb | 0.0 | 51.511296 |
| eng_llamacpp_32768_total_mb | 11478.217318 | 11697.500774 |
| eng_llamacpp_4096_params | 4945124864 | 5029010944 |
| eng_llamacpp_4096_state_mb | 0.0 | 51.511296 |
| eng_llamacpp_4096_total_mb | 10282.001818 | 10501.285274 |
| genz_chunked_kv_mb | 576.0 | 242.25 |
| genz_decode_kv_mb | 512.0 | 177.125 |
| genz_decode_weights_mb | 7132.5 | 8021.523438 |
| genz_kv_size | 65536 | 16384 |
| genz_layer_kinds | {"MHA-global": 32} | {"MHA-global": 8, "Recurrent": 24} |
| genz_prefill_kv_mb | 512.0 | 177.125 |
| genz_prefill_weights_mb | 8345.0 | 9234.023438 |
| prefix_kv_bytes_per_token | 131072 | 32768 |
| report_params | 4945124864 | 5029010944 |
| report_total_gb | 10.422165791 | 10.589937951 |
| report_weight_gb | 9.890249728 | 10.058021888 |
| runtime_chunked | [7.637411, 8345.0, 96.0] | [8.59528, 9234.023, 122.25] |
| runtime_decode | [5.911349, 7132.5, 64.5] | [6.398831, 8021.523, 212.625] |
| runtime_prefill | [17.133994, 8345.0, 64.0] | [18.879089, 9234.023, 65.125] |
| serving_bytes_per_token | 131072 | 32768 |
| serving_requests_8gib_1k | 64 | 100 |
| serving_state_bytes | 0 | 51511296 |
| serving_weight_bytes | 7478968320 | 9078531072 |

## fixture:Qwen_Qwen3.6-27B.json (multimodal)

| figure | before | after |
|---|---|---|
| conv_kv_size | 262144 | 32768 |
| conv_layers | [0, 32] | [0, 64] |
| conv_prefill_weights_mb | 12852.0 | 51299.21875 |
| conv_serving_weight_bytes | 13214154752 | 54713457120 |
| conv_state_bytes | 0 | 153944064 |
| disagg_transfer_bytes_640 | 167772160 | 195887104 |
| eng_None_32768_params | 27919350016 | 28422666496 |
| eng_None_32768_state_mb | 0.0 | 153.944064 |
| eng_None_32768_total_mb | 58973.596557 | 60134.173581 |
| eng_None_4096_params | 27919350016 | 28422666496 |
| eng_None_4096_state_mb | 0.0 | 153.944064 |
| eng_None_4096_total_mb | 56601.298214 | 57761.875238 |
| eng_llamacpp_32768_params | 27919350016 | 28422666496 |
| eng_llamacpp_32768_state_mb | 0.0 | 153.944064 |
| eng_llamacpp_32768_total_mb | 58973.596557 | 60134.173581 |
| eng_llamacpp_4096_params | 27919350016 | 28422666496 |
| eng_llamacpp_4096_state_mb | 0.0 | 153.944064 |
| eng_llamacpp_4096_total_mb | 56601.298214 | 57761.875238 |
| genz_chunked_kv_mb | 1152.0 | 581.625 |
| genz_decode_kv_mb | 1024.0 | 402.8125 |
| genz_decode_weights_mb | 44025.0 | 48874.21875 |
| genz_kv_size | 131072 | 32768 |
| genz_layer_kinds | {"MHA-global": 64} | {"MHA-global": 16, "Recurrent": 48} |
| genz_prefill_kv_mb | 1024.0 | 402.8125 |
| genz_prefill_weights_mb | 46450.0 | 51299.21875 |
| prefix_kv_bytes_per_token | 262144 | 65536 |
| report_params | 27919350016 | 28422666496 |
| report_total_gb | 56.938550728 | 57.945183688 |
| report_weight_gb | 55.838700032 | 56.845332992 |
| runtime_chunked | [42.094544, 46450.0, 192.0] | [46.883662, 51299.219, 341.625] |
| runtime_decode | [28.985134, 44025.0, 129.0] | [31.953231, 48874.219, 619.5] |
| runtime_prefill | [86.74405, 46450.0, 128.0] | [95.829491, 51299.219, 178.812] |
| serving_bytes_per_token | 262144 | 65536 |
| serving_requests_8gib_1k | 32 | 38 |
| serving_state_bytes | 0 | 153944064 |
| serving_weight_bytes | 46163558400 | 54713457120 |

## fixture:Salesforce_blip-image-captioning-base_hub.json (multimodal)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 282 | 245 |
| conv_kv_size | 262144 | 18432 |
| conv_layers | [0, 32] | [12, 12] |
| conv_prefill_weights_mb | 12852.0 | 578.148926 |
| conv_serving_weight_bytes | 13214154752 | 673806336 |
| conv_state_bytes | 0 | 21270528 |
| disagg_b_eff_4096 | 282 | 245 |
| disagg_transfer_bytes_640 | 23592960 | 44863488 |
| eng_None_32768_kv_mb | 1207.959552 | 1229.23008 |
| eng_None_32768_params | 123532032 | 223209216 |
| eng_None_32768_total_mb | 1568.151859 | 1788.776755 |
| eng_None_4096_kv_mb | 150.994944 | 172.265472 |
| eng_None_4096_params | 123532032 | 223209216 |
| eng_None_4096_total_mb | 458.339021 | 678.963917 |
| eng_llamacpp_32768_kv_mb | 1207.959552 | 1229.23008 |
| eng_llamacpp_32768_params | 123532032 | 223209216 |
| eng_llamacpp_32768_total_mb | 1568.151859 | 1788.776755 |
| eng_llamacpp_4096_kv_mb | 150.994944 | 172.265472 |
| eng_llamacpp_4096_params | 123532032 | 223209216 |
| eng_llamacpp_4096_total_mb | 458.339021 | 678.963917 |
| facts_kv_bytes_4096 | 150994944 | 172265472 |
| facts_weight_bytes | 447071232 | 673806336 |
| genz_chunked_kv_mb | 162.0 | 202.570312 |
| genz_decode_kv_mb | 144.0 | 164.285156 |
| genz_decode_weights_mb | 206.712891 | 424.686035 |
| genz_prefill_kv_mb | 144.0 | 184.570312 |
| genz_prefill_weights_mb | 251.425781 | 470.148926 |
| kv_cross_layers | 0 | 12 |
| kv_gb@32768 | 1.207959552 | 1.22923008 |
| kv_gb@4096 | 0.150994944 | 0.172265472 |
| kvgeo_32768 | [36864.0, 12, 0, 0.0, 16, 0.0] | [36864.0, 12, 0, 0.0, 16, 21270528.0] |
| kvgeo_4096 | [36864.0, 12, 0, 0.0, 16, 0.0] | [36864.0, 12, 0, 0.0, 16, 21270528.0] |
| report_kv_gb | 0.150994944 | 0.172265472 |
| report_params | 123532032 | 223209216 |
| report_total_gb | 0.500594012 | 0.725473014 |
| report_weight_gb | 0.247064064 | 0.446418432 |
| runtime_chunked | [0.25013, 251.426, 27.0] | [0.309209, 470.149, 67.57] |
| runtime_decode | [1.332181, 206.713, 18.141] | [1.616452, 424.686, 99.281] |
| runtime_prefill | [1.851682, 251.426, 18.0] | [3.024723, 470.149, 58.57] |
| serving_requests_8gib_1k | 227 | 145 |
| serving_state_bytes | 0 | 21270528 |
| serving_weight_bytes | 273377280 | 673806336 |
| slice_mib_16x4352 | [8960, true] | [9472, true] |
| vllm_limits | [126274, 2020384, 126274, 1, 0, 0.0, true] | [125890, 2014240, 125890, 1, 0, 21270528.0, true] |
| vllm_running_8192 | 239 | 223 |

## fixture:Zyphra_Zamba2-2.7B.json (hybrid)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 13 | 44 |
| conv_kv_size | 276480 | 92160 |
| conv_prefill_weights_mb | 11112.5 | 6670.771484 |
| conv_serving_weight_bytes | 11488460800 | 5894182720 |
| conv_state_bytes | 0 | 72479232 |
| disagg_b_eff_4096 | 13 | 44 |
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
| facts_kv_bytes_4096 | 377487360 | 754974720 |
| facts_kv_bytes_per_token | 92160 | 184320 |
| facts_weight_bytes | 5107183936 | 5894182720 |
| genz_chunked_kv_mb | 2430.0 | 948.243164 |
| genz_decode_kv_mb | 35.859375 | 789.121582 |
| genz_decode_weights_mb | 9842.5 | 6514.521484 |
| genz_ffn_kinds | {"Dense": 54} | {"Dense": 9, "no_op": 45} |
| genz_kv_size | 276480 | 92160 |
| genz_layer_kinds | {"Mamba": 54} | {"MHA-global": 9, "Recurrent": 45} |
| genz_prefill_kv_mb | 35.859375 | 789.121582 |
| genz_prefill_weights_mb | 9998.75 | 6670.771484 |
| kv_gb@32768 | 3.01989888 | 6.03979776 |
| kv_gb@4096 | 0.37748736 | 0.75497472 |
| kvgeo_32768 | [92160.0, 9, 54, 1342208.0, 144, 0.0] | [184320.0, 9, 54, 1342208.0, 80, 0.0] |
| kvgeo_4096 | [92160.0, 9, 54, 1342208.0, 144, 0.0] | [184320.0, 9, 54, 1342208.0, 80, 0.0] |
| prefix_kv_bytes_per_token | 552960 | 184320 |
| report_kv_gb | 0.37748736 | 0.75497472 |
| report_params | 2446020000 | 2577092000 |
| report_total_gb | 5.643392832 | 6.479317619 |
| report_weight_gb | 4.89204 | 5.154184 |
| runtime_chunked | [7.915387, 8412.5, 405.0] | [6.535956, 6670.771, 273.243] |
| runtime_decode | [8.760858, 9842.5, 137.109] | [5.675477, 6514.521, 367.189] |
| runtime_prefill | [51.836922, 9998.75, 35.859] | [14.889247, 6670.771, 159.122] |
| serving_bytes_per_token | 552960 | 184320 |
| serving_requests_8gib_1k | 15 | 32 |
| serving_state_bytes | 0 | 72479232 |
| serving_weight_bytes | 11488460800 | 5894182720 |
| sim_max_running | 376 | 240 |
| slice_mib_16x4352 | [16896, true] | [27136, true] |
| vllm_limits | [5261, 757584, 5261, 1, 0, 79626240.0, true] | [4682, 374560, 4682, 1, 0, 88473600.0, true] |
| vllm_running_1024 | 350 | 212 |
| vllm_running_8192 | 80 | 41 |

## fixture:ai21labs_Jamba-v0.1.json (hybrid)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 53 | 1 |
| conv_kv_size | 65536 | 8192 |
| conv_prefill_weights_mb | 175620.0 | 100410.0625 |
| conv_serving_weight_bytes | 14495514624 | 187699897344 |
| conv_state_bytes | 0 | 16056320 |
| decode_fit_refused | null | true |
| disagg_b_eff_4096 | 53 | 1 |
| disagg_transfer_bytes_640 | 83886080 | 26542080 |
| eng_None_32768_state_mb | 0.0 | 16.05632 |
| eng_None_32768_total_mb | 107731.409306 | 107747.465626 |
| eng_None_4096_state_mb | 0.0 | 16.05632 |
| eng_None_4096_total_mb | 106577.136845 | 106593.193165 |
| eng_llamacpp_32768_state_mb | 0.0 | 16.05632 |
| eng_llamacpp_32768_total_mb | 107731.409306 | 107747.465626 |
| eng_llamacpp_4096_state_mb | 0.0 | 16.05632 |
| eng_llamacpp_4096_total_mb | 106577.136845 | 106593.193165 |
| facts_weight_bytes | 187478807552 | 187699897344 |
| genz_chunked_kv_mb | 576.0 | 102.625 |
| genz_decode_kv_mb | null | 79.3125 |
| genz_decode_weights_mb | null | 97850.0625 |
| genz_error | "ValueError: More than 1 unique layers not supported. Work in progress" | null |
| genz_kv_size | 65536 | 8192 |
| genz_layer_kinds | {"Mamba": 32} | {"MHA-global": 4, "Recurrent": 28} |
| genz_prefill_error | "ValueError: More than 1 unique layers not supported. Work in progress" | null |
| genz_prefill_kv_mb | null | 79.3125 |
| genz_prefill_weights_mb | null | 100410.0625 |
| prefix_kv_bytes_per_token | 131072 | 16384 |
| runtime_chunked | [109.243784, 175620.0, 96.0] | [66.166645, 100410.062, 42.625] |
| runtime_decode | null | [31.680898, 97850.062, 69.312] |
| runtime_decode_error | "ValueError: More than 1 unique layers not supported. Work in progress" | null |
| runtime_prefill | null | [81.62546, 100410.062, 23.312] |
| runtime_prefill_error | "ValueError: More than 1 unique layers not supported. Work in progress" | null |
| serving_bytes_per_token | 131072 | 16384 |
| serving_requests_8gib_1k | 64 | 261 |
| serving_state_bytes | 0 | 16056320 |
| serving_weight_bytes | 14495514624 | 187699897344 |

## fixture:bart_transformers-5.12-default.json (encoder-decoder)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 1 | 104 |
| conv_kv_size | 98304 | 49152 |
| conv_prefill_weights_mb | 5417.390625 | 1062.347656 |
| conv_serving_weight_bytes | 5268774912 | 908623872 |
| decode_fit_refused | null | true |
| disagg_b_eff_4096 | 1 | 104 |
| disagg_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'architectures', 'attention_dropout', 'bos_token_id', 'chun" | null |
| disagg_transfer_bytes_640 | null | 62914560 |
| eng_None_32768_kv_mb | 1758.068736 | 3294.953472 |
| eng_None_32768_total_mb | 2718.961254 | 4255.84599 |
| eng_None_4096_kv_mb | 348.782592 | 476.381184 |
| eng_None_4096_total_mb | 1239.210803 | 1366.809395 |
| eng_llamacpp_32768_kv_mb | 1758.068736 | 3294.953472 |
| eng_llamacpp_32768_total_mb | 2718.961254 | 4255.84599 |
| eng_llamacpp_4096_kv_mb | 348.782592 | 476.381184 |
| eng_llamacpp_4096_total_mb | 1239.210803 | 1366.809395 |
| facts_kv_bytes_4096 | 201326592 | 402653184 |
| facts_weight_bytes | 908300288 | 908623872 |
| genz_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'architectures', 'attention_dropout', 'bos_token_id', 'chun" | null |
| genz_chunked_kv_mb | null | 432.0 |
| genz_decode_kv_mb | null | 384.0 |
| genz_decode_weights_mb | null | 770.173828 |
| genz_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'architectures', 'attention_dropout', 'bos_token_id', 'chun" | null |
| genz_ffn_kinds | null | {"Dense": 12} |
| genz_kv_size | null | 49152 |
| genz_layer_kinds | null | {"MHA-global": 12} |
| genz_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'architectures', 'attention_dropout', 'bos_token_id', 'chun" | null |
| genz_prefill_kv_mb | null | 576.0 |
| genz_prefill_weights_mb | null | 870.347656 |
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
| runtime_chunked | null | [0.53918, 870.348, 72.0] |
| runtime_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'architectures', 'a" | null |
| runtime_decode | null | [1.753666, 770.174, 120.188] |
| runtime_decode_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'architectures', 'a" | null |
| runtime_prefill | null | [3.715765, 870.348, 72.0] |
| runtime_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'architectures', 'a" | null |
| serving_bytes_per_token | null | 98304 |
| serving_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'architectures', 'attention_dropout', 'bos_token_id', 'chun" | null |
| serving_requests_8gib_1k | null | 85 |
| serving_state_bytes | null | 0 |
| serving_weight_bytes | null | 908623872 |
| serving_weight_bytes_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'architectures', 'attention_dropout', 'bos_token_id', 'chun" | null |
| slice_mib_16x4352 | [10240, true] | [14080, true] |
| vllm_limits | [94617, 1513872, 94617, 1, 0, 0.0, true] | [94617, 1513872, 94617, 1, 0, 50331648.0, true] |
| vllm_running_1024 | 1024 | 657 |
| vllm_running_8192 | 179 | 159 |

## fixture:blip2_flan-t5-xl_transformers-5.12.json (multimodal)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 1 | 21 |
| conv_kv_size | 131072 | 196608 |
| conv_layers | [16, 16] | [24, 24] |
| conv_prefill_weights_mb | 6676.0 | 5435.0 |
| conv_serving_weight_bytes | 6738149376 | 7763148032 |
| decode_fit_refused | null | true |
| disagg_b_eff_4096 | 1 | 21 |
| disagg_error | "TypeError: 'NoneType' object is not iterable" | null |
| disagg_transfer_bytes_640 | null | 251658240 |
| eng_None_32768_kv_mb | 0.0 | 12884.901888 |
| eng_None_32768_params | 1720216064 | 3942594816 |
| eng_None_32768_total_mb | 3768.924774 | 21098.584166 |
| eng_None_4096_kv_mb | 0.0 | 1610.612736 |
| eng_None_4096_params | 1720216064 | 3942594816 |
| eng_None_4096_total_mb | 3607.863501 | 9663.233741 |
| eng_llamacpp_32768_kv_mb | 0.0 | 12884.901888 |
| eng_llamacpp_32768_params | 1720216064 | 3942594816 |
| eng_llamacpp_32768_total_mb | 3768.924774 | 21098.584166 |
| eng_llamacpp_4096_kv_mb | 0.0 | 1610.612736 |
| eng_llamacpp_4096_params | 1720216064 | 3942594816 |
| eng_llamacpp_4096_total_mb | 3607.863501 | 9663.233741 |
| facts_kv_bytes_4096 | 805306368 | 1610612736 |
| facts_weight_bytes | 8553361408 | 7763148032 |
| genz_chunked_error | "TypeError: 'NoneType' object is not iterable" | null |
| genz_chunked_kv_mb | null | 1728.0 |
| genz_decode_kv_mb | null | 1536.0 |
| genz_decode_weights_mb | null | 5309.5 |
| genz_error | "TypeError: 'NoneType' object is not iterable" | null |
| genz_ffn_kinds | null | {"Dense": 24} |
| genz_kv_size | null | 196608 |
| genz_layer_kinds | null | {"MHA-global": 24} |
| genz_prefill_error | "TypeError: 'NoneType' object is not iterable" | null |
| genz_prefill_kv_mb | null | 2304.0 |
| genz_prefill_weights_mb | null | 5435.0 |
| kv_cross_layers | 0 | 24 |
| kv_gb@32768 | 0.0 | 12.884901888 |
| kv_gb@4096 | 0.0 | 1.610612736 |
| kvgeo_32768 | [0.0, 1, 0, 0.0, 16, 0.0] | [393216.0, 24, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [0.0, 1, 0, 0.0, 16, 0.0] | [393216.0, 24, 0, 0.0, 16, 0.0] |
| prefix_error | "TypeError: 'NoneType' object is not iterable" | null |
| prefix_kv_bytes_per_token | null | 393216 |
| report_kv_gb | 0.0 | 1.610612736 |
| report_params | 1720216064 | 3942594816 |
| report_total_gb | 3.641349775 | 10.018842563 |
| report_weight_gb | 3.440432128 | 7.885189632 |
| runtime_chunked | null | [2.787976, 5435.0, 288.0] |
| runtime_chunked_error | "TypeError: 'NoneType' object is not iterable" | null |
| runtime_decode | null | [3.790492, 5309.5, 480.75] |
| runtime_decode_error | "TypeError: 'NoneType' object is not iterable" | null |
| runtime_prefill | null | [13.28918, 5435.0, 288.0] |
| runtime_prefill_error | "TypeError: 'NoneType' object is not iterable" | null |
| serving_bytes_per_token | null | 393216 |
| serving_error | "TypeError: 'NoneType' object is not iterable" | null |
| serving_requests_8gib_1k | null | 21 |
| serving_state_bytes | null | 0 |
| serving_weight_bytes | null | 7763148032 |
| serving_weight_bytes_error | "TypeError: 'NoneType' object is not iterable" | null |
| sim_max_running | 322 | 66 |
| slice_mib_16x4352 | [29952, true] | [44800, true] |
| vllm_limits | [21227, 339632, 21227, 1, 0, 0.0, true] | [21478, 343648, 21478, 1, 0, 805306368.0, true] |
| vllm_running_1024 | 265 | 63 |
| vllm_running_8192 | 40 | 27 |

## fixture:config.json (decoder-only)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 1191968768 | 1192099840 |
| serving_weight_bytes | 1191968768 | 1192099840 |

## fixture:deepseek-ai_DeepSeek-V3.2-Exp.json (decoder-only)

| figure | before | after |
|---|---|---|
| conv_kv_size | 874496 | 35136 |
| conv_prefill_weights_mb | 73563.0 | 1281503.5 |
| conv_serving_weight_bytes | 75283038208 | 1342052838400 |
| disagg_transfer_bytes_640 | 1119354880 | 44974080 |
| genz_chunked_kv_mb | 7686.0 | 308.8125 |
| genz_decode_kv_mb | 152.5 | 274.5 |
| genz_decode_weights_mb | 1335494.375 | 1279736.0 |
| genz_ffn_kinds | {"MoE": 61} | {"Dense": 3, "MoE": 58} |
| genz_prefill_kv_mb | 152.5 | 274.5 |
| genz_prefill_weights_mb | 1337261.875 | 1281503.5 |
| prefix_kv_bytes_per_token | 1748992 | 70272 |
| runtime_chunked_error | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 678.35400390625 GB, KV Cache:1.2509765625. \n System:{'Flops': 400, 'Memory_size': 200, " | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 627.559326171875 GB, KV Cache:0.050262451171875. \n System:{'Flops': 400, 'Memory_size':" |
| runtime_decode_error | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 653.0641479492188 GB, KV Cache:0.027923583984375. \n System:{'Flops': 400, 'Memory_size'" | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 625.833251953125 GB, KV Cache:0.050262451171875. \n System:{'Flops': 400, 'Memory_size':" |
| runtime_prefill_error | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 654.7902221679688 GB, KV Cache:0.01861572265625. \n System:{'Flops': 400, 'Memory_size':" | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 627.559326171875 GB, KV Cache:0.03350830078125. \n System:{'Flops': 400, 'Memory_size': " |
| serving_bytes_per_token | 1748992 | 70272 |
| serving_requests_8gib_1k | 4 | 119 |
| serving_weight_bytes | 75283038208 | 1342052838400 |

## fixture:deepseek-v3.json (decoder-only)

| figure | before | after |
|---|---|---|
| conv_kv_size | 874496 | 35136 |
| conv_prefill_weights_mb | 73563.0 | 1281503.5 |
| conv_serving_weight_bytes | 75283038208 | 1342052838400 |
| disagg_transfer_bytes_640 | 1119354880 | 44974080 |
| genz_chunked_kv_mb | 7686.0 | 308.8125 |
| genz_decode_kv_mb | 152.5 | 274.5 |
| genz_decode_weights_mb | 1335494.375 | 1279736.0 |
| genz_ffn_kinds | {"MoE": 61} | {"Dense": 3, "MoE": 58} |
| genz_prefill_kv_mb | 152.5 | 274.5 |
| genz_prefill_weights_mb | 1337261.875 | 1281503.5 |
| prefix_kv_bytes_per_token | 1748992 | 70272 |
| runtime_chunked_error | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 678.35400390625 GB, KV Cache:1.2509765625. \n System:{'Flops': 400, 'Memory_size': 200, " | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 627.559326171875 GB, KV Cache:0.050262451171875. \n System:{'Flops': 400, 'Memory_size':" |
| runtime_decode_error | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 653.0641479492188 GB, KV Cache:0.027923583984375. \n System:{'Flops': 400, 'Memory_size'" | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 625.833251953125 GB, KV Cache:0.050262451171875. \n System:{'Flops': 400, 'Memory_size':" |
| runtime_prefill_error | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 654.7902221679688 GB, KV Cache:0.01861572265625. \n System:{'Flops': 400, 'Memory_size':" | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 627.559326171875 GB, KV Cache:0.03350830078125. \n System:{'Flops': 400, 'Memory_size': " |
| serving_bytes_per_token | 1748992 | 70272 |
| serving_requests_8gib_1k | 4 | 119 |
| serving_weight_bytes | 75283038208 | 1342052838400 |

## fixture:deepseek_v2_lite_chat_config.json (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 41 | 90 |
| conv_kv_size | 110592 | 15552 |
| conv_prefill_weights_mb | 5126.75 | 29957.5 |
| conv_serving_weight_bytes | 4956356608 | 31412971776 |
| decode_fit_refused | null | true |
| disagg_b_eff_4096 | 41 | 90 |
| disagg_transfer_bytes_640 | 141557760 | 19906560 |
| genz_chunked_kv_mb | 972.0 | 136.6875 |
| genz_decode_kv_mb | null | 121.5 |
| genz_decode_weights_mb | null | 29557.5 |
| genz_error | "ValueError: cannot convert float NaN to integer" | null |
| genz_ffn_kinds | {"MoE": 27} | {"Dense": 1, "MoE": 26} |
| genz_prefill_error | "ValueError: cannot convert float NaN to integer" | null |
| genz_prefill_kv_mb | null | 121.5 |
| genz_prefill_weights_mb | null | 29957.5 |
| prefix_kv_bytes_per_token | 221184 | 31104 |
| runtime_chunked | [25.484159, 37108.25, 162.0] | [19.466594, 29957.5, 22.781] |
| runtime_decode | null | [9.454812, 29557.5, 15.306] |
| runtime_decode_error | "ValueError: cannot convert float NaN to integer" | null |
| runtime_prefill | null | [24.395485, 29957.5, 15.188] |
| runtime_prefill_error | "ValueError: cannot convert float NaN to integer" | null |
| serving_bytes_per_token | 221184 | 31104 |
| serving_requests_8gib_1k | 37 | 269 |
| serving_weight_bytes | 4956356608 | 31412971776 |

## fixture:eagle3-gpt-oss-20b-redhatai.json (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 4940 | 4801 |
| conv_serving_weight_bytes | 1509949440 | 2668233600 |
| disagg_b_eff_4096 | 4940 | 4801 |
| eng_None_32768_params | 1132470720 | 1711604160 |
| eng_None_32768_total_mb | 3181.396864 | 4339.663744 |
| eng_None_4096_params | 1132470720 | 1711604160 |
| eng_None_4096_total_mb | 2698.003328 | 3856.270208 |
| eng_llamacpp_32768_params | 1132470720 | 1711604160 |
| eng_llamacpp_32768_total_mb | 3181.396864 | 4339.663744 |
| eng_llamacpp_4096_params | 1132470720 | 1711604160 |
| eng_llamacpp_4096_total_mb | 2698.003328 | 3856.270208 |
| report_params | 1132470720 | 1711604160 |
| report_total_gb | 2.784615706 | 3.942882586 |
| report_weight_gb | 2.26494144 | 3.42320832 |
| serving_weight_bytes | 1509949440 | 2668233600 |

## fixture:eagle3-gpt-oss-20b-zhuyksir.json (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 4933 | 4795 |
| conv_serving_weight_bytes | 1565245440 | 2723529600 |
| disagg_b_eff_4096 | 4933 | 4795 |
| eng_None_32768_params | 1160118720 | 1739252160 |
| eng_None_32768_total_mb | 3362.521984 | 4520.788864 |
| eng_None_4096_params | 1160118720 | 1739252160 |
| eng_None_4096_total_mb | 2816.213888 | 3974.480768 |
| eng_llamacpp_32768_params | 1160118720 | 1739252160 |
| eng_llamacpp_32768_total_mb | 3362.521984 | 4520.788864 |
| eng_llamacpp_4096_params | 1160118720 | 1739252160 |
| eng_llamacpp_4096_total_mb | 2816.213888 | 3974.480768 |
| report_params | 1160118720 | 1739252160 |
| report_total_gb | 2.915409178 | 4.073676058 |
| report_weight_gb | 2.32023744 | 3.47850432 |
| serving_weight_bytes | 1565245440 | 2723529600 |

## fixture:eagle3-qwen3-8b-angelslim.json (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 2462 | 2388 |
| conv_serving_weight_bytes | 1630535680 | 2875219968 |
| disagg_b_eff_4096 | 2462 | 2388 |
| serving_weight_bytes | 1630535680 | 2875219968 |

## fixture:eagle3-qwen3-8b-redhatai.json (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 2462 | 2388 |
| conv_serving_weight_bytes | 1630535680 | 2875219968 |
| disagg_b_eff_4096 | 2462 | 2388 |
| serving_weight_bytes | 1630535680 | 2875219968 |

## fixture:eagle3-qwen3-8b-tengyunw.json (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 2462 | 2388 |
| conv_serving_weight_bytes | 1630535680 | 2875219968 |
| disagg_b_eff_4096 | 2462 | 2388 |
| serving_weight_bytes | 1630535680 | 2875219968 |

## fixture:facebook_musicgen-small_hub.json (encoder-decoder)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 104 | 51 |
| conv_kv_size | 131072 | 98304 |
| conv_layers | [16, 16] | [12, 24] |
| conv_prefill_weights_mb | 6676.0 | 1260.5625 |
| conv_serving_weight_bytes | 6738149376 | 1185233920 |
| disagg_b_eff_4096 | 104 | 51 |
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
| facts_kv_bytes_4096 | 2147483648 | 805306368 |
| facts_kv_bytes_per_token | 524288 | 98304 |
| facts_layers | [32, 0, 0, null] | [24, 0, 0, null] |
| facts_weight_bytes | 34884558848 | 1185233920 |
| genz_chunked_kv_mb | 432.0 | 864.0 |
| genz_decode_kv_mb | 384.0 | 768.0 |
| genz_decode_weights_mb | 580.0 | 994.5625 |
| genz_kv_size | 49152 | 98304 |
| genz_prefill_kv_mb | 384.0 | 912.0 |
| genz_prefill_weights_mb | 584.0 | 1014.5625 |
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
| runtime_chunked | [0.589743, 584.0, 72.0] | [0.754985, 1014.562, 144.0] |
| runtime_decode | [1.771665, 580.0, 48.375] | [2.302822, 994.562, 240.375] |
| runtime_prefill | [2.859744, 584.0, 48.0] | [4.634156, 1014.562, 114.0] |
| serving_bytes_per_token | 98304 | 196608 |
| serving_requests_8gib_1k | 85 | 42 |
| serving_weight_bytes | 809500672 | 1185233920 |
| sim_max_running | 72 | 242 |
| slice_mib_16x4352 | [81582, false] | [22272, true] |
| vllm_limits | [4702, 75232, 4702, 1, 0, 0.0, true] | [46910, 750560, 46910, 1, 0, 201326592.0, true] |
| vllm_running_1024 | 58 | 225 |
| vllm_running_8192 | 8 | 71 |

## fixture:facebook_nougat-small_hub.json (encoder-decoder)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 1 | 552 |
| conv_kv_size | 131072 | 8192 |
| conv_layers | [16, 16] | [20, 4] |
| conv_prefill_weights_mb | 6676.0 | 548.675781 |
| conv_serving_weight_bytes | 6738149376 | 520832240 |
| conv_state_bytes | 0 | 9633792 |
| decode_fit_refused | null | true |
| disagg_b_eff_4096 | 1 | 552 |
| disagg_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'decoder', 'encoder', 'is_encoder_decoder', 'model_type', 'tie_word_embeddings', 'torch_dtype', 'transfo" | null |
| disagg_transfer_bytes_640 | null | 20119552 |
| eng_None_32768_kv_mb | 701.80864 | 669.384704 |
| eng_None_32768_params | 874782720 | 810360576 |
| eng_None_32768_total_mb | 2617.468518 | 2456.200294 |
| eng_None_4096_kv_mb | 114.60608 | 199.622656 |
| eng_None_4096_params | 874782720 | 810360576 |
| eng_None_4096_total_mb | 1947.218739 | 1903.391027 |
| eng_llamacpp_32768_kv_mb | 701.80864 | 669.384704 |
| eng_llamacpp_32768_params | 874782720 | 810360576 |
| eng_llamacpp_32768_total_mb | 2617.468518 | 2456.200294 |
| eng_llamacpp_4096_kv_mb | 114.60608 | 199.622656 |
| eng_llamacpp_4096_params | 874782720 | 810360576 |
| eng_llamacpp_4096_total_mb | 1947.218739 | 1903.391027 |
| facts_kv_bytes_4096 | 2147483648 | 76742656 |
| facts_kv_bytes_per_token | 524288 | 16384 |
| facts_layers | [32, 0, 0, null] | [4, 0, 0, null] |
| facts_weight_bytes | 34884558848 | 520832240 |
| genz_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'decoder', 'encoder', 'is_encoder_decoder', 'model_type', 'tie_word_embeddings', 'torch_dtype', 'transfo" | null |
| genz_chunked_kv_mb | null | 90.375 |
| genz_decode_kv_mb | null | 73.1875 |
| genz_decode_weights_mb | null | 366.769531 |
| genz_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'decoder', 'encoder', 'is_encoder_decoder', 'model_type', 'tie_word_embeddings', 'torch_dtype', 'transfo" | null |
| genz_ffn_kinds | null | {"Dense": 4} |
| genz_kv_size | null | 8192 |
| genz_layer_kinds | null | {"MHA-global": 4} |
| genz_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'decoder', 'encoder', 'is_encoder_decoder', 'model_type', 'tie_word_embeddings', 'torch_dtype', 'transfo" | null |
| genz_prefill_kv_mb | null | 197.21875 |
| genz_prefill_weights_mb | null | 471.425781 |
| kv_cross_layers | 0 | 4 |
| kv_full_layers | 0 | 4 |
| kv_gb@32768 | 0.0 | 0.546504704 |
| kv_gb@4096 | 0.0 | 0.076742656 |
| kvgeo_32768 | [0.0, 1, 0, 0.0, 16, 0.0] | [16384.0, 4, 0, 0.0, 16, 9633792.0] |
| kvgeo_4096 | [0.0, 1, 0, 0.0, 16, 0.0] | [16384.0, 4, 0, 0.0, 16, 9633792.0] |
| prefix_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'decoder', 'encoder', 'is_encoder_decoder', 'model_type', 'tie_w" | null |
| prefix_kv_bytes_per_token | null | 16384 |
| report_kv_gb | 0.11460608 | 0.199622656 |
| report_params | 874782720 | 810360576 |
| report_total_gb | 1.989206999 | 1.961921802 |
| report_weight_gb | 1.74956544 | 1.620721152 |
| runtime_chunked | null | [0.30712, 471.426, 30.375] |
| runtime_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'decoder', 'encoder', 'is_encoder_decoder', 'model_type', 'tie_w" | null |
| runtime_decode | null | [1.430026, 366.77, 44.812] |
| runtime_decode_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'decoder', 'encoder', 'is_encoder_decoder', 'model_type', 'tie_w" | null |
| runtime_prefill | null | [9.549937, 471.426, 141.219] |
| runtime_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'decoder', 'encoder', 'is_encoder_decoder', 'model_type', 'tie_w" | null |
| serving_bytes_per_token | null | 16384 |
| serving_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'decoder', 'encoder', 'is_encoder_decoder', 'model_type', 'tie_word_embeddings', 'torch_dtype', 'transfo" | null |
| serving_requests_8gib_1k | null | 325 |
| serving_state_bytes | null | 9633792 |
| serving_weight_bytes | null | 520832240 |
| serving_weight_bytes_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'decoder', 'encoder', 'is_encoder_decoder', 'model_type', 'tie_word_embeddings', 'torch_dtype', 'transfo" | null |
| sim_max_running | 72 | 400 |
| slice_mib_16x4352 | [81582, false] | [7680, true] |
| vllm_limits | [4702, 75232, 4702, 1, 0, 0.0, true] | [283434, 4534944, 283434, 1, 0, 9633792.0, true] |
| vllm_running_1024 | 58 | 1024 |
| vllm_running_8192 | 8 | 501 |

## fixture:google_deplot_hub.json (multimodal)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 1 | 139 |
| conv_kv_size | 262144 | 18432 |
| conv_layers | [0, 32] | [12, 12] |
| conv_prefill_weights_mb | 12852.0 | 538.325684 |
| conv_serving_weight_bytes | 13214154752 | 724623360 |
| conv_state_bytes | 0 | 150994944 |
| decode_fit_refused | null | true |
| disagg_b_eff_4096 | 1 | 139 |
| disagg_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_commit_hash', '_name_or_path', 'architectures', 'decoder_start_token_id', 'initializer_factor', 'initializer_range', 'i" | null |
| disagg_transfer_bytes_640 | null | 174587904 |
| eng_None_32768_kv_mb | 0.0 | 1358.954496 |
| eng_None_32768_params | 123532032 | 282385152 |
| eng_None_32768_total_mb | 375.422054 | 2052.08279 |
| eng_None_4096_kv_mb | 0.0 | 301.989888 |
| eng_None_4096_params | 123532032 | 282385152 |
| eng_None_4096_total_mb | 312.507494 | 932.203622 |
| eng_llamacpp_32768_kv_mb | 0.0 | 1358.954496 |
| eng_llamacpp_32768_params | 123532032 | 282385152 |
| eng_llamacpp_32768_total_mb | 375.422054 | 2052.08279 |
| eng_llamacpp_4096_kv_mb | 0.0 | 301.989888 |
| eng_llamacpp_4096_params | 123532032 | 282385152 |
| eng_llamacpp_4096_total_mb | 312.507494 | 932.203622 |
| facts_kv_bytes_4096 | 150994944 | 301989888 |
| facts_weight_bytes | 554535936 | 724623360 |
| genz_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_commit_hash', '_name_or_path', 'architectures', 'decoder_start_token_id', 'initializer_factor', 'initializer_range', 'i" | null |
| genz_chunked_kv_mb | null | 450.0 |
| genz_decode_kv_mb | null | 288.0 |
| genz_decode_weights_mb | null | 464.726074 |
| genz_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_commit_hash', '_name_or_path', 'architectures', 'decoder_start_token_id', 'initializer_factor', 'initializer_range', 'i" | null |
| genz_ffn_kinds | null | {"Dense": 12} |
| genz_kv_size | null | 18432 |
| genz_layer_kinds | null | {"MHA-global": 12} |
| genz_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_commit_hash', '_name_or_path', 'architectures', 'decoder_start_token_id', 'initializer_factor', 'initializer_range', 'i" | null |
| genz_prefill_kv_mb | null | 432.0 |
| genz_prefill_weights_mb | null | 538.325684 |
| kv_cross_layers | 0 | 12 |
| kv_gb@32768 | 0.0 | 1.358954496 |
| kv_gb@4096 | 0.0 | 0.301989888 |
| kvgeo_32768 | [0.0, 1, 0, 0.0, 16, 0.0] | [36864.0, 12, 0, 0.0, 16, 150994944.0] |
| kvgeo_4096 | [0.0, 1, 0, 0.0, 16, 0.0] | [36864.0, 12, 0, 0.0, 16, 150994944.0] |
| prefix_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_commit_hash', '_name_or_path', 'architectures', 'decoder_start_token_id', 'init" | null |
| prefix_kv_bytes_per_token | null | 36864 |
| report_kv_gb | 0.0 | 0.301989888 |
| report_params | 123532032 | 282385152 |
| report_total_gb | 0.32559618 | 1.005690286 |
| report_weight_gb | 0.247064064 | 0.564770304 |
| runtime_chunked | null | [0.546329, 538.326, 315.0] |
| runtime_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_commit_hash', '_name_or_path', 'architectures', 'decoder_start_token_id', 'init" | null |
| runtime_decode | null | [1.936737, 464.726, 594.141] |
| runtime_decode_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_commit_hash', '_name_or_path', 'architectures', 'decoder_start_token_id', 'init" | null |
| runtime_prefill | null | [7.693502, 538.326, 306.0] |
| runtime_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_commit_hash', '_name_or_path', 'architectures', 'decoder_start_token_id', 'init" | null |
| serving_bytes_per_token | null | 36864 |
| serving_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_commit_hash', '_name_or_path', 'architectures', 'decoder_start_token_id', 'initializer_factor', 'initializer_range', 'i" | null |
| serving_requests_8gib_1k | null | 45 |
| serving_state_bytes | null | 150994944 |
| serving_weight_bytes | null | 724623360 |
| serving_weight_bytes_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_commit_hash', '_name_or_path', 'architectures', 'decoder_start_token_id', 'initializer_factor', 'initializer_range', 'i" | null |
| sim_max_running | 400 | 390 |
| slice_mib_16x4352 | [8960, true] | [12032, true] |
| vllm_limits | [125989, 2015824, 125989, 1, 0, 0.0, true] | [125701, 2011216, 125701, 1, 0, 150994944.0, true] |
| vllm_running_1024 | 1024 | 374 |
| vllm_running_8192 | 238 | 160 |

## fixture:gpt-oss-20b-bf16.json (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 195 | 5 |
| conv_serving_weight_bytes | 3626680320 | 41829514368 |
| disagg_b_eff_4096 | 195 | 5 |
| genz_chunked_kv_mb | 216.0 | 123.0 |
| runtime_chunked | [25.702025, 39878.438, 36.0] | [25.696455, 39878.438, 27.0] |
| serving_weight_bytes | 3626680320 | 41829514368 |
| slice_mib_16x4352 | [49152, true] | [49408, true] |
| vllm_limits | [85423, 1366768, 85423, 1, 0, 0.0, true] | [85423, 1366768, 85423, 1, 1, 0.0, true] |
| vllm_running_1024 | 1024 | 959 |
| vllm_running_8192 | 161 | 159 |

## fixture:gpt-oss-20b.json (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 195 | 5 |
| conv_serving_weight_bytes | 3626680320 | 41829514368 |
| disagg_b_eff_4096 | 195 | 5 |
| genz_chunked_kv_mb | 216.0 | 123.0 |
| runtime_chunked | [9.364153, 13110.469, 36.0] | [9.358584, 13110.469, 27.0] |
| serving_weight_bytes | 3626680320 | 41829514368 |
| slice_mib_16x4352 | [20736, true] | [20992, true] |
| vllm_limits | [155360, 2485760, 155360, 1, 0, 0.0, true] | [155360, 2485760, 155360, 1, 1, 0.0, true] |
| vllm_running_8192 | 294 | 289 |

## fixture:ibm-ai-platform_Bamba-9B-v2.json (hybrid)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 52 | 134 |
| conv_kv_size | 65536 | 6144 |
| conv_prefill_weights_mb | 15316.0 | 18653.945312 |
| conv_serving_weight_bytes | 15009316864 | 19560470784 |
| conv_state_bytes | 0 | 123104768 |
| disagg_b_eff_4096 | 52 | 134 |
| disagg_transfer_bytes_640 | 83886080 | 130969088 |
| eng_None_32768_state_mb | 0.0 | 123.104768 |
| eng_None_32768_total_mb | 20768.962816 | 20892.067584 |
| eng_None_4096_state_mb | 0.0 | 123.104768 |
| eng_None_4096_total_mb | 20013.988096 | 20137.092864 |
| eng_llamacpp_32768_state_mb | 0.0 | 123.104768 |
| eng_llamacpp_32768_total_mb | 20768.962816 | 20892.067584 |
| eng_llamacpp_4096_state_mb | 0.0 | 123.104768 |
| eng_llamacpp_4096_total_mb | 20013.988096 | 20137.092864 |
| facts_weight_bytes | 19559980800 | 19560470784 |
| genz_chunked_kv_mb | 576.0 | 288.803711 |
| genz_decode_kv_mb | 66.0 | 165.401855 |
| genz_decode_weights_mb | 18282.0 | 17651.945312 |
| genz_kv_size | 65536 | 6144 |
| genz_layer_kinds | {"Mamba": 32} | {"MHA-global": 3, "Recurrent": 29} |
| genz_prefill_kv_mb | 66.0 | 165.401855 |
| genz_prefill_weights_mb | 19284.0 | 18653.945312 |
| prefix_kv_bytes_per_token | 131072 | 12288 |
| runtime_chunked | [13.908805, 15316.0, 96.0] | [17.216502, 18653.945, 243.804] |
| runtime_decode | [13.494652, 18282.0, 258.0] | [12.407145, 17651.945, 475.654] |
| runtime_prefill | [92.01464, 19284.0, 66.0] | [35.915163, 18653.945, 123.402] |
| serving_bytes_per_token | 131072 | 12288 |
| serving_requests_8gib_1k | 64 | 63 |
| serving_state_bytes | 0 | 123104768 |
| serving_weight_bytes | 15009316864 | 19560470784 |

## fixture:ibm-granite_granite-4.0-h-small.json (hybrid)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 56 | 1 |
| conv_kv_size | 81920 | 8192 |
| conv_prefill_weights_mb | 56630.5 | 62214.15625 |
| conv_serving_weight_bytes | 4932501504 | 64414681728 |
| conv_state_bytes | 0 | 152819712 |
| disagg_b_eff_4096 | 56 | 1 |
| disagg_transfer_bytes_640 | 104857600 | 163305472 |
| eng_None_32768_params | 31452695040 | 32207669760 |
| eng_None_32768_state_mb | 0.0 | 152.819712 |
| eng_None_32768_total_mb | 63985.84279 | 65648.611942 |
| eng_None_4096_params | 31452695040 | 32207669760 |
| eng_None_4096_state_mb | 0.0 | 152.819712 |
| eng_None_4096_total_mb | 63244.289843 | 64907.058995 |
| eng_llamacpp_32768_params | 31452695040 | 32207669760 |
| eng_llamacpp_32768_state_mb | 0.0 | 152.819712 |
| eng_llamacpp_32768_total_mb | 63985.84279 | 65648.611942 |
| eng_llamacpp_4096_params | 31452695040 | 32207669760 |
| eng_llamacpp_4096_state_mb | 0.0 | 152.819712 |
| eng_llamacpp_4096_total_mb | 63244.289843 | 64907.058995 |
| facts_weight_bytes | 62904124032 | 64414681728 |
| genz_chunked_kv_mb | 720.0 | 363.480469 |
| genz_decode_kv_mb | 82.5 | 209.740234 |
| genz_decode_weights_mb | 60806.5 | 61430.15625 |
| genz_kv_size | 81920 | 8192 |
| genz_layer_kinds | {"Mamba": 40} | {"MHA-global": 4, "Recurrent": 36} |
| genz_prefill_kv_mb | 82.5 | 209.740234 |
| genz_prefill_weights_mb | 61590.5 | 62214.15625 |
| prefix_kv_bytes_per_token | 163840 | 16384 |
| report_params | 31452695040 | 32207669760 |
| report_total_gb | 63.49545345 | 65.00540289 |
| report_weight_gb | 62.90539008 | 64.41533952 |
| runtime_chunked | [36.461028, 56630.5, 120.0] | [41.860657, 62214.156, 303.48] |
| runtime_decode | [22.885602, 60806.5, 322.5] | [22.547177, 61430.156, 591.023] |
| runtime_prefill | [121.566429, 61590.5, 82.5] | [54.365, 62214.156, 153.74] |
| serving_bytes_per_token | 163840 | 16384 |
| serving_requests_8gib_1k | 51 | 50 |
| serving_state_bytes | 0 | 152819712 |
| serving_weight_bytes | 4932501504 | 64414681728 |
| sim_max_running | 68 | 60 |
| vllm_limits | [689, 716560, 689, 1, 0, 153354240.0, false] | [601, 625040, 601, 1, 0, 153354240.0, false] |
| vllm_running_1024 | 62 | 54 |
| vllm_running_8192 | 38 | 33 |

## fixture:inclusionAI_Ling-flash-2.0.json (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 119 | 1 |
| conv_prefill_weights_mb | 201432.0 | 196246.0 |
| conv_serving_weight_bytes | 10951327744 | 205779410432 |
| disagg_b_eff_4096 | 119 | 1 |
| eng_None_32768_params | 102499885056 | 102889955328 |
| eng_None_32768_total_mb | 207710.968218 | 208491.108762 |
| eng_None_4096_params | 102499885056 | 102889955328 |
| eng_None_4096_total_mb | 205550.062797 | 206330.203341 |
| eng_llamacpp_32768_params | 102499885056 | 102889955328 |
| eng_llamacpp_32768_total_mb | 207710.968218 | 208491.108762 |
| eng_llamacpp_4096_params | 102499885056 | 102889955328 |
| eng_llamacpp_4096_total_mb | 205550.062797 | 206330.203341 |
| facts_weight_bytes | 205779394048 | 205779410432 |
| genz_decode_weights_mb | 200204.0 | 195018.0 |
| genz_ffn_kinds | {"MoE": 32} | {"Dense": 1, "MoE": 31} |
| genz_prefill_weights_mb | 201432.0 | 196246.0 |
| report_params | 102499885056 | 102889955328 |
| report_total_gb | 205.660121334 | 206.440261878 |
| report_weight_gb | 204.999770112 | 205.779910656 |
| runtime_chunked | null | [121.816444, 196246.0, 48.0] |
| runtime_chunked_error | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 203.4609375 GB, KV Cache:0.046875. \n System:{'Flops': 400, 'Memory_size': 200, 'Memory_" | null |
| runtime_decode | [18.456694, 200204.0, 32.25] | [18.705434, 195018.0, 32.25] |
| runtime_prefill | [131.489281, 201432.0, 32.0] | [129.743652, 196246.0, 32.0] |
| serving_weight_bytes | 10951327744 | 205779410432 |

## fixture:kyutai_helium-1-2b.json (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 83 | 82 |
| conv_serving_weight_bytes | 3785359360 | 4047736832 |
| disagg_b_eff_4096 | 83 | 82 |
| serving_weight_bytes | 3785359360 | 4047736832 |

## fixture:llama-3.1-8b.json (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 52 | 50 |
| conv_serving_weight_bytes | 15009316864 | 16060522496 |
| disagg_b_eff_4096 | 52 | 50 |
| serving_weight_bytes | 15009316864 | 16060522496 |

## fixture:measured_checkpoints/openai_gpt-oss-20b/config.json (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 195 | 5 |
| conv_serving_weight_bytes | 3626680320 | 41829514368 |
| disagg_b_eff_4096 | 195 | 5 |
| genz_chunked_kv_mb | 216.0 | 123.0 |
| runtime_chunked | [9.364153, 13110.469, 36.0] | [9.358584, 13110.469, 27.0] |
| serving_weight_bytes | 3626680320 | 41829514368 |
| slice_mib_16x4352 | [20736, true] | [20992, true] |
| vllm_limits | [155360, 2485760, 155360, 1, 0, 0.0, true] | [155360, 2485760, 155360, 1, 1, 0.0, true] |
| vllm_running_8192 | 294 | 289 |

## fixture:microsoft_Phi-4-mini-flash-reasoning.json (hybrid)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 53 | 102 |
| conv_kv_size | 81920 | 40960 |
| conv_prefill_weights_mb | 7953.75 | 8612.34375 |
| conv_serving_weight_bytes | 7315783680 | 8006702080 |
| conv_state_bytes | 0 | 5734400 |
| disagg_b_eff_4096 | 53 | 102 |
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
| facts_kv_bytes_4096 | 83886080 | 60293120 |
| facts_kv_bytes_per_token | 163840 | 81920 |
| facts_layers | [0, 32, 0, 512] | [1, 15, 16, 512] |
| facts_state_bytes | 0 | 5734400 |
| facts_weight_bytes | 7316116480 | 8006702080 |
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
| runtime_chunked | [7.285243, 7953.75, 120.0] | [7.909247, 8612.344, 70.938] |
| runtime_decode | [5.826644, 6976.875, 80.625] | [6.118478, 7635.469, 62.188] |
| runtime_prefill | [16.43708, 7953.75, 80.0] | [17.617703, 8612.344, 45.469] |
| serving_bytes_per_token | 163840 | 81920 |
| serving_requests_8gib_1k | 51 | 95 |
| serving_state_bytes | 0 | 5734400 |
| serving_weight_bytes | 7315783680 | 8006702080 |
| sim_max_running | 389 | 400 |
| slice_mib_16x4352 | [24832, true] | [19456, true] |
| state_gb | 0.0 | 0.0057344 |
| vllm_limits | [25739, 411824, 25739, 1, 0, 0.0, true] | [10133, 810640, 10133, 1, 0, 6553600.0, true] |
| vllm_running_1024 | 321 | 596 |
| vllm_running_8192 | 48 | 94 |

## fixture:mistralai_Mamba-Codestral-7B-v0.1.json (state-space)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 11 | 104 |
| conv_kv_size | 262144 | 0 |
| conv_serving_weight_bytes | 17448304640 | 14571331584 |
| conv_state_bytes | 0 | 272367616 |
| disagg_b_eff_4096 | 11 | 104 |
| disagg_transfer_bytes_640 | 335544320 | 272367616 |
| eng_None_32768_state_mb | 0.0 | 272.367616 |
| eng_None_32768_total_mb | 15136.10281 | 15408.470426 |
| eng_None_4096_state_mb | 0.0 | 272.367616 |
| eng_None_4096_total_mb | 14854.245581 | 15126.613197 |
| eng_llamacpp_32768_state_mb | 0.0 | 272.367616 |
| eng_llamacpp_32768_total_mb | 15136.10281 | 15408.470426 |
| eng_llamacpp_4096_state_mb | 0.0 | 272.367616 |
| eng_llamacpp_4096_total_mb | 14854.245581 | 15126.613197 |
| facts_weight_bytes | 14570020864 | 14571331584 |
| genz_kv_size | 262144 | 0 |
| prefix_kv_bytes_per_token | 524288 | 0 |
| serving_bytes_per_token | 524288 | 0 |
| serving_requests_8gib_1k | 16 | 31 |
| serving_state_bytes | 0 | 272367616 |
| serving_weight_bytes | 17448304640 | 14571331584 |

## fixture:mllama_transformers-5.12-default.json (multimodal)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 1 | 32 |
| conv_kv_size | 262144 | 65536 |
| conv_layers | [0, 32] | [0, 40] |
| conv_prefill_weights_mb | 12852.0 | 18644.0 |
| conv_serving_weight_bytes | 13214154752 | 21286212608 |
| conv_state_bytes | 0 | 134348800 |
| decode_fit_refused | null | true |
| disagg_b_eff_4096 | 1 | 32 |
| disagg_error | "TypeError: 'NoneType' object is not iterable" | null |
| disagg_transfer_bytes_640 | null | 218234880 |
| eng_None_32768_kv_mb | 5368.70912 | 4429.316096 |
| eng_None_32768_params | 123532032 | 10943653376 |
| eng_None_32768_total_mb | 6443.099648 | 27143.949312 |
| eng_None_4096_kv_mb | 671.08864 | 671.219712 |
| eng_None_4096_params | 123532032 | 10943653376 |
| eng_None_4096_total_mb | 1342.825984 | 22983.199744 |
| eng_llamacpp_32768_kv_mb | 5368.70912 | 4429.316096 |
| eng_llamacpp_32768_params | 123532032 | 10943653376 |
| eng_llamacpp_32768_total_mb | 6443.099648 | 27143.949312 |
| eng_llamacpp_4096_kv_mb | 671.08864 | 671.219712 |
| eng_llamacpp_4096_params | 123532032 | 10943653376 |
| eng_llamacpp_4096_total_mb | 1342.825984 | 22983.199744 |
| facts_kv_bytes_4096 | 671088640 | 671219712 |
| facts_kv_bytes_per_token | 163840 | 131072 |
| facts_layers | [40, 0, 0, null] | [32, 0, 0, null] |
| facts_weight_bytes | 20824956928 | 21286212608 |
| genz_chunked_error | "TypeError: 'NoneType' object is not iterable" | null |
| genz_chunked_kv_mb | null | 832.25 |
| genz_decode_kv_mb | null | 640.125 |
| genz_decode_weights_mb | null | 17642.0 |
| genz_error | "TypeError: 'NoneType' object is not iterable" | null |
| genz_ffn_kinds | null | {"Dense": 40} |
| genz_kv_size | null | 65536 |
| genz_layer_kinds | null | {"Cross": 8, "MHA-global": 32} |
| genz_prefill_error | "TypeError: 'NoneType' object is not iterable" | null |
| genz_prefill_kv_mb | null | 640.125 |
| genz_prefill_weights_mb | null | 18644.0 |
| kv_cross_layers | 0 | 8 |
| kv_full_layers | 40 | 32 |
| kv_gb@32768 | 5.36870912 | 4.429316096 |
| kv_gb@4096 | 0.67108864 | 0.671219712 |
| kv_source | "uniform" | "layer_plan:cross_attention_layers" |
| kvgeo_32768 | [163840.0, 40, 0, 0.0, 16, 0.0] | [131072.0, 32, 0, 0.0, 16, 134348800.0] |
| kvgeo_4096 | [163840.0, 40, 0, 0.0, 16, 0.0] | [131072.0, 32, 0, 0.0, 16, 134348800.0] |
| prefix_error | "TypeError: 'NoneType' object is not iterable" | null |
| prefix_kv_bytes_per_token | null | 131072 |
| report_kv_gb | 0.67108864 | 0.671219712 |
| report_params | 123532032 | 10943653376 |
| report_total_gb | 1.561978368 | 23.202378342 |
| report_weight_gb | 0.247064064 | 21.887306752 |
| runtime_chunked | null | [17.32036, 18644.0, 352.25] |
| runtime_chunked_error | "TypeError: 'NoneType' object is not iterable" | null |
| runtime_decode | null | [12.674338, 17642.0, 577.0] |
| runtime_decode_error | "TypeError: 'NoneType' object is not iterable" | null |
| runtime_prefill | null | [38.401893, 18644.0, 192.125] |
| runtime_prefill_error | "TypeError: 'NoneType' object is not iterable" | null |
| serving_bytes_per_token | null | 131072 |
| serving_error | "TypeError: 'NoneType' object is not iterable" | null |
| serving_requests_8gib_1k | null | 31 |
| serving_state_bytes | null | 134348800 |
| serving_weight_bytes | null | 21286212608 |
| serving_weight_bytes_error | "TypeError: 'NoneType' object is not iterable" | null |
| sim_max_running | 315 | 196 |
| slice_mib_16x4352 | [38912, true] | [39424, true] |
| vllm_limits | [20644, 330304, 20644, 1, 0, 0.0, true] | [25586, 409376, 25586, 1, 0, 134348800.0, true] |
| vllm_running_1024 | 258 | 177 |
| vllm_running_8192 | 39 | 43 |

## fixture:moonshotai_Kimi-Linear-48B-A3B-Instruct.json (hybrid)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 36 | 1 |
| conv_kv_size | 124416 | 4032 |
| conv_prefill_weights_mb | 95875.875 | 93693.880859 |
| conv_serving_weight_bytes | 5341446144 | 98245363456 |
| conv_state_bytes | 0 | 43417600 |
| decode_fit_refused | null | true |
| disagg_b_eff_4096 | 36 | 1 |
| disagg_transfer_bytes_640 | 159252480 | 48578560 |
| eng_None_32768_params | 48819169920 | 49004669568 |
| eng_None_32768_state_mb | 0.0 | 43.4176 |
| eng_None_32768_total_mb | 98400.864307 | 98815.281203 |
| eng_None_4096_params | 48819169920 | 49004669568 |
| eng_None_4096_state_mb | 0.0 | 43.4176 |
| eng_None_4096_total_mb | 97920.511642 | 98334.928538 |
| eng_llamacpp_32768_params | 48819169920 | 49004669568 |
| eng_llamacpp_32768_state_mb | 0.0 | 43.4176 |
| eng_llamacpp_32768_total_mb | 98400.864307 | 98815.281203 |
| eng_llamacpp_4096_params | 48819169920 | 49004669568 |
| eng_llamacpp_4096_state_mb | 0.0 | 43.4176 |
| eng_llamacpp_4096_total_mb | 97920.511642 | 98334.928538 |
| facts_weight_bytes | 97970012096 | 98245363456 |
| genz_chunked_kv_mb | 1093.5 | 118.25 |
| genz_decode_kv_mb | null | 72.90625 |
| genz_decode_weights_mb | null | 92973.880859 |
| genz_error | "ValueError: cannot convert float NaN to integer" | null |
| genz_ffn_kinds | {"MoE": 27} | {"Dense": 1, "MoE": 26} |
| genz_kv_size | 15552 | 4032 |
| genz_layer_kinds | {"MHA-global": 27} | {"MHA-global": 7, "Recurrent": 20} |
| genz_prefill_error | "ValueError: cannot convert float NaN to integer" | null |
| genz_prefill_kv_mb | null | 72.90625 |
| genz_prefill_weights_mb | null | 93693.880859 |
| prefix_kv_bytes_per_token | 248832 | 8064 |
| report_params | 48819169920 | 49004669568 |
| report_total_gb | 98.029047122 | 98.400046418 |
| report_weight_gb | 97.63833984 | 98.009339136 |
| runtime_chunked | [62.574435, 99156.375, 182.25] | [58.9165, 93693.881, 88.719] |
| runtime_decode | null | [10.671852, 92973.881, 169.593] |
| runtime_decode_error | "ValueError: cannot convert float NaN to integer" | null |
| runtime_prefill | null | [64.897031, 93693.881, 45.344] |
| runtime_prefill_error | "ValueError: cannot convert float NaN to integer" | null |
| serving_bytes_per_token | 248832 | 8064 |
| serving_requests_8gib_1k | 33 | 166 |
| serving_state_bytes | 0 | 43417600 |
| serving_weight_bytes | 5341446144 | 98245363456 |

## fixture:nari-labs_Dia-1.6B-0626.json (encoder-decoder)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 269 | 131 |
| conv_kv_size | 131072 | 18432 |
| conv_layers | [16, 16] | [12, 18] |
| conv_prefill_weights_mb | 6676.0 | 3072.78125 |
| conv_serving_weight_bytes | 6738149376 | 3305693184 |
| conv_state_bytes | 0 | 150994944 |
| disagg_b_eff_4096 | 269 | 131 |
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
| facts_kv_bytes_4096 | 2147483648 | 301989888 |
| facts_kv_bytes_per_token | 524288 | 36864 |
| facts_layers | [32, 0, 0, null] | [18, 0, 0, null] |
| facts_weight_bytes | 34884558848 | 3305693184 |
| genz_chunked_kv_mb | 162.0 | 450.0 |
| genz_decode_kv_mb | 144.0 | 288.0 |
| genz_decode_weights_mb | 2092.015625 | 3036.640625 |
| genz_prefill_kv_mb | 144.0 | 384.0 |
| genz_prefill_weights_mb | 2096.03125 | 3072.78125 |
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
| runtime_chunked | [1.926192, 2096.031, 27.0] | [2.459324, 3072.781, 315.0] |
| runtime_decode | [2.57908, 2092.016, 18.141] | [3.405099, 3036.641, 594.141] |
| runtime_prefill | [5.375457, 2096.031, 18.0] | [9.655511, 3072.781, 258.0] |
| serving_requests_8gib_1k | 227 | 45 |
| serving_state_bytes | 0 | 150994944 |
| serving_weight_bytes | 2193637376 | 3305693184 |
| sim_max_running | 72 | 376 |
| slice_mib_16x4352 | [81582, false] | [14848, true] |
| vllm_limits | [4702, 75232, 4702, 1, 0, 0.0, true] | [121194, 1939104, 121194, 1, 0, 150994944.0, true] |
| vllm_running_1024 | 58 | 360 |
| vllm_running_8192 | 8 | 154 |

## fixture:naver-clova-ix_donut-base_hub.json (encoder-decoder)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 1 | 291 |
| conv_kv_size | 131072 | 8192 |
| conv_layers | [16, 16] | [20, 4] |
| conv_prefill_weights_mb | 6676.0 | 574.185547 |
| conv_serving_weight_bytes | 6738149376 | 433963248 |
| conv_state_bytes | 0 | 78643200 |
| decode_fit_refused | null | true |
| disagg_b_eff_4096 | 1 | 291 |
| disagg_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'decoder', 'encoder', 'is_encoder_decoder', 'model_type', 'tie_word_embeddings', 'torch_dtype', 'transfo" | null |
| disagg_transfer_bytes_640 | null | 89128960 |
| eng_None_32768_kv_mb | 701.80864 | 738.394112 |
| eng_None_32768_params | 874782720 | 810360576 |
| eng_None_32768_total_mb | 2617.468518 | 2525.209702 |
| eng_None_4096_kv_mb | 114.60608 | 268.632064 |
| eng_None_4096_params | 874782720 | 810360576 |
| eng_None_4096_total_mb | 1947.218739 | 1972.400435 |
| eng_llamacpp_32768_kv_mb | 701.80864 | 738.394112 |
| eng_llamacpp_32768_params | 874782720 | 810360576 |
| eng_llamacpp_32768_total_mb | 2617.468518 | 2525.209702 |
| eng_llamacpp_4096_kv_mb | 114.60608 | 268.632064 |
| eng_llamacpp_4096_params | 874782720 | 810360576 |
| eng_llamacpp_4096_total_mb | 1947.218739 | 1972.400435 |
| facts_kv_bytes_4096 | 2147483648 | 145752064 |
| facts_kv_bytes_per_token | 524288 | 16384 |
| facts_layers | [32, 0, 0, null] | [4, 0, 0, null] |
| facts_weight_bytes | 34884558848 | 433963248 |
| genz_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'decoder', 'encoder', 'is_encoder_decoder', 'model_type', 'tie_word_embeddings', 'torch_dtype', 'transfo" | null |
| genz_chunked_kv_mb | null | 222.0 |
| genz_decode_kv_mb | null | 139.0 |
| genz_decode_weights_mb | null | 381.582031 |
| genz_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'decoder', 'encoder', 'is_encoder_decoder', 'model_type', 'tie_word_embeddings', 'torch_dtype', 'transfo" | null |
| genz_ffn_kinds | null | {"Dense": 4} |
| genz_kv_size | null | 8192 |
| genz_layer_kinds | null | {"MHA-global": 4} |
| genz_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'decoder', 'encoder', 'is_encoder_decoder', 'model_type', 'tie_word_embeddings', 'torch_dtype', 'transfo" | null |
| genz_prefill_kv_mb | null | 1151.5 |
| genz_prefill_weights_mb | null | 496.935547 |
| kv_cross_layers | 0 | 4 |
| kv_full_layers | 0 | 4 |
| kv_gb@32768 | 0.0 | 0.615514112 |
| kv_gb@4096 | 0.0 | 0.145752064 |
| kvgeo_32768 | [0.0, 1, 0, 0.0, 16, 0.0] | [16384.0, 4, 0, 0.0, 16, 78643200.0] |
| kvgeo_4096 | [0.0, 1, 0, 0.0, 16, 0.0] | [16384.0, 4, 0, 0.0, 16, 78643200.0] |
| prefix_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'decoder', 'encoder', 'is_encoder_decoder', 'model_type', 'tie_w" | null |
| prefix_kv_bytes_per_token | null | 16384 |
| report_kv_gb | 0.11460608 | 0.268632064 |
| report_params | 874782720 | 810360576 |
| report_total_gb | 1.989206999 | 2.044733092 |
| report_weight_gb | 1.74956544 | 1.620721152 |
| runtime_chunked | null | [0.42887, 496.936, 162.0] |
| runtime_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'decoder', 'encoder', 'is_encoder_decoder', 'model_type', 'tie_w" | null |
| runtime_decode | null | [1.599985, 381.582, 308.062] |
| runtime_decode_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'decoder', 'encoder', 'is_encoder_decoder', 'model_type', 'tie_w" | null |
| runtime_prefill | null | [410.064543, 496.936, 1095.5] |
| runtime_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'decoder', 'encoder', 'is_encoder_decoder', 'model_type', 'tie_w" | null |
| serving_bytes_per_token | null | 16384 |
| serving_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'decoder', 'encoder', 'is_encoder_decoder', 'model_type', 'tie_word_embeddings', 'torch_dtype', 'transfo" | null |
| serving_requests_8gib_1k | null | 90 |
| serving_state_bytes | null | 78643200 |
| serving_weight_bytes | null | 433963248 |
| serving_weight_bytes_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'decoder', 'encoder', 'is_encoder_decoder', 'model_type', 'tie_word_embeddings', 'torch_dtype', 'transfo" | null |
| sim_max_running | 72 | 400 |
| slice_mib_16x4352 | [81582, false] | [8704, true] |
| vllm_limits | [4702, 75232, 4702, 1, 0, 0.0, true] | [283677, 4538832, 283677, 1, 0, 78643200.0, true] |
| vllm_running_1024 | 58 | 746 |
| vllm_running_8192 | 8 | 342 |

## fixture:nvidia_Hymba-1.5B-Base.json (hybrid)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 242 | 227 |
| conv_prefill_weights_mb | 2182.818604 | 3170.123291 |
| conv_serving_weight_bytes | 2186448000 | 3221920000 |
| conv_state_bytes | 0 | 7168000 |
| disagg_b_eff_4096 | 242 | 227 |
| disagg_transfer_bytes_640 | 26214400 | 33382400 |
| eng_None_32768_state_mb | 0.0 | 7.168 |
| eng_None_32768_total_mb | 4875.103206 | 4882.271206 |
| eng_None_4096_state_mb | 0.0 | 7.168 |
| eng_None_4096_total_mb | 3545.299123 | 3552.467123 |
| eng_llamacpp_32768_state_mb | 0.0 | 7.168 |
| eng_llamacpp_32768_total_mb | 4875.103206 | 4882.271206 |
| eng_llamacpp_4096_state_mb | 0.0 | 7.168 |
| eng_llamacpp_4096_total_mb | 3545.299123 | 3552.467123 |
| facts_weight_bytes | 3179237056 | 3221920000 |
| genz_chunked_kv_mb | 180.0 | 84.921875 |
| genz_decode_kv_mb | 3.90625 | 58.085938 |
| genz_decode_weights_mb | 2692.971802 | 3072.463989 |
| genz_layer_kinds | {"Mamba": 32} | {"MHA-global": 3, "MHA-local": 29} |
| genz_prefill_kv_mb | 3.90625 | 58.085938 |
| genz_prefill_weights_mb | 2790.631104 | 3170.123291 |
| runtime_chunked | [2.022952, 2182.819, 30.0] | [3.040314, 3170.123, 43.672] |
| runtime_decode | [3.56883, 2692.972, 13.281] | [3.612223, 3072.464, 47.5] |
| runtime_prefill | [11.274933, 2790.631, 3.906] | [8.499001, 3170.123, 26.836] |
| serving_requests_8gib_1k | 204 | 174 |
| serving_state_bytes | 0 | 7168000 |
| serving_weight_bytes | 2186448000 | 3221920000 |
| vllm_limits | [9959, 1752784, 9959, 1, 0, 7208960.0, true] | [9953, 1751728, 9953, 1, 0, 7208960.0, true] |

## fixture:nvidia_NVIDIA-Nemotron-Nano-9B-v2.json (hybrid)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 12 | 102 |
| conv_kv_size | 114688 | 8192 |
| conv_prefill_weights_mb | 30629.375 | 21422.281494 |
| conv_serving_weight_bytes | 30942822400 | 21289276416 |
| conv_state_bytes | 0 | 143548416 |
| disagg_b_eff_4096 | 12 | 102 |
| disagg_transfer_bytes_640 | 146800640 | 154034176 |
| eng_None_32768_state_mb | 0.0 | 143.548416 |
| eng_None_32768_total_mb | 20370.048768 | 20513.597184 |
| eng_None_4096_state_mb | 0.0 | 143.548416 |
| eng_None_4096_total_mb | 19459.8848 | 19603.433216 |
| eng_llamacpp_32768_state_mb | 0.0 | 143.548416 |
| eng_llamacpp_32768_total_mb | 20370.048768 | 20513.597184 |
| eng_llamacpp_4096_state_mb | 0.0 | 143.548416 |
| eng_llamacpp_4096_total_mb | 19459.8848 | 19603.433216 |
| facts_weight_bytes | 20359294464 | 21289276416 |
| genz_chunked_kv_mb | 1008.0 | 345.796875 |
| genz_decode_kv_mb | 63.164062 | 200.898438 |
| genz_decode_weights_mb | 22925.0 | 15832.672119 |
| genz_ffn_kinds | {"Dense": 56} | {"Dense": 25, "no_op": 31} |
| genz_kv_size | 114688 | 8192 |
| genz_layer_kinds | {"Mamba": 56} | {"MHA-global": 4, "Recurrent": 27, "no_op": 25} |
| genz_prefill_kv_mb | 63.164062 | 200.898438 |
| genz_prefill_weights_mb | 24045.0 | 18072.672119 |
| prefix_kv_bytes_per_token | 229376 | 16384 |
| runtime_chunked | [21.081721, 23126.25, 168.0] | [16.809775, 18072.672, 285.797] |
| runtime_decode | [16.983359, 22925.0, 246.914] | [11.317602, 15832.672, 555.656] |
| runtime_prefill | [99.889553, 24045.0, 63.164] | [34.998366, 18072.672, 144.898] |
| serving_bytes_per_token | 229376 | 16384 |
| serving_requests_8gib_1k | 36 | 53 |
| serving_state_bytes | 0 | 143548416 |
| serving_weight_bytes | 30942822400 | 21289276416 |
| sim_max_running | 315 | 309 |
| slice_mib_16x4352 | [33536, true] | [34560, true] |
| vllm_limits | [2520, 3306240, 2520, 1, 0, 150470656.0, true] | [2477, 3249824, 2477, 1, 0, 150470656.0, true] |
| vllm_running_1024 | 315 | 309 |
| vllm_running_8192 | 180 | 176 |

## fixture:nvidia_Nemotron-H-8B-Base-8K.json (hybrid)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 11 | 133 |
| conv_kv_size | 106496 | 8192 |
| conv_prefill_weights_mb | 32416.0 | 19546.8125 |
| conv_serving_weight_bytes | 32916897792 | 20429989888 |
| conv_state_bytes | 0 | 102137856 |
| disagg_b_eff_4096 | 11 | 133 |
| disagg_transfer_bytes_640 | 136314880 | 112623616 |
| eng_None_32768_state_mb | 0.0 | 102.137856 |
| eng_None_32768_total_mb | 17894.134989 | 17996.272845 |
| eng_None_4096_state_mb | 0.0 | 102.137856 |
| eng_None_4096_total_mb | 16880.791142 | 16982.928998 |
| eng_llamacpp_32768_state_mb | 0.0 | 102.137856 |
| eng_llamacpp_32768_total_mb | 17894.134989 | 17996.272845 |
| eng_llamacpp_4096_state_mb | 0.0 | 102.137856 |
| eng_llamacpp_4096_total_mb | 16880.791142 | 16982.928998 |
| facts_weight_bytes | 20429498368 | 20429989888 |
| genz_chunked_kv_mb | 936.0 | 266.8125 |
| genz_decode_kv_mb | 107.25 | 161.40625 |
| genz_decode_weights_mb | 29104.0 | 14426.8125 |
| genz_ffn_kinds | {"Dense": 52} | {"Dense": 24, "no_op": 28} |
| genz_kv_size | 106496 | 8192 |
| genz_layer_kinds | {"Mamba": 52} | {"MHA-global": 4, "Recurrent": 24, "no_op": 24} |
| genz_prefill_kv_mb | 107.25 | 161.40625 |
| genz_prefill_weights_mb | 30128.0 | 15514.8125 |
| prefix_kv_bytes_per_token | 212992 | 16384 |
| runtime_chunked | [21.518349, 23680.0, 156.0] | [14.362543, 15514.812, 206.812] |
| runtime_decode | [20.933382, 29104.0, 419.25] | [10.340875, 14426.812, 397.688] |
| runtime_prefill | [146.729845, 30128.0, 107.25] | [30.130069, 15514.812, 105.406] |
| serving_bytes_per_token | 212992 | 16384 |
| serving_requests_8gib_1k | 39 | 72 |
| serving_state_bytes | 0 | 102137856 |
| serving_weight_bytes | 32916897792 | 20429989888 |

## fixture:onnx-community_cohere-transcribe-03-2026-ONNX.json (encoder-decoder)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 317 | 126 |
| conv_kv_size | 8192 | 16384 |
| conv_layers | [4, 4] | [48, 8] |
| conv_prefill_weights_mb | 192.0 | 2804.317383 |
| conv_serving_weight_bytes | 167772160 | 5283183616 |
| conv_state_bytes | 0 | 163840000 |
| disagg_b_eff_4096 | 317 | 126 |
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
| facts_kv_bytes_4096 | 134217728 | 298057728 |
| facts_weight_bytes | 604080128 | 5283183616 |
| genz_chunked_kv_mb | 144.0 | 456.5 |
| genz_decode_kv_mb | 128.0 | 284.25 |
| genz_decode_weights_mb | 224.0 | 2770.317383 |
| genz_prefill_kv_mb | 128.0 | 1456.125 |
| genz_prefill_weights_mb | 256.0 | 2804.317383 |
| kv_cross_layers | 0 | 8 |
| kv_gb@32768 | 1.073741824 | 1.237581824 |
| kv_gb@4096 | 0.134217728 | 0.298057728 |
| kvgeo_32768 | [32768.0, 8, 0, 0.0, 16, 0.0] | [32768.0, 8, 0, 0.0, 16, 163840000.0] |
| kvgeo_4096 | [32768.0, 8, 0, 0.0, 16, 0.0] | [32768.0, 8, 0, 0.0, 16, 163840000.0] |
| report_kv_gb | 0.183369728 | 0.666697728 |
| report_params | 576643072 | 580837376 |
| report_total_gb | 1.459883786 | 2.048726794 |
| report_weight_gb | 1.153286144 | 1.161674752 |
| runtime_chunked | [0.250475, 256.0, 24.0] | [0.574352, 2804.317, 336.5] |
| runtime_decode | [1.27734, 224.0, 16.125] | [2.183912, 2770.317, 641.125] |
| runtime_prefill | [1.732001, 256.0, 16.0] | [66.95409, 2804.317, 1344.125] |
| serving_requests_8gib_1k | 256 | 43 |
| serving_state_bytes | 0 | 163840000 |
| serving_weight_bytes | 301989888 | 5283183616 |
| sim_max_running | 400 | 350 |
| slice_mib_16x4352 | [8704, true] | [16640, true] |
| vllm_limits | [141755, 2268080, 141755, 1, 0, 0.0, true] | [132830, 2125280, 132830, 1, 0, 163840000.0, true] |
| vllm_running_1024 | 1024 | 338 |
| vllm_running_8192 | 268 | 158 |

## fixture:onnx-community_moonshine-base-ONNX.json (encoder-decoder)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 786 | 392 |
| conv_kv_size | 3328 | 13312 |
| conv_layers | [4, 4] | [8, 8] |
| conv_prefill_weights_mb | 73.125 | 147.0625 |
| conv_serving_weight_bytes | 49414144 | 154308544 |
| disagg_b_eff_4096 | 786 | 392 |
| disagg_transfer_bytes_640 | 8519680 | 17039360 |
| eng_None_32768_kv_mb | 456.175616 | 892.383232 |
| eng_None_32768_total_mb | 753.298125 | 1189.505741 |
| eng_None_4096_kv_mb | 74.493952 | 129.019904 |
| eng_None_4096_total_mb | 326.63255 | 381.158502 |
| eng_llamacpp_32768_kv_mb | 456.175616 | 892.383232 |
| eng_llamacpp_32768_total_mb | 753.298125 | 1189.505741 |
| eng_llamacpp_4096_kv_mb | 74.493952 | 129.019904 |
| eng_llamacpp_4096_total_mb | 326.63255 | 381.158502 |
| facts_kv_bytes_4096 | 54525952 | 109051904 |
| facts_weight_bytes | 143144768 | 154308544 |
| genz_chunked_kv_mb | 58.5 | 117.0 |
| genz_decode_kv_mb | 52.0 | 104.0 |
| genz_decode_weights_mb | 57.6875 | 121.0625 |
| genz_kv_size | 6656 | 13312 |
| genz_prefill_kv_mb | 52.0 | 156.0 |
| genz_prefill_weights_mb | 83.6875 | 147.0625 |
| kv_cross_layers | 0 | 8 |
| kv_gb@32768 | 0.436207616 | 0.872415232 |
| kv_gb@4096 | 0.054525952 | 0.109051904 |
| kvgeo_32768 | [13312.0, 8, 0, 0.0, 16, 0.0] | [26624.0, 8, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [13312.0, 8, 0, 0.0, 16, 0.0] | [26624.0, 8, 0, 0.0, 16, 0.0] |
| prefix_kv_bytes_per_token | 13312 | 26624 |
| report_kv_gb | 0.074493952 | 0.129019904 |
| report_total_gb | 0.351430523 | 0.416861665 |
| runtime_chunked | [0.092924, 83.688, 9.75] | [0.117825, 147.062, 19.5] |
| runtime_decode | [1.169651, 57.688, 6.551] | [1.340291, 121.062, 32.551] |
| runtime_prefill | [1.412893, 83.688, 6.5] | [1.992774, 147.062, 19.5] |
| serving_bytes_per_token | 13312 | 26624 |
| serving_requests_8gib_1k | 630 | 315 |
| serving_weight_bytes | 71565312 | 154308544 |
| slice_mib_16x4352 | [6656, true] | [7936, true] |
| vllm_limits | [351373, 5621968, 351373, 1, 0, 0.0, true] | [351321, 5621136, 351321, 1, 0, 6815744.0, true] |
| vllm_running_8192 | 665 | 627 |

## fixture:onnx-community_whisper-large-v3-turbo.json (encoder-decoder)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 1 | 359 |
| conv_kv_size | 32768 | 10240 |
| conv_prefill_weights_mb | 2354.40625 | 2118.325195 |
| conv_serving_weight_bytes | 2043887616 | 1787082240 |
| conv_state_bytes | 0 | 30720000 |
| decode_fit_refused | null | true |
| disagg_b_eff_4096 | 1 | 359 |
| disagg_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'apply_spec_augment', 'architectures', 'attention_dropout'," | null |
| disagg_transfer_bytes_640 | null | 43827200 |
| facts_kv_bytes_4096 | 671088640 | 114606080 |
| facts_kv_bytes_per_token | 163840 | 20480 |
| facts_layers | [32, 0, 0, null] | [4, 0, 0, null] |
| facts_weight_bytes | 3621163520 | 1787082240 |
| genz_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'apply_spec_augment', 'architectures', 'attention_dropout'," | null |
| genz_chunked_kv_mb | null | 148.59375 |
| genz_decode_kv_mb | null | 109.296875 |
| genz_decode_weights_mb | null | 1540.605469 |
| genz_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'apply_spec_augment', 'architectures', 'attention_dropout'," | null |
| genz_ffn_kinds | null | {"Dense": 4} |
| genz_kv_size | null | 10240 |
| genz_layer_kinds | null | {"MHA-global": 4} |
| genz_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'apply_spec_augment', 'architectures', 'attention_dropout'," | null |
| genz_prefill_kv_mb | null | 343.671875 |
| genz_prefill_weights_mb | null | 1668.325195 |
| kv_cross_layers | 0 | 4 |
| kv_full_layers | 32 | 4 |
| kv_gb@32768 | 0.0 | 0.70180864 |
| kv_gb@4096 | 0.0 | 0.11460608 |
| kvgeo_32768 | [0.0, 32, 0, 0.0, 16, 0.0] | [20480.0, 4, 0, 0.0, 16, 30720000.0] |
| kvgeo_4096 | [0.0, 32, 0, 0.0, 16, 0.0] | [20480.0, 4, 0, 0.0, 16, 30720000.0] |
| prefix_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'apply_spec_augment" | null |
| prefix_kv_bytes_per_token | null | 20480 |
| runtime_chunked | null | [0.44293, 1668.325, 73.594] |
| runtime_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'apply_spec_augment" | null |
| runtime_decode | null | [1.626689, 1540.605, 127.266] |
| runtime_decode_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'apply_spec_augment" | null |
| runtime_prefill | null | [10.684733, 1668.325, 273.672] |
| runtime_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'apply_spec_augment" | null |
| serving_bytes_per_token | null | 20480 |
| serving_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'apply_spec_augment', 'architectures', 'attention_dropout'," | null |
| serving_requests_8gib_1k | null | 166 |
| serving_state_bytes | null | 30720000 |
| serving_weight_bytes | null | 1787082240 |
| serving_weight_bytes_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'activation_dropout', 'activation_function', 'apply_spec_augment', 'architectures', 'attention_dropout'," | null |
| slice_mib_16x4352 | [22272, true] | [9472, true] |
| vllm_limits | [27344, 437504, 27344, 1, 0, 0.0, true] | [224350, 3589600, 224350, 1, 0, 30720000.0, true] |
| vllm_running_1024 | 341 | 1024 |
| vllm_running_8192 | 51 | 360 |

## fixture:openai_gpt-oss-120b.json (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 126 | 1 |
| conv_serving_weight_bytes | 4860887040 | 233658313344 |
| disagg_b_eff_4096 | 126 | 1 |
| genz_chunked_kv_mb | 324.0 | 184.5 |
| runtime_chunked | [39.655232, 62149.219, 54.0] | [39.646878, 62149.219, 40.5] |
| serving_weight_bytes | 4860887040 | 233658313344 |
| sim_max_running | 250 | 220 |
| vllm_limits | [16363, 261808, 16363, 1, 0, 0.0, true] | [16363, 261808, 16363, 1, 1, 0.0, true] |
| vllm_running_1024 | 204 | 183 |

## fixture:pix2struct_transformers-5.12-default.json (multimodal)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 1 | 139 |
| conv_kv_size | 131072 | 18432 |
| conv_layers | [16, 16] | [12, 12] |
| conv_prefill_weights_mb | 6676.0 | 538.325684 |
| conv_serving_weight_bytes | 6738149376 | 724346880 |
| conv_state_bytes | 0 | 150994944 |
| decode_fit_refused | null | true |
| disagg_b_eff_4096 | 1 | 139 |
| disagg_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'architectures', 'chunk_size_feed_forward', 'decoder_start_token_id', 'dtype', 'eos_token_id', 'id2label" | null |
| disagg_transfer_bytes_640 | null | 174587904 |
| eng_None_32768_kv_mb | 0.0 | 1358.954496 |
| eng_None_32768_params | 123532032 | 282385152 |
| eng_None_32768_total_mb | 375.422054 | 2052.08279 |
| eng_None_4096_kv_mb | 0.0 | 301.989888 |
| eng_None_4096_params | 123532032 | 282385152 |
| eng_None_4096_total_mb | 312.507494 | 932.203622 |
| eng_llamacpp_32768_kv_mb | 0.0 | 1358.954496 |
| eng_llamacpp_32768_params | 123532032 | 282385152 |
| eng_llamacpp_32768_total_mb | 375.422054 | 2052.08279 |
| eng_llamacpp_4096_kv_mb | 0.0 | 301.989888 |
| eng_llamacpp_4096_params | 123532032 | 282385152 |
| eng_llamacpp_4096_total_mb | 312.507494 | 932.203622 |
| facts_kv_bytes_4096 | 150994944 | 301989888 |
| facts_weight_bytes | 780751872 | 724346880 |
| genz_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'architectures', 'chunk_size_feed_forward', 'decoder_start_token_id', 'dtype', 'eos_token_id', 'id2label" | null |
| genz_chunked_kv_mb | null | 450.0 |
| genz_decode_kv_mb | null | 288.0 |
| genz_decode_weights_mb | null | 464.726074 |
| genz_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'architectures', 'chunk_size_feed_forward', 'decoder_start_token_id', 'dtype', 'eos_token_id', 'id2label" | null |
| genz_ffn_kinds | null | {"Dense": 12} |
| genz_kv_size | null | 18432 |
| genz_layer_kinds | null | {"MHA-global": 12} |
| genz_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'architectures', 'chunk_size_feed_forward', 'decoder_start_token_id', 'dtype', 'eos_token_id', 'id2label" | null |
| genz_prefill_kv_mb | null | 432.0 |
| genz_prefill_weights_mb | null | 538.325684 |
| kv_cross_layers | 0 | 12 |
| kv_gb@32768 | 0.0 | 1.358954496 |
| kv_gb@4096 | 0.0 | 0.301989888 |
| kvgeo_32768 | [0.0, 1, 0, 0.0, 16, 0.0] | [36864.0, 12, 0, 0.0, 16, 150994944.0] |
| kvgeo_4096 | [0.0, 1, 0, 0.0, 16, 0.0] | [36864.0, 12, 0, 0.0, 16, 150994944.0] |
| prefix_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'architectures', 'chunk_size_feed_forward', 'decoder_start_token" | null |
| prefix_kv_bytes_per_token | null | 36864 |
| report_kv_gb | 0.0 | 0.301989888 |
| report_params | 123532032 | 282385152 |
| report_total_gb | 0.32559618 | 1.005690286 |
| report_weight_gb | 0.247064064 | 0.564770304 |
| runtime_chunked | null | [0.546329, 538.326, 315.0] |
| runtime_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'architectures', 'chunk_size_feed_forward', 'decoder_start_token" | null |
| runtime_decode | null | [1.936737, 464.726, 594.141] |
| runtime_decode_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'architectures', 'chunk_size_feed_forward', 'decoder_start_token" | null |
| runtime_prefill | null | [7.693502, 538.326, 306.0] |
| runtime_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'architectures', 'chunk_size_feed_forward', 'decoder_start_token" | null |
| serving_bytes_per_token | null | 36864 |
| serving_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'architectures', 'chunk_size_feed_forward', 'decoder_start_token_id', 'dtype', 'eos_token_id', 'id2label" | null |
| serving_requests_8gib_1k | null | 45 |
| serving_state_bytes | null | 150994944 |
| serving_weight_bytes | null | 724346880 |
| serving_weight_bytes_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'architectures', 'chunk_size_feed_forward', 'decoder_start_token_id', 'dtype', 'eos_token_id', 'id2label" | null |
| sim_max_running | 400 | 390 |
| slice_mib_16x4352 | [9216, true] | [12032, true] |
| vllm_limits | [125606, 2009696, 125606, 1, 0, 0.0, true] | [125702, 2011232, 125702, 1, 0, 150994944.0, true] |
| vllm_running_1024 | 1024 | 374 |
| vllm_running_8192 | 237 | 160 |

## fixture:qwen1.5-moe-a2.7b.json (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 49 | 17 |
| conv_prefill_weights_mb | 25720.625 | 27304.625 |
| conv_serving_weight_bytes | 3088580608 | 28631178048 |
| disagg_b_eff_4096 | 49 | 17 |
| eng_None_32768_params | 13796380672 | 14626852864 |
| eng_None_32768_total_mb | 34377.467494 | 36038.411878 |
| eng_None_4096_params | 13796380672 | 14626852864 |
| eng_None_4096_total_mb | 28569.195315 | 30230.139699 |
| eng_llamacpp_32768_params | 13796380672 | 14626852864 |
| eng_llamacpp_32768_total_mb | 34377.467494 | 36038.411878 |
| eng_llamacpp_4096_params | 13796380672 | 14626852864 |
| eng_llamacpp_4096_total_mb | 28569.195315 | 30230.139699 |
| genz_decode_weights_mb | 25127.125 | 26711.125 |
| genz_prefill_weights_mb | 25720.625 | 27304.625 |
| report_params | 13796380672 | 14626852864 |
| report_total_gb | 28.764482109 | 30.425426493 |
| report_weight_gb | 27.592761344 | 29.253705728 |
| runtime_chunked | [16.520671, 25720.625, 144.0] | [17.943546, 27304.625, 144.0] |
| runtime_decode | [6.05469, 25127.125, 96.75] | [7.117747, 26711.125, 96.75] |
| runtime_prefill | [20.272026, 25720.625, 96.0] | [23.346704, 27304.625, 96.0] |
| serving_weight_bytes | 3088580608 | 28631178048 |

## fixture:qwen2_5_0_5b_config.json (decoder-only)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 987922432 | 988010240 |
| serving_weight_bytes | 987922432 | 988010240 |

## fixture:qwen3-30b-a3b-fp8.json (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 91 | 1 |
| conv_serving_weight_bytes | 6058147840 | 61064257536 |
| disagg_b_eff_4096 | 91 | 1 |
| serving_weight_bytes | 6058147840 | 61064257536 |

## fixture:qwen3-8b-fp8.json (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 46 | 43 |
| conv_serving_weight_bytes | 15136194560 | 16381470720 |
| disagg_b_eff_4096 | 46 | 43 |
| serving_weight_bytes | 15136194560 | 16381470720 |

## fixture:qwen3-8b.json (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 46 | 43 |
| conv_serving_weight_bytes | 15136194560 | 16381470720 |
| disagg_b_eff_4096 | 46 | 43 |
| serving_weight_bytes | 15136194560 | 16381470720 |

## fixture:qwen3.8-27b-fp8.json (multimodal)

| figure | before | after |
|---|---|---|
| conv_kv_size | 262144 | 32768 |
| conv_layers | [0, 32] | [0, 64] |
| conv_prefill_weights_mb | 12852.0 | 51299.21875 |
| conv_serving_weight_bytes | 13214154752 | 54713457120 |
| conv_state_bytes | 0 | 153944064 |
| disagg_transfer_bytes_640 | 167772160 | 195887104 |
| eng_None_32768_params | 27919350016 | 28422666496 |
| eng_None_32768_state_mb | 0.0 | 153.944064 |
| eng_None_32768_total_mb | 34685.072013 | 35342.332557 |
| eng_None_4096_params | 27919350016 | 28422666496 |
| eng_None_4096_state_mb | 0.0 | 153.944064 |
| eng_None_4096_total_mb | 32312.77367 | 32970.034214 |
| eng_llamacpp_32768_params | 27919350016 | 28422666496 |
| eng_llamacpp_32768_state_mb | 0.0 | 153.944064 |
| eng_llamacpp_32768_total_mb | 34685.072013 | 35342.332557 |
| eng_llamacpp_4096_params | 27919350016 | 28422666496 |
| eng_llamacpp_4096_state_mb | 0.0 | 153.944064 |
| eng_llamacpp_4096_total_mb | 32312.77367 | 32970.034214 |
| genz_chunked_kv_mb | 1152.0 | 581.625 |
| genz_decode_kv_mb | 1024.0 | 402.8125 |
| genz_decode_weights_mb | 44025.0 | 48874.21875 |
| genz_kv_size | 131072 | 32768 |
| genz_layer_kinds | {"MHA-global": 64} | {"MHA-global": 16, "Recurrent": 48} |
| genz_prefill_kv_mb | 1024.0 | 402.8125 |
| genz_prefill_weights_mb | 46450.0 | 51299.21875 |
| prefix_kv_bytes_per_token | 262144 | 65536 |
| report_params | 27919350016 | 28422666496 |
| report_total_gb | 32.650026184 | 33.153342664 |
| report_weight_gb | 31.550175488 | 32.053491968 |
| runtime_chunked | [42.094544, 30130.0, 192.0] | [46.883662, 34979.219, 341.625] |
| runtime_decode | [19.024197, 27705.0, 129.0] | [21.992294, 32554.219, 619.5] |
| runtime_prefill | [86.74405, 30130.0, 128.0] | [95.829491, 34979.219, 178.812] |
| serving_bytes_per_token | 262144 | 65536 |
| serving_requests_8gib_1k | 32 | 38 |
| serving_state_bytes | 0 | 153944064 |
| serving_weight_bytes | 46163558400 | 54713457120 |

## fixture:qwen3.8-27b-w4a16.json (multimodal)

| figure | before | after |
|---|---|---|
| conv_kv_size | 262144 | 32768 |
| conv_layers | [0, 32] | [0, 64] |
| conv_prefill_weights_mb | 12852.0 | 51299.21875 |
| conv_serving_weight_bytes | 13214154752 | 54713457120 |
| conv_state_bytes | 0 | 153944064 |
| disagg_transfer_bytes_640 | 167772160 | 195887104 |
| eng_None_32768_params | 27919350016 | 28422666496 |
| eng_None_32768_state_mb | 0.0 | 153.944064 |
| eng_None_32768_total_mb | 58973.596557 | 60134.173581 |
| eng_None_4096_params | 27919350016 | 28422666496 |
| eng_None_4096_state_mb | 0.0 | 153.944064 |
| eng_None_4096_total_mb | 56601.298214 | 57761.875238 |
| eng_llamacpp_32768_params | 27919350016 | 28422666496 |
| eng_llamacpp_32768_state_mb | 0.0 | 153.944064 |
| eng_llamacpp_32768_total_mb | 58973.596557 | 60134.173581 |
| eng_llamacpp_4096_params | 27919350016 | 28422666496 |
| eng_llamacpp_4096_state_mb | 0.0 | 153.944064 |
| eng_llamacpp_4096_total_mb | 56601.298214 | 57761.875238 |
| genz_chunked_kv_mb | 1152.0 | 581.625 |
| genz_decode_kv_mb | 1024.0 | 402.8125 |
| genz_decode_weights_mb | 44025.0 | 48874.21875 |
| genz_kv_size | 131072 | 32768 |
| genz_layer_kinds | {"MHA-global": 64} | {"MHA-global": 16, "Recurrent": 48} |
| genz_prefill_kv_mb | 1024.0 | 402.8125 |
| genz_prefill_weights_mb | 46450.0 | 51299.21875 |
| prefix_kv_bytes_per_token | 262144 | 65536 |
| report_params | 27919350016 | 28422666496 |
| report_total_gb | 56.938550728 | 57.945183688 |
| report_weight_gb | 55.838700032 | 56.845332992 |
| runtime_chunked | [42.094544, 46450.0, 192.0] | [46.883662, 51299.219, 341.625] |
| runtime_decode | [28.985134, 44025.0, 129.0] | [31.953231, 48874.219, 619.5] |
| runtime_prefill | [86.74405, 46450.0, 128.0] | [95.829491, 51299.219, 178.812] |
| serving_bytes_per_token | 262144 | 65536 |
| serving_requests_8gib_1k | 32 | 38 |
| serving_state_bytes | 0 | 153944064 |
| serving_weight_bytes | 46163558400 | 54713457120 |

## fixture:qwen3.8-27b.json (multimodal)

| figure | before | after |
|---|---|---|
| conv_kv_size | 262144 | 32768 |
| conv_layers | [0, 32] | [0, 64] |
| conv_prefill_weights_mb | 12852.0 | 51299.21875 |
| conv_serving_weight_bytes | 13214154752 | 54713457120 |
| conv_state_bytes | 0 | 153944064 |
| disagg_transfer_bytes_640 | 167772160 | 195887104 |
| eng_None_32768_params | 27919350016 | 28422666496 |
| eng_None_32768_state_mb | 0.0 | 153.944064 |
| eng_None_32768_total_mb | 58973.596557 | 60134.173581 |
| eng_None_4096_params | 27919350016 | 28422666496 |
| eng_None_4096_state_mb | 0.0 | 153.944064 |
| eng_None_4096_total_mb | 56601.298214 | 57761.875238 |
| eng_llamacpp_32768_params | 27919350016 | 28422666496 |
| eng_llamacpp_32768_state_mb | 0.0 | 153.944064 |
| eng_llamacpp_32768_total_mb | 58973.596557 | 60134.173581 |
| eng_llamacpp_4096_params | 27919350016 | 28422666496 |
| eng_llamacpp_4096_state_mb | 0.0 | 153.944064 |
| eng_llamacpp_4096_total_mb | 56601.298214 | 57761.875238 |
| genz_chunked_kv_mb | 1152.0 | 581.625 |
| genz_decode_kv_mb | 1024.0 | 402.8125 |
| genz_decode_weights_mb | 44025.0 | 48874.21875 |
| genz_kv_size | 131072 | 32768 |
| genz_layer_kinds | {"MHA-global": 64} | {"MHA-global": 16, "Recurrent": 48} |
| genz_prefill_kv_mb | 1024.0 | 402.8125 |
| genz_prefill_weights_mb | 46450.0 | 51299.21875 |
| prefix_kv_bytes_per_token | 262144 | 65536 |
| report_params | 27919350016 | 28422666496 |
| report_total_gb | 56.938550728 | 57.945183688 |
| report_weight_gb | 55.838700032 | 56.845332992 |
| runtime_chunked | [42.094544, 46450.0, 192.0] | [46.883662, 51299.219, 341.625] |
| runtime_decode | [28.985134, 44025.0, 129.0] | [31.953231, 48874.219, 619.5] |
| runtime_prefill | [86.74405, 46450.0, 128.0] | [95.829491, 51299.219, 178.812] |
| serving_bytes_per_token | 262144 | 65536 |
| serving_requests_8gib_1k | 32 | 38 |
| serving_state_bytes | 0 | 153944064 |
| serving_weight_bytes | 46163558400 | 54713457120 |

## fixture:qwen3_8_27b_config.json (multimodal)

| figure | before | after |
|---|---|---|
| conv_kv_size | 262144 | 32768 |
| conv_layers | [0, 32] | [0, 64] |
| conv_prefill_weights_mb | 12852.0 | 51299.21875 |
| conv_serving_weight_bytes | 13214154752 | 54713457120 |
| conv_state_bytes | 0 | 153944064 |
| disagg_transfer_bytes_640 | 167772160 | 195887104 |
| eng_None_32768_params | 27919350016 | 28422666496 |
| eng_None_32768_state_mb | 0.0 | 153.944064 |
| eng_None_32768_total_mb | 58973.596557 | 60134.173581 |
| eng_None_4096_params | 27919350016 | 28422666496 |
| eng_None_4096_state_mb | 0.0 | 153.944064 |
| eng_None_4096_total_mb | 56601.298214 | 57761.875238 |
| eng_llamacpp_32768_params | 27919350016 | 28422666496 |
| eng_llamacpp_32768_state_mb | 0.0 | 153.944064 |
| eng_llamacpp_32768_total_mb | 58973.596557 | 60134.173581 |
| eng_llamacpp_4096_params | 27919350016 | 28422666496 |
| eng_llamacpp_4096_state_mb | 0.0 | 153.944064 |
| eng_llamacpp_4096_total_mb | 56601.298214 | 57761.875238 |
| genz_chunked_kv_mb | 1152.0 | 581.625 |
| genz_decode_kv_mb | 1024.0 | 402.8125 |
| genz_decode_weights_mb | 44025.0 | 48874.21875 |
| genz_kv_size | 131072 | 32768 |
| genz_layer_kinds | {"MHA-global": 64} | {"MHA-global": 16, "Recurrent": 48} |
| genz_prefill_kv_mb | 1024.0 | 402.8125 |
| genz_prefill_weights_mb | 46450.0 | 51299.21875 |
| prefix_kv_bytes_per_token | 262144 | 65536 |
| report_params | 27919350016 | 28422666496 |
| report_total_gb | 56.938550728 | 57.945183688 |
| report_weight_gb | 55.838700032 | 56.845332992 |
| runtime_chunked | [42.094544, 46450.0, 192.0] | [46.883662, 51299.219, 341.625] |
| runtime_decode | [28.985134, 44025.0, 129.0] | [31.953231, 48874.219, 619.5] |
| runtime_prefill | [86.74405, 46450.0, 128.0] | [95.829491, 51299.219, 178.812] |
| serving_bytes_per_token | 262144 | 65536 |
| serving_requests_8gib_1k | 32 | 38 |
| serving_state_bytes | 0 | 153944064 |
| serving_weight_bytes | 46163558400 | 54713457120 |

## fixture:state-spaces_mamba-2.8b-hf.json (state-space)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 12 | 1619 |
| conv_kv_size | 327680 | 0 |
| conv_serving_weight_bytes | 8646041600 | 5794452480 |
| conv_state_bytes | 0 | 22937600 |
| disagg_b_eff_4096 | 12 | 1619 |
| disagg_transfer_bytes_640 | 419430400 | 22937600 |
| eng_None_32768_state_mb | 0.0 | 22.9376 |
| eng_None_32768_total_mb | 5890.000896 | 5912.938496 |
| eng_None_4096_state_mb | 0.0 | 22.9376 |
| eng_None_4096_total_mb | 5713.840128 | 5736.777728 |
| eng_llamacpp_32768_state_mb | 0.0 | 22.9376 |
| eng_llamacpp_32768_total_mb | 5890.000896 | 5912.938496 |
| eng_llamacpp_4096_state_mb | 0.0 | 22.9376 |
| eng_llamacpp_4096_total_mb | 5713.840128 | 5736.777728 |
| facts_weight_bytes | 5588716544 | 5794452480 |
| genz_kv_size | 327680 | 0 |
| prefix_kv_bytes_per_token | 655360 | 0 |
| serving_bytes_per_token | 655360 | 0 |
| serving_requests_8gib_1k | 12 | 374 |
| serving_state_bytes | 0 | 22937600 |
| serving_weight_bytes | 8646041600 | 5794452480 |
| sim_max_running | 96 | 95 |
| slice_mib_16x4352 | [62208, true] | [63232, true] |
| vllm_limits | [2202, 105696, 2202, 1, 0, 31457280.0, true] | [2196, 105408, 2196, 1, 0, 31457280.0, true] |

## fixture:t5_flan-xl-geometry.json (encoder-decoder)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 1 | 23 |
| conv_kv_size | 131072 | 196608 |
| conv_layers | [16, 16] | [24, 24] |
| conv_prefill_weights_mb | 6678.0 | 5435.0 |
| conv_serving_weight_bytes | 6739197952 | 5568753664 |
| decode_fit_refused | null | true |
| disagg_b_eff_4096 | 1 | 23 |
| disagg_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'architectures', 'chunk_size_feed_forward', 'classifier_dropout', 'd_ff', 'd_kv', 'd_model', 'dense_act_" | null |
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
| facts_kv_bytes_4096 | 805306368 | 1610612736 |
| facts_weight_bytes | 6574247936 | 5568753664 |
| genz_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'architectures', 'chunk_size_feed_forward', 'classifier_dropout', 'd_ff', 'd_kv', 'd_model', 'dense_act_" | null |
| genz_chunked_kv_mb | null | 1728.0 |
| genz_decode_kv_mb | null | 1536.0 |
| genz_decode_weights_mb | null | 5309.5 |
| genz_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'architectures', 'chunk_size_feed_forward', 'classifier_dropout', 'd_ff', 'd_kv', 'd_model', 'dense_act_" | null |
| genz_ffn_kinds | null | {"Dense": 24} |
| genz_kv_size | null | 196608 |
| genz_layer_kinds | null | {"MHA-global": 24} |
| genz_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'architectures', 'chunk_size_feed_forward', 'classifier_dropout', 'd_ff', 'd_kv', 'd_model', 'dense_act_" | null |
| genz_prefill_kv_mb | null | 2304.0 |
| genz_prefill_weights_mb | null | 5435.0 |
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
| runtime_chunked | null | [2.787976, 5435.0, 288.0] |
| runtime_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'architectures', 'chunk_size_feed_forward', 'classifier_dropout'" | null |
| runtime_decode | null | [3.790492, 5309.5, 480.75] |
| runtime_decode_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'architectures', 'chunk_size_feed_forward', 'classifier_dropout'" | null |
| runtime_prefill | null | [13.28918, 5435.0, 288.0] |
| runtime_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'architectures', 'chunk_size_feed_forward', 'classifier_dropout'" | null |
| serving_bytes_per_token | null | 393216 |
| serving_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'architectures', 'chunk_size_feed_forward', 'classifier_dropout', 'd_ff', 'd_kv', 'd_model', 'dense_act_" | null |
| serving_requests_8gib_1k | null | 21 |
| serving_state_bytes | null | 0 |
| serving_weight_bytes | null | 5568753664 |
| serving_weight_bytes_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['_name_or_path', 'architectures', 'chunk_size_feed_forward', 'classifier_dropout', 'd_ff', 'd_kv', 'd_model', 'dense_act_" | null |
| sim_max_running | 331 | 69 |
| slice_mib_16x4352 | [27904, true] | [42496, true] |
| vllm_limits | [21856, 349696, 21856, 1, 0, 0.0, true] | [22176, 354816, 22176, 1, 0, 805306368.0, true] |
| vllm_running_1024 | 273 | 66 |
| vllm_running_8192 | 41 | 28 |

## fixture:tencent_Hunyuan-A13B-Instruct.json (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 68 | 1 |
| conv_prefill_weights_mb | null | 154338.609375 |
| conv_serving_weight_bytes | 6150217728 | 160786370560 |
| conv_weights_error | "TypeError: unsupported operand type(s) for /: 'list' and 'int'" | null |
| decode_fit_refused | null | true |
| disagg_b_eff_4096 | 68 | 1 |
| eng_None_32768_params | 78379909120 | 79587868672 |
| eng_None_32768_total_mb | 162262.745088 | 164678.664192 |
| eng_None_4096_params | 78379909120 | 79587868672 |
| eng_None_4096_total_mb | 157900.668928 | 160316.588032 |
| eng_llamacpp_32768_params | 78379909120 | 79587868672 |
| eng_llamacpp_32768_total_mb | 162262.745088 | 164678.664192 |
| eng_llamacpp_4096_params | 78379909120 | 79587868672 |
| eng_llamacpp_4096_total_mb | 157900.668928 | 160316.588032 |
| facts_error | "TypeError: int() argument must be a string, a bytes-like object or a real number, not 'list'" | null |
| facts_full_attention | null | true |
| facts_kv_bytes_4096 | null | 536870912 |
| facts_kv_bytes_per_token | null | 131072 |
| facts_layers | null | [32, 0, 0, null] |
| facts_state_bytes | null | 0 |
| facts_weight_bytes | null | 160786370560 |
| genz_chunked_error | "TypeError: unsupported operand type(s) for /: 'list' and 'int'" | null |
| genz_chunked_kv_mb | null | 576.0 |
| genz_decode_kv_mb | null | 512.0 |
| genz_decode_weights_mb | null | 153337.304688 |
| genz_error | "TypeError: unsupported operand type(s) for /: 'list' and 'int'" | null |
| genz_prefill_error | "TypeError: unsupported operand type(s) for /: 'list' and 'int'" | null |
| genz_prefill_kv_mb | null | 512.0 |
| genz_prefill_weights_mb | null | 154338.609375 |
| report_params | 78379909120 | 79587868672 |
| report_total_gb | 158.128839066 | 160.54475817 |
| report_weight_gb | 156.75981824 | 159.175737344 |
| runtime_chunked | null | [97.081372, 154338.609, 96.0] |
| runtime_chunked_error | "TypeError: unsupported operand type(s) for /: 'list' and 'int'" | null |
| runtime_decode | null | [43.440622, 153337.305, 64.5] |
| runtime_decode_error | "TypeError: unsupported operand type(s) for /: 'list' and 'int'" | null |
| runtime_prefill | null | [106.517926, 154338.609, 64.0] |
| runtime_prefill_error | "TypeError: unsupported operand type(s) for /: 'list' and 'int'" | null |
| serving_weight_bytes | 6150217728 | 160786370560 |
| sim_error | null | "RuntimeError: the KV pool cannot hold one request at the step's budget" |
| slice_mib_16x4352 | null | [81582, false] |
| vllm_error | "TypeError: int() argument must be a string, a bytes-like object or a real number, not 'list'" | null |
| vllm_limits | null | [0, 0, 0, 1, 0, 0.0, false] |
| vllm_running_1024 | null | 1024 |
| vllm_running_8192 | null | 1024 |

## fixture:tiiuae_Falcon-H1-34B-Instruct.json (hybrid)

| figure | before | after |
|---|---|---|
| conv_prefill_weights_mb | 54780.0 | 64166.71875 |
| conv_serving_weight_bytes | 54767124480 | 67285032448 |
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
| facts_weight_bytes | 80878263808 | 67285032448 |
| genz_chunked_kv_mb | 648.0 | 1228.21875 |
| genz_decode_kv_mb | 365.625 | 866.109375 |
| genz_decode_weights_mb | 71130.0 | 61616.71875 |
| genz_layer_kinds | {"Mamba": 72} | {"MHA-global": 72} |
| genz_prefill_kv_mb | 365.625 | 866.109375 |
| genz_prefill_weights_mb | 73680.0 | 64166.71875 |
| runtime_chunked | [49.399305, 54780.0, 108.0] | [58.622617, 64166.719, 688.219] |
| runtime_decode | [48.848846, 71130.0, 1445.625] | [40.955966, 61616.719, 1233.0] |
| runtime_prefill | [440.484613, 73680.0, 365.625] | [120.493241, 64166.719, 362.109] |
| serving_requests_8gib_1k | 56 | 18 |
| serving_state_bytes | 0 | 304201728 |
| serving_weight_bytes | 54767124480 | 67285032448 |
| sim_error | "RuntimeError: the KV pool cannot hold one request at the step's budget" | null |
| sim_max_running | null | 11 |
| vllm_limits | [0, 0, 0, 1, 0, 304349184.0, false] | [22, 45408, 22, 1, 0, 304349184.0, false] |
| vllm_running_1024 | 1024 | 11 |
| vllm_running_8192 | 1024 | 3 |

## fixture:unknown-arch.json (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 78 | 77 |
| conv_serving_weight_bytes | 6030360576 | 6423926784 |
| disagg_b_eff_4096 | 78 | 77 |
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
| serving_weight_bytes | 6030360576 | 6423926784 |

## fixture:unsloth_gemma-3-27b-it.json (multimodal)

| figure | before | after |
|---|---|---|
| conv_kv_size | 262144 | 253952 |
| conv_layers | [0, 32] | [0, 62] |
| conv_prefill_weights_mb | 12852.0 | 54202.3125 |
| conv_serving_weight_bytes | 13214154752 | 57675936480 |
| eng_None_32768_params | 27425866752 | 27426201840 |
| eng_None_32768_total_mb | 72733.676749 | 72734.346925 |
| eng_None_4096_params | 27425866752 | 27426201840 |
| eng_None_4096_total_mb | 57589.722726 | 57590.392902 |
| eng_llamacpp_32768_params | 27425866752 | 27426201840 |
| eng_llamacpp_32768_total_mb | 72733.676749 | 72734.346925 |
| eng_llamacpp_4096_params | 27425866752 | 27426201840 |
| eng_llamacpp_4096_total_mb | 57589.722726 | 57590.392902 |
| facts_full_attention | true | false |
| facts_kv_bytes_4096 | 2080374784 | 771751936 |
| facts_layers | [62, 0, 0, null] | [10, 52, 0, 1024] |
| genz_chunked_kv_mb | 2232.0 | 984.0 |
| genz_decode_kv_mb | 1984.0 | 736.0 |
| genz_decode_weights_mb | 37842.65625 | 51513.65625 |
| genz_layer_kinds | {"MHA-global": 62} | {"MHA-global": 10, "MHA-local": 52} |
| genz_prefill_kv_mb | 1984.0 | 736.0 |
| genz_prefill_weights_mb | 40531.3125 | 54202.3125 |
| report_params | 27425866752 | 27426201840 |
| report_total_gb | 58.137320571 | 58.137990747 |
| report_weight_gb | 54.851733504 | 54.85240368 |
| runtime_chunked | [36.811458, 40531.312, 372.0] | [49.091845, 54202.312, 372.0] |
| runtime_decode | [25.257654, 37842.656, 249.938] | [33.60246, 51513.656, 249.938] |
| runtime_prefill | [75.992259, 40531.312, 248.0] | [100.457467, 54202.312, 248.0] |
| serving_weight_bytes | 54015983616 | 57675936480 |
| sim_max_running | 32 | 28 |
| slice_mib_16x4352 | [81582, false] | [80896, true] |
| vllm_limits | [2067, 33072, 2067, 1, 0, 0.0, false] | [12818, 205088, 12818, 1, 6, 0.0, true] |
| vllm_running_1024 | 25 | 27 |
| vllm_running_8192 | 3 | 13 |

## fixture:whisper-large-v3.json (encoder-decoder)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 13 | 42 |
| conv_kv_size | 262144 | 81920 |
| conv_prefill_weights_mb | 13162.40625 | 3868.325195 |
| conv_serving_weight_bytes | 13376897024 | 3622233600 |
| conv_state_bytes | 0 | 245760000 |
| disagg_b_eff_4096 | 13 | 42 |
| disagg_transfer_bytes_640 | 335544320 | 350617600 |
| facts_kv_bytes_4096 | 671088640 | 916848640 |
| facts_weight_bytes | 3621163520 | 3622233600 |
| genz_chunked_kv_mb | 2304.0 | 1188.75 |
| genz_decode_kv_mb | 2048.0 | 874.375 |
| genz_decode_weights_mb | 10005.203125 | 2940.605469 |
| genz_kv_size | 262144 | 81920 |
| genz_prefill_kv_mb | 2048.0 | 1108.75 |
| genz_prefill_weights_mb | 10410.40625 | 3068.325195 |
| kv_cross_layers | 0 | 32 |
| kv_full_layers | 0 | 32 |
| kv_gb@32768 | 0.0 | 5.61446912 |
| kv_gb@4096 | 0.0 | 0.91684864 |
| kvgeo_32768 | [0.0, 1, 0, 0.0, 16, 0.0] | [163840.0, 32, 0, 0.0, 16, 245760000.0] |
| kvgeo_4096 | [0.0, 1, 0, 0.0, 16, 0.0] | [163840.0, 32, 0, 0.0, 16, 245760000.0] |
| prefix_kv_bytes_per_token | 524288 | 163840 |
| runtime_chunked | [9.680075, 10410.406, 384.0] | [1.947247, 3068.325, 588.75] |
| runtime_decode | [7.78941, 10005.203, 258.0] | [3.650774, 2940.605, 1018.125] |
| runtime_prefill | [20.982092, 10410.406, 256.0] | [15.824888, 3068.325, 548.75] |
| serving_bytes_per_token | 524288 | 163840 |
| serving_requests_8gib_1k | 16 | 20 |
| serving_state_bytes | 0 | 245760000 |
| serving_weight_bytes | 13376897024 | 3622233600 |
| sim_max_running | 400 | 173 |
| slice_mib_16x4352 | [65536, true] | [70144, true] |
| vllm_limits | [27594, 441504, 27594, 1, 0, 0.0, true] | [27593, 441488, 27593, 1, 0, 245760000.0, true] |
| vllm_running_1024 | 344 | 158 |
| vllm_running_8192 | 52 | 44 |

## genz:HuggingFaceH4/zephyr-7b-beta (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 14220787712 | 14483464192 |

## genz:Hypothetical/SuperLLM-10T-MoE (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 702862589952 | 18909179440128 |

## genz:Hypothetical/SuperLLM-5T-Dense (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 8838438715392 | 8863629803520 |

## genz:Qwen/Qwen-14B (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 43605032960 | 45162997760 |

## genz:Qwen/Qwen-1_8B (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 47 | 46 |
| disagg_b_eff_4096 | 47 | 46 |
| serving_weight_bytes | 4674027520 | 5296558080 |

## genz:Qwen/Qwen-72B (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 238714617856 | 241208672256 |

## genz:Qwen/Qwen-7B (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 9 | 8 |
| disagg_b_eff_4096 | 9 | 8 |
| serving_weight_bytes | 22853713920 | 24098906112 |

## genz:Qwen/Qwen1.5-0.5B (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 104 | 103 |
| disagg_b_eff_4096 | 104 | 103 |
| serving_weight_bytes | 927727616 | 1238992896 |

## genz:Qwen/Qwen1.5-1.8B (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 49 | 48 |
| disagg_b_eff_4096 | 49 | 48 |
| serving_weight_bytes | 3050831872 | 3673362432 |

## genz:Qwen/Qwen1.5-110B (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 238714617856 | 241208672256 |

## genz:Qwen/Qwen1.5-14B (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 26775388160 | 28333352960 |

## genz:Qwen/Qwen1.5-32B (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 62961745920 | 64520202240 |

## genz:Qwen/Qwen1.5-4B (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 21 | 20 |
| disagg_b_eff_4096 | 21 | 20 |
| serving_weight_bytes | 7121797120 | 7900124160 |

## genz:Qwen/Qwen1.5-72B (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 142077853696 | 144571908096 |

## genz:Qwen/Qwen1.5-7B (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 13 | 12 |
| disagg_b_eff_4096 | 13 | 12 |
| serving_weight_bytes | 14196670464 | 15441862656 |

## genz:Qwen/Qwen1.5-MoE-A2.7B (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 28 | 1 |
| disagg_b_eff_4096 | 28 | 1 |
| genz_decode_weights_mb | 41482.875 | 44122.875 |
| genz_prefill_weights_mb | 42076.375 | 44716.375 |
| runtime_chunked | [36.310833, 52636.375, 240.0] | [29.196454, 44716.375, 240.0] |
| runtime_decode | [9.181176, 41482.875, 161.25] | [10.952938, 44122.875, 161.25] |
| runtime_prefill | [31.69723, 42076.375, 160.0] | [36.821694, 44716.375, 160.0] |
| serving_weight_bytes | 4732747776 | 46888849408 |

## genz:Qwen/Qwen2-0.5B (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 833 | 828 |
| disagg_b_eff_4096 | 833 | 828 |
| serving_weight_bytes | 987922432 | 1260279552 |

## genz:Qwen/Qwen2.5-1.5B (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 339 | 335 |
| disagg_b_eff_4096 | 339 | 335 |
| serving_weight_bytes | 3087138816 | 3554061312 |

## genz:Qwen/Qwen2.5-14B (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 18 | 16 |
| disagg_b_eff_4096 | 18 | 16 |
| serving_weight_bytes | 27981250560 | 29539379200 |

## genz:Qwen/Qwen2.5-32B (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 63968378880 | 65526835200 |

## genz:Qwen/Qwen2.5-3B (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 243 | 239 |
| disagg_b_eff_4096 | 243 | 239 |
| serving_weight_bytes | 6171394048 | 6794022912 |

## genz:Qwen/Qwen2.5-72B (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 142916714496 | 145410768896 |

## genz:Qwen/Qwen2.5-7B (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 122 | 118 |
| disagg_b_eff_4096 | 122 | 118 |
| serving_weight_bytes | 14140571648 | 15230974976 |

## genz:THUDM/glm-4-9b-chat (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 151 | 143 |
| disagg_b_eff_4096 | 151 | 143 |
| serving_weight_bytes | 17557356544 | 18799534080 |

## genz:ai21labs/AI21-Jamba-1.5-Large (None)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 188743680 | 95846400 |
| genz_kv_size | 147456 | 18432 |
| genz_layer_kinds | {"Mamba": 72} | {"MHA-global": 9, "Mamba": 63} |
| serving_bytes_per_token | 294912 | 36864 |
| serving_requests_8gib_1k | 28 | 78 |
| serving_state_bytes | 0 | 72253440 |
| serving_weight_bytes | 109790101504 | 797110222848 |

## genz:ai21labs/AI21-Jamba-1.5-Mini (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 53 | 1 |
| disagg_b_eff_4096 | 53 | 1 |
| disagg_transfer_bytes_640 | 83886080 | 26542080 |
| genz_kv_size | 65536 | 8192 |
| genz_layer_kinds | {"Mamba": 32} | {"MHA-global": 4, "Mamba": 28} |
| serving_bytes_per_token | 131072 | 16384 |
| serving_requests_8gib_1k | 64 | 261 |
| serving_state_bytes | 0 | 16056320 |
| serving_weight_bytes | 14495514624 | 103140630528 |

## genz:ai21labs/Jamba-v0.1 (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 53 | 1 |
| disagg_b_eff_4096 | 53 | 1 |
| disagg_transfer_bytes_640 | 83886080 | 26542080 |
| genz_kv_size | 65536 | 8192 |
| genz_layer_kinds | {"Mamba": 32} | {"MHA-global": 4, "Mamba": 28} |
| serving_bytes_per_token | 131072 | 16384 |
| serving_requests_8gib_1k | 64 | 261 |
| serving_state_bytes | 0 | 16056320 |
| serving_weight_bytes | 14495514624 | 103140630528 |

## genz:bigscience/bloom-176b (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 467560038400 | 474757312512 |

## genz:bigscience/bloom-7b1 (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 12 | 11 |
| disagg_b_eff_4096 | 12 | 11 |
| serving_weight_bytes | 18161336320 | 20217044992 |

## genz:databricks/dbrx-base (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 28 | 1 |
| disagg_b_eff_4096 | 28 | 1 |
| serving_weight_bytes | 24134025216 | 263193047040 |

## genz:deepseek-ai/DeepSeek-V2 (None)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 786432000 | 44236800 |
| genz_chunked_kv_mb | 5400.0 | 303.75 |
| genz_decode_kv_mb | 150.0 | 270.0 |
| genz_decode_weights_mb | 450171.25 | 455571.25 |
| genz_prefill_kv_mb | 150.0 | 270.0 |
| genz_prefill_weights_mb | 451171.25 | 456571.25 |
| runtime_chunked_error | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 477.825927734375 GB, KV Cache:0.87890625. \n System:{'Flops': 400, 'Memory_size': 200, '" | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 445.870361328125 GB, KV Cache:0.0494384765625. \n System:{'Flops': 400, 'Memory_size': 2" |
| runtime_decode_error | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 439.620361328125 GB, KV Cache:0.0274658203125. \n System:{'Flops': 400, 'Memory_size': 2" | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 444.893798828125 GB, KV Cache:0.0494384765625. \n System:{'Flops': 400, 'Memory_size': 2" |
| runtime_prefill_error | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 440.596923828125 GB, KV Cache:0.018310546875. \n System:{'Flops': 400, 'Memory_size': 20" | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 445.870361328125 GB, KV Cache:0.032958984375. \n System:{'Flops': 400, 'Memory_size': 20" |
| serving_bytes_per_token | 1228800 | 69120 |
| serving_requests_8gib_1k | 6 | 121 |
| serving_weight_bytes | 36280729600 | 473426749440 |

## genz:deepseek-ai/DeepSeek-V3-Base (None)

| figure | before | after |
|---|---|---|
| disagg_transfer_bytes_640 | 17489920 | 44974080 |
| genz_chunked_kv_mb | 120.09375 | 308.8125 |
| genz_decode_kv_mb | 152.5 | 274.5 |
| genz_decode_weights_mb | 1335494.375 | 1340618.375 |
| genz_prefill_kv_mb | 152.5 | 274.5 |
| genz_prefill_weights_mb | 1337261.875 | 1342385.875 |
| runtime_chunked_error | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 1341.5540161132812 GB, KV Cache:0.0195465087890625. \n System:{'Flops': 400, 'Memory_siz" | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 1310.9237060546875 GB, KV Cache:0.050262451171875. \n System:{'Flops': 400, 'Memory_size" |
| runtime_decode_error | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 1304.1937255859375 GB, KV Cache:0.027923583984375. \n System:{'Flops': 400, 'Memory_size" | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 1309.1976318359375 GB, KV Cache:0.050262451171875. \n System:{'Flops': 400, 'Memory_size" |
| runtime_prefill_error | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 1305.9197998046875 GB, KV Cache:0.01861572265625. \n System:{'Flops': 400, 'Memory_size'" | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 1310.9237060546875 GB, KV Cache:0.03350830078125. \n System:{'Flops': 400, 'Memory_size'" |
| serving_bytes_per_token | 27328 | 70272 |
| serving_requests_8gib_1k | 306 | 119 |
| serving_weight_bytes | 62942150656 | 1397501192192 |

## genz:deepseek-ai/deepseek-llm-67b-base (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 133169152000 | 134850002944 |

## genz:deepseek-ai/deepseek-moe-16b-base (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 325 | 81 |
| disagg_b_eff_4096 | 325 | 81 |
| genz_decode_weights_mb | 30479.0 | 31403.0 |
| genz_prefill_weights_mb | 30879.0 | 31803.0 |
| runtime_decode | [8.107177, 30479.0, 14.109] | [8.783323, 31403.0, 14.109] |
| runtime_prefill | [22.946663, 30879.0, 14.0] | [24.880225, 31803.0, 14.0] |
| serving_weight_bytes | 4713349120 | 33348096000 |

## genz:facebook/OPT-1.3b (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 49 | 48 |
| disagg_b_eff_4096 | 49 | 48 |
| serving_weight_bytes | 3427139584 | 3633254400 |

## genz:facebook/OPT-13B (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 34300988160 | 34818617000 |

## genz:facebook/OPT-350M (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 908263424 | 1011320832 |

## genz:facebook/opt-125M (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 282 | 281 |
| disagg_b_eff_4096 | 282 | 281 |
| serving_weight_bytes | 303710208 | 380966400 |

## genz:facebook/opt-175b (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 465091952640 | 466332180480 |

## genz:facebook/opt-30B (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 79640723456 | 80362813440 |

## genz:facebook/opt-6.7b (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 17591697408 | 18004058112 |

## genz:facebook/opt-66B (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 174872788992 | 175801780224 |

## genz:google/gemma-2-27b-it (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 54452551680 | 56812704768 |

## genz:google/gemma-2-2B (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 86 | 83 |
| disagg_b_eff_4096 | 86 | 83 |
| serving_weight_bytes | 5228199936 | 6408092160 |

## genz:google/gemma-2-9B (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 17 | 16 |
| disagg_b_eff_4096 | 17 | 16 |
| serving_weight_bytes | 18482200576 | 20317817856 |

## genz:google/gemma-2b-it (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 502 | 488 |
| disagg_b_eff_4096 | 502 | 488 |
| serving_weight_bytes | 5012193280 | 6060920832 |

## genz:google/gemma-3-12B (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 86 | 83 |
| disagg_b_eff_4096 | 86 | 83 |
| serving_weight_bytes | 5256511488 | 6464715264 |

## genz:google/gemma-3-1B (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 375 | 369 |
| disagg_b_eff_4096 | 375 | 369 |
| serving_weight_bytes | 1999503360 | 2603605248 |

## genz:google/gemma-3-27B (None)

| figure | before | after |
|---|---|---|
| genz_chunked_kv_mb | 2232.0 | 1008.0 |
| serving_weight_bytes | 54015295488 | 56835211776 |

## genz:google/gemma-3-4B (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 98 | 58 |
| disagg_b_eff_4096 | 98 | 58 |
| disagg_transfer_bytes_640 | 55705600 | 89128960 |
| genz_chunked_kv_mb | 382.5 | 276.0 |
| genz_decode_kv_mb | 130.0 | 208.0 |
| genz_decode_weights_mb | 7442.5 | 8080.0 |
| genz_kv_size | 43520 | 69632 |
| genz_prefill_kv_mb | 130.0 | 208.0 |
| genz_prefill_weights_mb | 8722.5 | 9360.0 |
| runtime_chunked | [7.930882, 8722.5, 63.75] | [8.558023, 9360.0, 102.0] |
| runtime_decode | [6.118561, 7442.5, 42.832] | [6.524453, 8080.0, 68.531] |
| runtime_prefill | [17.806114, 8722.5, 42.5] | [19.024784, 9360.0, 68.0] |
| serving_bytes_per_token | 87040 | 139264 |
| serving_requests_8gib_1k | 96 | 60 |
| serving_weight_bytes | 7804026880 | 9815024640 |

## genz:google/gemma-7b-it (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 13 | 12 |
| disagg_b_eff_4096 | 13 | 12 |
| serving_weight_bytes | 17075011584 | 18648225792 |

## genz:google/palm (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 1284032692224 | 1285221076992 |

## genz:intfloat/e5-mistral-7b-instruct (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 14220787712 | 14483464192 |

## genz:intfloat/multilingual-e5-base-instruct (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 280 | 277 |
| disagg_b_eff_4096 | 280 | 277 |
| serving_weight_bytes | 610495488 | 994536960 |

## genz:intfloat/multilingual-e5-large-instruct (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 103 | 102 |
| disagg_b_eff_4096 | 103 | 102 |
| serving_weight_bytes | 1317310464 | 1829414912 |

## genz:intfloat/multilingual-e5-small-instruct (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 565 | 563 |
| disagg_b_eff_4096 | 565 | 563 |
| serving_weight_bytes | 248651520 | 440699136 |

## genz:meta-llama/Llama-2-13B (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 25703219200 | 26031728640 |

## genz:meta-llama/Llama-2-7B (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 13214154752 | 13476831232 |

## genz:meta-llama/Llama-3.1-70B (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 139003428864 | 141107412992 |

## genz:meta-llama/Llama-3.2-1B (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 301 | 297 |
| disagg_b_eff_4096 | 301 | 297 |
| serving_weight_bytes | 2471493632 | 2996965376 |

## genz:meta-llama/Llama-3.2-3B (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 77 | 76 |
| disagg_b_eff_4096 | 77 | 76 |
| serving_weight_bytes | 6425149440 | 7213504512 |

## genz:meta-llama/Llama-4-Maverick-17B-128E (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 13 | 1 |
| disagg_b_eff_4096 | 13 | 1 |
| genz_decode_weights_mb | 1482353.125 | 1493873.125 |
| genz_prefill_weights_mb | 1484326.25 | 1495846.25 |
| runtime_decode_error | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 1447.6104736328125 GB, KV Cache:0.140625. \n System:{'Flops': 400, 'Memory_size': 200, '" | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 1458.8604736328125 GB, KV Cache:0.140625. \n System:{'Flops': 400, 'Memory_size': 200, '" |
| runtime_prefill_error | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 1449.537353515625 GB, KV Cache:0.09375. \n System:{'Flops': 400, 'Memory_size': 200, 'Me" | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 1460.787353515625 GB, KV Cache:0.09375. \n System:{'Flops': 400, 'Memory_size': 200, 'Me" |
| serving_weight_bytes | 32267960320 | 1568509470720 |

## genz:meta-llama/Llama-4-Scout-17B-16E (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 28 | 1 |
| decode_fit_refused | true | null |
| disagg_b_eff_4096 | 28 | 1 |
| genz_decode_weights_mb | 192060.625 | 203580.625 |
| genz_prefill_weights_mb | 194033.75 | 205553.75 |
| runtime_decode | [34.976174, 192060.625, 96.75] | [42.200289, 203580.625, 96.75] |
| runtime_prefill | [133.914737, 194033.75, 96.0] | null |
| runtime_prefill_error | null | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 200.736083984375 GB, KV Cache:0.09375. \n System:{'Flops': 400, 'Memory_size': 200, 'Mem" |
| serving_weight_bytes | 20188364800 | 215539722240 |

## genz:meta-llama/llama-2-33b (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 64630292480 | 65057887232 |

## genz:meta-llama/meta-llama-3.1-405b (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 807495794688 | 811706777600 |

## genz:meta-llama/meta-llama-3.1-70b (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 137426370560 | 137953296384 |

## genz:meta-llama/meta-llama-3.1-8b (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 52 | 50 |
| disagg_b_eff_4096 | 52 | 50 |
| serving_weight_bytes | 15009316864 | 16060522496 |

## genz:microsoft/Phi-3-medium (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 18 | 17 |
| disagg_b_eff_4096 | 18 | 17 |
| serving_weight_bytes | 27591311360 | 27920476160 |

## genz:microsoft/Phi-3-mini (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 22 | 21 |
| disagg_b_eff_4096 | 22 | 21 |
| serving_weight_bytes | 7444758528 | 7642159104 |

## genz:microsoft/Phi-3-small (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 52 | 50 |
| disagg_b_eff_4096 | 52 | 50 |
| serving_weight_bytes | 14780727296 | 15603343360 |

## genz:microsoft/Phi-3.5-MoE (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 65 | 1 |
| disagg_b_eff_4096 | 65 | 1 |
| serving_weight_bytes | 7980187648 | 83745054720 |

## genz:mistralai/Ministral-8B-Instruct-2410 (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 46 | 44 |
| disagg_b_eff_4096 | 46 | 44 |
| serving_weight_bytes | 14965276672 | 16039616512 |

## genz:mistralai/Mistral-7B (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 14220787712 | 14483464192 |

## genz:mistralai/Mistral-Large (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 244410482688 | 245220139008 |

## genz:mistralai/Mistral-NeMo-12B-Instruct (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 22 | 20 |
| disagg_b_eff_4096 | 22 | 20 |
| serving_weight_bytes | 24201134080 | 25544140800 |

## genz:mistralai/Mistral-Small (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 44090523648 | 44494565376 |

## genz:mistralai/Mixtral-8x22B (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 44090523648 | 281260142592 |

## genz:mistralai/Mixtral-8x7B (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 53 | 1 |
| disagg_b_eff_4096 | 53 | 1 |
| serving_weight_bytes | 14220787712 | 93405585408 |

## genz:nvidia/Llama-3_1-Nemotron-51B-Instruct (None)

| figure | before | after |
|---|---|---|
| decode_fit_refused | null | true |
| disagg_transfer_bytes_640 | 209715200 | 121569280 |
| genz_chunked_kv_mb | 1440.0 | 834.75 |
| genz_kv_size | 131840 | 94976 |
| runtime_chunked | [121.497516, 134568.0, 240.0] | [123.044728, 136532.0, 139.125] |
| runtime_decode | null | [84.183264, 134528.0, 93.475] |
| runtime_decode_error | "ValueError: Layer Name:Attn Linear not found in the breakdown function" | null |
| runtime_prefill | null | [248.132118, 136532.0, 92.75] |
| runtime_prefill_error | "ValueError: Layer Name:Attn Linear not found in the breakdown function" | null |
| serving_bytes_per_token | 327680 | 189952 |
| serving_requests_8gib_1k | 25 | 44 |
| serving_weight_bytes | 139003428864 | 141107412992 |

## genz:nvidia/Mistral-NeMo-Minitron-8B-Base (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 40 | 38 |
| disagg_b_eff_4096 | 40 | 38 |
| serving_weight_bytes | 15753805824 | 16828211200 |

## genz:nvidia/Nemotron-4-15B (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 9 | 3 |
| disagg_b_eff_4096 | 9 | 3 |
| serving_weight_bytes | 37773901824 | 40920428544 |

## genz:nvidia/Nemotron-4-340B-Instruct (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 933526241280 | 942970540032 |

## genz:openai/GPT-4 (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 476278161408 | 5806187369472 |

## genz:openai/gpt2 (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 282 | 281 |
| disagg_b_eff_4096 | 282 | 281 |
| serving_weight_bytes | 303687168 | 380920320 |

## genz:state-spaces/mamba-130m-hf (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 141 | 16514 |
| disagg_b_eff_4096 | 141 | 16514 |
| disagg_transfer_bytes_640 | 47185920 | 2580480 |
| genz_kv_size | 36864 | 0 |
| serving_bytes_per_token | 73728 | 0 |
| serving_requests_8gib_1k | 113 | 3328 |
| serving_state_bytes | 0 | 2580480 |
| serving_weight_bytes | 360345600 | 335537664 |

## genz:state-spaces/mamba-2.8b-hf (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 12 | 1619 |
| disagg_b_eff_4096 | 12 | 1619 |
| disagg_transfer_bytes_640 | 419430400 | 22937600 |
| genz_kv_size | 327680 | 0 |
| serving_bytes_per_token | 655360 | 0 |
| serving_requests_8gib_1k | 12 | 374 |
| serving_state_bytes | 0 | 22937600 |
| serving_weight_bytes | 8646041600 | 5794452480 |

## genz:tiiuae/falcon-7b-instruct (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 21734424576 | 22325953408 |

## genz:tiiuae/falcon-mamba-7b (None)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 7 | 773 |
| disagg_b_eff_4096 | 7 | 773 |
| disagg_transfer_bytes_640 | 671088640 | 36700160 |
| genz_decode_weights_mb | 13225.3125 | 13340.5 |
| genz_kv_size | 524288 | 0 |
| genz_prefill_weights_mb | 13618.125 | 13848.5 |
| runtime_chunked | [8.767965, 8978.125, 768.0] | [8.974503, 9208.5, 768.0] |
| runtime_decode | [11.003094, 13225.312, 68.0] | [11.073467, 13340.5, 68.0] |
| runtime_prefill | [45.179762, 13618.125, 20.0] | [45.592034, 13848.5, 20.0] |
| serving_bytes_per_token | 1048576 | 0 |
| serving_requests_8gib_1k | 8 | 234 |
| serving_state_bytes | 0 | 36700160 |
| serving_weight_bytes | 9001828352 | 14545854464 |

## genz:xai-org/grok-1 (None)

| figure | before | after |
|---|---|---|
| serving_weight_bytes | 128849018880 | 942216327168 |

## gguf:Qwen3.5-2B-Q4_K_M.gguf (hybrid)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 196 | 560 |
| conv_kv_size | 24576 | 6144 |
| conv_prefill_weights_mb | 4148.0 | 4559.164062 |
| conv_serving_weight_bytes | 3332374528 | 3763650176 |
| conv_state_bytes | 0 | 19537920 |
| disagg_b_eff_4096 | 196 | 560 |
| disagg_transfer_bytes_640 | 31457280 | 27402240 |
| eng_None_32768_kv_mb | 1610.612736 | 402.653184 |
| eng_None_32768_params | 1901268992 | 1881922368 |
| eng_None_32768_state_mb | 0.0 | 19.53792 |
| eng_None_32768_total_mb | 5775.538586 | 4548.423706 |
| eng_None_4096_kv_mb | 201.326592 | 50.331648 |
| eng_None_4096_params | 1901268992 | 1881922368 |
| eng_None_4096_state_mb | 0.0 | 19.53792 |
| eng_None_4096_total_mb | 4185.058509 | 4014.908237 |
| eng_llamacpp_32768_kv_mb | 1610.612736 | 402.653184 |
| eng_llamacpp_32768_params | 1901268992 | 1881922368 |
| eng_llamacpp_32768_state_mb | 0.0 | 19.53792 |
| eng_llamacpp_32768_total_mb | 5775.538586 | 4548.423706 |
| eng_llamacpp_4096_kv_mb | 201.326592 | 50.331648 |
| eng_llamacpp_4096_params | 1901268992 | 1881922368 |
| eng_llamacpp_4096_state_mb | 0.0 | 19.53792 |
| eng_llamacpp_4096_total_mb | 4185.058509 | 4014.908237 |
| facts_full_attention | true | false |
| facts_kv_bytes_4096 | 201326592 | 50331648 |
| facts_kv_bytes_per_token | 49152 | 12288 |
| facts_layers | [24, 0, 0, null] | [6, 0, 18, null] |
| facts_state_bytes | 0 | 19537920 |
| facts_weight_bytes | 4349718528 | 3763650176 |
| genz_chunked_kv_mb | 216.0 | 91.265625 |
| genz_decode_kv_mb | 192.0 | 66.632812 |
| genz_decode_weights_mb | 2602.0 | 3589.164062 |
| genz_kv_size | 24576 | 6144 |
| genz_layer_kinds | {"MHA-global": 24} | {"MHA-global": 6, "Recurrent": 18} |
| genz_prefill_kv_mb | 192.0 | 66.632812 |
| genz_prefill_weights_mb | 3572.0 | 4559.164062 |
| gguf_config_keys | ["head_dim", "hidden_size", "intermediate_size", "max_position_embeddings", "model_type", "num_attention_heads", "num_hidden_layers", "num_key_value_heads", "vocab_size"] | ["head_dim", "hidden_act", "hidden_size", "intermediate_size", "layer_types", "linear_conv_kernel_dim", "linear_key_head_dim", "linear_num_key_heads", "linear_num_value_heads", "linear_value_head_dim", "max_position_embeddings", "model_type", "num_attention_heads", "num_hidden_layers", "num_key_value_heads", "rope_theta", "ssm_conv_kernel", "ssm_inner_size", "ssm_n_groups", "ssm_state_size", "tie_word_embeddings", "time_step_rank", "vocab_size"] |
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
| report_params | 1901268992 | 1881922368 |
| report_state_gb | 0.0 | 0.01953792 |
| report_total_gb | 4.261562614 | 4.065120937 |
| report_weight_gb | 3.802537984 | 3.763844736 |
| runtime_chunked | [3.259829, 3572.0, 36.0] | [4.239539, 4559.164, 46.266] |
| runtime_decode | [2.991388, 2602.0, 24.188] | [3.51989, 3589.164, 80.578] |
| runtime_prefill | [8.208606, 3572.0, 24.0] | [10.054049, 4559.164, 24.633] |
| serving_bytes_per_token | 49152 | 12288 |
| serving_requests_8gib_1k | 170 | 267 |
| serving_state_bytes | 0 | 19537920 |
| serving_weight_bytes | 3332374528 | 3763650176 |
| slice_mib_16x4352 | [14080, true] | [12032, true] |
| state_gb | 0.0 | 0.01953792 |
| vllm_limits | [88602, 1417632, 88602, 1, 0, 0.0, true] | [10511, 5717984, 10511, 1, 0, 20054016.0, true] |
| vllm_running_8192 | 167 | 553 |

## gguf:Qwen3.5-2B-Q4_K_M.metadata.gguf (hybrid)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 196 | 560 |
| conv_kv_size | 24576 | 6144 |
| conv_prefill_weights_mb | 4148.0 | 4559.164062 |
| conv_serving_weight_bytes | 3332374528 | 3763650176 |
| conv_state_bytes | 0 | 19537920 |
| disagg_b_eff_4096 | 196 | 560 |
| disagg_transfer_bytes_640 | 31457280 | 27402240 |
| eng_None_32768_kv_mb | 1610.612736 | 402.653184 |
| eng_None_32768_params | 1901268992 | 1881922368 |
| eng_None_32768_state_mb | 0.0 | 19.53792 |
| eng_None_32768_total_mb | 5775.538586 | 4548.423706 |
| eng_None_4096_kv_mb | 201.326592 | 50.331648 |
| eng_None_4096_params | 1901268992 | 1881922368 |
| eng_None_4096_state_mb | 0.0 | 19.53792 |
| eng_None_4096_total_mb | 4185.058509 | 4014.908237 |
| eng_llamacpp_32768_kv_mb | 1610.612736 | 402.653184 |
| eng_llamacpp_32768_params | 1901268992 | 1881922368 |
| eng_llamacpp_32768_state_mb | 0.0 | 19.53792 |
| eng_llamacpp_32768_total_mb | 5775.538586 | 4548.423706 |
| eng_llamacpp_4096_kv_mb | 201.326592 | 50.331648 |
| eng_llamacpp_4096_params | 1901268992 | 1881922368 |
| eng_llamacpp_4096_state_mb | 0.0 | 19.53792 |
| eng_llamacpp_4096_total_mb | 4185.058509 | 4014.908237 |
| facts_full_attention | true | false |
| facts_kv_bytes_4096 | 201326592 | 50331648 |
| facts_kv_bytes_per_token | 49152 | 12288 |
| facts_layers | [24, 0, 0, null] | [6, 0, 18, null] |
| facts_state_bytes | 0 | 19537920 |
| facts_weight_bytes | 4349718528 | 3763650176 |
| genz_chunked_kv_mb | 216.0 | 91.265625 |
| genz_decode_kv_mb | 192.0 | 66.632812 |
| genz_decode_weights_mb | 2602.0 | 3589.164062 |
| genz_kv_size | 24576 | 6144 |
| genz_layer_kinds | {"MHA-global": 24} | {"MHA-global": 6, "Recurrent": 18} |
| genz_prefill_kv_mb | 192.0 | 66.632812 |
| genz_prefill_weights_mb | 3572.0 | 4559.164062 |
| gguf_config_keys | ["head_dim", "hidden_size", "intermediate_size", "max_position_embeddings", "model_type", "num_attention_heads", "num_hidden_layers", "num_key_value_heads", "vocab_size"] | ["head_dim", "hidden_act", "hidden_size", "intermediate_size", "layer_types", "linear_conv_kernel_dim", "linear_key_head_dim", "linear_num_key_heads", "linear_num_value_heads", "linear_value_head_dim", "max_position_embeddings", "model_type", "num_attention_heads", "num_hidden_layers", "num_key_value_heads", "rope_theta", "ssm_conv_kernel", "ssm_inner_size", "ssm_n_groups", "ssm_state_size", "tie_word_embeddings", "time_step_rank", "vocab_size"] |
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
| report_params | 1901268992 | 1881922368 |
| report_state_gb | 0.0 | 0.01953792 |
| report_total_gb | 4.261562614 | 4.065120937 |
| report_weight_gb | 3.802537984 | 3.763844736 |
| runtime_chunked | [3.259829, 3572.0, 36.0] | [4.239539, 4559.164, 46.266] |
| runtime_decode | [2.991388, 2602.0, 24.188] | [3.51989, 3589.164, 80.578] |
| runtime_prefill | [8.208606, 3572.0, 24.0] | [10.054049, 4559.164, 24.633] |
| serving_bytes_per_token | 49152 | 12288 |
| serving_requests_8gib_1k | 170 | 267 |
| serving_state_bytes | 0 | 19537920 |
| serving_weight_bytes | 3332374528 | 3763650176 |
| slice_mib_16x4352 | [14080, true] | [12032, true] |
| state_gb | 0.0 | 0.01953792 |
| vllm_limits | [88602, 1417632, 88602, 1, 0, 0.0, true] | [10511, 5717984, 10511, 1, 0, 20054016.0, true] |
| vllm_running_8192 | 167 | 553 |

## gguf:SmolVLM-500M-Instruct-Q8_0.gguf (decoder-only)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 723763200 | 818505600 |
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
| serving_weight_bytes | 723763200 | 818505600 |

## gguf:SmolVLM-500M-Instruct-Q8_0.metadata.gguf (decoder-only)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 723763200 | 818505600 |
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
| serving_weight_bytes | 723763200 | 818505600 |

## gguf:gpt-oss-20b-MXFP4.gguf (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 195 | 5 |
| conv_prefill_weights_mb | 4563.28125 | 39878.4375 |
| conv_serving_weight_bytes | 3626680320 | 41815856256 |
| disagg_b_eff_4096 | 195 | 5 |
| eng_None_32768_params | 1992044160 | 20908068480 |
| eng_None_32768_total_mb | 5945.764301 | 44049.60384 |
| eng_None_4096_params | 1992044160 | 20908068480 |
| eng_None_4096_total_mb | 4360.946534 | 42328.890624 |
| eng_llamacpp_32768_params | 1992044160 | 20908068480 |
| eng_llamacpp_32768_total_mb | 5945.764301 | 44049.60384 |
| eng_llamacpp_4096_params | 1992044160 | 20908068480 |
| eng_llamacpp_4096_total_mb | 4360.946534 | 42328.890624 |
| facts_weight_bytes | 4785229440 | 41815856256 |
| genz_decode_weights_mb | 3078.984375 | 38773.828125 |
| genz_ffn_kinds | {"Dense": 24} | {"MoE": 24} |
| genz_prefill_weights_mb | 4183.59375 | 39878.4375 |
| gguf_config_keys | ["head_dim", "hidden_size", "intermediate_size", "max_position_embeddings", "model_type", "num_attention_heads", "num_hidden_layers", "num_key_value_heads", "vocab_size"] | ["head_dim", "hidden_act", "hidden_size", "intermediate_size", "max_position_embeddings", "model_type", "moe_intermediate_size", "n_routed_experts", "num_attention_heads", "num_experts", "num_experts_per_tok", "num_hidden_layers", "num_key_value_heads", "rope_theta", "tie_word_embeddings", "vocab_size"] |
| report_params | 1992044160 | 20908068480 |
| report_total_gb | 4.436318177 | 42.431441357 |
| report_weight_gb | 3.98408832 | 41.81613696 |
| runtime_chunked | [3.861015, 4183.594, 36.0] | [25.702025, 39878.438, 36.0] |
| runtime_decode | [3.282565, 3078.984, 24.188] | [12.70061, 38773.828, 24.188] |
| runtime_prefill | [9.452835, 4183.594, 24.0] | [31.010111, 39878.438, 24.0] |
| serving_weight_bytes | 3626680320 | 41815856256 |
| slice_mib_16x4352 | [14336, true] | [52736, true] |
| vllm_limits | [88043, 1408688, 88043, 1, 0, 0.0, true] | [40956, 655296, 40956, 1, 0, 0.0, true] |
| vllm_running_1024 | 1024 | 511 |
| vllm_running_8192 | 166 | 77 |

## gguf:gpt-oss-20b-MXFP4.metadata.gguf (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 195 | 5 |
| conv_prefill_weights_mb | 4563.28125 | 39878.4375 |
| conv_serving_weight_bytes | 3626680320 | 41815856256 |
| disagg_b_eff_4096 | 195 | 5 |
| eng_None_32768_params | 1992044160 | 20908068480 |
| eng_None_32768_total_mb | 5945.764301 | 44049.60384 |
| eng_None_4096_params | 1992044160 | 20908068480 |
| eng_None_4096_total_mb | 4360.946534 | 42328.890624 |
| eng_llamacpp_32768_params | 1992044160 | 20908068480 |
| eng_llamacpp_32768_total_mb | 5945.764301 | 44049.60384 |
| eng_llamacpp_4096_params | 1992044160 | 20908068480 |
| eng_llamacpp_4096_total_mb | 4360.946534 | 42328.890624 |
| facts_weight_bytes | 4785229440 | 41815856256 |
| genz_decode_weights_mb | 3078.984375 | 38773.828125 |
| genz_ffn_kinds | {"Dense": 24} | {"MoE": 24} |
| genz_prefill_weights_mb | 4183.59375 | 39878.4375 |
| gguf_config_keys | ["head_dim", "hidden_size", "intermediate_size", "max_position_embeddings", "model_type", "num_attention_heads", "num_hidden_layers", "num_key_value_heads", "vocab_size"] | ["head_dim", "hidden_act", "hidden_size", "intermediate_size", "max_position_embeddings", "model_type", "moe_intermediate_size", "n_routed_experts", "num_attention_heads", "num_experts", "num_experts_per_tok", "num_hidden_layers", "num_key_value_heads", "rope_theta", "tie_word_embeddings", "vocab_size"] |
| report_params | 1992044160 | 20908068480 |
| report_total_gb | 4.436318177 | 42.431441357 |
| report_weight_gb | 3.98408832 | 41.81613696 |
| runtime_chunked | [3.861015, 4183.594, 36.0] | [25.702025, 39878.438, 36.0] |
| runtime_decode | [3.282565, 3078.984, 24.188] | [12.70061, 38773.828, 24.188] |
| runtime_prefill | [9.452835, 4183.594, 24.0] | [31.010111, 39878.438, 24.0] |
| serving_weight_bytes | 3626680320 | 41815856256 |
| slice_mib_16x4352 | [14336, true] | [52736, true] |
| vllm_limits | [88043, 1408688, 88043, 1, 0, 0.0, true] | [40956, 655296, 40956, 1, 0, 0.0, true] |
| vllm_running_1024 | 1024 | 511 |
| vllm_running_8192 | 166 | 77 |

## gguf:nomic-embed-text-v1.5.Q8_0.gguf (decoder-only)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 273374208 | 273412608 |
| eng_None_32768_params | 109986816 | 136725504 |
| eng_None_32768_total_mb | 1594.027622 | 1647.504998 |
| eng_None_4096_params | 109986816 | 136725504 |
| eng_None_4096_total_mb | 454.015795 | 507.493171 |
| eng_llamacpp_32768_params | 109986816 | 136725504 |
| eng_llamacpp_32768_total_mb | 1594.027622 | 1647.504998 |
| eng_llamacpp_4096_params | 109986816 | 136725504 |
| eng_llamacpp_4096_total_mb | 454.015795 | 507.493171 |
| facts_weight_bytes | 320294400 | 273412608 |
| genz_decode_weights_mb | 206.709961 | 260.709961 |
| genz_prefill_weights_mb | 251.419922 | 305.419922 |
| gguf_config_keys | ["hidden_size", "intermediate_size", "max_position_embeddings", "model_type", "num_attention_heads", "num_hidden_layers", "num_key_value_heads", "vocab_size"] | ["hidden_act", "hidden_size", "intermediate_size", "max_position_embeddings", "model_type", "num_attention_heads", "num_hidden_layers", "num_key_value_heads", "rope_theta", "tie_word_embeddings", "vocab_size"] |
| report_params | 109986816 | 136725504 |
| report_total_gb | 0.500824228 | 0.554301604 |
| report_weight_gb | 0.219973632 | 0.273451008 |
| runtime_chunked | [0.250125, 251.42, 27.0] | [0.298632, 305.42, 27.0] |
| runtime_decode | [1.332179, 206.71, 18.141] | [1.365157, 260.71, 18.141] |
| runtime_prefill | [1.851672, 251.42, 18.0] | [1.948309, 305.42, 18.0] |
| serving_weight_bytes | 273374208 | 273412608 |
| vllm_limits | [126489, 2023824, 126489, 1, 0, 0.0, true] | [126569, 2025104, 126569, 1, 0, 0.0, true] |

## gguf:nomic-embed-text-v1.5.Q8_0.metadata.gguf (decoder-only)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 273374208 | 273412608 |
| eng_None_32768_params | 109986816 | 136725504 |
| eng_None_32768_total_mb | 1594.027622 | 1647.504998 |
| eng_None_4096_params | 109986816 | 136725504 |
| eng_None_4096_total_mb | 454.015795 | 507.493171 |
| eng_llamacpp_32768_params | 109986816 | 136725504 |
| eng_llamacpp_32768_total_mb | 1594.027622 | 1647.504998 |
| eng_llamacpp_4096_params | 109986816 | 136725504 |
| eng_llamacpp_4096_total_mb | 454.015795 | 507.493171 |
| facts_weight_bytes | 320294400 | 273412608 |
| genz_decode_weights_mb | 206.709961 | 260.709961 |
| genz_prefill_weights_mb | 251.419922 | 305.419922 |
| gguf_config_keys | ["hidden_size", "intermediate_size", "max_position_embeddings", "model_type", "num_attention_heads", "num_hidden_layers", "num_key_value_heads", "vocab_size"] | ["hidden_act", "hidden_size", "intermediate_size", "max_position_embeddings", "model_type", "num_attention_heads", "num_hidden_layers", "num_key_value_heads", "rope_theta", "tie_word_embeddings", "vocab_size"] |
| report_params | 109986816 | 136725504 |
| report_total_gb | 0.500824228 | 0.554301604 |
| report_weight_gb | 0.219973632 | 0.273451008 |
| runtime_chunked | [0.250125, 251.42, 27.0] | [0.298632, 305.42, 27.0] |
| runtime_decode | [1.332179, 206.71, 18.141] | [1.365157, 260.71, 18.141] |
| runtime_prefill | [1.851672, 251.42, 18.0] | [1.948309, 305.42, 18.0] |
| serving_weight_bytes | 273374208 | 273412608 |
| vllm_limits | [126489, 2023824, 126489, 1, 0, 0.0, true] | [126569, 2025104, 126569, 1, 0, 0.0, true] |

## synth:blip2-flan-t5-xl (multimodal)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 282 | 22 |
| conv_kv_size | 131072 | 196608 |
| conv_layers | [16, 16] | [24, 24] |
| conv_prefill_weights_mb | 6676.0 | 5435.0 |
| conv_serving_weight_bytes | 6738149376 | 7295498496 |
| disagg_b_eff_4096 | 282 | 22 |
| disagg_transfer_bytes_640 | 23592960 | 251658240 |
| eng_None_32768_kv_mb | 0.0 | 12884.901888 |
| eng_None_32768_params | 1663986176 | 3391313664 |
| eng_None_32768_total_mb | 3656.150426 | 19995.70729 |
| eng_None_4096_kv_mb | 0.0 | 1610.612736 |
| eng_None_4096_params | 1663986176 | 3391313664 |
| eng_None_4096_total_mb | 3495.089152 | 8560.356864 |
| eng_llamacpp_32768_kv_mb | 0.0 | 12884.901888 |
| eng_llamacpp_32768_params | 1663986176 | 3391313664 |
| eng_llamacpp_32768_total_mb | 3656.150426 | 19995.70729 |
| eng_llamacpp_4096_kv_mb | 0.0 | 1610.612736 |
| eng_llamacpp_4096_params | 1663986176 | 3391313664 |
| eng_llamacpp_4096_total_mb | 3495.089152 | 8560.356864 |
| facts_kv_bytes_4096 | 805306368 | 1610612736 |
| facts_weight_bytes | 8572457984 | 7295498496 |
| genz_chunked_kv_mb | 162.0 | 1728.0 |
| genz_decode_kv_mb | 144.0 | 1536.0 |
| genz_decode_weights_mb | 208.875 | 4349.5 |
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
| report_params | 1663986176 | 3391313664 |
| report_total_gb | 3.528512512 | 8.915902771 |
| report_weight_gb | 3.327972352 | 6.782627328 |
| runtime_chunked | [0.254007, 255.75, 27.0] | [2.356801, 4475.0, 288.0] |
| runtime_decode | [1.333507, 208.875, 18.141] | [3.49746, 4349.5, 480.75] |
| runtime_prefill | [1.859421, 255.75, 18.0] | [11.571193, 4475.0, 288.0] |
| serving_bytes_per_token | 36864 | 393216 |
| serving_requests_8gib_1k | 227 | 21 |
| serving_weight_bytes | 275644416 | 7295498496 |
| sim_max_running | 322 | 67 |
| slice_mib_16x4352 | [30208, true] | [44288, true] |
| vllm_limits | [21221, 339536, 21221, 1, 0, 0.0, true] | [21627, 346032, 21627, 1, 0, 805306368.0, true] |
| vllm_running_1024 | 265 | 64 |
| vllm_running_8192 | 40 | 27 |

## synth:chatglm3 (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 13 | 14 |
| conv_kv_size | 262144 | 229376 |
| conv_layers | [0, 32] | [0, 28] |
| conv_prefill_weights_mb | 12852.0 | 13588.0 |
| conv_serving_weight_bytes | 13214154752 | 15557140480 |
| disagg_b_eff_4096 | 13 | 14 |
| disagg_transfer_bytes_640 | 335544320 | 293601280 |
| eng_None_32768_kv_mb | 6442.450944 | 15032.385536 |
| eng_None_32768_params | 2357735424 | 6797668352 |
| eng_None_32768_total_mb | 11938.062336 | 29407.862784 |
| eng_None_4096_kv_mb | 805.306368 | 1879.048192 |
| eng_None_4096_params | 2357735424 | 6797668352 |
| eng_None_4096_total_mb | 5910.847488 | 15864.455168 |
| eng_llamacpp_32768_kv_mb | 6442.450944 | 15032.385536 |
| eng_llamacpp_32768_params | 2357735424 | 6797668352 |
| eng_llamacpp_32768_total_mb | 11938.062336 | 29407.862784 |
| eng_llamacpp_4096_kv_mb | 805.306368 | 1879.048192 |
| eng_llamacpp_4096_params | 2357735424 | 6797668352 |
| eng_llamacpp_4096_total_mb | 5910.847488 | 15864.455168 |
| genz_chunked_kv_mb | 2304.0 | 2016.0 |
| genz_decode_kv_mb | 2048.0 | 1792.0 |
| genz_decode_weights_mb | 9850.0 | 13080.0 |
| genz_ffn_kinds | {"Dense": 32} | {"Dense": 28} |
| genz_kv_size | 262144 | 229376 |
| genz_layer_kinds | {"MHA-global": 32} | {"MHA-global": 28} |
| genz_prefill_kv_mb | 2048.0 | 1792.0 |
| genz_prefill_weights_mb | 10100.0 | 13588.0 |
| kv_gb@32768 | 6.442450944 | 15.032385536 |
| kv_gb@4096 | 0.805306368 | 1.879048192 |
| kvgeo_32768 | [458752.0, 1, 0, 0.0, 16, 0.0] | [458752.0, 28, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [458752.0, 1, 0, 0.0, 16, 0.0] | [458752.0, 28, 0, 0.0, 16, 0.0] |
| model_type | "unknown" | "decoder-only" |
| prefix_kv_bytes_per_token | 524288 | 458752 |
| report_kv_gb | 0.805306368 | 1.879048192 |
| report_params | 2357735424 | 6797668352 |
| report_total_gb | 6.149922816 | 16.318278861 |
| report_weight_gb | 4.715470848 | 13.595336704 |
| runtime_chunked | [9.401786, 10100.0, 384.0] | [12.492831, 13588.0, 336.0] |
| runtime_decode | [7.694589, 9850.0, 258.0] | [9.581397, 13080.0, 225.75] |
| runtime_prefill | [20.426598, 10100.0, 256.0] | [26.500875, 13588.0, 224.0] |
| serving_bytes_per_token | 524288 | 458752 |
| serving_requests_8gib_1k | 16 | 18 |
| serving_weight_bytes | 13214154752 | 15557140480 |

## synth:decoder-only-isencdec-false (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 52 | 50 |
| conv_serving_weight_bytes | 15009316864 | 16060522496 |
| disagg_b_eff_4096 | 52 | 50 |
| eng_None_32768_params | 7538479104 | 8063815680 |
| eng_None_32768_total_mb | 20177.231872 | 21227.905024 |
| eng_None_4096_params | 7538479104 | 8063815680 |
| eng_None_4096_total_mb | 16016.482304 | 17067.155456 |
| eng_llamacpp_32768_params | 7538479104 | 8063815680 |
| eng_llamacpp_32768_total_mb | 20177.231872 | 21227.905024 |
| eng_llamacpp_4096_params | 7538479104 | 8063815680 |
| eng_llamacpp_4096_total_mb | 16016.482304 | 17067.155456 |
| report_params | 7538479104 | 8063815680 |
| report_total_gb | 16.204387123 | 17.255060275 |
| report_weight_gb | 15.076958208 | 16.12763136 |
| serving_weight_bytes | 15009316864 | 16060522496 |

## synth:deepseek-v2-lite-mla (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 41 | 90 |
| conv_kv_size | 110592 | 15552 |
| conv_prefill_weights_mb | 5126.75 | 29957.5 |
| conv_serving_weight_bytes | 4956356608 | 31412971776 |
| decode_fit_refused | null | true |
| disagg_b_eff_4096 | 41 | 90 |
| disagg_transfer_bytes_640 | 141557760 | 19906560 |
| genz_chunked_kv_mb | 972.0 | 136.6875 |
| genz_decode_kv_mb | null | 121.5 |
| genz_decode_weights_mb | null | 29557.5 |
| genz_error | "ValueError: cannot convert float NaN to integer" | null |
| genz_ffn_kinds | {"MoE": 27} | {"Dense": 1, "MoE": 26} |
| genz_prefill_error | "ValueError: cannot convert float NaN to integer" | null |
| genz_prefill_kv_mb | null | 121.5 |
| genz_prefill_weights_mb | null | 29957.5 |
| prefix_kv_bytes_per_token | 221184 | 31104 |
| runtime_chunked | [25.484159, 37108.25, 162.0] | [19.466594, 29957.5, 22.781] |
| runtime_decode | null | [9.454812, 29557.5, 15.306] |
| runtime_decode_error | "ValueError: cannot convert float NaN to integer" | null |
| runtime_prefill | null | [24.395485, 29957.5, 15.188] |
| runtime_prefill_error | "ValueError: cannot convert float NaN to integer" | null |
| serving_bytes_per_token | 221184 | 31104 |
| serving_requests_8gib_1k | 37 | 269 |
| serving_weight_bytes | 4956356608 | 31412971776 |

## synth:enc-dec-llama-dec (encoder-decoder)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 304 | 147 |
| conv_kv_size | 131072 | 32768 |
| conv_layers | [16, 16] | [24, 16] |
| conv_prefill_weights_mb | 6676.0 | 3198.0 |
| conv_serving_weight_bytes | 6738149376 | 3349929984 |
| disagg_b_eff_4096 | 304 | 147 |
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
| facts_kv_bytes_4096 | 2147483648 | 268435456 |
| facts_kv_bytes_per_token | 524288 | 32768 |
| facts_layers | [32, 0, 0, null] | [16, 0, 0, null] |
| facts_weight_bytes | 34884558848 | 3349929984 |
| genz_chunked_kv_mb | 144.0 | 288.0 |
| genz_decode_kv_mb | 128.0 | 256.0 |
| genz_decode_weights_mb | 1981.0 | 3073.0 |
| genz_kv_size | 16384 | 32768 |
| genz_prefill_kv_mb | 128.0 | 640.0 |
| genz_prefill_weights_mb | 2106.0 | 3198.0 |
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
| runtime_chunked | [1.934278, 2106.0, 24.0] | [2.202712, 3198.0, 48.0] |
| runtime_decode | [2.478146, 1981.0, 16.125] | [3.030832, 3073.0, 80.125] |
| runtime_prefill | [5.342384, 2106.0, 16.0] | [8.622026, 3198.0, 80.0] |
| serving_bytes_per_token | 32768 | 65536 |
| serving_requests_8gib_1k | 256 | 128 |
| serving_weight_bytes | 2077229056 | 3349929984 |
| sim_max_running | 72 | 400 |
| slice_mib_16x4352 | [81582, false] | [14080, true] |
| vllm_limits | [4702, 75232, 4702, 1, 0, 0.0, true] | [137290, 2196640, 137290, 1, 0, 134217728.0, true] |
| vllm_running_1024 | 58 | 408 |
| vllm_running_8192 | 8 | 175 |

## synth:falcon-40b (decoder-only)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 65741520896 | 115882213376 |
| serving_weight_bytes | 65741520896 | 115882213376 |

## synth:florence2 (multimodal)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 1 | 104 |
| conv_kv_size | 131072 | 49152 |
| conv_layers | [16, 16] | [12, 12] |
| conv_prefill_weights_mb | 6676.0 | 1064.347656 |
| conv_serving_weight_bytes | 6738149376 | 1015760896 |
| decode_fit_refused | null | true |
| disagg_b_eff_4096 | 1 | 104 |
| disagg_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['is_encoder_decoder', 'model_type', 'text_config', 'vision_config']. Looked for ['num_hidden_layers', 'hidden_size'] at t" | null |
| disagg_transfer_bytes_640 | null | 62914560 |
| eng_None_32768_kv_mb | 0.0 | 3221.225472 |
| eng_None_32768_params | 290113024 | 491687424 |
| eng_None_32768_total_mb | 804.21417 | 4428.588442 |
| eng_None_4096_kv_mb | 0.0 | 402.653184 |
| eng_None_4096_params | 290113024 | 491687424 |
| eng_None_4096_total_mb | 693.484544 | 1499.286528 |
| eng_llamacpp_32768_kv_mb | 0.0 | 3221.225472 |
| eng_llamacpp_32768_params | 290113024 | 491687424 |
| eng_llamacpp_32768_total_mb | 804.21417 | 4428.588442 |
| eng_llamacpp_4096_kv_mb | 0.0 | 402.653184 |
| eng_llamacpp_4096_params | 290113024 | 491687424 |
| eng_llamacpp_4096_total_mb | 693.484544 | 1499.286528 |
| facts_kv_bytes_4096 | 201326592 | 402653184 |
| facts_weight_bytes | 1015437312 | 1015760896 |
| genz_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['is_encoder_decoder', 'model_type', 'text_config', 'vision_config']. Looked for ['num_hidden_layers', 'hidden_size'] at t" | null |
| genz_chunked_kv_mb | null | 432.0 |
| genz_decode_kv_mb | null | 384.0 |
| genz_decode_weights_mb | null | 772.173828 |
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
| report_params | 290113024 | 491687424 |
| report_total_gb | 0.716136243 | 1.602468864 |
| report_weight_gb | 0.580226048 | 0.983374848 |
| runtime_chunked | null | [0.540976, 872.348, 72.0] |
| runtime_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['is_encoder_decoder', 'model_type', 'text_config', 'vision_config']. Looked for [" | null |
| runtime_decode | null | [1.754892, 772.174, 120.188] |
| runtime_decode_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['is_encoder_decoder', 'model_type', 'text_config', 'vision_config']. Looked for [" | null |
| runtime_prefill | null | [3.719344, 872.348, 72.0] |
| runtime_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['is_encoder_decoder', 'model_type', 'text_config', 'vision_config']. Looked for [" | null |
| serving_bytes_per_token | null | 98304 |
| serving_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['is_encoder_decoder', 'model_type', 'text_config', 'vision_config']. Looked for ['num_hidden_layers', 'hidden_size'] at t" | null |
| serving_requests_8gib_1k | null | 85 |
| serving_state_bytes | null | 0 |
| serving_weight_bytes | null | 1015760896 |
| serving_weight_bytes_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['is_encoder_decoder', 'model_type', 'text_config', 'vision_config']. Looked for ['num_hidden_layers', 'hidden_size'] at t" | null |
| slice_mib_16x4352 | [10496, true] | [14336, true] |
| vllm_limits | [93844, 1501504, 93844, 1, 0, 0.0, true] | [93843, 1501488, 93843, 1, 0, 50331648.0, true] |
| vllm_running_1024 | 1024 | 651 |
| vllm_running_8192 | 177 | 158 |

## synth:gemma2-9b-hub (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 17 | 16 |
| conv_serving_weight_bytes | 18482200576 | 20317817856 |
| disagg_b_eff_4096 | 17 | 16 |
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
| serving_weight_bytes | 18482200576 | 20317817856 |
| slice_mib_16x4352 | [52992, true] | [52224, true] |
| vllm_limits | [9855, 157680, 9855, 1, 0, 0.0, true] | [19711, 315376, 19711, 1, 1, 0.0, true] |
| vllm_running_8192 | 18 | 25 |

## synth:gguf-swa-full-pattern (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 61 | 59 |
| conv_serving_weight_bytes | 7759462400 | 9101992960 |
| disagg_b_eff_4096 | 61 | 59 |
| eng_None_32768_params | 2988794880 | 3880084480 |
| eng_None_32768_total_mb | 11094.64064 | 12877.21984 |
| eng_None_4096_params | 2988794880 | 3880084480 |
| eng_None_4096_total_mb | 6824.839168 | 8607.418368 |
| eng_llamacpp_32768_params | 2988794880 | 3880084480 |
| eng_llamacpp_32768_total_mb | 11094.64064 | 12877.21984 |
| eng_llamacpp_4096_params | 2988794880 | 3880084480 |
| eng_llamacpp_4096_total_mb | 6824.839168 | 8607.418368 |
| facts_full_attention | true | false |
| facts_kv_bytes_4096 | 570425344 | 205520896 |
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
| report_params | 2988794880 | 3880084480 |
| report_total_gb | 6.99428905 | 8.77686825 |
| report_weight_gb | 5.97758976 | 7.76016896 |
| serving_weight_bytes | 7759462400 | 9101992960 |
| slice_mib_16x4352 | [24832, true] | [17664, true] |
| vllm_limits | [29394, 470304, 29394, 1, 0, 0.0, true] | [199885, 3198160, 199885, 1, 6, 0.0, true] |
| vllm_running_1024 | 367 | 425 |
| vllm_running_8192 | 55 | 217 |

## synth:gguf-swa-short-pattern (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 61 | 59 |
| conv_serving_weight_bytes | 7759462400 | 9101992960 |
| disagg_b_eff_4096 | 61 | 59 |
| eng_None_32768_params | 2988794880 | 3880084480 |
| eng_None_32768_total_mb | 11094.64064 | 12877.21984 |
| eng_None_4096_params | 2988794880 | 3880084480 |
| eng_None_4096_total_mb | 6824.839168 | 8607.418368 |
| eng_llamacpp_32768_params | 2988794880 | 3880084480 |
| eng_llamacpp_32768_total_mb | 11094.64064 | 12877.21984 |
| eng_llamacpp_4096_params | 2988794880 | 3880084480 |
| eng_llamacpp_4096_total_mb | 6824.839168 | 8607.418368 |
| facts_full_attention | true | false |
| facts_kv_bytes_4096 | 570425344 | 507510784 |
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
| report_params | 2988794880 | 3880084480 |
| report_total_gb | 6.99428905 | 8.77686825 |
| report_weight_gb | 5.97758976 | 7.76016896 |
| serving_weight_bytes | 7759462400 | 9101992960 |
| slice_mib_16x4352 | [24832, true] | [24064, true] |
| vllm_limits | [29394, 470304, 29394, 1, 0, 0.0, true] | [199885, 3198160, 33314, 6, 1, 0.0, true] |
| vllm_running_1024 | 367 | 366 |
| vllm_running_8192 | 55 | 61 |

## synth:gpt2 (unknown)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 13 | 281 |
| conv_kv_size | 262144 | 18432 |
| conv_layers | [0, 32] | [0, 12] |
| conv_prefill_weights_mb | 13137.265625 | 783.237305 |
| conv_serving_weight_bytes | 13363716096 | 380920320 |
| disagg_b_eff_4096 | 13 | 281 |
| disagg_transfer_bytes_640 | 335544320 | 23592960 |
| eng_None_32768_params | 123570432 | 151881984 |
| eng_None_32768_total_mb | 1621.194854 | 1677.817958 |
| eng_None_4096_params | 123570432 | 151881984 |
| eng_None_4096_total_mb | 481.183027 | 537.806131 |
| eng_llamacpp_32768_params | 123570432 | 151881984 |
| eng_llamacpp_32768_total_mb | 1621.194854 | 1677.817958 |
| eng_llamacpp_4096_params | 123570432 | 151881984 |
| eng_llamacpp_4096_total_mb | 481.183027 | 537.806131 |
| genz_chunked_kv_mb | 2304.0 | 162.0 |
| genz_decode_kv_mb | 2048.0 | 144.0 |
| genz_decode_weights_mb | 9992.632812 | 514.618652 |
| genz_ffn_kinds | {"Dense": 32} | {"Dense": 12} |
| genz_kv_size | 262144 | 18432 |
| genz_layer_kinds | {"MHA-global": 32} | {"MHA-global": 12} |
| genz_prefill_kv_mb | 2048.0 | 144.0 |
| genz_prefill_weights_mb | 10385.265625 | 589.737305 |
| kv_full_layers | 0 | 12 |
| kvgeo_32768 | [0.0, 12, 0, 0.0, 16, 0.0] | [36864.0, 12, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [0.0, 12, 0, 0.0, 16, 0.0] | [36864.0, 12, 0, 0.0, 16, 0.0] |
| prefix_kv_bytes_per_token | 524288 | 36864 |
| report_params | 123570432 | 151881984 |
| report_total_gb | 0.52799146 | 0.584614564 |
| report_weight_gb | 0.247140864 | 0.303763968 |
| runtime_chunked | [9.657536, 10385.266, 384.0] | [0.553922, 589.737, 27.0] |
| runtime_decode | [7.78173, 9992.633, 258.0] | [1.520302, 514.619, 18.141] |
| runtime_prefill | [20.937101, 10385.266, 256.0] | [2.457114, 589.737, 18.0] |
| serving_bytes_per_token | 524288 | 36864 |
| serving_requests_8gib_1k | 16 | 227 |
| serving_weight_bytes | 13363716096 | 380920320 |

## synth:instructblip-flan-t5-xl (multimodal)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 282 | 22 |
| conv_kv_size | 131072 | 196608 |
| conv_layers | [16, 16] | [24, 24] |
| conv_prefill_weights_mb | 6676.0 | 5435.0 |
| conv_serving_weight_bytes | 6738149376 | 7295498496 |
| disagg_b_eff_4096 | 282 | 22 |
| disagg_transfer_bytes_640 | 23592960 | 251658240 |
| eng_None_32768_kv_mb | 0.0 | 12884.901888 |
| eng_None_32768_params | 1663986176 | 3391313664 |
| eng_None_32768_total_mb | 3656.150426 | 19995.70729 |
| eng_None_4096_kv_mb | 0.0 | 1610.612736 |
| eng_None_4096_params | 1663986176 | 3391313664 |
| eng_None_4096_total_mb | 3495.089152 | 8560.356864 |
| eng_llamacpp_32768_kv_mb | 0.0 | 12884.901888 |
| eng_llamacpp_32768_params | 1663986176 | 3391313664 |
| eng_llamacpp_32768_total_mb | 3656.150426 | 19995.70729 |
| eng_llamacpp_4096_kv_mb | 0.0 | 1610.612736 |
| eng_llamacpp_4096_params | 1663986176 | 3391313664 |
| eng_llamacpp_4096_total_mb | 3495.089152 | 8560.356864 |
| facts_kv_bytes_4096 | 805306368 | 1610612736 |
| facts_weight_bytes | 8572457984 | 7295498496 |
| genz_chunked_kv_mb | 162.0 | 1728.0 |
| genz_decode_kv_mb | 144.0 | 1536.0 |
| genz_decode_weights_mb | 208.875 | 4349.5 |
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
| report_params | 1663986176 | 3391313664 |
| report_total_gb | 3.528512512 | 8.915902771 |
| report_weight_gb | 3.327972352 | 6.782627328 |
| runtime_chunked | [0.254007, 255.75, 27.0] | [2.356801, 4475.0, 288.0] |
| runtime_decode | [1.333507, 208.875, 18.141] | [3.49746, 4349.5, 480.75] |
| runtime_prefill | [1.859421, 255.75, 18.0] | [11.571193, 4475.0, 288.0] |
| serving_bytes_per_token | 36864 | 393216 |
| serving_requests_8gib_1k | 227 | 21 |
| serving_weight_bytes | 275644416 | 7295498496 |
| sim_max_running | 322 | 67 |
| slice_mib_16x4352 | [30208, true] | [44288, true] |
| vllm_limits | [21221, 339536, 21221, 1, 0, 0.0, true] | [21627, 346032, 21627, 1, 0, 805306368.0, true] |
| vllm_running_1024 | 265 | 64 |
| vllm_running_8192 | 40 | 27 |

## synth:llama3-8b (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 52 | 50 |
| conv_serving_weight_bytes | 15009316864 | 16060522496 |
| disagg_b_eff_4096 | 52 | 50 |
| eng_None_32768_params | 7538479104 | 8063815680 |
| eng_None_32768_total_mb | 20177.231872 | 21227.905024 |
| eng_None_4096_params | 7538479104 | 8063815680 |
| eng_None_4096_total_mb | 16016.482304 | 17067.155456 |
| eng_llamacpp_32768_params | 7538479104 | 8063815680 |
| eng_llamacpp_32768_total_mb | 20177.231872 | 21227.905024 |
| eng_llamacpp_4096_params | 7538479104 | 8063815680 |
| eng_llamacpp_4096_total_mb | 16016.482304 | 17067.155456 |
| report_params | 7538479104 | 8063815680 |
| report_total_gb | 16.204387123 | 17.255060275 |
| report_weight_gb | 15.076958208 | 16.12763136 |
| serving_weight_bytes | 15009316864 | 16060522496 |

## synth:llama4-maverick-text (unknown)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 28 | 1 |
| conv_kv_size | 262144 | 98304 |
| conv_layers | [0, 32] | [0, 48] |
| conv_prefill_weights_mb | 12852.0 | 764296.25 |
| conv_serving_weight_bytes | 13214154752 | 1568509483008 |
| disagg_b_eff_4096 | 28 | 1 |
| eng_None_32768_params | 151881984 | 400712345600 |
| eng_None_32768_total_mb | 7370.327347 | 808491.254579 |
| eng_None_4096_params | 151881984 | 400712345600 |
| eng_None_4096_total_mb | 1421.126554 | 802542.053786 |
| eng_llamacpp_32768_params | 151881984 | 400712345600 |
| eng_llamacpp_32768_total_mb | 7370.327347 | 808491.254579 |
| eng_llamacpp_4096_params | 151881984 | 400712345600 |
| eng_llamacpp_4096_total_mb | 1421.126554 | 802542.053786 |
| facts_weight_bytes | 1556429887488 | 1568509483008 |
| genz_decode_weights_mb | 990833.125 | 762323.125 |
| genz_ffn_kinds | {"MoE": 48} | {"Dense": 24, "MoE": 24} |
| genz_prefill_weights_mb | 992806.25 | 764296.25 |
| report_params | 151881984 | 400712345600 |
| report_total_gb | 1.644599071 | 802.765526303 |
| report_weight_gb | 0.303763968 | 801.4246912 |
| runtime_chunked_error | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 969.537353515625 GB, KV Cache:0.140625. \n System:{'Flops': 400, 'Memory_size': 200, 'Me" | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 746.383056640625 GB, KV Cache:0.140625. \n System:{'Flops': 400, 'Memory_size': 200, 'Me" |
| runtime_decode_error | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 967.6104736328125 GB, KV Cache:0.140625. \n System:{'Flops': 400, 'Memory_size': 200, 'M" | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 744.4561767578125 GB, KV Cache:0.140625. \n System:{'Flops': 400, 'Memory_size': 200, 'M" |
| runtime_prefill_error | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 969.537353515625 GB, KV Cache:0.09375. \n System:{'Flops': 400, 'Memory_size': 200, 'Mem" | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 746.383056640625 GB, KV Cache:0.09375. \n System:{'Flops': 400, 'Memory_size': 200, 'Mem" |
| serving_weight_bytes | 20188364800 | 1568509483008 |

## synth:llama4-scout-text (unknown)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 28 | 1 |
| conv_kv_size | 262144 | 98304 |
| conv_layers | [0, 32] | [0, 48] |
| conv_prefill_weights_mb | 12852.0 | 205553.75 |
| conv_serving_weight_bytes | 13214154752 | 215539723776 |
| decode_fit_refused | true | null |
| disagg_b_eff_4096 | 28 | 1 |
| eng_None_32768_params | 151881984 | 107770357760 |
| eng_None_32768_total_mb | 7370.327347 | 222607.278899 |
| eng_None_4096_params | 151881984 | 107770357760 |
| eng_None_4096_total_mb | 1421.126554 | 216658.078106 |
| eng_llamacpp_32768_params | 151881984 | 107770357760 |
| eng_llamacpp_32768_total_mb | 7370.327347 | 222607.278899 |
| eng_llamacpp_4096_params | 151881984 | 107770357760 |
| eng_llamacpp_4096_total_mb | 1421.126554 | 216658.078106 |
| facts_weight_bytes | 203460128256 | 215539723776 |
| genz_decode_weights_mb | 130620.625 | 203580.625 |
| genz_prefill_weights_mb | 132593.75 | 205553.75 |
| report_params | 151881984 | 107770357760 |
| report_total_gb | 1.644599071 | 216.881550623 |
| report_weight_gb | 0.303763968 | 215.54071552 |
| runtime_chunked | [84.301146, 132593.75, 144.0] | null |
| runtime_chunked_error | null | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 200.736083984375 GB, KV Cache:0.140625. \n System:{'Flops': 400, 'Memory_size': 200, 'Me" |
| runtime_decode | [25.60036, 130620.625, 96.75] | [42.200289, 203580.625, 96.75] |
| runtime_prefill | [96.180362, 132593.75, 96.0] | null |
| runtime_prefill_error | null | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 200.736083984375 GB, KV Cache:0.09375. \n System:{'Flops': 400, 'Memory_size': 200, 'Mem" |
| serving_weight_bytes | 20188364800 | 215539723776 |

## synth:marian-noflag (encoder-decoder)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 13 | 424 |
| conv_kv_size | 262144 | 12288 |
| conv_layers | [0, 32] | [6, 6] |
| conv_prefill_weights_mb | 13259.828125 | 221.478516 |
| conv_serving_weight_bytes | 13427974144 | 169335808 |
| disagg_b_eff_4096 | 13 | 424 |
| disagg_transfer_bytes_640 | 335544320 | 15728640 |
| eng_None_32768_kv_mb | 439.517184 | 823.738368 |
| eng_None_32768_total_mb | 701.818266 | 1086.03945 |
| eng_None_4096_kv_mb | 87.195648 | 119.095296 |
| eng_None_4096_total_mb | 294.131917 | 326.031565 |
| eng_llamacpp_32768_kv_mb | 439.517184 | 823.738368 |
| eng_llamacpp_32768_total_mb | 701.818266 | 1086.03945 |
| eng_llamacpp_4096_kv_mb | 87.195648 | 119.095296 |
| eng_llamacpp_4096_total_mb | 294.131917 | 326.031565 |
| facts_kv_bytes_4096 | 50331648 | 100663296 |
| genz_chunked_kv_mb | 2304.0 | 108.0 |
| genz_decode_kv_mb | 2048.0 | 96.0 |
| genz_decode_weights_mb | 10053.914062 | 140.739258 |
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
| runtime_chunked | [9.767417, 10507.828, 384.0] | [0.160886, 197.479, 18.0] |
| runtime_decode | [7.81917, 10053.914, 258.0] | [1.286146, 140.739, 30.047] |
| runtime_prefill | [21.156435, 10507.828, 256.0] | [1.905954, 197.479, 18.0] |
| serving_bytes_per_token | 524288 | 24576 |
| serving_requests_8gib_1k | 16 | 341 |
| serving_weight_bytes | 13427974144 | 169335808 |
| slice_mib_16x4352 | [6656, true] | [7680, true] |
| vllm_limits | [380038, 6080608, 380038, 1, 0, 0.0, true] | [380038, 6080608, 380038, 1, 0, 50331648.0, true] |
| vllm_running_8192 | 719 | 484 |

## synth:mistral-7b-window (decoder-only)

| figure | before | after |
|---|---|---|
| conv_serving_weight_bytes | 14220787712 | 14483464192 |
| eng_None_32768_params | 7144214528 | 7275286528 |
| eng_None_32768_total_mb | 15630.606336 | 15892.750336 |
| eng_None_4096_params | 7144214528 | 7275286528 |
| eng_None_4096_total_mb | 15227.953152 | 15490.097152 |
| eng_llamacpp_32768_params | 7144214528 | 7275286528 |
| eng_llamacpp_32768_total_mb | 15630.606336 | 15892.750336 |
| eng_llamacpp_4096_params | 7144214528 | 7275286528 |
| eng_llamacpp_4096_total_mb | 15227.953152 | 15490.097152 |
| report_params | 7144214528 | 7275286528 |
| report_total_gb | 15.415857971 | 15.678001971 |
| report_weight_gb | 14.288429056 | 14.550573056 |
| serving_weight_bytes | 14220787712 | 14483464192 |

## synth:mllama-own-cross-layers (multimodal)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 495 | 558 |
| conv_kv_size | 262144 | 8192 |
| conv_layers | [0, 32] | [0, 10] |
| conv_prefill_weights_mb | 12852.0 | 1410.0 |
| conv_serving_weight_bytes | 13214154752 | 3112028672 |
| conv_state_bytes | 0 | 4198400 |
| disagg_b_eff_4096 | 495 | 558 |
| disagg_transfer_bytes_640 | 13107200 | 14684160 |
| eng_None_32768_kv_mb | 671.08864 | 541.069312 |
| eng_None_32768_params | 1137317376 | 1370625536 |
| eng_None_32768_total_mb | 3410.66199 | 3747.258982 |
| eng_None_4096_kv_mb | 83.88608 | 71.307264 |
| eng_None_4096_params | 1137317376 | 1370625536 |
| eng_None_4096_total_mb | 2602.000179 | 3056.037683 |
| eng_llamacpp_32768_kv_mb | 671.08864 | 541.069312 |
| eng_llamacpp_32768_params | 1137317376 | 1370625536 |
| eng_llamacpp_32768_total_mb | 3410.66199 | 3747.258982 |
| eng_llamacpp_4096_kv_mb | 83.88608 | 71.307264 |
| eng_llamacpp_4096_params | 1137317376 | 1370625536 |
| eng_llamacpp_4096_total_mb | 2602.000179 | 3056.037683 |
| facts_kv_bytes_4096 | 83886080 | 71307264 |
| facts_kv_bytes_per_token | 20480 | 16384 |
| facts_layers | [10, 0, 0, null] | [8, 0, 0, null] |
| facts_weight_bytes | 2747973632 | 3112028672 |
| genz_chunked_kv_mb | 90.0 | 80.007812 |
| genz_decode_kv_mb | 80.0 | 68.003906 |
| genz_decode_weights_mb | 965.0 | 1285.0 |
| genz_kv_size | 10240 | 8192 |
| genz_layer_kinds | {"MHA-global": 10} | {"Cross": 2, "MHA-global": 8} |
| genz_prefill_kv_mb | 80.0 | 68.003906 |
| genz_prefill_weights_mb | 1090.0 | 1410.0 |
| kv_cross_layers | 0 | 2 |
| kv_full_layers | 10 | 8 |
| kv_gb@32768 | 0.67108864 | 0.541069312 |
| kv_gb@4096 | 0.08388608 | 0.071307264 |
| kv_source | "uniform" | "layer_plan:cross_attention_layers" |
| kvgeo_32768 | [20480.0, 10, 0, 0.0, 16, 0.0] | [16384.0, 8, 0, 0.0, 16, 4198400.0] |
| kvgeo_4096 | [20480.0, 10, 0, 0.0, 16, 0.0] | [16384.0, 8, 0, 0.0, 16, 4198400.0] |
| prefix_kv_bytes_per_token | 20480 | 16384 |
| report_kv_gb | 0.08388608 | 0.071307264 |
| report_params | 1137317376 | 1370625536 |
| report_total_gb | 2.667473265 | 3.118995005 |
| report_weight_gb | 2.274634752 | 2.741251072 |
| runtime_chunked | [1.00279, 1090.0, 15.0] | [1.295017, 1410.0, 20.008] |
| runtime_decode | [1.757883, 965.0, 10.078] | [1.952919, 1285.0, 24.078] |
| runtime_prefill | [3.301999, 1090.0, 10.0] | [3.919394, 1410.0, 12.004] |
| serving_bytes_per_token | 20480 | 16384 |
| serving_state_bytes | 0 | 4198400 |
| serving_weight_bytes | 1347420160 | 3112028672 |
| vllm_limits | [221502, 3544032, 221502, 1, 0, 0.0, true] | [275488, 4407808, 275488, 1, 0, 4198400.0, true] |
| vllm_running_8192 | 419 | 506 |

## synth:mllama-own-cross-layers-no-tower-geometry (multimodal)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 495 | 474 |
| conv_kv_size | 262144 | 10240 |
| conv_layers | [0, 32] | [0, 10] |
| conv_prefill_weights_mb | 12852.0 | 1410.0 |
| conv_serving_weight_bytes | 13214154752 | 3183014912 |
| disagg_b_eff_4096 | 495 | 474 |
| eng_None_32768_params | 1136487936 | 1369796096 |
| eng_None_32768_total_mb | 3391.197798 | 3857.814118 |
| eng_None_4096_params | 1136487936 | 1369796096 |
| eng_None_4096_total_mb | 2582.535987 | 3049.152307 |
| eng_llamacpp_32768_params | 1136487936 | 1369796096 |
| eng_llamacpp_32768_total_mb | 3391.197798 | 3857.814118 |
| eng_llamacpp_4096_params | 1136487936 | 1369796096 |
| eng_llamacpp_4096_total_mb | 2582.535987 | 3049.152307 |
| facts_kv_bytes_per_token | 20480 | 16384 |
| facts_layers | [10, 0, 0, null] | [8, 0, 0, null] |
| facts_weight_bytes | 2747973632 | 3183014912 |
| genz_decode_weights_mb | 965.0 | 1285.0 |
| genz_layer_kinds | {"MHA-global": 10} | {"Cross": 2, "MHA-global": 8} |
| genz_prefill_weights_mb | 1090.0 | 1410.0 |
| kv_cross_layers | 0 | 2 |
| kv_full_layers | 10 | 8 |
| kv_source | "uniform" | "layer_plan:cross_attention_layers" |
| kvgeo_32768 | [20480.0, 10, 0, 0.0, 16, 0.0] | [20480.0, 8, 0, 0.0, 16, 0.0] |
| kvgeo_4096 | [20480.0, 10, 0, 0.0, 16, 0.0] | [20480.0, 8, 0, 0.0, 16, 0.0] |
| report_params | 1136487936 | 1369796096 |
| report_total_gb | 2.64444801 | 3.11106433 |
| report_weight_gb | 2.272975872 | 2.739592192 |
| runtime_chunked | [1.00279, 1090.0, 15.0] | [1.283038, 1410.0, 15.0] |
| runtime_decode | [1.757883, 965.0, 10.078] | [1.948316, 1285.0, 16.062] |
| runtime_prefill | [3.301999, 1090.0, 10.0] | [3.890704, 1410.0, 10.0] |
| serving_weight_bytes | 1347420160 | 3183014912 |
| slice_mib_16x4352 | [8704, true] | [9216, true] |
| vllm_limits | [221502, 3544032, 221502, 1, 0, 0.0, true] | [275218, 4403488, 275218, 1, 0, 16777216.0, true] |
| vllm_running_8192 | 419 | 464 |

## synth:mpt-7b (unknown)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 13 | 11 |
| conv_serving_weight_bytes | 13365149696 | 18006679552 |
| disagg_b_eff_4096 | 13 | 11 |
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
| serving_weight_bytes | 13365149696 | 18006679552 |

## synth:mt5-noflag (encoder-decoder)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 13 | 421 |
| conv_kv_size | 262144 | 12288 |
| conv_layers | [0, 32] | [8, 8] |
| conv_prefill_weights_mb | 16260.0 | 572.5 |
| conv_serving_weight_bytes | 15000928256 | 549995520 |
| disagg_b_eff_4096 | 13 | 421 |
| disagg_transfer_bytes_640 | 335544320 | 15728640 |
| eng_None_32768_kv_mb | 0.0 | 829.882368 |
| eng_None_32768_params | 153052160 | 247492608 |
| eng_None_32768_total_mb | 376.568627 | 1395.331891 |
| eng_None_4096_kv_mb | 0.0 | 125.239296 |
| eng_None_4096_params | 153052160 | 247492608 |
| eng_None_4096_total_mb | 341.336474 | 655.456666 |
| eng_llamacpp_32768_kv_mb | 0.0 | 829.882368 |
| eng_llamacpp_32768_params | 153052160 | 247492608 |
| eng_llamacpp_32768_total_mb | 376.568627 | 1395.331891 |
| eng_llamacpp_4096_kv_mb | 0.0 | 125.239296 |
| eng_llamacpp_4096_params | 153052160 | 247492608 |
| eng_llamacpp_4096_total_mb | 341.336474 | 655.456666 |
| facts_kv_bytes_4096 | 67108864 | 100663296 |
| facts_kv_bytes_per_token | 16384 | 12288 |
| facts_weight_bytes | 579355648 | 549995520 |
| genz_chunked_kv_mb | 2304.0 | 108.0 |
| genz_decode_kv_mb | 2048.0 | 96.0 |
| genz_decode_weights_mb | 11554.0 | 312.25 |
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
| model_type | "unknown" | "encoder-decoder" |
| prefix_kv_bytes_per_token | 524288 | 24576 |
| report_kv_gb | 0.0 | 0.125239296 |
| report_params | 153052160 | 247492608 |
| report_total_gb | 0.348382904 | 0.688626156 |
| report_weight_gb | 0.30610432 | 0.494985216 |
| runtime_chunked | [12.457169, 13508.0, 384.0] | [0.498344, 556.5, 18.0] |
| runtime_decode | [8.735644, 11554.0, 258.0] | [1.464612, 312.25, 30.047] |
| runtime_prefill | [26.525452, 13508.0, 256.0] | [2.720889, 556.5, 18.0] |
| serving_bytes_per_token | 524288 | 24576 |
| serving_requests_8gib_1k | 16 | 341 |
| serving_weight_bytes | 15000928256 | 549995520 |
| slice_mib_16x4352 | [7424, true] | [8192, true] |
| vllm_limits | [281214, 4499424, 281214, 1, 0, 0.0, true] | [375102, 6001632, 375102, 1, 0, 50331648.0, true] |
| vllm_running_8192 | 532 | 478 |

## synth:musicgen-small (encoder-decoder)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 104 | 51 |
| conv_kv_size | 131072 | 98304 |
| conv_layers | [16, 16] | [12, 24] |
| conv_prefill_weights_mb | 6676.0 | 1260.5625 |
| conv_serving_weight_bytes | 6738149376 | 1185233920 |
| disagg_b_eff_4096 | 104 | 51 |
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
| facts_kv_bytes_4096 | 2147483648 | 805306368 |
| facts_kv_bytes_per_token | 524288 | 98304 |
| facts_layers | [32, 0, 0, null] | [24, 0, 0, null] |
| facts_weight_bytes | 34884558848 | 1185233920 |
| genz_chunked_kv_mb | 432.0 | 864.0 |
| genz_decode_kv_mb | 384.0 | 768.0 |
| genz_decode_weights_mb | 580.0 | 994.5625 |
| genz_kv_size | 49152 | 98304 |
| genz_prefill_kv_mb | 384.0 | 912.0 |
| genz_prefill_weights_mb | 584.0 | 1014.5625 |
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
| runtime_chunked | [0.589743, 584.0, 72.0] | [0.754985, 1014.562, 144.0] |
| runtime_decode | [1.771665, 580.0, 48.375] | [2.302822, 994.562, 240.375] |
| runtime_prefill | [2.859744, 584.0, 48.0] | [4.634156, 1014.562, 114.0] |
| serving_bytes_per_token | 98304 | 196608 |
| serving_requests_8gib_1k | 85 | 42 |
| serving_weight_bytes | 809500672 | 1185233920 |
| sim_max_running | 72 | 242 |
| slice_mib_16x4352 | [81582, false] | [22272, true] |
| vllm_limits | [4702, 75232, 4702, 1, 0, 0.0, true] | [46910, 750560, 46910, 1, 0, 201326592.0, true] |
| vllm_running_1024 | 58 | 225 |
| vllm_running_8192 | 8 | 71 |

## synth:nemotronh-list-depth (hybrid)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 1 | 133 |
| conv_kv_size | 65536 | 8192 |
| conv_layers | [0, 32] | [0, 52] |
| conv_prefill_weights_mb | 20736.0 | 19546.8125 |
| conv_serving_weight_bytes | 20669530112 | 20429662208 |
| conv_state_bytes | 0 | 102137856 |
| decode_fit_refused | null | true |
| disagg_b_eff_4096 | 1 | 133 |
| disagg_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'attention_bias', 'attention_dropout', 'attention_head_dim', 'auto_map', 'bos_token_id', 'chunk_size', '" | null |
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
| facts_kv_bytes_4096 | 536870912 | 67108864 |
| facts_kv_bytes_per_token | 131072 | 16384 |
| facts_layers | [32, 0, 0, null] | [4, 0, 24, null] |
| facts_state_bytes | 0 | 102137856 |
| facts_weight_bytes | 21743804416 | 20429662208 |
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
| genz_prefill_weights_mb | null | 15514.8125 |
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
| runtime_chunked | null | [14.362543, 15514.812, 206.812] |
| runtime_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'attention_bias', 'attention_dropout', 'attention_head_dim', 'au" | null |
| runtime_decode | null | [10.340875, 14426.812, 397.688] |
| runtime_decode_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'attention_bias', 'attention_dropout', 'attention_head_dim', 'au" | null |
| runtime_prefill | null | [30.130069, 15514.812, 105.406] |
| runtime_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'attention_bias', 'attention_dropout', 'attention_head_dim', 'au" | null |
| serving_bytes_per_token | null | 16384 |
| serving_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'attention_bias', 'attention_dropout', 'attention_head_dim', 'auto_map', 'bos_token_id', 'chunk_size', '" | null |
| serving_requests_8gib_1k | null | 72 |
| serving_state_bytes | null | 102137856 |
| serving_weight_bytes | null | 20429662208 |
| serving_weight_bytes_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'attention_bias', 'attention_dropout', 'attention_head_dim', 'auto_map', 'bos_token_id', 'chunk_size', '" | null |
| sim_max_running | 384 | 400 |
| slice_mib_16x4352 | [37376, true] | [32512, true] |
| state_gb | 0.025534464 | 0.102137856 |
| vllm_limits | [25363, 405808, 25363, 1, 0, 0.0, true] | [3176, 3303040, 3176, 1, 0, 102236160.0, true] |
| vllm_running_1024 | 317 | 397 |
| vllm_running_8192 | 48 | 211 |

## synth:nemotronh-list-depth-nested (hybrid)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 1 | 133 |
| conv_kv_size | 262144 | 8192 |
| conv_layers | [0, 32] | [0, 52] |
| conv_prefill_weights_mb | 12852.0 | 19546.8125 |
| conv_serving_weight_bytes | 13214154752 | 20429662208 |
| conv_state_bytes | 0 | 102137856 |
| decode_fit_refused | null | true |
| disagg_b_eff_4096 | 1 | 133 |
| disagg_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['model_type', 'text_config']. Looked for ['num_hidden_layers', 'hidden_size'] at the top level and in nested blocks. Refu" | null |
| disagg_transfer_bytes_640 | null | 112623616 |
| eng_None_32768_kv_mb | 1610.612736 | 536.870912 |
| eng_None_32768_params | 123570432 | 8134394880 |
| eng_None_32768_state_mb | 0.0 | 102.137856 |
| eng_None_32768_total_mb | 2944.917197 | 17994.962125 |
| eng_None_4096_kv_mb | 201.326592 | 67.108864 |
| eng_None_4096_params | 123570432 | 8134394880 |
| eng_None_4096_state_mb | 0.0 | 102.137856 |
| eng_None_4096_total_mb | 992.049254 | 16981.618278 |
| eng_llamacpp_32768_kv_mb | 1610.612736 | 536.870912 |
| eng_llamacpp_32768_params | 123570432 | 8134394880 |
| eng_llamacpp_32768_state_mb | 0.0 | 102.137856 |
| eng_llamacpp_32768_total_mb | 2944.917197 | 17994.962125 |
| eng_llamacpp_4096_kv_mb | 201.326592 | 67.108864 |
| eng_llamacpp_4096_params | 123570432 | 8134394880 |
| eng_llamacpp_4096_state_mb | 0.0 | 102.137856 |
| eng_llamacpp_4096_total_mb | 992.049254 | 16981.618278 |
| facts_full_attention | true | false |
| facts_kv_bytes_4096 | 536870912 | 67108864 |
| facts_kv_bytes_per_token | 131072 | 16384 |
| facts_layers | [32, 0, 0, null] | [4, 0, 24, null] |
| facts_state_bytes | 0 | 102137856 |
| facts_weight_bytes | 21743804416 | 20429662208 |
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
| genz_prefill_weights_mb | null | 15514.8125 |
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
| report_params | 123570432 | 8134394880 |
| report_state_gb | 0.0 | 0.102137856 |
| report_total_gb | 1.141030932 | 17.124183982 |
| report_weight_gb | 0.247140864 | 16.26878976 |
| runtime_chunked | null | [14.362543, 15514.812, 206.812] |
| runtime_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['model_type', 'text_config']. Looked for ['num_hidden_layers', 'hidden_size'] at " | null |
| runtime_decode | null | [10.340875, 14426.812, 397.688] |
| runtime_decode_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['model_type', 'text_config']. Looked for ['num_hidden_layers', 'hidden_size'] at " | null |
| runtime_prefill | null | [30.130069, 15514.812, 105.406] |
| runtime_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['model_type', 'text_config']. Looked for ['num_hidden_layers', 'hidden_size'] at " | null |
| serving_bytes_per_token | null | 16384 |
| serving_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['model_type', 'text_config']. Looked for ['num_hidden_layers', 'hidden_size'] at the top level and in nested blocks. Refu" | null |
| serving_requests_8gib_1k | null | 72 |
| serving_state_bytes | null | 102137856 |
| serving_weight_bytes | null | 20429662208 |
| serving_weight_bytes_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['model_type', 'text_config']. Looked for ['num_hidden_layers', 'hidden_size'] at the top level and in nested blocks. Refu" | null |
| sim_max_running | 384 | 400 |
| slice_mib_16x4352 | [37376, true] | [32512, true] |
| state_gb | 0.0 | 0.102137856 |
| vllm_limits | [25363, 405808, 25363, 1, 0, 0.0, true] | [3176, 3303040, 3176, 1, 0, 102236160.0, true] |
| vllm_running_1024 | 317 | 397 |
| vllm_running_8192 | 48 | 211 |

## synth:nllb-moe (encoder-decoder)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 12 | 1 |
| conv_prefill_weights_mb | 799675.21875 | 301533.609375 |
| conv_serving_weight_bytes | 11812847616 | 315377092608 |
| decode_fit_refused | null | true |
| disagg_b_eff_4096 | 12 | 1 |
| disagg_transfer_bytes_640 | 335544320 | 251658240 |
| eng_None_32768_kv_mb | 7032.274944 | 13179.813888 |
| eng_None_32768_total_mb | 14196.983398 | 20344.522342 |
| eng_None_4096_kv_mb | 1395.130368 | 1905.524736 |
| eng_None_4096_total_mb | 8338.379571 | 8848.773939 |
| eng_llamacpp_32768_kv_mb | 7032.274944 | 13179.813888 |
| eng_llamacpp_32768_total_mb | 14196.983398 | 20344.522342 |
| eng_llamacpp_4096_kv_mb | 1395.130368 | 1905.524736 |
| eng_llamacpp_4096_total_mb | 8338.379571 | 8848.773939 |
| facts_kv_bytes_4096 | 805306368 | 1610612736 |
| facts_weight_bytes | 315375806464 | 315377092608 |
| genz_chunked_kv_mb | 2304.0 | 1728.0 |
| genz_decode_kv_mb | 2048.0 | 1536.0 |
| genz_decode_weights_mb | 710641.609375 | 201460.804688 |
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
| runtime_chunked | null | [123.312477, 202461.609, 288.0] |
| runtime_chunked_error | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 695.9406433105469 GB, KV Cache:0.375. \n System:{'Flops': 400, 'Memory_size': 200, 'Memo" | null |
| runtime_decode | null | [11.099606, 201460.805, 480.75] |
| runtime_decode_error | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 693.9859466552734 GB, KV Cache:0.375. \n System:{'Flops': 400, 'Memory_size': 200, 'Memo" | null |
| runtime_prefill | null | [134.618911, 202461.609, 288.0] |
| runtime_prefill_error | "ValueError: All params would not fit on chip. System Memory Cap:200.0 GB , Weights : 695.9406433105469 GB, KV Cache:0.25. \n System:{'Flops': 400, 'Memory_size': 200, 'Memor" | null |
| serving_bytes_per_token | 524288 | 393216 |
| serving_requests_8gib_1k | 16 | 21 |
| serving_weight_bytes | 15050850304 | 315377092608 |
| vllm_limits | [0, 0, 0, 1, 0, 0.0, false] | [0, 0, 0, 1, 0, 805306368.0, false] |

## synth:pix2struct-seq-len (multimodal)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 1 | 88 |
| conv_kv_size | 131072 | 36864 |
| conv_layers | [16, 16] | [18, 18] |
| conv_prefill_weights_mb | 6676.0 | 1785.518555 |
| conv_serving_weight_bytes | 6738149376 | 2847540736 |
| conv_state_bytes | 0 | 150994944 |
| decode_fit_refused | null | true |
| disagg_b_eff_4096 | 1 | 88 |
| disagg_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['is_encoder_decoder', 'model_type', 'text_config', 'vision_config']. Looked for ['num_hidden_layers', 'hidden_size'] at t" | null |
| disagg_transfer_bytes_640 | null | 198180864 |
| eng_None_32768_kv_mb | 0.0 | 2566.914048 |
| eng_None_32768_params | 713446400 | 917650688 |
| eng_None_32768_total_mb | 1603.078144 | 4578.400768 |
| eng_None_4096_kv_mb | 0.0 | 452.984832 |
| eng_None_4096_params | 713446400 | 917650688 |
| eng_None_4096_total_mb | 1517.514342 | 2378.90775 |
| eng_llamacpp_32768_kv_mb | 0.0 | 2566.914048 |
| eng_llamacpp_32768_params | 713446400 | 917650688 |
| eng_llamacpp_32768_total_mb | 1603.078144 | 4578.400768 |
| eng_llamacpp_4096_kv_mb | 0.0 | 452.984832 |
| eng_llamacpp_4096_params | 713446400 | 917650688 |
| eng_llamacpp_4096_total_mb | 1517.514342 | 2378.90775 |
| facts_kv_bytes_4096 | 301989888 | 452984832 |
| facts_weight_bytes | 2443454464 | 2847540736 |
| genz_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['is_encoder_decoder', 'model_type', 'text_config', 'vision_config']. Looked for ['num_hidden_layers', 'hidden_size'] at t" | null |
| genz_chunked_kv_mb | null | 612.0 |
| genz_decode_kv_mb | null | 432.0 |
| genz_decode_weights_mb | null | 1687.385742 |
| genz_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['is_encoder_decoder', 'model_type', 'text_config', 'vision_config']. Looked for ['num_hidden_layers', 'hidden_size'] at t" | null |
| genz_ffn_kinds | null | {"Dense": 18} |
| genz_kv_size | null | 36864 |
| genz_layer_kinds | null | {"MHA-global": 18} |
| genz_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['is_encoder_decoder', 'model_type', 'text_config', 'vision_config']. Looked for ['num_hidden_layers', 'hidden_size'] at t" | null |
| genz_prefill_kv_mb | null | 648.0 |
| genz_prefill_weights_mb | null | 1785.518555 |
| kv_cross_layers | 0 | 18 |
| kv_gb@32768 | 0.0 | 2.566914048 |
| kv_gb@4096 | 0.0 | 0.452984832 |
| kvgeo_32768 | [0.0, 1, 0, 0.0, 16, 0.0] | [73728.0, 18, 0, 0.0, 16, 150994944.0] |
| kvgeo_4096 | [0.0, 1, 0, 0.0, 16, 0.0] | [73728.0, 18, 0, 0.0, 16, 150994944.0] |
| prefix_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['is_encoder_decoder', 'model_type', 'text_config', 'vision_config']. Looked for [" | null |
| prefix_kv_bytes_per_token | null | 73728 |
| report_kv_gb | 0.0 | 0.452984832 |
| report_params | 713446400 | 917650688 |
| report_total_gb | 1.535638651 | 2.487629025 |
| report_weight_gb | 1.4268928 | 1.835301376 |
| runtime_chunked | null | [0.903872, 1785.519, 342.0] |
| runtime_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['is_encoder_decoder', 'model_type', 'text_config', 'vision_config']. Looked for [" | null |
| runtime_decode | null | [2.365517, 1687.386, 612.281] |
| runtime_decode_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['is_encoder_decoder', 'model_type', 'text_config', 'vision_config']. Looked for [" | null |
| runtime_prefill | null | [13.267629, 1785.519, 396.0] |
| runtime_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['is_encoder_decoder', 'model_type', 'text_config', 'vision_config']. Looked for [" | null |
| serving_bytes_per_token | null | 73728 |
| serving_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['is_encoder_decoder', 'model_type', 'text_config', 'vision_config']. Looked for ['num_hidden_layers', 'hidden_size'] at t" | null |
| serving_requests_8gib_1k | null | 37 |
| serving_state_bytes | null | 150994944 |
| serving_weight_bytes | null | 2847540736 |
| serving_weight_bytes_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['is_encoder_decoder', 'model_type', 'text_config', 'vision_config']. Looked for ['num_hidden_layers', 'hidden_size'] at t" | null |
| sim_max_running | 400 | 315 |
| slice_mib_16x4352 | [14080, true] | [17152, true] |
| vllm_limits | [61354, 981664, 61354, 1, 0, 0.0, true] | [61012, 976192, 61012, 1, 0, 150994944.0, true] |
| vllm_running_1024 | 766 | 293 |
| vllm_running_8192 | 116 | 93 |

## synth:prophetnet (encoder-decoder)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 75 | 104 |
| conv_kv_size | 32768 | 49152 |
| conv_layers | [16, 16] | [12, 12] |
| conv_prefill_weights_mb | 1279.226562 | 984.226562 |
| conv_serving_weight_bytes | 1278857216 | 930699264 |
| disagg_b_eff_4096 | 75 | 104 |
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
| facts_kv_bytes_4096 | 536870912 | 402653184 |
| facts_kv_bytes_per_token | 131072 | 49152 |
| facts_layers | [32, 0, 0, null] | [12, 0, 0, null] |
| facts_weight_bytes | 2272634880 | 930699264 |
| genz_chunked_kv_mb | 576.0 | 432.0 |
| genz_decode_kv_mb | 512.0 | 384.0 |
| genz_decode_weights_mb | 1691.613281 | 731.613281 |
| genz_ffn_kinds | {"Dense": 32} | {"Dense": 12} |
| genz_kv_size | 65536 | 49152 |
| genz_layer_kinds | {"MHA-global": 32} | {"MHA-global": 12} |
| genz_prefill_kv_mb | 512.0 | 576.0 |
| genz_prefill_weights_mb | 1751.226562 | 792.226562 |
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
| runtime_chunked | [1.668514, 1751.227, 96.0] | [0.469143, 792.227, 72.0] |
| runtime_decode | [2.589122, 1691.613, 64.5] | [1.730039, 731.613, 120.188] |
| runtime_prefill | [5.251511, 1751.227, 64.0] | [3.575962, 792.227, 72.0] |
| serving_bytes_per_token | 131072 | 98304 |
| serving_requests_8gib_1k | 64 | 85 |
| serving_weight_bytes | 2495205376 | 930699264 |
| slice_mib_16x4352 | [18432, true] | [14336, true] |
| vllm_limits | [34622, 553952, 34622, 1, 0, 0.0, true] | [94033, 1504528, 94033, 1, 0, 25165824.0, true] |
| vllm_running_1024 | 432 | 839 |
| vllm_running_8192 | 65 | 167 |

## synth:qwen2-audio (audio-llm)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 13 | 12 |
| conv_prefill_weights_mb | 12852.0 | 14790.0 |
| conv_serving_weight_bytes | 13214154752 | 16812389376 |
| disagg_b_eff_4096 | 13 | 12 |
| facts_weight_bytes | 15508971520 | 16812389376 |
| serving_weight_bytes | 14230224896 | 16812389376 |
| sim_max_running | 108 | 106 |
| slice_mib_16x4352 | [62208, true] | [63744, true] |
| vllm_limits | [7075, 113200, 7075, 1, 0, 0.0, true] | [6919, 110704, 6919, 1, 0, 0.0, true] |
| vllm_running_1024 | 88 | 86 |

## synth:qwen35-list-depth (multimodal)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 1 | 182 |
| conv_kv_size | 262144 | 16384 |
| conv_prefill_weights_mb | 12852.0 | 9234.023438 |
| conv_serving_weight_bytes | 13214154752 | 9078531072 |
| conv_state_bytes | 0 | 51511296 |
| decode_fit_refused | null | true |
| disagg_b_eff_4096 | 1 | 182 |
| disagg_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'image_token_id', 'model_type', 'text_config', 'tie_word_embeddings', 'transformers_version', 'video_tok" | null |
| disagg_transfer_bytes_640 | null | 72482816 |
| eng_None_32768_params | 2622843904 | 5028806144 |
| eng_None_32768_state_mb | 0.0 | 51.511296 |
| eng_None_32768_total_mb | 6833.655398 | 11697.091174 |
| eng_None_4096_params | 2622843904 | 5028806144 |
| eng_None_4096_state_mb | 0.0 | 51.511296 |
| eng_None_4096_total_mb | 5637.439898 | 10500.875674 |
| eng_llamacpp_32768_params | 2622843904 | 5028806144 |
| eng_llamacpp_32768_state_mb | 0.0 | 51.511296 |
| eng_llamacpp_32768_total_mb | 6833.655398 | 11697.091174 |
| eng_llamacpp_4096_params | 2622843904 | 5028806144 |
| eng_llamacpp_4096_state_mb | 0.0 | 51.511296 |
| eng_llamacpp_4096_total_mb | 5637.439898 | 10500.875674 |
| facts_full_attention | true | false |
| facts_kv_bytes_4096 | 536870912 | 134217728 |
| facts_kv_bytes_per_token | 131072 | 32768 |
| facts_layers | [32, 0, 0, null] | [8, 0, 24, null] |
| facts_state_bytes | 0 | 51511296 |
| facts_weight_bytes | 8817451008 | 9078531072 |
| genz_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'image_token_id', 'model_type', 'text_config', 'tie_word_embeddings', 'transformers_version', 'video_tok" | null |
| genz_chunked_kv_mb | null | 242.25 |
| genz_decode_kv_mb | null | 177.125 |
| genz_decode_weights_mb | null | 8021.523438 |
| genz_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'image_token_id', 'model_type', 'text_config', 'tie_word_embeddings', 'transformers_version', 'video_tok" | null |
| genz_ffn_kinds | null | {"Dense": 32} |
| genz_kv_size | null | 16384 |
| genz_layer_kinds | null | {"MHA-global": 8, "Recurrent": 24} |
| genz_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'image_token_id', 'model_type', 'text_config', 'tie_word_embeddings', 'transformers_version', 'video_tok" | null |
| genz_prefill_kv_mb | null | 177.125 |
| genz_prefill_weights_mb | null | 9234.023438 |
| kv_free_layers | 0 | 24 |
| kv_source | "_layer_metadata" | "layer_plan:layer_types" |
| kvgeo_32768 | [32768.0, 1, 0, 0.0, 16, 0.0] | [32768.0, 8, 24, 2146304.0, 528, 0.0] |
| kvgeo_4096 | [32768.0, 1, 0, 0.0, 16, 0.0] | [32768.0, 8, 24, 2146304.0, 528, 0.0] |
| prefix_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'image_token_id', 'model_type', 'text_config', 'tie_word_embeddi" | null |
| prefix_kv_bytes_per_token | null | 32768 |
| report_params | 2622843904 | 5028806144 |
| report_state_gb | 0.0 | 0.051511296 |
| report_total_gb | 5.715790316 | 10.589528351 |
| report_weight_gb | 5.245687808 | 10.057612288 |
| runtime_chunked | null | [8.59528, 9234.023, 122.25] |
| runtime_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'image_token_id', 'model_type', 'text_config', 'tie_word_embeddi" | null |
| runtime_decode | null | [6.398831, 8021.523, 212.625] |
| runtime_decode_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'image_token_id', 'model_type', 'text_config', 'tie_word_embeddi" | null |
| runtime_prefill | null | [18.879089, 9234.023, 65.125] |
| runtime_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'image_token_id', 'model_type', 'text_config', 'tie_word_embeddi" | null |
| serving_bytes_per_token | null | 32768 |
| serving_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'image_token_id', 'model_type', 'text_config', 'tie_word_embeddings', 'transformers_version', 'video_tok" | null |
| serving_requests_8gib_1k | null | 100 |
| serving_state_bytes | null | 51511296 |
| serving_weight_bytes | null | 9078531072 |
| serving_weight_bytes_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['architectures', 'image_token_id', 'model_type', 'text_config', 'tie_word_embeddings', 'transformers_version', 'video_tok" | null |
| slice_mib_16x4352 | [23808, true] | [20480, true] |
| state_gb | 0.0 | 0.051511296 |
| vllm_limits | [31388, 502208, 31388, 1, 0, 0.0, true] | [3767, 1988976, 3767, 1, 0, 51904512.0, true] |
| vllm_running_1024 | 392 | 627 |
| vllm_running_8192 | 59 | 198 |

## synth:rwkv (state-space)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 49 | 39994 |
| conv_kv_size | 98304 | 0 |
| conv_prefill_weights_mb | 3464.789062 | 3465.257812 |
| conv_serving_weight_bytes | 3427160064 | 3633786880 |
| conv_state_bytes | 0 | 983040 |
| disagg_b_eff_4096 | 49 | 39994 |
| disagg_transfer_bytes_640 | 125829120 | 983040 |
| eng_None_32768_params | 1309554688 | 1414340608 |
| eng_None_32768_state_mb | 0.0 | 0.98304 |
| eng_None_32768_total_mb | 3062.027878 | 3272.582758 |
| eng_None_4096_params | 1309554688 | 1414340608 |
| eng_None_4096_state_mb | 0.0 | 0.98304 |
| eng_None_4096_total_mb | 2840.568627 | 3051.123507 |
| eng_llamacpp_32768_params | 1309554688 | 1414340608 |
| eng_llamacpp_32768_state_mb | 0.0 | 0.98304 |
| eng_llamacpp_32768_total_mb | 3062.027878 | 3272.582758 |
| eng_llamacpp_4096_params | 1309554688 | 1414340608 |
| eng_llamacpp_4096_state_mb | 0.0 | 0.98304 |
| eng_llamacpp_4096_total_mb | 2840.568627 | 3051.123507 |
| facts_full_attention | true | false |
| facts_kv_bytes_4096 | 805306368 | 0 |
| facts_kv_bytes_per_token | 196608 | 0 |
| facts_layers | [24, 0, 0, null] | [0, 0, 24, null] |
| facts_state_bytes | 0 | 983040 |
| facts_weight_bytes | 3633295360 | 3633786880 |
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
| report_params | 1309554688 | 1414340608 |
| report_state_gb | 0.00688128 | 0.00098304 |
| report_total_gb | 2.893118013 | 3.095611965 |
| report_weight_gb | 2.619109376 | 2.828681216 |
| runtime_chunked | [2.552108, 2696.789, 144.0] | [2.496627, 2697.258, 1.875] |
| runtime_decode | [2.975656, 2500.395, 96.75] | [2.771447, 2500.863, 3.75] |
| runtime_prefill | [6.725363, 2696.789, 96.0] | [6.583991, 2697.258, 0.938] |
| serving_bytes_per_token | 196608 | 0 |
| serving_requests_8gib_1k | 42 | 8738 |
| serving_state_bytes | 0 | 983040 |
| serving_weight_bytes | 3427160064 | 3633786880 |
| sim_max_running | 342 | 338 |
| state_gb | 0.00688128 | 0.00098304 |
| vllm_limits | [22571, 361136, 22571, 1, 0, 0.0, true] | [22571, 361136, 22571, 1, 0, 3145728.0, true] |
| vllm_running_1024 | 282 | 278 |

## synth:t5-asym (encoder-decoder)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 13 | 104 |
| conv_kv_size | 131072 | 49152 |
| conv_layers | [16, 16] | [24, 12] |
| conv_prefill_weights_mb | 6678.0 | 1103.5 |
| conv_serving_weight_bytes | 6739197952 | 1019324416 |
| disagg_b_eff_4096 | 13 | 104 |
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
| facts_kv_bytes_per_token | 98304 | 49152 |
| facts_layers | [24, 0, 0, null] | [12, 0, 0, null] |
| facts_weight_bytes | 1742309376 | 1019324416 |
| genz_chunked_kv_mb | 2304.0 | 432.0 |
| genz_decode_kv_mb | 2048.0 | 384.0 |
| genz_decode_weights_mb | 9851.0 | 842.75 |
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
| runtime_chunked | [9.403579, 10102.0, 384.0] | [0.419976, 905.5, 72.0] |
| runtime_decode | [7.6952, 9851.0, 258.0] | [1.791324, 842.75, 120.188] |
| runtime_prefill | [20.430178, 10102.0, 256.0] | [4.181622, 905.5, 96.0] |
| serving_bytes_per_token | 524288 | 98304 |
| serving_requests_8gib_1k | 16 | 85 |
| serving_weight_bytes | 13215203328 | 1019324416 |
| sim_max_running | 400 | 293 |
| slice_mib_16x4352 | [15104, true] | [14336, true] |
| vllm_limits | [46813, 749008, 46813, 1, 0, 0.0, true] | [94547, 1512752, 94547, 1, 0, 201326592.0, true] |
| vllm_running_1024 | 585 | 281 |
| vllm_running_8192 | 88 | 120 |

## synth:t5-small (encoder-decoder)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 13 | 425 |
| conv_kv_size | 131072 | 12288 |
| conv_layers | [16, 16] | [6, 6] |
| conv_prefill_weights_mb | 6678.0 | 170.75 |
| conv_serving_weight_bytes | 6739197952 | 166556672 |
| disagg_b_eff_4096 | 13 | 425 |
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
| facts_kv_bytes_4096 | 50331648 | 100663296 |
| facts_weight_bytes | 166474752 | 166556672 |
| genz_chunked_kv_mb | 2304.0 | 108.0 |
| genz_decode_kv_mb | 2048.0 | 96.0 |
| genz_decode_weights_mb | 9851.0 | 115.375 |
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
| runtime_chunked | [9.403579, 10102.0, 384.0] | [0.114413, 146.75, 18.0] |
| runtime_decode | [7.6952, 9851.0, 258.0] | [1.270544, 115.375, 30.047] |
| runtime_prefill | [20.430178, 10102.0, 256.0] | [1.815171, 146.75, 18.0] |
| serving_bytes_per_token | 524288 | 24576 |
| serving_requests_8gib_1k | 16 | 341 |
| serving_weight_bytes | 13215203328 | 166556672 |
| slice_mib_16x4352 | [6656, true] | [7424, true] |
| vllm_limits | [382640, 6122240, 382640, 1, 0, 0.0, true] | [382640, 6122240, 382640, 1, 0, 50331648.0, true] |
| vllm_running_8192 | 724 | 488 |

## synth:unknown-layer-name (decoder-only)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 1261 | 1257 |
| conv_serving_weight_bytes | 617611264 | 748720128 |
| disagg_b_eff_4096 | 1261 | 1257 |
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
| serving_weight_bytes | 617611264 | 748720128 |

## synth:vision-enc-dec (encoder-decoder)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 1 | 183 |
| conv_kv_size | 131072 | 24576 |
| conv_layers | [16, 16] | [12, 12] |
| conv_prefill_weights_mb | 6676.0 | 896.820801 |
| conv_serving_weight_bytes | 6738149376 | 879365120 |
| conv_state_bytes | 0 | 28360704 |
| decode_fit_refused | null | true |
| disagg_b_eff_4096 | 1 | 183 |
| disagg_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['decoder', 'encoder', 'is_encoder_decoder', 'model_type']. Looked for ['num_hidden_layers', 'hidden_size'] at the top lev" | null |
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
| facts_kv_bytes_4096 | 2147483648 | 229687296 |
| facts_kv_bytes_per_token | 524288 | 49152 |
| facts_layers | [32, 0, 0, null] | [12, 0, 0, null] |
| facts_weight_bytes | 34884558848 | 879365120 |
| genz_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['decoder', 'encoder', 'is_encoder_decoder', 'model_type']. Looked for ['num_hidden_layers', 'hidden_size'] at the top lev" | null |
| genz_chunked_kv_mb | null | 270.09375 |
| genz_decode_kv_mb | null | 219.046875 |
| genz_decode_weights_mb | null | 647.646973 |
| genz_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['decoder', 'encoder', 'is_encoder_decoder', 'model_type']. Looked for ['num_hidden_layers', 'hidden_size'] at the top lev" | null |
| genz_ffn_kinds | null | {"Dense": 12} |
| genz_kv_size | null | 24576 |
| genz_layer_kinds | null | {"MHA-global": 12} |
| genz_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['decoder', 'encoder', 'is_encoder_decoder', 'model_type']. Looked for ['num_hidden_layers', 'hidden_size'] at the top lev" | null |
| genz_prefill_kv_mb | null | 239.332031 |
| genz_prefill_weights_mb | null | 746.820801 |
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
| runtime_chunked | null | [0.553414, 746.821, 90.094] |
| runtime_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['decoder', 'encoder', 'is_encoder_decoder', 'model_type']. Looked for ['num_hidde" | null |
| runtime_decode | null | [1.761449, 647.647, 132.375] |
| runtime_decode_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['decoder', 'encoder', 'is_encoder_decoder', 'model_type']. Looked for ['num_hidde" | null |
| runtime_prefill | null | [3.552638, 746.821, 71.332] |
| runtime_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['decoder', 'encoder', 'is_encoder_decoder', 'model_type']. Looked for ['num_hidde" | null |
| serving_bytes_per_token | null | 49152 |
| serving_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['decoder', 'encoder', 'is_encoder_decoder', 'model_type']. Looked for ['num_hidden_layers', 'hidden_size'] at the top lev" | null |
| serving_requests_8gib_1k | null | 109 |
| serving_state_bytes | null | 28360704 |
| serving_weight_bytes | null | 879365120 |
| serving_weight_bytes_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['decoder', 'encoder', 'is_encoder_decoder', 'model_type']. Looked for ['num_hidden_layers', 'hidden_size'] at the top lev" | null |
| sim_max_running | 72 | 400 |
| slice_mib_16x4352 | [81582, false] | [11008, true] |
| vllm_limits | [4702, 75232, 4702, 1, 0, 0.0, true] | [94021, 1504336, 94021, 1, 0, 28360704.0, true] |
| vllm_running_1024 | 58 | 810 |
| vllm_running_8192 | 8 | 166 |

## synth:vision-enc-dec-gpt2 (encoder-decoder)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 1 | 536 |
| conv_kv_size | 131072 | 9216 |
| conv_layers | [16, 16] | [12, 6] |
| conv_prefill_weights_mb | 6676.0 | 708.903809 |
| conv_serving_weight_bytes | 6738149376 | 466079232 |
| conv_state_bytes | 0 | 3631104 |
| decode_fit_refused | null | true |
| disagg_b_eff_4096 | 1 | 536 |
| disagg_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['decoder', 'encoder', 'is_encoder_decoder', 'model_type']. Looked for ['num_hidden_layers', 'hidden_size'] at the top lev" | null |
| disagg_transfer_bytes_640 | null | 15427584 |
| eng_None_32768_kv_mb | 701.80864 | 662.90688 |
| eng_None_32768_params | 808394240 | 733372928 |
| eng_None_32768_total_mb | 2484.691558 | 2295.747174 |
| eng_None_4096_kv_mb | 114.60608 | 134.424576 |
| eng_None_4096_params | 808394240 | 733372928 |
| eng_None_4096_total_mb | 1814.441779 | 1684.217651 |
| eng_llamacpp_32768_kv_mb | 701.80864 | 662.90688 |
| eng_llamacpp_32768_params | 808394240 | 733372928 |
| eng_llamacpp_32768_total_mb | 2484.691558 | 2295.747174 |
| eng_llamacpp_4096_kv_mb | 114.60608 | 134.424576 |
| eng_llamacpp_4096_params | 808394240 | 733372928 |
| eng_llamacpp_4096_total_mb | 1814.441779 | 1684.217651 |
| facts_kv_bytes_4096 | 2147483648 | 79128576 |
| facts_kv_bytes_per_token | 524288 | 18432 |
| facts_layers | [32, 0, 0, null] | [6, 0, 0, null] |
| facts_weight_bytes | 34884558848 | 466079232 |
| genz_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['decoder', 'encoder', 'is_encoder_decoder', 'model_type']. Looked for ['num_hidden_layers', 'hidden_size'] at the top lev" | null |
| genz_chunked_kv_mb | null | 87.925781 |
| genz_decode_kv_mb | null | 75.462891 |
| genz_decode_weights_mb | null | 484.535156 |
| genz_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['decoder', 'encoder', 'is_encoder_decoder', 'model_type']. Looked for ['num_hidden_layers', 'hidden_size'] at the top lev" | null |
| genz_ffn_kinds | null | {"Dense": 6} |
| genz_kv_size | null | 9216 |
| genz_layer_kinds | null | {"MHA-global": 6} |
| genz_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['decoder', 'encoder', 'is_encoder_decoder', 'model_type']. Looked for ['num_hidden_layers', 'hidden_size'] at the top lev" | null |
| genz_prefill_kv_mb | null | 82.388672 |
| genz_prefill_weights_mb | null | 558.153809 |
| kv_cross_layers | 0 | 6 |
| kv_full_layers | 0 | 6 |
| kv_gb@32768 | 0.0 | 0.60761088 |
| kv_gb@4096 | 0.0 | 0.079128576 |
| kvgeo_32768 | [0.0, 1, 0, 0.0, 16, 0.0] | [18432.0, 6, 0, 0.0, 16, 3631104.0] |
| kvgeo_4096 | [0.0, 1, 0, 0.0, 16, 0.0] | [18432.0, 6, 0, 0.0, 16, 3631104.0] |
| prefix_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['decoder', 'encoder', 'is_encoder_decoder', 'model_type']. Looked for ['num_hidde" | null |
| prefix_kv_bytes_per_token | null | 18432 |
| report_kv_gb | 0.11460608 | 0.134424576 |
| report_params | 808394240 | 733372928 |
| report_total_gb | 1.856430039 | 1.72924801 |
| report_weight_gb | 1.61678848 | 1.466745856 |
| runtime_chunked | null | [0.361522, 558.154, 20.426] |
| runtime_chunked_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['decoder', 'encoder', 'is_encoder_decoder', 'model_type']. Looked for ['num_hidde" | null |
| runtime_decode | null | [1.458536, 484.535, 22.922] |
| runtime_decode_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['decoder', 'encoder', 'is_encoder_decoder', 'model_type']. Looked for ['num_hidde" | null |
| runtime_prefill | null | [2.563321, 558.154, 19.389] |
| runtime_prefill_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['decoder', 'encoder', 'is_encoder_decoder', 'model_type']. Looked for ['num_hidde" | null |
| serving_bytes_per_token | null | 18432 |
| serving_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['decoder', 'encoder', 'is_encoder_decoder', 'model_type']. Looked for ['num_hidden_layers', 'hidden_size'] at the top lev" | null |
| serving_requests_8gib_1k | null | 381 |
| serving_state_bytes | null | 3631104 |
| serving_weight_bytes | null | 466079232 |
| serving_weight_bytes_error | "ValueError: Could not locate the decoder's geometry in this model config. Top-level keys: ['decoder', 'encoder', 'is_encoder_decoder', 'model_type']. Looked for ['num_hidden_layers', 'hidden_size'] at the top lev" | null |
| sim_max_running | 72 | 400 |
| slice_mib_16x4352 | [81582, false] | [7424, true] |
| vllm_limits | [4702, 75232, 4702, 1, 0, 0.0, true] | [253850, 4061600, 253850, 1, 0, 3631104.0, true] |
| vllm_running_1024 | 58 | 1024 |
| vllm_running_8192 | 8 | 469 |

## synth:whisper-large-v3 (encoder-decoder)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 13 | 42 |
| conv_kv_size | 262144 | 81920 |
| conv_prefill_weights_mb | 13162.40625 | 3868.325195 |
| conv_serving_weight_bytes | 13376897024 | 3622233600 |
| conv_state_bytes | 0 | 245760000 |
| disagg_b_eff_4096 | 13 | 42 |
| disagg_transfer_bytes_640 | 335544320 | 350617600 |
| facts_kv_bytes_4096 | 671088640 | 916848640 |
| facts_weight_bytes | 3621163520 | 3622233600 |
| genz_chunked_kv_mb | 2304.0 | 1188.75 |
| genz_decode_kv_mb | 2048.0 | 874.375 |
| genz_decode_weights_mb | 10005.203125 | 2940.605469 |
| genz_kv_size | 262144 | 81920 |
| genz_prefill_kv_mb | 2048.0 | 1108.75 |
| genz_prefill_weights_mb | 10410.40625 | 3068.325195 |
| kv_cross_layers | 0 | 32 |
| kv_gb@32768 | 0.0 | 5.61446912 |
| kv_gb@4096 | 0.0 | 0.91684864 |
| kvgeo_32768 | [0.0, 32, 0, 0.0, 16, 0.0] | [163840.0, 32, 0, 0.0, 16, 245760000.0] |
| kvgeo_4096 | [0.0, 32, 0, 0.0, 16, 0.0] | [163840.0, 32, 0, 0.0, 16, 245760000.0] |
| prefix_kv_bytes_per_token | 524288 | 163840 |
| runtime_chunked | [9.680075, 10410.406, 384.0] | [1.947247, 3068.325, 588.75] |
| runtime_decode | [7.78941, 10005.203, 258.0] | [3.650774, 2940.605, 1018.125] |
| runtime_prefill | [20.982092, 10410.406, 256.0] | [15.824888, 3068.325, 548.75] |
| serving_bytes_per_token | 524288 | 163840 |
| serving_requests_8gib_1k | 16 | 20 |
| serving_state_bytes | 0 | 245760000 |
| serving_weight_bytes | 13376897024 | 3622233600 |
| sim_max_running | 400 | 171 |
| slice_mib_16x4352 | [22272, true] | [26624, true] |
| vllm_limits | [27344, 437504, 27344, 1, 0, 0.0, true] | [27343, 437488, 27343, 1, 0, 245760000.0, true] |
| vllm_running_1024 | 341 | 157 |
| vllm_running_8192 | 51 | 43 |

## synth:xlstm (state-space)

| figure | before | after |
|---|---|---|
| cluster_b_eff_4096 | 13 | 185 |
| conv_kv_size | 262144 | 0 |
| conv_prefill_weights_mb | 13138.0 | 13142.0 |
| conv_serving_weight_bytes | 13364101120 | 18008776704 |
| conv_state_bytes | 0 | 134480896 |
| disagg_b_eff_4096 | 13 | 185 |
| disagg_transfer_bytes_640 | 335544320 | 134480896 |
| eng_None_32768_params | 6646931456 | 6857170944 |
| eng_None_32768_state_mb | 0.0 | 134.480896 |
| eng_None_32768_total_mb | 14179.699917 | 14734.659789 |
| eng_None_4096_params | 6646931456 | 6857170944 |
| eng_None_4096_state_mb | 0.0 | 134.480896 |
| eng_None_4096_total_mb | 13736.781414 | 14291.741286 |
| eng_llamacpp_32768_params | 6646931456 | 6857170944 |
| eng_llamacpp_32768_state_mb | 0.0 | 134.480896 |
| eng_llamacpp_32768_total_mb | 14179.699917 | 14734.659789 |
| eng_llamacpp_4096_params | 6646931456 | 6857170944 |
| eng_llamacpp_4096_state_mb | 0.0 | 134.480896 |
| eng_llamacpp_4096_total_mb | 13736.781414 | 14291.741286 |
| facts_full_attention | true | false |
| facts_kv_bytes_4096 | 2147483648 | 0 |
| facts_kv_bytes_per_token | 524288 | 0 |
| facts_layers | [32, 0, 0, null] | [0, 0, 32, null] |
| facts_state_bytes | 0 | 134480896 |
| facts_weight_bytes | 18004582400 | 18008776704 |
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
| report_params | 6646931456 | 6857170944 |
| report_state_gb | 0.0 | 0.134480896 |
| report_total_gb | 13.825365115 | 14.407221166 |
| report_weight_gb | 13.293862912 | 13.714341888 |
| runtime_chunked | [9.658194, 10386.0, 384.0] | [9.549188, 10390.0, 256.502] |
| runtime_decode | [7.781955, 9993.0, 258.0] | [7.738026, 9997.0, 513.004] |
| runtime_prefill | [20.938415, 10386.0, 256.0] | [20.60359, 10390.0, 128.251] |
| serving_bytes_per_token | 524288 | 0 |
| serving_requests_8gib_1k | 16 | 63 |
| serving_state_bytes | 0 | 134480896 |
| serving_weight_bytes | 13364101120 | 18008776704 |
| sim_max_running | 103 | 78 |
| slice_mib_16x4352 | [66304, true] | [68608, true] |
| state_gb | 0.0 | 0.134480896 |
| vllm_limits | [6707, 107312, 6707, 1, 0, 0.0, true] | [394, 107168, 394, 1, 0, 142606336.0, false] |
| vllm_running_1024 | 83 | 65 |
| vllm_running_8192 | 12 | 11 |

