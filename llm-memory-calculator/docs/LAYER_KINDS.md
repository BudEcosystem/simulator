# Layer kinds: which layers hold a KV cache

A model's KV cache is not `layers x tokens x KV width`. Only layers that run softmax attention
over their own keys and values keep a per-token cache, and how far that cache grows depends on
the kind of attention. Everything else -- Gated DeltaNet, Mamba, RWKV, short convolutions,
FFN-only slots -- keeps a **fixed** per-sequence state that does not grow with the context.

The simulator reads which kind every layer is from the model's **own config keys**, through one
table, and every part of the simulator answers from it:

| Caller | What it asks |
|---|---|
| `ModelMemoryCalculator.calculate_kv_cache` / `kv_cache_breakdown` / `calculate_state_memory` | KV bytes per kind, fixed state |
| `bud_component.py --gguf` (via `gguf_layers.hf_layer_keys_from_gguf`) | the same kinds from a GGUF's llama.cpp keys |
| GenZ graph (`get_language_model.layer_configs_from_hf`, `ModelConfig.get_kv_size`) -- the prefill, decode **and chunked** builders (`create_full_chunked_model`: chunked serving, speculative decoding, the simulation engine's chunked feature) | per-layer operators; KV only on attention layers, the window on windowed ones; which FFNs are MoE (`layer_plan.moe_layers`) |
| GenZ serving (`MemoryModel.bytes_per_token_kv`, disaggregation KV transfer) | per-token KV |
| `UniversalParameterCounter`, `model.facts` (FRD-027), `model.recognise` | weights per mixer, KV per token, the known vocabulary. `model.facts` reads the layer kinds for every config: a `model_type` outside the recogniser is still flagged `architecture: unrecognised`, but which layers hold KV comes from the table, not from that allowlist |
| BudSimulator `ModelConverter.hf_to_genz` | database models graphed like HF configs |

The table is `layer_plan.LAYER_KINDS`; the per-model answer is `layer_plan.layer_kinds(config)`
(or `resolve_layer_plan(config)` for hybrids), whose depth is `num_hidden_layers` or, when a
config states only a per-layer list (transformers 5.12's Nemotron-H), that list's length.
Which layers route to experts is `layer_plan.moe_layers(config)`. Nothing keys off a model name.

## The root cause this replaced

Layer kinds used to be decided in several places, and each got a different subset wrong:

* `calculator._count_attention_layers` and `parameter_counter._calculate_hybrid_params` counted a
  layer as attention when its `layer_types` entry *contained the substring* `"attention"`, so
  Gated DeltaNet's `"linear_attention"` layers (Qwen3-Next, Qwen3.5/3.6) were charged a full KV
  cache, and a hybrid with no `layer_types` was guessed at 50 % attention.
* `bud_component.gguf_to_hf_config` dropped every GGUF key that says which layers attend
  (`{arch}.full_attention_interval`, per-layer `attention.head_count_kv`, `ssm.*`), so a hybrid
  GGUF reached the calculator as a dense stack: **Qwen3.5-2B was charged KV on all 24 layers,
  4x the 6 that llama.cpp allocates**.
* The GenZ performance graph built every decoder layer as attention unless the config was a
  sliding/full `layer_types` list, so its KV traffic, KV memory and offload decisions for every
  hybrid (Qwen3.5: 512 MiB at 4,096 tokens against a measured 128 MiB) were inflated, and Mamba
  hybrids lost their attention layers entirely. `ModelConfig.get_kv_size` and the serving
  memory model's fallback counted all decoder layers. The chunked graph builder had no per-layer
  branch at all: every layer of every model was graphed as full attention (Qwen3.5: 32 layers of
  KV), windowed layers were not bounded, and MLA was graphed as multi-head KV.

Measured truth (GB10, 27 Oct 2026, llama.cpp and mistral.rs): Qwen3.5-4B holds a paged pool on its
8 `full_attention` layers only -- 129 MiB at 4,096 tokens -- and a fixed Gated DeltaNet state on
the other 24. The simulator now answers 128 MiB (8 x 2 x 4 KV heads x 256 x 2 B x 4,096).

## Categories and the config keys that state them

Each key was checked against the shipped configs in `tests/fixtures/hybrid_configs/` (verbatim
HF `config.json` files) and against transformers 5.12's configuration classes; GGUF keys against
gguf-py's `Keys` (`gguf/constants.py`).

| Category | KV per layer | Config keys read | Examples |
|---|---|---|---|
| Full / global attention | every token | `layer_types: full_attention / attention / global`, or a uniform stack | Llama, Qwen2, most dense |
| Sliding-window attention | at most `sliding_window` tokens | `layer_types: sliding_attention`; `sliding_window_pattern: N` (one full layer in N, the last of each group); `sliding_window_layout` (0/1 mask); `global_attn_idx`; window spellings `sliding_window`, `sliding_window_size`, `attention_window_size`, `window_size`; `use_sliding_window: false` disables. A window with **no** layout key takes the layout the model type's own config class derives (transformers: `Gemma2Config` alternates sliding and full, `Qwen2Config` windows from `max_window_layers` on), or every layer when that class states no per-layer list (Mistral) | Gemma 2/3, gpt-oss, Mistral v0.1, SmallThinker, Hymba |
| Chunked local attention | at most `attention_chunk_size` | `layer_types: chunked_attention` | Llama 4 |
| Sparse / compressed attention | every token (upper bound) | `deepseek_sparse_attention`, `compressed_sparse_attention`, `heavily_compressed_attention`, `minimax_m3_sparse`, `sparse` | DeepSeek-V3.2/V4, MiniMax-M3 |
| MLA (latent KV) | `kv_lora_rank + qk_rope_head_dim`, no factor 2 | `kv_lora_rank` (non-null), `qk_rope_head_dim` | DeepSeek-V2/V3, Kimi |
| Cross-attention | the encoder's tokens, fixed per request | `cross_attention_layers`; encoder tokens = `max_num_tiles x ((image_size / patch_size)^2 + 1)` from `vision_config` | Mllama |
| KV-shared layers | none of their own | `num_kv_shared_layers` (the last N layers) | Gemma 3n |
| Gated DeltaNet | fixed: `nv x dk x dv` fp32 + conv | `layer_types: linear_attention` with `linear_num_value_heads`, `linear_key_head_dim`, `linear_value_head_dim`, `linear_num_key_heads`, `linear_conv_kernel_dim`; `full_attention_interval` | Qwen3-Next, Qwen3.5/3.6 |
| Plain linear attention | fixed: `heads x d x d` | `linear_attention` without GDN keys; `attn_type_list` (0 = linear) | MiniMax-Text-01 |
| Kimi Delta Attention | fixed | `linear_attn_config.{full_attn_layers, kda_layers}` (one-indexed) | Kimi Linear |
| Mamba-1 / Mamba-2 | fixed: `heads x d_head x d_state` fp32 + conv | `layer_types: mamba`, `layers_block_type`, `hybrid_override_pattern` (`M`/`E`/`*`/`-`, as transformers' `NemotronHConfig` translates them), `attn_layer_indices`, `attn_layer_period`/`attn_layer_offset`, state keys `state_size`/`mamba_d_state`/`ssm_state_size`, heads `mamba_n_heads`/`mamba_num_heads`, `mamba_head_dim`, groups `n_groups`/`mamba_n_groups` | Mamba, Jamba, Nemotron-H, granite-4.0-h, Bamba, Zamba2, Falcon-H1 |
| Short convolution | fixed conv buffer | `layer_types: conv`, `conv_L_cache` | LFM2 |
| RWKV | fixed: token shift + WKV | no attention heads + `attention_hidden_size`; GGUF `wkv.head_size` | RWKV |
| xLSTM (mLSTM) | fixed matrix memory | `qk_dim_factor`, `v_dim_factor`, `num_heads` | xLSTM |
| RG-LRU | fixed vector + conv | `block_types` (repeating), `lru_width`, `conv1d_width`, `attention_window_size` | RecurrentGemma |
| FFN-only slot | none | `moe`, `dense`, `mlp` in a layer list; `-` and `E` in `hybrid_override_pattern`. A stack that names such slots gives the FFN its own layer: only those carry one | Nemotron-H |
| MoE | does not change KV; decides which FFN a layer runs | `mlp_layer_types` (dense / sparse), `moe_layers`, a per-layer expert count, `expert_layer_period`/`expert_layer_offset`, `decoder_sparse_step` + `mlp_only_layers`, `interleave_moe_layer_step`, `moe_layer_start_index`/`moe_layer_end_index`/`moe_layer_interval`, `first_k_dense_replace`/`moe_layer_freq`; a `moe` FFN slot | Mixtral, DeepSeek, Kimi, Jamba, Qwen-MoE, Llama 4, ERNIE |

Parallel hybrids (attention and a mixer in the same layer: Falcon-H1, Hymba, Zamba's `hybrid`
layers) set both axes on that layer: it holds KV *and* a state.

Zamba / Zamba2 name their layers through the same table (`mamba`, `hybrid`; any other name is
charged full attention with the warning). A `hybrid` layer invokes the shared attention+MLP
block, so it runs that block's MLP too; a plain `mamba` layer has no FFN. The block's weights
exist once per shared block (`num_mem_blocks`, one when the config states none), so the
parameter counts (`UniversalParameterCounter`, `model.facts`) count them that many times, while
the graph streams them on every layer that invokes the block.

A multimodal config nests the decoder under `text_config`; every reader looks there first, so a
model that is both multimodal and hybrid (Qwen3.5) gets the hybrid layout.

## What is conservative, and why

Under-sizing KV is what OOMs a deployment, so every unknown errs high, and says so:

* **An unknown layer-kind name** is charged full-attention KV, the plan is marked `approximate`,
  its `notes` name the kind, and a `UserWarning` is raised. It is never silently treated as free.
* **A hybrid whose layout no key states** (a hybrid `model_type` with no per-layer key) is charged
  KV on every layer and state on every layer, unless the config states `attention_ratio`.
* **A recurrent mechanism with none of its sizing keys** is sized from fallback geometry and
  marked as an `ESTIMATE` in the plan's notes.
* **A GGUF with a sliding window but no window pattern** (gpt-oss: which layers slide is set in
  llama.cpp's code) is charged full KV on every layer rather than the window on every layer.
* **An HF config with a window but no layout, of a model type the installed transformers does
  not know** is likewise charged full KV on every layer, with a note and a warning. Clamping every
  layer to the window -- what the uniform path used to do for every such config -- under-counts
  any model that alternates (the original Gemma 2 hub config is exactly that shape: 8x low at
  32k tokens). A known type takes its config class's layout, as above. That lookup reads the
  installed `transformers` (BudSimulator requires it; the calculator package alone does not):
  without it, with a release that predates the model type, or with a release before 4.53 (the
  first whose config classes derive `layer_types`: 4.50's `Gemma2Config` states none though
  Gemma 2 alternates), such a config -- Mistral v0.1's included -- is charged full KV on every
  layer with the same warning, an over-estimate.
* **SambaY (Phi-4-mini-flash)**, recognised by its `mb_per_layer` key: its YOCO cross-decoder
  reuses one full-attention cache that grows with the context, and the config does not say which
  layer writes it. Charged: a Mamba mixer every `mb_per_layer` layers, every other layer the
  window, and one layer the full context -- an upper bound on the published design (the
  self-decoder's windowed layers plus one growing cache), never below it. The Mamba state, with
  no SSM key in the config, is an `ESTIMATE`.
* **Windowed layers inside a stack that also has full layers** keep the full sequence unless the
  engine is declared able to hold a per-layer KV layout
  (`engine_capabilities={"heterogeneous_kv_layout": True}`); see `EngineKVCapabilities`.
* **Sparse / compressed attention** is charged as full attention.
* **A pure state-space stack whose config states no state key and no attention-head count** is
  recognised by its `model_type` (`mamba`, `mamba2`, `falcon_mamba`, ...): the one place a family
  name decides a layer kind, and only after every structural key is absent. Its state is then
  sized from fallback geometry and marked as an `ESTIMATE`.
* **Cross-attention** without a vision geometry in the config is charged the full sequence.

## The recurrent state

`calculate_state_memory` (and `EngineMemory.state_bytes`, which `bud_component` reports as
`state_bytes` and counts in `total_runtime_bytes`) is the per-sequence state, constant in
context length: the recurrent matrix in fp32 (unless `mamba_ssm_cache_dtype` / `mamba_ssm_dtype`
/ `inference_state_dtype` says otherwise) plus the convolution's `d_conv - 1` ring buffer at
model precision. One function, `state_memory.recurrent_state_shape`, gives the shape for every
mechanism; the GenZ graph's recurrent operators read the same shape and the same bytes per
element (the state dtype travels in the operator's name: `Rec State` fp32, `Rec State16` 16-bit,
for the recurrent matrix and for the small vectors some mechanisms keep beside it -- RWKV's token
shifts, xLSTM's normaliser; `Rec Conv` at the activation precision), so the graph's state equals
`calculate_state_memory` for every mechanism.

## GGUF

`gguf_layers.hf_layer_keys_from_gguf` translates llama.cpp's keys into the HF ones above, by
llama.cpp's own rules, so a GGUF and its checkpoint resolve to the same plan
(`tests/test_layer_kinds_by_category.py::test_a_gguf_and_its_checkpoint_resolve_to_the_same_kv_and_state`):
`full_attention_interval`; per-layer `attention.head_count_kv` arrays (0 = no KV; with
`feed_forward_length` telling FFN-only slots apart); `attention.sliding_window` with
`attention.sliding_window_pattern`; `attention.shared_kv_layers`; `ssm.{conv_kernel, inner_size,
state_size, group_count, time_step_rank}` (a file with no `ssm.group_count` is Mamba-1, as
llama.cpp's Mamba-1 has no B/C groups); `wkv.head_size`; `shortconv.l_cache`. The reader also
states what the parameter counter needs from the file itself: RoPE from `rope.freq_base` /
`rope.dimension_count` (without it the counter assumed a learned position table: 537M phantom
parameters on Qwen3.5-2B), a gated FFN from an `ffn_gate` tensor, and tied embeddings from the
absence of an `output.weight` tensor. Counted against the files' own tensors: Qwen3.5-2B 98.6 %,
gpt-oss-20b 99.97 %, SmolVLM-500M 100 %.

## The GenZ graph

A recurrent layer is three operators (`genz/Models/recurrent.py`): `Rec Proj`, a GEMM over the
mixer's own parameter count (`mixer_params`); `Rec State`, the fixed state read as an Attend over
`key_dim` rows; `Rec Conv`, the conv buffer. GenZ's "KV Cache" column has always carried Mamba's
state; it now carries every mixer's state, and grows with the context only through the attention
layers. A pure Mamba stack keeps GenZ's own Mamba operators.

All three graph builders read the same per-layer configs, and a windowed layer's window from the
same table: any window spelling, and `attention_chunk_size` for Llama 4's chunked layers. In the
chunked graph a windowed layer reads no context beyond its window (`attention.window_kv_sizes`), an MLA layer reads its one
latent head (`attention.mla_attention_chunked`), and a recurrent layer holds one state per
sequence in the chunk however much each has cached (`recurrent.recurrent_chunked`). A dense layer
of an MoE model (Jamba's even layers, DeepSeek's and Kimi's leading layers) is graphed as one FFN
of `intermediate_size`, not as the experts.

Known approximations: KV-shared layers are still graphed as attention (their KV read traffic is
real, their memory is over-counted); a cross-attention layer is graphed as a full one; a
*uniformly* windowed stack (Mistral v0.1) keeps GenZ's uniform full-attention graph -- the memory
calculator applies its window, the graph does not; a per-layer MoE list (Hunyuan's
`num_experts` per layer) does not graph.

## Adding a new family

If its config uses an existing key, nothing to do. If it introduces a new per-layer kind name,
add the name to `LAYER_KINDS` (until then it is charged full KV with a warning). If it introduces
a new layout key, add a dialect to `resolve_layer_plan`; if a new mechanism, add its state shape to
`state_memory.recurrent_state_shape` and its parameters to `mixer_params`. Add the shipped
`config.json` to `tests/fixtures/hybrid_configs/` and a case to
`tests/test_layer_kinds_by_category.py`.

## Evidence

`docs/proofs/kv-layer-kinds-corpus-diff.md` holds the before/after figures for every model config
the simulator ships or this machine caches (`scripts/kv_corpus.py`).
