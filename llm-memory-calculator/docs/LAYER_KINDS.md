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
| GenZ graph (`get_language_model.layer_configs_from_hf`, `ModelConfig.get_kv_size`) -- the prefill, decode **and chunked** builders (`create_full_chunked_model`: chunked serving, speculative decoding, the simulation engine's chunked feature) | per-layer operators; KV only on attention layers, the window on windowed ones; which FFNs are MoE (`layer_plan.moe_layers`) and how wide a dense one is (`layer_plan.dense_ffn_width`) |
| GenZ serving (`MemoryModel.bytes_per_token_kv` and `state_bytes_per_sequence`, `request_bytes`, the scheduler's admission check, cluster and disaggregation `B_eff`, the disaggregation transfer; the prefix cache's price of a cached token (`PrefixCacheAnalyzer`, BudSimulator `/cache/analyze`); BudSimulator `/api/v2/memory/tiers`) | per-token KV, and the per-request fixed state (`ModelConfig.get_state_bytes`); the weights the device holds beside them (`memory_model.estimate_weight_bytes`: a `ServingSimulator` run's weight load and the `B_eff` budget) from the model's own parameter count -- see *The weights the serving path holds* |
| `UniversalParameterCounter`, `model.facts` (FRD-027), `model.recognise` | weights per mixer, KV per token, the known vocabulary. `model.facts` reads the layer kinds for every config: a `model_type` outside the recogniser is still flagged `architecture: unrecognised`, but which layers hold KV comes from the table, not from that allowlist |
| BudSimulator `ModelConverter.hf_to_genz` | database models graphed like HF configs: the same depth, per-layer kinds, window, Mamba decision and all-recurrent stack's state (`get_language_model.graph_layer_kinds`), the same MLA latent (`kv_lora_rank`, `qk_rope_head_dim`), the decoder read from under `text_config` or an encoder-decoder's own spellings (`layer_plan.decoder_view`), and the encoder's depth from its own stack (`layer_plan.encoder_decoder_stacks`) |
| The vLLM planner (`engine/vllm.py`: `limits`, `predict`; the profile scheduler `perf.sched`; the shared-slice sizing `runtime.modules.capacity`) | the facts' full and sliding layers as KV-cache groups; a running sequence's blocks in every group (`Limits.blocks_per_sequence`) |

The table is `layer_plan.LAYER_KINDS`; the per-model answer is `layer_plan.layer_kinds(config)`
(or `resolve_layer_plan(config)` for hybrids), whose depth is `num_hidden_layers` or, when a
config states only a per-layer list (transformers 5.12's Nemotron-H), that list's length
(`layer_plan.depth`). The GenZ loader, its decoder-block detection and BudSimulator's converter
read the same depth: they used to fall back to 32 layers, so the per-layer configs did not match
and GenZ graphed the saved Nemotron-H-8B as 32 Mamba layers -- `get_kv_size` 0, the serving
memory model, the prefix cache and the disaggregation transfer charging none of its 4 attention
layers' KV (16,384 B a token).
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
| Sliding-window attention | at most `sliding_window` tokens | `layer_types: sliding_attention`; GPT-Neo's `attention_types` (`[[["global", "local"], 12]]`, which its config class expands into `attention_layers`; both read, names `global` / `local`); `sliding_window_pattern: N` (one full layer in N, the last of each group); `sliding_window_layout` (0/1 mask); `global_attn_idx`; window spellings `sliding_window`, `sliding_window_size`, `attention_window_size`, `window_size`; `use_sliding_window: false` disables. A window with **no** layout key takes the layout the model type's own config class derives (transformers: `Gemma2Config` alternates sliding and full, `Qwen2Config` windows from `max_window_layers` on), or, when that class states no per-layer list under any key the reader knows, every layer (Mistral) -- unless the config states `max_window_layers`, whose documented meaning (transformers) is that the first N layers are full and the rest windowed: 4.53's `Qwen2MoeConfig` states no list while its modelling windows only from layer N on | Gemma 2/3, gpt-oss, GPT-Neo, Mistral v0.1, SmallThinker, Hymba |
| Chunked local attention | at most `attention_chunk_size` | `layer_types: chunked_attention` | Llama 4 |
| Sparse / compressed attention | every token (upper bound) | `deepseek_sparse_attention`, `compressed_sparse_attention`, `heavily_compressed_attention`, `minimax_m3_sparse`, `sparse` | DeepSeek-V3.2/V4, MiniMax-M3 |
| MLA (latent KV) | `kv_lora_rank + qk_rope_head_dim`, no factor 2 | `kv_lora_rank` (non-null), `qk_rope_head_dim` | DeepSeek-V2/V3, Kimi |
| Cross-attention layers of their own | the tower's tokens, fixed per request; **no** self-attention cache on those layers | `cross_attention_layers` (in `text_config`); K / V at the stack's own heads and width (`cross_num_key_value_heads` / `cross_head_dim` where stated); the tower's output read as an encoder's is (below): a stated length, else `max_num_tiles x ((image_size / patch_size)^2 + 1)` from `vision_config` -- **one image's**: no config states how many images a request carries, and the plan says so in a note (see *What is conservative*) | Mllama |
| Encoder-decoder (the decoder's self-attention) | every token, on the **decoder's** layers | the decoder's depth: `decoder_layers` (BART, Whisper, Marian, M2M-100, Pegasus), `num_decoder_layers` (T5, mT5, UMT5, LongT5, Switch, ProphetNet; T5's `num_layers` is the encoder's), `decoder_num_hidden_layers` (Moonshine), a nested `decoder_config` / `decoder` block (Dia, EncoderDecoderModel, VisionEncoderDecoder, MusicGen; a block spelling its depth `decoder_layers`, as TrOCR's, included; a block that is itself a whole encoder-decoder config -- Donut's and Nougat's `mbart` -- is read by its `decoder_layers` before its `num_hidden_layers`, which MBartConfig aliases to `encoder_layers`, and its own `encoder_*` keys and `is_encoder_decoder` / `add_cross_attention` flags describe the block, so they are not lifted over the model's), else the stated depth; heads `decoder_attention_heads`, `decoder_num_attention_heads`, `num_decoder_attention_heads` (ProphetNet), `decoder_num_key_value_heads`, T5's `num_heads`; width `d_model`; FFN `decoder_ffn_dim`, `d_ff`; head width `d_kv`; a nested decoder spelled the GPT-2 way (`n_layer`, `n_embd`, `n_head`, `n_positions`, `n_inner`: a ViT-GPT2 captioner's `decoder`) under the decoder-only names its class aliases them to (`layer_plan.DECODER_NAME_ALIASES`, transformers' `attribute_map`), as a flat GPT-2 config is. The encoder's depth: `encoder_layers`, `num_encoder_layers`, `encoder_num_hidden_layers`, a nested `encoder_config` / `encoder` / `text_encoder` (MusicGen) block, else the stated depth; its heads `encoder_attention_heads`, `encoder_num_attention_heads`, `num_encoder_attention_heads`. A whole encoder-decoder nested under `text_config` (BLIP-2 / InstructBLIP with flan-T5, Florence-2) is read there | T5, BART, Whisper, Moonshine, Dia, cohere_asr, ProphetNet, BLIP-2 (flan-T5), Florence-2 |
| Encoder-decoder (the decoder's cross-attention) | the encoder's output tokens, fixed per request, on every decoder layer | the cross-attention's K / V heads `cross_num_key_value_heads` (else `cross_num_attention_heads`, else the decoder's self-attention heads) and width `cross_head_dim` (else the decoder's head width): Dia's are 16 KV heads of 128 against 4 for self-attention. The encoder's output length: `max_source_positions` (Whisper, Speech2Text; the config, its encoder block or `audio_config`), the nested encoder block's `max_position_embeddings` (Dia's 1,024-token text encoder, cohere_asr's 5,000-position audio encoder), an image encoder's patches (`image_size` / `patch_size` of a VisionEncoderDecoder's ViT, each a square int or a [height, width] pair, plus the class token; a staged tower that states `depths` -- Swin, Donut-Swin, FocalNet, NAT, ConvNeXt -- merges 2 x 2 between stages and has no class token: Donut's 2,560 x 1,920 at patch 4 over 4 stages is 80 x 60 = 4,800 tokens, Nougat's 896 x 672 is 28 x 21 = 588), an image encoder's `seq_len` (Pix2Struct: its encoder is the `vision_config` beside a `text_config` that says `is_decoder` and states no encoder of its own -- read as the decoder of that tower whatever the block's own `is_encoder_decoder` says, as transformers' composite config overwrites it: the hub files of google/deplot and pix2struct-* omit the outer flag, the class default `true`, and keep a stale `false` in `text_config`; BLIP's `text_config` is the same shape, and BlipTextLayer builds its cross-attention exactly when `is_decoder`); else unstated (T5, BART, Moonshine, BLIP-2's flan-T5, Florence-2): charged over the sequence, see *What is conservative* | Whisper, Dia, cohere_asr, TrOCR, Donut, Nougat, Pix2Struct, BLIP, MusicGen, T5, BART |
| KV-shared layers | none of their own | `num_kv_shared_layers` (the last N layers) | Gemma 3n |
| Gated DeltaNet | fixed: `nv x dk x dv` fp32 + conv | `layer_types: linear_attention` with `linear_num_value_heads`, `linear_key_head_dim`, `linear_value_head_dim`, `linear_num_key_heads`, `linear_conv_kernel_dim`; `full_attention_interval` | Qwen3-Next, Qwen3.5/3.6 |
| Plain linear attention | fixed: `heads x d x d` | `linear_attention` without GDN keys; `attn_type_list` (0 = linear) | MiniMax-Text-01 |
| Kimi Delta Attention | fixed | `linear_attn_config.{full_attn_layers, kda_layers}` (one-indexed, as Kimi's modelling code reads them, `layer_idx + 1`; only a list naming layer 0 is read 0-indexed) | Kimi Linear |
| Mamba-1 / Mamba-2 | fixed: `heads x d_head x d_state` fp32 + conv | `layer_types: mamba`, `layers_block_type`, `hybrid_override_pattern` (`M`/`E`/`*`/`-`, as transformers' `NemotronHConfig` translates them), `attn_layer_indices`, `attn_layer_period`/`attn_layer_offset`, state keys `state_size`/`mamba_d_state`/`ssm_state_size`, heads `mamba_n_heads`/`mamba_num_heads`, `mamba_head_dim`, groups `n_groups`/`mamba_n_groups` | Mamba, Jamba, Nemotron-H, granite-4.0-h, Bamba, Zamba2, Falcon-H1 |
| Short convolution | fixed conv buffer | `layer_types: conv`, `conv_L_cache` | LFM2 |
| RWKV | fixed: token shift + WKV | no attention heads + `attention_hidden_size`; GGUF `wkv.head_size` | RWKV |
| xLSTM (mLSTM) | fixed matrix memory | `qk_dim_factor`, `v_dim_factor`, `num_heads` | xLSTM |
| RG-LRU | fixed vector + conv | `block_types` (repeating), `lru_width`, `conv1d_width`, `attention_window_size` | RecurrentGemma |
| FFN-only slot | none | `moe`, `dense`, `mlp` in a layer list; `-` and `E` in `hybrid_override_pattern`. A stack that names such slots gives the FFN its own layer: only those carry one | Nemotron-H |
| No mixer | none | `no_op`: a layer whose mixer holds no KV and no state, and that runs its FFN like every other layer of its stack (not an FFN-only slot). Named by the GGUF reader for a 0-KV-head layer with no recurrent mixer | Deci GGUFs (Llama-3.x-Nemotron Super / Ultra) |
| MoE | does not change KV; decides which FFN a layer runs | `mlp_layer_types` (dense / sparse), `moe_layers`, a per-layer expert count, `expert_layer_period`/`expert_layer_offset`, `decoder_sparse_step` + `mlp_only_layers`, `interleave_moe_layer_step`, `moe_layer_start_index`/`moe_layer_end_index`/`moe_layer_interval`, `first_k_dense_replace`/`moe_layer_freq`; a `moe` FFN slot. A dense layer's width when the config names it apart from the experts': `intermediate_size_mlp` (Llama 4), `dense_intermediate_size` (MiniMax-M3), `prefix_dense_intermediate_size` (Cohere2-MoE); otherwise `intermediate_size` | Mixtral, DeepSeek, Kimi, Jamba, Qwen-MoE, Llama 4, ERNIE |

Parallel hybrids (attention and a mixer in the same layer: Falcon-H1, Hymba, Zamba's `hybrid`
layers) set both axes on that layer: it holds KV *and* a state.

Zamba / Zamba2 name their layers through the same table (`mamba`, `hybrid`; any other name is
charged full attention with the warning). A `hybrid` layer invokes the shared attention+MLP
block, so it runs that block's MLP too; a plain `mamba` layer has no FFN. The block's weights
exist once per shared block (`num_mem_blocks`, one when the config states none), so the
parameter counts (`UniversalParameterCounter`, `model.facts`) count them that many times, while
the graph streams them on every layer that invokes the block.

### How wide a KV layer is

A KV layer holds `2 x kv_heads x head_dim` elements a token (MLA: its latent). The head width is
read once, by `layer_plan.attention_head_dim`, for every reader -- the calculator's KV and
breakdown, the engine estimate, the parameter counter, the FRD-027 facts, the GenZ graph (and so
`get_kv_size`, the serving `MemoryModel`, the prefix cache and the disaggregation transfer) and
BudSimulator's model converter. In order:

1. a stated width, under any spelling transformers aliases `head_dim` to (`attribute_map`):
   `head_dim`, `attention_head_dim` (Zamba, Zamba2), `kv_channels` (JetMoE), `d_kv` (T5). Zamba2's
   hub config also ships Megatron's `kv_channels: 80` beside `attention_head_dim: 160`; transformers
   builds its K and V projections at 160, so `attention_head_dim` comes first;
2. the attention's own input width over the heads, `attention_hidden_size` (Zamba's attention reads
   the hidden state concatenated with the embeddings: 2 x hidden);
3. the width the model type's config class derives under another name -- its `head_dim` alias, or
   a width argument of its constructor in a release without the alias (transformers installed:
   `ZambaConfig` gives `2 x hidden / heads` to a config that leaves both keys out, in 4.50 and
   5.12 alike);
4. only then `hidden_size / heads`.

Step 3 imports the installed `transformers` (once a process, then cached per config) for a config
that states no width under any spelling; without transformers it is skipped.

Reading `head_dim` alone and deriving the rest from the hidden size charged Zamba2-2.7B half its KV
(92,160 B a token against 184,320) everywhere, and counted its shared attention's weights at a
quarter. The parameter counter also takes Q / K / V's input width from `attention_hidden_size`
(`attention_input_width`). A test builds every fixture's own transformers config class and checks
each reader's per-layer width against it; another checks that the spellings above cover every
`head_dim` alias transformers states (GLM-4-MoE-Lite's, `qk_rope_head_dim`, is MLA's RoPE width:
an MLA cache is its latent, sized by its own path).

A multimodal config nests the decoder under `text_config`; every reader looks there first, so a
model that is both multimodal and hybrid (Qwen3.5) gets the hybrid layout.

### Encoder-decoder models

An encoder-decoder's per-token KV cache is its **decoder's** self-attention. Such a config states
the decoder under its own spellings (the encoder-decoder row above, each checked against
transformers 5.12's configuration classes and the hub configs in
`tests/fixtures/encoder_decoder/`), and `layer_plan.decoder_view` puts them under the
decoder-only names (lifting a nested decoder block as `text_config` is lifted) for every reader:
`calculate_kv_cache` and `kv_cache_breakdown`, `calculate_memory`'s encoder-decoder getters (and
so the engine estimate), the FRD-027 facts, the GenZ loader (and so `get_kv_size`, the serving
memory model and the disaggregation transfer) and BudSimulator's converter.
`layer_plan.encoder_decoder_stacks` gives both depths. A config is one when it says so
(`is_encoder_decoder`), states a decoder depth of its own or nests both stacks; any other config
is read as it stands. A multimodal config whose `text_config` is itself an encoder-decoder
(BLIP-2 and InstructBLIP with flan-T5, Florence-2: the outer config may say `is_encoder_decoder`
and state no stack) is read in its `text_config`, which `decoder_view` views in place; the GenZ
loader's and the converter's search for a nested decoder views every candidate block the same
way. Passing such a block over (it has no `hidden_size` or `num_hidden_layers`) graphed BLIP-2's
12-layer, 768-wide Q-Former as the language model in GenZ and, from round 7, in the converter: KV
18,432 elements a token against the decoder's 98,304. Read under the decoder-only names alone:

* `calculate_kv_cache` and the facts charged Whisper the encoder's depth (32 layers, not the
  turbo decoder's 4), `calculate_kv_cache` charged Whisper, BART and Dia no KV at all (no head
  count where it looked), and the facts gave flan-t5-xl a default 16 heads for its 32;
* `calculate_memory`'s getters read Whisper's spellings only: flan-t5-xl's decoder was 4 layers of
  20 heads (Whisper defaults) against 24 of 32, Dia's 4 of 20 against 18 of 4 KV heads;
* GenZ refused T5, BART and Whisper outright (a width spelled `d_model` beside a nested block);
* BudSimulator's converter split the decoder's depth in two (`num_hidden_layers // 2` whenever
  `encoder_layers` was absent): Dia's 18 decoder layers became 9 and flan-t5-xl's 24 became 12
  with a default 32 heads -- under the true decoder KV, and under origin/main's.

The decoder's **cross-attention** cache -- each decoder layer's K and V over the encoder's
output, computed once a request and read at every decode step (transformers'
`EncoderDecoderCache.cross_attention_cache`, vLLM's `CrossAttentionSpec`) -- is read by one
function, `layer_plan.cross_attention_cache` (the decoder's depth, the cross-attention's own K / V
heads and width, the encoder's output length; the cross-attention row above), and charged by every
reader of a request's KV:

* `calculate_kv_cache` / `kv_cache_breakdown`: a cross group (`num_cross_layers`, `kv_width`) over
  the stated encoder length, else over the sequence; it adds to `marginal_bytes_per_token` only
  when unstated (a stated one does not grow).
* `calculate_memory` and the engine estimate: `calculate_cross_attention_kv_cache` reads the same
  function, so the cross-attention is as wide as it is built (Dia: 16 KV heads of 128, 2,048 wide;
  it was taken as the 1,024-wide encoder's) over `encoder_seq_length`, else the stated length, else
  the sequence (it was Whisper's 1,500 frames for every encoder). The report still adds its own
  encoder self-attention term.
* GenZ's `ModelConfig` (the HF loader and BudSimulator's converter build it through one function,
  `encoder_decoder_fields`): `get_cross_kv_bytes` is the stated-length cache a request,
  `get_request_fixed_bytes` adds it to the recurrent state, and the serving `MemoryModel` charges
  a request those fixed bytes once, with its first blocks (`state_bytes_per_sequence`, so the
  cluster and disaggregation `B_eff` too); the prefill-to-decode transfer moves them. Unstated,
  `get_kv_size` carries the cache a token. whisper-large-v3-turbo, a 1,024-token request: 20.97 MB
  of self-attention + 30.72 MB of cross-attention = 51.69 MB (round 8: 20.97 MB; origin/main
  refused the config).
* The FRD-027 facts: `cross_bytes_per_sequence` (the stated length, else the caller's context) is
  in `kv_bytes_for_context`; `kv_bytes_per_token` stays the self-attention a token adds. The vLLM
  planner holds it per running sequence beside its self-attention blocks, at the stated length or
  the request's `context.max_tokens` (a text encoder's input), in `Limits.state_bytes_per_sequence`
  -- so `predict`, the profile scheduler (`perf.predict.sim_config`) and the shared-slice sizing
  (`runtime.modules.capacity.shared_slice_mib`) all charge it -- and in the one-request boot
  check. Dia at 1,024 + 256 tokens on an 80 GB card: 357 running sequences (self 47 MB + cross
  151 MB each), where round 8 gave 1,024 and origin/main 58.
* `KVGeometry` (the KV planner): `cross_bytes_per_sequence`, and the KV layers are the decoder's
  (Whisper-turbo's 4, not the encoder's 32). Its per-token bytes, and the prefix-cache analyser's,
  stay the self-attention a cached token holds: the cross cache is the encoder's output, which no
  prefix tier stores (vLLM keeps no prefix cache for an encoder-decoder) -- unless the length is
  unstated, when it is in the per-token figure (an over-count for a prefix).
* The GenZ graphs: every decoder layer's cross-attention (`attention.cross_attention`: its query
  and output projections, its scores and values over each sequence's cached encoder K / V) in the
  decode, prefill and chunked graphs, so their `KV Cache` carries it (whisper-large-v3-turbo at
  4,096 tokens: 80 MiB self + 29.3 MiB cross). The prefill graph also runs the encoder once
  (`encoder_stack`: its layers over the encoder's tokens, bidirectional attention and a dense FFN
  at the encoder's own width, `layer_plan.encoder_view`) and each decoder layer's K / V projection
  of the encoder's output, so its weights are the whole model's (whisper-large-v3-turbo 1,653 MiB
  against 809M parameters = 1,542 MiB, the difference GenZ's separately graphed tied classifier).
  The prefill graph's `KV Cache` includes the encoder's self-attention K / V over its tokens, as
  `calculate_memory`'s encoder term does.
* The device holds the whole model while it decodes, so the decode and chunked graphs carry the
  encoder (`encoder_stack(..., resident=True)`) and the decoder's cross K / V projections as GEMMs
  over no tokens: their weights count in `Total Weights`, which `decode_moddeling`'s and
  `chunked_moddeling`'s fit checks and `platform_size.get_minimum_system_size` add to the KV, and
  they take no step time (an operator with no compute takes none -- the mechanism already holding
  an MoE's unrouted experts). Round 8 left them out: Whisper large-v3's decode graph weighed
  1,526.6 MiB of its ~2,942 MiB, so a 3 GB device passed the decode fit check that the prefill
  check refuses; it now weighs 2,926.6 MiB and is refused. The one table no GenZ decode graph
  carries, for any model, is the input embedding (the prefill graph's `embeddings` GEMM): a
  pre-existing gap of `vocab x hidden` in every decode fit check (Llama-3-8B: 1,002 MiB), left as
  origin/main has it.
* GenZ's runtime modelling (`prefill_moddeling`, `decode_moddeling`, `chunked_moddeling`, and the
  `estimate_*_performance` built on them) books every operator these graphs emit in
  `get_runtime_breakdown` (`attention.CROSS_PROJECTION_OPS`, `CROSS_SCORE_OPS`,
  `CROSS_COLLECTIVE_OPS`: MHA, its QKVO / attention-kernel / all-reduce buckets). Round 8 added
  the operators without booking them, and every encoder-decoder raised `Layer Name:Enc QKV not
  found in the breakdown function` (prefill) or `Cross Q` (decode, chunked), where origin/main
  answered.

A decoder-only stack's **own cross-attention layers** (Mllama's `cross_attention_layers`: 8 of its
40 layers attend to the vision tower's output and to no text) are read by the same function:
`cross_attention_cache` returns them with `own_layers` -- the layers the table marks
`ATTN_CROSS`, at the stack's K / V width, over the tower's tokens (the tower is read as an
encoder block is: a stated length, else its patches and tiles; MllamaConfig's 4 tiles of
448 / 14 patches + 1 = 4,100). Those layers hold no self-attention cache, and every reader
charges them the cross cache only: the calculator (one cross group, as for an encoder-decoder),
the FRD-027 facts (they are not counted as full layers; their projections still count in the
weights), GenZ (layer kind `Cross`: `attention.cross_attention(..., own_layer=True)`, the query /
output projections and the reads over the encoder's K / V, the K / V projection at prefill and
held at no time elsewhere) and so the serving model, the disaggregation transfer and
BudSimulator's converter. MllamaConfig at 1,024 tokens, bf16: 32 x 131,072 B a token x 1,024 +
8 x 2 x 1,024 x 4,100 x 2 B = 268,566,528 B in every reader; round 8's GenZ, converter and facts
graphed the cross layers as self-attention, 167,772,160 B (origin/main's converter: 536,870,912 B,
every layer full).

The composition is read from the configs **as the hub ships them**, which older transformers
releases saved with flags the composite class overwrites on load. Three shapes, each in
`tests/fixtures/cross_attention/*_hub.json` and `tests/test_cross_attention_hub_shapes.py`:

* google/deplot (and pix2struct-docvqa-base / -textcaps-base) omit the outer
  `is_encoder_decoder` (Pix2StructConfig's class default, `true`) and keep `is_encoder_decoder:
  false`, `is_decoder: true`, `num_heads: 12`, `d_kv: 64` in `text_config`.
  `layer_plan.composed_text_config` reads a `text_config` that says `is_decoder`, holds no
  encoder of its own and sits beside a tower as that tower's decoder, flagged as transformers
  flags it; the parameter counter reads the same block. BLIP's hub config is the same shape.
  The FRD-027 facts read the head count by `layer_plan.attention_heads`, the reader that goes
  with the head width (`num_heads` beside `d_kv`): they had read 6 default heads of 64. deplot at
  1,024 tokens: 36,864 B a token + 150,994,944 B of cross-attention, 45 requests in 8 GiB (round
  10: 18,432 B a token in the facts, 98,304 in GenZ and the serving model, no cross anywhere,
  85 requests).
* facebook/musicgen-small's nested `decoder` carries the `to_dict` defaults
  `is_encoder_decoder: false`, `add_cross_attention: false` under the outer `true`. Lifting them
  over the model made `decoder_view` deny its own cross-attention, and the calculator -- which
  viewed the decoder before asking for the cross cache -- charged none while every other reader
  charged 24 layers. A nested block's composition flags and its own `encoder_*` keys are no
  longer lifted (`layer_plan._BLOCK_OWN_KEYS`), so `cross_attention_cache(decoder_view(c))` is
  `cross_attention_cache(c)` for every config fixture (a test walks them all), and the
  calculator's plan reads the config as stated. musicgen-small at 4,096 tokens: 0.805 GB
  (round 10: 0.403). The facts now also count its text encoder and cross-attention weights, as
  for every encoder-decoder (the planner's slice grows 512 MiB).
* naver-clova-ix/donut-base and facebook/nougat-small nest an `mbart` decoder block whose
  `num_hidden_layers` (12) is MBartConfig's alias of `encoder_layers`; its decoder is
  `decoder_layers` (4). A nested block's depth is read for its role (`_block_depth(block,
  "decoder")`: the role's own spelling first). Their Swin encoders state `image_size` as
  [height, width]; read with their stages, Donut hands the decoder 4,800 tokens and Nougat 588.
  donut-base at 1,024 tokens: 4 x 2,048 x 2 B x (1,024 + 4,800) = 95,420,416 B in every reader
  (round 10: 12 layers in the facts, KV geometry, GenZ and converter, 4 over the sequence in the
  calculator).

A **parameter count** of a multimodal model whose `text_config` is an encoder-decoder (BLIP-2 /
InstructBLIP with flan-T5, Florence-2) counts that language model as one
(`UniversalParameterCounter._calculate_text_model_params`): the decoder under `decoder_view`, its
cross-attention, and the encoder under `encoder_view` (the embedding shared). The attention's head
count is read by the function that goes with its head width (`layer_plan.attention_heads`, T5's
`num_heads` beside `d_kv`), and a config that states its FFN gated is three matrices in the
counter and in GenZ alike (`layer_plan.stated_gated_ffn`: the T5 family's `feed_forward_proj:
gated-<act>` / `is_gated_act`, DINOv3's `use_gated_mlp`; flan-T5 names its activation
`gelu_new`, which neither the counter's model-type list nor GenZ's activation test reads as gated).
BLIP-2 with flan-t5-xl (`Blip2Config(text_config=T5Config(...)).to_dict()`): 3,835,230,720
counted, against transformers' meta count of 3,876,648,448 (98.9 %; the counter has no Q-Former
term, and its vision-tower term is its own approximation);
origin/main 1,720,216,064, round 8 1,595,173,376 (12 heads of 64 over a default 12 layers). The
flan-t5-xl fixture's GenZ prefill graph: 5,435 MiB, transformers' 2,783,959,040 parameters plus
GenZ's separately graphed tied classifier (round 8: 4,475 MiB, a 2-matrix FFN).

An attention head count spelled `num_heads` or `n_heads` (GPT-Neo, T5, MPT) is read as the
attention heads (`ConfigNormalizer`) only when the table says the stack has attention layers:
xLSTM's `num_heads` belong to its recurrent matrix memory. Unread, GPT-Neo was charged no KV at
all. Nor is GPT-Neo's `attention_layers` list a hybrid marker any more (`detect_model_type`).

The weights beside the decoder are each read at their own geometry, by the readers that size
them (round 12). The FRD-027 facts, which the vLLM planner reserves before it sizes the KV blocks,
count an encoder-decoder's encoder through `stack_params.encoder_params` -- a hierarchical tower's
stages (Donut's and Nougat's Swin: 20 blocks over 128-1,024 wide, which had been 4 blocks at the
decoder's 1,024; `layer_plan.encoder_stages` also gives every reader its depth, 20 not 4), T5's
gated FFN, a conformer's two FFNs and convolution module -- and an adapter stack between a tower
and the language model through `stack_params.adapter_params` (BLIP-2's and InstructBLIP's
Q-Former, its cross-attention layers and its projection: 106.7M the facts had left out). A tower's
class defaults stand in for the keys its config leaves out (Mllama's 8 global layers, its
7,680-wide output and its tiled position tables). The parameter counter counts a tower at its own
geometry the same way, a wrapper around a language model alone from its `text_config`, and the
activation and tied-embedding flags at the config class's defaults where a config states none
(Llama 4's "silu" and untied head). Those defaults ship with the package
(`src/llm_memory_calculator/config_class_defaults.json`, written from transformers 5.12's 642
config classes by `scripts/config_class_defaults.py`; `layer_plan.CLASS_DEFAULT_KEYS` names every
key a reader asks for), so every figure is the same whether transformers is installed, absent, or
a release that predates the class (4.50 has no Llama 4 or Nemotron-H); the installed transformers
is asked only about a model type the snapshot does not hold. Read from the installed release
alone, a machine without transformers counted Mllama's tower without its global layers and a
tied head on RWKV and Llama 4 -- weights below origin/main's and below the model's own count. A
query / key norm is read under every spelling (`layer_plan.QK_NORM_KEYS`: `qk_norm`,
Hunyuan's `use_qk_norm`, `qk_layernorm`). Every reader takes an encoder-decoder's shape from the same
reader: a flat config that states two stacks under an encoder-decoder's spellings
(`encoder_decoder_stacks`: mT5's `num_layers` and `num_decoder_layers`) is an encoder-decoder to
`detect_model_type` and the parameter count as it is to the layer plan.

A nested decoder spelled the GPT-2 way -- a VisionEncoderDecoder's or EncoderDecoderModel's
`decoder` with `n_layer`, `n_embd`, `n_head` (a ViT-GPT2 captioner) -- is read under the
decoder-only names its class aliases those to (`layer_plan.DECODER_NAME_ALIASES`: GPT2Config's
`attribute_map`, and GPT-J's, CodeGen's, GPT-BigCode's, CTRL's, OpenAI-GPT's and BLOOM's, which
a test checks against transformers; GPT-2's FFN width `n_inner`), by `decoder_view` for the
nested block and for a flat config alike, and `layer_plan.depth` reads `n_layer`. Unread, the
block was no decoder: `calculate_kv_cache` charged 0 KV, and GenZ, the serving path and the
prefix cache raised (`Could not locate the decoder's geometry`), on origin/main as on round 13.
The harness's ViT-GPT2 shape now holds 6 x 2 x 768 K + V a token and 6 x 2 x 768 x 197 of
cross-attention in every reader (the ViT's (224 / 16)^2 + 1 tokens); a flat GPT-2 config is
graphed at its own 12 x 768 (GenZ graphed the generic 32 x 4,096) and the KV planner charges its
36,864 B a token (it charged 0).

## The weights the serving path holds

`genz.serving.memory_model.estimate_weight_bytes` is the weight a serving simulation loads into
the device tier (`ServingSimulator._estimate_weight_bytes`) and what the cluster and
disaggregation `B_eff` take off the device before they divide the rest by a request's KV and
fixed bytes. It is the model's own parameter count (`memory_model.model_parameter_count`) at the
serving precision, split by tensor parallelism: a GenZ `ModelConfig` read from a model config --
by the HF loader or by BudSimulator's converter -- carries that config (`source_config`), and its
parameters are the FRD-027 facts' count of it (`params_total + vision_params`): the decoder, an
encoder-decoder's encoder and cross-attention, every tower and adapter beside the language model
(a vision tower, an audio encoder -- `audio_config` / `audio_encoder_config`, read at its own
geometry by `stack_params.tower_view`, with its convolutional front end and its projector -- and
BLIP-2's Q-Former), every routed and shared expert. That is the count the vLLM planner reserves
before it sizes the KV blocks. A `ModelConfig` built by hand (GenZ's built-in dictionary) has no
config to count: it is counted from its own fields, with its routed and shared experts on the
layers it places them (`layer_type`), each Mamba layer as its Mamba mixer
(`mixer_params.mamba1_params` over its `mamba_*` fields, with no FFN in a stack of Mamba layers
alone), its token embedding and an LM head (no tying stated: an upper bound) and its norms; an
object with the decoder fields alone keeps the decoder formula. The facts count every recurrent
layer other than a Gated DeltaNet one by its own mechanism's count (`mixer_params`: Mamba-1,
Mamba-2, Kimi Delta Attention, lightning attention, short convolution, RWKV, mLSTM, RG-LRU),
where they had counted them all with one copy of a Mamba-2 formula that left out the
convolution's bias: Mamba-Codestral sat 393,216 parameters below transformers', and Kimi-Linear
137,675,680 below its checkpoint (its KDA layers counted as Mamba-2; KDA's own count,
`mixer_params.kda_params`, lacked the per-head beta projection `b_proj` [32, 2304] too). Bamba-9B-v2, Falcon-H1-34B and Kimi-Linear are now
their truths to the parameter. GenZ's built-in Falcon-Mamba 7B states the hub config's 65,024-token vocabulary (it
said 50,280, state-spaces/mamba's).

The formula it replaced counted `num_decoder_layers` x (attention + 3 x hidden x intermediate)
plus one embedding of whatever the `ModelConfig` held. Once the loader and the converter gave an
encoder-decoder's `ModelConfig` its own decoder (round 7 on), and the converter a multimodal
model's language model, that was the decoder alone: Whisper large-v3 1.81 GB of its 3.09 GB, a
flan-T5-xl 2.45 of 5.57, Qwen3.5-4B 7.48 of 9.08, NLLB-MoE 4.27 of 109 -- below origin/main
(whose reading of the same configs fell back to a generic 7B and happened to sit higher) and
below the model -- and the cluster admitted more requests than the device holds: Whisper
large-v3 on a 40 GB A100 at 4,096 tokens, 44 where the memory holds 43 (origin/main 13). It now
loads 3.62 GB (the facts' 1,811M parameters: 17 % over transformers' 1,543M) and admits 42. A
built-in MoE model counted one expert: Mixtral-8x7B weighed 14.2 GB on origin/main and was given
53 concurrent requests on a 40 GB card it does not fit; it now weighs 93.4 GB (its 46,702,792,704 parameters) and is given 1.
`scripts/kv_corpus.py` measures the figure (`serving_weight_bytes` through a `ServingSimulator`,
`conv_serving_weight_bytes` on the converter's `ModelConfig`) and both concurrencies
(`cluster_b_eff_4096`, `disagg_b_eff_4096`, uncapped by the scheduler's batch) for every row, and
the gate holds them: the weights to the model's every parameter at 2 B, a concurrency to the three
memory figures it is sized from.

## What is conservative, and why

Under-sizing KV is what OOMs a deployment, so every unknown **layer kind** errs high, and says so
(the weights are a separate matter: see *What errs low, measured* below, which names every figure
the gate holds below its ground truth and why):

* **An unknown layer-kind name** is charged full-attention KV, the plan is marked `approximate`,
  its `notes` name the kind, and a `UserWarning` is raised. It is never silently treated as free.
* **Layers named sliding / local whose window the config states under no key the table reads**
  (`sliding_window`, `sliding_window_size`, `attention_window_size`, `window_size`, or
  `attention_chunk_size` for chunked layers -- ModernBERT's `local_attention` is not one) are
  charged full-attention KV, in the table itself (`layer_plan._sliding_bound`), with a note, a
  warning and the `approximate` flag. They used to stay sliding with no window: the calculator,
  GenZ and the facts' own KV treated them as full, but the vLLM planner charged a sliding group
  with no window zero blocks a request. Every such config in the corpus is a ModernBERT-family
  encoder (back to all-full, as on main).
* **A per-layer list shorter than the depth** (`layer_types`, `layers_block_type`,
  `hybrid_override_pattern`, `attn_type_list`, `sliding_window_layout`, GPT-Neo's lists): the
  layers it names keep their kinds, the layers it does not reach are charged full-attention KV,
  with the same note, warning and `approximate` flag. They used to be left with no kind at all,
  so charged nothing.
* **A hybrid whose layout no key states** (a hybrid `model_type` with no per-layer key) is charged
  KV on every layer and state on every layer, unless the config states `attention_ratio`.
* **A config that states an SSM state key and attention heads but no layout key** (none of the
  per-layer lists, indices, periods or patterns above) is read as a parallel hybrid: every layer
  charged a growing KV *and* a fixed state, with a note and a warning. That is exactly Falcon-H1,
  and an upper bound for anything that interleaves instead. The calculator, the facts, the GenZ
  graph and the serving path all read it from the table; GenZ used to take the state key alone as
  "every layer is Mamba", so the serving path charged such a model 0 KV a token while the
  calculator charged full KV and no state.
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
  layer with the same warning, an over-estimate. Config classes gained `layer_types` in different
  releases after 4.53 (4.53's `Qwen2MoeConfig` has none, while its modelling windows only layers
  >= `max_window_layers`), so a class that states none makes the window uniform only when the
  config states no `max_window_layers`. That key only ever takes the window off layers, so a
  release whose modelling windows every layer anyway is over-counted, never under-counted.
* **SambaY (Phi-4-mini-flash)**, recognised by its `mb_per_layer` key: its YOCO cross-decoder
  reuses one full-attention cache that grows with the context, and the config does not say which
  layer writes it. Charged: a Mamba mixer every `mb_per_layer` layers, every other layer the
  window, and one layer the full context -- an upper bound on the published design (the
  self-decoder's windowed layers plus one growing cache), never below it. The Mamba state, with
  no SSM key in the config, is an `ESTIMATE`.
* **Windowed layers inside a stack that also has full layers** keep the full sequence *in the
  memory calculator* (`calculate_kv_cache`, `kv_cache_breakdown`, the engine estimate) unless the
  engine is declared able to hold a per-layer KV layout
  (`engine_capabilities={"heterogeneous_kv_layout": True}`; see `EngineKVCapabilities`): the
  measured vLLM gpt-oss behaviour. That is a statement about the engine, not the model, and only
  the calculator takes an engine. The two readers that describe the **model** apply the window
  on every windowed layer, whatever its stack: the GenZ prefill, decode and chunked graphs (a
  windowed layer reads and holds no more than its window, as an engine with per-layer KV -- vLLM's
  hybrid allocator, llama.cpp's SWA cache -- holds it), and the FRD-027 facts (whose vLLM model
  then lays out its own sliding groups, `engine/vllm.py`). The vLLM planner charges a running
  sequence its blocks in **every** group of that layout -- each full group the whole sequence,
  each sliding group up to the window plus one block -- in its concurrency (`predict`), its profile
  scheduler and its shared-slice sizing (`Limits.blocks_per_sequence`). Sizing concurrency as
  `kv_tokens // tokens` counted one group: Gemma-3-27b (1 full and 6 sliding groups of 10 layers)
  was given 160 running 1,280-token sequences where the allocator holds 27, Gemma-2-9B 246 where
  it holds 123 -- more than main, which read those stacks as all full and gave 25 and 123. So for an engine without per-layer KV,
  the graph's `KV Cache (MB)` -- and `platform_size.get_minimum_system_size`, which adds it to the
  weights -- is below the calculator's default: Gemma-3-27b-it at 4,096 tokens, decode graph
  736 MiB against 1,984 MiB with the window withheld; a Llama 4 text stack at 32k, 2,688 MiB
  against the calculator's 6.44 GB. Size a deployment on such an engine from the calculator.
* **Sparse / compressed attention** is charged as full attention.
* **A pure state-space stack** is read structurally: an SSM state key and no attention-head count
  of any spelling. Only a config with no attention-head count *and* no state key is recognised by
  its `model_type` (`mamba`, `mamba2`, `falcon_mamba`, ...): the one place a family name decides a
  layer kind, and only after every structural key is absent; its state is then sized from fallback
  geometry and marked as an `ESTIMATE`. A config that states attention heads is never made
  attention-free by its name (a `mamba`-named config with heads and a state key is the parallel
  hybrid above).
* **Cross-attention layers of their own** (Mllama) whose tower states no length or geometry are
  charged the full sequence, with a note in the plan -- **not** a bound when the tower outputs more
  tokens than the sequence holds.
* **Cross-attention layers of their own are charged one image** (Mllama: 8 layers x 2 x 1,024 x
  6,404 tokens x 2 B = 209,846,272 B for the 11B-Vision config's 4 tiles of (560 / 14)^2 + 1).
  transformers reshapes the cross-attention states to `num_concurrent_media` x tiles x patches,
  so a request carrying N images holds N times that; the image count is the request's, as its
  length is, and no config states it. The plan says so in a note (`one image's 6404 tower
  tokens`). This is **not** a bound for a multi-image request: such a request is under-counted
  by (N - 1) x that figure, and at long context the per-request total can fall below
  origin/main's, which charged the 8 layers as self-attention over the sequence (11B-Vision at
  32k: 4.505 GB against origin/main's 5.369; the vLLM planner admits 41 sequences at 8k against
  39). An encoder-decoder's image encoder (Donut, Nougat, Pix2Struct, BLIP, TrOCR) reads one
  image a request by construction.
* **A staged image tower's tokens** (`depths`) assume the 2 x 2 patch merging between stages that
  Swin, Donut-Swin, FocalNet, NAT and ConvNeXt perform. A staged tower that kept its full grid
  would hold 4^(stages - 1) times more; none is known among the configs read.
* **A `text_config` that says `is_decoder` beside a tower** is charged the cross-attention cache
  over the tower's output on every layer (Pix2Struct, BLIP build exactly that). A decoder that
  states `is_decoder` and does not attend to the tower would be over-counted, never under.
* **An encoder-decoder whose encoder's output length the config does not state** (T5, BART,
  Marian, ProphetNet, BLIP-2's flan-T5, Florence-2, Moonshine, MusicGen) has its cross-attention
  cache charged over the whole sequence (the calculator, GenZ, the serving model, the converter)
  or over the request's `context.max_tokens` (the facts and the vLLM planner), with a note in the
  plan. It is an upper bound only for a text encoder whose input is the request's prompt alone
  (T5, BART, Marian, ProphetNet, MusicGen's T5): that input is never longer than the sequence. It
  is **not** a bound where the encoder's input is longer than the prompt or is not the prompt:
  Florence-2's BART encoder reads the image's tokens (577 at 768 px, a processor setting no model
  config states) before the prompt; BLIP-2's and InstructBLIP's flan-T5 encoder reads the
  Q-Former's `num_query_tokens` (32) before it; and an audio encoder that states no length
  (Moonshine's frames follow the audio's duration, not the text) holds as many frames as the
  recording has -- a long recording over a short transcript is charged less than it holds. State
  `encoder_seq_length` to `calculate_memory` for those. Stated lengths are each the encoder's
  maximum (`max_source_positions`, an encoder block's `max_position_embeddings` or `seq_len`):
  cohere_asr's 5,000 positions over-count a 30-second clip's 375 frames 13x, and Pix2Struct's
  4,096 patches a request processed at `max_patches` 2,048 (its processor's default) 2x.
* **An encoder-decoder's weights in the GenZ decode graph** are the resident model's (the encoder
  and the cross K / V projections held at no step time; see "Encoder-decoder models"), so its fit
  check is not below the prefill check's. The input embedding table is the exception for every
  model, as on origin/main.

## What errs low, measured

`scripts/kv_corpus.py --gate` holds every figure the branch moves in the unsafe direction against
origin/main to a ground truth (`tests/fixtures/kv_gate/allowlist.json`; *The gate* below), and
over the whole corpus none is below it but by modules its entry names as no config key's (the
serving weights of two checkpoints, below). The readers' figures that stay below the model's own
count -- each the same or higher than origin/main, or held by such a named allowance -- are
these, stated rather than hidden:

* **The GenZ graphs' weights hold matrices only.** No operator carries a norm or bias vector, nor a
  few small matrices (T5's relative-position bias, Mllama's 8 extra embedding rows): the graphs
  sit up to 5 parts in 100,000 below transformers' parameter count (DeepSeek-V3: 1.9 MiB of
  1.28 TiB). The decode graph looks a token up in its input embedding and its learned position
  table, if any (gathers), and graphs no operator for either, on origin/main as here, so its
  weights are the prefill graph's less those tables (Nougat: 7.0 MiB of positions). The chunked
  graph holds them, and is held to the prefill graph's truth.
* **The calculator's count for BLIP's BERT text decoder** leaves out its bias vectors, its learned
  positions and its LM head's transform: 223.21M of 223.97M (origin/main 223.11M).
* **A conformer encoder in the calculator and the GenZ graph** (cohere-transcribe's Parakeet
  encoder: two FFNs and a convolution module a layer) is counted as plain blocks: 581M and 1.47B of
  2.05B (origin/main 302M, and no graph). The FRD-027 facts count the conformer block
  (`stack_params.stack_params`: 2.64B).
* **Whisper's calculator count** leaves out its encoder's convolutional frame embedding:
  1,542.5M of 1,543.5M (origin/main the same).
* **DeepSeek Sparse Attention's indexer key cache** (V3.2: `index_head_dim` 128 a token a layer,
  stored at 8 bits beside the 576-element MLA latent) is in no reader's KV, on origin/main as
  here; its weights are in the GenZ graph (`attention.mla_indexer_ops`).
* **The serving path's weights of a model whose checkpoint holds modules no config key describes**
  (round 14). The weights are the count of what the config states, so a module the modelling code
  builds at a width written in the code, or a codec the config names no geometry for, is not in
  them. The gate holds each such figure to the whole checkpoint less exactly those modules, named
  with their source in its allowlist entry (`unstated`, from `truths.json`), and fails it below
  that: Voxtral-Mini-4B-Realtime, 4,424,886,272 of 4,429,679,360 parameters -- the 26 layers'
  `VoxtralRealtimeTextAdaRmsNorm` (5,111,808: Linear(3,072, 32) and back, the 32 in
  modeling_voxtral_realtime.py); higgs-audio-v3-tts-4b, 4,022,473,216 of its checkpoint's
  4,654,850,537 -- an acoustic codec and codebook tables (`tied.embedding.modality_embeddings`,
  222,413,801; its `audio_encoder_config` says `encoder_type: discrete` with no geometry), a second
  stored copy of the token table (388,956,160, beside `tie_word_embeddings: true`) and 8 codebook
  heads (21,012,480). Both sit above origin/main's reading on the HF loader's path and below it on
  the converter's, where origin/main fell back to a generic 7B (13.2 GB).
* **A neural codec or a speech encoder** (a TTS model's speech tokenizer, Mimi, w2v-BERT) states
  a stack's geometry but no token vocabulary: it is not a language model, which every reader
  sizes, and the corpus names it (`_not_language_models`) without measuring it. Read as one, the
  readers size its transformer blocks and not its convolutional decoder or quantizers (the
  Qwen3-TTS speech tokenizer: 62M of its checkpoint's 171M; origin/main sized its default 4096 x
  32 shape, 8.9B).

## The gate

`tests/test_kv_gate.py` runs the corpus harness over every config the repository ships
(prepopulated.db, every fixture, the synthetic shapes, GenZ's built-in models, and the GGUF
reader -- `bud_component.gguf_to_hf_config` with `gguf_layers` -- on `tests/fixtures/gguf`'s
metadata-only copies of four real GGUFs: Qwen3.5-2B's hybrid, gpt-oss-20b's window pattern,
SmolVLM's Llama, nomic-embed's encoder; `scripts/gguf_metadata_fixture.py` copies every metadata
key and tensor name verbatim, which is all the reader reads) through every reader, and compares
each figure with origin/main's (`tests/fixtures/kv_gate/origin-main.json`, measured by the same
harness on `c17ef89`). It fails on:

* a memory figure below origin/main's, or a capacity figure above it,
* a figure the branch reports and origin/main did not (main's reader raised, or had no such
  reader) -- held to its ground truth like a drop, and
* a figure origin/main reported and the branch does not (its reader now raises) -- its entry
  names the error the branch records for that reader, and the gate checks it is that error,

unless the allowlist names it: the config, the reader, both figures, its cause, and its ground
truth -- transformers 5.12's own model for the config on the meta device (`scripts/kv_truth.py`:
every parameter; the K / V projections each decoder layer's self-attention caches, and those of
its cross-attention; the recurrent state transformers' mixer modules hold; the cache kinds and
windows `DynamicCache(config=...)` builds), the checkpoint's or GGUF's tensors (`kv_truth.py
--checkpoint REPO` reads every safetensors shard's header from the hub, no weights downloaded), or
the config arithmetic the entry cites (`tests/fixtures/kv_gate/truths.json`, which also states a
fixed-length encoder's token count: a tower's patches, Whisper's 1,500 frames) -- and the
branch's figure is not below that truth. A capacity figure names the memory figures it is sized
from, each held to its own truth. Never a model card's rounded total: a hub card's "100B" sat 2.8%
below Ling-flash-2.0's 102,889,705,216 tensors (`tests/test_kv_truth.py` rejects one).

The truth itself says when it cannot read a model: a softmax attention whose K / V projections
its table does not name, or a layer the class builds as a convolution where the cache expects
attention (LFM2 reads only `full_attention`), leaves the width undetermined -- no truth, so the
entry fails -- never 0, which would accept any figure. The decoder's layer list is a list of
blocks, never of output heads (MusicGen's `lm_heads`). Each GenZ graph is held to the weights it
graphs: the prefill and chunked graphs to the language model's matrices (both hold the input
embedding), the decode graph to them less every table a step gathers one row from (the token
embedding, a learned position table, codebook embeddings). A per-token figure with no stated
context (GenZ's and the converter's `get_kv_size`, the serving model's and the prefix cache's
bytes a token, the facts') is held to the K + V a token adds on every layer that caches it --
full, and windowed while the token is inside the window -- and the KV planner's rate at a stated
length (`kvgeo_N[0]`) to what the N-th token adds (the full layers, and the windowed layers whose
window still holds it). Counting the full layers alone gave a stack whose every layer is windowed
a per-token truth of 0, which accepted any figure (round 13: Voxtral-Mini-4B-Realtime's 26
layers of 2,048, the LFM2.5-Audio detokenizers' 3 of 512). The serving path's weights
(`serving_weight_bytes`, `conv_serving_weight_bytes`) are held to every parameter of the model at
2 B, and its concurrency (`cluster_b_eff_4096`, `disagg_b_eff_4096`) by the weights, the KV a token
and the fixed bytes a request it is sized from.

The gate needs no transformers or torch: the figures read the shipped class defaults (above) and
the allowlist's truths are committed. Two readers have third-party dependencies of their own
(`kv_corpus.READER_DEPENDENCIES`): the GGUF reader needs `gguf`, and BudSimulator's converter
imports BudSimulator's database package, which needs `requests`, `lxml` and `json_repair`. Where
one of those is not installed, that reader cannot run at all: its figures are reported as not
measured here, by module and figure count, and `test_the_gguf_fixtures_are_read_through_the_gguf_reader`
skips without `gguf`. A reader that cannot import any other module fails the gate --
transformers and torch above all, so a reader that came to depend on one fails where it is absent
instead of passing as "not measured" (round 13 accepted any module outside the repository). The
gate passes, every figure measured, with transformers 5.12, with 4.50, and with neither
transformers nor torch installed (with `gguf`, `requests`, `lxml` and `json_repair`); a venv that
also lacks those passes with the GGUF reader's and the converter's figures reported as not
measured. `python scripts/kv_corpus.py --gate main.json branch.json --allowlist
tests/fixtures/kv_gate/allowlist.json` runs the same check over the machine's HF cache, model
folders and GGUFs too. A figure that moves again, or a new drop, fails until its entry is
re-measured (`scripts/kv_truth.py --allowlist-from`).

## The recurrent state

`calculate_state_memory` (and `EngineMemory.state_bytes`, which `bud_component` reports as
`state_bytes` and counts in `total_runtime_bytes`; and in the serving simulator
`ModelConfig.get_state_bytes`, below) is the per-sequence state, constant in
context length: the recurrent matrix in fp32 (unless `mamba_ssm_cache_dtype` / `mamba_ssm_dtype`
/ `inference_state_dtype` says otherwise) plus the convolution's `d_conv - 1` ring buffer at
model precision. One function, `state_memory.recurrent_state_shape`, gives the shape for every
mechanism; the GenZ graph's recurrent operators read the same shape and the same bytes per
element (the state dtype travels in the operator's name: `Rec State` fp32, `Rec State16` 16-bit,
for the recurrent matrix and for the small vectors some mechanisms keep beside it -- RWKV's token
shifts, xLSTM's normaliser; `Rec Conv` at the activation precision), so the graph's state equals
`calculate_state_memory` for every mechanism.

The serving simulator prices the same shapes with the same bytes
(`state_memory.shape_state_bytes`): `ModelConfig.get_state_bytes(precision_bytes)` sums the
recurrent layers' state (each per-layer config's shape; a pure state-space stack's shape from the
config's table entry, or for a built-in GenZ Mamba model from its `mamba_*` fields as Mamba-1).
`MemoryModel` charges a request that state, split by tensor parallelism, once, with its first KV
blocks and on the same tier, and frees it with the request (or when eviction takes the whole
request); `request_bytes(tokens)` is what the scheduler's never-fits check compares with every
tier. The cluster and disaggregation concurrency (`B_eff`) divide the free memory by KV x context
**plus** the state, and the prefill-to-decode transfer moves the state with the KV. Without it, a
hybrid's requests were admitted 2.5x (Qwen3.5-4B) to 10x (granite-4.0-h-small) too often and a
pure SSM's cost nothing.

## GGUF

`gguf_layers.hf_layer_keys_from_gguf` translates llama.cpp's keys into the HF ones above, by
llama.cpp's own rules, so a GGUF and its checkpoint resolve to the same plan
(`tests/test_layer_kinds_by_category.py::test_a_gguf_and_its_checkpoint_resolve_to_the_same_kv_and_state`):
`full_attention_interval`; per-layer `attention.head_count_kv` arrays (0 = no KV; with
`feed_forward_length` telling FFN-only slots apart: a stack has them only when its KV layers run
no FFN and some 0-KV layer runs one, as Nemotron-H's; any other 0-KV layer is the file's
recurrent mixer, or `no_op` when it states none -- naming Deci's no-op and linear-replacement
layers `mlp` slots had taken the FFN off every attention layer in the GenZ graph; an FFN no-op
on a layer with KV heads, in a stack without FFN slots, is still given its FFN -- an over-count); `attention.sliding_window` with
`attention.sliding_window_pattern` (a pattern array shorter than `block_count`, which llama.cpp
never writes, windows only the layers it reaches: the rest attend in full, beside a per-layer KV
array as without one); `attention.shared_kv_layers`; `ssm.{conv_kernel, inner_size,
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
layers. A pure Mamba stack keeps GenZ's own Mamba operators. GenZ's own stack builder turns every
layer into Mamba once it is given an SSM state, so it is given one only when the table names
recurrent layers and no attention layer: a stack with attention layers is graphed as attention,
with its KV (an upper bound, with a warning, should its per-layer configs ever not be built).

All three graph builders read the same per-layer configs, and a windowed layer's window from the
same table: any window spelling, and `attention_chunk_size` for Llama 4's chunked layers. In the
chunked graph a windowed layer reads no context beyond its window (`attention.window_kv_sizes`), an MLA layer reads its one
latent head (`attention.mla_attention_chunked`), and a recurrent layer holds one state per
sequence in the chunk however much each has cached (`recurrent.recurrent_chunked`). A dense layer
of an MoE model (Jamba's even layers, DeepSeek's and Kimi's leading layers, Llama 4's interleaved
ones) is graphed as one FFN, not as the experts, at the dense width the config names apart from
the experts' (`layer_plan.dense_ffn_width`: Llama 4's `intermediate_size_mlp`, MiniMax-M3's
`dense_intermediate_size`, Cohere2-MoE's `prefix_dense_intermediate_size`), else at
`intermediate_size`. Llama-4-Maverick's 24 dense layers had been graphed at the experts' 8,192
instead of 16,384: 3.02B parameters short.

Every MoE layer graphs its shared expert(s) beside the routed ones (`ffn.shared_expert_ops`, the
width from `layer_plan.shared_expert_width`: DeepSeek's and Kimi's `n_shared_experts` x the expert
width, Qwen MoE's `shared_expert_intermediate_size`, Granite 4's `shared_intermediate_size`,
Ling's `moe_shared_expert_intermediate_size`, Llama 4's and Hunyuan's class-built one), and the
routed experts are read under every spelling (`layer_plan.routed_expert_count` /
`routed_expert_width`: SmallThinker's `moe_num_primary_experts` of `moe_ffn_hidden_size`). The
per-layer MoE placement had removed the extra MoE layers that hid the missing shared expert:
DeepSeek-V3's graph fell to 2.55B parameters below the checkpoint (1,275,007 MiB against
transformers' 1,281,505). MLA's cache is its latent (`attention.mla_latent_op`: kv_lora_rank +
qk_rope_head_dim a token a layer; the per-head K / V its expanded score operators read are compute,
not cache), and V3.2's lightning indexer is graphed. An attention that gates its output from a
second query-width projection (`layer_plan.attention_output_gate`: Qwen3-Next, Qwen3.5) graphs
it. A gated FFN is read from the config's activation, else its config class's default
(`layer_plan.ffn_activation`: Llama 4's "silu"), else its family (`layer_plan.GATED_FFN_MODEL_TYPES`,
the parameter counter's table: SmallThinker, ChatGLM, Pix2Struct's `wi_0` / `wi_1`). An audio
decoder with several codebooks graphs as many embedding tables and heads
(`layer_plan.decoder_streams`: MusicGen's `num_codebooks`, Dia's `num_channels`), and a decoder with a
learned absolute position table graphs it (`layer_plan.learned_positions`: MBart, Whisper, GPT-2,
BERT-style decoders). An encoder-decoder's encoder is graphed at its own geometry: a hierarchical
tower stage by stage (`layer_plan.encoder_stages`: Swin's `depths` at `embed_dim` x 2^i, with the
2 x 2 merges between stages and its windows' position tables), a tower's patch or frame embedding
and positions (`stack_params.tower_embedding_params`), a nested encoder's own token table, and a
composite's projection to the decoder's width (`enc_to_dec_proj`, MusicGen's 768 -> 1,024).

Known approximations: the graph's KV memory is the model's (windowed layers bounded by their
window in every stack), not that of an engine without per-layer KV, which would hold the full
sequence on windowed layers of a mixed stack -- see *What is conservative*; KV-shared layers are
still graphed as attention (their KV read traffic is real, their memory is over-counted); a
cross-attention layer of its own (Mllama) is graphed as cross-attention over the tower's tokens,
but the vision tower that produces them is not graphed (as for every multimodal decoder: GenZ
graphs the language model, and an adapter such as BLIP-2's Q-Former is the tower's); a
*uniformly* windowed stack (Mistral v0.1) keeps GenZ's uniform full-attention graph -- the memory
calculator applies its window, the graph does not; a per-layer MoE list (Hunyuan's) is graphed
at its largest entry.

## Adding a new family

If its config uses an existing key, nothing to do. If it introduces a new per-layer kind name,
add the name to `LAYER_KINDS` (until then it is charged full KV with a warning). If it introduces
a new layout key, add a dialect to `resolve_layer_plan`; if a new mechanism, add its state shape to
`state_memory.recurrent_state_shape` and its parameters to `mixer_params`. Add the shipped
`config.json` to `tests/fixtures/hybrid_configs/` and a case to
`tests/test_layer_kinds_by_category.py`.

## Evidence

`docs/proofs/kv-layer-kinds-corpus-diff.md` holds the before/after figures for every model config
the simulator ships or this machine caches (`scripts/kv_corpus.py`), and `tests/test_kv_gate.py`
holds every figure that moves in the unsafe direction to its ground truth (*The gate* above).
