#!/usr/bin/env python3
"""KV / memory figures for every model config the simulator ships or this machine caches.

Run it against two source trees (before and after a change) and diff the two JSON files
with ``--diff``: every figure that moved must be explained by a layer-kind correction, and
a dense model must not move at all. The import root is whatever ``PYTHONPATH`` puts first,
so the same script measures both trees.

    PYTHONPATH=<tree>/llm-memory-calculator/src python kv_corpus.py --out after.json \
        --db BudSimulator/data/prepopulated.db --fixtures tests/fixtures \
        --hf-cache ~/.cache/huggingface/hub --models /home/bud/models --genz-data <scratch>

    python kv_corpus.py --diff before.json after.json --report diff.md

    # THE GATE: every unsafe-direction move (a memory figure down, a capacity figure up), and every
    # figure one tree produced and the other did not, must be in the allowlist with both figures
    # and a ground truth it is not below (a lost figure: the error that replaced it)
    python kv_corpus.py --gate main.json branch.json \
        --allowlist ../tests/fixtures/kv_gate/allowlist.json
    python kv_corpus.py --unsafe main.json branch.json        # list the moves as JSON

Corpus: every row of the BudSimulator database that carries a config (rows whose
`config_json` is empty are counted and reported, not measured -- they would only measure the
calculator's defaults), every hybrid fixture, every config.json in the HuggingFace cache and
the model folders that names a decoder geometry and states a token vocabulary (one that states
none -- a codec, a speech encoder -- is no language model: named in `_not_language_models`, not
measured), every GGUF in the model folders and the fixtures' metadata-only GGUFs (read
through bud_component's GGUF reader, beside the file's own tensor count), and every built-in
GenZ model. Model folders and caches are only read.

Figures, per config: the calculator's KV / state / weights / totals; the engine estimate
(`estimate_memory_for_engine`, llama.cpp and generic, at 4k and 32k: KV, state, total,
parameters); the FRD-027 `ModelFacts` layer counts, weight bytes (what the vLLM planner
reserves), KV bytes a token and at 4k, and state bytes; GenZ's per-layer attention and FFN kinds,
its KV elements per token (`get_kv_size`, which the disaggregation KV transfer multiplies by
2 bytes), its decode and chunked graphs' KV and weights; and the serving `MemoryModel`'s KV
bytes per token, per-request state, the 1,024-token requests 8 GiB admits, and the
disaggregation transfer for 512 + 128 tokens, read from `DisaggregationAnalyzer` itself; GenZ's
runtime modelling (`prefill_moddeling`, `decode_moddeling` -- whose fit check adds the decode
graph's weights to its KV -- and `chunked_moddeling`, which `estimate_*_performance` wrap), each
run to completion on a fixed 200 GB device, its latency, weights and KV, or its error, and the
decode fit check on a device the resident model's weights alone overflow; with ``--repo-root``, BudSimulator's database
converter (`ModelConverter.hf_to_genz`): its encoder / decoder depths, KV elements per token,
per-sequence fixed bytes and its graph's weights; and every other reader of the layer kinds: GenZ's prefill graph
(weights, KV), the prefix-cache analyser's bytes per token, the KV planner's `KVGeometry`, the
vLLM planner (`engine/vllm.py`: limits, and the running sequences `predict` gives 1,024 clients
at 1k and 8k prompts), its profile scheduler (`perf.predict.sim_config` + `perf.sched.simulate`)
and its shared-slice sizing (`runtime.modules.capacity.shared_slice_mib`).

Fixture roots are read recursively, keeping the files that name a decoder (or an encoder-decoder's)
geometry; the planner's engine records beside them are not model configs. Built-in synthetic
shapes (``synthetic_configs``; ``--no-synthetic`` leaves them out) cover what no shipped or cached
config does: depth stated only in a per-layer list, nested decoders (VisionEncoderDecoder,
EncoderDecoderModel, MusicGen, BLIP-2 / InstructBLIP with a flan-T5 `text_config`, Florence-2),
ProphetNet, RWKV, xLSTM, Llama 4's interleaved MoE, GGUF-derived short window patterns, a
decoder-only stack's own cross-attention layers (Mllama's `cross_attention_layers`, with and
without the tower's geometry), an image encoder whose length is its `seq_len` (Pix2Struct), and
the encoder-decoder and dense spellings around them; the fixture folder `cross_attention/` holds
transformers 5.12's own Mllama, Pix2Struct and BLIP-2 (flan-T5) configs.
"""

import argparse
import glob
import json
import os
import sqlite3
import sys
import traceback
import warnings

SEQS = (4096, 32768)


_DEPTH = ("num_hidden_layers", "n_layers", "num_layers", "decoder_layers", "num_decoder_layers",
          "decoder_num_hidden_layers")
_WIDTH = ("hidden_size", "d_model", "n_embd")


def _is_llm_config(cfg):
    if not isinstance(cfg, dict):
        return False
    tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    n = any(tc.get(k) for k in _DEPTH[:3])
    h = any(tc.get(k) for k in _WIDTH)
    return bool(n and h)


def _is_model_config(cfg):
    """A decoder config, or an encoder-decoder's (its decoder's depth under its own spelling or
    in a nested block: BART's defaults state no `num_hidden_layers`, Dia nests both stacks)."""
    if _is_llm_config(cfg):
        return True
    if not isinstance(cfg, dict):
        return False
    for scope in (cfg, cfg.get("text_config")):
        if not isinstance(scope, dict):
            continue
        if any(scope.get(k) for k in _DEPTH) and any(scope.get(k) for k in _WIDTH):
            return True
        block = scope.get("decoder_config") or scope.get("decoder")
        if isinstance(block, dict) and _is_llm_config(block):
            return True
    return False


EMPTY_DB_ROWS = []
#: Configs that state a stack's geometry but no token vocabulary anywhere (`vocab_size` or its
#: spellings, on the config or a nested block): neural codecs and speech encoders (a speech
#: tokenizer's codec decoder, w2v-BERT, Mimi). They are not language models, which every reader
#: here sizes, so they are named in the output (``_not_language_models``) and not measured.
NOT_LANGUAGE_MODELS = []
_VOCAB_KEYS = ("vocab_size", "padded_vocab_size", "n_vocab")


def _states_vocabulary(cfg, depth=0):
    if not isinstance(cfg, dict) or depth > 3:
        return False
    if any(isinstance(cfg.get(k), int) and cfg.get(k) > 0 for k in _VOCAB_KEYS):
        return True
    return any(_states_vocabulary(v, depth + 1) for v in cfg.values() if isinstance(v, dict))


def collect(args):
    corpus = []
    if args.db and os.path.exists(args.db):
        con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
        for model_id, cfg_json in con.execute("select model_id, config_json from models"):
            try:
                cfg = json.loads(cfg_json)
            except Exception:
                cfg = None
            if not _is_model_config(cfg):
                EMPTY_DB_ROWS.append(model_id)
                continue
            corpus.append(("db:" + model_id, cfg))
        con.close()
    seen_fixtures = set()
    for root in args.fixtures or []:
        for path in sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True)):
            real = os.path.realpath(path)
            if real in seen_fixtures or os.sep + "kv_gate" + os.sep in path:
                continue  # the gate's own baseline, allowlist and truths are no model configs
            seen_fixtures.add(real)
            try:
                cfg = json.load(open(path))
            except Exception:
                continue
            if not _is_model_config(cfg):
                continue
            # a file's own name, or its path under the root where two share a name
            # (measured_checkpoints/<model>/config.json)
            name = "fixture:" + os.path.basename(path)
            if any(n == name for n, _ in corpus):
                name = "fixture:" + os.path.relpath(path, root)
            corpus.append((name, cfg))
    seen = set()
    for root in [args.hf_cache, args.models]:
        if not root or not os.path.isdir(root):
            continue
        for path in sorted(glob.glob(os.path.join(root, "**", "config.json"), recursive=True)):
            real = os.path.realpath(path)
            if real in seen:
                continue
            seen.add(real)
            try:
                cfg = json.load(open(path))
            except Exception:
                continue
            if _is_model_config(cfg):
                rel = os.path.relpath(path, root)
                if not _states_vocabulary(cfg):
                    NOT_LANGUAGE_MODELS.append("config:" + rel)
                    continue
                corpus.append(("config:" + rel, cfg))
    return corpus


def collect_ggufs(args):
    """Every GGUF the corpus reads: the fixture roots' (metadata-only copies of real files,
    ``scripts/gguf_metadata_fixture.py``: tests/fixtures/gguf) and the model folders' (a file
    under 10 MiB there is a stub or a vocabulary, not a model)."""
    out = []
    seen = set()
    roots = [(r, False) for r in (getattr(args, "fixtures", None) or [])]
    roots.append((getattr(args, "models", None), True))
    for root, sized in roots:
        if not root or not os.path.isdir(root):
            continue
        for path in sorted(glob.glob(os.path.join(root, "**", "*.gguf"), recursive=True)):
            real = os.path.realpath(path)
            base = os.path.basename(path).lower()
            if not os.path.isfile(real):  # dangling link into another machine's layout
                continue
            if real in seen or "mmproj" in base or (
                    sized and os.path.getsize(real) < 10 * 1024 * 1024):
                continue
            seen.add(real)
            out.append(path)
    return out


def _gb(x):
    return None if x is None else round(float(x), 9)


def measure_config(name, cfg, genz_data):
    from llm_memory_calculator import calculate_memory
    from llm_memory_calculator.calculator import ModelMemoryCalculator

    row = {"name": name}
    try:
        calc = ModelMemoryCalculator()
        calc.model_type = calc.detect_model_type(cfg)
        row["model_type"] = calc.model_type
        row["attention_type"] = calc.detect_attention_type(cfg)
        for seq in SEQS:
            row[f"kv_gb@{seq}"] = _gb(calc.calculate_kv_cache(cfg, 1, seq, "bf16"))
        bd = calc.kv_cache_breakdown(cfg, 1, 4096, "bf16")
        row["kv_full_layers"] = bd.get("num_full_layers")
        row["kv_sliding_layers"] = bd.get("num_sliding_layers")
        row["kv_free_layers"] = bd.get("num_no_kv_layers")
        row["kv_source"] = bd.get("source")
        row["kv_cross_layers"] = bd.get("num_cross_layers") or 0
        row["state_gb"] = _gb(calc.calculate_state_memory(cfg, 1, "bf16"))
    except Exception as e:  # recorded, not fatal: a crash is itself a figure to diff
        row["calc_error"] = f"{type(e).__name__}: {e}"
    try:
        rep = calculate_memory(cfg, batch_size=1, seq_length=4096, precision="bf16")
        row["report_kv_gb"] = _gb(rep.kv_cache_bytes / 1e9)
        row["report_state_gb"] = _gb(rep.state_memory_bytes / 1e9)
        row["report_weight_gb"] = _gb(rep.weight_memory_bytes / 1e9)
        row["report_params"] = int(rep.parameter_count)
        row["report_total_gb"] = _gb(rep.total_memory_bytes / 1e9)
    except Exception as e:
        row["report_error"] = f"{type(e).__name__}: {e}"
    row.update(measure_genz(cfg, genz_data))
    row.update(measure_engine_and_facts(cfg))
    row.update(measure_converter(cfg))
    row.update(measure_readers(cfg))
    return row


#: The tree whose BudSimulator converter is measured (``--repo-root``).
REPO_ROOT = None
#: Where the GenZ graphs are written (``--genz-data``).
GENZ_DATA = None


def measure_converter(cfg):
    """BudSimulator's database path: `ModelConverter.hf_to_genz`, which graphs a model stored in
    prepopulated.db (or added through hf_integration / model_loader) for the API."""
    out = {}
    if not REPO_ROOT:
        return out
    bud = os.path.join(REPO_ROOT, "BudSimulator")
    sys.path.insert(0, bud)
    try:
        import contextlib
        import io

        from src.db.model_converter import ModelConverter

        with contextlib.redirect_stdout(io.StringIO()):
            mc = ModelConverter().hf_to_genz(json.loads(json.dumps(cfg)))
        out["conv_layers"] = [mc.num_encoder_layers, mc.num_decoder_layers]
        out["conv_kv_size"] = int(mc.get_kv_size())
        # A tree whose ModelConfig prices no state (origin/main) charges none: 0, as the serving
        # path's state is recorded. A request's fixed bytes: the state, and (round 9) an
        # encoder-decoder's cross-attention cache.
        out["conv_state_bytes"] = _fixed_bytes(mc)
        # the weights of the model the converter graphs (its prefill graph)
        try:
            from llm_memory_calculator.genz.Models.get_language_model import (
                create_full_prefill_model,
            )
            from llm_memory_calculator.genz.analyse_model import get_model_df, get_summary_table

            with contextlib.redirect_stdout(io.StringIO()):
                path = create_full_prefill_model(name=mc, input_sequence_length=512,
                                                 data_path=GENZ_DATA)
                df = get_model_df(path, batch_size=1, data_path=GENZ_DATA,
                                  model_characterstics=True)
                st = get_summary_table(df, model_characterstics=True)
            out["conv_prefill_weights_mb"] = round(float(st["Total Weights (MB)"].values[0]), 6)
        except Exception as e:
            out["conv_weights_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    except Exception as e:
        out["conv_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    finally:
        sys.path.remove(bud)
    return out


def _fixed_bytes(mc):
    """A GenZ ModelConfig's fixed bytes per request at 2 B, by whatever the tree measured prices."""
    for name in ("get_request_fixed_bytes", "get_state_bytes"):
        f = getattr(mc, name, None)
        if callable(f):
            return int(f(2))
    return 0


#: The engine record and device the vLLM-planner figures are measured with (this script's own
#: fixture, so both trees are measured against the same record).
PLANNER_RECORD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "tests", "fixtures", "planner", "records", "vllm-0.30-upstream.json")
PLANNER_DEVICE = "h100xm-80c-vgpu"


def measure_readers(cfg):
    """Every other reader of the layer kinds: the prefix-cache analyser, the KV planner's geometry,
    the vLLM planner (limits, predict), its profile scheduler and its shared-slice sizing."""
    import contextlib
    import copy
    import io

    out = {}
    try:
        from llm_memory_calculator.genz.Models.get_language_model import get_configs
        from llm_memory_calculator.genz.serving.prefix_cache import PrefixCacheAnalyzer

        with contextlib.redirect_stdout(io.StringIO()):
            pa = PrefixCacheAnalyzer(get_configs(copy.deepcopy(cfg)))
        out["prefix_kv_bytes_per_token"] = int(pa._kv_bytes_per_token)
    except Exception as e:
        out["prefix_error"] = f"{type(e).__name__}: {str(e)[:160]}"
    try:
        from llm_memory_calculator.kv.geometry import KVGeometry

        for seq in SEQS:
            g = KVGeometry.from_model(copy.deepcopy(cfg), seq_length=seq)
            out[f"kvgeo_{seq}"] = [round(g.bytes_per_token_total, 3), g.kv_layers, g.recurrent_layers,
                                   round(g.state_bytes_per_layer, 1), g.block_size,
                                   round(float(getattr(g, "cross_bytes_per_sequence", 0.0)), 1)]
    except Exception as e:
        out["kvgeo_error"] = f"{type(e).__name__}: {str(e)[:160]}"
    try:
        from llm_memory_calculator.device import catalogue
        from llm_memory_calculator.engine import EngineRecord, engine_for
        from llm_memory_calculator.engine.base import Workload
        from llm_memory_calculator.model.facts import ModelFacts

        rec = json.load(open(PLANNER_RECORD))
        eng, dev = engine_for(EngineRecord.from_dict(rec)), catalogue.get(PLANNER_DEVICE)
        facts = ModelFacts.from_config(copy.deepcopy(cfg))
        lim = eng.limits({}, facts, dev)
        out["vllm_limits"] = [int(lim.num_blocks), int(lim.kv_tokens), int(lim.sequence_ceiling or 0),
                              int(lim.attn_groups), int(getattr(lim, "sliding_groups", 0)),
                              round(float(lim.state_bytes_per_sequence or 0), 1), bool(lim.boots)]
        for inp in (1024, 8192):
            p = eng.predict({}, Workload(concurrency=1024, input_tokens=inp, output_tokens=256), facts, dev)
            out[f"vllm_running_{inp}"] = int(p.running)
        try:
            from llm_memory_calculator.perf.predict import sim_config
            from llm_memory_calculator.perf.sched import simulate

            class _Profile:
                api_ms = 0.0

            class _Steps:
                def decode_ms(self, b, ctx):
                    return 10.0

                def chunk_ms(self, tokens, request_len):
                    return 10.0

            sc = sim_config(eng, {}, facts, dev, _Profile())
            res = simulate(_Steps(), sc, [(1024, 256)] * 400, concurrency=400)
            out["sim_max_running"] = int(res.max_running)
        except Exception as e:
            out["sim_error"] = f"{type(e).__name__}: {str(e)[:160]}"
        try:
            from llm_memory_calculator.runtime.modules.capacity import shared_slice_mib

            mib, fits = shared_slice_mib(eng, facts, dev, {}, concurrency=16, request_tokens=4352)
            out["slice_mib_16x4352"] = [int(mib), bool(fits)]
        except Exception as e:
            out["slice_error"] = f"{type(e).__name__}: {str(e)[:160]}"
    except Exception as e:
        out["vllm_error"] = f"{type(e).__name__}: {str(e)[:160]}"
    return out


def synthetic_configs(fixtures_root):
    """Shapes no shipped or cached config has, each with the spelling a real config class or hub
    file uses (transformers 5.12 unless noted)."""
    import copy

    def fx(name, sub="hybrid_configs"):
        return json.load(open(os.path.join(fixtures_root, sub, name + ".json")))

    s = {}
    s["gpt2"] = dict(model_type="gpt2", architectures=["GPT2LMHeadModel"], n_layer=12, n_head=12,
                     n_embd=768, n_positions=1024, vocab_size=50257)
    s["mpt-7b"] = dict(model_type="mpt", d_model=4096, n_heads=32, n_layers=32, expansion_ratio=4,
                       max_seq_len=2048, vocab_size=50432)
    s["chatglm3"] = dict(model_type="chatglm", num_layers=28, hidden_size=4096,
                         num_attention_heads=32, multi_query_attention=True,
                         multi_query_group_num=2, kv_channels=128, ffn_hidden_size=13696,
                         padded_vocab_size=65024, seq_length=8192)
    s["falcon-40b"] = dict(model_type="falcon", hidden_size=8192, num_attention_heads=128,
                           num_hidden_layers=60, num_kv_heads=8, new_decoder_architecture=True,
                           vocab_size=65024)
    s["llama3-8b"] = dict(model_type="llama", hidden_size=4096, num_attention_heads=32,
                          num_key_value_heads=8, num_hidden_layers=32, intermediate_size=14336,
                          vocab_size=128256, max_position_embeddings=8192)
    s["mistral-7b-window"] = dict(s["llama3-8b"], model_type="mistral", sliding_window=4096,
                                  vocab_size=32000)
    s["gemma2-9b-hub"] = dict(model_type="gemma2", architectures=["Gemma2ForCausalLM"],
                              hidden_size=3584, num_hidden_layers=42, num_attention_heads=16,
                              num_key_value_heads=8, head_dim=256, intermediate_size=14336,
                              sliding_window=4096, vocab_size=256000, max_position_embeddings=8192)
    s["unknown-layer-name"] = dict(model_type="zzunknown", hidden_size=2048, num_attention_heads=16,
                                   num_key_value_heads=4, num_hidden_layers=4,
                                   intermediate_size=8192, vocab_size=32000,
                                   layer_types=["full_attention", "weird_mixer"] * 2)
    # Llama 4: chunked attention, MoE on every layer (Scout) or every other layer (Maverick)
    s["llama4-scout-text"] = dict(
        model_type="llama4", architectures=["Llama4ForConditionalGeneration"],
        text_config=dict(model_type="llama4_text", hidden_size=5120, num_attention_heads=40,
                         num_key_value_heads=8, head_dim=128, num_hidden_layers=48,
                         intermediate_size=8192, intermediate_size_mlp=16384,
                         num_local_experts=16, num_experts_per_tok=1,
                         interleave_moe_layer_step=1, attention_chunk_size=8192,
                         no_rope_layers=[], vocab_size=202048))
    s["llama4-maverick-text"] = copy.deepcopy(s["llama4-scout-text"])
    s["llama4-maverick-text"]["text_config"].update(num_local_experts=128,
                                                    interleave_moe_layer_step=2)
    # recurrent stacks without a layout key
    s["rwkv"] = dict(model_type="rwkv", hidden_size=2048, num_hidden_layers=24,
                     attention_hidden_size=2048, intermediate_size=8192, vocab_size=50277,
                     context_length=4096)
    s["xlstm"] = dict(model_type="xlstm", hidden_size=4096, embedding_dim=4096, num_blocks=32,
                      num_hidden_layers=32, num_heads=8, qk_dim_factor=0.5, v_dim_factor=1.0,
                      vocab_size=50304)
    # depth only in the per-layer list (transformers 5.12's NemotronH save), flat and nested
    nh = fx("nvidia_Nemotron-H-8B-Base-8K")
    pattern = nh.pop("hybrid_override_pattern")
    nh.pop("num_hidden_layers")
    kinds = {"M": "mamba", "-": "mlp", "*": "attention", "E": "moe"}
    nh["layers_block_type"] = [kinds[c] for c in pattern]
    s["nemotronh-list-depth"] = nh
    s["nemotronh-list-depth-nested"] = {"model_type": "zzwrap", "text_config": copy.deepcopy(nh)}
    q = fx("Qwen_Qwen3.5-4B")
    q["text_config"].pop("num_hidden_layers", None)
    s["qwen35-list-depth"] = q
    # encoder-decoders, flat
    s["t5-small"] = dict(model_type="t5", is_encoder_decoder=True, d_model=512, d_kv=64, d_ff=2048,
                         num_heads=8, num_layers=6, vocab_size=32128)
    s["t5-asym"] = dict(model_type="t5", is_encoder_decoder=True, d_model=1024, d_kv=64,
                        d_ff=2816, num_heads=16, num_layers=24, num_decoder_layers=12,
                        vocab_size=32128)
    s["mt5-noflag"] = dict(model_type="mt5", d_model=512, d_kv=64, d_ff=1024, num_heads=6,
                           num_layers=8, num_decoder_layers=8, vocab_size=250112)
    s["whisper-large-v3"] = dict(model_type="whisper", is_encoder_decoder=True, d_model=1280,
                                 encoder_layers=32, decoder_layers=32, encoder_attention_heads=20,
                                 decoder_attention_heads=20, encoder_ffn_dim=5120,
                                 decoder_ffn_dim=5120, num_hidden_layers=32,
                                 max_source_positions=1500, max_target_positions=448,
                                 vocab_size=51866, num_mel_bins=128)
    s["marian-noflag"] = dict(model_type="marian", d_model=512, encoder_layers=6, decoder_layers=6,
                              encoder_attention_heads=8, decoder_attention_heads=8,
                              encoder_ffn_dim=2048, decoder_ffn_dim=2048, vocab_size=58101)
    # ProphetNet's spellings (ProphetNetConfig): num_{encoder,decoder}_attention_heads
    s["prophetnet"] = dict(model_type="prophetnet", is_encoder_decoder=True, hidden_size=1024,
                           num_encoder_layers=12, num_decoder_layers=12,
                           num_encoder_attention_heads=16, num_decoder_attention_heads=16,
                           encoder_ffn_dim=4096, decoder_ffn_dim=4096, vocab_size=30522,
                           max_position_embeddings=512)
    s["nllb-moe"] = dict(model_type="nllb-moe", is_encoder_decoder=True, d_model=2048,
                         encoder_layers=24, decoder_layers=24, encoder_attention_heads=16,
                         decoder_attention_heads=16, encoder_ffn_dim=8192, decoder_ffn_dim=8192,
                         num_experts=128, vocab_size=256206)
    # encoder-decoders, nested
    s["vision-enc-dec"] = dict(
        model_type="vision-encoder-decoder", is_encoder_decoder=True,
        encoder=dict(model_type="vit", hidden_size=768, num_hidden_layers=12,
                     num_attention_heads=12, intermediate_size=3072, image_size=384,
                     patch_size=16),
        decoder=dict(model_type="trocr", d_model=1024, decoder_layers=12,
                     decoder_attention_heads=16, decoder_ffn_dim=4096, vocab_size=50265,
                     max_position_embeddings=512))
    # GPT-2 spellings (`n_layer` / `n_embd` / `n_head`) for the nested decoder
    s["vision-enc-dec-gpt2"] = dict(
        model_type="vision-encoder-decoder", is_encoder_decoder=True,
        encoder=dict(model_type="vit", hidden_size=768, num_hidden_layers=12,
                     num_attention_heads=12, intermediate_size=3072, image_size=224,
                     patch_size=16),
        decoder=dict(model_type="gpt2", n_embd=768, n_layer=6, n_head=12, vocab_size=50257,
                     add_cross_attention=True))
    s["enc-dec-llama-dec"] = dict(
        model_type="encoder-decoder", is_encoder_decoder=True,
        encoder=dict(model_type="bert", hidden_size=1024, num_hidden_layers=24,
                     num_attention_heads=16, intermediate_size=4096),
        decoder=dict(model_type="llama", hidden_size=2048, num_hidden_layers=16,
                     num_attention_heads=32, num_key_value_heads=8, intermediate_size=8192,
                     vocab_size=32000))
    # MusicGen: a T5 text encoder (`text_encoder`) feeding a nested decoder
    s["musicgen-small"] = dict(
        model_type="musicgen", is_encoder_decoder=True,
        text_encoder=dict(model_type="t5", d_model=768, d_kv=64, d_ff=3072, num_heads=12,
                          num_layers=12, vocab_size=32128),
        audio_encoder=dict(model_type="encodec", hidden_size=128),
        decoder=dict(model_type="musicgen_decoder", hidden_size=1024, num_hidden_layers=24,
                     num_attention_heads=16, ffn_dim=4096, vocab_size=2048,
                     max_position_embeddings=2048, num_codebooks=4))
    # a whole encoder-decoder under `text_config` (Blip2Config / InstructBlipConfig with a
    # T5Config(d_model=2048, d_kv=64, num_heads=32, num_layers=24) text model: flan-t5-xl)
    t5xl = dict(model_type="t5", is_encoder_decoder=True, d_model=2048, d_kv=64, d_ff=5120,
                num_heads=32, num_layers=24, num_decoder_layers=24, vocab_size=32128)
    for name in ("blip2-flan-t5-xl", "instructblip-flan-t5-xl"):
        s[name] = dict(model_type=name.split("-")[0], is_encoder_decoder=True,
                       use_decoder_only_language_model=False, num_query_tokens=32,
                       text_config=copy.deepcopy(t5xl),
                       qformer_config=dict(model_type=name.split("-")[0] + "_qformer",
                                           hidden_size=768, num_hidden_layers=12,
                                           num_attention_heads=12, intermediate_size=3072),
                       vision_config=dict(hidden_size=1408, num_hidden_layers=39,
                                          num_attention_heads=16, image_size=224, patch_size=14))
    s["florence2"] = dict(model_type="florence2", is_encoder_decoder=True,
                          text_config=dict(model_type="florence2_language", d_model=1024,
                                           encoder_layers=12, decoder_layers=12,
                                           encoder_attention_heads=16, decoder_attention_heads=16,
                                           encoder_ffn_dim=4096, decoder_ffn_dim=4096,
                                           is_encoder_decoder=True, vocab_size=51289,
                                           max_position_embeddings=1024),
                          vision_config=dict(model_type="davit"))
    # an audio-LLM: no cross-attention, the audio enters the decoder's input
    s["qwen2-audio"] = dict(model_type="qwen2_audio",
                            audio_config=dict(model_type="qwen2_audio_encoder", d_model=1280,
                                              encoder_layers=32, encoder_attention_heads=20,
                                              encoder_ffn_dim=5120),
                            text_config=dict(model_type="qwen2", hidden_size=4096,
                                             num_attention_heads=32, num_key_value_heads=32,
                                             num_hidden_layers=32, intermediate_size=11008,
                                             vocab_size=156032))
    s["decoder-only-isencdec-false"] = dict(s["llama3-8b"], is_encoder_decoder=False)
    # GGUF-derived: llama.cpp's keys through gguf_layers (a window pattern shorter than the stack,
    # which llama.cpp never writes; and a per-layer KV-head array with 0-KV recurrent layers)
    def gguf(name, n, meta, base):
        try:
            from llm_memory_calculator.gguf_layers import hf_layer_keys_from_gguf
        except Exception:  # a tree without the GGUF reader: the row records the shape only
            return dict(base, _gguf_reader="absent")
        return dict(base, **hf_layer_keys_from_gguf(lambda k: meta.get(k), n))

    g3 = dict(model_type="gemma3_text", hidden_size=2560, num_attention_heads=8,
              num_key_value_heads=4, head_dim=256, num_hidden_layers=34, intermediate_size=10240,
              vocab_size=262144)
    s["gguf-swa-short-pattern"] = gguf("g3", 34, {"attention.sliding_window": 1024,
                                                  "attention.sliding_window_pattern":
                                                  [True] * 5 + [False]}, g3)
    s["gguf-swa-full-pattern"] = gguf("g3", 34, {"attention.sliding_window": 1024,
                                                 "attention.sliding_window_pattern":
                                                 ([True] * 5 + [False]) * 5 + [True] * 4}, g3)
    s["deepseek-v2-lite-mla"] = fx("deepseek_v2_lite_chat_config", sub=".")
    # A decoder-only stack's own cross-attention layers (MllamaConfig's `cross_attention_layers`,
    # reduced): the tower states its patches (448 / 14, one tile: 1,025 tokens), and a variant
    # whose tower states no geometry (the cache then over the sequence)
    mll = dict(model_type="mllama",
               text_config=dict(model_type="mllama_text_model", hidden_size=2048,
                                num_hidden_layers=10, num_attention_heads=16,
                                num_key_value_heads=4, intermediate_size=8192, vocab_size=32000,
                                cross_attention_layers=[3, 8]),
               vision_config=dict(model_type="mllama_vision_model", hidden_size=1280,
                                  num_hidden_layers=32, image_size=448, patch_size=14,
                                  max_num_tiles=1))
    s["mllama-own-cross-layers"] = mll
    s["mllama-own-cross-layers-no-tower-geometry"] = dict(
        copy.deepcopy(mll), vision_config=dict(model_type="mllama_vision_model", hidden_size=1280,
                                               num_hidden_layers=32))
    # Pix2Struct's shape: a decoder-only `text_config` (is_decoder) cross-attending to an image
    # encoder whose output length is `seq_len`, not image_size / patch_size
    s["pix2struct-seq-len"] = dict(
        model_type="pix2struct", is_encoder_decoder=True,
        text_config=dict(model_type="pix2struct_text_model", hidden_size=1024, d_kv=64, d_ff=2816,
                         num_heads=16, num_layers=18, is_decoder=True, is_encoder_decoder=True,
                         vocab_size=50244),
        vision_config=dict(model_type="pix2struct_vision_model", hidden_size=1536, d_kv=64,
                           d_ff=3968, num_attention_heads=24, num_hidden_layers=18, seq_len=2048))
    return s


def measure_engine_and_facts(cfg):
    """The engine estimate and the FRD-027 facts, the two other callers of the layer kinds."""
    out = {}
    try:
        from llm_memory_calculator.inference_engines import estimate_memory_for_engine

        for engine in ("llamacpp", None):
            for seq in SEQS:
                em = estimate_memory_for_engine(dict(cfg), engine, batch_size=1, seq_length=seq,
                                                weight_precision="fp16", runtime_precision="fp16")
                tag = f"eng_{engine}_{seq}"
                out[tag + "_kv_mb"] = round(em.kv_cache_bytes / 1e6, 6)
                out[tag + "_state_mb"] = round(getattr(em, "state_bytes", 0.0) / 1e6, 6)
                out[tag + "_total_mb"] = round(em.total_bytes / 1e6, 6)
                out[tag + "_params"] = int(em.parameter_count)
    except Exception as e:
        out["eng_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    try:
        from llm_memory_calculator.model.facts import ModelFacts

        f = ModelFacts.from_config(dict(cfg))
        layers = f.layers
        out["facts_layers"] = [layers.full, layers.sliding, layers.recurrent, layers.sliding_window]
        out["facts_full_attention"] = bool(f.full_attention)
        # what the vLLM planner reserves for the weights before it sizes the KV blocks, and what
        # it sizes them with: KV bytes per token (every attention layer), a sequence's KV at 4k
        # (windows and the cross-attention cache applied) and its recurrent state
        figures = {
            "facts_weight_bytes": lambda: f.weight_bytes(),
            "facts_kv_bytes_per_token": lambda: f.kv_bytes_per_token("bf16"),
            "facts_kv_bytes_4096": lambda: f.kv_bytes_for_context(4096, "bf16"),
            "facts_state_bytes": lambda: f.state_bytes_per_sequence(),
        }
        for key, figure in figures.items():
            try:  # a tree without the figure records its error, not the others'
                out[key] = int(figure())
            except Exception as e:
                out[key + "_error"] = f"{type(e).__name__}: {str(e)[:120]}"
    except Exception as e:
        out["facts_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return out


def measure_genz(cfg_or_name, genz_data):
    """GenZ's own view: per-layer kinds, its KV size, and its decode graph's KV bytes."""
    out = {}
    try:
        from llm_memory_calculator.genz.Models.get_language_model import (
            create_full_decode_model,
            get_configs,
        )
        from llm_memory_calculator.genz.analyse_model import get_model_df, get_summary_table

        mc = get_configs(cfg_or_name)
        kinds = {}
        reps = mc.num_decoder_layers / max(len(mc.layer_type), 1)
        for lt in mc.layer_type:
            kinds[lt[0]] = kinds.get(lt[0], 0) + reps
        out["genz_layer_kinds"] = {k: int(v) for k, v in sorted(kinds.items())}
        ffn = {}
        for lt in mc.layer_type:
            ffn[lt[1]] = ffn.get(lt[1], 0) + reps
        out["genz_ffn_kinds"] = {k: int(v) for k, v in sorted(ffn.items())}
        out["genz_kv_size"] = int(mc.get_kv_size())
        path = create_full_decode_model(
            name=mc, input_sequence_length=4096, output_gen_tokens=0, data_path=genz_data
        )
        df = get_model_df(path, batch_size=1, data_path=genz_data, model_characterstics=True)
        st = get_summary_table(df, model_characterstics=True)
        out["genz_decode_kv_mb"] = round(float(st["KV Cache (MB)"].values[0]), 6)
        out["genz_decode_weights_mb"] = round(float(st["Total Weights (MB)"].values[0]), 6)
    except Exception as e:
        out["genz_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    try:
        # The prefill graph: where an encoder-decoder runs its encoder (round 9)
        from llm_memory_calculator.genz.Models.get_language_model import (
            create_full_prefill_model,
            get_configs,
        )
        from llm_memory_calculator.genz.analyse_model import get_model_df, get_summary_table

        path = create_full_prefill_model(name=get_configs(cfg_or_name), input_sequence_length=4096,
                                         data_path=genz_data)
        df = get_model_df(path, batch_size=1, data_path=genz_data, model_characterstics=True)
        st = get_summary_table(df, model_characterstics=True)
        out["genz_prefill_kv_mb"] = round(float(st["KV Cache (MB)"].values[0]), 6)
        out["genz_prefill_weights_mb"] = round(float(st["Total Weights (MB)"].values[0]), 6)
    except Exception as e:
        out["genz_prefill_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    try:
        # The chunked graph (chunked serving, speculative decoding, the simulation engine's
        # chunked feature): one 512-token prefill chunk beside one sequence decoding at 4k.
        from llm_memory_calculator.genz.Models.get_language_model import (
            create_full_chunked_model,
            get_configs,
        )
        from llm_memory_calculator.genz.analyse_model import get_model_df, get_summary_table

        path = create_full_chunked_model(name=get_configs(cfg_or_name),
                                         prefill_kv_sizes=[(0, 512)], decode_kv_sizes=[4096],
                                         data_path=genz_data)
        df = get_model_df(path, batch_size=1, data_path=genz_data, model_characterstics=True)
        st = get_summary_table(df, model_characterstics=True)
        out["genz_chunked_kv_mb"] = round(float(st["KV Cache (MB)"].values[0]), 6)
    except Exception as e:
        out["genz_chunked_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    try:
        from llm_memory_calculator.genz.Models.get_language_model import get_configs
        from llm_memory_calculator.genz.serving.memory_model import MemoryModel

        mc = get_configs(cfg_or_name)
        mm = MemoryModel(mc, [], tensor_parallel=1, precision_bytes=2)
        out["serving_bytes_per_token"] = int(mm.bytes_per_token_kv)
        # A request's fixed recurrent state (absent before the serving path counted it: 0), the
        # 1,024-token requests that fit 8 GiB, and the prefill-to-decode transfer for 512 + 128
        # tokens (KV x 2 B, plus the state where the tree counts it).
        state = int(getattr(mm, "state_bytes_per_sequence", 0) or 0)
        out["serving_state_bytes"] = state
        per_request = -(-1024 // 16) * 16 * mm.bytes_per_token_kv + state
        out["serving_requests_8gib_1k"] = (8 * 1024**3 // per_request) if per_request else None
    except Exception as e:
        out["serving_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    try:
        # The prefill-to-decode transfer, as DisaggregationAnalyzer computes it (512 + 128 tokens)
        from llm_memory_calculator.genz.Models.get_language_model import get_configs
        from llm_memory_calculator.genz.serving.disaggregation import DisaggregationAnalyzer

        an = DisaggregationAnalyzer(get_configs(cfg_or_name), "H100_GPU")
        out["disagg_transfer_bytes_640"] = int(an._estimate_kv_bytes(512, 128))
    except Exception as e:
        out["disagg_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    out.update(measure_runtime(cfg_or_name))
    return out


#: The device the runtime modelling runs on: big enough that every model in the corpus fits.
RUNTIME_SYSTEM = {"Flops": 400, "Memory_size": 200, "Memory_BW": 2000, "ICN": 300,
                  "real_values": True}


def measure_runtime(cfg_or_name):
    """GenZ's runtime modelling, the paths that hold `get_runtime_breakdown` and the decode
    graph's memory-fit check: prefill (512 tokens), decode (512 + 64), chunked (a 256-token chunk
    beside one 512-token decode), each run to completion -- latency (ms), weights and KV (MB) -- or
    its error; and the decode fit check on a device of 0.6 x the prefill graph's weights (the
    resident model: it must refuse, `decode_fit_refused`)."""
    import contextlib
    import io

    out = {}
    try:
        from llm_memory_calculator.genz.LLM_inference import (
            chunked_moddeling,
            decode_moddeling,
            prefill_moddeling,
        )
        from llm_memory_calculator.genz.Models.get_language_model import get_configs
    except Exception as e:
        return {"runtime_error": f"{type(e).__name__}: {str(e)[:200]}"}

    def summary(res):
        st = res.summary_table if hasattr(res, "summary_table") else None
        row = [round(float(res.Latency), 6)]
        if st is not None and "Total Weights (MB)" in st:
            row += [round(float(st["Total Weights (MB)"].values[0]), 3),
                    round(float(st["KV Cache (MB)"].values[0]), 3)]
        return row

    runs = {
        "prefill": lambda mc: prefill_moddeling(model=mc, batch_size=1, input_tokens=512,
                                                system_name=RUNTIME_SYSTEM, bits="bf16"),
        "decode": lambda mc: decode_moddeling(model=mc, batch_size=1, input_tokens=512,
                                              output_tokens=64, system_name=RUNTIME_SYSTEM,
                                              bits="bf16"),
        "chunked": lambda mc: chunked_moddeling(model=mc, prefill_kv_sizes=[(0, 256)],
                                                decode_kv_sizes=[512], system_name=RUNTIME_SYSTEM,
                                                bits="bf16"),
    }
    weights = None
    for stage, run in runs.items():
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                res = run(get_configs(cfg_or_name))
            out[f"runtime_{stage}"] = summary(res)
            if stage == "prefill" and len(out[f"runtime_{stage}"]) > 1:
                weights = out[f"runtime_{stage}"][1]
        except Exception as e:
            out[f"runtime_{stage}_error"] = f"{type(e).__name__}: {str(e)[:160]}"
    if weights:
        try:
            small = dict(RUNTIME_SYSTEM, Memory_size=0.6 * weights / 1024)
            with contextlib.redirect_stdout(io.StringIO()):
                decode_moddeling(model=get_configs(cfg_or_name), batch_size=1, input_tokens=512,
                                 output_tokens=64, system_name=small, bits="bf16")
            out["decode_fit_refused"] = False
        except ValueError as e:
            out["decode_fit_refused"] = "would not fit" in str(e)
        except Exception as e:
            out["decode_fit_error"] = f"{type(e).__name__}: {str(e)[:160]}"
    return out


def measure_gguf(path, repo_root, genz_data):
    sys.path.insert(0, repo_root)
    row = {"name": "gguf:" + os.path.basename(path)}
    try:
        import bud_component

        cfg, _, _ = bud_component.gguf_to_hf_config(path)
        row["gguf_config_keys"] = sorted(k for k in cfg if k != "name")
        try:
            import gguf
            import numpy as np

            row["gguf_tensor_params"] = int(
                sum(int(np.prod(t.shape)) for t in gguf.GGUFReader(path).tensors)
            )
        except Exception as e:
            row["gguf_tensor_error"] = f"{type(e).__name__}: {e}"
        row.update(
            {k: v for k, v in measure_config(row["name"], cfg, genz_data).items() if k != "name"}
        )
    except Exception as e:
        row["gguf_error"] = f"{type(e).__name__}: {e}"
    finally:
        sys.path.pop(0)
    return row


def run(args):
    rows = measure_corpus(args)
    json.dump(rows, open(args.out, "w"), indent=1, sort_keys=True, default=str)
    print(f"{len(rows)} rows -> {args.out}")


def measure_corpus(args):
    """Every row's figures (see the module docstring), as a list of dicts."""
    global REPO_ROOT, GENZ_DATA
    REPO_ROOT = args.repo_root
    GENZ_DATA = args.genz_data
    EMPTY_DB_ROWS.clear()
    NOT_LANGUAGE_MODELS.clear()
    warnings.simplefilter("ignore")
    os.makedirs(os.path.join(args.genz_data, "model"), exist_ok=True)
    os.makedirs(os.path.join(args.genz_data, "sparsity"), exist_ok=True)
    rows = []
    for name, cfg in collect(args):
        rows.append(measure_config(name, cfg, args.genz_data))
    if not args.no_synthetic:
        fixtures_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "tests", "fixtures")
        for name, cfg in synthetic_configs(fixtures_root).items():
            rows.append(measure_config("synth:" + name, cfg, args.genz_data))
    for path in collect_ggufs(args):
        rows.append(measure_gguf(path, args.repo_root, args.genz_data))
    try:
        from llm_memory_calculator.genz.Models.default_models import MODEL_DICT

        for model_name in sorted(MODEL_DICT.list_models()):
            row = {"name": "genz:" + model_name}
            row.update(measure_genz(model_name, args.genz_data))
            rows.append(row)
    except Exception:
        traceback.print_exc()
    if EMPTY_DB_ROWS:
        rows.append({"name": "_empty_db_rows", "count": len(EMPTY_DB_ROWS),
                     "model_ids": sorted(EMPTY_DB_ROWS)})
    if NOT_LANGUAGE_MODELS:
        rows.append({"name": "_not_language_models", "count": len(NOT_LANGUAGE_MODELS),
                     "configs": sorted(NOT_LANGUAGE_MODELS)})
    return rows


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def diff(before_path, after_path, report):
    before = {r["name"]: r for r in json.load(open(before_path))}
    after = {r["name"]: r for r in json.load(open(after_path))}
    changed, unchanged = [], 0
    for name in sorted(set(before) | set(after)):
        b, a = before.get(name, {}), after.get(name, {})
        moves = []
        for key in sorted(set(b) | set(a)):
            if key == "name":
                continue
            bv, av = b.get(key), a.get(key)
            if bv == av:
                continue
            if _num(bv) and _num(av) and abs(bv - av) <= 1e-9 * max(1.0, abs(bv)):
                continue
            moves.append((key, bv, av))
        if moves:
            changed.append((name, a.get("model_type") or b.get("model_type"), moves))
        else:
            unchanged += 1
    lines = [
        "# KV / memory corpus diff",
        "",
        f"before: `{os.path.basename(before_path)}`  after: `{os.path.basename(after_path)}`",
        "",
        f"{len(set(before) | set(after))} configs measured; {unchanged} unchanged; "
        f"{len(changed)} changed.",
        "",
    ]
    for name, mtype, moves in changed:
        lines.append(f"## {name} ({mtype})")
        lines.append("")
        lines.append("| figure | before | after |")
        lines.append("|---|---|---|")
        for key, bv, av in moves:
            lines.append(f"| {key} | {json.dumps(bv)} | {json.dumps(av)} |")
        lines.append("")
    text = "\n".join(lines) + "\n"
    if report:
        open(report, "w").write(text)
    print(text[:4000])


# ------------------------------------------------------------------------------------- the gate
#
# A figure moves in the UNSAFE direction when the branch sizes memory below origin/main (a KV,
# state or weight figure goes down) or admits more than origin/main (a capacity figure goes up:
# vLLM's blocks, tokens and running sequences, the serving path's requests, the scheduler's
# peak). Each such move must be named in the allowlist -- config, reader (the figure), both
# figures, and the ground truth the branch's figure is held against -- or the gate fails.
#
#   python kv_corpus.py --gate main.json branch.json --allowlist kv_gate_allowlist.json

import re  # noqa: E402

#: Capacity figures: higher admits more requests into the same memory.
CAPACITY_FIGURES = re.compile(
    r"^(vllm_limits\[[012]\]|vllm_running_\d+|serving_requests_\w+|sim_max_running)$")
#: Neither memory nor capacity: kinds and layer counts, latencies (`runtime_*[0]`), vLLM's KV
#: group counts, the KV planner's layer counts and block size, flags, GGUF bookkeeping.
NOT_AN_AMOUNT = re.compile(
    r"^(count|genz_layer_kinds\..*|genz_ffn_kinds\..*|kv_(full|sliding|free|cross)_layers"
    r"|facts_layers\[\d+\]|conv_layers\[\d+\]|runtime_\w+\[0\]|vllm_limits\[[34]\]"
    r"|kvgeo_\d+\[[124]\]|gguf_.*)$")


def _leaves(key, value):
    """Every numeric leaf of a row's figure, as ``(path, number)`` (booleans are flags)."""
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        yield key, value
    elif isinstance(value, list):
        for i, x in enumerate(value):
            yield from _leaves(f"{key}[{i}]", x)
    elif isinstance(value, dict):
        for k, x in value.items():
            yield from _leaves(f"{key}.{k}", x)


def figure_kind(path):
    """``"capacity"``, ``"memory"`` or None (not an amount)."""
    if path.endswith("error") or NOT_AN_AMOUNT.match(path):
        return None
    return "capacity" if CAPACITY_FIGURES.match(path) else "memory"


def unsafe_moves(main_rows, branch_rows, names=None):
    """``[(config, reader, main, branch, kind)]``: every figure the branch moves in the unsafe
    direction against origin/main, over the configs both measured (or ``names``)."""
    main = {r["name"]: r for r in main_rows}
    branch = {r["name"]: r for r in branch_rows}
    out = []
    for name in sorted(set(main) & set(branch) if names is None else names):
        if name not in main or name not in branch:
            continue
        a, b = main[name], branch[name]
        for key in sorted(set(a) & set(b)):
            fa, fb = dict(_leaves(key, a[key])), dict(_leaves(key, b[key]))
            for path in sorted(set(fa) & set(fb)):
                kind = figure_kind(path)
                if kind is None:
                    continue
                x, y = fa[path], fb[path]
                tol = 1e-9 * max(1.0, abs(x))
                if (kind == "capacity" and y > x + tol) or (kind == "memory" and y < x - tol):
                    out.append((name, path, x, y, kind))
    return out


def _figures(row):
    """``{path: number}``: every numeric figure of a row."""
    out = {}
    for key, value in row.items():
        out.update(_leaves(key, value))
    return out


def _error_of(row, path):
    """The error a row records for the reader behind a figure (``runtime_prefill[1]`` ->
    ``runtime_prefill_error``, ``facts_weight_bytes`` -> ``facts_weight_bytes_error`` or
    ``facts_error``), or None."""
    top = re.split(r"[\[.]", path)[0]
    for key in (top + "_error", top.split("_")[0] + "_error"):
        if row.get(key):
            return str(row[key])
    return None


_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _repo_owned(module):
    """Whether a top-level module name is this repository's own (a package or module under the
    repo root, BudSimulator/ or the calculator's src/), not a third-party dependency."""
    for root in (_REPO, os.path.join(_REPO, "BudSimulator"),
                 os.path.join(_REPO, "llm-memory-calculator", "src")):
        if os.path.isdir(os.path.join(root, module)) or os.path.isfile(
                os.path.join(root, module + ".py")):
            return True
    return False


def unavailable(row, path):
    """The third-party module a figure's reader could not import in this environment (the GGUF
    reader's `gguf`, the converter's `lxml`), or None. Such a figure is not measured here -- the
    reader cannot run at all -- which is not a figure the branch lost; a reader that raises
    anything else, or cannot import this repository's own code, is."""
    import re as _re

    err = _error_of(row, path) or row.get("gguf_error") or ""
    m = _re.match(r"(ModuleNotFoundError|ImportError): No module named '([\w.]+)'", err)
    if not m:
        return None
    top = m.group(2).split(".")[0]
    return None if _repo_owned(top) else top


def lost_and_new(main_rows, branch_rows, names=None):
    """``[(config, reader, main, branch, kind, change)]``: every figure (memory or capacity) one
    tree produced and the other did not, over the configs both measured (or ``names``) --
    ``"lost"`` where origin/main had a figure and the branch has none (its reader raised, or
    stopped reporting it), ``"new"`` where the branch reports a figure origin/main did not (its
    reader raised on main, or main had no such reader). Either one can hide an under-count the
    unsafe-move check never sees."""
    main = {r["name"]: r for r in main_rows}
    branch = {r["name"]: r for r in branch_rows}
    out = []
    for name in sorted(set(main) & set(branch) if names is None else names):
        if name not in main or name not in branch:
            continue
        fa, fb = _figures(main[name]), _figures(branch[name])
        for path in sorted(set(fa) ^ set(fb)):
            kind = figure_kind(path)
            if kind is None:
                continue
            if path in fa:
                out.append((name, path, fa[path], None, kind, "lost"))
            else:
                out.append((name, path, None, fb[path], kind, "new"))
    return out


def _close(x, y):
    return x is not None and y is not None and abs(x - y) <= 1e-6 * max(1.0, abs(x), abs(y))


def gate(main_rows, branch_rows, allowlist, names=None):
    """The failures: an unsafe move -- or a figure lost or new (``lost_and_new``) -- no allowlist
    entry names, an entry whose figures are not the measured ones, a memory entry (moved or new)
    without a ground truth or whose branch figure is below it, a capacity entry whose memory
    figures (``held_by``) are not all held to their truths, a lost figure whose entry does not
    name the error the branch now records for it."""
    entries = {(e["config"], e["reader"]): e for e in allowlist.get("entries", [])}
    failures = []
    branch_by_name = {r["name"]: r for r in branch_rows}
    changes = [(n, p, x, y, k, "moved") for n, p, x, y, k in
               unsafe_moves(main_rows, branch_rows, names)]
    changes += lost_and_new(main_rows, branch_rows, names)
    for name, path, x, y, kind, change in changes:
        if change == "lost" and unavailable(branch_by_name.get(name, {}), path):
            continue  # not measured in this environment (unavailable_readers says which)
        e = entries.get((name, path))
        where = f"{name} {path}: origin/main {x} -> branch {y}"
        if e is None:
            failures.append(f"{where}: not in the allowlist")
            continue
        if e.get("change", "moved") != change:
            failures.append(f"{where}: {change}, the allowlist says {e.get('change', 'moved')}")
            continue
        if change == "lost":
            # the figure is gone: the entry names main's figure, the error the branch now
            # records for that reader, and why that error is right
            err = _error_of(branch_by_name.get(name, {}), path)
            if not _close(e.get("main"), x):
                failures.append(f"{where}: the allowlist says {e.get('main')} -> None")
            elif not e.get("error") or not (err or "").startswith(e["error"]):
                failures.append(f"{where}: the branch's error is {err!r}, the allowlist says "
                                f"{e.get('error')!r}")
            continue
        if not ((change == "new" and e.get("main") is None) or _close(e.get("main"), x)) \
                or not _close(e.get("branch"), y):
            failures.append(f"{where}: the allowlist says {e.get('main')} -> {e.get('branch')}")
            continue
        if kind == "memory":
            truth = e.get("truth")
            if not isinstance(truth, (int, float)):
                failures.append(f"{where}: no ground truth")
            elif y < truth * (1 - float(e.get("vectors", 0.0))):
                failures.append(f"{where}: below its ground truth {truth}")
        else:
            held = e.get("held_by") or []
            if not held:
                failures.append(f"{where}: a capacity rise names no memory figure it follows")
            for reader in held:
                h = entries.get((name, reader))
                if h is None or not isinstance(h.get("truth"), (int, float)):
                    failures.append(f"{where}: follows {reader}, which has no ground truth")
                elif not _close(h.get("branch"), _figure(branch_rows, name, reader)):
                    failures.append(f"{where}: follows {reader}, whose figure moved")
                elif h["branch"] < h["truth"] * (1 - float(h.get("vectors", 0.0))):
                    failures.append(f"{where}: follows {reader}, below its ground truth "
                                    f"{h['truth']}")
    return failures


def unavailable_readers(main_rows, branch_rows, names=None):
    """``{module: figures}``: the figures origin/main has that this environment's branch run could
    not measure, by the third-party module their reader could not import."""
    branch = {r["name"]: r for r in branch_rows}
    out = {}
    for name, path, _, _, _, change in lost_and_new(main_rows, branch_rows, names):
        mod = change == "lost" and unavailable(branch.get(name, {}), path)
        if mod:
            out[mod] = out.get(mod, 0) + 1
    return out


def _figure(rows, name, path):
    for r in rows:
        if r["name"] == name:
            for key in r:
                for p, v in _leaves(key, r[key]):
                    if p == path:
                        return v
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--db")
    ap.add_argument("--fixtures", nargs="*")
    ap.add_argument("--hf-cache")
    ap.add_argument("--models")
    ap.add_argument("--repo-root", default=os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    ap.add_argument("--genz-data")
    ap.add_argument("--diff", nargs=2)
    ap.add_argument("--report")
    ap.add_argument("--no-synthetic", action="store_true")
    ap.add_argument("--gate", nargs=2, metavar=("MAIN", "BRANCH"))
    ap.add_argument("--allowlist")
    ap.add_argument("--unsafe", nargs=2, metavar=("MAIN", "BRANCH"),
                    help="list every unsafe-direction move as JSON")
    args = ap.parse_args()
    if args.unsafe:
        a, b = json.load(open(args.unsafe[0])), json.load(open(args.unsafe[1]))
        moves = [(n, p, x, y, k, "moved") for n, p, x, y, k in unsafe_moves(a, b)]
        moves += lost_and_new(a, b)
        print(json.dumps([dict(config=n, reader=p, main=x, branch=y, kind=k, change=c)
                          for n, p, x, y, k, c in moves], indent=1))
        return
    if args.gate:
        allow = json.load(open(args.allowlist)) if args.allowlist else {}
        failures = gate(json.load(open(args.gate[0])), json.load(open(args.gate[1])), allow)
        for f in failures:
            print(f)
        skipped = unavailable_readers(json.load(open(args.gate[0])), json.load(open(args.gate[1])))
        if skipped:
            print("not measured here (a reader's third-party module is not installed): "
                  + ", ".join(f"{m}: {n} figures" for m, n in sorted(skipped.items())))
        print(f"{len(failures)} gate failures")
        sys.exit(1 if failures else 0)
    if args.diff:
        diff(args.diff[0], args.diff[1], args.report)
    else:
        run(args)


if __name__ == "__main__":
    main()
