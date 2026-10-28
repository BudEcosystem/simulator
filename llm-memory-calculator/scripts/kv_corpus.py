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

Corpus: every row of the BudSimulator database that carries a config (rows whose
`config_json` is empty are counted and reported, not measured -- they would only measure the
calculator's defaults), every hybrid fixture, every config.json in the HuggingFace cache and
the model folders that names a decoder geometry, every GGUF in the model folders (read
through bud_component's GGUF reader, beside the file's own tensor count), and every built-in
GenZ model. Model folders and caches are only read.

Figures, per config: the calculator's KV / state / weights / totals; the engine estimate
(`estimate_memory_for_engine`, llama.cpp and generic, at 4k and 32k: KV, state, total,
parameters); the FRD-027 `ModelFacts` layer counts; GenZ's per-layer attention and FFN kinds,
its KV elements per token (`get_kv_size`, which the disaggregation KV transfer multiplies by
2 bytes), its decode and chunked graphs' KV and weights; and the serving `MemoryModel`'s KV
bytes per token, per-request state, the 1,024-token requests 8 GiB admits, and the
disaggregation transfer for 640 tokens; with ``--repo-root``, BudSimulator's database
converter (`ModelConverter.hf_to_genz`): its encoder / decoder depths, KV elements per token and
per-sequence fixed bytes; and every other reader of the layer kinds: GenZ's prefill graph
(weights, KV), the prefix-cache analyser's bytes per token, the KV planner's `KVGeometry`, the
vLLM planner (`engine/vllm.py`: limits, and the running sequences `predict` gives 1,024 clients
at 1k and 8k prompts), its profile scheduler (`perf.predict.sim_config` + `perf.sched.simulate`)
and its shared-slice sizing (`runtime.modules.capacity.shared_slice_mib`).

Fixture roots are read recursively, keeping the files that name a decoder (or an encoder-decoder's)
geometry; the planner's engine records beside them are not model configs. Built-in synthetic
shapes (``synthetic_configs``; ``--no-synthetic`` leaves them out) cover what no shipped or cached
config does: depth stated only in a per-layer list, nested decoders (VisionEncoderDecoder,
EncoderDecoderModel, MusicGen, BLIP-2 / InstructBLIP with a flan-T5 `text_config`, Florence-2),
ProphetNet, RWKV, xLSTM, Llama 4's interleaved MoE, GGUF-derived short window patterns, and the
encoder-decoder and dense spellings around them.
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
            if real in seen_fixtures:
                continue
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
                corpus.append(("config:" + rel, cfg))
    return corpus


def collect_ggufs(args):
    out = []
    if not args.models or not os.path.isdir(args.models):
        return out
    seen = set()
    for path in sorted(glob.glob(os.path.join(args.models, "**", "*.gguf"), recursive=True)):
        real = os.path.realpath(path)
        base = os.path.basename(path).lower()
        if not os.path.isfile(real):  # dangling link into another machine's layout
            continue
        if real in seen or "mmproj" in base or os.path.getsize(real) < 10 * 1024 * 1024:
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
        out["disagg_transfer_bytes_640"] = int(mc.get_kv_size() * 2 * 640 + _fixed_bytes(mc))
    except Exception as e:
        out["serving_error"] = f"{type(e).__name__}: {str(e)[:200]}"
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
    global REPO_ROOT
    REPO_ROOT = args.repo_root
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
    json.dump(rows, open(args.out, "w"), indent=1, sort_keys=True, default=str)
    print(f"{len(rows)} rows -> {args.out}")


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
    args = ap.parse_args()
    if args.diff:
        diff(args.diff[0], args.diff[1], args.report)
    else:
        run(args)


if __name__ == "__main__":
    main()
