#!/usr/bin/env python3
"""Ground truth for the KV / memory gate, from transformers' own modelling.

For a config, builds the model transformers builds for it (on the meta device: no memory, no
weights) and reads off what the GPU must hold:

* ``params``: every parameter of the model class (`ForConditionalGeneration` /
  `ForCausalLM` / the base model, in that order), and ``lm_params``: the language model alone
  (a multimodal decoder-only model's text model and LM head; an encoder-decoder is whole) --
  what GenZ graphs; ``input_embedding``: the token embedding's parameters (a decode step only
  gathers from it).
* ``kv``: per decoder layer, the K and V elements one token adds to its self-attention cache
  and the window that bounds it -- the widths from the layer's own K / V projections (or the
  fused QKV less its queries, or MLA's cached latent), the kinds and windows from the cache
  transformers builds for the config (``DynamicCache(config=...)``: a full layer grows, a
  sliding or chunked layer is bounded by its window, a linear-attention / Mamba / conv layer
  holds no per-token KV, a KV-shared layer none of its own).

Used to write each allowlist entry's ground truth (``--allowlist-from``); the gate itself
(`kv_corpus.py --gate`, `tests/test_kv_gate.py`) only reads the committed allowlist, so it runs
without torch. A config transformers cannot build as written has its truth written by hand, with
its source (``tests/fixtures/kv_gate/truths.json``, ``--manual``).

    python kv_truth.py --allowlist-from main.json branch.json \
        --manual ../tests/fixtures/kv_gate/truths.json --causes ../tests/fixtures/kv_gate/causes.json \
        --out ../tests/fixtures/kv_gate/allowlist.json <the corpus arguments kv_corpus.py took>
    python kv_truth.py config.json            # one config's truth profile
"""

import json
import logging
import warnings

#: K / V projection names, per family (transformers 5.12): separate projections.
_K_NAMES = ("k_proj", "key", "k", "wk", "k_lin")
_V_NAMES = ("v_proj", "value", "v", "wv", "v_lin")
#: Fused projections: (name, which parts it holds).
_FUSED = {"qkv_proj": "qkv", "c_attn": "qkv", "query_key_value": "qkv", "Wqkv": "qkv",
          "qkv": "qkv", "in_proj": None, "kv_proj": "kv", "wkv": "kv"}
#: MLA's cached latent: K and V share it (kv_lora_rank + qk_rope_head_dim per token).
_LATENT = ("kv_a_proj_with_mqa",)
#: A path through any of these is a cross-attention, not a self-attention cache.
_CROSS = ("encoder_attn", "crossattention", "cross_attn", "encdecattention", "cross_attention",
          "EncDecAttention", "encoder_decoder_attention")


def _build(cfg_dict):
    import torch
    import transformers as T
    from transformers import AutoConfig

    d = json.loads(json.dumps(cfg_dict))
    d.pop("quantization_config", None)
    cfg = AutoConfig.for_model(**d)
    errors = []
    loaders = ("AutoModelForSeq2SeqLM", "AutoModelForSpeechSeq2Seq", "AutoModelForTextToWaveform",
               "AutoModelForImageTextToText", "AutoModelForCausalLM", "AutoModel")
    with torch.device("meta"):
        for name in loaders:
            try:
                return cfg, getattr(T, name).from_config(cfg)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{name}: {str(e)[:120]}")
    raise RuntimeError("; ".join(errors))


def _linear_out(mod):
    import torch

    w = getattr(mod, "weight", None)
    if w is None or w.dim() != 2:
        return None
    if isinstance(mod, torch.nn.Linear):
        return int(mod.out_features)
    if type(mod).__name__ == "Conv1D":  # GPT-2: weight is (in, out)
        return int(w.shape[1])
    return int(w.shape[0])


def _layer_kv_width(layer, heads_width):
    """K + V elements one token adds to this layer's self-attention cache, or 0."""
    k = v = latent = fused = 0
    parents = {path: type(mod).__name__.lower() for path, mod in layer.named_modules()}
    for path, mod in layer.named_modules():
        low = path.lower()
        if any(c.lower() in low for c in _CROSS):
            continue
        leaf = path.split(".")[-1]
        parent = parents.get(path.rsplit(".", 1)[0] if "." in path else "", "")
        # a projection of a softmax attention module (its class says so), not a linear
        # attention's or a cross-attention's
        attention = ("attention" in parent or "attn" in parent) and not any(
            t in parent for t in ("linear", "cross", "lightning", "deltanet", "gated_delta"))
        out = _linear_out(mod)
        if out is None or not attention:
            continue
        if leaf in _LATENT:
            latent += out
        elif leaf in _K_NAMES:
            k += out
        elif leaf in _V_NAMES:
            v += out
        elif leaf in _FUSED and _FUSED[leaf]:
            fused += out - (heads_width if _FUSED[leaf] == "qkv" else 0)
    if latent:
        return latent
    return k + v + fused


def _decoder_layers(model, n, decoder_only=False):
    """The decoder's layer list: a ModuleList of the decoder's depth ``n`` (any depth when ``n``
    is 0) outside a vision or audio tower -- an encoder-decoder's decoder's, else the one
    outside any encoder."""
    import torch

    found = []
    for path, mod in model.named_modules():
        if not isinstance(mod, torch.nn.ModuleList) or (n and len(mod) != n) or len(mod) < 1:
            continue
        if not len(list(mod[0].parameters())):
            continue
        low = path.lower()
        if any(t in low for t in ("vision", "visual", "audio", "qformer", "tower")):
            continue
        found.append(path)
    if not found:
        return None
    # an encoder-decoder's decoder, else the stack outside any encoder, else the shortest path
    # (BLIP's BERT text decoder keeps its layers under `bert.encoder.layer`)
    dec = [p for p in found if "decoder" in p.lower()]
    if decoder_only:
        return dict(model.named_modules())[min(dec, key=len)] if dec else None
    plain = [p for p in found if "encoder" not in p.lower()]
    path = min(dec or plain or found, key=len)
    return dict(model.named_modules())[path]


def truth_profile(cfg_dict):
    """The ground truth for one config (see the module docstring), or ``{"error": ...}``."""
    warnings.simplefilter("ignore")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    try:
        cfg, model = _build(cfg_dict)
    except Exception as e:  # noqa: BLE001
        return {"error": f"transformers cannot build it: {str(e)[:300]}"}
    out = {"class": type(model).__name__,
           "params": int(sum(p.numel() for p in model.parameters()))}
    # the language model GenZ graphs: an encoder-decoder whole; a multimodal decoder's text
    # model and its head
    lm_set = list(model.parameters())
    for attr in ("language_model", "text_model"):
        sub = getattr(getattr(model, "model", model), attr, None) or getattr(model, attr, None)
        if sub is not None:
            lm_set = list(sub.parameters())
            ids = {id(p) for p in lm_set}
            head = getattr(model, "lm_head", None)
            if head is not None and id(head.weight) not in ids:
                lm_set.append(head.weight)
            break
    # an audio codec beside the generator (MusicGen's EnCodec) turns codes into a waveform: not
    # the language model
    codec = getattr(model, "audio_encoder", None)
    if codec is not None:
        drop = {id(p) for p in codec.parameters()}
        lm_set = [p for p in lm_set if id(p) not in drop]
    out["lm_params"] = int(sum(p.numel() for p in lm_set))
    # GenZ graphs matrices: the norm and bias vectors are no operator's weights
    out["lm_matrix_params"] = int(sum(p.numel() for p in lm_set if p.dim() >= 2))
    try:
        out["input_embedding"] = int(model.get_input_embeddings().weight.numel())
    except Exception:  # noqa: BLE001
        out["input_embedding"] = 0
    # per-layer KV
    try:
        from transformers import DynamicCache

        tc = cfg.get_text_config(decoder=True)
        cache = DynamicCache(config=cfg)
        # the decoder's own layer list: an encoder-decoder's decoder stack whatever its depth
        # (T5's decoder may be shallower than `num_layers`), else the stack of the cache's depth
        # (a KV-shared stack's cache is shorter: its last layers read an earlier layer's)
        layers = _decoder_layers(model, 0, decoder_only=True) or _decoder_layers(
            model, len(cache.layers)) or _decoder_layers(model, 0)
        n = len(layers) if layers is not None else len(cache.layers)
        if layers is not None and len(cache.layers) > n:
            cache.layers = cache.layers[:n]
        heads = int(getattr(tc, "num_attention_heads", 0) or getattr(tc, "num_heads", 0) or 0)
        hd = getattr(tc, "head_dim", None) or (
            int(getattr(tc, "hidden_size", 0) or getattr(tc, "d_model", 0)) // max(heads, 1))
        kv = []
        for i in range(n):
            kind, window = "full", None
            if i < len(cache.layers):
                c = cache.layers[i]
                name = type(c).__name__
                if "LinearAttention" in name and "Full" not in name:
                    kind = "none"
                elif "Sliding" in name:
                    kind, window = "sliding", int(getattr(c, "sliding_window", 0) or 0) or None
            else:
                kind = "none"  # a KV-shared layer (Gemma 3n): reads an earlier layer's cache
            width = _layer_kv_width(layers[i], heads * hd) if layers is not None else None
            if kind == "none":
                width = 0
            kv.append([kind, window, width])
        out["kv"] = kv
        out["kv_layers_found"] = layers is not None
    except Exception as e:  # noqa: BLE001
        out["kv_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return out


def kv_elements(profile, tokens=None):
    """K + V elements the self-attention caches hold for ``tokens`` tokens; with no tokens, per
    token on the layers that grow without bound (the full ones). None when a width is unknown.
    A layer entry is ``[kind, window, width]``, or ``[kind, window, width, count]`` for ``count``
    identical layers (a truth written by hand)."""
    if not profile or "kv" not in profile:
        return None
    total = 0
    for entry in profile["kv"]:
        kind, window, width = entry[:3]
        count = entry[3] if len(entry) > 3 else 1
        if width is None:
            return None
        width = width * count
        if tokens is None:
            total += width if kind == "full" else 0
        elif kind == "sliding" and window:
            total += width * min(tokens, window)
        else:
            total += width * tokens if kind != "none" else 0
    return total


# ------------------------------------------------------------------ a reader's ground truth
#
# Each corpus figure is held against what the model holds for the same workload, in the
# figure's own units, at 2 bytes an element (the corpus measures bf16 / fp16). Where a reader's
# workload is a range (the chunked graph: a 512-token chunk beside a 4k decode), the truth is
# its lower bound (the longest context alone), so a figure below it is below any reading.

MIB, MB, GB = 2 ** 20, 1e6, 1e9


def _kv_bytes(profile, tokens=None):
    e = kv_elements(profile, tokens)
    return None if e is None else 2 * e


def _scaled(value, unit):
    return None if value is None else value / unit


def figure_truth(profile, reader):
    """``(truth, source)`` for one corpus figure, or ``(None, why)``."""
    import re

    if not profile or profile.get("error"):
        return None, (profile or {}).get("error", "no truth profile")
    p = profile
    params = p.get("params")
    lm = p.get("lm_matrix_params", p.get("lm_params", params))
    emb = p.get("input_embedding", 0) or 0
    src = p.get("source") or f"transformers 5.12 {p.get('class', '')}"
    if reader in p.get("figures", {}):  # a figure's truth written by hand
        v = p["figures"][reader]
        return v["truth"], v["source"]
    per_token = _kv_bytes(p)
    rules = [
        (r"^(conv_kv_size|genz_kv_size)$", lambda: kv_elements(p), "K+V elements a token adds"
         " on the full-attention layers"),
        (r"^(serving_bytes_per_token|prefix_kv_bytes_per_token|kvgeo_\d+\[0\])$",
         lambda: per_token, "K+V bytes a token adds on the full-attention layers"),
        (r"^facts_kv_bytes_per_token$", lambda: per_token,
         "K+V bytes a token adds on the full-attention layers"),
        (r"^facts_kv_bytes_4096$", lambda: _kv_bytes(p, 4096), "a 4k sequence's self-attention KV"),
        (r"^disagg_transfer_bytes_640$", lambda: _kv_bytes(p, 640), "640 tokens' KV"),
        (r"^genz_(decode|prefill|chunked)_kv_mb$", lambda: _scaled(_kv_bytes(p, 4096), MIB),
         "a 4k sequence's KV (MiB)"),
        (r"^runtime_(prefill|decode|chunked)\[2\]$", lambda: _scaled(_kv_bytes(p, 512), MIB),
         "a 512-token sequence's KV (MiB)"),
        (r"^kv_gb@(\d+)$", lambda: _scaled(_kv_bytes(p, int(reader.split("@")[1])), GB),
         "the sequence's KV (GB)"),
        (r"^report_kv_gb$", lambda: _scaled(_kv_bytes(p, 4096), GB), "a 4k sequence's KV (GB)"),
        (r"^eng_\w+_(\d+)_kv_mb$", lambda: _scaled(_kv_bytes(p, int(reader.split("_")[-3])), MB),
         "the sequence's KV (MB)"),
        (r"^slice_mib_16x4352\[0\]$", lambda: _scaled(
            None if _kv_bytes(p, 4352) is None else 16 * _kv_bytes(p, 4352), MIB),
         "16 sequences' KV at 4,352 tokens (MiB)"),
        (r"^(genz_prefill_weights_mb|conv_prefill_weights_mb|runtime_prefill\[1\])$",
         lambda: _scaled(None if lm is None else 2 * lm, MIB),
         "the language model's weight matrices (MiB; GenZ graphs no norm or bias vector)"),
        (r"^(genz_decode_weights_mb|runtime_(decode|chunked)\[1\])$",
         lambda: _scaled(None if lm is None else 2 * (lm - emb), MIB),
         "the language model's weight matrices less the input embedding a decode step gathers"
         " from (MiB)"),
        (r"^facts_weight_bytes$", lambda: None if params is None or p.get("quantized")
         else 2 * params, "the model's weights at 2 B"),
        (r"^(report_params|eng_\w+_params)$", lambda: params, "the model's parameters"),
        (r"^report_weight_gb$", lambda: _scaled(None if params is None else 2 * params, GB),
         "the model's weights (GB)"),
        (r"^report_total_gb$", lambda: None if params is None or _kv_bytes(p, 4096) is None
         else (2 * params + _kv_bytes(p, 4096)) / GB, "weights and a 4k sequence's KV (GB)"),
        (r"^eng_\w+_(\d+)_total_mb$", lambda: None if params is None
         or _kv_bytes(p, int(reader.split("_")[-3])) is None
         else (2 * params + _kv_bytes(p, int(reader.split("_")[-3]))) / MB,
         "weights and the sequence's KV (MB)"),
    ]
    for pattern, value, what in rules:
        if re.match(pattern, reader):
            v = value()
            if v is None:
                return None, f"{what}: not determined by {src}"
            return v, f"{what}: {src}"
    return None, f"no ground-truth rule for {reader}"


#: GenZ's graph weights, held against the matrices with this relative slack.
GRAPH_WEIGHTS = r"^(genz_(prefill|decode)_weights_mb|conv_prefill_weights_mb|runtime_\w+\[1\])$"
GRAPH_SLACK = 5e-5

#: The memory figures a capacity figure is sized from, per capacity figure.
HELD_BY = {
    "vllm": ["facts_weight_bytes", "facts_kv_bytes_per_token", "facts_kv_bytes_4096"],
    "serving_requests": ["serving_bytes_per_token"],
}


def _held_by(reader):
    if reader.startswith("serving_requests"):
        return HELD_BY["serving_requests"]
    return HELD_BY["vllm"]


def allowlist_from(main_rows, branch_rows, configs, manual, causes):
    """Allowlist entries for every unsafe move between two corpus runs, each with its ground
    truth (``truth_profile`` of the row's config, or a hand-written profile in ``manual``), and
    the truth-only entries the capacity moves are held by."""
    import re

    import kv_corpus

    moves = kv_corpus.unsafe_moves(main_rows, branch_rows)
    branch = {r["name"]: r for r in branch_rows}
    main = {r["name"]: r for r in main_rows}
    profiles = {}

    def profile(name):
        if name not in profiles:
            if name in manual:
                profiles[name] = manual_profile(manual[name], configs.get(name))
            elif name in configs:
                profiles[name] = truth_profile(configs[name])
            else:
                profiles[name] = {"error": "no config for this row"}
        return profiles[name]

    entries = {}
    for name, reader, x, y, kind in moves:
        e = {"config": name, "reader": reader, "main": x, "branch": y, "kind": kind,
             "cause": _cause(causes, name, reader) or derived_cause(main.get(name, {}),
                                                                   branch.get(name, {}), reader)}
        if kind == "memory":
            e["truth"], e["source"] = figure_truth(profile(name), reader)
            if re.match(GRAPH_WEIGHTS, reader):
                # the truth is the matrices; a few small ones have no GenZ operator (T5's
                # relative-position bias, Mllama's 8 extra embedding rows)
                e["vectors"] = GRAPH_SLACK
        else:
            e["held_by"] = _held_by(reader)
            if not _cause(causes, name, reader):
                e["cause"] = ("a capacity sized from " + ", ".join(e["held_by"]) + " -- each held "
                              "to its ground truth (its own entry) -- admits what that memory holds")
        entries[(name, reader)] = e
    for (name, reader), e in list(entries.items()):
        for held in e.get("held_by", []):
            if (name, held) in entries:
                continue
            fig = kv_corpus._figure(branch_rows, name, held)
            truth, source = figure_truth(profile(name), held)
            entries[(name, held)] = {"config": name, "reader": held,
                                     "main": kv_corpus._figure(main_rows, name, held),
                                     "branch": fig, "kind": "memory", "truth": truth,
                                     "source": source, "cause": "held by a capacity figure"}
    return sorted(entries.values(), key=lambda e: (e["config"], e["reader"]))


_KV_READER = (r"kv|bytes_per_token|disagg|slice_mib|runtime_\w+\[2\]")
_WEIGHT_READER = (r"weight|params|runtime_\w+\[1\]|total")


def derived_cause(a, b, reader):
    """Why a figure moved, read off the two rows' own layer counts and kinds (both trees)."""
    import re

    notes = []
    fa, fb = a.get("facts_layers") or [None] * 4, b.get("facts_layers") or [None] * 4
    if re.search(_KV_READER, reader):
        pairs = [("kv_free_layers", "layers without a KV cache (recurrent / linear attention / "
                  "FFN-only / KV-shared)"),
                 ("kv_sliding_layers", "windowed layers, bounded by their window"),
                 ("kv_full_layers", "full-attention layers")]
        for key, what in pairs:
            if a.get(key) != b.get(key) and b.get(key) is not None:
                notes.append(f"{what}: {a.get(key)} -> {b.get(key)}")
        if fa[2] != fb[2] and fb[2] is not None:
            notes.append(f"the FRD-027 facts' recurrent layers: {fa[2]} -> {fb[2]}")
        if fa[1] != fb[1] and fb[1] is not None:
            notes.append(f"the facts' windowed layers: {fa[1]} -> {fb[1]}")
        ka, kb = a.get("genz_layer_kinds"), b.get("genz_layer_kinds")
        if ka != kb and kb:
            notes.append(f"GenZ's layer kinds: {ka} -> {kb}")
        if a.get("conv_layers") != b.get("conv_layers") and b.get("conv_layers"):
            notes.append(f"the converter's [encoder, decoder] depths: {a.get('conv_layers')} -> "
                         f"{b.get('conv_layers')}")
    elif re.search(_WEIGHT_READER, reader):
        ga, gb = a.get("genz_ffn_kinds"), b.get("genz_ffn_kinds")
        if ga != gb and gb:
            notes.append(f"GenZ's FFN per layer: {ga} -> {gb}")
        ka, kb = a.get("genz_layer_kinds"), b.get("genz_layer_kinds")
        if ka != kb and kb:
            notes.append(f"GenZ's layer kinds: {ka} -> {kb}")
        if a.get("conv_layers") != b.get("conv_layers") and b.get("conv_layers"):
            notes.append(f"the converter's [encoder, decoder] depths: {a.get('conv_layers')} -> "
                         f"{b.get('conv_layers')}")
        if a.get("model_type") != b.get("model_type"):
            notes.append(f"the calculator's model type: {a.get('model_type')} -> "
                         f"{b.get('model_type')}")
        if any(k.endswith("error") for k in a) and not notes:
            errs = sorted(k for k in a if k.endswith("error"))
            notes.append("origin/main errored in " + ", ".join(errs))
    if not notes:
        notes.append("the config's own shape, read as stated (origin/main read a default or "
                     "another stack's); the ground truth is the model's own")
    return "; ".join(notes)


def manual_profile(spec, config):
    """A truth written by hand (``kv_gate_truths.json``): its own fields, over transformers' truth
    for the row's config when it says ``"base": "transformers"`` -- built from its `text_config`
    (``"build_from"``), with keys replaced (``"build_as"``) or dropped (``"drop_keys"``) where the
    row's config is a shape transformers' class does not take as written."""
    import copy

    spec = dict(spec)
    out = {}
    if spec.pop("base", None) == "transformers" and config is not None:
        cfg = copy.deepcopy(config)
        frm = spec.pop("build_from", None)
        if frm:
            cfg = copy.deepcopy(cfg[frm])
        for k in spec.pop("drop_keys", []):
            cfg.pop(k, None)
        for k, v in spec.pop("build_as", {}).items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k] = {**cfg[k], **v}
            else:
                cfg[k] = v
        out = truth_profile(cfg)
    for k in ("build_from", "drop_keys", "build_as"):
        spec.pop(k, None)
    out.update(spec)
    return out


def _cause(causes, name, reader):
    import re

    for c in causes:
        if re.search(c["config"], name) and re.search(c.get("reader", "."), reader):
            return c["cause"]
    return None


def corpus_configs(args):
    """``{row name: config}`` for every row the corpus measures, as kv_corpus collects them."""
    import os
    import sys

    import kv_corpus

    configs = dict(kv_corpus.collect(args))
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests",
                        "fixtures")
    for n, c in kv_corpus.synthetic_configs(root).items():
        configs["synth:" + n] = c
    sys.path.insert(0, args.repo_root)
    try:
        import bud_component

        for path in kv_corpus.collect_ggufs(args):
            try:
                cfg, _, _ = bud_component.gguf_to_hf_config(path)
                configs["gguf:" + os.path.basename(path)] = cfg
            except Exception:  # noqa: BLE001
                pass
    finally:
        sys.path.pop(0)
    return configs


def main():
    import argparse
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="*")
    ap.add_argument("--allowlist-from", nargs=2, metavar=("MAIN", "BRANCH"))
    ap.add_argument("--manual", help="hand-written truth profiles, {config: profile}")
    ap.add_argument("--causes", help="[{config: regex, reader: regex, cause: text}]")
    ap.add_argument("--out")
    ap.add_argument("--db")
    ap.add_argument("--fixtures", nargs="*")
    ap.add_argument("--hf-cache")
    ap.add_argument("--models")
    ap.add_argument("--repo-root", default=os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    args = ap.parse_args()
    if args.allowlist_from:
        manual = json.load(open(args.manual)) if args.manual else {}
        causes = json.load(open(args.causes)) if args.causes else []
        entries = allowlist_from(json.load(open(args.allowlist_from[0])),
                                 json.load(open(args.allowlist_from[1])),
                                 corpus_configs(args), manual.get("profiles", {}), causes)
        json.dump({"about": __doc__.split("\n\n")[0], "entries": entries},
                  open(args.out, "w"), indent=1, sort_keys=True, default=str)
        print(f"{len(entries)} entries -> {args.out}")
        return
    for p in args.config:
        prof = truth_profile(json.load(open(p)))
        prof["kv_per_token_full"] = kv_elements(prof)
        json.dump(prof, sys.stdout)
        print()


if __name__ == "__main__":
    main()
