#!/usr/bin/env python3
"""Ground truth for the KV / memory gate, from transformers' own modelling.

For a config, builds the model transformers builds for it (on the meta device: no memory, no
weights) and reads off what the GPU must hold:

* ``params``: every parameter of the model class (`ForConditionalGeneration` /
  `ForCausalLM` / the base model, in that order), and ``lm_params``: the language model alone
  (a multimodal decoder-only model's text model and LM head; an encoder-decoder is whole) --
  what GenZ graphs; ``input_embedding``: the token embedding's parameters, and ``gathered``: every
  embedding table's (token, learned positions, codebooks), which a decode step only gathers from.
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
    python kv_truth.py --checkpoint inclusionAI/Ling-flash-2.0   # a checkpoint's tensor counts
"""

import json
import logging
import re
import warnings

#: K / V projection names, per family (transformers 5.12): separate projections (ProphetNet's
#: `key_proj` / `value_proj`).
_K_NAMES = ("k_proj", "key", "k", "wk", "k_lin", "key_proj")
_V_NAMES = ("v_proj", "value", "v", "wv", "v_lin", "value_proj")
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
    """K + V elements one token adds to this layer's self-attention cache: 0 for a layer with no
    softmax self-attention, None for one whose attention projects K and V under names this table
    does not know (undetermined, so the figure gets no truth -- never 0, which would accept any
    figure)."""
    k = v = latent = fused = 0
    projections = 0  # linear maps inside a softmax self-attention module
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
        projections += 1
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
    width = k + v + fused
    if projections and not width:
        return None
    return width


def _is_cross(path, parent):
    return any(c.lower() in path.lower() for c in _CROSS) or "cross" in parent


def _layer_cross_kv_width(layer):
    """K + V elements one encoder token adds to this layer's cross-attention cache (its K / V
    projections of the encoder's output), 0 for a layer without one."""
    width = 0
    parents = {path: type(mod).__name__.lower() for path, mod in layer.named_modules()}
    for path, mod in layer.named_modules():
        parent = parents.get(path.rsplit(".", 1)[0] if "." in path else "", "")
        if not _is_cross(path, parent):
            continue
        leaf = path.split(".")[-1]
        out = _linear_out(mod)
        if out is not None and (leaf in _K_NAMES or leaf in _V_NAMES):
            width += out
        elif out is not None and _FUSED.get(leaf) == "kv":
            width += out
    return width


def _layer_state(layer):
    """``[conv, matrix]``: the elements a sequence's recurrent state holds in this layer, read off
    transformers' own mixer modules -- the causal depthwise convolution's ring buffer (each
    `Conv1d` with one input channel per group: channels x (kernel - 1)) and the recurrent matrix
    (Gated DeltaNet: `num_v_heads x head_k_dim x head_v_dim`; Mamba-2: `num_heads x head_dim x
    ssm_state_size`; Zamba: `n_mamba_heads x mamba_head_dim x ssm_state_size`; Mamba-1:
    `intermediate_size x ssm_state_size`). ``[0, 0]`` for a layer with neither."""
    import torch

    conv = matrix = 0
    for path, mod in layer.named_modules():
        parent_cross = _is_cross(path, "")
        if parent_cross:
            continue
        if isinstance(mod, torch.nn.Conv1d) and mod.groups == mod.in_channels \
                and mod.kernel_size[0] > 1:
            conv += int(mod.in_channels) * (int(mod.kernel_size[0]) - 1)
        g = lambda k: getattr(mod, k, None)  # noqa: E731
        if all(isinstance(g(k), int) for k in ("num_v_heads", "head_k_dim", "head_v_dim")):
            matrix += g("num_v_heads") * g("head_k_dim") * g("head_v_dim")
        elif all(isinstance(g(k), int) for k in ("num_heads", "head_dim", "ssm_state_size")):
            matrix += g("num_heads") * g("head_dim") * g("ssm_state_size")
        elif all(isinstance(g(k), int) for k in ("n_mamba_heads", "mamba_head_dim",
                                                  "ssm_state_size")):
            matrix += g("n_mamba_heads") * g("mamba_head_dim") * g("ssm_state_size")
        elif all(isinstance(g(k), int) for k in ("intermediate_size", "ssm_state_size")):
            matrix += g("intermediate_size") * g("ssm_state_size")
    return [conv, matrix]


def _decoder_layers(model, n, decoder_only=False):
    """The decoder's layer list: a ModuleList of blocks (each holding modules of its own -- not a
    list of output heads, MusicGen's `lm_heads`) of the decoder's depth ``n`` (any depth when
    ``n`` is 0) outside a vision or audio tower -- an encoder-decoder's decoder's, else the one
    outside any encoder."""
    import torch

    found = []
    for path, mod in model.named_modules():
        if not isinstance(mod, torch.nn.ModuleList) or (n and len(mod) != n) or len(mod) < 1:
            continue
        if not len(list(mod[0].parameters())) or not len(list(mod[0].children())):
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


def _untied_head(cfg_dict, cfg, model):
    """The LM head's parameters when the checkpoint holds it apart but the build shares it with
    the token embedding: a config class that ties whatever the config says (transformers 5.12's
    T5, mT5 and UMT5: "we tie weights always"), built for a config that states an untied head
    (`tie_word_embeddings: false`: T5 v1.1, flan-T5, mT5, whose checkpoints carry `lm_head`) or
    leaves the flag out (transformers 4.x's MT5Config defaulted to untied; the simulator reads
    an omitted flag on such a class as untied). 0 otherwise."""
    try:
        out_emb, in_emb = model.get_output_embeddings(), model.get_input_embeddings()
    except Exception:  # noqa: BLE001
        return 0
    w_out = getattr(out_emb, "weight", None)
    if w_out is None or w_out is not getattr(in_emb, "weight", None):
        return 0
    try:
        tc = cfg.get_text_config(decoder=True)
    except Exception:  # noqa: BLE001
        tc = cfg
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forced = type(tc)(tie_word_embeddings=False).tie_word_embeddings is True
    except Exception:  # noqa: BLE001
        forced = False
    if not forced:
        return 0
    text = cfg_dict.get("text_config") if isinstance(cfg_dict.get("text_config"), dict) else {}
    stated = text.get("tie_word_embeddings", cfg_dict.get("tie_word_embeddings"))
    return int(w_out.numel()) if stated is not True else 0


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
    untied = _untied_head(cfg_dict, cfg, model)
    if untied:
        out["params"] += untied
        out["untied_head"] = untied
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
    out["lm_params"] = int(sum(p.numel() for p in lm_set)) + untied
    # GenZ graphs matrices: the norm and bias vectors are no operator's weights
    out["lm_matrix_params"] = int(sum(p.numel() for p in lm_set if p.dim() >= 2)) + untied
    try:
        out["input_embedding"] = int(model.get_input_embeddings().weight.numel())
    except Exception:  # noqa: BLE001
        out["input_embedding"] = 0
    # every table a step looks rows up in rather than multiplies by (the token embedding, a
    # learned position table, an audio decoder's codebook embeddings): a decode step gathers one
    # row from each, and GenZ's decode graph graphs no operator for a gather
    import torch

    lm_ids, seen, gathered = {id(p) for p in lm_set}, set(), 0
    for mod in model.modules():
        w = getattr(mod, "weight", None)
        if isinstance(mod, torch.nn.Embedding) and id(w) in lm_ids and id(w) not in seen:
            seen.add(id(w))
            gathered += int(w.numel())
    out["gathered"] = max(gathered, out["input_embedding"])
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
            elif width == 0 and layers is not None and any(_layer_state(layers[i])):
                # the cache expects attention here, the class built a recurrent / conv mixer: a
                # layer name the class does not read (LFM2 builds a `sliding_attention` layer as
                # a short convolution) -- transformers does not build this config as written
                width = None
            kv.append([kind, window, width])
        out["kv"] = kv
        out["kv_layers_found"] = layers is not None
        if layers is not None:
            # what a sequence holds whatever its length: each layer's recurrent state, and the
            # K / V each encoder token adds to the decoder's cross-attention caches
            out["state"] = [_layer_state(layer) for layer in layers]
            out["cross_kv"] = sum(_layer_cross_kv_width(layer) for layer in layers)
    except Exception as e:  # noqa: BLE001
        out["kv_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return out


#: A token embedding's tensor name, last component(s), in a checkpoint.
_EMBEDDING_TENSORS = ("embed_tokens.weight", "word_embeddings.weight", "wte.weight",
                      "embedding.weight", "tok_embeddings.weight")


def _hub_get(url, headers=None, tries=10):
    import time
    import urllib.request

    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:  # noqa: BLE001
            if attempt == tries - 1:
                raise
            time.sleep(min(5 * (attempt + 1), 30))
    return b""


def checkpoint_profile(repo, revision="main"):
    """A truth from a hub checkpoint's own tensors: every safetensors shard's header (its first
    bytes, read with HTTP range requests -- no weights downloaded) gives each tensor's shape.
    ``params`` counts every tensor once (a tied head is stored once), ``lm_matrix_params`` the
    tensors of two or more dimensions, ``input_embedding`` the token embedding's."""
    import struct

    api = json.loads(_hub_get(f"https://huggingface.co/api/models/{repo}/revision/{revision}"))
    sha = api.get("sha") or revision
    files = [s["rfilename"] for s in api.get("siblings", [])]
    shards = sorted(f for f in files if f.endswith(".safetensors"))
    # One copy of the weights: a repo can ship them twice, in the transformers layout
    # (`model*.safetensors`, named by its index) and in another (Mistral's `consolidated
    # .safetensors`), and summing both doubled Voxtral-Mini-4B-Realtime's 4,429,679,360.
    if "model.safetensors.index.json" in files:
        index = json.loads(_hub_get(
            f"https://huggingface.co/{repo}/resolve/{sha}/model.safetensors.index.json"))
        shards = sorted(set(index.get("weight_map", {}).values())) or shards
    elif "model.safetensors" in files:
        shards = ["model.safetensors"]
    def header(shard):
        url = f"https://huggingface.co/{repo}/resolve/{sha}/{shard}"
        n = struct.unpack("<Q", _hub_get(url, {"Range": "bytes=0-7"})[:8])[0]
        return json.loads(_hub_get(url, {"Range": f"bytes=8-{7 + n}"}))

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=8) as pool:
        headers = list(pool.map(header, shards))
    params = matrices = embedding = 0
    for header in headers:
        for name, t in header.items():
            if name == "__metadata__":
                continue
            count = 1
            for d in t["shape"]:
                count *= int(d)
            params += count
            if len(t["shape"]) >= 2:
                matrices += count
            if any(name.endswith(e) for e in _EMBEDDING_TENSORS) and "position" not in name:
                embedding = max(embedding, count)
    return {"params": params, "lm_params": params, "lm_matrix_params": matrices,
            "input_embedding": embedding, "checkpoint": f"{repo}@{sha}",
            "shards": len(shards)}


def kv_elements(profile, tokens=None, at=None):
    """K + V elements the self-attention caches hold for ``tokens`` tokens; with no tokens, the
    elements one token adds: with ``at``, the ``at``-th token of a sequence (every full layer, and
    every windowed layer whose window still holds it: window >= ``at``); without, a token inside
    every window (every layer that caches, full or windowed -- what each token adds until the
    windows fill). None when a width is unknown. A layer entry is ``[kind, window, width]``, or
    ``[kind, window, width, count]`` for ``count`` identical layers (a truth written by hand).

    Counting the full layers alone made the per-token truth of a stack whose every layer is
    windowed 0, which accepts any figure (Voxtral-Realtime, LFM2.5-Audio's detokenizers)."""
    if not profile or "kv" not in profile:
        return None
    total = 0
    for entry in profile["kv"]:
        kind, window, width = entry[:3]
        count = entry[3] if len(entry) > 3 else 1
        if width is None:
            return None
        width = width * count
        if kind == "none":
            continue
        if tokens is None:
            if kind == "full" or at is None or not window or window >= at:
                total += width
        elif kind == "sliding" and window:
            total += width * min(tokens, window)
        else:
            total += width * tokens
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


def _fixed_bytes(profile, cross):
    """Bytes a sequence holds whatever its length, at 2 B an element (transformers' caches hold
    the state in the model's dtype; an engine that keeps the matrix in fp32 holds more): the
    recurrent state (``state``, per layer ``[conv, matrix]``, or a hand-written ``state_bytes``),
    and with ``cross`` the cross-attention cache of the encoder's ``encoder_tokens`` (written by
    hand, with its arithmetic: the tower's patches, the audio frames). None when a part is not
    determined: a built profile without its layers, a hand profile whose layers include one
    without a cache (``none``: recurrent, or KV-shared) and no ``state_bytes``, a
    cross-attention cache whose encoder length nobody wrote down."""
    if "state_bytes" in profile:
        state = profile["state_bytes"]
    elif "state" in profile:
        state = 2 * sum(conv + matrix for conv, matrix in profile["state"])
    elif any(e[0] == "none" for e in profile.get("kv", [])):
        return None
    else:
        state = 0
    if not cross:
        return state
    cross_kv = profile.get("cross_kv", 0) or 0
    if not cross_kv:
        return state
    if profile.get("encoder_tokens") is None:
        return None
    return state + 2 * cross_kv * int(profile["encoder_tokens"])


#: The fixed per-sequence figures that carry a fixed-length encoder's cross-attention cache.
STATE_WITH_CROSS = r"^(serving_state_bytes|conv_state_bytes|vllm_limits\[5\])$"


def figure_truth(profile, reader):
    """``(truth, source)`` for one corpus figure, or ``(None, why)``."""
    import re

    if not profile or profile.get("error"):
        return None, (profile or {}).get("error", "no truth profile")
    p = profile
    params = p.get("params")
    lm = p.get("lm_matrix_params", p.get("lm_params", params))
    emb = p.get("input_embedding", 0) or 0
    gathered = p.get("gathered", emb) or emb
    src = p.get("source") or f"transformers 5.12 {p.get('class', '')}"
    if p.get("untied_head"):
        src += (f", with the {p['untied_head']:,}-parameter LM head the class ties whatever the"
                " config says and the checkpoint holds apart")
    if reader in p.get("figures", {}):  # a figure's truth written by hand
        v = p["figures"][reader]
        return v["truth"], v["source"]
    per_token = _kv_bytes(p)
    at = re.match(r"^kvgeo_(\d+)\[0\]$", reader)
    rules = [
        (r"^(conv_kv_size|genz_kv_size)$", lambda: kv_elements(p), "K+V elements a token adds"
         " on every layer that caches it (full, and windowed while inside the window)"),
        (r"^(serving_bytes_per_token|prefix_kv_bytes_per_token|facts_kv_bytes_per_token)$",
         lambda: per_token, "K+V bytes a token adds on every layer that caches it (full, and"
         " windowed while inside the window)"),
        (r"^kvgeo_\d+\[0\]$",
         lambda: None if kv_elements(p, at=int(at.group(1))) is None
         else 2 * kv_elements(p, at=int(at.group(1))),
         "K+V bytes the context's last token adds (the full layers, and the windowed layers whose"
         " window still holds it)"),
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
        # the prefill and chunked graphs look their tokens up in the input embedding and hold it
        # (create_full_chunked_model graphs input_embedding)
        (r"^(genz_prefill_weights_mb|conv_prefill_weights_mb|runtime_(prefill|chunked)\[1\])$",
         lambda: _scaled(None if lm is None else 2 * lm, MIB),
         "the language model's weight matrices (MiB; GenZ graphs no norm or bias vector)"),
        (r"^(genz_decode_weights_mb|runtime_decode\[1\])$",
         lambda: _scaled(None if lm is None else 2 * (lm - gathered), MIB),
         "the language model's weight matrices less the tables a decode step gathers a row from"
         " -- the input embedding, a learned position table, codebook embeddings (MiB)"),
        (r"^facts_weight_bytes$", lambda: None if params is None or p.get("quantized")
         else 2 * params, "the model's weights at 2 B"),
        # the weights the serving path holds on the device (a ServingSimulator's weight load, the
        # cluster / disaggregation B_eff), at the precision measured (bf16) whatever the checkpoint
        (r"^(serving_weight_bytes|conv_serving_weight_bytes)$",
         lambda: None if params is None else 2 * params,
         "the model's weights at 2 B (every parameter: the device holds the whole model)"),
        (STATE_WITH_CROSS,
         lambda: _fixed_bytes(p, cross=True), "a sequence's fixed bytes: its recurrent state and"
         " the cross-attention cache of a fixed-length encoder (2 B an element)"),
        (r"^facts_state_bytes$", lambda: _fixed_bytes(p, cross=False),
         "a sequence's recurrent state (2 B an element)"),
        (r"^(state_gb|report_state_gb)$", lambda: _scaled(_fixed_bytes(p, cross=False), GB),
         "a sequence's recurrent state (GB, 2 B an element)"),
        (r"^eng_\w+_\d+_state_mb$", lambda: _scaled(_fixed_bytes(p, cross=False), MB),
         "a sequence's recurrent state (MB, 2 B an element)"),
        (r"^(report_params|eng_\w+_params|counter_params|traingenz_params)$", lambda: params,
         "the model's parameters"),
        # TrainingMemoryCalculator, full fine-tuning in bf16 with AdamW: its own accounting -- bf16
        # weights and their fp32 master copy (6 B a parameter), fp32 gradients (4 B), AdamW's two
        # fp32 moments (8 B); the total's activations are above it
        (r"^trainmem_weight_gb$", lambda: None if params is None else 6 * params / 1e9,
         "the weights the trainer holds (bf16 and an fp32 master copy: 6 B a parameter, GB)"),
        (r"^trainmem_total_gb$", lambda: None if params is None else 18 * params / 1e9,
         "weights, master copy, gradients and AdamW state (18 B a parameter, GB; activations"
         " on top)"),
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
            if re.match(STATE_WITH_CROSS, reader) and p.get("cross_kv"):
                what += (f"; {p['cross_kv']:,} cross-attention K+V elements an encoder token, "
                         f"{p.get('encoder_tokens'):,} encoder tokens ("
                         f"{p.get('encoder_tokens_source', 'stated by hand')})")
            return v, f"{what}: {src}"
    return None, f"no ground-truth rule for {reader}"


#: GenZ's graph weights, held against the matrices with this relative slack.
GRAPH_WEIGHTS = r"^(genz_(prefill|decode)_weights_mb|conv_prefill_weights_mb|runtime_\w+\[1\])$"
GRAPH_SLACK = 5e-5

#: The memory figures a capacity figure is sized from, per capacity figure.
HELD_BY = {
    "vllm": ["facts_weight_bytes", "facts_kv_bytes_per_token", "facts_kv_bytes_4096"],
    "serving_requests": ["serving_bytes_per_token"],
    # B_eff = (device - weights) // (KV a token x context + a request's fixed bytes)
    "b_eff": ["serving_weight_bytes", "serving_bytes_per_token", "serving_state_bytes"],
}


def _held_by(reader):
    if reader.startswith("serving_requests"):
        return HELD_BY["serving_requests"]
    if re.match(r"^(cluster|disagg)_b_eff_\d+$", reader):
        return HELD_BY["b_eff"]
    return HELD_BY["vllm"]


def allowlist_from(main_rows, branch_rows, configs, manual, causes):
    """Allowlist entries for every unsafe move between two corpus runs -- and every figure one
    run produced and the other did not (``kv_corpus.lost_and_new``) -- each with its ground
    truth (``truth_profile`` of the row's config, or a hand-written profile in ``manual``), a lost
    figure with the error that replaced it, and the truth-only entries the capacity moves are
    held by."""
    import re

    import kv_corpus

    moves = [m + ("moved",) for m in kv_corpus.unsafe_moves(main_rows, branch_rows)]
    moves += kv_corpus.lost_and_new(main_rows, branch_rows)
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
    for name, reader, x, y, kind, change in moves:
        e = {"config": name, "reader": reader, "main": x, "branch": y, "kind": kind,
             "cause": _cause(causes, name, reader) or derived_cause(main.get(name, {}),
                                                                   branch.get(name, {}), reader)}
        if change != "moved":
            e["change"] = change
        if change == "lost":
            # the branch's reader raises where origin/main's gave a figure: the entry names the
            # error (its first words, before any figure in it) and, for a weight or KV figure,
            # the truth the error is read against
            err = kv_corpus._error_of(branch.get(name, {}), reader) or ""
            e["error"] = re.split(r"\d", err, maxsplit=1)[0][:80]
            if kind == "memory":
                e["truth"], e["source"] = figure_truth(profile(name), reader)
            if not _cause(causes, name, reader):
                e["cause"] = f"the branch's reader raises: {err[:200]}"
                if isinstance(e.get("truth"), (int, float)):
                    e["cause"] += (f"; the figure it replaced is held to {e['truth']:,.3f} "
                                   f"({e['source']}), origin/main's {x:,.3f}")
            entries[(name, reader)] = e
            continue
        if change == "new" and not _cause(causes, name, reader):
            errs = sorted(k for k, v in main.get(name, {}).items() if k.endswith("error") and v)
            e["cause"] = ("origin/main gave no figure" + (" (its readers raised: " + ", ".join(
                errs) + ")" if errs else "") + "; the branch's is held to its ground truth")
        if kind == "memory":
            e["truth"], e["source"] = figure_truth(profile(name), reader)
            unstated = profile(name).get("unstated") or {}
            if unstated and re.match(unstated.get("readers", "$^"), reader):
                # modules the checkpoint holds that no config key describes: the figure may sit
                # that many parameters (at 2 B) below the truth, named in the entry
                e["unstated"] = 2 * int(unstated["params"])
                e["unstated_source"] = unstated["source"]
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


ALLOWLIST_ABOUT = (
    "Every figure the branch moves in the unsafe direction against origin/main (a memory figure "
    "down, a capacity figure up), every figure the branch reports that origin/main did not "
    "(change: new) and every figure origin/main reported that the branch does not (change: lost, "
    "with the error the branch now records), over the whole corpus (committed rows -- the GGUF "
    "fixtures among them -- and this machine's HF cache, model folders and GGUFs), each with both "
    "figures, its ground truth and the truth's source, and its cause; and the memory figures each "
    "capacity figure is held by. Written by scripts/kv_truth.py --allowlist-from from two "
    "kv_corpus.py runs; read by kv_corpus.py --gate and tests/test_kv_gate.py.")


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
    ap.add_argument("--checkpoint", nargs="+", metavar="REPO[@REV]",
                    help="a hub checkpoint's tensor counts, from its safetensors headers")
    args = ap.parse_args()
    if args.checkpoint:
        for spec in args.checkpoint:
            repo, _, rev = spec.partition("@")
            json.dump(checkpoint_profile(repo, rev or "main"), sys.stdout)
            print()
        return
    if args.allowlist_from:
        manual = json.load(open(args.manual)) if args.manual else {}
        causes = json.load(open(args.causes)) if args.causes else []
        entries = allowlist_from(json.load(open(args.allowlist_from[0])),
                                 json.load(open(args.allowlist_from[1])),
                                 corpus_configs(args), manual.get("profiles", {}), causes)
        json.dump({"about": ALLOWLIST_ABOUT, "entries": entries},
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
