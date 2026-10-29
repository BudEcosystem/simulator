"""Parameters of the stacks around a decoder: an encoder-decoder's encoder, and the adapter stacks
a multimodal model puts between its tower and its language model (BLIP-2's Q-Former).

Read from the stack's own geometry, as ``layer_plan.encoder_view`` and ``layer_plan.encoder_stages``
give it -- never from the decoder's: Donut's Swin encoder runs 20 blocks over four widths
(128 to 1,024), and was sized as 4 blocks at the decoder's 1,024.

Every count is of the matrices and the vectors beside them (biases, norms, a windowed tower's
relative-position tables) as transformers' modelling builds them (5.12), so a reader that adds
these to a decoder's weights holds at least what the checkpoint holds.
"""

from typing import Any, Dict, Optional

from .layer_plan import (
    TOWER_CLASS_KEYS,
    _first,
    _pos_int,
    _text_config,
    attention_head_dim,
    config_class_default,
    encoder_view,
    ffn_activation,
    gated_ffn_by_family,
    nested_config_class,
    stated_gated_ffn,
    with_class_geometry,
)

#: Gated activations (three FFN matrices: gate, up, down) by name.
_GATED_ACTIVATIONS = ("silu", "swiglu", "geglu", "glu", "swish")


def _gated(view: Dict[str, Any]) -> bool:
    stated = stated_gated_ffn(view)
    if stated is not None:
        return stated
    if gated_ffn_by_family(view):
        return True
    act = ffn_activation(view)
    return any(a in act for a in _GATED_ACTIVATIONS)


#: Spellings a tower or stack states its FFN width under.
_FFN_WIDTH_KEYS = ("intermediate_size", "d_ff", "ffn_dim", "encoder_ffn_dim", "mlp_dim")


def tower_view(block: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """A tower's or stack's block (a `vision_config`, an `audio_config`) under the names
    ``stack_params`` reads: its width, depth, heads, head width (`d_kv`) and FFN width under
    any of their spellings, and its stages when it is hierarchical. None for anything else."""
    from .layer_plan import ATTENTION_HEAD_KEYS, ENCODER_HEAD_KEYS, _block_depth, _first, encoder_stages

    if not isinstance(block, dict):
        return None
    # a role the block leaves out is its config class's (Blip2VisionConfig's 39 x 1,408)
    block = with_class_geometry(block)
    view = {k: v for k, v in block.items() if not isinstance(v, dict)}
    width = _pos_int(_first(block, "hidden_size", "d_model", "embed_dim", "n_embd"))
    if width:
        view["hidden_size"] = width
    view["num_hidden_layers"] = _block_depth(block)
    # a Whisper-shaped audio tower (Qwen2-Audio's, MOSS-Transcribe's `audio_config`) states its
    # heads as an encoder's
    heads = (_pos_int(_first(block, *ATTENTION_HEAD_KEYS)) or _pos_int(block.get("num_heads"))
             or _pos_int(_first(block, *ENCODER_HEAD_KEYS)))
    if heads:
        view["num_attention_heads"] = heads
    ffn = _pos_int(_first(block, *_FFN_WIDTH_KEYS))
    if ffn:
        view["intermediate_size"] = ffn
    if _pos_int(block.get("d_kv")) and not _pos_int(block.get("head_dim")):
        view["head_dim"] = int(block["d_kv"])
    stages = encoder_stages(block)
    if stages:
        view["stages"] = stages
    return view


def _block(h: int, heads: int, kv_heads: int, head_dim: int, ffn: int, gated: bool) -> int:
    """One transformer block: Q / K / V / O with their biases, the FFN with its biases, two
    norms (weight and bias)."""
    q = h * heads * head_dim + heads * head_dim
    kv = 2 * (h * kv_heads * head_dim + kv_heads * head_dim)
    o = heads * head_dim * h + h
    mlp = (3 if gated else 2) * h * ffn + ffn + h
    return q + kv + o + mlp + 4 * h


def stack_params(view: Optional[Dict[str, Any]]) -> int:
    """The parameters of one stack, given under the decoder-only names (``encoder_view``): a
    uniform stack's blocks, or a hierarchical tower's stages (``stages``) with the 2 x 2 patch
    merging between them, its patch embedding and its windows' relative-position tables. 0 for
    None."""
    if not isinstance(view, dict):
        return 0
    gated = _gated(view)
    stages = view.get("stages")
    if stages:
        total = 0
        window = _pos_int(view.get("window_size"))
        for i, st in enumerate(stages):
            c, heads = st["hidden_size"], st["num_attention_heads"]
            hd = max(c // heads, 1)
            total += st["layers"] * _block(c, heads, heads, hd, st["intermediate_size"], gated)
            if window:
                total += st["layers"] * (2 * window - 1) ** 2 * heads
        total += stage_embedding_params(view)
        return int(total)
    layers = _pos_int(view.get("num_hidden_layers")) or 0
    h = _pos_int(view.get("hidden_size")) or 0
    heads = _pos_int(view.get("num_attention_heads")) or 0
    if not (layers and h and heads):
        return 0
    kv_heads = _pos_int(view.get("num_key_value_heads")) or heads
    hd = attention_head_dim(view, heads, h) or max(h // heads, 1)
    ffn = _pos_int(view.get("intermediate_size")) or 4 * h
    per_layer = _block(h, heads, kv_heads, hd, ffn, gated)
    kernel = _pos_int(view.get("conv_kernel_size"))
    if kernel:
        # a conformer block (its stack states a convolution module: Parakeet / FastConformer,
        # Wav2Vec2-Conformer): a second, macaron FFN, and the convolution module -- a pointwise
        # projection to two widths (GLU), a depthwise convolution, a pointwise projection back,
        # their biases and a norm. Counted as a plain block, cohere-transcribe's 48-layer encoder
        # was half its 1.89B.
        per_layer += (3 if gated else 2) * h * ffn + ffn + h + 2 * h
        per_layer += h * 2 * h + 2 * h + h * kernel + h + h * h + h + 4 * h
    return int(layers * per_layer + 2 * h)


def _block_features(h: int, heads: int, kv_heads: int, head_dim: int, ffn: int, gated: bool) -> int:
    """One transformer block's linear maps (``_block``'s matrices), as the sum of each map's
    (in + out) features: Q / K / V / O and the FFN's."""
    attn = 4 * h + 2 * heads * head_dim + 2 * kv_heads * head_dim
    return attn + (3 if gated else 2) * (h + ffn)


def stack_linear_features(view: Optional[Dict[str, Any]]) -> int:
    """The sum of (in + out) over every linear map of one stack's blocks (``stack_params``'
    geometry: a uniform stack, a hierarchical tower's stages and the 2 x 2 merging between them, a
    conformer block's second FFN and pointwise convolutions), what an engine that wraps every
    linear layer with LoRA (vLLM) allocates A and B buffers over. 0 for None."""
    if not isinstance(view, dict):
        return 0
    gated = _gated(view)
    stages = view.get("stages")
    if stages:
        total = 0
        for i, st in enumerate(stages):
            c, heads = st["hidden_size"], st["num_attention_heads"]
            hd = max(c // heads, 1)
            total += st["layers"] * _block_features(c, heads, heads, hd, st["intermediate_size"],
                                                    gated)
            if i + 1 < len(stages):
                total += 4 * c + 2 * c  # the 2 x 2 merge: 4c -> 2c
        return int(total)
    layers = _pos_int(view.get("num_hidden_layers")) or 0
    h = _pos_int(view.get("hidden_size")) or 0
    heads = _pos_int(view.get("num_attention_heads")) or 0
    if not (layers and h and heads):
        return 0
    kv_heads = _pos_int(view.get("num_key_value_heads")) or heads
    hd = attention_head_dim(view, heads, h) or max(h // heads, 1)
    ffn = _pos_int(view.get("intermediate_size")) or 4 * h
    per_layer = _block_features(h, heads, kv_heads, hd, ffn, gated)
    if _pos_int(view.get("conv_kernel_size")):
        # a conformer block: the macaron FFN and the two pointwise projections (h -> 2h, h -> h)
        per_layer += (3 if gated else 2) * (h + ffn) + 3 * h + 2 * h
    return int(layers * per_layer)


def encoder_linear_features(config: Dict[str, Any]) -> int:
    """``stack_linear_features`` of an encoder-decoder's encoder (``layer_plan.encoder_view``), 0
    for any other config."""
    return stack_linear_features(encoder_view(config)) if isinstance(config, dict) else 0


def stage_embedding_params(view: Dict[str, Any]) -> int:
    """How a hierarchical tower's stages are entered: a tower that states a patch size per stage
    (DaViT / Florence-2: `patch_size` [7, 3, 3, 3]) embeds each stage with its own convolution
    from the previous stage's width (the image's channels for the first) and a norm; any other
    (Swin) embeds the image once (`patch_size`, default 4) and merges 2 x 2 neighbours between
    stages (Linear(4c, 2c) and LayerNorm(4c)). Counted as Swin's merges, Florence-2's
    convolutions were 3.5M short."""
    stages = view.get("stages") or []
    if not stages:
        return 0
    channels = _pos_int(view.get("num_channels")) or _pos_int(view.get("in_channels")) or 3
    patch = view.get("patch_size")
    if isinstance(patch, (list, tuple)) and len(patch) == len(stages) and all(
            _pos_int(p) for p in patch):
        total, prev = 0, channels
        for p, st in zip(patch, stages):
            c = st["hidden_size"]
            total += int(p) ** 2 * prev * c + c + 2 * max(prev, c)  # conv, bias, a norm
            prev = c
        return int(total)
    total = 0
    for i, st in enumerate(stages[:-1]):
        c = st["hidden_size"]
        total += 4 * c * 2 * c + 2 * 4 * c  # PatchMerging: Linear(4c, 2c), LayerNorm(4c)
    c0 = stages[0]["hidden_size"]
    patch = patch if _pos_int(patch) else 4
    total += channels * patch * patch * c0 + c0 + 2 * c0  # patch embedding and its norm
    return int(total)


def tower_embedding_params(view: Optional[Dict[str, Any]]) -> int:
    """What a tower holds before its first block, from its own keys (transformers 5.12): a patch
    encoder whose length is its `seq_len` (Pix2Struct: a linear patch projection from
    `patch_embed_hidden_size`, and row and column position tables of `seq_len` each); a ViT's
    convolutional patch embedding with its class token and a position per patch; an audio
    encoder's two convolutions over `num_mel_bins` and a position per frame
    (`max_source_positions`: Whisper). 0 for a text encoder (its token table is counted as the
    embedding) and for a hierarchical tower (``stack_params`` counts its patch embedding)."""
    if not isinstance(view, dict):
        return 0
    stages = view.get("stages")
    if stages:
        # a hierarchical tower: its patch embedding and its windows' relative-position tables
        # (the blocks and the merges between stages are graphed as operators)
        c0 = stages[0]["hidden_size"]
        patch = _pos_int(view.get("patch_size")) or 4
        channels = _pos_int(view.get("num_channels")) or 3
        window = _pos_int(view.get("window_size")) or 0
        tables = sum(st["layers"] * (2 * window - 1) ** 2 * st["num_attention_heads"]
                     for st in stages) if window else 0
        return channels * patch * patch * c0 + c0 + tables
    h = _pos_int(view.get("hidden_size")) or 0
    if not h:
        return 0
    seq_len = _pos_int(view.get("seq_len"))
    if seq_len:
        patch_in = _pos_int(view.get("patch_embed_hidden_size")) or h
        return 2 * seq_len * h + patch_in * h + h
    mels = _pos_int(view.get("num_mel_bins"))
    if mels:
        frames = _pos_int(view.get("max_source_positions")) or 0
        return mels * h * 3 + h + h * h * 3 + h + frames * h
    size, patch = view.get("image_size"), view.get("patch_size")
    if _pos_int(size) and _pos_int(patch):
        channels = _pos_int(view.get("num_channels")) or 3
        patches = (int(size) // int(patch)) ** 2
        return channels * int(patch) ** 2 * h + h + (patches + 1) * h + h
    return 0


def encoder_params(config: Dict[str, Any]) -> int:
    """An encoder-decoder's encoder stack (``layer_plan.encoder_view``), 0 for any other config."""
    return stack_params(encoder_view(config)) if isinstance(config, dict) else 0


#: Adapter stacks between a tower and the language model, by the key their config sits under:
#: BLIP-2's and InstructBLIP's Q-Former (`qformer_config`), a BERT-like stack whose every
#: `cross_attention_frequency`-th layer cross-attends to the tower (`encoder_hidden_size` wide),
#: fed `num_query_tokens` learned queries and projected to the language model's width.
ADAPTER_KEYS = ("qformer_config",)

#: Q-Former classes that read the prompt's text beside the queries: transformers 5.12's
#: InstructBlipQFormerModel (and the video variant) embeds text tokens (`word_embeddings` at its
#: `vocab_size`, `position_embeddings` at its `max_position_embeddings`, a norm) and every layer
#: runs a text FFN (`intermediate`, `output`) beside the query FFN; Blip2QFormerModel builds
#: neither. Left out, the hub's InstructBLIP flan-T5-xl was 80.5M parameters short.
CLASS_TEXT_QFORMER = frozenset({"instructblip_qformer", "instructblipvideo_qformer"})


def adapter_params(config: Dict[str, Any]) -> int:
    """The adapter stacks a config states (``ADAPTER_KEYS``), 0 when it states none. BLIP-2's
    Q-Former and its projection are 106.7M parameters the FRD-027 facts left out."""
    if not isinstance(config, dict):
        return 0
    # a Q-Former that states only its model type is its class's (Blip2QFormerConfig: 12 x 768)
    config = with_class_geometry(config)
    total = 0
    text_width = _pos_int(_first_width(_text_config(config)))
    for key in ADAPTER_KEYS:
        block = config.get(key)
        if not isinstance(block, dict):
            continue
        layers = _pos_int(block.get("num_hidden_layers")) or 0
        h = _pos_int(block.get("hidden_size")) or 0
        if not (layers and h):
            continue
        total += stack_params(block)
        every = _pos_int(block.get("cross_attention_frequency")) or 1
        enc_w = _pos_int(block.get("encoder_hidden_size")) or h
        cross_layers = -(-layers // every)
        # each cross-attention: Q and O at its own width (with biases), K and V from the tower's,
        # and its output norm
        total += cross_layers * (2 * (h * h + h) + 2 * (enc_w * h + h) + 2 * h)
        queries = _pos_int(config.get("num_query_tokens")) or 0
        total += queries * h + 2 * h  # the learned queries and the stack's input norm
        qformer_class = nested_config_class(config.get("model_type"), key) or block.get(
            "model_type")
        if qformer_class in CLASS_TEXT_QFORMER:
            ffn = _pos_int(block.get("intermediate_size")) or 4 * h
            vocab = _pos_int(block.get("vocab_size")) or 0
            positions = _pos_int(block.get("max_position_embeddings")) or 0
            total += layers * (2 * h * ffn + ffn + h + 2 * h)  # the text FFN and its norm
            total += (vocab + positions) * h + 2 * h  # the text embeddings and their norm
        if text_width:
            total += h * text_width + text_width  # the projection to the language model
    return int(total + resampler_params(config))


#: Perceiver resamplers between a tower and the language model, by the key their config sits
#: under (Idefics2's `perceiver_config`): `resampler_depth` layers in which `resampler_n_latents`
#: learned latents attend over the tower's projected output and themselves.
RESAMPLER_KEYS = ("perceiver_config",)


def resampler_params(config: Dict[str, Any]) -> int:
    """The perceiver resamplers a config states (``RESAMPLER_KEYS``), 0 when it states none, read
    from the block's own keys, else its class's (``with_class_geometry``: the hub's Idefics2
    `perceiver_config` states only a model type): per layer the latents' attention --
    `resampler_n_heads` x `resampler_head_dim` queries, `num_key_value_heads` K / V heads -- at
    the block's width, a gated MLP four times as wide and three norms; the latents and a final
    norm; and the projection that feeds it from the tower's width, a gated MLP at the language
    model's FFN width (transformers 5.12 Idefics2Connector's `modality_projection`). Left out, the
    hub's Idefics2-8B was 743.2M parameters short."""
    if not isinstance(config, dict):
        return 0
    config = with_class_geometry(config)
    text = _text_config(config)
    text_width = _pos_int(_first_width(text)) or 0
    total = 0
    for key in RESAMPLER_KEYS:
        block = config.get(key)
        if not isinstance(block, dict):
            continue
        layers = _pos_int(block.get("resampler_depth")) or 0
        h = _pos_int(block.get("hidden_size")) or text_width
        heads = _pos_int(block.get("resampler_n_heads")) or 0
        if not (layers and h and heads):
            continue
        hd = _pos_int(block.get("resampler_head_dim")) or max(h // heads, 1)
        kv = _pos_int(block.get("num_key_value_heads")) or heads
        attention = 2 * h * heads * hd + 2 * h * kv * hd
        total += layers * (attention + 3 * h * 4 * h + 3 * h)
        total += (_pos_int(block.get("resampler_n_latents")) or 0) * h + h
        vision = tower_view(config.get("vision_config"))
        vh = _pos_int((vision or {}).get("hidden_size")) or h
        ffn = _pos_int(_first(text, *_FFN_WIDTH_KEYS)) or 4 * h
        total += 2 * vh * ffn + ffn * h
    return int(total)


def _first_width(tc: Dict[str, Any]) -> Optional[int]:
    for k in ("hidden_size", "d_model", "n_embd"):
        v = _pos_int(tc.get(k)) if isinstance(tc, dict) else None
        if v:
            return v
    return None


#: An audio encoder beside a decoder-only language model, by the key its config sits under
#: (transformers 5.12: Qwen2-Audio's, Voxtral's and Voxtral-Realtime's `audio_config`; a hub
#: model's `audio_encoder_config`). Its blocks are read at their own geometry
#: (``stack_params.tower_view``), as an encoder-decoder's encoder is.
AUDIO_TOWER_KEYS = ("audio_config", "audio_encoder_config")
#: The frames an audio projector concatenates into one language-model position, by spelling
#: (Voxtral's `downsample_factor`, MOSS-Transcribe's `audio_merge_size`), or the projector's
#: stated input width (`adaptor_input_dim`).
_AUDIO_MERGE_KEYS = ("downsample_factor", "audio_merge_size")


def audio_tower_params(config: Dict[str, Any]) -> int:
    """The audio encoder a decoder-only language model holds beside it, and the projector to the
    language model: its blocks (``stack_params``), its convolutional front end and frame
    positions (``tower_embedding_params``), and the projector from the tower's width times the
    frames it merges (else its stated input width) to the language model's, then the language
    model's width to itself -- two matrices, as Voxtral's (`linear_1`, act, `linear_2`) and
    MOSS-Transcribe's `vq_adaptor` build it; Qwen2-Audio's single projection is over-counted by
    the second. No config states the projector's depth, and one matrix left MOSS-Transcribe 1.05M
    below its checkpoint. 0 for a config with no such tower, or a tower that states no stack (a
    discrete audio tokenizer)."""
    if not isinstance(config, dict):
        return 0
    config = with_class_geometry(config)
    total = 0
    out_h = int(_first_width(_text_config(config)) or 0)
    for key in AUDIO_TOWER_KEYS:
        view = tower_view(config.get(key))
        blocks = stack_params(view)
        if not blocks:
            continue
        total += blocks + tower_embedding_params(view)
        if out_h:
            vh = int(view["hidden_size"])
            merge = int(_pos_int(_first(config, *_AUDIO_MERGE_KEYS)) or 1)
            proj_in = int(_pos_int(config.get("adaptor_input_dim")) or vh * merge)
            total += proj_in * out_h + out_h + out_h * out_h + out_h
    return int(total)


def _get(block: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = block.get(key)
        if value is not None:
            return value
    return default


def vision_tower_params(config: Dict[str, Any]) -> int:
    """A config's image tower (`vision_config`) and its projector to the language model, 0 for a
    config with none: a hierarchical tower's stages (Florence-2's DaViT), else a ViT's blocks
    (and a tiled tower's extra global layers and positions: Mllama), its patch embedding, its
    positions, its norms and the projector (``_projector_params``). The one count the FRD-027
    facts and the parameter counter take."""
    if not isinstance(config, dict):
        return 0
    config = with_class_geometry(config)
    vision = config.get("vision_config")
    if not isinstance(vision, dict):
        return 0
    # what the tower leaves out is its config class's default (Mllama's tower builds 8 global
    # layers and a 7680-wide output whether or not the config says so)
    mt = vision.get("model_type")
    vision = {**{k: config_class_default(mt, k) for k in TOWER_CLASS_KEYS
                 if config_class_default(mt, k) is not None}, **dict(vision)}
    out_text = _first_width(_text_config(config))
    view = tower_view(vision)
    if view and view.get("stages"):
        # a hierarchical tower (Florence-2's DaViT): its stages at their own widths
        # (stack_params), and a projector from its last stage's width to the language model's
        vh = int(view["stages"][-1]["hidden_size"])
        return int(stack_params(view) + _projector_params(vh, int(out_text or vh)))
    vh = _get(vision, "hidden_size", "embed_dim", default=0)
    vh = int(vh) if isinstance(vh, (int, float)) and not isinstance(vh, bool) else 0
    if not vh:
        return 0
    # a tower's stated extra layers count too (Mllama's `num_global_layers` after its 32)
    depth = int(_get(vision, "depth", "num_hidden_layers", default=24)) + int(
        _get(vision, "num_global_layers", default=0) or 0)
    vi = int(_get(vision, "intermediate_size", "mlp_dim", default=4 * vh))
    patch = int(_get(vision, "patch_size", default=14))
    temporal = int(_get(vision, "temporal_patch_size", default=1))
    channels = int(_get(vision, "in_channels", "num_channels", default=3))
    merge = int(_get(vision, "spatial_merge_size", default=1))
    out_h = int(
        _get(vision, "out_hidden_size", default=_first_width(_text_config(config)) or vh)
    )
    pos = int(_get(vision, "num_position_embeddings", default=0) or 0)
    tiles = int(_get(vision, "max_num_tiles", default=0) or 0)
    image = _get(vision, "image_size", default=None)
    if not pos and tiles and isinstance(image, int) and patch:
        # a tiled tower (Mllama): a position per patch and class token, and per supported aspect
        # ratio a position table over every tile's patches, and a pre- and post-tile embedding
        per_tile = (image // patch) ** 2 + 1
        ratios = len(vision.get("supported_aspect_ratios") or []) + 1
        pos = per_tile + ratios * tiles * per_tile + 2 * ratios * tiles
    elif not pos and merge == 1 and isinstance(image, int) and isinstance(patch, int) and patch:
        # a ViT tower (CLIP, SigLIP: no merge) learns a position per patch and its class token's
        pos = (image // patch) ** 2 + 1
    block = 4 * vh * vh + 4 * vh + 2 * vh * vi + vi + vh + 4 * vh
    proj_in = int(_get(vision, "vision_output_dim", default=0) or 0)
    if proj_in:
        # the projector from the tower's stated output width (Mllama concatenates intermediate
        # layers: 7680) to the language model's
        merger = proj_in * out_h + out_h
    else:
        merger = _projector_params(vh * merge * merge, out_h) + 2 * vh
    # the blocks, the patch embedding, the positions, the projector, and the tower's own norms
    # before its first block and after its last (CLIP's pre- and post-layernorm)
    if _pos_int(vision.get("seq_len")):
        # a patch encoder whose length is its `seq_len` (Pix2Struct): a linear patch projection
        # and row and column positions (tower_embedding_params), no convolution
        embed = tower_embedding_params(tower_view(vision))
    else:
        embed = channels * temporal * patch * patch * vh + vh + pos * vh
    return int(depth * block + embed + merger + 4 * vh)


def _projector_params(merged: int, out_h: int) -> int:
    """The projector from a tower's (merged) output width to the language model's: two matrices,
    `merged` -> hidden -> `out_h`, with their biases. No config states the hidden width: Qwen-VL's
    merger keeps the merged width (5,120 -> 5,120 -> 3,584), LLaVA's projector the language
    model's (1,024 -> 4,096 -> 4,096), so the wider of the two is read -- the reading that is
    never below either. Read as the merged width alone, LLaVA-1.5's projector was 16.3M short."""
    hidden = max(merged, out_h)
    return int(merged * hidden + hidden + hidden * out_h + out_h)
