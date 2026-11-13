"""What a decision-only checkpoint is made of, read from its family's files (Bud spec 031 D-45, FRD §5.16.3).

Bud-Decision-Runtime serves a handful of model *families*. Each family keeps its backbone's configuration in its own
place and adds a head of its own; the tables below are copied from the runtime's loaders
(``engine/python/basal_engine/models/*.py``) and must change with them (``tests/test_decision_runtime.py`` compares the
family list with the runtime's ``FAMILIES``).

The facts are counted from the configuration alone, so budsim can size a checkpoint from its small JSON files without
downloading the weights. The counts match the checkpoints' safetensors headers within 1% (TC-DO-07).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

#: How a call's questions become rows:
#: - ``per_question``: one row per question, each holding the state and that question (Laya, Julia);
#: - ``one_sequence``: every question of a call and the state in one row (GLiNER);
#: - ``branch``: the state once, each question continuing it (Kev).
PER_QUESTION, ONE_SEQUENCE, BRANCH = "per_question", "one_sequence", "branch"


@dataclass(frozen=True)
class Family:
    name: str
    kind: str                       # "encoder" | "decoder": the calibration's fallback group
    layout: str                     # how a call becomes rows (above)
    backbone_config: str            # where the backbone's config.json lives, relative to the checkpoint
    family_config: Optional[str]    # the family's own config file, if any
    required_files: Tuple[str, ...]  # what makes a checkpoint this family's (budmodel's detection reads the same list)
    keeps_state: bool = False       # states kept between calls (needs the runtime's tenant salt, D-40)


FAMILIES: Dict[str, Family] = {
    "laya": Family(
        "laya", "encoder", PER_QUESTION, "encoder/config.json", "rl_agent_config.json",
        ("rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json", "model.safetensors"),
    ),
    "julia": Family(
        "julia", "encoder", PER_QUESTION, "encoder/config.json", "julia_config.json",
        ("julia_config.json", "encoder/config.json", "tokenizer/tokenizer.json", "model.safetensors"),
    ),
    "gliner": Family(
        "gliner", "encoder", ONE_SEQUENCE, "encoder_config/config.json", "config.json",
        ("config.json", "encoder_config/config.json", "tokenizer.json", "model.safetensors"),
    ),
    "kev": Family(
        "kev", "decoder", BRANCH, "config.json", "adapter_config.json",
        ("adapter_config.json", "adapter_model.safetensors", "head.pt", "tokenizer.json"),
        keeps_state=True,
    ),
}

#: Families the runtime serves that Bud does not size: ``instant`` is the runtime's built-in test model.
NOT_SIZED = ("instant",)

#: GLiNER's span, count and label heads, measured on GLiNER2.5-Decide's checkpoint as multiples of hidden² (52.5M at
#: hidden 1024). Its config names the head's kind but not its widths.
_GLINER_HEAD_PER_HIDDEN2 = 50.1
#: GLiNER refuses a request whose questions and state together pass this (basal_engine.models.gliner.MAX_LEN).
_GLINER_MAX_LEN = 4096
#: Laya's default row length (rl_agent_config.json ``max_len``) when the file names none.
_LAYA_MAX_LEN = 512
#: Kev's longest state and the room a question has beyond it (basal_engine.models.kev).
_KEV_MAX_STATE, _KEV_QUESTION_TOKENS = 65536, 8192


@dataclass(frozen=True)
class DecisionFacts:
    """One decision checkpoint, as the sizing needs it."""

    family: str
    kind: str
    layout: str
    backbone: str                 # the backbone config's model_type
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    full_layers: int              # layers attending over the whole row
    window_layers: int            # layers attending over a local window
    window: Optional[int]         # the window's width in tokens (both sides together)
    relative_buckets: int         # DeBERTa's relative-position buckets (its extra c2p / p2c scores); 0 otherwise
    max_len: int                  # the longest row the family serves, in tokens
    embedding_params: int
    layer_params: int             # every backbone layer together
    head_params: int
    lora_params: int = 0
    bidirectional: bool = True
    glu: bool = False             # the MLP's input projection is gated (two halves of 2 x intermediate)
    keeps_state: bool = False

    @property
    def params_total(self) -> int:
        return int(self.embedding_params + self.layer_params + self.head_params + self.lora_params)

    @property
    def linear_params(self) -> int:
        """Parameters every token is multiplied through (the embedding table is a gather)."""
        return int(self.layer_params + self.head_params + self.lora_params)

    def linear_flops_per_token(self) -> float:
        return 2.0 * self.linear_params

    def attention_flops_per_token(self, row: int, context: Optional[float] = None) -> float:
        """Score and value FLOPs of one token of a ``row``-token row.

        An encoder token attends over the whole row (``bidirectional``); a window layer over at most its window. A
        decoder token attends over what precedes it: ``context`` tokens (the mean, row / 2, when not given).
        DeBERTa adds its two relative-position scores (content-to-position and position-to-content), each over at
        most 2 x buckets relative distances.
        """
        width = self.num_heads * self.head_dim
        if self.bidirectional:
            ctx = float(max(int(row), 1))
        else:
            ctx = float(context) if context is not None else max(int(row), 1) / 2.0
        window = float(self.window or ctx)
        attn = 4.0 * width * (self.full_layers * ctx + self.window_layers * min(ctx, window))
        if self.relative_buckets:
            attn += 4.0 * width * self.num_layers * min(ctx, 2.0 * self.relative_buckets)
        return attn

    def flops_per_token(self, row: int, context: Optional[float] = None) -> float:
        return self.linear_flops_per_token() + self.attention_flops_per_token(row, context)

    def activation_bytes_per_token(self, compute_bytes: float) -> float:
        """Live activation bytes per token of a pass, from the backbones' forward (``basal_engine.backbones``).

        The residual stream is float32 between layers; inside a layer the projections are in the compute
        precision: q, k, v and the attention output, the MLP's input (two halves when gated) and its activation.
        DeBERTa also keeps its two relative-position score tensors, ``heads x 2 x buckets`` float32 each. Attention
        itself is fused (never a row x row matrix).
        """
        h, i = self.hidden_size, self.intermediate_size
        mlp = (2 * i + i) if self.glu else (i + i)
        per = 4.0 * 2 * h + compute_bytes * (3 * h + h + mlp)
        if self.relative_buckets:
            per += 2 * 4.0 * self.num_heads * 2 * self.relative_buckets
        return per


class FamilyError(ValueError):
    """The checkpoint is not one the family's loader would open."""


def _read(root: str, name: str) -> Dict[str, Any]:
    path = os.path.join(root, name)
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise FamilyError(f"{name} is missing: the checkpoint is not a complete {os.path.basename(root)} checkpoint") from exc
    except (OSError, ValueError) as exc:
        raise FamilyError(f"{name} cannot be read: {exc}") from exc
    if not isinstance(data, dict):
        raise FamilyError(f"{name} is not a JSON object")
    return data


def facts(checkpoint_dir: str, family: str, *, base_dir: Optional[str] = None) -> DecisionFacts:
    """Read a checkpoint's facts from its files. Kev also needs ``base_dir``, its base model's directory."""
    fam = _family(family)
    family_config = _read(checkpoint_dir, fam.family_config) if fam.family_config else {}
    if fam.layout == BRANCH:
        if not base_dir:
            raise FamilyError("a Kev checkpoint is sized with its base model's config.json (base_dir)")
        backbone = _read(base_dir, fam.backbone_config)
    else:
        backbone = _read(checkpoint_dir, fam.backbone_config)
    return facts_from_configs(family, backbone, family_config)


def facts_from_configs(
    family: str, backbone: Mapping[str, Any], family_config: Optional[Mapping[str, Any]] = None
) -> DecisionFacts:
    """The facts from configs already read: ``backbone`` is the backbone's config.json, ``family_config`` the
    family's own file (``rl_agent_config.json``, ``julia_config.json``, GLiNER's ``config.json``, Kev's
    ``adapter_config.json``)."""
    fam = _family(family)
    cfg = dict(backbone or {})
    own = dict(family_config or {})
    model_type = str(cfg.get("model_type") or "").lower()
    try:
        h = int(cfg["hidden_size"])
        i = int(cfg["intermediate_size"])
        layers = int(cfg["num_hidden_layers"])
        heads = int(cfg["num_attention_heads"])
        vocab = int(cfg["vocab_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FamilyError(f"the {fam.name} backbone config lacks {exc}") from exc
    if min(h, i, layers, heads, vocab) <= 0:
        raise FamilyError(f"the {fam.name} backbone config has a non-positive size")
    head_dim = int(cfg.get("head_dim") or h // heads)

    if model_type == "modernbert":
        every = max(int(cfg.get("global_attn_every_n_layers") or 1), 1)
        full = sum(1 for n in range(layers) if n % every == 0)
        window = int(cfg.get("local_attention") or 0) or None
        layer = 4 * h * h + 3 * h * i
        kw = dict(full_layers=full, window_layers=layers - full, window=window, relative_buckets=0, glu=True,
                  num_kv_heads=heads, bidirectional=True)
        layer_params = layers * (layer + 2 * h)
        embedding = vocab * h + h
    elif model_type in ("deberta-v2", "deberta"):
        buckets = int(cfg.get("position_buckets") or 0) or int(cfg.get("max_relative_positions") or 0)
        buckets = buckets if buckets > 0 else int(cfg.get("max_position_embeddings") or 512)
        layer = 4 * h * h + 4 * h + 2 * h * i + i + h
        kw = dict(full_layers=layers, window_layers=0, window=None, relative_buckets=buckets, glu=False,
                  num_kv_heads=heads, bidirectional=True)
        layer_params = layers * (layer + 4 * h) + 2 * buckets * h + 2 * h
        embedding = vocab * h + 2 * h
    elif model_type in ("qwen2", "qwen3", "llama", "mistral"):
        kv_heads = int(cfg.get("num_key_value_heads") or heads)
        kv = kv_heads * head_dim
        layer = 2 * h * heads * head_dim + 2 * h * kv + 3 * h * i
        bias = (heads * head_dim + 2 * kv) if model_type == "qwen2" else 0
        kw = dict(full_layers=layers, window_layers=0, window=None, relative_buckets=0, glu=True,
                  num_kv_heads=kv_heads, bidirectional=False)
        layer_params = layers * (layer + bias + 2 * h)
        tied = bool(cfg.get("tie_word_embeddings", True))
        embedding = vocab * h * (1 if tied else 2) + h
    else:
        raise FamilyError(f"the {fam.name} family's backbone '{model_type or 'unknown'}' is not one the runtime serves")

    head, lora, max_len = 0, 0, 0
    if fam.name in ("laya", "julia"):
        head_layers = int(own.get("head_layers") or 2)
        head = head_layers * 12 * h * h + h * h + 256 * h      # the marker head's layers, scorer and act head
        if fam.name == "laya":
            max_len = int(own.get("max_len") or _LAYA_MAX_LEN)
        else:
            max_len = int(cfg.get("max_position_embeddings") or 8192)
    elif fam.name == "gliner":
        head = int(_GLINER_HEAD_PER_HIDDEN2 * h * h)
        max_len = _GLINER_MAX_LEN
    elif fam.name == "kev":
        r = int(own.get("r") or 0)
        targets = own.get("target_modules") or []
        per_layer = 0
        kv_width = int(kw["num_kv_heads"]) * head_dim
        for name in targets if isinstance(targets, (list, tuple)) else []:
            out = {"q_proj": heads * head_dim, "k_proj": kv_width, "v_proj": kv_width, "o_proj": h,
                   "gate_proj": i, "up_proj": i, "down_proj": h}.get(str(name))
            inp = {"o_proj": heads * head_dim, "down_proj": i}.get(str(name), h)
            if out:
                per_layer += r * (inp + out)
        lora = layers * per_layer
        head = 2 * (h * h + h)                                # the pointer head's q and k projections
        max_len = _KEV_MAX_STATE + _KEV_QUESTION_TOKENS

    return DecisionFacts(
        family=fam.name, kind=fam.kind, layout=fam.layout, backbone=model_type,
        hidden_size=h, intermediate_size=i, num_layers=layers, num_heads=heads, head_dim=head_dim,
        vocab_size=vocab, max_len=max_len, embedding_params=int(embedding), layer_params=int(layer_params),
        head_params=int(head), lora_params=int(lora), keeps_state=fam.keeps_state, **kw,
    )


def _family(name: str) -> Family:
    fam = FAMILIES.get(str(name or "").lower())
    if fam is None:
        raise FamilyError(f"unknown decision family '{name}': sized families are {', '.join(sorted(FAMILIES))}")
    return fam


def detect(files: Any) -> Optional[str]:
    """The family whose required files are all among ``files`` (relative paths), or None. Families are tried most
    specific first, so a GLiNER checkpoint (whose root config.json any model has) is matched by its other files."""
    present = {str(f).strip("/").replace("\\", "/") for f in files or ()}
    for name in ("laya", "julia", "kev", "gliner"):
        if all(req in present for req in FAMILIES[name].required_files):
            return name
    return None
