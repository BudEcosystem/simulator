"""The shape of a model's KV cache as vLLM stores it, and the bytes a tier moves for a prefix.

KV bytes per token come from :meth:`ModelMemoryCalculator.kv_cache_breakdown` (FR-PLAN-2). For a
hybrid model (linear-attention or Mamba layers) vLLM also keeps a fixed-size state per cached
prefix, and sizes its block so one attention page holds at least one state page
(``platforms/interface.py`` in vLLM 0.30). Both checked on TCS (spike S10): Qwen3.8-27B gets
784-token blocks, and the pool moved exactly 3,920 x 64 KiB + 48 x 3,211,264 bytes for a 4K prefix.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

from ..calculator import ModelMemoryCalculator
from ..layer_plan import resolve_layer_plan
from ..state_memory import calculate_recurrent_state_bytes
from .constants import (
    BUD_VLLM_MINOR_CEILING,
    VLLM_BLOCK_SIZE,
    VLLM_BLOCK_SIZE_V034,
    VLLM_BLOCK_SIZE_V034_BUD,
    VLLM_BLOCK_SIZE_V034_UPSTREAM,
    VLLM_KERNEL_BLOCK_ALIGNMENT,
)


def vllm_default_block_size(engine_version: Optional[str]) -> int:
    """The KV block vLLM pages a dense model with when no ``--block-size`` is passed: 64 from v0.34.0, 16 before.

    ``engine_version`` is :func:`kv.budsim.engine_version`'s ``"<version>|<image>"`` (or a bare version): the
    upstream vLLM base when BudConnect lists one, else the Bud build number. A 0.x minor below
    ``BUD_VLLM_MINOR_CEILING`` reads as a Bud build. No parsable version keeps the measured default (16), which is
    what every calibration and fixture here ran on.
    """
    head = str(engine_version or "").split("|", 1)[0].strip().lstrip("v")
    match = re.match(r"^(\d+)\.(\d+)", head)
    if not match:
        return VLLM_BLOCK_SIZE
    major_minor = (int(match.group(1)), int(match.group(2)))
    is_bud_build = major_minor[0] == 0 and major_minor[1] < BUD_VLLM_MINOR_CEILING
    threshold = VLLM_BLOCK_SIZE_V034_BUD if is_bud_build else VLLM_BLOCK_SIZE_V034_UPSTREAM
    return VLLM_BLOCK_SIZE_V034 if major_minor >= threshold else VLLM_BLOCK_SIZE


def _text_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    text = config.get("text_config")
    return text if isinstance(text, Mapping) else config


def kv_bytes_per_token_per_rank(
    model_config: Mapping[str, Any],
    *,
    seq_length: int,
    tensor_parallel: int = 1,
    precision: str = "bf16",
    engine_capabilities: Any = None,
    data_parallel_attention: bool = False,
) -> Tuple[float, float, str]:
    """Return ``(bytes per token per rank, bytes per token for the whole model, attention type)``.

    From :meth:`ModelMemoryCalculator.kv_cache_breakdown`, the one source of per-token KV
    (FR-PLAN-2). An MLA latent is not split by head-wise tensor parallelism -- every rank holds the
    whole latent -- unless attention is data-parallel.
    """
    breakdown = ModelMemoryCalculator().kv_cache_breakdown(
        dict(model_config),
        batch_size=1,
        seq_length=max(int(seq_length), 1),
        precision=precision,
        engine_capabilities=engine_capabilities,
    )
    total = float(breakdown.get("marginal_bytes_per_token") or 0.0)
    attention = str(breakdown.get("attention_type") or "")
    tp = max(int(tensor_parallel or 1), 1)
    replicated = attention == "mla" and not data_parallel_attention
    per_rank = total if replicated else total / tp
    return per_rank, total, attention


@dataclass(frozen=True)
class KVGeometry:
    """A model's KV layout on one deployment: bytes per token, layers, block, hybrid state."""

    bytes_per_token_rank: float
    bytes_per_token_total: float
    attention_type: str
    kv_layers: int  # layers whose KV grows with the sequence (full attention)
    recurrent_layers: int  # linear-attention / Mamba layers with a fixed state
    state_bytes_per_layer: float  # one recurrent layer's state for one sequence, whole model
    block_size: int  # vLLM's KV block, tokens
    tp: int = 1

    @classmethod
    def from_model(
        cls,
        model_config: Mapping[str, Any],
        *,
        seq_length: int,
        tensor_parallel: int = 1,
        precision: str = "bf16",
        engine_capabilities: Any = None,
        data_parallel_attention: bool = False,
        default_block_size: int = VLLM_BLOCK_SIZE,
    ) -> "KVGeometry":
        """``default_block_size`` is the engine's dense block (:func:`vllm_default_block_size`)."""
        per_rank, total, attention = kv_bytes_per_token_per_rank(
            model_config,
            seq_length=seq_length,
            tensor_parallel=tensor_parallel,
            precision=precision,
            engine_capabilities=engine_capabilities,
            data_parallel_attention=data_parallel_attention,
        )
        config = dict(model_config)
        plan = resolve_layer_plan(config)
        text = _text_config(config)
        layers = int(text.get("num_hidden_layers") or text.get("n_layer") or 0)
        if plan:
            kv_layers = int(plan.get("num_full_layers") or 0) or int(
                plan.get("num_attention_layers") or 0
            )
            recurrent = int(plan.get("num_recurrent_layers") or 0)
        else:
            kv_layers, recurrent = layers, 0
        kv_layers = max(kv_layers, 1)
        state_per_layer = 0.0
        if recurrent:
            state_bytes = calculate_recurrent_state_bytes(config, plan, batch_size=1, model_bytes=2)
            state_per_layer = float(state_bytes) / recurrent
        geometry = cls(
            bytes_per_token_rank=per_rank,
            bytes_per_token_total=total,
            attention_type=attention,
            kv_layers=kv_layers,
            recurrent_layers=recurrent,
            state_bytes_per_layer=state_per_layer,
            block_size=int(default_block_size),
            tp=max(int(tensor_parallel or 1), 1),
        )
        return cls(**{**geometry.__dict__, "block_size": geometry.vllm_block_size()})

    @property
    def hybrid(self) -> bool:
        return self.recurrent_layers > 0

    @property
    def attention_page_per_token(self) -> float:
        """One full-attention layer's KV bytes for one token, whole model."""
        return self.bytes_per_token_total / self.kv_layers if self.kv_layers else 0.0

    def vllm_block_size(self) -> int:
        """vLLM's block: the engine's dense default (``self.block_size`` as constructed), or for a hybrid the
        smallest aligned block whose attention page holds a state page (vLLM 0.30, ``mamba_cache_mode=align``).

        The alignment unit is the larger of the kernel's 16 and the dense default, as vLLM v0.34.0's
        ``Platform._align_hybrid_block_size`` takes it: a 784-token block at 16 is 832 at 64.
        """
        if not self.hybrid or self.attention_page_per_token <= 0:
            return self.block_size
        align = max(VLLM_KERNEL_BLOCK_ALIGNMENT, self.block_size)
        pages = math.ceil(self.state_bytes_per_layer / (align * self.attention_page_per_token))
        return align * max(pages, 1)

    def segment_bytes(self, block_size: Optional[int] = None) -> float:
        """One layer's bytes for one block on one rank: the unit vLLM's Mooncake connector moves."""
        block = block_size or self.block_size
        return block * self.bytes_per_token_rank / self.kv_layers if self.kv_layers else 0.0

    def state_bytes_stored(self, block_size: Optional[int] = None, scope: str = "total") -> float:
        """A hybrid prefix's stored state: one page per recurrent layer, padded to the attention
        page (so a larger block stores more padding, FR-T3-5)."""
        if not self.hybrid:
            return 0.0
        block = block_size or self.block_size
        page = max(self.state_bytes_per_layer, block * self.attention_page_per_token)
        total = self.recurrent_layers * page
        return total / self.tp if scope == "rank" else total

    @staticmethod
    def cached_tokens(prefix_tokens: int, block_size: int) -> int:
        """vLLM caches whole blocks only."""
        block = max(int(block_size), 1)
        return (max(int(prefix_tokens), 0) // block) * block

    def prefix_bytes(
        self, prefix_tokens: int, *, block_size: Optional[int] = None, scope: str = "rank"
    ) -> Tuple[int, float]:
        """Return ``(cached tokens, bytes)`` a tier moves for one prefix.

        ``scope`` is ``rank`` (one GPU's share: T1 over that GPU's PCIe link) or ``total`` (every
        rank's share: T2 and T3, whose link the node's ranks share). Every rank stores its own
        share, so ``total`` is the per-rank bytes times TP; for a replicated MLA latent that is
        TP whole copies.
        """
        block = block_size or self.block_size
        tokens = self.cached_tokens(prefix_tokens, block)
        per_token = self.bytes_per_token_rank * (1 if scope == "rank" else self.tp)
        return tokens, tokens * per_token + self.state_bytes_stored(block, scope)

    def offload_block(self, requested: int) -> int:
        """An offload chunk is a whole number of vLLM blocks (budcluster rounds the same way)."""
        return max(self.block_size, math.ceil(requested / self.block_size) * self.block_size)
