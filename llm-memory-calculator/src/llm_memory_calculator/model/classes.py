"""Architecture class, attributes and size bucket, derived from model facts (FRD-027 FR-MDL-6).

The class is what grounding falls back to when a family is unmeasured; it is computed, never assigned, so a
new checkpoint of a known shape lands in the right class without a table edit.
"""

from __future__ import annotations

from typing import Tuple

ARCH_CLASSES = ("dense-GQA", "MoE", "hybrid-linear", "MLA", "encoder-decoder")

#: Size buckets by ACTIVE parameters (FR-MDL-6): the roofline already scales with bytes and FLOPs; grounding
#: only needs the efficiency at that size.
SIZE_BUCKETS: Tuple[Tuple[str, float], ...] = (
    ("<=4B", 4e9),
    ("4-10B", 10e9),
    ("10-40B", 40e9),
    ("40-120B", 120e9),
    (">120B", float("inf")),
)


def size_bucket(active_params: float) -> str:
    for name, upper in SIZE_BUCKETS:
        if active_params <= upper:
            return name
    return SIZE_BUCKETS[-1][0]


def arch_class(*, encoder_decoder: bool, recurrent_layers: int, mla: bool, moe: bool) -> str:
    """Precedence: encoder-decoder > hybrid-linear > MLA > MoE > dense-GQA (MoE stays an attribute of the
    first three)."""
    if encoder_decoder:
        return "encoder-decoder"
    if recurrent_layers > 0:
        return "hybrid-linear"
    if mla:
        return "MLA"
    if moe:
        return "MoE"
    return "dense-GQA"
