"""The KV namespace keys on the layout a group's blocks have, not on the backend's name (FRD-028 FR-MIX-13).

On accubits (2026-11-03) the RTX 3050 ran FLASH_ATTN and the GTX 1650 TRITON_ATTN, both picked by vLLM 0.32 on its own.
The 1650 loaded 416 tokens the 3050 had written to the Mooncake pool and answered word for word as the 3050 did: the two
backends keep vLLM's paged cache in the same shape. The namespace hashed ``auto@<card>``, so a mix of the two, planned
group by group, would never have shared a block. Backends that share the layout now share a namespace; FlashInfer keeps
its own, and so does a card whose auto-pick isn't known (Blackwell).
"""

import pytest

from llm_memory_calculator.kv.budsim import kv_layout_family, layout_facts


def _layout(raw_name, backend=None):
    group = {"raw_name": raw_name, "args": {}, "envs": {"VLLM_ATTENTION_BACKEND": backend} if backend else {}}
    return layout_facts(group, {})["attention_backend"]


@pytest.mark.parametrize("backend", ["FLASH_ATTN", "TRITON_ATTN", "FLEX_ATTENTION"])
def test_the_backends_that_share_the_paged_layout_share_a_family(backend):
    assert kv_layout_family(backend) == "paged-nhd"


def test_flashinfer_keeps_its_own_family():
    assert kv_layout_family("FLASHINFER") == "FLASHINFER"


@pytest.mark.parametrize(
    "card",
    [
        "NVIDIA-GeForce-RTX-3050-Laptop-GPU",
        "NVIDIA-GeForce-GTX-1650",
        "NVIDIA-H100-80GB-HBM3",
        "NVIDIA-A100-SXM4-80GB",
        "NVIDIA-L40S",
        "Tesla-T4",
        "NVIDIA-GeForce-RTX-4090",
    ],
)
def test_cards_whose_auto_pick_is_known_resolve_to_the_paged_family(card):
    """Volta to Hopper: vLLM picks FLASH_ATTN where it runs (Ampere and newer), else TRITON_ATTN."""
    assert _layout(card) == "auto@paged-nhd"


def test_the_two_accubits_cards_get_one_namespace_family():
    assert _layout("NVIDIA-GeForce-RTX-3050-Laptop-GPU") == _layout("NVIDIA-GeForce-GTX-1650")


def test_a_card_whose_auto_pick_is_unknown_keeps_its_own():
    """Blackwell's default backend isn't pinned down: no sharing with anything until it is."""
    assert _layout("NVIDIA-B200") == "auto@NVIDIA-B200"
    assert _layout("NVIDIA-B200") != _layout("NVIDIA-H100-80GB-HBM3")


def test_a_pinned_backend_resolves_to_its_family_on_any_card():
    assert _layout("NVIDIA-B200", "TRITON_ATTN") == "paged-nhd"
    assert _layout("NVIDIA-GeForce-GTX-1650", "FLASHINFER") == "FLASHINFER"


def test_an_unknown_pinned_backend_is_kept_by_name():
    assert _layout("NVIDIA-H100-80GB-HBM3", "SOME_NEW_BACKEND") == "SOME_NEW_BACKEND"


def test_a_bf16_model_on_a_card_without_bf16_is_hashed_as_the_fp16_it_runs():
    """vLLM computes in float16 on a pre-Ampere card, so its blocks must not share a namespace with a bf16 group's."""
    bf16 = {"torch_dtype": "bfloat16"}
    turing = layout_facts({"raw_name": "NVIDIA-GeForce-GTX-1650", "args": {}, "envs": {}}, bf16)
    ampere = layout_facts({"raw_name": "NVIDIA-GeForce-RTX-3050-Laptop-GPU", "args": {}, "envs": {}}, bf16)
    assert turing["resolved_dtype"] == "float16" and ampere["resolved_dtype"] == "bfloat16"
