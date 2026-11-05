"""The accubits02 lab's GPUs have hardware entries (FRD-023 audit A14), so the KV planner can estimate recompute and
judge break-even there instead of dropping every tier as unknown (FR-PLAN-11)."""

import pytest

from llm_memory_calculator.kv.cost import RecomputeModel
from llm_memory_calculator.kv.calibration import QWEN3_06B
from llm_memory_calculator.kv.group import genz_hardware


@pytest.mark.parametrize(
    ("label", "entry", "flops", "memory_gb"),
    [
        ("NVIDIA-GeForce-GTX-1650", "GTX1650_MOBILE_GPU", 6.4, 4),
        ("NVIDIA GeForce GTX 1650", "GTX1650_MOBILE_GPU", 6.4, 4),
        ("NVIDIA-GeForce-RTX-3050-Laptop-GPU", "RTX3050_LAPTOP_GPU", 17.7, 4),
        ("RTX 3050 Laptop", "RTX3050_LAPTOP_GPU", 17.7, 4),
    ],
)
def test_the_lab_cards_match_their_entries(label, entry, flops, memory_gb):
    hardware, name = genz_hardware(label, "cuda")
    assert name == entry and hardware["Flops"] == flops and hardware["Memory_size"] == memory_gb


def test_a_lab_card_gives_the_planner_a_recompute_estimate():
    hardware, _ = genz_hardware("NVIDIA-GeForce-GTX-1650", "cuda")
    model = RecomputeModel(model_config=QWEN3_06B, model_id="Qwen/Qwen3-0.6B", hardware=hardware, device_tflops=6.4)
    ms, source = model.ms(2048)
    assert ms is not None and ms > 0, source


def test_an_unknown_card_still_has_no_entry():
    hardware, name = genz_hardware("NVIDIA-NOT-A-GPU", "cuda")
    assert hardware is None and name is None
