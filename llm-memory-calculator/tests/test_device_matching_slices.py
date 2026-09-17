"""A GPU must be recognised whatever slice of it the cluster reports.

Cluster inventory rarely reports the physical card. A HAMi slice reports its `nvidia.com/gpumem`, an
NVIDIA vGPU profile its framebuffer, and a partly-used card its free memory. The matcher used to require
the reported memory to be within 15% of a card's, so every one of those matched nothing. budsim then
fell back to a generic A100-40GB, predicting an H100 slice 1.2-2.1x too slow and sizing it as a 40 GB
card -- or, when even that failed, produced a plan with no latency metrics at all.

Observed memory is a LOWER BOUND on the card. These tests pin that contract generically: exact whole
cards keep their exact variant, slices resolve to their family, and a family token never leaks into a
neighbouring family (A10 vs A100).
"""
import pytest

from llm_memory_calculator import HardwareManager
from llm_memory_calculator.hardware.device_matcher import DeviceParser


@pytest.fixture(scope="module")
def manager():
    return HardwareManager()


def _matched(manager, raw_name, memory_mb=None):
    info = {"raw_name": raw_name}
    if memory_mb is not None:
        info["memory_mb"] = memory_mb
    manager.clear_match_cache()
    specs = manager.get_cluster_hardware_specs(info)
    return specs["device_name"] if specs["matched"] else None


# ------------------------------------------------------------------ slices of a known card


@pytest.mark.parametrize("memory_mb", [None, 81920, 80000, 59790, 33446, 20480, 8192])
def test_every_slice_of_an_h100_vgpu_is_an_h100(manager, memory_mb):
    """The vGPU device string, at the slice sizes budcluster actually rendered."""
    assert _matched(manager, "NVIDIA-H100XM-80C", memory_mb) == "H100_GPU"


def test_the_budsim_call_gets_h100_performance_not_the_a100_fallback(manager):
    manager.clear_match_cache()
    specs = manager.get_cluster_hardware_specs({"raw_name": "NVIDIA-H100XM-80C", "memory_mb": 33446})
    assert specs["matched"]
    assert specs["flops_fp16"] == 989.5
    assert specs["memory_bandwidth_gbs"] == 3350
    assert specs["memory_size_gb"] == 80, "the card, not the slice, is the device's memory size"


@pytest.mark.parametrize(
    "raw_name, memory_mb, expected",
    [
        # the size in the name is the card; a smaller reported size is a slice of it
        ("NVIDIA A100-SXM4-40GB", 20000, "A100_40GB_GPU"),
        ("NVIDIA A100-SXM4-80GB", 30000, "A100_80GB_GPU"),
        # no size in the name: the smallest card that can hold what was observed
        ("NVIDIA A100", 61000, "A100_80GB_GPU"),
        ("NVIDIA A100", 20000, "A100_40GB_GPU"),
        # vGPU profile framebuffer is a lower bound too
        ("NVIDIA L40S-48C", 12000, "L40S_48GB_GPU"),
    ],
)
def test_memory_variant_is_the_smallest_card_that_holds_the_observation(manager, raw_name, memory_mb, expected):
    assert _matched(manager, raw_name, memory_mb) == expected


# ------------------------------------------------------------------ nothing else moves


@pytest.mark.parametrize(
    "raw_name, memory_mb, expected",
    [
        ("NVIDIA A100-SXM4-80GB", 81920, "A100_80GB_GPU"),
        ("NVIDIA A100-SXM4-40GB", 40960, "A100_40GB_GPU"),
        ("Tesla V100-SXM2-16GB", None, "V100_16GB_GPU"),
        ("NVIDIA H100", 81559, "H100_GPU"),
    ],
)
def test_whole_cards_keep_their_exact_match(manager, raw_name, memory_mb, expected):
    assert _matched(manager, raw_name, memory_mb) == expected


def test_a_family_token_does_not_leak_into_a_neighbouring_family(manager):
    """'A10' is a prefix of 'A100'; a slice of an A10 must never be timed as an A100."""
    assert _matched(manager, "NVIDIA A10", 20000) == "A10_GPU"


def test_memory_no_card_in_the_family_can_hold_is_not_forced_onto_that_family(manager):
    assert _matched(manager, "NVIDIA H100", 400 * 1024) != "H100_GPU"


# ------------------------------------------------------------------ parsing


@pytest.mark.parametrize(
    "info, name_memory, lower_bound",
    [
        ({"raw_name": "NVIDIA-H100XM-80C"}, 80.0, 80.0),
        ({"raw_name": "NVIDIA-H100XM-80C", "memory_mb": 33446}, 80.0, 80.0),
        ({"raw_name": "NVIDIA A100-SXM4-40GB", "memory_mb": 20000}, 40.0, 40.0),
        ({"raw_name": "NVIDIA A100", "memory_mb": 61000}, None, 59.6),
        ({"raw_name": "Intel Xeon Platinum 8480C"}, None, None),
    ],
)
def test_reported_and_named_memory_are_both_lower_bounds(info, name_memory, lower_bound):
    identity = DeviceParser.parse(info)
    assert identity.name_memory_gb == name_memory
    assert identity.min_physical_memory_gb == lower_bound


# ------------------------------------------------------------------ every record, not examples


def _unambiguous_labels():
    """(label, memory_mb, record name) for every name/alias that exactly one hardware record carries."""
    import re
    from collections import defaultdict
    from llm_memory_calculator.hardware.configs import HARDWARE_CONFIGS

    norm = lambda label: re.sub(r"[\s_\-]+", "", str(label)).upper()  # noqa: E731
    owners = defaultdict(set)
    for key, cfg in HARDWARE_CONFIGS.items():
        for label in (key, cfg.get("name"), *cfg.get("aliases", [])):
            if label:
                owners[norm(label)].add(key)
    cases = []
    for key, cfg in HARDWARE_CONFIGS.items():
        memory_mb = int((cfg.get("Memory_size") or 0) * 1024) or None
        for label in dict.fromkeys((key, *cfg.get("aliases", []))):
            if len(owners[norm(label)]) == 1:
                cases.append((label, None, cfg.get("name", key)))
                cases.append((label, memory_mb, cfg.get("name", key)))
    return cases


def test_every_unambiguous_label_resolves_to_its_own_record(manager):
    """Swept over the whole hardware table. Before the lower-bound and digit-aware fixes 227 of 512
    name/alias probes resolved to their own record (an A10 became an A100, every RTX label matched
    nothing); a label only one record carries has exactly one right answer."""
    wrong = [(label, memory_mb, expected, got)
             for label, memory_mb, expected in _unambiguous_labels()
             if (got := _matched(manager, label, memory_mb)) != expected]
    assert not wrong, f"{len(wrong)} labels resolve elsewhere, e.g. {wrong[:5]}"
