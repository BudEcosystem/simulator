"""The Ada data-center cards are told apart, and the L20 carries its own price.

gtech (2026-10-20) runs one NVIDIA L20 (48 GB GDDR6, reported by HAMi as 44.99 GB). The hardware table had no L20,
so budsim priced it with its generic CUDA fallback, an A100's $27,164, at $0.62/h: almost an H100's $0.685/h, against
the L40S's $0.18/h. The PCI table also had the Ada IDs shifted: pci.ids lists 10de:26b5 as L40, 26b9 as L40S and
26ba as L20, and the matcher read 26b9 as L40 and 26ba (the L20) as L40S.
"""

import pytest

from llm_memory_calculator import HardwareManager
from llm_memory_calculator.hardware.device_matcher import DeviceMatcher


@pytest.fixture(scope="module")
def manager():
    return HardwareManager()


def _specs(manager, **info):
    manager.clear_match_cache()
    return manager.get_cluster_hardware_specs(info)


@pytest.mark.parametrize("memory_mb", [None, 49152, 46070, 20480])
def test_an_l20_is_an_l20_whatever_slice_is_reported(manager, memory_mb):
    """The whole card, the 44.99 GB HAMi reports on gtech, and a smaller slice all resolve to the L20."""
    info = {"raw_name": "NVIDIA-L20"}
    if memory_mb:
        info["memory_mb"] = memory_mb
    specs = _specs(manager, **info)
    assert specs["matched"] and specs["device_name"] == "L20_48GB_GPU"
    assert specs["memory_size_gb"] == 48 and specs["memory_bandwidth_gbs"] == 864


def test_the_l20_has_its_own_price(manager):
    """The street price (2026-10: $4,299-4,550 from resellers), not the A100 fallback.

    No cloud lists the L20 by the hour, so it has a purchase price and no rental rate.
    """
    specs = _specs(manager, raw_name="NVIDIA L20", memory_mb=46070)
    assert specs["purchase_price_usd"] == 4500 and specs["tdp_watts"] == 275


@pytest.mark.parametrize(
    "pci_device, card", [("26ba", "L20_48GB_GPU"), ("26b9", "L40S_48GB_GPU")]
)
def test_the_ada_pci_ids_name_the_right_card(manager, pci_device, card):
    """10de:26ba is the L20 and 10de:26b9 the L40S (pci.ids AD102GL)."""
    specs = _specs(manager, raw_name="NVIDIA Device", pci_vendor="10de", pci_device=pci_device, memory_mb=46070)
    assert specs["matched"] and specs["device_name"] == card


def test_the_l20_never_claims_a_neighbour():
    """The L20 token stays the L20, and the L40S and L4 keep theirs."""
    assert DeviceMatcher.PCI_ID_MAP["10de:26b5"] == "L40"
    assert DeviceMatcher.PCI_ID_MAP["10de:26b9"] == "L40S"
    assert DeviceMatcher.PCI_ID_MAP["10de:26ba"] == "L20"
    from llm_memory_calculator.hardware.gpu_specs import get_gpu_info_by_pci_id

    assert [get_gpu_info_by_pci_id("10de", d)["model"] for d in ("26b5", "26b9", "26ba")] == ["L40", "L40S", "L20"]
    from llm_memory_calculator.hardware.configs import HARDWARE_CONFIGS

    assert HARDWARE_CONFIGS["L40S_48GB_GPU"]["pci_ids"] == ["26b9"]
    assert HARDWARE_CONFIGS["L20_48GB_GPU"]["pci_ids"] == ["26ba"]
