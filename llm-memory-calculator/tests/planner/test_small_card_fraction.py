"""A dedicated card keeps an absolute reserve outside vLLM's fraction, not only a percentage.

``default_gpu_memory_utilization`` is 0.92: 6.4 GiB left outside on an 80 GB card, 0.3 GiB on a 4 GiB one. Activation memory
lives there, so on accubits (2026-10-30) a pod that booted cleanly died with a GPU out-of-memory error at 5 requests in flight on
the RTX 3050 and at 8 on the GTX 1650. A plan accepted by the fit check, which assumes 1.5 GiB outside, rendered a fraction that
left a fifth of that. Measured at maximum activation on the RTX 3050: 0.85 held, 0.88 did not.
"""

import math
from dataclasses import replace

import pytest

from llm_memory_calculator.device import HardwareMode, catalogue
from llm_memory_calculator.engine import record as record_module
from llm_memory_calculator.runtime.modules import legacy

from .test_runtime_spec import UP, VGPU

GIB = 1024**3


def card(gib):
    """Return a dedicated card of ``gib`` GiB."""
    device = catalogue.get("rtx3050-laptop-4gb")
    return replace(device, memory_visible_bytes=int(gib * GIB))


def test_the_reserve_is_the_one_the_fit_check_assumes():
    """One number: a plan the fit check accepts renders a fraction that leaves what it assumed."""
    assert record_module.DEDICATED_OUTSIDE_RESERVE_BYTES == legacy.DEVICE_CTX_BYTES


def test_a_4_gib_card_keeps_the_reserve_outside():
    """The accubits cards: 1.5 GiB outside, so well under the 0.88 that crashed."""
    assert UP.default_fraction(card(4)) == pytest.approx(0.625)


@pytest.mark.parametrize(("gib", "fraction"), [(8, 0.8125), (12, 0.875), (16, 0.90625)])
def test_mid_size_cards_get_what_the_reserve_leaves(gib, fraction):
    """Between 4 GiB and the crossover, the fraction rises with the card."""
    assert UP.default_fraction(card(gib)) == pytest.approx(fraction)


@pytest.mark.parametrize("gib", [24, 48, 80])
def test_a_large_card_keeps_the_records_fraction(gib):
    """Above about 19 GiB the percentage already leaves more than the reserve: no change."""
    assert UP.default_fraction(card(gib)) == 0.92


def test_the_vgpu_measured_on_tcs_is_unchanged():
    """The H100 vGPU's plans don't move."""
    assert UP.default_fraction(VGPU) == 0.92


def test_no_device_gets_the_records_fraction():
    """Callers with no device, such as the engine's own defaults, are unchanged."""
    assert UP.default_fraction(None) == 0.92


def test_a_shared_slice_is_left_to_its_own_reserve():
    """Shared mode sizes the slice so the reserve stays outside already (``SLICE_OUTSIDE_RESERVE``)."""
    device = card(80).with_mode(HardwareMode(kind="shared", slice_bytes=2 * GIB))
    assert UP.default_fraction(device) == 0.92


def test_a_very_small_card_doesnt_go_below_half():
    """The fit check refuses a plan here anyway; this keeps a usable number if something asks."""
    assert UP.default_fraction(card(2)) == 0.5


def test_the_ceiling_follows_so_nothing_can_raise_it_back():
    """``max_fraction`` caps the KV plan's T0 target (0.92) and the boot check: a small card can't be raised past its reserve."""
    assert UP.max_fraction(card(4)) == pytest.approx(0.625)
    assert UP.max_fraction(card(80)) == 0.95


def test_the_fraction_leaves_the_reserve_on_every_size():
    """The property the rule exists for, across sizes."""
    for gib in (3, 4, 6, 8, 10, 12, 16, 18):
        left = (1 - UP.default_fraction(card(gib))) * gib * GIB
        assert left >= record_module.DEDICATED_OUTSIDE_RESERVE_BYTES - 1 or math.isclose(UP.default_fraction(card(gib)), 0.5)
