"""Golden tests for R2-SV4/M14: per-device serving power from datasheet TDP.

Before this fix, every catalog device that lacked a TDP field collapsed to a
flat 300W default, so physically distinct accelerators (e.g. A10 150W vs Gaudi3
900W) reported identical accelerator power. These tests assert the PHYSICAL
outcome: accelerator power scales with each device's published datasheet TDP via
P = idle + util*(active - idle), and never collapses to a single flat number for
distinct device classes.

All expected TDPs are vendor datasheet figures (cited inline at each constant in
configs.py / power_model.py).
"""
import pytest

from llm_memory_calculator.genz.serving.constants import (
    PowerComponent,
    GPU_IDLE_FRACTION,
    GPU_ACTIVE_FRACTION,
    GPU_STANDBY_FRACTION,
)
from llm_memory_calculator.genz.serving.power_model import (
    PowerConfig,
    PowerModel,
    _derive_class_tdp,
    _DEFAULT_TDP_HBM_ACCEL_W,
    _DEFAULT_TDP_HBM_GPU_W,
    _DEFAULT_TDP_GDDR_GPU_W,
    _DEFAULT_TDP_GENERIC_W,
)


def _accel(name: str):
    pm = PowerModel.from_hardware_name(name)
    return pm._config.components[PowerComponent.ACCELERATOR]


# Datasheet TDP (watts) for devices that previously fell back to the flat 300W
# default because they carried no TDP field. Each value is sourced inline in
# configs.py. This is the heart of the golden assertion.
DATASHEET_TDP_W = {
    "A10_GPU": 150,       # NVIDIA A10 datasheet
    "A30_GPU": 165,       # NVIDIA A30 datasheet
    "A40_GPU": 300,       # NVIDIA A40 datasheet
    "A6000_GPU": 300,     # NVIDIA RTX A6000 datasheet
    "TPUv4": 192,         # Jouppi et al. ISCA 2023
    "TPUv5e": 170,        # TPU v5e cost-optimized per-chip power
    "TPUv5p": 450,        # TPU v5p per-chip TDP (TSMC N5)
    "TPUv6": 300,         # TPU v6e (Trillium) per-chip TDP
    "MAX1550": 600,       # Intel Data Center GPU Max 1550 product spec
    "MAX1100": 300,       # Intel Data Center GPU Max 1100 product spec
    "ARC770": 225,        # Intel Arc A770 total board power
    "Gaudi3": 900,        # Intel Gaudi 3 OAM product brief
}


class TestPerDeviceDatasheetTDP:
    @pytest.mark.parametrize("name,tdp", sorted(DATASHEET_TDP_W.items()))
    def test_accelerator_power_tracks_datasheet_tdp(self, name, tdp):
        a = _accel(name)
        assert a.active_power_w == pytest.approx(tdp * GPU_ACTIVE_FRACTION)
        assert a.idle_power_w == pytest.approx(tdp * GPU_IDLE_FRACTION)
        assert a.standby_power_w == pytest.approx(tdp * GPU_STANDBY_FRACTION)

    def test_no_device_collapses_to_flat_300w_default(self):
        """The regression being fixed: distinct devices must NOT all report the
        flat 300W -> 210W active default. Devices with genuinely distinct
        datasheet TDPs must report distinct accelerator power."""
        # A10 (150W) and Gaudi3 (900W) are wildly different physically.
        a10 = _accel("A10_GPU")
        gaudi3 = _accel("Gaudi3")
        assert a10.active_power_w != pytest.approx(gaudi3.active_power_w)
        # Neither equals the old flat 300W default's 210W active.
        flat_default_active = 300 * GPU_ACTIVE_FRACTION  # 210W
        assert a10.active_power_w != pytest.approx(flat_default_active)
        assert gaudi3.active_power_w != pytest.approx(flat_default_active)

    def test_power_spans_real_dynamic_range(self):
        """Lowest-TDP device (A10, 150W) draws far less than highest (Gaudi3,
        900W): a 6x datasheet ratio must show up as a 6x power ratio."""
        a10 = _accel("A10_GPU")
        gaudi3 = _accel("Gaudi3")
        ratio = gaudi3.active_power_w / a10.active_power_w
        assert ratio == pytest.approx(900 / 150, rel=1e-6)  # exactly 6x


class TestUtilizationShape:
    def test_power_interpolates_idle_to_active_with_util(self):
        """P = idle + util*(active - idle): power varies with utilization AND
        with device. Verify on two distinct devices at the same util."""
        for name, tdp in [("A10_GPU", 150), ("Gaudi3", 900)]:
            pm = PowerModel.from_hardware_name(name)
            r = pm.estimate_from_simulation_result(
                latency_ms=100.0, compute_util=0.5, num_accel=1,
            )
            idle = tdp * GPU_IDLE_FRACTION
            active = tdp * GPU_ACTIVE_FRACTION
            expected = idle + (active - idle) * 0.5
            assert r["accelerator_w"] == pytest.approx(expected, rel=1e-6)

    def test_zero_util_is_idle_full_util_is_active(self):
        pm = PowerModel.from_hardware_name("Gaudi3")
        r0 = pm.estimate_from_simulation_result(latency_ms=10.0, compute_util=0.0, num_accel=1)
        pm2 = PowerModel.from_hardware_name("Gaudi3")
        r1 = pm2.estimate_from_simulation_result(latency_ms=10.0, compute_util=1.0, num_accel=1)
        assert r0["accelerator_w"] == pytest.approx(900 * GPU_IDLE_FRACTION, rel=1e-6)
        assert r1["accelerator_w"] == pytest.approx(900 * GPU_ACTIVE_FRACTION, rel=1e-6)


class TestClassDerivationForUndocumentedDevices:
    """Devices with NO published datasheet TDP (AWS Trainium1/Inferentia2) derive
    a class-representative TDP rather than the flat generic 300W."""

    def test_aws_asics_use_hbm_accel_class_default(self):
        for name in ("Trainium1", "Inferentia2"):
            a = _accel(name)
            assert a.active_power_w == pytest.approx(
                _DEFAULT_TDP_HBM_ACCEL_W * GPU_ACTIVE_FRACTION
            )

    def test_class_derivation_by_type_and_memory(self):
        # HBM ASIC / accelerator
        assert _derive_class_tdp({"type": "asic", "memory_type": "HBM"}) == _DEFAULT_TDP_HBM_ACCEL_W
        assert _derive_class_tdp({"type": "accelerator"}) == _DEFAULT_TDP_HBM_ACCEL_W
        # HBM GPU vs GDDR GPU diverge
        assert _derive_class_tdp({"type": "gpu", "memory_type": "HBM3"}) == _DEFAULT_TDP_HBM_GPU_W
        assert _derive_class_tdp({"type": "gpu", "memory_type": "GDDR6"}) == _DEFAULT_TDP_GDDR_GPU_W
        # Typeless / empty hits generic fallback (preserves legacy behaviour)
        assert _derive_class_tdp({}) == _DEFAULT_TDP_GENERIC_W

    def test_empty_config_preserves_legacy_300w(self):
        """Byte-identical guard: an empty {} config still yields the historical
        300W -> 210W active default."""
        pc = PowerConfig.from_hardware_config({})
        a = pc.components[PowerComponent.ACCELERATOR]
        assert a.active_power_w == pytest.approx(300 * GPU_ACTIVE_FRACTION)


class TestPowerFieldResolution:
    def test_power_field_used_when_present(self):
        """DB-format entries carry TDP under the 'Power' key (datasheet TDP)."""
        pc = PowerConfig.from_hardware_config({"Power": 250, "type": "gpu"})
        a = pc.components[PowerComponent.ACCELERATOR]
        assert a.active_power_w == pytest.approx(250 * GPU_ACTIVE_FRACTION)

    def test_resolution_priority_cost_over_toplevel_over_power(self):
        # cost.tdp_watts wins
        pc = PowerConfig.from_hardware_config(
            {"cost": {"tdp_watts": 400}, "tdp_watts": 200, "Power": 100}
        )
        assert pc.components[PowerComponent.ACCELERATOR].active_power_w == pytest.approx(
            400 * GPU_ACTIVE_FRACTION
        )
        # tdp_watts wins over Power when no cost.tdp_watts
        pc2 = PowerConfig.from_hardware_config({"tdp_watts": 200, "Power": 100})
        assert pc2.components[PowerComponent.ACCELERATOR].active_power_w == pytest.approx(
            200 * GPU_ACTIVE_FRACTION
        )
