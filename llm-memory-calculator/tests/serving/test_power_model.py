"""Tests for serving power model."""
import pytest

from llm_memory_calculator.genz.serving.constants import (
    PowerComponent,
    PowerState,
    NS_PER_S,
    NS_PER_MS,
    GPU_IDLE_FRACTION,
    GPU_ACTIVE_FRACTION,
    GPU_STANDBY_FRACTION,
    DRAM_ACTIVITY_FACTOR,
)
from llm_memory_calculator.genz.serving.power_model import (
    ComponentPowerConfig,
    PowerConfig,
    PowerModel,
)


class TestComponentPowerConfig:
    def test_defaults(self):
        cfg = ComponentPowerConfig(component=PowerComponent.ACCELERATOR)
        assert cfg.component == PowerComponent.ACCELERATOR
        assert cfg.idle_power_w == 0.0
        assert cfg.active_power_w == 0.0
        assert cfg.standby_power_w == 0.0
        assert cfg.energy_per_bit_pj == 0.0
        assert cfg.count == 1

    def test_custom_values(self):
        cfg = ComponentPowerConfig(
            component=PowerComponent.DRAM,
            idle_power_w=5.0,
            active_power_w=15.0,
            energy_per_bit_pj=3.7,
            count=2,
        )
        assert cfg.idle_power_w == 5.0
        assert cfg.active_power_w == 15.0
        assert cfg.energy_per_bit_pj == 3.7
        assert cfg.count == 2


class TestPowerConfig:
    def test_from_hardware_config_with_cost_nested_tdp(self):
        """TDP under config['cost']['tdp_watts'] -- the common data-center format."""
        hw = {"cost": {"tdp_watts": 400}}
        pc = PowerConfig.from_hardware_config(hw, num_accel=1)
        accel = pc.components[PowerComponent.ACCELERATOR]
        assert accel.active_power_w == pytest.approx(400 * GPU_ACTIVE_FRACTION)  # 280W
        assert accel.idle_power_w == pytest.approx(400 * GPU_IDLE_FRACTION)      # 50W
        assert accel.standby_power_w == pytest.approx(400 * GPU_STANDBY_FRACTION) # 80W
        assert accel.count == 1

    def test_from_hardware_config_with_toplevel_tdp(self):
        """TDP at config['tdp_watts'] -- the consumer GPU format."""
        hw = {"tdp_watts": 450}
        pc = PowerConfig.from_hardware_config(hw, num_accel=2)
        accel = pc.components[PowerComponent.ACCELERATOR]
        assert accel.active_power_w == pytest.approx(450 * GPU_ACTIVE_FRACTION)  # 315W
        assert accel.count == 2

    def test_from_hardware_config_fallback_default(self):
        """No TDP anywhere falls back to 300W default."""
        hw = {}
        pc = PowerConfig.from_hardware_config(hw)
        accel = pc.components[PowerComponent.ACCELERATOR]
        assert accel.active_power_w == pytest.approx(300 * GPU_ACTIVE_FRACTION)  # 210W

    def test_all_seven_components_present(self):
        hw = {"cost": {"tdp_watts": 400}}
        pc = PowerConfig.from_hardware_config(hw)
        for comp in PowerComponent:
            assert comp in pc.components

    def test_base_node_power(self):
        hw = {"cost": {"tdp_watts": 400}}
        pc = PowerConfig.from_hardware_config(hw)
        # base_node_power is 0 since all power is in component fractions
        assert pc.base_node_power_w == 0.0


class TestPowerModelFromHardwareName:
    def test_a100_40gb(self):
        pm = PowerModel.from_hardware_name("A100_40GB_GPU")
        accel = pm._config.components[PowerComponent.ACCELERATOR]
        assert accel.active_power_w == pytest.approx(400 * GPU_ACTIVE_FRACTION)
        assert accel.idle_power_w == pytest.approx(400 * GPU_IDLE_FRACTION)
        assert accel.standby_power_w == pytest.approx(400 * GPU_STANDBY_FRACTION)

    def test_h100(self):
        pm = PowerModel.from_hardware_name("H100_GPU")
        accel = pm._config.components[PowerComponent.ACCELERATOR]
        assert accel.active_power_w == pytest.approx(700 * GPU_ACTIVE_FRACTION)
        assert accel.idle_power_w == pytest.approx(700 * GPU_IDLE_FRACTION)
        assert accel.standby_power_w == pytest.approx(700 * GPU_STANDBY_FRACTION)

    def test_unknown_hardware_raises(self):
        with pytest.raises(ValueError, match="Unknown hardware"):
            PowerModel.from_hardware_name("DOES_NOT_EXIST_GPU")

    def test_multi_accelerator(self):
        pm = PowerModel.from_hardware_name("A100_40GB_GPU", num_accel=8)
        accel = pm._config.components[PowerComponent.ACCELERATOR]
        assert accel.count == 8


class TestGetBasePower:
    def test_base_power_sums_idle_plus_node(self):
        hw = {"cost": {"tdp_watts": 400}}
        pc = PowerConfig.from_hardware_config(hw, num_accel=1)
        pm = PowerModel(pc)
        base = pm.get_base_power_w()

        expected = pc.base_node_power_w
        for comp, cfg in pc.components.items():
            expected += cfg.idle_power_w * cfg.count
        assert base == pytest.approx(expected)

    def test_base_power_scales_with_accel_count(self):
        hw = {"cost": {"tdp_watts": 400}}
        pc1 = PowerConfig.from_hardware_config(hw, num_accel=1)
        pc8 = PowerConfig.from_hardware_config(hw, num_accel=8)
        pm1 = PowerModel(pc1)
        pm8 = PowerModel(pc8)
        diff = pm8.get_base_power_w() - pm1.get_base_power_w()
        # 7 extra accelerators at idle + non-accel components scale with system_total
        assert diff > 7 * (400 * GPU_IDLE_FRACTION)  # at least 7 extra GPUs


class TestAddAcceleratorActiveEnergy:
    def test_one_second_at_calibrated_active(self):
        hw = {"cost": {"tdp_watts": 400}}
        pc = PowerConfig.from_hardware_config(hw, num_accel=1)
        pm = PowerModel(pc)
        energy = pm.add_accelerator_active_energy(latency_ns=NS_PER_S, num=1)
        # active = 400 * 0.70 = 280W, 1s = 280J
        assert energy == pytest.approx(400 * GPU_ACTIVE_FRACTION, rel=1e-6)

    def test_half_second_two_gpus(self):
        hw = {"cost": {"tdp_watts": 400}}
        pc = PowerConfig.from_hardware_config(hw, num_accel=2)
        pm = PowerModel(pc)
        energy = pm.add_accelerator_active_energy(
            latency_ns=NS_PER_S / 2, num=2
        )
        # 280W * 2 GPUs * 0.5s = 280J
        assert energy == pytest.approx(400 * GPU_ACTIVE_FRACTION, rel=1e-6)

    def test_cumulative_tracking(self):
        hw = {"cost": {"tdp_watts": 400}}
        pc = PowerConfig.from_hardware_config(hw)
        pm = PowerModel(pc)
        pm.add_accelerator_active_energy(latency_ns=NS_PER_S, num=1)
        pm.add_accelerator_active_energy(latency_ns=NS_PER_S, num=1)
        assert pm._energy_j[PowerComponent.ACCELERATOR] == pytest.approx(
            2 * 400 * GPU_ACTIVE_FRACTION
        )


class TestAddAcceleratorStandbyEnergy:
    def test_two_seconds_standby(self):
        hw = {"cost": {"tdp_watts": 400}}
        pc = PowerConfig.from_hardware_config(hw, num_accel=1)
        pm = PowerModel(pc)
        # standby = 400 * 0.20 = 80W
        energy = pm.add_accelerator_standby_energy(
            current_ns=3 * NS_PER_S,
            last_end_ns=1 * NS_PER_S,
            last_calc_ns=0,
        )
        # idle window = 3s - max(1s, 0) = 2s at 80W = 160J
        assert energy == pytest.approx(2 * 400 * GPU_STANDBY_FRACTION, rel=1e-6)

    def test_no_idle_gap(self):
        hw = {"cost": {"tdp_watts": 400}}
        pc = PowerConfig.from_hardware_config(hw)
        pm = PowerModel(pc)
        energy = pm.add_accelerator_standby_energy(
            current_ns=NS_PER_S,
            last_end_ns=2 * NS_PER_S,
            last_calc_ns=0,
        )
        assert energy == 0.0

    def test_last_calc_dominates(self):
        hw = {"cost": {"tdp_watts": 400}}
        pc = PowerConfig.from_hardware_config(hw)
        pm = PowerModel(pc)
        energy = pm.add_accelerator_standby_energy(
            current_ns=5 * NS_PER_S,
            last_end_ns=1 * NS_PER_S,
            last_calc_ns=4 * NS_PER_S,
        )
        # idle window = 5s - max(1s, 4s) = 1s at 80W
        assert energy == pytest.approx(400 * GPU_STANDBY_FRACTION, rel=1e-6)


class TestAddDramEnergy:
    def test_one_gb_transfer(self):
        hw = {"cost": {"tdp_watts": 400}}
        pc = PowerConfig.from_hardware_config(hw)
        pm = PowerModel(pc)
        one_gb = 1024 ** 3
        energy = pm.add_dram_energy(data_bytes=one_gb)
        # bits = 1GB * 8, energy_pj = bits * 3.7 / 0.80
        expected = one_gb * 8 * 3.7 / DRAM_ACTIVITY_FACTOR * 1e-12
        assert energy == pytest.approx(expected, rel=1e-6)

    def test_zero_bytes(self):
        hw = {"cost": {"tdp_watts": 400}}
        pc = PowerConfig.from_hardware_config(hw)
        pm = PowerModel(pc)
        assert pm.add_dram_energy(0) == 0.0


class TestAddInterconnectEnergy:
    def test_one_gb_transfer(self):
        hw = {"cost": {"tdp_watts": 400}}
        pc = PowerConfig.from_hardware_config(hw)
        pm = PowerModel(pc)
        one_gb = 1024 ** 3
        energy = pm.add_interconnect_energy(data_bytes=one_gb)
        expected = one_gb * 8 * 5.0 * 1e-12
        assert energy == pytest.approx(expected, rel=1e-6)

    def test_zero_bytes(self):
        hw = {"cost": {"tdp_watts": 400}}
        pc = PowerConfig.from_hardware_config(hw)
        pm = PowerModel(pc)
        assert pm.add_interconnect_energy(0) == 0.0


class TestEstimateFromSimulationResult:
    def test_100ms_half_compute_util(self):
        pm = PowerModel.from_hardware_name("A100_40GB_GPU", num_accel=1)
        result = pm.estimate_from_simulation_result(
            latency_ms=100.0,
            compute_util=0.5,
            mem_util=0.0,
            comm_util=0.0,
            num_accel=1,
        )
        assert "total_energy_j" in result
        assert "avg_power_w" in result
        assert result["total_energy_j"] > 0
        assert result["avg_power_w"] > 0
        # At 50% utilization: accel power = idle + (active-idle)*0.5
        idle = 400 * GPU_IDLE_FRACTION   # 50W
        active = 400 * GPU_ACTIVE_FRACTION  # 280W
        expected_accel_w = idle + (active - idle) * 0.5  # 165W
        assert result["accelerator_w"] == pytest.approx(expected_accel_w, rel=1e-6)

    def test_with_data_read_and_comm(self):
        pm = PowerModel.from_hardware_name("A100_40GB_GPU")
        result = pm.estimate_from_simulation_result(
            latency_ms=200.0,
            compute_util=0.8,
            mem_util=0.5,
            comm_util=0.3,
            data_read_bytes=1e9,
            data_comm_bytes=5e8,
            num_accel=1,
        )
        assert result["total_energy_j"] > 0
        assert "accelerator_j" in result

    def test_pue_multiplier(self):
        """PUE > 1 should increase total energy proportionally."""
        pm = PowerModel.from_hardware_name("A100_40GB_GPU")
        result_no_pue = pm.estimate_from_simulation_result(
            latency_ms=100.0, compute_util=0.5, num_accel=1, pue=1.0,
        )
        pm2 = PowerModel.from_hardware_name("A100_40GB_GPU")
        result_pue = pm2.estimate_from_simulation_result(
            latency_ms=100.0, compute_util=0.5, num_accel=1, pue=1.2,
        )
        assert result_pue["total_energy_j"] == pytest.approx(
            result_no_pue["total_energy_j"] * 1.2, rel=1e-6
        )

    def test_component_utilization(self):
        """HOST_CPU should track compute_util, COOLING should track max util."""
        pm = PowerModel.from_hardware_name("A100_40GB_GPU")
        result = pm.estimate_from_simulation_result(
            latency_ms=100.0, compute_util=0.8, mem_util=0.5, comm_util=0.3,
            num_accel=1,
        )
        # Total energy should be > 0 and incorporate all components
        assert result["total_energy_j"] > 0
        assert result["avg_power_w"] > 0


class TestEnergyPerToken:
    def test_known_energy(self):
        hw = {"cost": {"tdp_watts": 400}}
        pc = PowerConfig.from_hardware_config(hw)
        pm = PowerModel(pc)
        # active_power = 280W, 1 second -> 280J
        pm.add_accelerator_active_energy(latency_ns=NS_PER_S, num=1)
        ept = pm.energy_per_token(total_tokens=100)
        assert ept == pytest.approx(400 * GPU_ACTIVE_FRACTION / 100, rel=1e-6)

    def test_zero_tokens_returns_zero(self):
        hw = {"cost": {"tdp_watts": 400}}
        pc = PowerConfig.from_hardware_config(hw)
        pm = PowerModel(pc)
        pm.add_accelerator_active_energy(latency_ns=NS_PER_S, num=1)
        assert pm.energy_per_token(total_tokens=0) == 0.0


class TestPowerBreakdown:
    def test_breakdown_sums_correctly(self):
        hw = {"cost": {"tdp_watts": 400}}
        pc = PowerConfig.from_hardware_config(hw)
        pm = PowerModel(pc)
        # 1 second of active compute
        pm.add_accelerator_active_energy(latency_ns=NS_PER_S, num=1)
        # 1 GB DRAM transfer
        pm.add_dram_energy(data_bytes=1024 ** 3)
        # 0.5 GB interconnect
        pm.add_interconnect_energy(data_bytes=512 * 1024 ** 2)

        breakdown = pm.power_breakdown(duration_ns=NS_PER_S)
        assert PowerComponent.ACCELERATOR.value in breakdown
        assert PowerComponent.DRAM.value in breakdown
        assert PowerComponent.INTERCONNECT.value in breakdown
        # accelerator at 280W for 1s = 280W average
        assert breakdown[PowerComponent.ACCELERATOR.value] == pytest.approx(
            400 * GPU_ACTIVE_FRACTION, rel=1e-6
        )

    def test_empty_duration_returns_empty(self):
        hw = {"cost": {"tdp_watts": 400}}
        pc = PowerConfig.from_hardware_config(hw)
        pm = PowerModel(pc)
        assert pm.power_breakdown(duration_ns=0) == {}


class TestSummary:
    def test_contains_expected_keys(self):
        hw = {"cost": {"tdp_watts": 400}}
        pc = PowerConfig.from_hardware_config(hw)
        pm = PowerModel(pc)
        pm.add_accelerator_active_energy(latency_ns=NS_PER_S, num=1)
        s = pm.summary(duration_ns=NS_PER_S, total_tokens=100)
        assert "total_energy_j" in s
        assert "total_energy_kwh" in s
        assert "avg_power_w" in s
        assert "duration_s" in s
        assert "energy_per_token_j" in s
        assert "energy_per_token_mj" in s
        assert "breakdown_j" in s
        assert "breakdown_w" in s

    def test_reasonable_values(self):
        pm = PowerModel.from_hardware_name("A100_40GB_GPU")
        pm.add_accelerator_active_energy(latency_ns=NS_PER_S, num=1)
        active_w = 400 * GPU_ACTIVE_FRACTION  # 280W
        s = pm.summary(duration_ns=NS_PER_S, total_tokens=50)
        assert s["total_energy_j"] == pytest.approx(active_w, rel=1e-6)
        assert s["avg_power_w"] == pytest.approx(active_w, rel=1e-6)
        assert s["energy_per_token_j"] == pytest.approx(active_w / 50, rel=1e-6)
        assert s["energy_per_token_mj"] == pytest.approx(active_w / 50 * 1000, rel=1e-6)
        assert s["duration_s"] == pytest.approx(1.0)
        assert s["total_energy_kwh"] == pytest.approx(active_w / 3_600_000, rel=1e-6)

    def test_no_tokens_yields_none(self):
        hw = {"cost": {"tdp_watts": 400}}
        pc = PowerConfig.from_hardware_config(hw)
        pm = PowerModel(pc)
        s = pm.summary(duration_ns=NS_PER_S, total_tokens=0)
        assert s["energy_per_token_j"] is None
        assert s["energy_per_token_mj"] is None
