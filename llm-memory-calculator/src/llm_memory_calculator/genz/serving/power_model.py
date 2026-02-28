"""7-component power model for LLM serving simulation.

Tracks energy consumption across accelerators, DRAM, interconnect,
host CPU, cooling, storage, and misc components.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

from .constants import PowerComponent, PowerState, NS_PER_S, NS_PER_MS


@dataclass
class ComponentPowerConfig:
    """Power configuration for a single hardware component."""
    component: PowerComponent
    idle_power_w: float = 0.0
    active_power_w: float = 0.0
    standby_power_w: float = 0.0
    energy_per_bit_pj: float = 0.0  # picojoules per bit (for memory/interconnect)
    count: int = 1


@dataclass
class PowerConfig:
    """Power configuration for an entire hardware node."""
    components: Dict[PowerComponent, ComponentPowerConfig] = field(default_factory=dict)
    base_node_power_w: float = 0.0  # Base power draw (fans, VRMs, etc.)

    @classmethod
    def from_hardware_config(cls, hw_config: dict, num_accel: int = 1) -> 'PowerConfig':
        """Create PowerConfig from HARDWARE_CONFIGS entry.

        Derives power breakdown from TDP:
        - Active = TDP
        - Idle = 30% of TDP
        - Standby = 50% of TDP

        Handles TDP stored under ``config['cost']['tdp_watts']`` (data-center GPUs)
        or directly at ``config['tdp_watts']`` (consumer GPUs).
        """
        tdp = hw_config.get('cost', {}).get('tdp_watts')
        if tdp is None:
            tdp = hw_config.get('tdp_watts', 300)

        components = {
            PowerComponent.ACCELERATOR: ComponentPowerConfig(
                component=PowerComponent.ACCELERATOR,
                idle_power_w=tdp * 0.30,
                active_power_w=tdp,
                standby_power_w=tdp * 0.50,
                count=num_accel,
            ),
            PowerComponent.DRAM: ComponentPowerConfig(
                component=PowerComponent.DRAM,
                idle_power_w=5.0,
                active_power_w=15.0,
                energy_per_bit_pj=3.7,  # HBM2e typical
                count=1,
            ),
            PowerComponent.INTERCONNECT: ComponentPowerConfig(
                component=PowerComponent.INTERCONNECT,
                idle_power_w=2.0,
                active_power_w=10.0,
                energy_per_bit_pj=5.0,  # NVLink typical
                count=1,
            ),
            PowerComponent.HOST_CPU: ComponentPowerConfig(
                component=PowerComponent.HOST_CPU,
                idle_power_w=30.0,
                active_power_w=80.0,
                count=1,
            ),
            PowerComponent.COOLING: ComponentPowerConfig(
                component=PowerComponent.COOLING,
                idle_power_w=20.0,
                active_power_w=50.0,
                count=1,
            ),
            PowerComponent.STORAGE: ComponentPowerConfig(
                component=PowerComponent.STORAGE,
                idle_power_w=5.0,
                active_power_w=10.0,
                count=1,
            ),
            PowerComponent.MISC: ComponentPowerConfig(
                component=PowerComponent.MISC,
                idle_power_w=10.0,
                active_power_w=15.0,
                count=1,
            ),
        }

        return cls(components=components, base_node_power_w=20.0)


class PowerModel:
    """7-component power model for LLM serving simulation.

    Tracks energy consumption across accelerators, DRAM, interconnect,
    host CPU, cooling, storage, and misc components.
    """

    def __init__(self, power_config: PowerConfig):
        self._config = power_config
        # Cumulative energy in joules per component
        self._energy_j: Dict[PowerComponent, float] = {
            comp: 0.0 for comp in PowerComponent
        }

    def get_base_power_w(self) -> float:
        """Get the base idle power of the system (all components at idle)."""
        total = self._config.base_node_power_w
        for comp, cfg in self._config.components.items():
            total += cfg.idle_power_w * cfg.count
        return total

    def add_accelerator_active_energy(self, latency_ns: float, num: int = 1) -> float:
        """Add energy from active accelerator computation. Returns energy in joules."""
        cfg = self._config.components.get(PowerComponent.ACCELERATOR)
        if not cfg:
            return 0.0
        duration_s = latency_ns / NS_PER_S
        energy = cfg.active_power_w * num * duration_s
        self._energy_j[PowerComponent.ACCELERATOR] += energy
        return energy

    def add_accelerator_standby_energy(
        self, current_ns: float, last_end_ns: float, last_calc_ns: float
    ) -> float:
        """Add standby energy for idle accelerator time between batches."""
        cfg = self._config.components.get(PowerComponent.ACCELERATOR)
        if not cfg:
            return 0.0
        idle_ns = current_ns - max(last_end_ns, last_calc_ns)
        if idle_ns <= 0:
            return 0.0
        duration_s = idle_ns / NS_PER_S
        energy = cfg.standby_power_w * duration_s
        self._energy_j[PowerComponent.ACCELERATOR] += energy
        return energy

    def add_dram_energy(self, data_bytes: float) -> float:
        """Add energy for DRAM access. Returns energy in joules."""
        cfg = self._config.components.get(PowerComponent.DRAM)
        if not cfg:
            return 0.0
        data_bits = data_bytes * 8
        energy_pj = data_bits * cfg.energy_per_bit_pj
        energy_j = energy_pj * 1e-12
        self._energy_j[PowerComponent.DRAM] += energy_j
        return energy_j

    def add_interconnect_energy(self, data_bytes: float) -> float:
        """Add energy for interconnect data transfer. Returns energy in joules."""
        cfg = self._config.components.get(PowerComponent.INTERCONNECT)
        if not cfg:
            return 0.0
        data_bits = data_bytes * 8
        energy_pj = data_bits * cfg.energy_per_bit_pj
        energy_j = energy_pj * 1e-12
        self._energy_j[PowerComponent.INTERCONNECT] += energy_j
        return energy_j

    def estimate_from_simulation_result(
        self,
        latency_ms: float,
        compute_util: float = 0.0,
        mem_util: float = 0.0,
        comm_util: float = 0.0,
        data_read_bytes: float = 0,
        data_comm_bytes: float = 0,
        num_accel: int = 1,
    ) -> Dict[str, Any]:
        """Estimate power from high-level simulation results (analytical mode).

        Uses utilization to interpolate between idle and active power.
        """
        duration_s = latency_ms / 1000.0
        result: Dict[str, Any] = {}
        total_energy = 0.0

        accel_cfg = self._config.components.get(PowerComponent.ACCELERATOR)
        if accel_cfg:
            accel_power = (
                accel_cfg.idle_power_w
                + (accel_cfg.active_power_w - accel_cfg.idle_power_w) * compute_util
            )
            accel_energy = accel_power * num_accel * duration_s
            self._energy_j[PowerComponent.ACCELERATOR] += accel_energy
            result["accelerator_w"] = accel_power * num_accel
            result["accelerator_j"] = accel_energy
            total_energy += accel_energy

        # DRAM energy from data reads
        if data_read_bytes > 0:
            dram_e = self.add_dram_energy(data_read_bytes)
            total_energy += dram_e
        dram_cfg = self._config.components.get(PowerComponent.DRAM)
        if dram_cfg:
            dram_power = (
                dram_cfg.idle_power_w
                + (dram_cfg.active_power_w - dram_cfg.idle_power_w) * mem_util
            )
            dram_duration_e = dram_power * duration_s
            self._energy_j[PowerComponent.DRAM] += dram_duration_e
            total_energy += dram_duration_e
            result["dram_w"] = dram_power

        # Interconnect energy
        if data_comm_bytes > 0:
            comm_e = self.add_interconnect_energy(data_comm_bytes)
            total_energy += comm_e
        icn_cfg = self._config.components.get(PowerComponent.INTERCONNECT)
        if icn_cfg:
            icn_power = (
                icn_cfg.idle_power_w
                + (icn_cfg.active_power_w - icn_cfg.idle_power_w) * comm_util
            )
            icn_duration_e = icn_power * duration_s
            self._energy_j[PowerComponent.INTERCONNECT] += icn_duration_e
            total_energy += icn_duration_e
            result["interconnect_w"] = icn_power

        # Other components at idle-ish
        for comp in [
            PowerComponent.HOST_CPU,
            PowerComponent.COOLING,
            PowerComponent.STORAGE,
            PowerComponent.MISC,
        ]:
            cfg = self._config.components.get(comp)
            if cfg:
                comp_energy = cfg.idle_power_w * cfg.count * duration_s
                self._energy_j[comp] += comp_energy
                total_energy += comp_energy

        # Base node power
        base_energy = self._config.base_node_power_w * duration_s
        total_energy += base_energy

        result["total_energy_j"] = total_energy
        result["avg_power_w"] = total_energy / duration_s if duration_s > 0 else 0.0
        return result

    def energy_per_token(self, total_tokens: int, duration_ns: float = 0) -> float:
        """Energy per token in joules. Uses cumulative tracked energy."""
        if total_tokens == 0:
            return 0.0
        total = sum(self._energy_j.values())
        return total / total_tokens

    def power_breakdown(self, duration_ns: float) -> Dict[str, float]:
        """Average power per component over a duration. Returns watts."""
        duration_s = duration_ns / NS_PER_S
        if duration_s <= 0:
            return {}
        return {
            comp.value: energy / duration_s
            for comp, energy in self._energy_j.items()
            if energy > 0
        }

    def summary(self, duration_ns: float, total_tokens: int = 0) -> Dict[str, Any]:
        """Complete power/energy summary."""
        duration_s = duration_ns / NS_PER_S
        total_energy = sum(self._energy_j.values())
        return {
            "total_energy_j": total_energy,
            "total_energy_kwh": total_energy / 3_600_000,
            "avg_power_w": total_energy / duration_s if duration_s > 0 else 0.0,
            "duration_s": duration_s,
            "energy_per_token_j": self.energy_per_token(total_tokens) if total_tokens > 0 else None,
            "energy_per_token_mj": (self.energy_per_token(total_tokens) * 1000) if total_tokens > 0 else None,
            "breakdown_j": {comp.value: e for comp, e in self._energy_j.items() if e > 0},
            "breakdown_w": self.power_breakdown(duration_ns),
        }

    @classmethod
    def from_hardware_name(cls, name: str, num_accel: int = 1) -> 'PowerModel':
        """Create PowerModel from a hardware config name in HARDWARE_CONFIGS."""
        from llm_memory_calculator.hardware.configs import HARDWARE_CONFIGS
        if name not in HARDWARE_CONFIGS:
            raise ValueError(
                f"Unknown hardware: {name}. "
                f"Available: {list(HARDWARE_CONFIGS.keys())[:10]}..."
            )
        hw_config = HARDWARE_CONFIGS[name]
        power_config = PowerConfig.from_hardware_config(hw_config, num_accel)
        return cls(power_config)
