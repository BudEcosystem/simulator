"""7-component power model for LLM serving simulation.

Tracks energy consumption across accelerators, DRAM, interconnect,
host CPU, cooling, storage, and misc components.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

from .constants import (
    PowerComponent, PowerState, NS_PER_S, NS_PER_MS,
    GPU_IDLE_FRACTION, GPU_ACTIVE_FRACTION, GPU_STANDBY_FRACTION,
    SERVER_POWER_FRACTION_GPU, SERVER_POWER_FRACTION_CPU,
    SERVER_POWER_FRACTION_DRAM, SERVER_POWER_FRACTION_NVLINK,
    SERVER_POWER_FRACTION_NIC, SERVER_POWER_FRACTION_COOLING,
    SERVER_POWER_FRACTION_SSD,
    DRAM_ENERGY_PJ_PER_BIT_HBM2E, DRAM_ACTIVITY_FACTOR,
    CPU_IDLE_FRACTION, CPU_ACTIVE_FRACTION, CPU_STANDBY_FRACTION,
    DRAM_ENERGY_PJ_PER_BIT_DDR5,
    SERVER_POWER_FRACTION_CPU_COMPUTE,
    SERVER_POWER_FRACTION_CPU_DRAM,
    SERVER_POWER_FRACTION_CPU_COOLING,
    SERVER_POWER_FRACTION_CPU_NIC,
    SERVER_POWER_FRACTION_CPU_SSD,
    SERVER_POWER_FRACTION_CPU_MISC,
)


@dataclass
class ComponentPowerConfig:
    """Power configuration for a single hardware component."""
    component: PowerComponent
    idle_power_w: float = 0.0
    active_power_w: float = 0.0
    standby_power_w: float = 0.0
    energy_per_bit_pj: float = 0.0  # picojoules per bit (for memory/interconnect)
    count: int = 1


# ---------------------------------------------------------------------------
# Per-device-class default TDP (watts), used ONLY when a device carries no
# datasheet TDP in any of its config fields (cost.tdp_watts / tdp_watts / Power).
# Each value is a representative datasheet TDP for that device class so that two
# distinct devices never collapse to one flat number. Keyed on
# (hardware ``type``, memory technology family).
#
# Sources (all vendor datasheets / published per-chip power):
#   HBM data-center accelerators (asic/accelerator): 350W
#     - AWS Trainium1 / Inferentia2 are 7nm HBM ASICs in the same class as
#       Google TPU v4 (192W, Jouppi et al. ISCA 2023) and below Gaudi2/3; AWS
#       does not publish a per-chip TDP, so we use the class-representative
#       350W (Trn1.32xl ~ 16 chips within the published instance envelope; well
#       under the ~500-700W Trainium2/3 figures, SemiAnalysis 2024).
#   HBM data-center GPU: 400W (NVIDIA A100 PCIe datasheet 300W .. SXM 400W).
#   GDDR data-center / pro GPU: 230W (NVIDIA A30 165W .. A40/A6000 300W class).
#   Other / unknown: 300W (final fallback, preserves legacy default).
# ---------------------------------------------------------------------------
_DEFAULT_TDP_HBM_ACCEL_W: float = 350.0   # AWS Trainium1/Inferentia2-class HBM ASIC
_DEFAULT_TDP_HBM_GPU_W: float = 400.0     # NVIDIA A100-class HBM GPU
_DEFAULT_TDP_GDDR_GPU_W: float = 230.0    # GDDR data-center/pro GPU class
_DEFAULT_TDP_GENERIC_W: float = 300.0     # legacy generic fallback


def _derive_class_tdp(hw_config: dict) -> float:
    """Derive a defensible per-device-class TDP (watts) for a device that lacks
    a datasheet TDP field. Never returns a single flat number for distinct
    device classes -- GPUs, HBM ASICs, and GDDR cards each get a class-specific
    datasheet-representative value. See module constants for sourcing."""
    hw_type = str(hw_config.get('type', '')).lower()
    mem_type = str(hw_config.get('memory_type', '')).lower()
    is_hbm = mem_type.startswith('hbm')
    if hw_type in ('asic', 'accelerator'):
        return _DEFAULT_TDP_HBM_ACCEL_W
    if hw_type == 'gpu':
        return _DEFAULT_TDP_HBM_GPU_W if is_hbm else _DEFAULT_TDP_GDDR_GPU_W
    return _DEFAULT_TDP_GENERIC_W


@dataclass
class PowerConfig:
    """Power configuration for an entire hardware node."""
    components: Dict[PowerComponent, ComponentPowerConfig] = field(default_factory=dict)
    base_node_power_w: float = 0.0  # Base power draw (fans, VRMs, etc.)

    @classmethod
    def from_hardware_config(cls, hw_config: dict, num_accel: int = 1) -> 'PowerConfig':
        """Create PowerConfig from a unified hardware-source entry.

        Power state breakdown is derived from the device TDP via the calibrated
        GPU fractions (idle/active/standby) in ``constants.py``.

        TDP resolution (R2-SV4/M14 fix): the device TDP is a DATASHEET figure
        resolved, in priority order, from
          1. ``config['cost']['tdp_watts']`` (data-center GPU format),
          2. ``config['tdp_watts']``        (consumer/static GPU format),
          3. ``config['Power']``            (DB / CPU-spec format).
        A device that carries none of these (e.g. a DB-only or undocumented
        accelerator) no longer collapses to a single flat 300W: its TDP is
        derived from its device class via :func:`_derive_class_tdp`, so an
        A100-class HBM GPU, a GDDR card, and an HBM ASIC report distinct power.
        Only a genuinely empty/typeless config hits the generic 300W fallback.
        """
        tdp = hw_config.get('cost', {}).get('tdp_watts')
        if tdp is None:
            tdp = hw_config.get('tdp_watts')
        if tdp is None:
            tdp = hw_config.get('Power')
        if tdp is None:
            tdp = _derive_class_tdp(hw_config)

        # Derive total system power from GPU TDP using physics-based fractions
        system_total = num_accel * tdp / SERVER_POWER_FRACTION_GPU

        components = {
            PowerComponent.ACCELERATOR: ComponentPowerConfig(
                component=PowerComponent.ACCELERATOR,
                idle_power_w=tdp * GPU_IDLE_FRACTION,
                active_power_w=tdp * GPU_ACTIVE_FRACTION,
                standby_power_w=tdp * GPU_STANDBY_FRACTION,
                count=num_accel,
            ),
            PowerComponent.HOST_CPU: ComponentPowerConfig(
                component=PowerComponent.HOST_CPU,
                idle_power_w=system_total * SERVER_POWER_FRACTION_CPU * 0.4,
                active_power_w=system_total * SERVER_POWER_FRACTION_CPU,
                count=1,
            ),
            PowerComponent.DRAM: ComponentPowerConfig(
                component=PowerComponent.DRAM,
                idle_power_w=system_total * SERVER_POWER_FRACTION_DRAM * 0.3,
                active_power_w=system_total * SERVER_POWER_FRACTION_DRAM,
                energy_per_bit_pj=DRAM_ENERGY_PJ_PER_BIT_HBM2E,
                count=1,
            ),
            PowerComponent.INTERCONNECT: ComponentPowerConfig(
                component=PowerComponent.INTERCONNECT,
                idle_power_w=system_total * SERVER_POWER_FRACTION_NVLINK * 0.2,
                active_power_w=system_total * SERVER_POWER_FRACTION_NVLINK,
                energy_per_bit_pj=5.0,
                count=1,
            ),
            PowerComponent.COOLING: ComponentPowerConfig(
                component=PowerComponent.COOLING,
                idle_power_w=system_total * SERVER_POWER_FRACTION_COOLING * 0.4,
                active_power_w=system_total * SERVER_POWER_FRACTION_COOLING,
                count=1,
            ),
            PowerComponent.STORAGE: ComponentPowerConfig(
                component=PowerComponent.STORAGE,
                idle_power_w=system_total * SERVER_POWER_FRACTION_SSD * 0.5,
                active_power_w=system_total * SERVER_POWER_FRACTION_SSD,
                count=1,
            ),
            PowerComponent.MISC: ComponentPowerConfig(
                component=PowerComponent.MISC,
                idle_power_w=system_total * SERVER_POWER_FRACTION_NIC * 0.3,
                active_power_w=system_total * SERVER_POWER_FRACTION_NIC,
                count=1,
            ),
        }

        return cls(components=components, base_node_power_w=0.0)

    @classmethod
    def from_cpu_config(cls, hw_config: dict, num_sockets: int = 1) -> 'PowerConfig':
        """Create PowerConfig for a CPU-based system.

        For CPU servers the CPU itself is the compute element (ACCELERATOR).
        There is no separate HOST_CPU component — its budget is folded into MISC.
        Uses DDR5 energy-per-bit and UPI/IF for interconnect.
        """
        tdp = hw_config.get('Power')
        if tdp is None:
            tdp = hw_config.get('cost', {}).get('tdp_watts', 300)

        # Derive total system power from CPU TDP
        system_total = num_sockets * tdp / SERVER_POWER_FRACTION_CPU_COMPUTE

        components = {
            PowerComponent.ACCELERATOR: ComponentPowerConfig(
                component=PowerComponent.ACCELERATOR,
                idle_power_w=tdp * CPU_IDLE_FRACTION,
                active_power_w=tdp * CPU_ACTIVE_FRACTION,
                standby_power_w=tdp * CPU_STANDBY_FRACTION,
                count=num_sockets,
            ),
            PowerComponent.DRAM: ComponentPowerConfig(
                component=PowerComponent.DRAM,
                idle_power_w=system_total * SERVER_POWER_FRACTION_CPU_DRAM * 0.3,
                active_power_w=system_total * SERVER_POWER_FRACTION_CPU_DRAM,
                energy_per_bit_pj=DRAM_ENERGY_PJ_PER_BIT_DDR5,
                count=1,
            ),
            PowerComponent.INTERCONNECT: ComponentPowerConfig(
                component=PowerComponent.INTERCONNECT,
                idle_power_w=system_total * SERVER_POWER_FRACTION_CPU_NIC * 0.2,
                active_power_w=system_total * SERVER_POWER_FRACTION_CPU_NIC,
                energy_per_bit_pj=8.0,  # UPI/Infinity Fabric pJ/bit
                count=1,
            ),
            PowerComponent.COOLING: ComponentPowerConfig(
                component=PowerComponent.COOLING,
                idle_power_w=system_total * SERVER_POWER_FRACTION_CPU_COOLING * 0.4,
                active_power_w=system_total * SERVER_POWER_FRACTION_CPU_COOLING,
                count=1,
            ),
            PowerComponent.STORAGE: ComponentPowerConfig(
                component=PowerComponent.STORAGE,
                idle_power_w=system_total * SERVER_POWER_FRACTION_CPU_SSD * 0.5,
                active_power_w=system_total * SERVER_POWER_FRACTION_CPU_SSD,
                count=1,
            ),
            PowerComponent.MISC: ComponentPowerConfig(
                component=PowerComponent.MISC,
                idle_power_w=system_total * SERVER_POWER_FRACTION_CPU_MISC * 0.3,
                active_power_w=system_total * SERVER_POWER_FRACTION_CPU_MISC,
                count=1,
            ),
        }

        return cls(components=components, base_node_power_w=0.0)


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
        energy_pj = data_bits * cfg.energy_per_bit_pj / DRAM_ACTIVITY_FACTOR
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
        pue: float = 1.0,
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

        # Other components with utilization-based power
        for comp in [PowerComponent.HOST_CPU, PowerComponent.COOLING,
                     PowerComponent.STORAGE, PowerComponent.MISC]:
            cfg = self._config.components.get(comp)
            if cfg:
                if comp == PowerComponent.HOST_CPU:
                    util = compute_util
                elif comp == PowerComponent.COOLING:
                    util = max(compute_util, mem_util, comm_util)
                else:
                    util = 0.1  # minimal background activity
                comp_power = cfg.idle_power_w + (cfg.active_power_w - cfg.idle_power_w) * util
                comp_energy = comp_power * cfg.count * duration_s
                self._energy_j[comp] += comp_energy
                total_energy += comp_energy

        # Base node power
        base_energy = self._config.base_node_power_w * duration_s
        total_energy += base_energy

        # Apply Power Usage Effectiveness (facility overhead)
        total_energy *= pue

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
        """Create PowerModel from a hardware name. F2/C5: resolve via the unified hardware source
        (static configs + DB) so every catalog device works, not only the ~73 static configs. Falls
        back to the static dict if the manager is unavailable."""
        from llm_memory_calculator.hardware.configs import HARDWARE_CONFIGS
        hw_config = None
        try:
            from llm_memory_calculator.hardware import get_hardware_config
            hw_config = get_hardware_config(name) or None
        except Exception:
            hw_config = None
        if hw_config is None:
            hw_config = HARDWARE_CONFIGS.get(name)
        if hw_config is None:
            raise ValueError(
                f"Unknown hardware: {name}. "
                f"Available: {list(HARDWARE_CONFIGS.keys())[:10]}..."
            )
        power_config = PowerConfig.from_hardware_config(hw_config, num_accel)
        return cls(power_config)

    @classmethod
    def from_cpu_hardware(cls, name: str, num_sockets: int = 1) -> 'PowerModel':
        """Create PowerModel for a CPU hardware entry.

        Looks up the hardware in HARDWARE_CONFIGS first, then falls back to
        CPU_PRESETS.  Uses :meth:`PowerConfig.from_cpu_config` which models
        the CPU as the accelerator with DDR5 memory and UPI interconnect.
        """
        from llm_memory_calculator.hardware.configs import HARDWARE_CONFIGS
        if name in HARDWARE_CONFIGS:
            hw_config = HARDWARE_CONFIGS[name]
        else:
            from llm_memory_calculator.genz.cpu.cpu_configs import CPU_PRESETS
            preset_key = next(
                (k for k in CPU_PRESETS if k.lower() == name.lower()), None,
            )
            if preset_key is None:
                raise ValueError(
                    f"Unknown CPU hardware: {name}. "
                    f"Not found in HARDWARE_CONFIGS or CPU_PRESETS."
                )
            preset = CPU_PRESETS[preset_key]
            # Synthesize a minimal hw_config dict from preset base_params
            bp = preset['base_params']
            cpu_cfg = preset.get('cpu_specific')
            tdp = 300  # default
            if cpu_cfg and hasattr(cpu_cfg, 'sockets'):
                num_sockets = cpu_cfg.sockets
            hw_config = {'Power': tdp, 'Memory_size': bp.get('off_chip_mem_size', 512 * 1024) / 1024}
        power_config = PowerConfig.from_cpu_config(hw_config, num_sockets)
        return cls(power_config)
