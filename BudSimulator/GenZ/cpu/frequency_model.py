from dataclasses import dataclass
import numpy as np
from .isa_model import ISAType


@dataclass
class ThermalState:
    temperature: float  # Celsius
    power_consumed: float  # Watts
    throttling_active: bool = False


class FrequencyGovernor:
    """Model CPU frequency scaling behavior"""
    
    def __init__(self, cpu_config):
        self.cpu_config = cpu_config
        self.thermal_state = ThermalState(temperature=50.0, power_consumed=100.0)
        
    def get_frequency(self, isa: ISAType, active_cores: int, 
                     workload_type: str = 'compute') -> float:
        """Get effective frequency for given conditions"""
        
        # Base frequency
        base_freq = self.cpu_config.base_frequency
        
        # Turbo boost based on active cores
        if active_cores in self.cpu_config.turbo_frequency_curve:
            turbo_freq = self.cpu_config.turbo_frequency_curve[active_cores]
        else:
            # Interpolate
            turbo_freq = self._interpolate_turbo(active_cores)
            
        # ISA-specific reduction
        isa_offset = 0
        if isa == ISAType.AVX512:
            isa_offset = self.cpu_config.avx_frequency_offset.get('avx512', -200e6)
        elif isa == ISAType.AMX:
            isa_offset = self.cpu_config.avx_frequency_offset.get('amx', -300e6)
        elif isa == ISAType.AVX2:
            isa_offset = self.cpu_config.avx_frequency_offset.get('avx2', 0)
            
        # Thermal throttling
        thermal_multiplier = self._get_thermal_multiplier()
        
        # Memory-bound workloads can maintain higher frequency
        workload_multiplier = 1.0 if workload_type == 'compute' else 1.1
        
        # Apply offset (negative values reduce frequency)
        effective_freq = (turbo_freq + isa_offset) * thermal_multiplier * workload_multiplier
        
        return max(effective_freq, base_freq * 0.8)  # Don't go below 80% base
        
    def _interpolate_turbo(self, active_cores: int) -> float:
        """Interpolate turbo frequency for core count"""
        curve = self.cpu_config.turbo_frequency_curve
        core_counts = sorted(curve.keys())
        
        # Find surrounding points
        for i, cores in enumerate(core_counts):
            if cores >= active_cores:
                if i == 0:
                    return curve[cores]
                else:
                    # Linear interpolation
                    c1, c2 = core_counts[i-1], cores
                    f1, f2 = curve[c1], curve[c2]
                    ratio = (active_cores - c1) / (c2 - c1)
                    return f1 + (f2 - f1) * ratio
                    
        return curve[core_counts[-1]]
        
    def _get_thermal_multiplier(self) -> float:
        """Calculate frequency reduction due to thermal constraints"""
        temp = self.thermal_state.temperature
        
        if temp < 70:
            return 1.0
        elif temp < 85:
            # Linear reduction 
            return 1.0 - (temp - 70) / 15 * 0.2
        else:
            # Aggressive throttling
            return 0.6
            
    def update_thermal_state(self, power: float, duration: float):
        """Update thermal model based on power consumption"""
        # Simplified thermal model
        thermal_resistance = 0.3  # C/W
        thermal_capacitance = 100  # J/C
        
        # Temperature rise
        temp_rise = power * thermal_resistance
        
        # Exponential rise to steady state
        tau = thermal_resistance * thermal_capacitance
        self.thermal_state.temperature += (
            temp_rise * (1 - np.exp(-duration / tau))
        )
        
        self.thermal_state.power_consumed = power
        self.thermal_state.throttling_active = self.thermal_state.temperature > 85 