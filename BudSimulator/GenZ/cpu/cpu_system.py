from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class CacheConfig:
    size: int  # bytes
    latency: int  # cycles
    bandwidth: float  # GB/s
    associativity: int = 8
    line_size: int = 64
    is_shared: bool = False
    is_inclusive: bool = True


@dataclass 
class CPUConfig:
    # Core configuration
    cores_per_socket: int
    sockets: int
    threads_per_core: int
    
    # Cache hierarchy
    l1i_config: CacheConfig
    l1d_config: CacheConfig
    l2_config: CacheConfig
    l3_config: CacheConfig
    
    # NUMA configuration
    numa_nodes: int
    cores_per_numa: int
    numa_distance_matrix: np.ndarray
    
    # ISA capabilities
    isa_support: List[str]  # ['amx', 'avx512', 'avx2', 'sse']
    
    # Frequency
    base_frequency: float
    turbo_frequency_curve: Dict[int, float]  # active_cores -> frequency
    avx_frequency_offset: Dict[str, float]  # isa -> frequency_reduction
    
    # Memory
    dram_bandwidth_per_channel: float
    memory_channels_per_socket: int
    
    # Vendor specific
    vendor: str  # 'intel', 'amd', 'arm'
    microarchitecture: str  # 'icelake', 'zen3', 'neoverse-n1'

    # Calibration factors
    framework_overhead_ms: float = 0.0


class CPUSystem:
    """CPU-aware system model that extends base System"""
    
    def __init__(self, base_system, cpu_config: CPUConfig):
        self.base_system = base_system
        self.cpu_config = cpu_config
        self.config = cpu_config  # Add alias for compatibility
        
        # Override base system parameters with CPU-specific values
        self._setup_cpu_parameters()
        
        # Initialize CPU-specific components
        self.cache_hierarchy = self._init_cache_hierarchy()
        self.numa_topology = self._init_numa_topology()
        self.isa_selector = self._init_isa_selector()
        self.frequency_governor = self._init_frequency_governor()
        self.threading_model = self._init_threading_model()
        
    def _setup_cpu_parameters(self):
        """Override base system params with CPU-specific values"""
        # Total FLOPS depends on ISA and frequency
        max_vector_width = self._get_max_vector_width()
        total_cores = self.cpu_config.cores_per_socket * self.cpu_config.sockets
        
        # Peak FLOPS calculation
        if 'amx' in self.cpu_config.isa_support:
            # AMX provides massive INT8/BF16 throughput
            self.peak_int8_tops = total_cores * 1024 * self.cpu_config.base_frequency / 1e12
        if 'avx512' in self.cpu_config.isa_support:
            self.peak_fp32_tflops = total_cores * 16 * 2 * self.cpu_config.base_frequency / 1e12
        
        # Memory bandwidth is NUMA-aware
        self.peak_memory_bandwidth = (
            self.cpu_config.dram_bandwidth_per_channel * 
            self.cpu_config.memory_channels_per_socket * 
            self.cpu_config.sockets
        )
        
    def _get_max_vector_width(self):
        """Get maximum vector width based on ISA support"""
        if 'avx512' in self.cpu_config.isa_support:
            return 16  # 16 FP32 elements
        elif 'avx2' in self.cpu_config.isa_support:
            return 8   # 8 FP32 elements
        elif 'sve2' in self.cpu_config.isa_support:
            return 16  # Scalable, but typically 512-bit
        elif 'neon' in self.cpu_config.isa_support:
            return 4   # 128-bit NEON
        else:
            return 1   # Scalar
            
    def _init_cache_hierarchy(self):
        """Initialize cache hierarchy model"""
        from .cache_model import CacheHierarchy
        return CacheHierarchy(self.cpu_config)
        
    def _init_numa_topology(self):
        """Initialize NUMA topology model"""
        from .numa_model import NUMATopology
        return NUMATopology(self.cpu_config)
        
    def _init_isa_selector(self):
        """Initialize ISA selector"""
        from .isa_model import ISASelector
        return ISASelector(self.cpu_config)
        
    def _init_frequency_governor(self):
        """Initialize frequency governor"""
        from .frequency_model import FrequencyGovernor
        return FrequencyGovernor(self.cpu_config)
        
    def _init_threading_model(self):
        """Initialize threading model"""
        from .threading_model import ThreadingModel
        return ThreadingModel(self.cpu_config, self.numa_topology)
        
    def get_active_cores(self) -> int:
        """Get number of active cores (placeholder)"""
        return self.cpu_config.cores_per_socket * self.cpu_config.sockets
        
    def get_effective_bandwidth(self, data_location: str, accessing_core: int, batch_size: int = 1) -> float:
        """Get effective bandwidth considering cache level, NUMA, and batch contention"""
        if data_location == 'L1':
            return self.cpu_config.l1d_config.bandwidth
        elif data_location == 'L2':
            return self.cpu_config.l2_config.bandwidth
        elif data_location == 'L3':
            # L3 bandwidth shared among cores and affected by batch contention
            active_cores = self.get_active_cores()
            # Batch contention: higher batch sizes create more memory pressure
            batch_contention = max(1.0, np.sqrt(batch_size / 10))  # Contention grows with batch size
            return self.cpu_config.l3_config.bandwidth / (np.sqrt(active_cores) * batch_contention)
        else:  # DRAM
            numa_penalty = self.numa_topology.get_access_penalty(
                accessing_core, data_location
            ) if self.numa_topology else 1.0
            # High batch sizes saturate memory bandwidth more quickly
            batch_contention = max(1.0, batch_size / 20)  # Linear scaling for DRAM contention
            return self.peak_memory_bandwidth / (numa_penalty * batch_contention)
            
    def get_effective_flops(self, operation_type: str, data_type: str, 
                          isa: str, active_cores: int) -> float:
        """Get effective FLOPS for given operation"""
        # Base FLOPS for the ISA
        base_flops = self.isa_selector.get_peak_performance(isa, data_type) if self.isa_selector else 16 * 2
        
        # Frequency scaling
        frequency = self.frequency_governor.get_frequency(isa, active_cores) if self.frequency_governor else self.cpu_config.base_frequency
        
        # Efficiency factors
        vectorization_efficiency = self._get_vectorization_efficiency(
            operation_type, isa
        )
        
        return base_flops * frequency * active_cores * vectorization_efficiency
        
    def _get_vectorization_efficiency(self, operation_type: str, isa: str) -> float:
        """Get vectorization efficiency for operation type"""
        # Placeholder implementation
        if operation_type == 'GEMM':
            return 0.85 if isa in ['avx512', 'amx'] else 0.7
        else:
            return 0.6
            
    # Delegation methods to make CPUSystem compatible with System interface
    def __getattr__(self, name):
        """Delegate unknown attributes to base_system"""
        return getattr(self.base_system, name)
        
    def get_bit_multiplier(self, type='M', data='w', **kwargs):
        """Delegate to base system"""
        return self.base_system.get_bit_multiplier(type, data, **kwargs)
        
    def get_off_chip_mem_size(self):
        """Delegate to base system"""
        return self.base_system.get_off_chip_mem_size()
        
    def get_external_mem_bw(self):
        """Delegate to base system"""
        return self.base_system.get_external_mem_bw()
        
    def set_offchip_mem_bw(self, bw):
        """Delegate to base system"""
        return self.base_system.set_offchip_mem_bw(bw) 