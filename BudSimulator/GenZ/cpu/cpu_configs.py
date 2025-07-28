import numpy as np
from .cpu_system import CPUConfig, CacheConfig
from dataclasses import dataclass
from typing import List, Dict


CPU_PRESETS = {
    'intel_xeon_8380': {
        'base_params': {
            'unit': None,
            'frequency': 2.3e9,
            'bits': 'bf16',
            'compute_efficiency': 0.85,
            'memory_efficiency': 0.75,
        },
        'cpu_specific': CPUConfig(
            # Core configuration
            cores_per_socket=40,
            sockets=2,
            threads_per_core=2,
            
            # Cache hierarchy (per core)
            l1i_config=CacheConfig(
                size=32*1024, latency=4, bandwidth=3200, 
                associativity=8, is_shared=False
            ),
            l1d_config=CacheConfig(
                size=48*1024, latency=5, bandwidth=3200,
                associativity=12, is_shared=False  
            ),
            l2_config=CacheConfig(
                size=1280*1024, latency=14, bandwidth=1600,
                associativity=20, is_shared=False
            ),
            l3_config=CacheConfig(
                size=60*1024*1024, latency=42, bandwidth=800,
                associativity=12, is_shared=True, is_inclusive=False
            ),
            
            # NUMA
            numa_nodes=2,
            cores_per_numa=40,
            numa_distance_matrix=np.array([[10, 21], [21, 10]]),
            
            # ISA
            isa_support=['amx', 'avx512', 'avx2', 'sse'],
            
            # Frequency  
            base_frequency=2.3e9,
            turbo_frequency_curve={
                1: 3.4e9, 2: 3.4e9, 4: 3.2e9, 8: 3.0e9,
                16: 2.8e9, 32: 2.6e9, 40: 2.4e9
            },
            avx_frequency_offset={'avx2': 0, 'avx512': -200e6, 'amx': -300e6},
            
            # Memory
            dram_bandwidth_per_channel=25.6,  # GB/s
            memory_channels_per_socket=8,
            
            # Vendor
            vendor='intel',
            microarchitecture='icelake',
            framework_overhead_ms=1250.0
        )
    },
    
    'amd_epyc_7763': {
        'base_params': {
            'unit': None, 
            'frequency': 2.45e9,
            'bits': 'bf16',
            'compute_efficiency': 0.85,
            'memory_efficiency': 0.75,
        },
        'cpu_specific': CPUConfig(
            cores_per_socket=64,
            sockets=2,
            threads_per_core=2,
            
            # AMD has different L3 architecture (CCX-based)
            l1i_config=CacheConfig(size=32*1024, latency=4, bandwidth=3200),
            l1d_config=CacheConfig(size=32*1024, latency=4, bandwidth=3200),
            l2_config=CacheConfig(size=512*1024, latency=12, bandwidth=1600),
            l3_config=CacheConfig(
                size=256*1024*1024, latency=46, bandwidth=900,
                is_shared=True  # Shared within CCX
            ),
            
            numa_nodes=8,  # 8 CCDs
            cores_per_numa=16,
            numa_distance_matrix=np.array([
                [10, 16, 16, 16, 32, 32, 32, 32],
                [16, 10, 16, 16, 32, 32, 32, 32],
                [16, 16, 10, 16, 32, 32, 32, 32],
                [16, 16, 16, 10, 32, 32, 32, 32],
                [32, 32, 32, 32, 10, 16, 16, 16],
                [32, 32, 32, 32, 16, 10, 16, 16],
                [32, 32, 32, 32, 16, 16, 10, 16],
                [32, 32, 32, 32, 16, 16, 16, 10]
            ]),
            
            isa_support=['avx2'],  # No AVX-512 on Zen3
            
            base_frequency=2.45e9,
            turbo_frequency_curve={1: 3.5e9, 8: 3.2e9, 16: 3.0e9, 64: 2.8e9},
            avx_frequency_offset={'avx2': 0},
            
            dram_bandwidth_per_channel=25.6,
            memory_channels_per_socket=8,
            
            vendor='amd',
            microarchitecture='zen3',
            framework_overhead_ms=1250.0
        )
    },
    
    'aws_graviton3': {
        # ARM-based configuration
        'base_params': {
            'unit': None,
            'frequency': 2.6e9, 
            'bits': 'bf16',
            'compute_efficiency': 0.85,
            'memory_efficiency': 0.75,
        },
        'cpu_specific': CPUConfig(
            cores_per_socket=64,
            sockets=1,
            threads_per_core=1,  # No SMT
            
            l1i_config=CacheConfig(size=64*1024, latency=4, bandwidth=3200),
            l1d_config=CacheConfig(size=64*1024, latency=4, bandwidth=3200),
            l2_config=CacheConfig(size=1024*1024, latency=13, bandwidth=1600),
            l3_config=CacheConfig(size=32*1024*1024, latency=35, bandwidth=800),
            
            numa_nodes=1,  # Single socket
            cores_per_numa=64,
            numa_distance_matrix=np.array([[10]]),
            
            isa_support=['sve2', 'neon'],  # ARM ISAs
            
            base_frequency=2.6e9,
            turbo_frequency_curve={1: 2.6e9, 64: 2.6e9},  # Fixed frequency
            avx_frequency_offset={},
            
            dram_bandwidth_per_channel=25.6,
            memory_channels_per_socket=8,
            
            vendor='arm', 
            microarchitecture='neoverse-v1',
            framework_overhead_ms=1250.0
        )
    },
    
    # Example: Adding a new CPU - Intel Xeon 6430
    'intel_xeon_6430': {
        'base_params': {
            'unit': None,
            'frequency': 2.6e9,
            'bits': 'bf16',
            'compute_efficiency': 0.85,
            'memory_efficiency': 0.75,
        },
        'cpu_specific': CPUConfig(
            # Core configuration
            cores_per_socket=32,    # 32 cores for Sapphire Rapids
            sockets=1,              # Single socket configuration
            threads_per_core=2,
            
            # Cache hierarchy (per core) - based on Sapphire Rapids architecture
            l1i_config=CacheConfig(
                size=32*1024,
                latency=4,
                bandwidth=2500,
                associativity=8,
                is_shared=False
            ),
            l1d_config=CacheConfig(
                size=48*1024,
                latency=5,
                bandwidth=2500,
                associativity=12,
                is_shared=False  
            ),
            l2_config=CacheConfig(
                size=2*1024*1024,        # 2MB L2 cache per core
                latency=17,              # Increased L2 latency
                bandwidth=1200,
                associativity=16,
                is_shared=False
            ),
            l3_config=CacheConfig(
                size=60*1024*1024,       # 60MB L3 cache (shared)
                latency=45,
                bandwidth=850,
                associativity=15,
                is_shared=True,
                is_inclusive=False
            ),
            
            # NUMA configuration
            numa_nodes=1,
            cores_per_numa=32,
            numa_distance_matrix=np.array([[10]]),
            
            # ISA support
            isa_support=['amx', 'avx512', 'avx2', 'sse'],
            
            # Frequency
            base_frequency=2.6e9,
            turbo_frequency_curve={
                1: 3.4e9, 2: 3.4e9, 4: 3.2e9, 8: 3.1e9,
                16: 3.0e9, 32: 2.8e9
            },
            avx_frequency_offset={'avx2': 0, 'avx512': -200e6, 'amx': -300e6},
            
            # Memory
            dram_bandwidth_per_channel=38.4,  # DDR5-4800
            memory_channels_per_socket=8,
            
            # Vendor information
            vendor='intel',
            microarchitecture='sapphire_rapids',
            framework_overhead_ms=1038.0
        )
    },
    'intel_xeon_8592_plus': {
    'base_params': {
        'unit': None,
        'frequency': 1.9e9,          # nominal/base clock (Hz)
        'bits': 'bf16',              # AMX BF16 native
        'compute_efficiency': 0.88,  # perf/W head-room over Ice Lake
        'memory_efficiency': 0.78,   # faster DDR5 helps a bit
    },
    'cpu_specific': CPUConfig(
        # Core configuration
        cores_per_socket=64,
        sockets=4,
        threads_per_core=2,

        # Cache hierarchy  (sizes are *per core* except the shared L3)
        l1i_config=CacheConfig(
            size=32*1024,  latency=4,  bandwidth=3400,
            associativity=8,  is_shared=False
        ),
        l1d_config=CacheConfig(
            size=48*1024,  latency=5,  bandwidth=3400,
            associativity=12, is_shared=False
        ),
        l2_config=CacheConfig(
            size=2048*1024, latency=13, bandwidth=1800,
            associativity=16, is_shared=False
        ),
        l3_config=CacheConfig(
            size=320*1024*1024, latency=40, bandwidth=900,
            associativity=16, is_shared=True,  is_inclusive=False
        ),

        # NUMA (two big tiles per package)
        numa_nodes=4,
        cores_per_numa=32,
        numa_distance_matrix=np.array([[10, 21],
                                       [21, 10]]),

        # ISA extensions
        isa_support=['amx', 'avx512', 'avx2'],

        # Frequency behaviour
        base_frequency       = 1.9e9,
        turbo_frequency_curve={
            1: 3.9e9,  2: 3.9e9,  4: 3.7e9,  8: 3.6e9,
            16: 3.4e9, 32: 3.1e9, 64: 2.9e9
        },
        avx_frequency_offset={'avx2': 0,
                              'avx512': -200e6,
                              'amx':   -300e6},

        # Memory subsystem
        dram_bandwidth_per_channel = 44.8,  # GB/s @ DDR5-5600
        memory_channels_per_socket = 8,

        # Metadata
        vendor            = 'intel',
        microarchitecture = 'emeraldrapids',

        # Calibration
        framework_overhead_ms=1250.0 # Default based on initial diagnosis
    )
},
    
    # Intel Xeon 6747P (Granite Rapids) - Latest generation server CPU
    'intel_xeon_6747p': {
        'base_params': {
            'unit': None,
            'frequency': 2.7e9,          # 2.7 GHz base frequency
            'bits': 'bf16',              # Native BF16 support with AMX
            'compute_efficiency': 0.90,  # Improved efficiency over previous generations
            'memory_efficiency': 0.82,   # Better memory efficiency with DDR5-6400
        },
        'cpu_specific': CPUConfig(
            # Core configuration - 48 cores, 96 threads
            cores_per_socket=48,
            sockets=2,                   # Can be used in dual-socket configuration
            threads_per_core=2,          # Intel Hyper-Threading
            
            # Cache hierarchy based on Granite Rapids specifications
            l1i_config=CacheConfig(
                size=32*1024,            # 32KB L1 instruction cache per core
                latency=4,               # Estimated 4 cycles
                bandwidth=3600,          # Higher bandwidth for latest gen
                associativity=8,
                is_shared=False
            ),
            l1d_config=CacheConfig(
                size=48*1024,            # 48KB L1 data cache per core  
                latency=5,               # Estimated 5 cycles
                bandwidth=3600,          # Higher bandwidth
                associativity=12,
                is_shared=False
            ),
            l2_config=CacheConfig(
                size=2048*1024,          # 2MB L2 cache per core (estimated)
                latency=12,              # Improved latency
                bandwidth=2000,          # GB/s
                associativity=16,
                is_shared=False
            ),
            l3_config=CacheConfig(
                size=288*1024*1024,      # 288MB L3 cache (shared across all cores)
                latency=38,              # Improved L3 latency for Granite Rapids
                bandwidth=1000,          # GB/s
                associativity=16,
                is_shared=True,
                is_inclusive=False       # Non-inclusive L3 for Intel
            ),
            
            # NUMA configuration - single socket initially, can be expanded to dual
            numa_nodes=2,                # 2 NUMA nodes per socket (typical for high-core count)
            cores_per_numa=24,           # 24 cores per NUMA node
            numa_distance_matrix=np.array([[10, 16], [16, 10]]),
            
            # ISA support - Granite Rapids has latest Intel features
            isa_support=['amx', 'avx512', 'avx2', 'sse'],
            
            # Frequency scaling with realistic turbo curve
            base_frequency=2.7e9,
            turbo_frequency_curve={
                1: 3.9e9,                # Single core max turbo
                2: 3.9e9,                # 2 cores
                4: 3.8e9,                # 4 cores  
                8: 3.7e9,                # 8 cores
                16: 3.5e9,               # 16 cores
                24: 3.3e9,               # 24 cores
                32: 3.1e9,               # 32 cores
                48: 2.9e9,               # All cores (single socket)
                96: 2.8e9                # All cores (dual socket)
            },
            avx_frequency_offset={
                'avx2': 0,               # No reduction for AVX2
                'avx512': -100e6,        # 100MHz reduction for AVX-512 (improved over older gens)
                'amx': -200e6            # 200MHz reduction for AMX
            },
            
            # Memory configuration - DDR5-6400 with 8-channel
            dram_bandwidth_per_channel=51.2,  # GB/s per channel (DDR5-6400 = 51.2 GB/s)
            memory_channels_per_socket=8,     # 8-channel memory interface
            
            # Vendor information
            vendor='intel',
            microarchitecture='granite_rapids',

            # Calibration
            framework_overhead_ms=1250.0 # Default based on initial diagnosis
        )
    }
}


def get_cpu_preset(cpu_model):
    """
    Get CPU configuration preset
    
    Args:
        cpu_model: One of 'intel_xeon_8380', 'amd_epyc_7763', 'aws_graviton3'
        
    Returns:
        CPUConfig: CPU configuration
    """
    if cpu_model not in CPU_PRESETS:
        raise ValueError(f"Unknown CPU model: {cpu_model}. Available: {list(CPU_PRESETS.keys())}")
    
    return CPU_PRESETS[cpu_model]['cpu_specific'] 