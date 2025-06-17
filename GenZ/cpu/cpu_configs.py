import numpy as np
from .cpu_system import CPUConfig, CacheConfig


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
            microarchitecture='icelake'
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
            microarchitecture='zen3'
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
            microarchitecture='neoverse-v1'
        )
    },
    
    # Example: Adding a new CPU - Intel Xeon 6430
    'intel_xeon_6430': {
        'base_params': {
            'unit': None,
            'frequency': 2.1e9,
            'bits': 'bf16',
            'compute_efficiency': 0.85,
            'memory_efficiency': 0.75,
        },
        'cpu_specific': CPUConfig(
            # Core configuration
            cores_per_socket=32,
            sockets=2,
            threads_per_core=2,  # SMT/Hyper-Threading
            
            # Cache hierarchy (per core)
            l1i_config=CacheConfig(
                size=32*1024,      # 32KB L1 instruction cache
                latency=4,         # 4 cycles
                bandwidth=3200,    # GB/s
                associativity=8,
                is_shared=False
            ),
            l1d_config=CacheConfig(
                size=48*1024,      # 48KB L1 data cache
                latency=5,         # 5 cycles
                bandwidth=3200,    # GB/s
                associativity=12,
                is_shared=False
            ),
            l2_config=CacheConfig(
                size=2048*1024,    # 2MB L2 cache per core
                latency=14,        # 14 cycles
                bandwidth=1600,    # GB/s
                associativity=16,
                is_shared=False
            ),
            l3_config=CacheConfig(
                size=60*1024*1024, # 60MB L3 cache shared
                latency=42,        # 42 cycles
                bandwidth=800,     # GB/s
                associativity=12,
                is_shared=True,
                is_inclusive=True
            ),
            
            # NUMA configuration
            numa_nodes=2,          # 2 NUMA nodes (one per socket)
            cores_per_numa=32,     # 32 cores per NUMA node
            numa_distance_matrix=np.array([[10, 21], [21, 10]]),
            
            # ISA support
            isa_support=['amx', 'avx512', 'avx2', 'sse'],
            
            # Frequency configuration
            base_frequency=2.1e9,
            turbo_frequency_curve={
                1: 3.7e9,   # Single core turbo
                2: 3.7e9,   # 2 cores
                4: 3.5e9,   # 4 cores
                8: 3.3e9,   # 8 cores
                16: 3.1e9,  # 16 cores
                32: 2.9e9,  # 32 cores (all cores in one socket)
                64: 2.7e9   # All cores
            },
            avx_frequency_offset={
                'avx2': 0,          # No reduction for AVX2
                'avx512': -200e6,   # 200MHz reduction for AVX-512
                'amx': -400e6       # 400MHz reduction for AMX
            },
            
            # Memory configuration
            dram_bandwidth_per_channel=25.6,  # GB/s per channel
            memory_channels_per_socket=8,     # 8 memory channels
            
            # Vendor information
            vendor='intel',
            microarchitecture='sapphire_rapids'
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
        microarchitecture = 'emeraldrapids'
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