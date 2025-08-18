"""
Detailed CPU specifications for specific SKUs.

Each CPU SKU has different core counts, frequencies, and capabilities,
resulting in different FLOPS and performance characteristics.

FLOPS Calculation:
- Intel with AVX-512: cores × frequency × 16 (FP32 FLOPS/cycle per core)
- AMD with AVX-512: cores × frequency × 16 (FP32 FLOPS/cycle per core)
- For FP16/BF16: multiply by 2 (with appropriate instruction support)
"""

# Intel Xeon Scalable Processors (Sapphire Rapids - 4th Gen)
INTEL_XEON_SAPPHIRE_RAPIDS = {
    'XEON_PLATINUM_8490H': {
        'name': 'Intel Xeon Platinum 8490H',
        'Flops': 58.9,  # 60 cores × 1.9 GHz × 16 × 2 sockets / 1000 = 60.8 TFLOPS FP32
        'Memory_size': 4096,  # Max 4TB per socket, 8TB dual socket
        'Memory_BW': 307.2,  # 8 channels DDR5-4800
        'ICN': 128,  # UPI 16 GT/s × 4 links
        'Power': 350,  # TDP per socket
        'cores': 60,
        'base_freq_ghz': 1.9,
        'turbo_freq_ghz': 3.5,
        'real_values': True,
        'type': 'cpu',
        'manufacturer': 'Intel',
        'generation': 'Sapphire Rapids',
        'sku': '8490H'
    },
    'XEON_PLATINUM_8480PLUS': {
        'name': 'Intel Xeon Platinum 8480+',
        'Flops': 57.3,  # 56 cores × 2.0 GHz × 16 × 2 sockets / 1000 = 57.3 TFLOPS FP32
        'Memory_size': 4096,
        'Memory_BW': 307.2,
        'ICN': 128,
        'Power': 350,
        'cores': 56,
        'base_freq_ghz': 2.0,
        'turbo_freq_ghz': 3.8,
        'real_values': True,
        'type': 'cpu',
        'manufacturer': 'Intel',
        'generation': 'Sapphire Rapids',
        'sku': '8480+'
    },
    'XEON_PLATINUM_8470': {
        'name': 'Intel Xeon Platinum 8470',
        'Flops': 53.2,  # 52 cores × 2.0 GHz × 16 × 2 sockets / 1000
        'Memory_size': 4096,
        'Memory_BW': 307.2,
        'ICN': 128,
        'Power': 350,
        'cores': 52,
        'base_freq_ghz': 2.0,
        'turbo_freq_ghz': 3.8,
        'real_values': True,
        'type': 'cpu',
        'manufacturer': 'Intel',
        'generation': 'Sapphire Rapids',
        'sku': '8470'
    },
    'XEON_PLATINUM_8460YPLUS': {
        'name': 'Intel Xeon Platinum 8460Y+',
        'Flops': 44.8,  # 40 cores × 2.2 GHz × 16 × 2 sockets / 1000
        'Memory_size': 4096,
        'Memory_BW': 307.2,
        'ICN': 128,
        'Power': 300,
        'cores': 40,
        'base_freq_ghz': 2.2,
        'turbo_freq_ghz': 3.9,
        'real_values': True,
        'type': 'cpu',
        'manufacturer': 'Intel',
        'generation': 'Sapphire Rapids',
        'sku': '8460Y+'
    },
    'XEON_GOLD_6430': {
        'name': 'Intel Xeon Gold 6430',
        'Flops': 32.8,  # 32 cores × 2.1 GHz × 16 × 2 sockets / 1000
        'Memory_size': 4096,
        'Memory_BW': 307.2,
        'ICN': 128,
        'Power': 270,
        'cores': 32,
        'base_freq_ghz': 2.1,
        'turbo_freq_ghz': 3.4,
        'real_values': True,
        'type': 'cpu',
        'manufacturer': 'Intel',
        'generation': 'Sapphire Rapids',
        'sku': '6430'
    }
}

# Intel Xeon Scalable Processors (Emerald Rapids - 5th Gen)
INTEL_XEON_EMERALD_RAPIDS = {
    'XEON_PLATINUM_8592PLUS': {
        'name': 'Intel Xeon Platinum 8592+',
        'Flops': 76.8,  # 64 cores × 1.9 GHz × 16 × 2 sockets × 1.05 (IPC improvement) / 1000
        'Memory_size': 4096,
        'Memory_BW': 358.4,  # 8 channels DDR5-5600
        'ICN': 144,  # UPI 20 GT/s × 4 links
        'Power': 350,
        'cores': 64,
        'base_freq_ghz': 1.9,
        'turbo_freq_ghz': 3.9,
        'real_values': True,
        'type': 'cpu',
        'manufacturer': 'Intel',
        'generation': 'Emerald Rapids',
        'sku': '8592+'
    },
    'XEON_PLATINUM_8580': {
        'name': 'Intel Xeon Platinum 8580',
        'Flops': 61.4,  # 56 cores × 2.0 GHz × 16 × 2 sockets × 1.05 / 1000
        'Memory_size': 4096,
        'Memory_BW': 358.4,
        'ICN': 144,
        'Power': 350,
        'cores': 56,
        'base_freq_ghz': 2.0,
        'turbo_freq_ghz': 4.0,
        'real_values': True,
        'type': 'cpu',
        'manufacturer': 'Intel',
        'generation': 'Emerald Rapids',
        'sku': '8580'
    }
}

# AMD EPYC Processors (Milan/Milan-X - Zen 3)
AMD_EPYC_MILAN = {
    'EPYC_7763': {
        'name': 'AMD EPYC 7763',
        'Flops': 39.4,  # 64 cores × 2.45 GHz × 16 × 2 sockets / 1000
        'Memory_size': 4096,
        'Memory_BW': 204.8,  # 8 channels DDR4-3200
        'ICN': 128,  # Infinity Fabric
        'Power': 280,
        'cores': 64,
        'base_freq_ghz': 2.45,
        'boost_freq_ghz': 3.5,
        'real_values': True,
        'type': 'cpu',
        'manufacturer': 'AMD',
        'generation': 'Milan',
        'sku': '7763'
    },
    'EPYC_7773X': {
        'name': 'AMD EPYC 7773X (Milan-X)',
        'Flops': 40.0,  # 64 cores × 2.2 GHz × 16 × 2 sockets × 1.15 (V-Cache boost) / 1000
        'Memory_size': 4096,
        'Memory_BW': 204.8,
        'ICN': 128,
        'Power': 280,
        'cores': 64,
        'base_freq_ghz': 2.2,
        'boost_freq_ghz': 3.5,
        'l3_cache_mb': 768,  # 3D V-Cache
        'real_values': True,
        'type': 'cpu',
        'manufacturer': 'AMD',
        'generation': 'Milan-X',
        'sku': '7773X'
    },
    'EPYC_7713': {
        'name': 'AMD EPYC 7713',
        'Flops': 39.4,  # 64 cores × 2.0 GHz × 16 × 2 sockets / 1000
        'Memory_size': 4096,
        'Memory_BW': 204.8,
        'ICN': 128,
        'Power': 225,
        'cores': 64,
        'base_freq_ghz': 2.0,
        'boost_freq_ghz': 3.675,
        'real_values': True,
        'type': 'cpu',
        'manufacturer': 'AMD',
        'generation': 'Milan',
        'sku': '7713'
    }
}

# AMD EPYC Processors (Genoa - Zen 4)
AMD_EPYC_GENOA = {
    'EPYC_9654': {
        'name': 'AMD EPYC 9654',
        'Flops': 73.7,  # 96 cores × 2.4 GHz × 16 × 2 sockets / 1000
        'Memory_size': 6144,  # 12 channels DDR5, up to 6TB
        'Memory_BW': 460.8,  # 12 channels DDR5-4800
        'ICN': 144,  # Infinity Fabric 3.0
        'Power': 360,
        'cores': 96,
        'base_freq_ghz': 2.4,
        'boost_freq_ghz': 3.7,
        'real_values': True,
        'type': 'cpu',
        'manufacturer': 'AMD',
        'generation': 'Genoa',
        'sku': '9654'
    },
    'EPYC_9554': {
        'name': 'AMD EPYC 9554',
        'Flops': 63.5,  # 64 cores × 3.1 GHz × 16 × 2 sockets / 1000
        'Memory_size': 6144,
        'Memory_BW': 460.8,
        'ICN': 144,
        'Power': 360,
        'cores': 64,
        'base_freq_ghz': 3.1,
        'boost_freq_ghz': 3.75,
        'real_values': True,
        'type': 'cpu',
        'manufacturer': 'AMD',
        'generation': 'Genoa',
        'sku': '9554'
    },
    'EPYC_9454': {
        'name': 'AMD EPYC 9454',
        'Flops': 42.2,  # 48 cores × 2.75 GHz × 16 × 2 sockets / 1000
        'Memory_size': 6144,
        'Memory_BW': 460.8,
        'ICN': 144,
        'Power': 290,
        'cores': 48,
        'base_freq_ghz': 2.75,
        'boost_freq_ghz': 3.8,
        'real_values': True,
        'type': 'cpu',
        'manufacturer': 'AMD',
        'generation': 'Genoa',
        'sku': '9454'
    },
    'EPYC_9684X': {
        'name': 'AMD EPYC 9684X (Genoa-X)',
        'Flops': 78.6,  # 96 cores × 2.55 GHz × 16 × 2 sockets × 1.05 (V-Cache) / 1000
        'Memory_size': 6144,
        'Memory_BW': 460.8,
        'ICN': 144,
        'Power': 400,
        'cores': 96,
        'base_freq_ghz': 2.55,
        'boost_freq_ghz': 3.7,
        'l3_cache_mb': 1152,  # 3D V-Cache
        'real_values': True,
        'type': 'cpu',
        'manufacturer': 'AMD',
        'generation': 'Genoa-X',
        'sku': '9684X'
    }
}

# AMD EPYC Processors (Bergamo - Zen 4c)
AMD_EPYC_BERGAMO = {
    'EPYC_9754': {
        'name': 'AMD EPYC 9754',
        'Flops': 65.5,  # 128 cores × 2.25 GHz × 16 × 2 sockets × 0.9 (efficiency cores) / 1000
        'Memory_size': 6144,
        'Memory_BW': 460.8,
        'ICN': 144,
        'Power': 360,
        'cores': 128,
        'base_freq_ghz': 2.25,
        'boost_freq_ghz': 3.1,
        'real_values': True,
        'type': 'cpu',
        'manufacturer': 'AMD',
        'generation': 'Bergamo',
        'sku': '9754'
    },
    'EPYC_9734': {
        'name': 'AMD EPYC 9734',
        'Flops': 66.4,  # 112 cores × 2.2 GHz × 16 × 2 sockets × 0.9 / 1000
        'Memory_size': 6144,
        'Memory_BW': 460.8,
        'ICN': 144,
        'Power': 320,
        'cores': 112,
        'base_freq_ghz': 2.2,
        'boost_freq_ghz': 3.0,
        'real_values': True,
        'type': 'cpu',
        'manufacturer': 'AMD',
        'generation': 'Bergamo',
        'sku': '9734'
    }
}

# Combine all CPU configurations
CPU_CONFIGS = {
    **INTEL_XEON_SAPPHIRE_RAPIDS,
    **INTEL_XEON_EMERALD_RAPIDS,
    **AMD_EPYC_MILAN,
    **AMD_EPYC_GENOA,
    **AMD_EPYC_BERGAMO
}

# Mapping from generic names to specific SKUs (for backward compatibility)
GENERATION_TO_SKUS = {
    'SapphireRapids': ['XEON_PLATINUM_8490H', 'XEON_PLATINUM_8480PLUS', 'XEON_PLATINUM_8470'],
    'EmeraldRapids': ['XEON_PLATINUM_8592PLUS', 'XEON_PLATINUM_8580'],
    'Milan': ['EPYC_7763', 'EPYC_7713'],
    'MilanX': ['EPYC_7773X'],
    'Genoa': ['EPYC_9654', 'EPYC_9554', 'EPYC_9454'],
    'GenoaX': ['EPYC_9684X'],
    'Bergamo': ['EPYC_9754', 'EPYC_9734']
}