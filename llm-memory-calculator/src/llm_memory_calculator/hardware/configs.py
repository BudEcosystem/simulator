"""
Static hardware configurations for LLM Memory Calculator.

This module contains predefined hardware configurations including GPUs, CPUs, TPUs, and ASICs.
Configurations are merged from multiple sources to provide a comprehensive hardware library.
"""

from typing import Any, Dict
from llm_memory_calculator.hardware.cpu_specs import CPU_CONFIGS

HARDWARE_CONFIGS: Dict[str, Dict[str, Any]] = {
    # NVIDIA GPUs
    'A100_40GB_GPU': {
        'name': 'A100_40GB_GPU',
        'Flops': 312,
        'Memory_size': 40,
        'Memory_BW': 1600,
        'ICN': 150,
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'NVIDIA',
        'pci_ids': ['20b0', '20f1'],  # PCIe and SXM4 variants
        'aliases': ['A100', 'TESLA A100', 'A100-SXM4-40GB', 'A100-PCIE-40GB', 'NVIDIA A100 40GB']
    },
    'A100_80GB_GPU': {
        'Flops': 312,
        'Memory_size': 80,
        'Memory_BW': 2039,
        'ICN': 150,
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'NVIDIA',
        'pci_ids': ['20b2', '20b5'],  # SXM4 and PCIe variants
        'aliases': ['A100 80GB', 'TESLA A100 80GB', 'A100-SXM4-80GB', 'A100-PCIE-80GB', 'NVIDIA A100 80GB']
    },
    'H100_GPU': {
        'Flops': 989,
        'Memory_size': 80,
        'Memory_BW': 3400,
        'ICN': 450,
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'NVIDIA',
        'pci_ids': ['2330', '2331', '2339'],  # SXM5, PCIe, NVL variants
        'aliases': ['H100', 'HOPPER', 'H100-SXM5', 'H100-PCIE', 'H100-NVL', 'NVIDIA H100']
    },
    'GH200_GPU': {
        'Flops': 1979,
        'Memory_size': 144,
        'Memory_BW': 4900,
        'ICN': 450,
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'NVIDIA',
        'pci_ids': ['233a'],  # GH200
        'aliases': ['GH200', 'GRACE HOPPER', 'GH200-144GB', 'NVIDIA GH200']
    },
    'B100': {
        'Flops': 3500,
        'Memory_size': 192,
        'Memory_BW': 8000,
        'ICN': 900,
        'ICN_LL': 0.25,
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'NVIDIA'
    },
    'GB200': {
        'Flops': 4500,
        'Memory_size': 192,
        'Memory_BW': 8000,
        'ICN': 900,
        'ICN_LL': 0.25,
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'NVIDIA'
    },
    
    # Google TPUs
    'TPUv6': {
        'Flops': 926,
        'Memory_size': 32,
        'Memory_BW': 1640,
        'ICN': 100,
        'real_values': True,
        'type': 'asic',
        'manufacturer': 'Google'
    },
    'TPUv5e': {
        'Flops': 197,
        'Memory_size': 16,
        'Memory_BW': 820,
        'ICN': 50,
        'real_values': True,
        'type': 'asic',
        'manufacturer': 'Google'
    },
    'TPUv5p': {
        'Flops': 459,
        'Memory_size': 95,
        'Memory_BW': 2765,
        'ICN': 450,
        'real_values': True,
        'type': 'asic',
        'manufacturer': 'Google'
    },
    'TPUv4': {
        'Flops': 275,
        'Memory_size': 32,
        'Memory_BW': 1228,
        'ICN': 24,
        'real_values': True,
        'type': 'asic',
        'manufacturer': 'Google'
    },
    
    # AMD GPUs
    'MI300X': {
        'Flops': 1307,
        'Memory_size': 192,
        'Memory_BW': 5300,
        'ICN': 400,
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'AMD'
    },
    'MI325X': {
        'Flops': 1307,
        'Memory_size': 256,
        'Memory_BW': 6000,
        'ICN': 400,
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'AMD'
    },
    
    # Intel Accelerators
    'Gaudi3': {
        'Flops': 1835,
        'Memory_size': 128,
        'Memory_BW': 3675,
        'ICN': 300,
        'real_values': True,
        'type': 'accelerator',
        'manufacturer': 'Intel'
    },
    
    # Intel CPUs
    'SapphireRapids_CPU': {
        'Flops': 33,
        'Memory_size': 300,
        'Memory_BW': 180,
        'ICN': 100,
        'Power': 434,
        'real_values': False,
        'type': 'cpu',
        'manufacturer': 'Intel',
        'name': 'Intel 4th Gen Xeon Scalable (Sapphire Rapids)',
        'url': 'https://www.intel.com/content/www/us/en/products/docs/processors/xeon/4th-gen-xeon-scalable-processors.html',
        'description': 'Up to 56-core server CPU featuring Advanced Matrix Extensions (AMX) for accelerated BF16/INT8 AI operations.',
        'aliases': ['Sapphire Rapids', 'Xeon Sapphire Rapids', 'Intel Sapphire Rapids', 'Xeon Platinum 8480+']
    },
    'EmeraldRapids_CPU': {
        'Flops': 47,
        'Memory_size': 300,
        'Memory_BW': 350,
        'ICN': 125,
        'Power': 289,
        'real_values': False,
        'type': 'cpu',
        'manufacturer': 'Intel',
        'name': 'Intel 5th Gen Xeon Scalable (Emerald Rapids)',
        'url': 'https://www.intel.com/content/www/us/en/newsroom/news/future-xeon-emerald-rapids.html',
        'description': 'Successor to Sapphire Rapids with more cores/cache, improved DDR5 speeds, and faster UPI links.',
        'aliases': ['Emerald Rapids', 'Xeon Emerald Rapids', 'Intel Emerald Rapids', 'Xeon Platinum 8580']
    },
    'GraniteRapids_CPU': {
        'Flops': 86,
        'Memory_size': 300,
        'Memory_BW': 500,
        'ICN': 175,
        'Power': 450,
        'real_values': False,
        'type': 'cpu',
        'manufacturer': 'Intel',
        'name': 'Intel Granite Rapids (Next-Gen Xeon)',
        'url': 'https://www.intel.com/content/www/us/en/newsroom/news/intel-advances-ai-everywhere.html',
        'description': 'Expected to offer significant enhancements in core count, memory bandwidth, and AI acceleration.'
    },
    'SierraForest_CPU': {
        'Flops': 47,
        'Memory_size': 300,
        'Memory_BW': 400,
        'ICN': 125,
        'Power': 256,
        'real_values': False,
        'type': 'cpu',
        'manufacturer': 'Intel',
        'name': 'Intel Sierra Forest (E-Core Xeon)',
        'url': 'https://www.intel.com/content/www/us/en/newsroom/news/intel-advances-ai-everywhere.html',
        'description': 'First Intel Xeon based on E-cores, targeting cloud-native workloads with high core density and energy efficiency.'
    },
    
    # AMD CPUs
    'MilanX_CPU': {
        'Flops': 36,
        'Memory_size': 512,
        'Memory_BW': 205,
        'ICN': 80,
        'Power': 225,
        'real_values': False,
        'type': 'cpu',
        'manufacturer': 'AMD',
        'name': 'AMD EPYC Milan-X (3rd Gen)',
        'url': 'https://www.amd.com/en/products/processors/server/epyc/3rd-generation.html',
        'description': 'Features 3D V-Cache technology providing up to 768MB L3 cache per CPU for data-intensive workloads.',
        'aliases': ['Milan-X', 'EPYC Milan-X', 'AMD Milan-X', 'AMD EPYC Milan-X', 'EPYC 7773X']
    },
    'Genoa_CPU': {
        'Flops': 60,
        'Memory_size': 300,
        'Memory_BW': 460,
        'ICN': 125,
        'Power': 360,
        'real_values': False,
        'type': 'cpu',
        'manufacturer': 'AMD',
        'name': 'AMD EPYC Genoa (4th Gen)',
        'url': 'https://www.amd.com/en/products/processors/server/epyc/9004-series.html',
        'description': 'Up to 96 Zen 4 cores with support for DDR5, PCIe Gen5, and CXL 1.1+ for enhanced performance.',
        'aliases': ['Genoa', 'EPYC Genoa', 'AMD Genoa', 'AMD EPYC Genoa', 'EPYC 9654']
    },
    'GenoaX_CPU': {
        'Flops': 60,
        'Memory_size': 512,
        'Memory_BW': 460,
        'ICN': 125,
        'Power': 380,
        'real_values': False,
        'type': 'cpu',
        'manufacturer': 'AMD',
        'name': 'AMD EPYC Genoa-X (4th Gen with 3D V-Cache)',
        'url': 'https://www.amd.com/en/products/processors/server/epyc/9004-series.html',
        'description': 'Genoa with 3D V-Cache technology, providing up to 1.2GB L3 cache for memory-intensive applications.'
    },
    'Bergamo_CPU': {
        'Flops': 45,
        'Memory_size': 300,
        'Memory_BW': 460,
        'ICN': 125,
        'Power': 360,
        'real_values': False,
        'type': 'cpu',
        'manufacturer': 'AMD',
        'name': 'AMD EPYC Bergamo (4th Gen Cloud Native)',
        'url': 'https://www.amd.com/en/products/processors/server/epyc/9004-series.html',
        'description': 'Up to 128 Zen 4c cores optimized for cloud-native workloads with high core density.'
    },
    'Turin_CPU': {
        'Flops': 98,
        'Memory_size': 300,
        'Memory_BW': 600,
        'ICN': 175,
        'Power': 426,
        'real_values': False,
        'type': 'cpu',
        'manufacturer': 'AMD',
        'name': 'AMD EPYC Turin (5th Gen)',
        'url': 'https://www.amd.com/en/products/processors/server/epyc.html',
        'description': 'Expected to feature Zen 5 cores with enhanced IPC, memory bandwidth, and AI acceleration capabilities.'
    },
    
    # ARM CPUs
    'Grace_CPU': {
        'Flops': 74,
        'Memory_size': 512,
        'Memory_BW': 500,
        'ICN': 200,
        'Power': 500,
        'real_values': False,
        'type': 'cpu',
        'manufacturer': 'NVIDIA',
        'name': 'NVIDIA Grace CPU',
        'url': 'https://www.nvidia.com/en-us/data-center/grace-cpu/',
        'description': 'ARM-based server CPU with 72 cores, designed for AI and HPC workloads with LPDDR5X memory.',
        'aliases': ['Grace', 'Grace CPU', 'NVIDIA Grace', 'NVIDIA Grace CPU']
    },
    'Graviton3_CPU': {
        'Flops': 20,
        'Memory_size': 300,
        'Memory_BW': 307,
        'ICN': 100,
        'Power': 100,
        'real_values': False,
        'type': 'cpu',
        'manufacturer': 'AWS',
        'name': 'AWS Graviton3',
        'url': 'https://aws.amazon.com/ec2/graviton/',
        'description': '64-core ARM Neoverse V1 CPU with DDR5 support, optimized for cloud workloads.',
        'aliases': ['Graviton3', 'Graviton 3', 'AWS Graviton3', 'AWS Graviton 3']
    },
    'Graviton4_CPU': {
        'Flops': 40,
        'Memory_size': 300,
        'Memory_BW': 450,
        'ICN': 150,
        'Power': 135,
        'real_values': False,
        'type': 'cpu',
        'manufacturer': 'AWS',
        'name': 'AWS Graviton4',
        'url': 'https://aws.amazon.com/ec2/graviton/',
        'description': 'Next-generation ARM CPU with improved performance, memory bandwidth, and AI inference capabilities.'
    },
    'AmpereOne_CPU': {
        'Flops': 23,
        'Memory_size': 300,
        'Memory_BW': 307,
        'ICN': 100,
        'Power': 350,
        'real_values': False,
        'type': 'cpu',
        'manufacturer': 'Ampere',
        'name': 'Ampere One',
        'url': 'https://amperecomputing.com/products/processors/ampere-one',
        'description': 'Up to 192-core ARM CPU designed for cloud-native applications with high single-thread performance.'
    },
    
    # Additional Accelerators
    'Trainium1': {
        'Flops': 190,
        'Memory_size': 32,
        'Memory_BW': 820,
        'ICN': 800,
        'real_values': False,
        'type': 'asic',
        'manufacturer': 'AWS',
        'name': 'AWS Trainium1',
        'url': 'https://aws.amazon.com/machine-learning/trainium/',
        'description': 'Purpose-built for training machine learning models with high performance and efficiency.'
    },
    'Inferentia2': {
        'Flops': 190,
        'Memory_size': 32,
        'Memory_BW': 820,
        'ICN': 800,
        'real_values': False,
        'type': 'asic',
        'manufacturer': 'AWS',
        'name': 'AWS Inferentia2',
        'url': 'https://aws.amazon.com/machine-learning/inferentia/',
        'description': 'Optimized for large-scale ML inference workloads with high throughput and low latency.'
    },
    'Cerebras_WSE2': {
        'Flops': 7500,
        'Memory_size': 40,
        'Memory_BW': 20000,
        'ICN': 2000,
        'Power': 15000,
        'real_values': False,
        'type': 'asic',
        'manufacturer': 'Cerebras',
        'name': 'Cerebras WSE-2',
        'url': 'https://www.cerebras.net/product-system/',
        'description': 'Wafer-scale processor with 850,000 cores and 40GB on-chip SRAM for massive parallel processing.'
    },
    'Cerebras_WSE3': {
        'Flops': 125000,
        'Memory_size': 44,
        'Memory_BW': 21000,
        'ICN': 7000,
        'Power': 23000,
        'real_values': False,
        'type': 'asic',
        'manufacturer': 'Cerebras',
        'name': 'Cerebras WSE-3',
        'url': 'https://www.cerebras.net/press-release/cerebras-announces-third-generation-wafer-scale-engine',
        'description': 'Third-generation wafer-scale processor with 900,000 cores at 5nm technology.'
    },
    'Groq_LPU': {
        'Flops': 250,
        'Memory_size': 230,
        'Memory_BW': 80000,
        'ICN': 400,
        'Power': 500,
        'real_values': False,
        'type': 'asic',
        'manufacturer': 'Groq',
        'name': 'Groq Language Processing Unit',
        'url': 'https://groq.com/',
        'description': 'Tensor Streaming Processor optimized for sequential processing and LLM inference.'
    },
    'SambaNova_SN40L': {
        'Flops': 688,
        'Memory_size': 1500,
        'Memory_BW': 1638,
        'ICN': 400,
        'Power': 1000,
        'real_values': False,
        'type': 'asic',
        'manufacturer': 'SambaNova',
        'name': 'SambaNova SN40L',
        'url': 'https://sambanova.ai/products/sn40l/',
        'description': 'Reconfigurable dataflow architecture for AI training and inference at scale.'
    },
    
    # Merge specific CPU SKUs
    **CPU_CONFIGS,
}

def get_hardware_names() -> list:
    """Get list of all hardware names."""
    return list(HARDWARE_CONFIGS.keys())

def get_hardware_by_manufacturer(manufacturer: str) -> Dict[str, Dict[str, Any]]:
    """Get all hardware from a specific manufacturer."""
    return {
        name: config 
        for name, config in HARDWARE_CONFIGS.items() 
        if config.get('manufacturer', '').lower() == manufacturer.lower()
    }

def get_hardware_by_type(hw_type: str) -> Dict[str, Dict[str, Any]]:
    """Get all hardware of a specific type (gpu, cpu, asic, accelerator)."""
    return {
        name: config 
        for name, config in HARDWARE_CONFIGS.items() 
        if config.get('type', '').lower() == hw_type.lower()
    }