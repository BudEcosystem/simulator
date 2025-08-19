"""
GPU architecture specifications and detailed hardware information.

This module contains comprehensive GPU architecture and model specifications,
organized by vendor and architecture for precise hardware matching and
performance modeling.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class GPUArchitecture:
    """GPU architecture metadata."""
    name: str
    vendor: str
    release_year: int
    process_node: str
    description: str
    compute_focus: str  # 'gaming', 'datacenter', 'professional', 'mixed'


# NVIDIA GPU Architectures
NVIDIA_ARCHITECTURES = {
    'PASCAL': GPUArchitecture(
        name='Pascal',
        vendor='NVIDIA',
        release_year=2016,
        process_node='16nm/14nm',
        description='Gaming and datacenter architecture with first-gen Tensor operations',
        compute_focus='mixed'
    ),
    'VOLTA': GPUArchitecture(
        name='Volta',
        vendor='NVIDIA', 
        release_year=2017,
        process_node='12nm',
        description='Datacenter-focused with first dedicated Tensor cores',
        compute_focus='datacenter'
    ),
    'TURING': GPUArchitecture(
        name='Turing',
        vendor='NVIDIA',
        release_year=2018,
        process_node='12nm',
        description='Gaming architecture with RT cores and second-gen Tensor cores',
        compute_focus='gaming'
    ),
    'AMPERE': GPUArchitecture(
        name='Ampere',
        vendor='NVIDIA',
        release_year=2020,
        process_node='8nm/7nm',
        description='Unified gaming/datacenter with third-gen RT and Tensor cores',
        compute_focus='mixed'
    ),
    'ADA_LOVELACE': GPUArchitecture(
        name='Ada Lovelace',
        vendor='NVIDIA',
        release_year=2022,
        process_node='4nm',
        description='Gaming-focused with third-gen RT cores and AV1 encoding',
        compute_focus='gaming'
    ),
    'HOPPER': GPUArchitecture(
        name='Hopper',
        vendor='NVIDIA',
        release_year=2022,
        process_node='4nm',
        description='Datacenter architecture with Transformer Engine and fourth-gen Tensor cores',
        compute_focus='datacenter'
    ),
    'BLACKWELL': GPUArchitecture(
        name='Blackwell',
        vendor='NVIDIA',
        release_year=2024,
        process_node='4nm',
        description='Next-gen datacenter/gaming with 208B transistors and fifth-gen Tensor cores',
        compute_focus='mixed'
    )
}

# AMD GPU Architectures  
AMD_ARCHITECTURES = {
    'GCN': GPUArchitecture(
        name='GCN',
        vendor='AMD',
        release_year=2012,
        process_node='28nm-14nm',
        description='Graphics Core Next architecture spanning multiple generations',
        compute_focus='mixed'
    ),
    'RDNA': GPUArchitecture(
        name='RDNA',
        vendor='AMD',
        release_year=2019,
        process_node='7nm',
        description='Gaming-optimized architecture with improved performance per watt',
        compute_focus='gaming'
    ),
    'RDNA2': GPUArchitecture(
        name='RDNA 2',
        vendor='AMD',
        release_year=2020,
        process_node='7nm',
        description='Second-gen RDNA with ray tracing and variable rate shading',
        compute_focus='gaming'
    ),
    'RDNA3': GPUArchitecture(
        name='RDNA 3',
        vendor='AMD',
        release_year=2022,
        process_node='5nm',
        description='Chiplet-based gaming architecture with AI acceleration',
        compute_focus='gaming'
    ),
    'CDNA': GPUArchitecture(
        name='CDNA',
        vendor='AMD',
        release_year=2020,
        process_node='7nm',
        description='Compute-focused architecture for datacenter and HPC',
        compute_focus='datacenter'
    ),
    'CDNA2': GPUArchitecture(
        name='CDNA 2',
        vendor='AMD',
        release_year=2021,
        process_node='6nm',
        description='Enhanced compute architecture with matrix engines',
        compute_focus='datacenter'
    ),
    'CDNA3': GPUArchitecture(
        name='CDNA 3',
        vendor='AMD',
        release_year=2023,
        process_node='5nm',
        description='Third-gen compute architecture with unified memory and AI focus',
        compute_focus='datacenter'
    )
}

# Intel GPU Architectures
INTEL_ARCHITECTURES = {
    'XE_LP': GPUArchitecture(
        name='Xe-LP',
        vendor='Intel',
        release_year=2020,
        process_node='10nm',
        description='Low-power graphics architecture for integrated GPUs',
        compute_focus='gaming'
    ),
    'XE_HPG': GPUArchitecture(
        name='Xe-HPG',
        vendor='Intel',
        release_year=2022,
        process_node='6nm',
        description='High-performance gaming architecture (Arc series)',
        compute_focus='gaming'
    ),
    'XE_HPC': GPUArchitecture(
        name='Xe-HPC',
        vendor='Intel',
        release_year=2022,
        process_node='7nm',
        description='Datacenter-focused architecture (Ponte Vecchio)',
        compute_focus='datacenter'
    )
}

# Comprehensive PCI ID to Architecture and Model mapping
PCI_ARCHITECTURE_MAP = {
    # NVIDIA Pascal
    '10de:1b06': {'arch': 'PASCAL', 'model': 'P100', 'variant': 'PCIe'},
    '10de:15f8': {'arch': 'PASCAL', 'model': 'P100', 'variant': 'SXM2'},
    '10de:1b38': {'arch': 'PASCAL', 'model': 'P40', 'variant': 'PCIe'},
    '10de:1cb3': {'arch': 'PASCAL', 'model': 'P4', 'variant': 'PCIe'},
    '10de:1db1': {'arch': 'PASCAL', 'model': 'P100', 'variant': 'SXM2-16GB'},
    
    # NVIDIA Volta
    '10de:1db4': {'arch': 'VOLTA', 'model': 'V100', 'variant': 'PCIe-16GB'},
    '10de:1db5': {'arch': 'VOLTA', 'model': 'V100', 'variant': 'PCIe-32GB'},
    '10de:1db6': {'arch': 'VOLTA', 'model': 'V100', 'variant': 'SXM2-16GB'},
    '10de:1dbe': {'arch': 'VOLTA', 'model': 'V100', 'variant': 'SXM2-32GB'},
    '10de:1df6': {'arch': 'VOLTA', 'model': 'V100S', 'variant': 'PCIe-32GB'},
    
    # NVIDIA Turing
    '10de:1e04': {'arch': 'TURING', 'model': 'RTX8000', 'variant': 'Quadro'},
    '10de:1e30': {'arch': 'TURING', 'model': 'RTX6000', 'variant': 'Quadro'},
    '10de:1eb8': {'arch': 'TURING', 'model': 'T4', 'variant': 'Tesla'},
    '10de:2080': {'arch': 'TURING', 'model': 'RTX2080Ti', 'variant': 'GeForce'},
    '10de:1e87': {'arch': 'TURING', 'model': 'RTX2080', 'variant': 'GeForce'},
    
    # NVIDIA Ampere
    '10de:20b0': {'arch': 'AMPERE', 'model': 'A100', 'variant': 'PCIe-40GB'},
    '10de:20b2': {'arch': 'AMPERE', 'model': 'A100', 'variant': 'SXM4-80GB'},
    '10de:20b5': {'arch': 'AMPERE', 'model': 'A100', 'variant': 'PCIe-80GB'},
    '10de:20f1': {'arch': 'AMPERE', 'model': 'A100', 'variant': 'SXM4-40GB'},
    '10de:20f5': {'arch': 'AMPERE', 'model': 'A100', 'variant': 'SXM4-80GB-HBM2e'},
    '10de:2230': {'arch': 'AMPERE', 'model': 'RTXA6000', 'variant': 'Professional'},
    '10de:2204': {'arch': 'AMPERE', 'model': 'RTX3090', 'variant': 'GeForce'},
    '10de:220a': {'arch': 'AMPERE', 'model': 'RTX3080', 'variant': 'GeForce'},
    '10de:2216': {'arch': 'AMPERE', 'model': 'RTX3070', 'variant': 'GeForce'},
    
    # NVIDIA Ada Lovelace
    '10de:2684': {'arch': 'ADA_LOVELACE', 'model': 'RTX4090', 'variant': 'GeForce'},
    '10de:2757': {'arch': 'ADA_LOVELACE', 'model': 'RTX4090', 'variant': 'Mobile'},
    '10de:2782': {'arch': 'ADA_LOVELACE', 'model': 'RTX4080', 'variant': 'GeForce'},
    '10de:27b8': {'arch': 'ADA_LOVELACE', 'model': 'L4', 'variant': 'Tesla'},
    '10de:26b9': {'arch': 'ADA_LOVELACE', 'model': 'L40', 'variant': 'Professional'},
    '10de:26ba': {'arch': 'ADA_LOVELACE', 'model': 'L40S', 'variant': 'Professional'},
    
    # NVIDIA Hopper
    '10de:2330': {'arch': 'HOPPER', 'model': 'H100', 'variant': 'SXM5'},
    '10de:2331': {'arch': 'HOPPER', 'model': 'H100', 'variant': 'PCIe'},
    '10de:2339': {'arch': 'HOPPER', 'model': 'H100', 'variant': 'NVL'},
    '10de:233a': {'arch': 'HOPPER', 'model': 'GH200', 'variant': 'Grace-Hopper'},
    '10de:2335': {'arch': 'HOPPER', 'model': 'H200', 'variant': 'SXM'},
    '10de:2336': {'arch': 'HOPPER', 'model': 'H200', 'variant': 'NVL'},
    
    # NVIDIA Blackwell (preliminary - may need updates as more IDs are revealed)
    '10de:2900': {'arch': 'BLACKWELL', 'model': 'B200', 'variant': 'SXM'},
    '10de:2901': {'arch': 'BLACKWELL', 'model': 'B100', 'variant': 'SXM'},
    '10de:2902': {'arch': 'BLACKWELL', 'model': 'GB200', 'variant': 'Grace-Blackwell'},
    
    # AMD RDNA2
    '1002:73a5': {'arch': 'RDNA2', 'model': 'RX6900XT', 'variant': 'Navi21'},
    '1002:73a8': {'arch': 'RDNA2', 'model': 'RX6900XT', 'variant': 'Navi21-GL'},
    '1002:73a9': {'arch': 'RDNA2', 'model': 'RX6950XT', 'variant': 'Navi21-KXT'},
    '1002:73ac': {'arch': 'RDNA2', 'model': 'RX6800XT', 'variant': 'Navi21-XT'},
    '1002:73ad': {'arch': 'RDNA2', 'model': 'RX6800', 'variant': 'Navi21-XL'},
    '1002:73da': {'arch': 'RDNA2', 'model': 'RX6700XT', 'variant': 'Navi22-XT'},
    '1002:73db': {'arch': 'RDNA2', 'model': 'RX6700', 'variant': 'Navi22-XL'},
    '1002:73e8': {'arch': 'RDNA2', 'model': 'RX6600XT', 'variant': 'Navi23-XT'},
    '1002:73e9': {'arch': 'RDNA2', 'model': 'RX6600', 'variant': 'Navi23-XL'},
    
    # AMD RDNA3
    '1002:744c': {'arch': 'RDNA3', 'model': 'RX7900XTX', 'variant': 'Navi31-XTX'},
    '1002:7448': {'arch': 'RDNA3', 'model': 'RX7900XT', 'variant': 'Navi31-XT'},
    '1002:747e': {'arch': 'RDNA3', 'model': 'RX7800XT', 'variant': 'Navi32-XT'},
    '1002:7479': {'arch': 'RDNA3', 'model': 'RX7700XT', 'variant': 'Navi32-XL'},
    '1002:7480': {'arch': 'RDNA3', 'model': 'RX7600', 'variant': 'Navi33-XL'},
    '1002:7483': {'arch': 'RDNA3', 'model': 'RX7600XT', 'variant': 'Navi33-XT'},
    
    # AMD CDNA2
    '1002:740c': {'arch': 'CDNA2', 'model': 'MI250X', 'variant': 'Instinct-OAM'},
    '1002:740f': {'arch': 'CDNA2', 'model': 'MI250', 'variant': 'Instinct-OAM'},
    '1002:7408': {'arch': 'CDNA2', 'model': 'MI210', 'variant': 'Instinct-PCIe'},
    
    # AMD CDNA3
    '1002:744c': {'arch': 'CDNA3', 'model': 'MI300X', 'variant': 'Instinct-OAM'},
    '1002:744f': {'arch': 'CDNA3', 'model': 'MI300A', 'variant': 'Instinct-APU'},
    '1002:7460': {'arch': 'CDNA3', 'model': 'MI325X', 'variant': 'Instinct-OAM'},
    
    # AMD CDNA (first-gen)
    '1002:738c': {'arch': 'CDNA', 'model': 'MI100', 'variant': 'Instinct-PCIe'},
    '1002:738e': {'arch': 'CDNA', 'model': 'MI50', 'variant': 'Instinct-PCIe'},
    '1002:7362': {'arch': 'CDNA', 'model': 'MI25', 'variant': 'Instinct-PCIe'},
    
    # Intel Xe-HPC (Ponte Vecchio)
    '8086:0bd5': {'arch': 'XE_HPC', 'model': 'MAX1550', 'variant': 'Data-Center'},
    '8086:0bd9': {'arch': 'XE_HPC', 'model': 'MAX1100', 'variant': 'Data-Center'},
    '8086:0bda': {'arch': 'XE_HPC', 'model': 'MAX1350', 'variant': 'Data-Center'},
    
    # Intel Xe-HPG (Arc)
    '8086:56c0': {'arch': 'XE_HPG', 'model': 'ARC770', 'variant': 'Arc-Desktop'},
    '8086:5690': {'arch': 'XE_HPG', 'model': 'A580', 'variant': 'Arc-Desktop'},
    '8086:5691': {'arch': 'XE_HPG', 'model': 'A750', 'variant': 'Arc-Desktop'},
    '8086:5692': {'arch': 'XE_HPG', 'model': 'A730M', 'variant': 'Arc-Mobile'},
    '8086:5693': {'arch': 'XE_HPG', 'model': 'A550M', 'variant': 'Arc-Mobile'},
}

# Architecture-specific performance characteristics
ARCHITECTURE_FEATURES = {
    'AMPERE': {
        'tensor_cores': 'gen3',
        'rt_cores': 'gen2', 
        'memory_types': ['HBM2', 'HBM2e', 'GDDR6X'],
        'compute_capabilities': ['8.0', '8.6', '8.7'],
        'key_features': ['sparsity', 'multi_instance_gpu', 'nvlink3']
    },
    'ADA_LOVELACE': {
        'tensor_cores': None,
        'rt_cores': 'gen3',
        'memory_types': ['GDDR6X'],
        'compute_capabilities': ['8.9'],
        'key_features': ['av1_encode', 'dlss3', 'ada_sparsity']
    },
    'HOPPER': {
        'tensor_cores': 'gen4',
        'rt_cores': None,
        'memory_types': ['HBM3', 'HBM2e'],
        'compute_capabilities': ['9.0'],
        'key_features': ['transformer_engine', 'nvlink4', 'confidential_computing']
    },
    'BLACKWELL': {
        'tensor_cores': 'gen5',
        'rt_cores': 'gen4',  # For gaming variants
        'memory_types': ['HBM3e'],
        'compute_capabilities': ['10.0'],  # Projected
        'key_features': ['fp4_precision', 'secure_ai', 'nvlink5']
    },
    'RDNA2': {
        'ray_accelerators': 'gen1',
        'memory_types': ['GDDR6'],
        'key_features': ['infinity_cache', 'smart_access_memory', 'variable_rate_shading']
    },
    'RDNA3': {
        'ray_accelerators': 'gen2',
        'memory_types': ['GDDR6'],
        'key_features': ['chiplet_design', 'av1_encode', 'displayport2.1']
    },
    'CDNA2': {
        'matrix_cores': 'gen2',
        'memory_types': ['HBM2e'],
        'key_features': ['infinity_cache', 'rocm_support', 'multi_die']
    },
    'CDNA3': {
        'matrix_cores': 'gen3',
        'memory_types': ['HBM3'],
        'key_features': ['unified_memory', 'fp8_support', 'apu_integration']
    }
}

def get_architecture_info(arch_name: str) -> Optional[GPUArchitecture]:
    """Get architecture information by name."""
    all_archs = {**NVIDIA_ARCHITECTURES, **AMD_ARCHITECTURES, **INTEL_ARCHITECTURES}
    return all_archs.get(arch_name.upper())

def get_gpu_info_by_pci_id(pci_vendor: str, pci_device: str) -> Optional[Dict[str, Any]]:
    """Get comprehensive GPU information from PCI ID."""
    pci_key = f"{pci_vendor.lower()}:{pci_device.lower()}"
    gpu_info = PCI_ARCHITECTURE_MAP.get(pci_key)
    
    if gpu_info:
        arch_info = get_architecture_info(gpu_info['arch'])
        return {
            'pci_id': pci_key,
            'architecture': gpu_info['arch'],
            'model': gpu_info['model'],
            'variant': gpu_info['variant'],
            'architecture_info': arch_info,
            'features': ARCHITECTURE_FEATURES.get(gpu_info['arch'], {})
        }
    
    return None

def get_compute_capability(arch_name: str, model: str) -> Optional[str]:
    """Get NVIDIA compute capability for a specific architecture and model."""
    if arch_name not in ARCHITECTURE_FEATURES:
        return None
    
    compute_caps = ARCHITECTURE_FEATURES[arch_name].get('compute_capabilities', [])
    if compute_caps:
        # For simplicity, return the primary compute capability
        # In reality, this might vary by specific model
        return compute_caps[0]
    
    return None

# Export all architectures for easy access
ALL_ARCHITECTURES = {**NVIDIA_ARCHITECTURES, **AMD_ARCHITECTURES, **INTEL_ARCHITECTURES}