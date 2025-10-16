"""
Device matching utilities for hardware identification.

This module provides utilities to match cluster-reported device information
to known hardware configurations based on various identifiers like PCI IDs,
device names, and memory sizes.
"""

import re
import logging
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass

try:
    from .gpu_specs import get_gpu_info_by_pci_id, PCI_ARCHITECTURE_MAP, get_architecture_info
except ImportError:
    # Fallback for when gpu_specs is not available
    def get_gpu_info_by_pci_id(pci_vendor: str, pci_device: str) -> Optional[Dict[str, Any]]:
        return None
    PCI_ARCHITECTURE_MAP = {}
    def get_architecture_info(arch_name: str) -> Optional[Any]:
        return None

logger = logging.getLogger(__name__)


@dataclass
class DeviceIdentity:
    """Parsed device identity from cluster information."""
    vendor: Optional[str] = None
    model: Optional[str] = None
    memory_gb: Optional[float] = None
    variant: Optional[str] = None
    pci_vendor: Optional[str] = None
    pci_device: Optional[str] = None
    raw_name: Optional[str] = None
    cpu_family: Optional[str] = None  # CPU family ID (e.g., "6" for Intel)
    cpu_model: Optional[str] = None   # CPU model number (e.g., "143" for Sapphire Rapids)
    cpu_stepping: Optional[str] = None  # CPU stepping
    device_type: Optional[str] = None  # 'cpu' or 'gpu'
    # GPU-specific fields
    gpu_architecture: Optional[str] = None  # e.g., "AMPERE", "RDNA3", "HOPPER"
    compute_capability: Optional[str] = None  # NVIDIA compute capability
    architecture_features: Optional[Dict[str, Any]] = None  # Architecture-specific features


class DeviceParser:
    """Parse device information from various formats."""
    
    # Known vendor prefixes and their normalized names
    VENDOR_PATTERNS = {
        'nvidia': ['nvidia', 'tesla', 'grace'],
        'amd': ['amd', 'radeon', 'mi', 'epyc', 'milan', 'genoa', 'bergamo', 'turin'],
        'intel': ['intel', 'xe', 'iris', 'xeon', 'sapphire', 'emerald', 'granite', 'sierra'],
        'google': ['tpu', 'google'],
        'apple': ['apple', 'm1', 'm2', 'm3'],
        'aws': ['graviton', 'aws'],
        'ampere': ['ampere', 'altra'],
        'arm': ['arm', 'neoverse'],
        'qualcomm': ['qualcomm', 'adreno'],
        'graphcore': ['graphcore', 'ipu'],
        'cerebras': ['cerebras', 'wse'],
        'habana': ['habana', 'gaudi', 'goya'],
        'sambanova': ['sambanova', 'sn'],
        'groq': ['groq', 'tsp']
    }
    
    # Model extraction patterns
    MODEL_PATTERNS = [
        # NVIDIA GPU patterns
        (r'(?:nvidia\s+)?(?:tesla\s+)?([ahvtl]\d{2,3})', 'nvidia'),  # A100, H100, V100, T4, L4, L40
        (r'(?:nvidia\s+)?(?:geforce\s+)?rtx\s*(\d{4})', 'nvidia'),  # RTX 3090, RTX 4090
        (r'(?:nvidia\s+)?(?:quadro\s+)?rtx\s*([a-z]?\d{4})', 'nvidia'),  # Quadro RTX A6000
        
        # NVIDIA CPU patterns
        (r'(?:nvidia\s+)?(grace)(?:\s+cpu)?', 'nvidia'),  # NVIDIA Grace CPU
        
        # Intel CPU patterns
        (r'(?:intel[\s\(R\)]*)?(?:xeon[\s\(R\)]*)?(?:platinum[\s\(R\)]*)?(\d{4,5}[h\+a-z]?)', 'intel'),  # Xeon Platinum 8480+, 8490H
        (r'(sapphire\s*rapids?)', 'intel'),  # Sapphire Rapids
        (r'(emerald\s*rapids?)', 'intel'),  # Emerald Rapids
        (r'(granite\s*rapids?)', 'intel'),  # Granite Rapids
        (r'(sierra\s*forest)', 'intel'),  # Sierra Forest
        
        # Intel GPU patterns
        (r'(?:intel\s+)?(?:xe\s+)?max\s*(\d{3,4})', 'intel'),  # Max1550, Max1100
        (r'(?:intel\s+)?(?:arc\s+)?a(\d{3})', 'intel'),  # Arc A770
        
        # AMD GPU patterns  
        (r'(?:amd\s+)?(?:instinct\s+)?mi(\d{2,3}[xa]?)', 'amd'),  # MI100, MI250X, MI300A
        (r'(?:amd\s+)?(?:radeon\s+)?rx\s*(\d{4})', 'amd'),  # RX 7900
        
        # AMD CPU patterns
        (r'(?:amd\s+)?epyc\s+(milan-?x?)', 'amd'),  # EPYC Milan/Milan-X
        (r'(?:amd\s+)?epyc\s+(genoa-?x?)', 'amd'),  # EPYC Genoa/Genoa-X
        (r'(?:amd\s+)?epyc\s+(bergamo)', 'amd'),  # EPYC Bergamo
        (r'(?:amd\s+)?epyc\s+(turin)', 'amd'),  # EPYC Turin
        (r'(?:amd\s+)?epyc\s+(rome)', 'amd'),  # EPYC Rome
        (r'(?:amd\s+)?epyc\s+(\d{4,5}[a-z]?)', 'amd'),  # EPYC 7763, 9654
        (r'^(milan-?x?)(?:\s+cpu)?', 'amd'),  # Standalone Milan/Milan-X
        (r'^(genoa-?x?)(?:\s+cpu)?', 'amd'),  # Standalone Genoa/Genoa-X
        (r'^(bergamo)(?:\s+cpu)?', 'amd'),  # Standalone Bergamo
        (r'^(turin)(?:\s+cpu)?', 'amd'),  # Standalone Turin
        
        # ARM CPU patterns
        (r'(?:aws\s+)?(graviton)\s*(\d)', 'aws'),  # AWS Graviton3, Graviton4, Graviton 4
        (r'(ampere\s*(?:altra|one))', 'ampere'),  # Ampere Altra
        (r'(neoverse\s*[nv]\d)', 'arm'),  # ARM Neoverse N2, V1
        
        # Google TPU patterns
        (r'tpu\s*v?(\d+[a-z]?)', 'google'),  # TPUv4, TPU v4, TPU4
        
        # Apple Silicon patterns
        (r'(?:apple\s+)?m(\d+)\s*(?:pro|max|ultra)?', 'apple'),  # M1, M2 Pro, M3 Max
        
        # Other accelerators
        (r'gaudi(\d+)?', 'habana'),  # Gaudi, Gaudi2
        (r'ipu-?(\w+)', 'graphcore'),  # IPU-POD16
        (r'wse-?(\d+)', 'cerebras'),  # WSE-2
    ]
    
    # Memory size patterns
    MEMORY_PATTERNS = [
        (r'(\d+)\s*gb', 1),  # 40GB, 80 GB
        (r'(\d+)\s*g\b', 1),  # 40G
        (r'(\d+)\s*gib', 1),  # 40GiB
        (r'(\d+)\s*mb', 1/1024),  # 40960MB - correct conversion
        (r'(\d+)\s*mib', 1/1024),  # 40960MiB - correct conversion
    ]
    
    # Variant patterns (form factor, generation, etc.)
    VARIANT_PATTERNS = [
        r'sxm[0-9]?',  # SXM4, SXM5
        r'pcie',  # PCIe
        r'oam',  # OAM
        r'nvl',  # NVL
        r'dgx',  # DGX
        r'hgx',  # HGX
    ]
    
    @classmethod
    def parse(cls, device_info: Dict[str, Any]) -> DeviceIdentity:
        """
        Parse device information from cluster data.
        
        Args:
            device_info: Dictionary containing device information from cluster.
                Expected keys: raw_name, pci_vendor, pci_device, memory_mb, etc.
        
        Returns:
            DeviceIdentity with parsed information.
        """
        identity = DeviceIdentity()
        
        # Store raw information
        identity.raw_name = device_info.get('raw_name', '')
        identity.pci_vendor = device_info.get('pci_vendor', '')
        identity.pci_device = device_info.get('pci_device', '')
        
        # CPU-specific identifiers
        identity.cpu_family = device_info.get('cpu_family', '')
        identity.cpu_model = device_info.get('cpu_model', '')
        identity.cpu_stepping = device_info.get('cpu_stepping', '')
        
        # Determine device type and get GPU architecture info
        if identity.pci_vendor and identity.pci_device:
            identity.device_type = 'gpu'  # Has PCI IDs, likely a GPU
            
            # Get comprehensive GPU info from PCI ID
            gpu_info = get_gpu_info_by_pci_id(identity.pci_vendor, identity.pci_device)
            if gpu_info:
                identity.gpu_architecture = gpu_info['architecture']
                identity.model = gpu_info['model']  # Use precise model from PCI ID
                identity.variant = gpu_info['variant']
                identity.architecture_features = gpu_info.get('features', {})
                
                # Get compute capability for NVIDIA GPUs
                if identity.vendor == 'nvidia' or identity.gpu_architecture in ['AMPERE', 'ADA_LOVELACE', 'HOPPER', 'BLACKWELL', 'VOLTA', 'TURING', 'PASCAL']:
                    compute_caps = identity.architecture_features.get('compute_capabilities', [])
                    if compute_caps:
                        identity.compute_capability = compute_caps[0]
                        
        elif identity.cpu_family and identity.cpu_model:
            identity.device_type = 'cpu'  # Has CPU IDs, definitely a CPU
        
        # Parse memory from MB if provided
        if 'memory_mb' in device_info:
            try:
                memory_mb = float(device_info['memory_mb'])
                identity.memory_gb = round(memory_mb / 1024, 1)
            except (ValueError, TypeError):
                pass
        
        # Parse device name if available (but don't override PCI-derived info)
        if identity.raw_name:
            name_lower = identity.raw_name.lower()
            
            # Extract vendor only if not already set
            if not identity.vendor:
                for vendor, patterns in cls.VENDOR_PATTERNS.items():
                    if any(p in name_lower for p in patterns):
                        identity.vendor = vendor
                        break
            
            # Extract model only if not already set from PCI ID
            if not identity.model:
                for pattern, vendor in cls.MODEL_PATTERNS:
                    match = re.search(pattern, name_lower)
                    if match:
                        model_str = match.group(1).upper()
                        # Special handling for TPUs to add 'V' prefix
                        if vendor == 'google' and model_str.isdigit():
                            identity.model = f'V{model_str}'
                        # Special handling for AMD MI series  
                        elif vendor == 'amd' and not model_str.startswith('MI'):
                            identity.model = f'MI{model_str}'
                        # Special handling for Intel CPUs - preserve H suffix
                        elif vendor == 'intel' and model_str.endswith('H'):
                            identity.model = model_str  # Keep as is (e.g., "8490H")
                        else:
                            identity.model = model_str
                        if not identity.vendor:
                            identity.vendor = vendor
                        break
            
            # Extract memory if not already set
            if not identity.memory_gb:
                for pattern, multiplier in cls.MEMORY_PATTERNS:
                    match = re.search(pattern, name_lower)
                    if match:
                        try:
                            memory_value = float(match.group(1))
                            identity.memory_gb = round(memory_value * multiplier, 1)
                            break
                        except (ValueError, TypeError):
                            pass
            
            # Extract variant
            for pattern in cls.VARIANT_PATTERNS:
                match = re.search(pattern, name_lower)
                if match:
                    identity.variant = match.group(0).upper()
                    break
        
        return identity


class DeviceMatcher:
    """Match device identities to hardware configurations."""
    
    # CPU Family/Model mappings (family:model -> CPU name)
    # Based on Intel/AMD CPU identification standards
    CPU_ID_MAP = {
        # Intel CPUs (Family 6)
        '6:143': 'SAPPHIRERAPIDS',  # Sapphire Rapids (Model 0x8F)
        '6:207': 'EMERALDRAPIDS',   # Emerald Rapids (Model 0xCF)
        '6:173': 'GRANITERAPIDS',   # Granite Rapids (Model 0xAD)
        '6:175': 'SIERRAFOREST',    # Sierra Forest (Model 0xAF)
        
        # AMD CPUs (Family 25/0x19 = Zen 3/4)
        '25:1': 'MILAN',      # Milan (Zen 3)
        '25:8': 'MILANX',     # Milan-X (Zen 3 + V-Cache)
        '25:17': 'GENOA',     # Genoa (Zen 4)
        '25:18': 'GENOAX',    # Genoa-X (Zen 4 + V-Cache)
        '25:160': 'BERGAMO',  # Bergamo (Zen 4c)
        '26:1': 'TURIN',      # Turin (Zen 5)
    }
    
    # CPUID to specific SKU mapping
    CPUID_MODEL_MAP = {
        # Intel Xeon Sapphire Rapids
        '8490H': 'XEON_PLATINUM_8490H',
        '8480+': 'XEON_PLATINUM_8480PLUS',
        '8480': 'XEON_PLATINUM_8480PLUS',  # Map 8480 to 8480+
        '8470': 'XEON_PLATINUM_8470',
        '8460Y+': 'XEON_PLATINUM_8460YPLUS',
        '8460Y': 'XEON_PLATINUM_8460YPLUS',
        '6430': 'XEON_GOLD_6430',
        
        # Intel Xeon Emerald Rapids
        '8592+': 'XEON_PLATINUM_8592PLUS',
        '8592': 'XEON_PLATINUM_8592PLUS',
        '8580': 'XEON_PLATINUM_8580',
        
        # AMD EPYC Milan
        '7763': 'EPYC_7763',
        '7773X': 'EPYC_7773X',
        '7713': 'EPYC_7713',
        
        # AMD EPYC Genoa
        '9654': 'EPYC_9654',
        '9554': 'EPYC_9554',
        '9454': 'EPYC_9454',
        '9684X': 'EPYC_9684X',
        
        # AMD EPYC Bergamo
        '9754': 'EPYC_9754',
        '9734': 'EPYC_9734',
    }
    
    # PCI Device ID mappings (vendor_id:device_id -> base model)
    PCI_ID_MAP = {
        # NVIDIA GPUs
        '10de:20b0': 'A100',  # A100 PCIe 40GB
        '10de:20b2': 'A100',  # A100 SXM4 80GB  
        '10de:20b5': 'A100',  # A100 PCIe 80GB
        '10de:20f1': 'A100',  # A100 SXM4 40GB
        '10de:2330': 'H100',  # H100 SXM5
        '10de:2331': 'H100',  # H100 PCIe
        '10de:2339': 'H100',  # H100 NVL
        '10de:2335': 'H200',  # H200 SXM
        '10de:2336': 'H200',  # H200 NVL
        '10de:233a': 'GH200', # GH200
        '10de:26b9': 'L40',   # L40
        '10de:26ba': 'L40S',  # L40S
        '10de:27b8': 'L4',    # L4
        '10de:1db4': 'V100',  # V100 PCIe 16GB
        '10de:1db5': 'V100',  # V100 PCIe 32GB
        '10de:1db6': 'V100',  # V100 SXM2 16GB
        '10de:1df6': 'V100',  # V100S PCIe 32GB
        '10de:1e30': 'RTX6000', # Quadro RTX 6000
        '10de:2204': 'RTX3090', # GeForce RTX 3090
        '10de:2684': 'RTX4090', # GeForce RTX 4090
        '10de:2230': 'RTXA6000', # RTX A6000
        '10de:1e04': 'RTX8000', # Quadro RTX 8000
        '10de:1b38': 'P40',   # Tesla P40
        '10de:1b06': 'P100',  # P100 PCIe
        '10de:15f8': 'P100',  # P100 SXM2
        '10de:1eb8': 'T4',    # Tesla T4
        
        # AMD GPUs
        '1002:740c': 'MI250X', # Instinct MI250X
        '1002:740f': 'MI250',  # Instinct MI250
        '1002:7408': 'MI210',  # Instinct MI210  
        '1002:738c': 'MI100',  # Instinct MI100
        '1002:744c': 'MI300X', # Instinct MI300X
        '1002:744f': 'MI300A', # Instinct MI300A
        
        # Intel GPUs
        '8086:0bd5': 'MAX1550', # Data Center GPU Max 1550
        '8086:0bd9': 'MAX1100', # Data Center GPU Max 1100
        '8086:0bda': 'MAX1350', # Data Center GPU Max 1350
        '8086:56c0': 'ARC770',  # Arc A770
    }
    
    # Model name aliases and variations
    MODEL_ALIASES = {
        # NVIDIA GPU aliases
        'A100': ['A100', 'TESLA A100', 'A100-SXM4', 'A100-PCIE', 'A100-SXM', 'NVIDIA A100'],
        'H100': ['H100', 'HOPPER', 'H100-SXM5', 'H100-PCIE', 'H100-NVL', 'NVIDIA H100'],
        'H200': ['H200', 'H200-SXM', 'H200-NVL', 'NVIDIA H200'],
        'GH200': ['GH200', 'GRACE HOPPER', 'GH200-96GB', 'GH200-144GB'],
        'L40': ['L40', 'L40-48GB', 'NVIDIA L40'],
        'L40S': ['L40S', 'L40S-48GB', 'NVIDIA L40S', 'NVIDIA-L40S'],
        'L4': ['L4', 'L4-24GB', 'NVIDIA L4'],
        'V100': ['V100', 'TESLA V100', 'V100-SXM2', 'V100-PCIE', 'VOLTA'],
        'T4': ['T4', 'TESLA T4', 'T4-16GB'],
        'P40': ['P40', 'TESLA P40', 'P40-24GB'],
        'P100': ['P100', 'TESLA P100', 'P100-SXM2', 'P100-PCIE'],
        
        # NVIDIA CPU aliases
        'GRACE': ['GRACE', 'GRACE CPU', 'NVIDIA GRACE', 'NVIDIA GRACE CPU'],
        
        # Intel CPU aliases - Specific SKUs
        'XEON_PLATINUM_8490H': ['XEON PLATINUM 8490H', 'XEON 8490H', '8490H', 'PLATINUM 8490H', 'INTEL XEON 8490H'],
        'XEON_PLATINUM_8480PLUS': ['XEON PLATINUM 8480+', 'XEON 8480+', '8480+', 'PLATINUM 8480+', 'XEON PLATINUM 8480', 'INTEL XEON 8480+', 'INTEL XEON 8480', 'INTEL(R) XEON(R) PLATINUM 8480+'],
        'XEON_PLATINUM_8470': ['XEON PLATINUM 8470', 'XEON 8470', '8470', 'PLATINUM 8470', 'INTEL XEON 8470'],
        'XEON_PLATINUM_8460YPLUS': ['XEON PLATINUM 8460Y+', 'XEON 8460Y+', '8460Y+', 'PLATINUM 8460Y+', 'INTEL XEON 8460Y+'],
        'XEON_GOLD_6430': ['XEON GOLD 6430', 'XEON 6430', '6430', 'GOLD 6430', 'INTEL XEON 6430'],
        'XEON_PLATINUM_8592PLUS': ['XEON PLATINUM 8592+', 'XEON 8592+', '8592+', 'PLATINUM 8592+', 'INTEL XEON 8592+'],
        'XEON_PLATINUM_8580': ['XEON PLATINUM 8580', 'XEON 8580', '8580', 'PLATINUM 8580', 'INTEL XEON 8580'],
        
        # Keep generation aliases for backward compatibility
        'SAPPHIRERAPIDS': ['SAPPHIRE RAPIDS', 'SAPPHIRERAPIDS', 'XEON SAPPHIRE RAPIDS', 'INTEL SAPPHIRE RAPIDS'],
        'EMERALDRAPIDS': ['EMERALD RAPIDS', 'EMERALDRAPIDS', 'XEON EMERALD RAPIDS', 'INTEL EMERALD RAPIDS'],
        'GRANITERAPIDS': ['GRANITE RAPIDS', 'GRANITERAPIDS', 'XEON GRANITE RAPIDS', 'INTEL GRANITE RAPIDS'],
        'SIERRAFOREST': ['SIERRA FOREST', 'SIERRAFOREST', 'XEON SIERRA FOREST', 'INTEL SIERRA FOREST'],
        
        # AMD GPU aliases
        'MI250X': ['MI250X', 'INSTINCT MI250X', 'AMD MI250X'],
        'MI250': ['MI250', 'INSTINCT MI250', 'AMD MI250'],
        'MI210': ['MI210', 'INSTINCT MI210', 'AMD MI210'],
        'MI100': ['MI100', 'INSTINCT MI100', 'AMD MI100'],
        'MI300X': ['MI300X', 'INSTINCT MI300X', 'AMD MI300X'],
        'MI300A': ['MI300A', 'INSTINCT MI300A', 'AMD MI300A'],
        
        # AMD CPU aliases - Specific SKUs
        'EPYC_7763': ['EPYC 7763', 'AMD EPYC 7763', '7763', 'MILAN 7763', 'EPYC MILAN 7763'],
        'EPYC_7773X': ['EPYC 7773X', 'AMD EPYC 7773X', '7773X', 'MILAN-X 7773X', 'EPYC MILAN-X 7773X'],
        'EPYC_7713': ['EPYC 7713', 'AMD EPYC 7713', '7713', 'MILAN 7713', 'EPYC MILAN 7713'],
        'EPYC_9654': ['EPYC 9654', 'AMD EPYC 9654', '9654', 'GENOA 9654', 'EPYC GENOA 9654'],
        'EPYC_9554': ['EPYC 9554', 'AMD EPYC 9554', '9554', 'GENOA 9554', 'EPYC GENOA 9554'],
        'EPYC_9454': ['EPYC 9454', 'AMD EPYC 9454', '9454', 'GENOA 9454', 'EPYC GENOA 9454'],
        'EPYC_9684X': ['EPYC 9684X', 'AMD EPYC 9684X', '9684X', 'GENOA-X 9684X', 'EPYC GENOA-X 9684X'],
        'EPYC_9754': ['EPYC 9754', 'AMD EPYC 9754', '9754', 'BERGAMO 9754', 'EPYC BERGAMO 9754'],
        'EPYC_9734': ['EPYC 9734', 'AMD EPYC 9734', '9734', 'BERGAMO 9734', 'EPYC BERGAMO 9734'],
        
        # Keep generation aliases for backward compatibility
        'MILANX': ['MILAN-X', 'MILANX', 'EPYC MILAN-X', 'AMD MILAN-X', 'AMD EPYC MILAN-X'],
        'MILAN': ['MILAN', 'EPYC MILAN', 'AMD MILAN', 'AMD EPYC MILAN'],
        'GENOA': ['GENOA', 'EPYC GENOA', 'AMD GENOA', 'AMD EPYC GENOA'],
        'GENOAX': ['GENOA-X', 'GENOAX', 'EPYC GENOA-X', 'AMD GENOA-X', 'AMD EPYC GENOA-X'],
        'BERGAMO': ['BERGAMO', 'EPYC BERGAMO', 'AMD BERGAMO', 'AMD EPYC BERGAMO'],
        'TURIN': ['TURIN', 'EPYC TURIN', 'AMD TURIN', 'AMD EPYC TURIN'],
        
        # ARM CPU aliases
        'GRAVITON3': ['GRAVITON3', 'GRAVITON 3', 'AWS GRAVITON3', 'AWS GRAVITON 3'],
        'GRAVITON4': ['GRAVITON4', 'GRAVITON 4', 'AWS GRAVITON4', 'AWS GRAVITON 4'],
        
        # Intel GPU aliases
        'MAX1550': ['MAX1550', 'MAX 1550', 'PONTE VECCHIO', 'PVC', 'INTEL MAX 1550'],
        'MAX1100': ['MAX1100', 'MAX 1100', 'INTEL MAX 1100'],
        
        # TPU aliases
        'TPUV4': ['TPUV4', 'TPU V4', 'TPU-V4'],
        'TPUV5E': ['TPUV5E', 'TPU V5E', 'TPU-V5E', 'TPUV5 LITE'],
        'TPUV5P': ['TPUV5P', 'TPU V5P', 'TPU-V5P', 'TPUV5 POD'],
        'TPUV6': ['TPUV6', 'TPU V6', 'TPU-V6'],
    }
    
    @classmethod
    def match_by_cpu_id(cls, cpu_family: str, cpu_model: str) -> Optional[str]:
        """
        Match CPU by family and model IDs.
        
        Args:
            cpu_family: CPU family ID (e.g., "6" for Intel, "25" for AMD Zen3/4)
            cpu_model: CPU model ID (e.g., "143" for Sapphire Rapids)
            
        Returns:
            Matched CPU configuration name or None.
        """
        if not cpu_family or not cpu_model:
            return None
        
        cpu_key = f"{cpu_family}:{cpu_model}"
        return cls.CPU_ID_MAP.get(cpu_key)
    
    @classmethod
    def match_by_cpu_model_number(cls, model_number: str) -> Optional[str]:
        """
        Match CPU by model number (e.g., "8480+" for Xeon, "9654" for EPYC).
        
        Args:
            model_number: CPU model number from the name
            
        Returns:
            Matched CPU configuration name or None.
        """
        if not model_number:
            return None
        
        # Try exact match first
        if model_number in cls.CPUID_MODEL_MAP:
            return cls.CPUID_MODEL_MAP[model_number]
        
        # Try without suffix (e.g., "8480+" -> "8480")
        base_number = model_number.rstrip('+')
        if base_number in cls.CPUID_MODEL_MAP:
            return cls.CPUID_MODEL_MAP[base_number]
        
        return None
    
    @classmethod
    def match_by_pci_id(cls, pci_vendor: str, pci_device: str, 
                        memory_gb: Optional[float] = None) -> Optional[str]:
        """
        Match device by PCI ID with architecture awareness.
        
        Args:
            pci_vendor: PCI vendor ID (e.g., "10de" for NVIDIA)
            pci_device: PCI device ID (e.g., "20f1" for A100)
            memory_gb: Optional memory size for variant selection
            
        Returns:
            Matched hardware configuration name or None.
        """
        if not pci_vendor or not pci_device:
            return None
        
        # Normalize PCI IDs (remove 0x prefix if present)
        pci_vendor = pci_vendor.lower().replace('0x', '')
        pci_device = pci_device.lower().replace('0x', '')
        
        # First, try architecture-aware matching from gpu_specs
        gpu_info = get_gpu_info_by_pci_id(pci_vendor, pci_device)
        if gpu_info:
            base_model = gpu_info['model']
            # Add memory variant if available
            if memory_gb:
                return cls._add_memory_variant(base_model, memory_gb)
            return base_model
        
        # Fallback to legacy PCI_ID_MAP
        pci_key = f"{pci_vendor}:{pci_device}"
        base_model = cls.PCI_ID_MAP.get(pci_key)
        
        if not base_model:
            return None
        
        # Add memory variant if available
        if memory_gb:
            return cls._add_memory_variant(base_model, memory_gb)
        
        return base_model
    
    @classmethod
    def match_by_architecture(cls, architecture: str, model: str, 
                             memory_gb: Optional[float] = None) -> Optional[str]:
        """
        Match GPU by architecture and model name.
        
        Args:
            architecture: GPU architecture (e.g., "AMPERE", "HOPPER")
            model: GPU model within architecture
            memory_gb: Optional memory size for variant selection
            
        Returns:
            Matched hardware configuration name or None.
        """
        if not architecture or not model:
            return None
        
        # Direct model match with architecture context
        if memory_gb:
            return cls._add_memory_variant(model, memory_gb)
        
        return model
    
    @classmethod
    def match_by_name(cls, model: str, memory_gb: Optional[float] = None,
                     vendor: Optional[str] = None) -> Optional[str]:
        """
        Match device by parsed model name.
        
        Args:
            model: Parsed model name (e.g., "A100")
            memory_gb: Optional memory size for variant selection
            vendor: Optional vendor for disambiguation
            
        Returns:
            Matched hardware configuration name or None.
        """
        if not model:
            return None
        
        model_upper = model.upper()
        
        # Direct match in aliases
        for base_model, aliases in cls.MODEL_ALIASES.items():
            if model_upper in [a.upper() for a in aliases]:
                if memory_gb:
                    return cls._add_memory_variant(base_model, memory_gb)
                return base_model
        
        # Check if model contains any known model name
        for base_model, aliases in cls.MODEL_ALIASES.items():
            for alias in aliases:
                if alias.upper() in model_upper or model_upper in alias.upper():
                    if memory_gb:
                        return cls._add_memory_variant(base_model, memory_gb)
                    return base_model
        
        return None
    
    @classmethod
    def fuzzy_match(cls, raw_name: str) -> Optional[str]:
        """
        Perform fuzzy matching on raw device name.
        
        Args:
            raw_name: Raw device name from cluster
            
        Returns:
            Best matched hardware configuration name or None.
        """
        if not raw_name:
            return None
        
        raw_upper = raw_name.upper()
        best_match = None
        best_score = 0
        
        # First check if any model name is contained in the raw name
        for base_model, aliases in cls.MODEL_ALIASES.items():
            for alias in aliases:
                alias_upper = alias.upper()
                # Check if the base model name is in the raw name
                base_upper = base_model.upper()
                if base_upper in raw_upper or raw_upper in alias_upper:
                    return base_model
        
        # Score based on matching tokens
        for base_model, aliases in cls.MODEL_ALIASES.items():
            for alias in aliases:
                # Calculate similarity score
                alias_upper = alias.upper()
                if alias_upper == raw_upper:
                    return base_model  # Exact match
                
                # Token-based matching
                raw_tokens = set(raw_upper.split())
                alias_tokens = set(alias_upper.split())
                
                if raw_tokens and alias_tokens:
                    intersection = len(raw_tokens & alias_tokens)
                    union = len(raw_tokens | alias_tokens)
                    score = intersection / union if union > 0 else 0
                    
                    if score > best_score:
                        best_score = score
                        best_match = base_model
        
        # Return match if score is good enough (>40% similarity for more flexibility)
        if best_score > 0.4:
            return best_match
        
        return None
    
    @classmethod
    def _add_memory_variant(cls, base_model: str, memory_gb: float) -> str:
        """
        Add memory variant suffix to base model name.
        
        Args:
            base_model: Base model name (e.g., "A100")
            memory_gb: Memory size in GB
            
        Returns:
            Model name with memory variant (e.g., "A100_40GB")
        """
        # Round memory to common sizes
        memory_variants = {
            8: '8GB',
            16: '16GB',
            24: '24GB',
            32: '32GB',
            40: '40GB',
            48: '48GB',
            64: '64GB',
            80: '80GB',
            96: '96GB',
            128: '128GB',
            144: '144GB',
            192: '192GB',
        }
        
        # Find closest memory variant
        closest_memory = min(memory_variants.keys(), 
                           key=lambda x: abs(x - memory_gb))
        
        # Only add variant if close enough (within 15% for more flexibility)
        if abs(closest_memory - memory_gb) / closest_memory < 0.15:
            # Check for specific naming conventions
            if base_model in ['A100', 'H100', 'V100', 'P100']:
                return f"{base_model}_{memory_variants[closest_memory]}_GPU"
            elif base_model.startswith('TPU'):
                return base_model  # TPUs don't typically have memory variants in name
            elif base_model.startswith('ARC') or base_model.startswith('MAX'):
                return base_model  # Intel GPUs have fixed memory per model
            elif base_model in ['MI300X', 'MI325X', 'MI250X', 'MI250', 'MI210', 'MI100']:
                return base_model  # AMD Instinct GPUs have fixed memory per model
            else:
                return f"{base_model}_{memory_variants[closest_memory]}"
        
        return base_model


def match_device(device_info: Dict[str, Any], 
                 hardware_configs: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Match cluster device information to hardware configuration.
    
    This is the main entry point for device matching. It tries multiple
    matching strategies in order of reliability.
    
    Args:
        device_info: Device information from cluster containing:
            - raw_name: Full device name string
            - pci_vendor: PCI vendor ID
            - pci_device: PCI device ID  
            - memory_mb: Memory size in MB
            - gpu_count: Number of GPUs (optional)
        hardware_configs: Dictionary of available hardware configurations
        
    Returns:
        Matched hardware configuration dictionary or None if no match found.
    """
    # Parse device identity
    identity = DeviceParser.parse(device_info)
    
    logger.debug(f"Parsed device identity: {identity}")
    
    # Helper function to find config by name pattern
    def find_config_by_pattern(pattern: str, verify_memory: bool = True) -> Optional[Dict[str, Any]]:
        pattern_upper = pattern.upper()
        for config_name, config in hardware_configs.items():
            # Check if pattern matches config name
            if pattern_upper in config_name.upper():
                # For CPUs, be more lenient with memory verification since they can have varying amounts of system RAM
                if verify_memory and identity.memory_gb:
                    config_memory = config.get('Memory_size', config.get('memory_size', 0))
                    is_cpu = config.get('type', '').lower() == 'cpu'
                    if is_cpu:
                        # For CPUs, just check that device memory is reasonable (between 64GB and max config)
                        if 64 <= identity.memory_gb <= config_memory:
                            return config
                    else:
                        # For GPUs, use strict memory matching
                        if config_memory > 0 and abs(config_memory - identity.memory_gb) / config_memory < 0.15:
                            return config
                elif not verify_memory or not identity.memory_gb:
                    return config
            # Also check aliases
            aliases = config.get('aliases', [])
            if any(pattern_upper in alias.upper() for alias in aliases):
                if verify_memory and identity.memory_gb:
                    config_memory = config.get('Memory_size', config.get('memory_size', 0))
                    is_cpu = config.get('type', '').lower() == 'cpu'
                    if is_cpu:
                        # For CPUs, just check that device memory is reasonable (between 64GB and max config)
                        if 64 <= identity.memory_gb <= config_memory:
                            return config
                    else:
                        # For GPUs, use strict memory matching
                        if config_memory > 0 and abs(config_memory - identity.memory_gb) / config_memory < 0.15:
                            return config
                elif not verify_memory or not identity.memory_gb:
                    return config
        return None
    
    # Strategy 1a: Match CPU by CPU family/model (most reliable for CPUs)
    if identity.cpu_family and identity.cpu_model:
        matched_name = DeviceMatcher.match_by_cpu_id(
            identity.cpu_family,
            identity.cpu_model
        )
        if matched_name:
            config = find_config_by_pattern(matched_name, verify_memory=False)
            if config:
                logger.info(f"Matched CPU by family/model ID: {config.get('name', matched_name)}")
                return config
    
    # Strategy 1b: Match GPU by PCI ID with architecture (most reliable for GPUs)
    if identity.pci_vendor and identity.pci_device:
        matched_name = DeviceMatcher.match_by_pci_id(
            identity.pci_vendor, 
            identity.pci_device,
            identity.memory_gb
        )
        if matched_name:
            config = find_config_by_pattern(matched_name, verify_memory=True)
            if config:
                logger.info(f"Matched GPU by PCI ID: {config.get('name', matched_name)} "
                           f"(Architecture: {identity.gpu_architecture})")
                return config
    
    # Strategy 1c: Match GPU by architecture and model (reliable when PCI ID mapping fails)
    if identity.gpu_architecture and identity.model:
        matched_name = DeviceMatcher.match_by_architecture(
            identity.gpu_architecture,
            identity.model,
            identity.memory_gb
        )
        if matched_name:
            config = find_config_by_pattern(matched_name, verify_memory=True)
            if config:
                logger.info(f"Matched GPU by architecture: {config.get('name', matched_name)} "
                           f"({identity.gpu_architecture})")
                return config
    
    # Strategy 2: Match by parsed model and memory
    if identity.model:
        matched_name = DeviceMatcher.match_by_name(
            identity.model,
            identity.memory_gb,
            identity.vendor
        )
        if matched_name:
            config = find_config_by_pattern(matched_name, verify_memory=True)
            if config:
                logger.info(f"Matched device by model name: {config.get('name', matched_name)}")
                return config
    
    # Strategy 3: Fuzzy name matching (least reliable)
    if identity.raw_name:
        matched_name = DeviceMatcher.fuzzy_match(identity.raw_name)
        if matched_name:
            config = find_config_by_pattern(matched_name, verify_memory=True)
            if config:
                logger.info(f"Matched device by fuzzy matching: {config.get('name', matched_name)}")
                return config
    
    logger.warning(f"No match found for device: {device_info}")
    return None