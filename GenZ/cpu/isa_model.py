from enum import Enum
from typing import Dict, Tuple, Optional, List
import numpy as np
from dataclasses import dataclass, field


class ISAType(Enum):
    SCALAR = "scalar"
    SSE = "sse"
    AVX2 = "avx2"
    AVX512 = "avx512"
    AMX = "amx"
    NEON = "neon"
    SVE = "sve"
    SVE2 = "sve2"


@dataclass
class ISAConfig:
    """Configuration for a specific ISA"""
    name: ISAType
    vector_width: Dict[str, int]  # data_type -> elements
    throughput: Dict[str, float]  # operation -> ops/cycle
    supported_types: List[str]
    latency: Dict[str, int]  # operation -> cycles
    
    # Constraints
    alignment_required: int = 1
    min_vector_length: int = 1
    special_constraints: Dict = field(default_factory=dict)  # e.g., AMX tile constraints


class ISASelector:
    """Select optimal ISA for operations"""
    
    def __init__(self, cpu_config):
        self.cpu_config = cpu_config
        self.isa_configs = self._initialize_isa_configs()
        
    def _initialize_isa_configs(self) -> Dict[ISAType, ISAConfig]:
        """Initialize ISA configurations based on CPU"""
        configs = {}
        
        if 'avx512' in self.cpu_config.isa_support:
            configs[ISAType.AVX512] = ISAConfig(
                name=ISAType.AVX512,
                vector_width={'fp32': 16, 'fp16': 32, 'bf16': 32, 'int8': 64},
                throughput={
                    'fma': 2.0,  # 2 FMA units per core
                    'add': 2.0,
                    'mul': 2.0,
                    'load': 2.0,
                    'store': 1.0
                },
                supported_types=['fp32', 'fp16', 'bf16', 'int32', 'int8'],
                latency={'fma': 4, 'add': 3, 'mul': 3},
                alignment_required=64
            )
            
        if 'avx2' in self.cpu_config.isa_support:
            configs[ISAType.AVX2] = ISAConfig(
                name=ISAType.AVX2,
                vector_width={'fp32': 8, 'fp16': 16, 'bf16': 16, 'int8': 32},
                throughput={
                    'fma': 2.0,
                    'add': 2.0,
                    'mul': 2.0,
                    'load': 2.0,
                    'store': 1.0
                },
                supported_types=['fp32', 'fp16', 'bf16', 'int32', 'int8'],
                latency={'fma': 4, 'add': 3, 'mul': 3},
                alignment_required=32
            )
            
        if 'amx' in self.cpu_config.isa_support:
            configs[ISAType.AMX] = ISAConfig(
                name=ISAType.AMX,
                vector_width={'int8': 131072, 'bf16': 131072},  # Actual ops per tile: 2*16*64*64
                throughput={'tilemmul': 0.125},  # 1 tile op per 8 cycles
                supported_types=['int8', 'bf16'],
                latency={'tilemmul': 8},  # 8 cycles per tile operation
                special_constraints={
                    'tile_m': 16,
                    'tile_n': 64,
                    'tile_k': 64,
                    'num_tiles': 8,
                    'amx_units_per_core': 2  # 2 AMX units per core
                }
            )
            
        if 'sve2' in self.cpu_config.isa_support:
            configs[ISAType.SVE2] = ISAConfig(
                name=ISAType.SVE2,
                vector_width={'fp32': 16, 'fp16': 32, 'int8': 64},  # 512-bit vectors
                throughput={
                    'fma': 2.0,
                    'add': 2.0,
                    'mul': 2.0,
                    'load': 2.0,
                    'store': 1.0
                },
                supported_types=['fp32', 'fp16', 'int32', 'int8'],
                latency={'fma': 4, 'add': 3, 'mul': 3},
                alignment_required=64
            )
            
        if 'neon' in self.cpu_config.isa_support:
            configs[ISAType.NEON] = ISAConfig(
                name=ISAType.NEON,
                vector_width={'fp32': 4, 'fp16': 8, 'int8': 16},  # 128-bit vectors
                throughput={
                    'fma': 2.0,
                    'add': 2.0,
                    'mul': 2.0,
                    'load': 1.0,
                    'store': 1.0
                },
                supported_types=['fp32', 'fp16', 'int32', 'int8'],
                latency={'fma': 4, 'add': 3, 'mul': 3},
                alignment_required=16
            )
            
        # Always have scalar as fallback
        configs[ISAType.SCALAR] = ISAConfig(
            name=ISAType.SCALAR,
            vector_width={'fp32': 1, 'fp16': 1, 'int8': 1},
            throughput={
                'fma': 1.0,
                'add': 1.0,
                'mul': 1.0,
                'load': 1.0,
                'store': 1.0
            },
            supported_types=['fp32', 'fp16', 'int32', 'int8'],
            latency={'fma': 4, 'add': 3, 'mul': 3},
            alignment_required=1
        )
            
        return configs
        
    def select_isa(self, operator, data_type: str) -> Tuple[ISAType, float]:
        """
        Select optimal ISA for given operator
        Returns: (selected_isa, expected_efficiency)
        """
        # Get operator type from class name
        op_type = operator.__class__.__name__
        
        if op_type == 'GEMM':
            return self._select_isa_for_gemm(operator, data_type)
        elif op_type in ['Logit', 'Attend']:
            return self._select_isa_for_attention(operator, data_type)
        else:
            return self._select_isa_default(operator, data_type)
            
    def _select_isa_for_gemm(self, operator, data_type: str) -> Tuple[ISAType, float]:
        """Select ISA for GEMM operations"""
        M, N, K = operator.get_gemms()[:3]
        
        # Check if AMX is available and applicable
        if (ISAType.AMX in self.isa_configs and 
            data_type in ['int8', 'bf16'] and
            self._check_amx_constraints(M, N, K)):
            
            # Calculate AMX efficiency
            amx_config = self.isa_configs[ISAType.AMX]
            constraints = amx_config.special_constraints
            
            # How well do dimensions map to tiles?
            tile_efficiency = 1.0
            if M % constraints['tile_m'] != 0:
                tile_efficiency *= 0.9
            if N % constraints['tile_n'] != 0:
                tile_efficiency *= 0.9
            if K % constraints['tile_k'] != 0:
                tile_efficiency *= 0.9
            
            if tile_efficiency > 0.7:
                return ISAType.AMX, tile_efficiency
                
        # Otherwise try AVX-512
        if ISAType.AVX512 in self.isa_configs and data_type in ['fp32', 'fp16']:
            # Check if problem size is large enough
            if M * N * K > 1000:  # Arbitrary threshold
                return ISAType.AVX512, 0.85
                
        # Fall back to AVX2
        if ISAType.AVX2 in self.isa_configs:
            return ISAType.AVX2, 0.7
            
        # ARM alternatives
        if ISAType.SVE2 in self.isa_configs:
            return ISAType.SVE2, 0.8
        elif ISAType.NEON in self.isa_configs:
            return ISAType.NEON, 0.6
            
        return ISAType.SCALAR, 0.3
        
    def _select_isa_for_attention(self, operator, data_type: str) -> Tuple[ISAType, float]:
        """Select ISA for attention operations"""
        # Attention has less regular patterns, prefer wider vectors
        if ISAType.AVX512 in self.isa_configs and data_type in ['fp32', 'fp16', 'bf16']:
            return ISAType.AVX512, 0.75
        elif ISAType.AVX2 in self.isa_configs:
            return ISAType.AVX2, 0.65
        elif ISAType.SVE2 in self.isa_configs:
            return ISAType.SVE2, 0.7
        elif ISAType.NEON in self.isa_configs:
            return ISAType.NEON, 0.5
            
        return ISAType.SCALAR, 0.25
        
    def _select_isa_default(self, operator, data_type: str) -> Tuple[ISAType, float]:
        """Default ISA selection"""
        # Try to use widest available vector ISA
        if ISAType.AVX512 in self.isa_configs:
            return ISAType.AVX512, 0.7
        elif ISAType.AVX2 in self.isa_configs:
            return ISAType.AVX2, 0.6
        elif ISAType.SVE2 in self.isa_configs:
            return ISAType.SVE2, 0.65
        elif ISAType.NEON in self.isa_configs:
            return ISAType.NEON, 0.5
            
        return ISAType.SCALAR, 0.3
        
    def _check_amx_constraints(self, M: int, N: int, K: int) -> bool:
        """Check if dimensions are suitable for AMX"""
        if ISAType.AMX not in self.isa_configs:
            return False
            
        constraints = self.isa_configs[ISAType.AMX].special_constraints
        
        # Dimensions should be multiples of tile size or at least larger
        return (M >= constraints['tile_m'] and
                N >= constraints['tile_n'] and  
                K >= constraints['tile_k'])
                
    def get_compute_time(self, operator, isa: ISAType, 
                        num_threads: int, system) -> float:
        """Calculate compute time using selected ISA"""
        
        isa_config = self.isa_configs[isa]
        op_type = operator.__class__.__name__
        
        if op_type == 'GEMM' and isa == ISAType.AMX:
            return self._calculate_amx_gemm_time(operator, isa_config, num_threads, system)
        else:
            return self._calculate_vector_compute_time(operator, isa_config, num_threads, system)
            
    def _calculate_amx_gemm_time(self, operator, isa_config: ISAConfig,
                                num_threads: int, system) -> float:
        """Calculate GEMM time using AMX tiles"""
        M, N, K = operator.get_gemms()[:3]
        constraints = isa_config.special_constraints
        
        # Number of tile operations needed
        num_tile_ops = (
            np.ceil(M / constraints['tile_m']) *
            np.ceil(N / constraints['tile_n']) *
            np.ceil(K / constraints['tile_k'])
        )
        
        # Account for 2 AMX units per core
        amx_units = constraints.get('amx_units_per_core', 1) * num_threads
        
        # Cycles accounting for multiple AMX units
        cycles = num_tile_ops / amx_units * isa_config.latency['tilemmul']
        
        # Account for frequency reduction
        frequency = system.frequency_governor.get_frequency(ISAType.AMX, num_threads) if hasattr(system, 'frequency_governor') else system.cpu_config.base_frequency
        
        return cycles / frequency
        
    def _calculate_vector_compute_time(self, operator, isa_config: ISAConfig,
                                     num_threads: int, system) -> float:
        """Calculate compute time for vector operations"""
        num_ops = operator.get_num_ops()
        
        # Operations per cycle based on vector width and throughput
        data_type = system.base_system.bits if hasattr(system, 'base_system') else 'fp32'
        vector_width = isa_config.vector_width.get(data_type, 1)
        throughput = isa_config.throughput.get('fma', 1.0)
        
        ops_per_cycle = vector_width * throughput
        
        # Total cycles
        total_cycles = num_ops / ops_per_cycle / num_threads
        
        # Get frequency
        frequency = system.frequency_governor.get_frequency(isa_config.name, num_threads) if hasattr(system, 'frequency_governor') else system.cpu_config.base_frequency
        
        return total_cycles / frequency
        
    def get_peak_performance(self, isa: ISAType, data_type: str) -> float:
        """Get peak performance in ops/cycle for ISA and data type"""
        if isa not in self.isa_configs:
            return 1.0
            
        config = self.isa_configs[isa]
        vector_width = config.vector_width.get(data_type, 1)
        throughput = config.throughput.get('fma', 1.0)
        
        return vector_width * throughput 