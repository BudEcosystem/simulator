import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
from .cpu_system import CacheConfig


class AccessPattern(Enum):
    SEQUENTIAL = "sequential"
    RANDOM = "random" 
    STRIDED = "strided"
    TILED = "tiled"


@dataclass
class MemoryAccess:
    """Represents a memory access pattern"""
    address: int
    size: int
    is_read: bool
    stride: int = 1
    

class CacheLevel:
    """Model a single cache level"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.num_sets = config.size // (config.line_size * config.associativity)
        self.tag_bits = 48 - int(np.log2(self.num_sets)) - int(np.log2(config.line_size))
        
        # Cache state
        self.tags = np.zeros((self.num_sets, config.associativity), dtype=np.uint64)
        self.valid = np.zeros((self.num_sets, config.associativity), dtype=bool)
        self.lru = np.zeros((self.num_sets, config.associativity), dtype=np.uint8)
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
    def access(self, address: int, size: int, is_write: bool) -> Tuple[bool, int]:
        """
        Simulate cache access
        Returns: (hit, cycles)
        """
        # Calculate cache line(s) accessed
        start_line = address // self.config.line_size
        end_line = (address + size - 1) // self.config.line_size
        
        total_cycles = 0
        all_hit = True
        
        for line_addr in range(start_line, end_line + 1):
            hit, cycles = self._access_line(line_addr, is_write)
            total_cycles += cycles
            all_hit &= hit
            
        return all_hit, total_cycles
        
    def _access_line(self, line_address: int, is_write: bool) -> Tuple[bool, int]:
        """Access a single cache line"""
        set_index = line_address % self.num_sets
        tag = line_address // self.num_sets
        
        # Check for hit
        for way in range(self.config.associativity):
            if self.valid[set_index, way] and self.tags[set_index, way] == tag:
                # Hit!
                self.hits += 1
                self._update_lru(set_index, way)
                return True, self.config.latency
                
        # Miss
        self.misses += 1
        
        # Find victim (LRU)
        victim_way = np.argmax(self.lru[set_index])
        
        if self.valid[set_index, victim_way]:
            self.evictions += 1
            
        # Install new line
        self.tags[set_index, victim_way] = tag
        self.valid[set_index, victim_way] = True
        self._update_lru(set_index, victim_way)
        
        # Miss penalty depends on where data comes from
        miss_penalty = self.config.latency * 10  # Simplified
        return False, miss_penalty
        
    def _update_lru(self, set_index: int, accessed_way: int):
        """Update LRU counters - higher value means less recently used"""
        # Increment all LRU counters
        self.lru[set_index, :] += 1
        # Reset the accessed way to 0 (most recently used)
        self.lru[set_index, accessed_way] = 0
        

class CacheHierarchy:
    """Model complete cache hierarchy"""
    
    def __init__(self, cpu_config):
        self.l1i = CacheLevel(cpu_config.l1i_config)
        self.l1d = CacheLevel(cpu_config.l1d_config)
        self.l2 = CacheLevel(cpu_config.l2_config)
        self.l3 = CacheLevel(cpu_config.l3_config)
        
        self.cpu_config = cpu_config
        
    def simulate_data_access(self, accesses: List[MemoryAccess]) -> Dict[str, float]:
        """
        Simulate a sequence of memory accesses
        Returns timing breakdown
        """
        total_cycles = 0
        l1_hits = 0
        l2_hits = 0 
        l3_hits = 0
        dram_accesses = 0
        
        for access in accesses:
            # Try L1
            hit, cycles = self.l1d.access(access.address, access.size, not access.is_read)
            if hit:
                l1_hits += 1
                total_cycles += cycles
                continue
                
            # Try L2
            hit, cycles = self.l2.access(access.address, access.size, not access.is_read)
            if hit:
                l2_hits += 1
                total_cycles += self.cpu_config.l1d_config.latency + cycles
                continue
                
            # Try L3
            hit, cycles = self.l3.access(access.address, access.size, not access.is_read)
            if hit:
                l3_hits += 1
                total_cycles += (self.cpu_config.l1d_config.latency + 
                               self.cpu_config.l2_config.latency + cycles)
            else:
                # DRAM access
                dram_accesses += 1
                total_cycles += (self.cpu_config.l1d_config.latency +
                               self.cpu_config.l2_config.latency +
                               self.cpu_config.l3_config.latency + 200)  # ~200 cycles for DRAM
                
        return {
            'total_cycles': total_cycles,
            'l1_hit_rate': l1_hits / len(accesses) if accesses else 0,
            'l2_hit_rate': l2_hits / len(accesses) if accesses else 0,
            'l3_hit_rate': l3_hits / len(accesses) if accesses else 0,
            'dram_access_rate': dram_accesses / len(accesses) if accesses else 0
        }
        
    def analyze_operator_access_pattern(self, operator) -> List[MemoryAccess]:
        """Convert operator access pattern to memory accesses"""
        accesses = []
        
        # Analyze based on operator type
        op_type = operator.__class__.__name__
        
        if op_type == 'GEMM':
            # GEMM has predictable access patterns
            M, N, K = operator.get_gemms()[:3]
            
            # Simplified access pattern for A matrix (M x K)
            # In reality, this would be tiled for cache blocking
            for i in range(min(M, 16)):  # Limit for testing
                for k in range(min(K, 16)):
                    addr = i * K + k
                    accesses.append(MemoryAccess(addr * 4, 4, True))  # 4 bytes for FP32
                    
            # Access pattern for B matrix (K x N) - different base address
            base_b = M * K * 4
            for k in range(min(K, 16)):
                for j in range(min(N, 16)):
                    addr = base_b + k * N + j
                    accesses.append(MemoryAccess(addr * 4, 4, True))
                    
        elif op_type == 'Logit':
            # Attention patterns are more complex
            B, H, M, N, D, _ = operator.dim[:6]
            
            # Simplified Q x K^T access pattern
            for b in range(min(B, 2)):
                for h in range(min(H, 2)):
                    for i in range(min(M, 8)):
                        for j in range(min(N, 8)):
                            # Access Q[b,h,i,:] and K[b,h,j,:]
                            q_addr = ((b * H + h) * M + i) * D
                            k_addr = ((b * H + h) * N + j) * D
                            
                            # D contiguous accesses (simplified)
                            accesses.append(MemoryAccess(q_addr * 4, min(D, 16) * 4, True, D))
                            accesses.append(MemoryAccess(k_addr * 4, min(D, 16) * 4, True, D))
                            
        return accesses 