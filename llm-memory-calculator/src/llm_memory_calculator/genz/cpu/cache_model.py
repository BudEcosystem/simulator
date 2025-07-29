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
            
            # Use sampling approach for large matrices
            # Scale samples with matrix size
            total_accesses_a = M * K
            total_accesses_b = K * N
            
            # Use sqrt scaling for balanced performance
            num_samples_a = min(max(100, int(np.sqrt(total_accesses_a) * 5)), 10000)
            num_samples_b = min(max(100, int(np.sqrt(total_accesses_b) * 5)), 10000)
            
            # Access pattern for A matrix (M x K) using random sampling
            for _ in range(num_samples_a):
                i = np.random.randint(0, M)
                k = np.random.randint(0, K)
                addr = i * K + k
                accesses.append(MemoryAccess(addr * 4, 4, True))  # 4 bytes for FP32
                    
            # Access pattern for B matrix (K x N) - different base address
            base_b = M * K * 4
            for _ in range(num_samples_b):
                k = np.random.randint(0, K)
                j = np.random.randint(0, N)
                addr = base_b + k * N + j
                accesses.append(MemoryAccess(addr * 4, 4, True))
                    
        elif op_type == 'Logit':
            # Attention patterns are more complex
            B, H, M, N, D, _ = operator.dim[:6]
            
            # Distinguish between prefill (M=N=seq_len) and decode (M=1, N=KV_len)
            is_decode = (M == 1)
            
            if is_decode:
                # Decode: M=1, N=KV_size. Need to model full KV cache read
                # Each decode step reads the entire KV cache
                total_kv_accesses = B * H * N * D * 2  # K and V reads
                
                # For decode, use higher sampling rate to capture KV scaling
                # Scale with batch size to model contention
                sample_rate = min(0.1, 50000 / total_kv_accesses) if total_kv_accesses > 1000 else 1.0
                num_samples = max(1000, int(total_kv_accesses * sample_rate))
                
                # Model decode KV cache access pattern
                for _ in range(min(num_samples, 50000)):
                    b = np.random.randint(0, B)
                    h = np.random.randint(0, H)
                    n = np.random.randint(0, N)  # Position in KV cache
                    
                    # Access K[b,h,n,:] and V[b,h,n,:]
                    k_addr = ((b * H + h) * N + n) * D
                    v_addr = k_addr + B * H * N * D  # V follows K in memory
                    
                    # Sample within the D dimension
                    d_accesses = min(D, 64)
                    accesses.append(MemoryAccess(k_addr * 4, d_accesses * 4, True, D))
                    accesses.append(MemoryAccess(v_addr * 4, d_accesses * 4, True, D))
            else:
                # Prefill: M=N=seq_len, quadratic scaling
                total_accesses = B * H * M * N  # Total Q*K^T operations
                
                # Calculate number of samples to maintain scaling
                # Use sqrt scaling to balance accuracy and performance
                base_samples = int(np.sqrt(total_accesses) * 10)
                num_samples = min(max(100, base_samples), 50000)  # Cap at 50k for very large sequences
                
                # Q x K^T access pattern using random sampling
                for _ in range(num_samples):
                    b = np.random.randint(0, B)
                    h = np.random.randint(0, H)
                    i = np.random.randint(0, M)
                    j = np.random.randint(0, N)
                    
                    # Access Q[b,h,i,:] and K[b,h,j,:]
                    q_addr = ((b * H + h) * M + i) * D
                    k_addr = ((b * H + h) * N + j) * D
                    
                    # Simulate accessing part of the D dimension
                    d_accesses = min(D, 64)  # Access up to 64 elements
                    accesses.append(MemoryAccess(q_addr * 4, d_accesses * 4, True, D))
                    accesses.append(MemoryAccess(k_addr * 4, d_accesses * 4, True, D))
                
        elif op_type == 'Attend':
            # Attend operation (processing attention scores)
            B, H, M, N, D, _ = operator.dim[:6]
            
            # For decode phase (M=1), add special handling
            if M == 1:
                # Decode: read entire KV cache
                total_kv_accesses = B * H * N * D * 2  # K and V
                sample_rate = min(0.05, 10000 / total_kv_accesses) if total_kv_accesses > 1000 else 1.0
                num_samples = max(100, int(total_kv_accesses * sample_rate))
                
                for _ in range(min(num_samples, 10000)):
                    b = np.random.randint(0, B)
                    h = np.random.randint(0, H)
                    n = np.random.randint(0, N)  # KV position
                    
                    # Access K and V cache
                    k_addr = ((b * H + h) * N + n) * D
                    v_addr = k_addr + B * H * N * D  # V follows K in memory
                    
                    d_accesses = min(D, 64)
                    accesses.append(MemoryAccess(k_addr * 4, d_accesses * 4, True, D))
                    accesses.append(MemoryAccess(v_addr * 4, d_accesses * 4, True, D))
            else:
                # Prefill: similar to Logit pattern
                total_accesses = B * H * M * N
                sample_rate = min(0.01, 10000 / total_accesses) if total_accesses > 1000 else 1.0
                num_samples = max(100, int(total_accesses * sample_rate))
                
                for _ in range(min(num_samples, 10000)):
                    b = np.random.randint(0, B)
                    h = np.random.randint(0, H)
                    i = np.random.randint(0, M)
                    j = np.random.randint(0, N)
                    
                    # Access attention scores
                    addr = ((b * H + h) * M + i) * N + j
                    accesses.append(MemoryAccess(addr * 4, 4, True))
                    
        elif op_type == 'FC':
            # Fully connected layer
            M = operator.get_M()
            N = operator.get_N()
            K = operator.get_K()
            
            # Similar to GEMM but often smaller
            total_accesses = M * K + K * N
            sample_rate = min(0.1, 10000 / total_accesses) if total_accesses > 100 else 1.0
            num_samples = max(50, int(total_accesses * sample_rate))
            
            for _ in range(min(num_samples, 5000)):
                # Randomly sample input access
                if np.random.random() < 0.5:
                    # Input matrix access
                    i = np.random.randint(0, M)
                    k = np.random.randint(0, K)
                    addr = i * K + k
                else:
                    # Weight matrix access
                    k = np.random.randint(0, K)
                    j = np.random.randint(0, N)
                    addr = M * K + k * N + j
                    
                accesses.append(MemoryAccess(addr * 4, 4, True))
        else:
            # Default pattern for other operators
            # Estimate based on compute complexity
            if hasattr(operator, 'get_gemms'):
                M, N, K = operator.get_gemms()[:3]
                total_size = M * N
            else:
                total_size = 1000  # Default
                
            num_samples = min(100, total_size)
            for i in range(num_samples):
                addr = i * 64  # Assume some stride
                accesses.append(MemoryAccess(addr, 4, True))
                            
        return accesses 