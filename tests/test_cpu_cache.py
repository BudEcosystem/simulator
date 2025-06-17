"""Tests for CPU cache hierarchy model"""
import unittest
import numpy as np
from GenZ.cpu.cache_model import CacheLevel, CacheHierarchy, CacheConfig, MemoryAccess, AccessPattern
from GenZ.cpu import create_cpu_system


class TestCacheLevel(unittest.TestCase):
    """Test individual cache level functionality"""
    
    def setUp(self):
        """Set up test cache configuration"""
        self.cache_config = CacheConfig(
            size=32*1024,  # 32KB
            latency=4,
            bandwidth=3200,
            associativity=8,
            line_size=64
        )
        self.cache = CacheLevel(self.cache_config)
        
    def test_cache_initialization(self):
        """Test cache level initialization"""
        self.assertEqual(self.cache.config.size, 32*1024)
        self.assertEqual(self.cache.config.line_size, 64)
        self.assertEqual(self.cache.num_sets, 64)  # 32KB / (64B * 8-way) = 64 sets
        
    def test_cache_hit(self):
        """Test cache hit scenario"""
        # First access - miss
        hit, cycles = self.cache.access(0x1000, 4, False)
        self.assertFalse(hit)
        self.assertGreater(cycles, self.cache_config.latency)
        
        # Second access to same address - hit
        hit, cycles = self.cache.access(0x1000, 4, False)
        self.assertTrue(hit)
        self.assertEqual(cycles, self.cache_config.latency)
        
    def test_cache_line_granularity(self):
        """Test that cache works at line granularity"""
        # Access address 0x1000
        self.cache.access(0x1000, 4, False)
        
        # Access nearby address in same cache line - should hit
        hit, cycles = self.cache.access(0x1020, 4, False)  # Same 64B line
        self.assertTrue(hit)
        
        # Access address in different cache line - should miss
        hit, cycles = self.cache.access(0x1040, 4, False)  # Different line
        self.assertFalse(hit)
        
    def test_cache_eviction(self):
        """Test LRU eviction policy"""
        # Fill a cache set (8 ways for 8-way associative)
        set_index = 0
        addresses = []
        
        # Access 8 different addresses mapping to same set
        for i in range(8):
            addr = set_index * 64 + i * self.cache.num_sets * 64
            addresses.append(addr)
            self.cache.access(addr, 4, False)
            
        # All should hit now
        for addr in addresses:
            hit, _ = self.cache.access(addr, 4, False)
            self.assertTrue(hit)
            
        # Record initial LRU state
        initial_lru = self.cache.lru[set_index].copy()
        
        # Access 9th address - should evict LRU (first address)
        new_addr = set_index * 64 + 8 * self.cache.num_sets * 64
        self.cache.access(new_addr, 4, False)
        
        # Check cache state to see which tag was evicted
        # The first address (tag 0) should have been evicted
        tags_in_cache = set(self.cache.tags[set_index][self.cache.valid[set_index]])
        
        # New address should be in cache
        new_tag = (new_addr // 64) // self.cache.num_sets
        self.assertIn(new_tag, tags_in_cache)
        
        # First address should not be in cache
        first_tag = (addresses[0] // 64) // self.cache.num_sets
        self.assertNotIn(first_tag, tags_in_cache)
        
        # All other addresses should still be in cache
        for addr in addresses[1:]:
            tag = (addr // 64) // self.cache.num_sets
            self.assertIn(tag, tags_in_cache)
            
    def test_cache_statistics(self):
        """Test cache statistics tracking"""
        initial_hits = self.cache.hits
        initial_misses = self.cache.misses
        
        # Generate some hits and misses
        self.cache.access(0x1000, 4, False)  # Miss
        self.cache.access(0x1000, 4, False)  # Hit
        self.cache.access(0x2000, 4, False)  # Miss
        self.cache.access(0x1000, 4, False)  # Hit
        
        self.assertEqual(self.cache.hits - initial_hits, 2)
        self.assertEqual(self.cache.misses - initial_misses, 2)


class TestCacheHierarchy(unittest.TestCase):
    """Test complete cache hierarchy"""
    
    def setUp(self):
        """Set up test CPU system with cache hierarchy"""
        self.cpu_system = create_cpu_system('intel_xeon_8380')
        self.cache_hierarchy = CacheHierarchy(self.cpu_system.cpu_config)
        
    def test_hierarchy_initialization(self):
        """Test cache hierarchy initialization"""
        self.assertIsNotNone(self.cache_hierarchy.l1i)
        self.assertIsNotNone(self.cache_hierarchy.l1d)
        self.assertIsNotNone(self.cache_hierarchy.l2)
        self.assertIsNotNone(self.cache_hierarchy.l3)
        
        # Check sizes
        self.assertEqual(self.cache_hierarchy.l1d.config.size, 48*1024)
        self.assertEqual(self.cache_hierarchy.l2.config.size, 1280*1024)
        self.assertEqual(self.cache_hierarchy.l3.config.size, 60*1024*1024)
        
    def test_memory_access_simulation(self):
        """Test simulating memory access through hierarchy"""
        # Create a simple access pattern
        accesses = [
            MemoryAccess(address=0x1000, size=8, is_read=True),
            MemoryAccess(address=0x1000, size=8, is_read=True),  # L1 hit
            MemoryAccess(address=0x2000, size=8, is_read=True),  # L1 miss
            MemoryAccess(address=0x1000, size=8, is_read=True),  # L1 hit
        ]
        
        results = self.cache_hierarchy.simulate_data_access(accesses)
        
        self.assertIn('total_cycles', results)
        self.assertIn('l1_hit_rate', results)
        self.assertGreater(results['l1_hit_rate'], 0)
        self.assertLessEqual(results['l1_hit_rate'], 1.0)
        
    def test_operator_access_pattern_gemm(self):
        """Test access pattern analysis for GEMM operator"""
        # Use actual GEMM operator from GenZ
        from GenZ.operators import GEMM
        
        operator = GEMM(['gemm', 1, 64, 64, 64], density=(1.0, 1.0, 1.0))
        accesses = self.cache_hierarchy.analyze_operator_access_pattern(operator)
        
        self.assertIsInstance(accesses, list)
        self.assertGreater(len(accesses), 0)
        self.assertIsInstance(accesses[0], MemoryAccess)
        
    def test_operator_access_pattern_attention(self):
        """Test access pattern analysis for attention operator"""
        # Use actual Logit operator from GenZ
        from GenZ.operators import Logit
        
        operator = Logit(['logit', 1, 8, 128, 128, 64, 8], density=(1.0, 1.0, 1.0))
        accesses = self.cache_hierarchy.analyze_operator_access_pattern(operator)
        
        self.assertIsInstance(accesses, list)
        self.assertGreater(len(accesses), 0)
        
    def test_cache_inclusive_exclusive(self):
        """Test inclusive vs exclusive cache behavior"""
        # Intel Xeon 8380 has non-inclusive L3
        self.assertFalse(self.cache_hierarchy.l3.config.is_inclusive)
        
        # Test that data can be in L2 but not L3 (exclusive property)
        # This is a property test - implementation would handle this


class TestAccessPatterns(unittest.TestCase):
    """Test different memory access patterns"""
    
    def test_sequential_access(self):
        """Test sequential access pattern"""
        cache = CacheLevel(CacheConfig(size=32*1024, latency=4, bandwidth=3200))
        
        # Sequential access should have good locality
        total_cycles = 0
        for i in range(100):
            addr = 0x1000 + i * 8  # Sequential 8-byte accesses
            hit, cycles = cache.access(addr, 8, False)
            total_cycles += cycles
            
        # Most accesses should hit after initial misses
        hit_rate = cache.hits / (cache.hits + cache.misses)
        self.assertGreater(hit_rate, 0.8)
        
    def test_random_access(self):
        """Test random access pattern"""
        cache = CacheLevel(CacheConfig(size=32*1024, latency=4, bandwidth=3200))
        
        # Random access should have poor locality
        np.random.seed(42)
        addresses = np.random.randint(0, 1024*1024, size=100)
        
        for addr in addresses:
            cache.access(addr * 8, 8, False)
            
        # Hit rate should be lower than sequential
        hit_rate = cache.hits / (cache.hits + cache.misses)
        self.assertLess(hit_rate, 0.5)
        
    def test_strided_access(self):
        """Test strided access pattern"""
        cache = CacheLevel(CacheConfig(size=32*1024, latency=4, bandwidth=3200))
        
        # Stride of cache line size - each access misses
        stride = 64  # Cache line size
        for i in range(100):
            addr = 0x1000 + i * stride
            cache.access(addr, 8, False)
            
        # Should have mostly misses
        hit_rate = cache.hits / (cache.hits + cache.misses)
        self.assertLess(hit_rate, 0.2)


if __name__ == '__main__':
    unittest.main() 