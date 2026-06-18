import logging
import os
from typing import Optional, Dict, Tuple
import numpy as np
from .cache_model import AccessPattern

# Setup a dedicated logger for CPU performance debugging
cpu_log_file = '/tmp/cpu_perf_log.txt'
cpu_logger = logging.getLogger('cpu_perf')
cpu_logger.setLevel(logging.DEBUG)

# Avoid adding multiple handlers if the module is reloaded
if not cpu_logger.handlers:
    # Clear log file at the start of a run
    if os.path.exists(cpu_log_file):
        os.remove(cpu_log_file)
    handler = logging.FileHandler(cpu_log_file)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    cpu_logger.addHandler(handler)


class CPUOperatorMixin:
    """Mixin to add CPU-specific functionality to operators"""
    
    def get_cpu_memory_time(self, system: 'CPUSystem') -> float:
        """Memory time = the physical bandwidth-limited roofline over the per-cache-level data volume.

        R2-CPU2: the prior implementation took ``max(total_cycles/base_frequency, bandwidth_time)``,
        where ``total_cycles`` came from a hand-rolled per-line cache simulator driven by an *unseeded*
        RNG. For memory-bound decode that fictional cycle term won the ``max()`` and produced a
        non-deterministic latency ~20x slower than the physical roofline. That second timing model is
        removed: memory time is now purely the bandwidth roofline (the cache hit-rate columns are kept
        only as informational metadata, computed from a now-seeded RNG so they are reproducible).
        """
        cache_sim = system.cache_hierarchy
        accesses = cache_sim.analyze_operator_access_pattern(self)
        results = cache_sim.simulate_data_access(accesses)

        data_volume = self._calculate_data_volume(system)
        final_memory_time = self._calculate_bandwidth_limited_time(
            data_volume, results, system
        )

        cpu_logger.debug(f"MEMORY_TIME;{self.__class__.__name__};{final_memory_time}")
        return final_memory_time

    def _get_batch_size(self) -> int:
        """Recover the batch (B) dimension of this operator.

        Under GenZ's layout the leading post-name dim is always the batch slot: FC dim=[B,O,I],
        GEMM dim=[B,M,N,K], Logit/Attend dim=[B,H,M,N,D,Hkv] (see operators.py). So ``self.dim[0]``
        is B for every weight-projection / attention op the prefill+decode graphs emit. Returns 1 if
        the dim is unavailable (conservative: treat as a single stream)."""
        d = getattr(self, 'dim', None)
        if d is not None and len(d) >= 1:
            try:
                return int(d[0])
            except Exception:
                return 1
        return 1

    def _is_single_stream_decode(self) -> bool:
        """A decode op is bound by ONE socket's DRAM bandwidth only when it is a SINGLE stream
        (batch==1). Physics: under default first-touch NUMA the streamed operand (the weight tensor
        for FC/GEMM; the KV cache for Logit/Attend) is allocated in one node, so a lone batch=1
        decode token sees a single socket's channels. A BATCHED decode (batch>1) runs one thread
        group per request, the runtime spreads those threads across all sockets, and the streamed
        operand (KV cache scales with batch; weights are re-read by every socket's threads) is served
        from the AGGREGATE multi-socket bandwidth. R2-CPU5: the previous ``_is_decode_op`` flagged
        decode purely by the token dim (get_gemms()[0]==1 for FC/GEMM, or dim[2]==1 for attention),
        so a batched decode's attention ops (dim[2]==1 for ANY batch) were charged single-socket
        bandwidth — under-counting bandwidth on multi-socket parts. The single-socket roofline now
        applies ONLY to batch==1; batched decode falls through to the aggregate band, same as prefill.

        Decode-token detection (token dim == 1) keeps the original convention: FC/GEMM expose the
        token dim in get_gemms()[0] (FC: B; decode B==1 collapses it), attention in dim[2]==M."""
        # Token-dimension decode detection (unchanged convention).
        M = None
        try:
            g = self.get_gemms()
            if g:
                M = g[0]
        except Exception:
            pass
        if M is None:
            d = getattr(self, 'dim', None)
            if d is not None and len(d) >= 3:
                M = d[2]
        try:
            is_decode_token = int(M) == 1
        except Exception:
            return False
        if not is_decode_token:
            return False
        # Single-socket binding only for a single (batch==1) stream.
        return self._get_batch_size() <= 1

    # Backward-compatible alias: callers/tests referencing the old name get the batch-aware behavior.
    def _is_decode_op(self) -> bool:
        return self._is_single_stream_decode()
        
    def _determine_access_pattern(self) -> AccessPattern:
        """Determine the memory access pattern for this operator"""
        op_type = self.__class__.__name__
        
        if op_type in ['GEMM', 'FC']:
            return AccessPattern.TILED  # Assumes cache-blocked GEMM
        elif op_type in ['Logit', 'Attend']:
            # Attention has less regular patterns
            return AccessPattern.STRIDED
        else:
            return AccessPattern.SEQUENTIAL
            
    def _calculate_data_volume(self, system=None) -> Dict[str, float]:
        """Calculate data volume for this operator"""
        # Derive bytes_per_element from system precision
        _precision_bytes = {'fp32': 4, 'f32': 4, 'bf16': 2, 'fp16': 2, 'int8': 1, 'int4': 0.5}
        data_type = 'fp32'
        if system is not None and hasattr(system, 'base_system'):
            data_type = getattr(system.base_system, 'bits', 'fp32')
        bytes_per_element = _precision_bytes.get(data_type, 4)

        # Get tensors
        tensors = self.get_tensors()
        if len(tensors) >= 3:
            input_a, input_w, output = tensors[:3]

            input_a_size = np.prod(input_a) * bytes_per_element
            input_w_size = np.prod(input_w) * bytes_per_element
            output_size = np.prod(output) * bytes_per_element

            return {
                'input_a': input_a_size,
                'input_w': input_w_size,
                'output': output_size,
                'total': input_a_size + input_w_size + output_size
            }
        else:
            # Fallback for operators without standard tensor structure
            num_ops = self.get_num_ops()
            return {'total': num_ops * bytes_per_element}
            
    def _calculate_bandwidth_limited_time(self, data_volume: Dict[str, float],
                                        cache_results: Dict[str, float],
                                        system: 'CPUSystem') -> float:
        """Physical bandwidth roofline over the operator's unique data footprint.

        R2-CPU2/CPU3: time is derived from a DETERMINISTIC capacity model — does the unique footprint
        fit in a cache level? — instead of the sampled cache hit-rates, which found false locality and
        under-counted the DRAM weight-stream for decode (a 14 GB weight tensor cannot reside in a
        ~0.6 GB L3, yet the sampler reported it ~70% cache-resident, making decode ~3x too fast). A
        single operator pass reads its footprint once: above L3 it is a pure DRAM stream (compulsory
        misses, no intra-op reuse); otherwise it is served from the smallest cache level that holds it.
        ``cache_results`` (the seeded sampler's hit-rates) is retained only as informational metadata.
        """
        footprint = data_volume['total']
        if footprint <= 0:
            return 0.0

        l2c = system.cpu_config.l2_config.size

        # Operand-aware split. The WEIGHT operand (and, for attention, the KV cache) is COLD: it is
        # read once per token from memory it was not previously resident in (other layers evicted it),
        # so it is a compulsory DRAM stream regardless of whether a single matrix happens to be smaller
        # than the aggregate L3 — there is no intra-op reuse for an M=1 GEMV. Only the small
        # activations (input_a + output) are cache-resident, overflowing to DRAM if they exceed L2.
        if 'input_w' in data_volume:
            weight_bytes = data_volume.get('input_w', 0.0)
            act_bytes = data_volume.get('input_a', 0.0) + data_volume.get('output', 0.0)
        else:
            # operator without standard (a, w, out) tensors: treat the whole footprint as streamed
            weight_bytes, act_bytes = footprint, 0.0

        dram_volume = weight_bytes + max(0.0, act_bytes - l2c)
        cached_volume = max(0.0, footprint - dram_volume)  # activations served from cache (L3 tier)

        l1_time = 0
        l2_time = 0
        l3_time = cached_volume / (system.cpu_config.l3_config.bandwidth * 1e9) if cached_volume > 0 else 0

        # DRAM bandwidth (R2-CPU3/CPU5): the access-pattern-correct DRAM bandwidth, derated by the
        # sustained-streaming efficiency (DDR band, same source as the GPU/static path's eta_mem). A
        # SINGLE-STREAM batch=1 decode streams the operand from a single NUMA node -> one socket's
        # bandwidth; prefill AND batched decode (batch>1) spread threads across sockets -> the
        # aggregate. The 1/sockets factor is a consequence of first-touch placement, not a tuned
        # constant; the batch>1 -> aggregate selection follows from per-request thread groups spanning
        # sockets (R2-CPU5).
        per_socket_bw = getattr(system, 'per_socket_mem_bw', system.peak_memory_bandwidth)
        aggregate_bw = getattr(system, 'aggregate_mem_bw', system.peak_memory_bandwidth)
        dram_peak_bw = per_socket_bw if self._is_single_stream_decode() else aggregate_bw
        dram_eff = getattr(system, 'dram_efficiency', 1.0)
        dram_time = dram_volume / (dram_peak_bw * dram_eff * 1e9) if dram_volume > 0 else 0

        # Vendor-aware combination. Only one tier is active per pass, so additive (Intel ECM serial
        # data path) and max (AMD/ARM overlapping caches) coincide here; the structure is retained so
        # the model stays vendor-aware for future multi-tier extensions.
        vendor = getattr(system.cpu_config, 'vendor', 'intel')
        if vendor == 'amd':
            return max(l1_time, l2_time, l3_time, dram_time)
        else:
            return l1_time + l2_time + l3_time + dram_time
        
    def get_cpu_compute_time(self, system: 'CPUSystem') -> float:
        """Calculate compute time on CPU system"""
        # Select optimal ISA
        data_type = system.base_system.bits if hasattr(system, 'base_system') else 'fp32'
        isa, efficiency = system.isa_selector.select_isa(self, data_type)
        
        # Get optimal thread configuration
        thread_config = system.threading_model.get_optimal_thread_config(self, system) if hasattr(system, 'threading_model') else None
        
        if thread_config:
            num_threads = thread_config.num_threads
            # Base compute time assuming ideal utilisation
            base_time = system.isa_selector.get_compute_time(
                self, isa, num_threads, system
            )
            # Parallel-efficiency (threading) penalty
            parallel_eff = system.threading_model.calculate_parallel_efficiency(
                self, thread_config, system
            )
        else:
            num_threads = system.get_active_cores()
            base_time = system.isa_selector.get_compute_time(
                self, isa, num_threads, system
            )
            parallel_eff = 1.0

        # ---------------------------------------------------------------------------------
        # NEW: incorporate ISA-level efficiency returned by select_isa (e.g. 0.85 or 0.70).
        # This accounts for pipeline bubbles, vector utilisation and non-FMA instructions
        # that make real kernels slower than the theoretical peak.
        # ---------------------------------------------------------------------------------
        isa_efficiency = efficiency if efficiency > 0 else 0.5  # safety guard

        # Effective compute time is longer when efficiency < 1 and/or parallel_eff < 1.
        effective_time = base_time / (isa_efficiency * parallel_eff)

        cpu_logger.debug(f"COMPUTE_TIME;{self.__class__.__name__};{effective_time}")
        return effective_time
            
    def optimize_for_cpu_cache(self, system: 'CPUSystem') -> 'Operator':
        """Return cache-optimized version of this operator"""
        op_type = self.__class__.__name__
        
        if op_type == 'GEMM':
            return self._optimize_gemm_tiling(system)
        elif op_type in ['Logit', 'Attend']:
            return self._optimize_attention_blocking(system)
        else:
            return self  # No optimization
            
    def _optimize_gemm_tiling(self, system: 'CPUSystem') -> 'Operator':
        """Optimize GEMM tiling for cache hierarchy"""
        # This is a placeholder - in a real implementation,
        # this would return a modified operator with optimal tiling
        return self
        
    def _optimize_attention_blocking(self, system: 'CPUSystem') -> 'Operator':
        """Optimize attention blocking for cache"""
        # This is a placeholder - in a real implementation,
        # this would return a modified operator with optimal blocking
        return self 