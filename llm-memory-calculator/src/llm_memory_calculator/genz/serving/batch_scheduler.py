"""Batch scheduler for LLM serving simulation.

Manages request queuing, batch formation, and latency estimation
using GenZ's analytical prefill/decode modeling functions.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings

from .constants import RequestStatus, MemoryTier, NS_PER_MS
from .request import Request, Batch
from .memory_model import MemoryModel


@dataclass
class SchedulerConfig:
    """Configuration for the batch scheduler."""
    max_batch_size: int = 256
    max_num_batched_tokens: int = 8192
    enable_chunked_prefill: bool = False
    chunk_size: int = 512
    prioritize_prefill: bool = False


class BatchScheduler:
    """Continuous batching scheduler with GenZ analytical latency estimation.

    Manages the lifecycle of requests through prefill and decode phases,
    forming batches that respect memory and token limits. Uses GenZ's
    prefill_moddeling() and decode_moddeling() for latency estimation.
    """

    def __init__(
        self,
        model: str,
        hardware: str,
        memory_model: MemoryModel,
        config: SchedulerConfig,
        precision: str = "bf16",
        tensor_parallel: int = 1,
        pipeline_parallel: int = 1,
    ):
        self._model = model
        self._hardware = hardware
        self._memory_model = memory_model
        self._config = config
        self._precision = precision
        self._tensor_parallel = tensor_parallel
        self._pipeline_parallel = pipeline_parallel

        # Request queues
        self._pending: List[Request] = []  # Waiting for first scheduling
        self._running: List[Request] = []  # In decode phase (between iterations)
        self._batch_counter = 0

        # Latency cache to avoid redundant GenZ calls
        self._latency_cache: Dict[Tuple, float] = {}

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def inflight_count(self) -> int:
        return len(self._running)

    def add_request(self, request: Request) -> None:
        """Add a request to the pending queue."""
        request.set_queued(request.arrival_time_ns)
        self._pending.append(request)

    def schedule(self, current_time_ns: int) -> Optional[Batch]:
        """Form the next batch from pending and running requests.

        Returns None if no requests are available to schedule.
        """
        if not self._pending and not self._running:
            return None

        prefill_requests: List[Request] = []
        decode_requests: List[Request] = []
        total_tokens = 0

        # Add running (decode) requests first — they use 1 token each
        for req in list(self._running):
            if len(prefill_requests) + len(decode_requests) >= self._config.max_batch_size:
                break
            if total_tokens + 1 > self._config.max_num_batched_tokens:
                break
            decode_requests.append(req)
            total_tokens += 1

        # Add pending (prefill) requests
        if self._config.prioritize_prefill:
            # Sort pending by arrival time (FIFO)
            self._pending.sort(key=lambda r: r.arrival_time_ns)

        remaining_pending = []
        for req in self._pending:
            if len(prefill_requests) + len(decode_requests) >= self._config.max_batch_size:
                remaining_pending.append(req)
                continue

            req_tokens = req.input_tokens
            if self._config.enable_chunked_prefill:
                req_tokens = min(req_tokens, self._config.chunk_size)

            if total_tokens + req_tokens > self._config.max_num_batched_tokens:
                remaining_pending.append(req)
                continue

            # Allocate KV cache for input tokens
            try:
                self._memory_model.allocate_kv_blocks(req, req.input_tokens)
            except MemoryError:
                # Attempt eviction before dropping request
                retry_success = False
                if hasattr(self._memory_model, 'evict_blocks'):
                    needed_blocks = (req.input_tokens + self._memory_model._block_size - 1) // self._memory_model._block_size
                    evicted_count, evicted_rids = self._memory_model.evict_blocks(self._memory_model.primary_tier, needed_blocks)
                    # Move evicted running requests back to pending
                    if evicted_rids:
                        still_running = []
                        for running_req in self._running:
                            if running_req.request_id in evicted_rids:
                                running_req.status = RequestStatus.QUEUED
                                self._pending.append(running_req)
                            else:
                                still_running.append(running_req)
                        self._running = still_running
                    if evicted_count > 0:
                        try:
                            self._memory_model.allocate_kv_blocks(req, req.input_tokens)
                            retry_success = True
                        except MemoryError:
                            pass
                if not retry_success:
                    remaining_pending.append(req)
                    continue

            req.set_prefilling(current_time_ns)
            prefill_requests.append(req)
            total_tokens += req_tokens

        self._pending = remaining_pending

        # Remove scheduled decode requests from running
        for req in decode_requests:
            if req in self._running:
                self._running.remove(req)

        if not prefill_requests and not decode_requests:
            return None

        all_requests = prefill_requests + decode_requests
        self._batch_counter += 1

        return Batch(
            batch_id=self._batch_counter,
            model=self._model,
            requests=all_requests,
            prefill_requests=prefill_requests,
            decode_requests=decode_requests,
        )

    def complete_batch(self, batch: Batch, current_time_ns: int) -> List[Request]:
        """Process a completed batch iteration.

        - Prefill requests transition to decode phase
        - Decode requests get one token generated
        - Requests that reached max_output_tokens are completed

        Returns list of newly completed requests.
        """
        completed = []

        # Prefill requests -> transition to decode
        # The prefill step produces the first output token from logits
        for req in batch.prefill_requests:
            req.set_decoding(current_time_ns)
            req.record_token(current_time_ns)  # First token from prefill logits
            if req.tokens_generated >= req.max_output_tokens:
                req.set_complete(current_time_ns)
                self._memory_model.deallocate_kv_blocks(req)
                completed.append(req)
            else:
                self._running.append(req)

        # Decode requests -> generate one token
        for req in batch.decode_requests:
            req.record_token(current_time_ns)
            if req.tokens_generated >= req.max_output_tokens:
                req.set_complete(current_time_ns)
                self._memory_model.deallocate_kv_blocks(req)
                completed.append(req)
            else:
                # Still decoding, put back in running
                self._running.append(req)

        return completed

    def estimate_batch_latency_ms(self, batch: Batch) -> float:
        """Estimate batch latency using GenZ analytical engine.

        Calls prefill_moddeling() for the prefill portion and
        decode_moddeling() for the decode portion, returning
        the maximum (they run together in a continuous batch).
        """
        prefill_latency_ms = 0.0
        decode_latency_ms = 0.0

        if batch.prefill_count > 0:
            prefill_latency_ms = self._estimate_prefill_latency(batch)

        if batch.decode_count > 0:
            decode_latency_ms = self._estimate_decode_latency(batch)

        # In continuous batching, prefill and decode run in the same iteration
        if prefill_latency_ms > 0 and decode_latency_ms > 0:
            if self._config.enable_chunked_prefill:
                # With chunked prefill, chunks are sized to fit within one decode step
                # so prefill and decode overlap on the GPU. Only scheduling overhead
                # adds to the dominant phase.
                # Overhead: CUDA kernel launch + block table updates (Sarathi-Serve OSDI 2024)
                scheduling_overhead_ms = 0.1 + 0.01 * batch.size
                return max(prefill_latency_ms, decode_latency_ms) + scheduling_overhead_ms
            else:
                # Without chunked prefill, prefill and decode use different CUDA
                # kernels that execute sequentially (Orca-style)
                return prefill_latency_ms + decode_latency_ms
        return prefill_latency_ms + decode_latency_ms

    def _estimate_prefill_latency(self, batch: Batch) -> float:
        """Estimate prefill latency via GenZ prefill_moddeling()."""
        total_prefill_tokens = sum(r.input_tokens for r in batch.prefill_requests)
        avg_input = total_prefill_tokens // batch.prefill_count if batch.prefill_count > 0 else 0
        bs = batch.prefill_count

        cache_key = ("prefill", bs, avg_input, self._hardware, self._precision,
                     self._tensor_parallel, self._pipeline_parallel)
        if cache_key in self._latency_cache:
            return self._latency_cache[cache_key]

        try:
            from llm_memory_calculator.genz.serving import get_modeling_functions
            prefill_fn, _ = get_modeling_functions(self._hardware)
            result = prefill_fn(
                model=self._model,
                batch_size=bs,
                input_tokens=avg_input,
                system_name=self._hardware,
                bits=self._precision,
                tensor_parallel=self._tensor_parallel,
                pipeline_parallel=self._pipeline_parallel,
            )
            latency = result.Latency  # in milliseconds
        except Exception as e:
            warnings.warn(f"GenZ prefill modeling failed: {e}. Using heuristic estimate.")
            # Heuristic fallback: ~0.01ms per token per batch item
            latency = avg_input * 0.01 * bs / max(self._tensor_parallel, 1)

        self._latency_cache[cache_key] = latency
        return latency

    def _estimate_decode_latency(self, batch: Batch) -> float:
        """Estimate decode latency via GenZ decode_moddeling()."""
        # For decode, each request generates 1 token per iteration
        # Use average context length for KV cache sizing
        avg_context = sum(
            r.input_tokens + r.tokens_generated for r in batch.decode_requests
        ) // max(batch.decode_count, 1)
        bs = batch.decode_count

        cache_key = ("decode", bs, avg_context, self._hardware, self._precision,
                     self._tensor_parallel, self._pipeline_parallel)
        if cache_key in self._latency_cache:
            return self._latency_cache[cache_key]

        try:
            from llm_memory_calculator.genz.serving import get_modeling_functions
            _, decode_fn = get_modeling_functions(self._hardware)
            result = decode_fn(
                model=self._model,
                batch_size=bs,
                input_tokens=avg_context,
                output_tokens=0,
                system_name=self._hardware,
                bits=self._precision,
                tensor_parallel=self._tensor_parallel,
                pipeline_parallel=self._pipeline_parallel,
            )
            latency = result.Latency  # in milliseconds
        except Exception as e:
            warnings.warn(f"GenZ decode modeling failed: {e}. Using heuristic estimate.")
            # Heuristic fallback: ~0.5ms per token for decode
            latency = 0.5 * bs / max(self._tensor_parallel, 1)

        self._latency_cache[cache_key] = latency
        return latency
