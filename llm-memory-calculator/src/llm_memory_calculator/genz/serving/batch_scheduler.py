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

        # Latency cache to avoid redundant GenZ calls. Each entry is the full per-phase
        # signal tuple (latency_ms, is_offload, mfu, mbu) so SV3/SV4 reuse the engine's
        # offload flag and roofline utilization, not just the latency number.
        self._latency_cache: Dict[Tuple, Tuple[float, bool, float, float]] = {}

        # Cached GenZ modeling functions (resolved once); also an injection point for tests.
        self._modeling_fns: Optional[Tuple] = None

        # SV3 / SV4 per-batch signals exposed to the simulator loop:
        #   last_batch_offloaded — True if either phase's GenZ result had is_offload=True
        #   last_batch_mfu/mbu   — real roofline utilization for the batch's dominant phase
        # (computed via the shared roofline_utilization helper on the cached GenZ result).
        self.last_batch_offloaded: bool = False
        self.last_batch_mfu: float = 0.0
        self.last_batch_mbu: float = 0.0

        # Requests that exceed the TOTAL multi-tier memory budget (SV1/SV2): set_failed and
        # drained by the simulator so they are counted as failed, never re-queued forever.
        self._failed: List[Request] = []

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
            # SV1/SV2: a request whose KV footprint exceeds the TOTAL capacity of ALL tiers
            # combined can never be admitted on any tier (allocate_kv_blocks already walks the
            # full HBM->DRAM->HOST_DDR->CXL->NVME hierarchy). This is a PERMANENT property of the
            # request, independent of the current batch fill, so it is checked FIRST — before the
            # max_batch_size / max_num_batched_tokens guards below. (Previously this sat after those
            # guards, so a prompt longer than max_num_batched_tokens tripped the token guard every
            # iteration and was re-queued forever, never reaching the admission check.) The
            # evict->QUEUED retry path below is for requests that DO fit but are momentarily blocked.
            total_cap = getattr(self._memory_model, 'total_capacity_bytes', None)
            if total_cap is not None:
                num_blocks = (req.input_tokens + self._memory_model._block_size - 1) // self._memory_model._block_size
                required_bytes = num_blocks * self._memory_model.bytes_per_block
                if required_bytes > total_cap:
                    req.set_failed(current_time_ns)
                    self._failed.append(req)
                    continue  # dropped: NOT appended to remaining_pending

            # Second permanent-failure mode: a prompt longer than the per-iteration token budget can
            # never fill a batch unless chunked prefill is enabled. Without chunking it would be
            # re-queued forever (neither completed nor failed). Fail it honestly instead of hanging.
            if (not self._config.enable_chunked_prefill
                    and req.input_tokens > self._config.max_num_batched_tokens):
                req.set_failed(current_time_ns)
                self._failed.append(req)
                continue

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

    def drain_failed(self) -> List[Request]:
        """Return and clear requests failed by admission control (SV1/SV2).

        The simulator drains these each loop iteration to count them as
        ``total_requests_failed`` without ever appending them to ``raw_requests``.
        """
        failed = self._failed
        self._failed = []
        return failed

    def _get_modeling_fns(self) -> Tuple:
        """Resolve (and cache) the GenZ prefill/decode modeling functions for this hardware."""
        if self._modeling_fns is None:
            from llm_memory_calculator.genz.serving import get_modeling_functions
            self._modeling_fns = get_modeling_functions(self._hardware)
        return self._modeling_fns

    def estimate_batch_latency_ms(self, batch: Batch) -> float:
        """Estimate batch latency using GenZ analytical engine.

        Calls prefill_moddeling() for the prefill portion and
        decode_moddeling() for the decode portion, returning
        the maximum (they run together in a continuous batch).
        """
        prefill_latency_ms = 0.0
        decode_latency_ms = 0.0

        # SV3/SV4: reset per-batch signals, then OR/MAX in each phase's contribution.
        self.last_batch_offloaded = False
        self.last_batch_mfu = 0.0
        self.last_batch_mbu = 0.0

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

    def _record_phase_signals(self, is_offload: bool, mfu: float, mbu: float) -> None:
        """Fold a single phase's SV3/SV4 signals into the per-batch aggregates.

        ``offloaded`` is the OR across phases; ``mfu``/``mbu`` are the MAX (the dominant phase
        sets the batch's roofline draw, used by the power model in the sim loop)."""
        self.last_batch_offloaded = self.last_batch_offloaded or bool(is_offload)
        self.last_batch_mfu = max(self.last_batch_mfu, mfu)
        self.last_batch_mbu = max(self.last_batch_mbu, mbu)

    def _estimate_prefill_latency(self, batch: Batch) -> float:
        """Estimate prefill latency via GenZ prefill_moddeling().

        Captures (SV3) ``result.is_offload`` and (SV4) the real roofline MFU/MBU so the
        scheduler can surface them; caches the full signal tuple per (phase, bs, tokens, hw...).
        """
        total_prefill_tokens = sum(r.input_tokens for r in batch.prefill_requests)
        avg_input = total_prefill_tokens // batch.prefill_count if batch.prefill_count > 0 else 0
        bs = batch.prefill_count

        cache_key = ("prefill", bs, avg_input, self._hardware, self._precision,
                     self._tensor_parallel, self._pipeline_parallel)
        cached = self._latency_cache.get(cache_key)
        if cached is not None:
            latency, is_offload, mfu, mbu = cached
            self._record_phase_signals(is_offload, mfu, mbu)
            return latency

        is_offload = False
        mfu = mbu = 0.0
        try:
            prefill_fn, _ = self._get_modeling_fns()
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
            is_offload = bool(getattr(result, 'is_offload', False))
            mfu, mbu = self._roofline(result)
        except Exception as e:
            warnings.warn(f"GenZ prefill modeling failed: {e}. Using heuristic estimate.")
            # Heuristic fallback for unresolvable models (e.g. mocked "test"): ~0.01ms/token/item.
            latency = avg_input * 0.01 * bs / max(self._tensor_parallel, 1)

        self._latency_cache[cache_key] = (latency, is_offload, mfu, mbu)
        self._record_phase_signals(is_offload, mfu, mbu)
        return latency

    def _estimate_decode_latency(self, batch: Batch) -> float:
        """Estimate decode latency via GenZ decode_moddeling()."""
        # For decode, each request generates 1 token per iteration
        # Use average context length for KV cache sizing
        avg_context = sum(
            r.input_tokens + r.tokens_generated for r in batch.decode_requests
        ) // max(batch.decode_count, 1)
        bs = batch.decode_count

        # Bucket the context length in the CACHE KEY so consecutive decode steps
        # (avg_context grows ~1 per generated token) reuse the cached GenZ result
        # instead of recomputing every step -- the O(requests x output_tokens)
        # cliff that made serving sims take minutes. The computation below still
        # uses the exact avg_context; only the cache key is bucketed.
        context_bucket = (int(avg_context) // 128) * 128
        cache_key = ("decode", bs, context_bucket, self._hardware, self._precision,
                     self._tensor_parallel, self._pipeline_parallel)
        cached = self._latency_cache.get(cache_key)
        if cached is not None:
            latency, is_offload, mfu, mbu = cached
            self._record_phase_signals(is_offload, mfu, mbu)
            return latency

        is_offload = False
        mfu = mbu = 0.0
        try:
            _, decode_fn = self._get_modeling_fns()
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
            is_offload = bool(getattr(result, 'is_offload', False))
            mfu, mbu = self._roofline(result)
        except Exception as e:
            warnings.warn(f"GenZ decode modeling failed: {e}. Using heuristic estimate.")
            # Heuristic fallback for unresolvable models: ~0.5ms per token for decode.
            latency = 0.5 * bs / max(self._tensor_parallel, 1)

        self._latency_cache[cache_key] = (latency, is_offload, mfu, mbu)
        self._record_phase_signals(is_offload, mfu, mbu)
        return latency

    def _roofline(self, result) -> Tuple[float, float]:
        """SV4: real per-device (mfu, mbu) for ``result`` on this hardware via the shared helper.

        num_devices=1: GenZ's ``summary_table`` is ALREADY per-device (it shards FLOPs/bytes across
        TP, and latency is the sharded per-device time) — verified: summary MACs x tp is constant
        across tp. So MFU = per-device-work / (per-device-peak x time); dividing the peak by tp*pp a
        second time would understate utilization by tp*pp. Returns (0, 0) when the helper cannot
        derive them (e.g. a mocked result with no summary_table) so the power model degrades to idle."""
        from llm_memory_calculator.genz.serving import roofline_utilization
        try:
            return roofline_utilization(result, self._hardware, num_devices=1)
        except Exception:
            return 0.0, 0.0
