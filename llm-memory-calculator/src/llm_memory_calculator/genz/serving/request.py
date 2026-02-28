"""Request and Batch dataclasses for the serving simulation.

Tracks the full lifecycle of inference requests including timing metrics
(TTFT, TPOT, E2E, ITL) and organizes requests into batches for scheduling.
"""
from dataclasses import dataclass, field
from typing import List

from .constants import RequestStatus


@dataclass
class Request:
    """A single inference request flowing through the serving pipeline.

    Tracks state transitions and accumulates latency metrics as the request
    progresses through: ARRIVED -> QUEUED -> PREFILLING -> DECODING -> COMPLETE.
    """
    request_id: int
    model: str
    input_tokens: int
    max_output_tokens: int
    arrival_time_ns: int
    instance_id: int = 0
    status: RequestStatus = RequestStatus.ARRIVED
    tokens_generated: int = 0
    ttft_ns: int = 0
    tpot_ns: int = 0
    e2e_ns: int = 0
    queuing_delay_ns: int = 0
    itl_ns: List[int] = field(default_factory=list)
    prefill_start_ns: int = 0
    decode_start_ns: int = 0
    completion_time_ns: int = 0

    def set_queued(self, time_ns: int) -> None:
        """Transition to QUEUED state and record queuing delay."""
        self.status = RequestStatus.QUEUED
        self.queuing_delay_ns = time_ns - self.arrival_time_ns

    def set_prefilling(self, time_ns: int) -> None:
        """Transition to PREFILLING state, record prefill start and queuing delay."""
        self.status = RequestStatus.PREFILLING
        self.prefill_start_ns = time_ns
        self.queuing_delay_ns = time_ns - self.arrival_time_ns

    def set_decoding(self, time_ns: int) -> None:
        """Transition to DECODING state, record decode start and TTFT."""
        self.status = RequestStatus.DECODING
        self.decode_start_ns = time_ns
        self.ttft_ns = time_ns - self.arrival_time_ns

    def record_token(self, time_ns: int) -> None:
        """Record generation of one output token and its inter-token latency."""
        self.tokens_generated += 1
        if len(self.itl_ns) == 0:
            self.itl_ns.append(time_ns - self.decode_start_ns)
        else:
            last_token_time = self.decode_start_ns + sum(self.itl_ns)
            self.itl_ns.append(time_ns - last_token_time)

    def set_complete(self, time_ns: int) -> None:
        """Transition to COMPLETE state and finalize all timing metrics."""
        self.status = RequestStatus.COMPLETE
        self.completion_time_ns = time_ns
        self.e2e_ns = time_ns - self.arrival_time_ns
        if self.tokens_generated > 0:
            decode_duration = time_ns - self.decode_start_ns
            self.tpot_ns = decode_duration // self.tokens_generated

    def set_failed(self, time_ns: int) -> None:
        """Transition to FAILED state and record failure timing."""
        self.status = RequestStatus.FAILED
        self.completion_time_ns = time_ns
        self.e2e_ns = time_ns - self.arrival_time_ns


@dataclass
class Batch:
    """A batch of requests scheduled for execution together.

    Separates requests into prefill and decode groups to support
    continuous batching / chunked-prefill scheduling strategies.
    """
    batch_id: int
    model: str
    requests: List[Request] = field(default_factory=list)
    prefill_requests: List[Request] = field(default_factory=list)
    decode_requests: List[Request] = field(default_factory=list)

    @property
    def prefill_count(self) -> int:
        """Number of requests in the prefill phase."""
        return len(self.prefill_requests)

    @property
    def decode_count(self) -> int:
        """Number of requests in the decode phase."""
        return len(self.decode_requests)

    @property
    def size(self) -> int:
        """Total number of requests in the batch."""
        return len(self.requests)
