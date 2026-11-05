"""T1 is planned only where the node's free host memory holds the pod's whole request (FRD-023 FR-T1).

Without T1 a cuda pod requests no host memory. With T1 it requests its engine base (host memory per rank, the
framework floor, the compile peak, /dev/shm), the T1 region, and the T3 client buffers when there is a T3 tier
(``legacy.host_base_bytes``). The planner sized T1 as a share of the free host memory and ignored the rest, so on
accubits (2026-11-02) it planned 4 GiB of T1 on a node with 8.02 GiB free, and the pod's 13.15 GiB request was refused
by the deploy's node-memory check. The base now comes off the free memory before T1 is sized.

The base is the pod's engine limit less T1, which the chart requests whole: it includes the weights' host copy
(``legacy.limit_headroom_mib``, capped at 2 GiB). Leaving that out still over-sized T1 by the weights, so on accubits
(2026-11-04) a 0.5B model's 16.11 GiB pod was refused on a node with 15.27 GiB unreserved.
"""

import math

import pytest
from tests.test_kv_tiering import QWEN3_06B, _chat, _plan, _t1

from llm_memory_calculator.kv.constants import GIB, T1_HOST_RAM_SHARE
from llm_memory_calculator.runtime.modules.legacy import MIB, host_base_bytes, limit_headroom_mib

TCP_POOL = {"backend": "mooncake", "transport": "tcp", "read_gbps": 1.45, "capacity_gib": 512}
T3_ENGINE = ["kv_events", "cache_salt", "cpu_offload_shm", "mooncake_store"]

WEIGHT_GIB = 0.45  # per rank, as _small plans it
HEADROOM_GIB = limit_headroom_mib(WEIGHT_GIB * GIB, 1) * MIB / GIB
BASE_GIB = host_base_bytes(1) / GIB + HEADROOM_GIB  # 7.75 + the weights' host copy at TP 1
BASE_T3_GIB = host_base_bytes(1, t3=True) / GIB + HEADROOM_GIB  # and 1 GiB more for the T3 client


def _floor_q(gib):
    """T1 sizes come in quarter-GiB steps."""
    return math.floor(gib * 4) / 4


def _small(host_free_gib, **over):
    """Qwen-0.6B on a 4 GiB card, as on accubits: T1 pays for itself here when the memory allows it."""
    args = dict(
        model_config=QWEN3_06B,
        device_memory_gib=4.0,
        weight_gib_per_rank=WEIGHT_GIB,
        kv_demand_gib_per_rank=0.3,
        device_tflops=17.0,
        workload=_chat(concurrency=1, input_tokens=1024, output_tokens=256),
        host_free_gib=host_free_gib,
    )
    args.update(over)
    return _plan(**args)


def test_the_base_request_is_the_one_the_runtime_spec_renders():
    """One formula: 2.5 GiB a rank, 3 GiB framework, 2 GiB compile peak, the /dev/shm floor, 1 GiB a rank for T3."""
    assert host_base_bytes(1) / GIB == pytest.approx(2.5 + 3 + 2 + 0.25)
    assert HEADROOM_GIB == pytest.approx(WEIGHT_GIB, abs=1 / 1024)
    assert BASE_T3_GIB - BASE_GIB == pytest.approx(1.0)
    assert host_base_bytes(2) - host_base_bytes(1) == (2.5 + 0.25) * GIB


def test_the_accubits_node_gets_no_t1():
    """8.02 GiB free can't hold a 7.75 GiB base plus a T1 the size of the GPU pool: T1 is dropped, and says why."""
    plan = _small(8.02)
    assert _t1(plan) is None
    (reason,) = [d for d in plan.decisions if d.startswith("T1 dropped")]
    assert f"{BASE_GIB:.2f} GiB base request" in reason


def test_ample_host_memory_keeps_the_share_bound():
    """Where the base fits easily, T1 is sized as before: the share of free memory (or the working set) binds."""
    tier = _t1(_small(30.0))
    assert tier is not None and tier["size_gib"] <= T1_HOST_RAM_SHARE * 30.0


def test_t1_fills_only_what_the_base_leaves():
    """12 GiB free: the share would give 6 GiB, but the pod's base leaves 4.25."""
    tier = _t1(_small(12.0))
    assert tier is not None
    assert tier["size_gib"] == _floor_q(12.0 - BASE_GIB)


@pytest.mark.parametrize("host_free", [10.0, 12.0, 13.0, 16.0, 30.0, 60.0])
def test_a_kept_t1_always_leaves_the_whole_request_schedulable(host_free):
    """Whatever T1 is kept, base + T1 fits the node's free memory."""
    tier = _t1(_small(host_free))
    if tier is not None:
        assert BASE_GIB + tier["size_gib"] <= host_free


def test_colocated_replicas_each_carry_their_base():
    """Two replicas on the node: each pod needs its base and its T1 from half the free memory."""
    tier = _t1(_small(24.0, replicas=2))
    assert tier is not None
    assert tier["size_gib"] == _floor_q(24.0 / 2 - BASE_GIB)


def test_a_t3_tier_that_can_follow_reserves_its_client_buffers():
    """A cluster with a Mooncake pool and a cross-node need: the T3 client's GiB a rank is reserved too."""
    with_t3 = dict(engine_kv_features=T3_ENGINE, cluster={"t3": TCP_POOL}, cross_node_need=True)
    tier = _t1(_small(12.0, **with_t3))
    assert tier is not None
    assert tier["size_gib"] == _floor_q(12.0 - BASE_T3_GIB)
    plan = _small(9.5, **with_t3)
    assert _t1(plan) is None
    assert any(f"{BASE_T3_GIB:.2f} GiB base request" in d for d in plan.decisions)


def test_no_pool_reserves_no_t3_buffers():
    """Without a T3 backend, or without a cross-node need, nothing is reserved for a tier that can't come."""
    for over in ({"cluster": {"t3": TCP_POOL}}, {"cross_node_need": True}):
        tier = _t1(_small(12.0, engine_kv_features=T3_ENGINE, **over))
        assert tier["size_gib"] == _floor_q(12.0 - BASE_GIB)


def test_the_weights_host_copy_counts_against_t1():
    """accubits 2026-11-04: 15.27 GiB unreserved. The pod (base + weights' copy + T1) must fit, so T1 leaves room."""
    tier = _t1(_small(15.27))
    assert tier is not None
    assert BASE_GIB + tier["size_gib"] <= 15.27


def test_the_limit_headroom_is_capped():
    """A large model's host copy is capped at 2 GiB, as the pod limit caps it."""
    assert limit_headroom_mib(40 * GIB, 2) == 2048
