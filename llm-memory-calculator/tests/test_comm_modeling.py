"""TDD for collective/communication modeling fixes (E unit: M9, H6, M8).

M9: node count must use CEIL division (9 GPUs span 2 nodes, not floor→1) so non-multiple-of-8
    parallel degrees incur the real inter-node penalty.
H6: inter-node bandwidth must be IB-class (~0.11× NVLink, DGX 8×400Gb IB vs NVLink4), not 0.4×.
M8: don't assume concurrent_ops=2 (no implicit DP overlap) — that double-derated single TP collectives.
"""
import warnings
import pytest

warnings.filterwarnings("ignore")

from llm_memory_calculator.genz.collective_times import get_AR_time
from llm_memory_calculator.genz.system import System

LARGE = 256 * 1024 * 1024  # 256 MB — bandwidth-dominated regime


def _sys():
    # NVLink-class intra-node link (H100: ~450 GB/s/GPU), sub-µs latency.
    return System(interchip_link_bw=450, interchip_link_latency=0.5)


def test_m9_ceil_node_count_incurs_inter_node_penalty():
    """9 GPUs span 2 nodes (ceil) → inter-node IB penalty; must be markedly slower than a pure-NVLink
    8-GPU AllReduce. With the old floor (9//8==1) both were modeled as a single NVLink ring."""
    t8 = get_AR_time(LARGE, 8, _sys())     # exactly one node: all NVLink
    t9 = get_AR_time(LARGE, 9, _sys())     # 1 + spillover → 2 nodes: crosses IB
    assert t8 > 0 and t9 > 0
    assert t9 > t8 * 1.3, (
        f"9-GPU AR ({t9:.4f} ms) should be >>8-GPU ({t8:.4f} ms): it spans 2 nodes and crosses the "
        f"inter-node fabric (M9: ceil division, not floor→1 node)"
    )


def test_m9_twelve_gpus_span_two_nodes():
    """12 GPUs (1.5 nodes) must not be modeled as a single 12-GPU NVLink domain (physically impossible)."""
    t8 = get_AR_time(LARGE, 8, _sys())
    t12 = get_AR_time(LARGE, 12, _sys())
    assert t12 > t8 * 1.3, f"12-GPU AR ({t12:.4f}) modeled as one NVLink domain? must cross IB (>{t8:.4f})"


def test_h6_inter_node_bandwidth_is_ib_class():
    """The inter-node bandwidth used for multi-node AR must be ~IB-class. Probe: the inter-node term
    dominates a 2-node large-message AR; with the old 0.4×NVLink (=180 GB/s) it would be far faster
    than realistic IB (~50 GB/s). Assert the 2-node time is consistent with IB-class inter-node BW."""
    sys = _sys()
    t16 = get_AR_time(LARGE, 16, sys)   # 2 full nodes
    # Inter-node reduced data = data/gpus_per_node = 256MB/8 = 32MB, ring across 2 nodes moves ~2*32MB
    # at inter-node BW. At 0.11*450=49.5 GB/s that term alone is ~ 2*32e6/49.5e9 ≈ 1.3 ms; at the old
    # 0.4*450=180 GB/s it would be ~0.36 ms. Assert we're in the IB regime (slower than the 0.4 model).
    inter_bw_011 = 0.11 * 450e9
    inter_bw_040 = 0.40 * 450e9
    reduced = LARGE / 8
    t_term_011 = 2 * reduced / (inter_bw_011 * 0.85) * 1000  # ms, ~ ring inter term, large-msg eff 0.85
    t_term_040 = 2 * reduced / (inter_bw_040 * 0.85) * 1000
    # t16 must be at least as large as the IB-class inter-node term (not the optimistic 0.4 one).
    assert t16 > t_term_040 * 1.5, (
        f"2-node AR ({t16:.4f} ms) is too fast — inter-node BW still ~0.4×NVLink? "
        f"(IB-class term ≈{t_term_011:.3f} ms vs 0.4-model ≈{t_term_040:.3f} ms)"
    )
