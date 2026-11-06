"""A CPU serving pod: the cores its engine runs on, the host cost of a decode step, and what one session sees.

Measured on PNAP (one Xeon 6767P, 64 cores, SNC2), vllm-cpu 0.10.0, 12 engine threads bound to one NUMA node,
2026-11-06. The fitted values and the corpus are in hardware/configs.py (_GRANITE_RAPIDS_VLLM_CPU_DECODE).
"""

import copy

import pytest

from llm_memory_calculator import (
    estimate_end_to_end_performance,
    estimate_prefill_performance,
)
from llm_memory_calculator.hardware.configs import (
    HARDWARE_CONFIGS,
    resolve_inference_realism,
)
from llm_memory_calculator.performance_estimator import _scope_cpu_to_pod

STABLE_MODEL = "llama2_7b"


def _record(name):
    return copy.deepcopy(HARDWARE_CONFIGS[name])


class TestPodScope:
    def test_compute_is_the_pods_share_of_the_chip(self):
        chip = _record("Xeon6767P_1socket_CPU")
        pod = _scope_cpu_to_pod(chip, cpu_threads=12, host_cores=64)
        assert pod["Flops"] == pytest.approx(chip["Flops"] * 12 / 64)

    def test_bandwidth_is_not_scaled(self):
        """Decode measured as fast on 12 threads as on 31: the pod is not bandwidth-starved."""
        chip = _record("Xeon6767P_1socket_CPU")
        assert (
            _scope_cpu_to_pod(chip, cpu_threads=12, host_cores=64)["Memory_BW"]
            == chip["Memory_BW"]
        )

    def test_the_records_own_core_count_stands_in_for_the_hosts(self):
        chip = _record("Xeon6767P_1socket_CPU")
        assert _scope_cpu_to_pod(chip, cpu_threads=16)["Flops"] == pytest.approx(
            chip["Flops"] * 16 / 64
        )

    def test_the_record_is_not_modified(self):
        chip = _record("Xeon6767P_1socket_CPU")
        flops = chip["Flops"]
        _scope_cpu_to_pod(chip, cpu_threads=12, host_cores=64)
        assert chip["Flops"] == flops

    @pytest.mark.parametrize(
        "threads, host_cores", [(None, 64), (12, None), (64, 64), (128, 64)]
    )
    def test_unchanged_without_a_smaller_share(self, threads, host_cores):
        chip = _record("GraniteRapids_CPU")  # declares no cores
        assert (
            _scope_cpu_to_pod(chip, cpu_threads=threads, host_cores=host_cores) is chip
        )

    def test_an_accelerator_is_unchanged(self):
        gpu = (
            _record("A100_GPU")
            if "A100_GPU" in HARDWARE_CONFIGS
            else {"Flops": 312, "Memory_BW": 2039}
        )
        assert _scope_cpu_to_pod(gpu, cpu_threads=12, host_cores=64) is gpu

    def test_the_prefill_prices_the_pods_cores(self):
        chip = _record("Xeon6767P_1socket_CPU")
        kw = dict(
            model=STABLE_MODEL,
            batch_size=1,
            input_tokens=1024,
            system_name=chip,
            bits="bf16",
        )
        whole = estimate_prefill_performance(**kw)["Latency"]
        pod = estimate_prefill_performance(cpu_threads=12, host_cores=64, **kw)[
            "Latency"
        ]
        assert pod > 3 * whole


class TestDecodeHostCost:
    @pytest.mark.parametrize(
        "name, step", [("GraniteRapids_CPU", 8.8), ("Xeon6767P_1socket_CPU", 6.1)]
    )
    def test_granite_rapids_declares_its_measured_decode_cost(self, name, step):
        decode = resolve_inference_realism(name, phase="decode")
        assert decode["step_overhead_ms"] == pytest.approx(step)
        assert decode["per_sequence_overhead_ms"] == pytest.approx(0.55)

    def test_prefill_was_not_measured_apart_from_its_compute(self):
        prefill = resolve_inference_realism("GraniteRapids_CPU", phase="prefill")
        assert prefill["step_overhead_ms"] == 0.0
        assert prefill["per_sequence_overhead_ms"] == 0.0

    @pytest.mark.parametrize("batch", [1, 4, 8])
    def test_each_step_pays_it_once_plus_per_sequence(self, batch):
        declared = _record("GraniteRapids_CPU")
        bare = _record("GraniteRapids_CPU")
        bare.pop("step_overhead_ms")
        bare.pop("per_sequence_overhead_ms")
        kw = dict(
            model=STABLE_MODEL,
            batch_size=batch,
            input_tokens=1024,
            output_tokens=64,
            bits="bf16",
        )
        with_cost = estimate_end_to_end_performance(system_name=declared, **kw)[
            "average_tpot"
        ]
        without = estimate_end_to_end_performance(system_name=bare, **kw)[
            "average_tpot"
        ]
        assert with_cost - without == pytest.approx(8.8 + 0.55 * batch, rel=1e-6)


class TestOneSession:
    def test_a_cpu_session_waits_for_its_own_prompt(self):
        kw = dict(
            model=STABLE_MODEL,
            input_tokens=1024,
            output_tokens=64,
            system_name="GraniteRapids_CPU",
            bits="bf16",
            cpu_threads=12,
            host_cores=64,
        )
        batch = estimate_end_to_end_performance(batch_size=4, **kw)
        alone = estimate_end_to_end_performance(batch_size=1, **kw)
        assert batch["session_ttft"] == pytest.approx(alone["ttft"])
        assert batch["ttft"] > 3 * batch["session_ttft"]

    def test_an_accelerator_session_waits_for_the_batch(self):
        r = estimate_end_to_end_performance(
            model=STABLE_MODEL,
            batch_size=4,
            input_tokens=1024,
            output_tokens=64,
            bits="bf16",
        )
        assert r["session_ttft"] == r["ttft"]

    @pytest.mark.parametrize("system", [None, "GraniteRapids_CPU"])
    def test_session_throughput_counts_generated_tokens_only(self, system):
        r = estimate_end_to_end_performance(
            model=STABLE_MODEL,
            batch_size=4,
            input_tokens=1024,
            output_tokens=64,
            system_name=system,
            bits="bf16",
        )
        assert r["session_throughput"] == pytest.approx(1000 / r["average_tpot"])
        assert r["session_throughput"] < r["total_throughput"] / 4


class TestClusterMatch:
    def test_a_granite_rapids_node_hands_over_its_record(self):
        """A node's NFD id names no SKU; the matched record must still reach the estimators with its declarations."""
        from llm_memory_calculator import HardwareManager

        specs = HardwareManager().get_cluster_hardware_specs(
            {'raw_name': 'Intel Family 6 Model 173 CPU Xeon 6 (Granite Rapids, P-core)', 'memory_gb': 125.23}
        )
        assert specs['matched']
        record = specs['hardware_config']
        assert record['type'] == 'cpu'
        assert resolve_inference_realism(record, phase='decode')['step_overhead_ms'] == pytest.approx(8.8)
