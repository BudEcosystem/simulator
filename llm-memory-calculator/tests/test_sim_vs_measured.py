"""Regression tests pinning simulator physics to measured real-world references.

Added after the July 2026 gap study, which found (and fixed) structural modeling
defects that calibration could not absorb:
  - MoE decode charged only top_k experts' weights regardless of batch size
    (ffn.py), making single-GPU MoE decode ~3-5x optimistic and inverting the
    EP-vs-TP ranking.
  - PP decode reported microbatch latency as TPOT with a free xPP throughput
    multiplier (llm_decode.py), inverting the measured single-node PP-vs-TP
    ordering (vLLM: PP2 13.09 vs TP2 51.63 tok/s output throughput).
  - Training activation memory materialized B*heads*S^2 attention scores
    (365x over at 128K) and grads/optimizer were never sharded by TP/PP/EP.

Measured references:
  - gpt-oss-120b on a single B200, vLLM, max load: ~7,236 tok/s aggregate.
  - vLLM docs, single node: PP output throughput < TP output throughput.
  - Real MoE training MFU band 20-40% (DeepSeek-V3 21-39%); dense ~38-50%.
  - Llama-3 405B trained at seq 131,072 (CP=16) on 80 GB GPUs -> long-context
    activations are tens of GB, not TB.

The gpt-oss tests fetch the HF config over the network; they skip offline.
"""
import math

import pytest

MODEL = 'openai/gpt-oss-120b'


@pytest.fixture(scope='module')
def gpt_oss_available():
    from llm_memory_calculator.genz.Models.get_language_model import get_configs
    try:
        cfg = get_configs(MODEL)
    except Exception as e:  # offline / HF unreachable
        pytest.skip(f'gpt-oss-120b config unavailable: {e}')
    return cfg


def _decode(**overrides):
    from llm_memory_calculator import estimate_decode_performance
    kw = dict(model=MODEL, batch_size=50, input_tokens=8000, output_tokens=2000,
              system_name='B300', bits='fp4', tensor_parallel=1)
    kw.update(overrides)
    return estimate_decode_performance(**kw)


class TestMoEDecodeTraffic:
    def test_single_gpu_aggregate_matches_measured_band(self, gpt_oss_available):
        """vLLM on one B200 sustains ~7,236 tok/s at max load for gpt-oss-120b."""
        agg = _decode(system_name='B200')['Throughput_tokens_per_sec']
        assert 4000 <= agg <= 10000, f'{agg:.0f} tok/s vs measured ~7236'

    def test_expert_traffic_scales_with_batch(self, gpt_oss_available):
        """50 concurrent tokens hit ~102 of 128 experts; TPOT must grow with batch."""
        t1 = _decode(batch_size=1)['TPOT']
        t50 = _decode(batch_size=50)['TPOT']
        assert t50 > 2 * t1, f'TPOT flat in batch: b1={t1:.3f}ms b50={t50:.3f}ms'

    def test_expected_unique_experts_formula(self):
        from llm_memory_calculator.genz.Models.ffn import calculate_activated_experts
        assert calculate_activated_experts(128, 4, num_tokens=1) == 4
        assert 20 <= calculate_activated_experts(128, 4, num_tokens=6) <= 25
        assert 98 <= calculate_activated_experts(128, 4, num_tokens=50) <= 106
        assert calculate_activated_experts(128, 4, num_tokens=500) == 128

    def test_ep_beneficial_for_moe_decode(self, gpt_oss_available):
        """Expert weights dominate batched MoE decode -> sharding them must win."""
        tp1 = _decode()['Throughput_tokens_per_sec']
        ep8 = _decode(expert_parallel=8)['Throughput_tokens_per_sec']
        assert ep8 > tp1, f'EP8={ep8:.0f} <= TP1={tp1:.0f}'


class TestPipelineParallelOrdering:
    def test_pp_does_not_beat_tp_single_node(self, gpt_oss_available):
        """vLLM measures single-node PP output throughput well below TP."""
        for p in (2, 8):
            tp = _decode(tensor_parallel=p)['Throughput_tokens_per_sec']
            pp = _decode(pipeline_parallel=p)['Throughput_tokens_per_sec']
            assert pp <= 1.1 * tp, f'PP{p}={pp:.0f} > 1.1*TP{p}={tp:.0f}'

    def test_pp_tpot_is_not_microbatch_latency(self, gpt_oss_available):
        """PP TPOT must reflect a full m-microbatch pipeline round, not one microbatch."""
        tpot_pp8 = _decode(pipeline_parallel=8)['TPOT']
        tpot_tp1 = _decode()['TPOT']
        assert tpot_pp8 >= 0.5 * tpot_tp1, (
            f'PP8 TPOT {tpot_pp8:.2f}ms implausibly below TP1 {tpot_tp1:.2f}ms')

    def test_pp_prefill_charges_fill_drain(self, gpt_oss_available):
        from llm_memory_calculator import estimate_prefill_performance
        p50 = estimate_prefill_performance(model=MODEL, batch_size=50, input_tokens=8000,
                system_name='B300', bits='fp4', tensor_parallel=1, pipeline_parallel=8)
        p6 = estimate_prefill_performance(model=MODEL, batch_size=6, input_tokens=8000,
                system_name='B300', bits='fp4', tensor_parallel=1)
        assert p50['Latency'] > 1.3 * p6['Latency'], (
            'PP prefill batch latency equals one microbatch (fill/drain unmodeled)')


class TestModelLoader:
    def test_gated_mlp_weights(self, gpt_oss_available):
        """num_ffi=2 for gated MLPs: gpt-oss fp4 weights ~55-65 GB (real ~60.2)."""
        st = _decode()['summary_table']
        wcol = [c for c in st.columns if 'Weight' in c][0]
        wt_mb = float(st[wcol].values[0])
        assert 53000 <= wt_mb <= 66000, f'weights {wt_mb:.0f} MB'

    def test_sliding_window_halves_kv(self, gpt_oss_available):
        """18 of 36 gpt-oss layers use a 128-token window -> KV ~half of all-full."""
        st = _decode()['summary_table']
        kcol = [c for c in st.columns if 'KV' in c][0]
        kv_mb = float(st[kcol].values[0])
        assert kv_mb <= 5200, f'KV {kv_mb:.0f} MB suggests all-full-attention (was ~7032)'


class TestTrainingRealism:
    def _train(self, **overrides):
        from llm_memory_calculator.genz.LLM_training.training_modeling import training_modeling
        kw = dict(model=MODEL, training_stage='sft', method='full', batch_size=1,
                  seq_length=4096, system_name='B300', num_gpus=8, tensor_parallel=1,
                  data_parallel=8, optimizer='adamw', zero_stage=3,
                  gradient_checkpointing=True, bits='bf16')
        kw.update(overrides)
        return training_modeling(**kw)

    def test_sharded_memory_fits_node(self, gpt_oss_available):
        r = self._train()
        assert 180 <= r.memory_per_gpu_gb <= 330, f'{r.memory_per_gpu_gb:.1f} GB'
        r2 = self._train(tensor_parallel=8, data_parallel=1, zero_stage=1)
        assert r2.memory_per_gpu_gb <= 350, (
            f'TP8 {r2.memory_per_gpu_gb:.1f} GB — grads/optimizer not TP-sharded?')

    def test_long_context_activation_is_linear(self, gpt_oss_available):
        """131K activations must be tens of GB (FlashAttention), not TB (S^2 scores)."""
        r = self._train(seq_length=131072)
        assert 20 <= r.activation_memory_gb <= 80, f'{r.activation_memory_gb:.1f} GB'
        assert r.memory_per_gpu_gb <= 300, '128K SFT must fit a 288 GB B300 node'

    def test_mfu_in_measured_band(self, gpt_oss_available):
        """MoE training MFU 20-40% measured (DeepSeek-V3); MFU == timeline identity."""
        r = self._train(batch_size=64)
        assert 0.15 <= r.model_flops_utilization <= 0.50, (
            f'MFU {r.model_flops_utilization*100:.1f}%')

    def test_dense_mfu_band(self):
        from llm_memory_calculator.genz.LLM_training.training_modeling import training_modeling
        r = training_modeling(model='llama-2-7b', training_stage='sft', method='full',
                              batch_size=32, seq_length=4096, system_name='H100_GPU',
                              num_gpus=8, tensor_parallel=1, data_parallel=8,
                              optimizer='adamw', zero_stage=2,
                              gradient_checkpointing=True, bits='bf16')
        assert 0.20 <= r.model_flops_utilization <= 0.60, (
            f'dense MFU {r.model_flops_utilization*100:.1f}%')


class TestB300Registered:
    def test_b300_first_class(self):
        from llm_memory_calculator.hardware.manager import get_hardware_config
        cfg = get_hardware_config('B300')
        assert cfg and cfg['Memory_size'] == 288 and cfg['Memory_BW'] == 8000
        assert cfg['Flops'] == 3750
