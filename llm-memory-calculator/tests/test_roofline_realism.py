"""Roofline realism: achieved efficiency, and the fixed per-step cost the roofline omits.

Two systematic errors, both properties of the DEVICE + its host serving runtime and therefore
unfixable from the model side, both measured on an H100XM-80C vGPU (tp=1, vLLM V1 engine, driven
IN-CLUSTER straight at the engine service so no network overhead is included):

  1. NO ACHIEVED-EFFICIENCY DERATING.  `system_eff` defaulted to 1 in llm_decode/llm_prefill, i.e.
     every device was modelled as sustaining 100% of its datasheet FLOPs AND 100% of its datasheet
     DRAM bandwidth. Nothing does. Qwen3.8-27B must stream 55.93 GB per decode step; against the
     3350 GB/s HBM3 peak that is a 16.7 ms floor, and it measures 20.197 ms — 83% achieved.
     (That 83% attributes the WHOLE step to streaming. Subtract the fixed host cost in (2) first and
     the achieved bandwidth is ~92%; see the note in hardware/configs.py — the two terms must not be
     fitted to the same residual, which is why nothing here hardcodes 0.83 for one device.)

  2. NO FIXED PER-STEP COST.  The roofline charges bytes and FLOPs and nothing else, so it claims a
     small enough decode step is free. Measured Qwen3-0.6B TPOT never drops below ~2.0 ms at ANY
     batch/context while the roofline predicts 0.506 ms: kernel dispatch, the scheduler, sampling
     and detokenisation were modelled as exactly zero. The missing term is ADDITIVE, not a scale.

Both are now DECLARED on the hardware record (hardware/configs.py :: resolve_inference_realism) with
documented per-device-class defaults, and are read at inference-system construction. These tests pin
that they are read from configuration rather than hardcoded, that a device declaring efficiency 1.0
and zero overhead reproduces the untouched roofline EXACTLY (so the change is opt-in and auditable),
and that the measured floor is reproduced.

MEASURED GROUND TRUTH lives in section 6, with the measurement rules. Two corpora, two unrelated
architectures, each spanning batch 1..32 and context 2k..32k on one H100XM-80C engine:
    Qwen3-0.6B   dense, 12 points          gpt-oss-20b  MoE + MXFP4 experts + sliding window, 10 points

A CORRECTION TO THE FIRST VERSION OF THIS MODULE.  It was written against a batch-1/batch-10 corpus
whose batch-10 points were measured through a 2-replica gateway (5 sequences per engine), with
identical prompts (the prefix cache shared their KV) and requests stopping at their own EOS (the
batch drained). Each made batch 10 look cheaper. A least-squares fit over that corpus produced a
0.245 ms per-SEQUENCE term and a 0.35 coefficient on attention, and both were read as physics: the
first became the per_sequence_overhead_ms default, the second a reported "attention over-charged at
long context" defect. Neither survives a clean measurement. With the artifacts removed:

    predicted / measured TPOT        per-seq 0.25 (old default)     per-seq 0.0 (current)
    Qwen3-0.6B,  12 points           1.08 .. 2.69x                  0.95 .. 1.08x   mean |ln| 0.037
    gpt-oss-20b, 10 points           0.93 .. 2.09x                  0.86 .. 0.99x   mean |ln| 0.075

and the attention coefficient is ~1: the long-context points are the best-fitting ones. The old
default is physically refuted, not just a worse fit — at batch 32 / 2k it charges 8 ms of host cost
to a step that measures 4.75 ms in total. step_overhead_ms = 1.0 is independently confirmed by the
same data (0.5 and 1.5 both fit worse), so nothing was re-fitted to absorb the change.

(The gpt-oss column also depends on sizing its MXFP4 experts at their stored precision rather than at
the caller's bf16 — genz/weight_precision.py, pinned by test_checkpoint_weight_precision.py. Timed at
bf16 it was 1.26 .. 3.42x.)

TTFT is not validated here: the clean corpus shows prefill under-predicted (~0.5-0.7x at batch 1), a
separate open item that `compute_efficiency` for prefill is the declared place to address.

These terms are NOT expected to fix Qwen3.8-27B: that model's error is a model-config defect (its
dimensions live in a nested `text_config`, so a generic ~5.5B LLaMA was built instead of the real
25.9B hybrid), owned elsewhere. `test_fixed_overhead_cannot_explain_the_27b_gap` pins that split.
"""
import math
import os

import pytest

from llm_memory_calculator.genz.LLM_inference.llm_decode import decode_moddeling
from llm_memory_calculator.genz.LLM_inference.llm_prefill import prefill_moddeling
from llm_memory_calculator.genz.analyse_model import count_repeat_aware_ops
from llm_memory_calculator.genz.unit import Unit
from llm_memory_calculator.hardware.configs import (
    DEFAULT_DEVICE_CLASS,
    HARDWARE_CONFIGS,
    INFERENCE_PHASES,
    INFERENCE_REALISM_DEFAULTS,
    INFERENCE_REALISM_KEYS,
    apply_inference_realism,
    resolve_inference_realism,
)

UNIT = Unit()
LAT = f'Latency ({UNIT.unit_time})'

#: A built-in model config (MODEL_DICT), deliberately NOT read from a HuggingFace checkpoint, so the
#: exactness tests below cannot be perturbed by concurrent model-config work.
STABLE_MODEL = 'llama2_7b'

_OVERHEAD_KEYS = ('kernel_launch_latency_ms', 'step_overhead_ms', 'per_sequence_overhead_ms')


def _hw(**overrides):
    """An H100 record with no measured calibration block, plus the given declarations."""
    cfg = {k: v for k, v in HARDWARE_CONFIGS['H100_GPU'].items() if k != 'inference_calibration'}
    cfg.update(overrides)
    return cfg


#: "Raw roofline" device: sustains 100% of peak and pays nothing per step. This is the model the
#: simulator used to apply to EVERY device; declaring it explicitly keeps it reproducible.
RAW = _hw(compute_efficiency=1.0, memory_efficiency=1.0,
          kernel_launch_latency_ms=0.0, step_overhead_ms=0.0, per_sequence_overhead_ms=0.0)

#: Today-before-this-change device: the documented per-technology efficiency bands (undeclared, so
#: get_inference_system supplies them) with the fixed overheads switched off.
NO_OVERHEAD = _hw(kernel_launch_latency_ms=0.0, step_overhead_ms=0.0, per_sequence_overhead_ms=0.0)


def _decode(system_name, **overrides):
    kw = dict(model=STABLE_MODEL, batch_size=1, input_tokens=2048, output_tokens=1, bits='bf16')
    kw.update(overrides)
    return decode_moddeling(system_name=system_name, **kw)


def _decode_roofline(system_name, **overrides):
    """The pre-overhead roofline latency for the same shape (model_profilling returns before the
    additive terms are applied)."""
    kw = dict(model=STABLE_MODEL, batch_size=1, input_tokens=2048, output_tokens=1, bits='bf16')
    kw.update(overrides)
    model_df, summary = decode_moddeling(system_name=system_name, model_profilling=True, **kw)
    return summary[LAT].values[0], count_repeat_aware_ops(model_df)


# ==================================================================================================
# 1. The change is opt-in and auditable
# ==================================================================================================
class TestUnitEfficiencyAndZeroOverheadReproduceTheRawRoofline:
    """A device declaring efficiency 1.0 and zero overhead must land on the untouched roofline —
    every millisecond of difference has to be traceable to a declared number."""

    def test_decode_matches_the_roofline_exactly(self):
        roofline, _ = _decode_roofline(RAW, batch_size=4)
        full = _decode(RAW, batch_size=4)['Latency']
        assert full == pytest.approx(roofline, rel=1e-12), (
            f'declared 1.0/zero device drifted from the raw roofline: {full} vs {roofline}')

    @pytest.mark.parametrize('eta', [1.0, 0.5, 0.25])
    def test_prefill_is_the_roofline_over_the_declared_efficiency(self, eta):
        """Prefill mixes compute-bound and memory-bound operators, so scaling BOTH efficiencies by
        the same eta must scale max(compute, memory) by exactly 1/eta. eta=1.0 with zero overhead is
        therefore the untouched roofline, and no term escapes the declaration."""
        def run(e):
            return prefill_moddeling(model=STABLE_MODEL, batch_size=2, input_tokens=2048,
                                     bits='bf16',
                                     system_name=_hw(compute_efficiency=e, memory_efficiency=e,
                                                     kernel_launch_latency_ms=0.0,
                                                     step_overhead_ms=0.0,
                                                     per_sequence_overhead_ms=0.0))['Latency']
        assert run(eta) == pytest.approx(run(1.0) / eta, rel=1e-9)

    def test_the_default_device_is_slower_than_the_raw_roofline_in_both_phases(self):
        """The whole point: the default is no longer 100%-of-peak-and-free."""
        assert _decode('H100_GPU')['Latency'] > _decode(RAW)['Latency']
        raw_ttft = prefill_moddeling(model=STABLE_MODEL, batch_size=1, input_tokens=2048,
                                     system_name=RAW, bits='bf16')['Latency']
        default_ttft = prefill_moddeling(model=STABLE_MODEL, batch_size=1, input_tokens=2048,
                                         system_name='H100_GPU', bits='bf16')['Latency']
        assert default_ttft > raw_ttft


# ==================================================================================================
# 2. Achieved efficiency is read from configuration
# ==================================================================================================
class TestAchievedEfficiencyIsDeclared:

    def test_memory_bound_decode_at_083_lands_exactly_one_over_083_slower(self):
        """Batch-1 decode is weight-streaming bound, so a declared 83% of peak bandwidth must cost
        exactly 1/0.83. Measured H100: 55.93 GB / 3350 GB/s = 16.7 ms floor vs 20.197 ms -> 0.83."""
        fast = _decode(RAW)['Latency']
        slow = _decode(_hw(compute_efficiency=1.0, memory_efficiency=0.83,
                           kernel_launch_latency_ms=0.0, step_overhead_ms=0.0,
                           per_sequence_overhead_ms=0.0))['Latency']
        assert slow == pytest.approx(fast / 0.83, rel=1e-9), (
            f'declared memory efficiency did not scale a memory-bound decode: {slow} vs {fast/0.83}')

    def test_efficiency_is_not_hardcoded_anywhere(self):
        """Three declared efficiencies, three distinct latencies, each the roofline over eta."""
        base = _decode(RAW)['Latency']
        for eta in (0.9, 0.75, 0.5):
            got = _decode(_hw(compute_efficiency=1.0, memory_efficiency=eta,
                              kernel_launch_latency_ms=0.0, step_overhead_ms=0.0,
                              per_sequence_overhead_ms=0.0))['Latency']
            assert got == pytest.approx(base / eta, rel=1e-9), f'eta={eta}'

    def test_an_explicit_caller_system_eff_still_wins(self):
        """system_eff stays a caller override; the declaration is only the default."""
        declared = _decode(_hw(compute_efficiency=0.9, memory_efficiency=0.9,
                               kernel_launch_latency_ms=0.0, step_overhead_ms=0.0,
                               per_sequence_overhead_ms=0.0))['Latency']
        overridden = decode_moddeling(model=STABLE_MODEL, batch_size=1, input_tokens=2048,
                                      output_tokens=1, bits='bf16', system_eff=0.5,
                                      system_name=_hw(compute_efficiency=0.9, memory_efficiency=0.9,
                                                      kernel_launch_latency_ms=0.0,
                                                      step_overhead_ms=0.0,
                                                      per_sequence_overhead_ms=0.0))['Latency']
        assert overridden == pytest.approx(declared * 0.9 / 0.5, rel=1e-9)

    def test_no_device_is_left_running_at_100_percent_of_peak(self):
        """Peak is a ceiling, never an operating point — for every device in the library."""
        from llm_memory_calculator.genz.LLM_inference.utils import get_inference_system
        for name, cfg in HARDWARE_CONFIGS.items():
            for phase in INFERENCE_PHASES:
                realism = resolve_inference_realism(cfg, phase=phase)
                system = get_inference_system(
                    system_name=dict(cfg), bits='bf16', phase=phase,
                    ceff=realism['compute_efficiency'] or 1,
                    meff=realism['memory_efficiency'] or 1)
                assert 0 < system.compute_efficiency < 1, f'{name}/{phase} compute at peak'
                assert 0 < system.memory_efficiency < 1, f'{name}/{phase} memory at peak'


# ==================================================================================================
# 3. The fixed per-step cost is read from configuration
# ==================================================================================================
class TestFixedPerStepOverheadIsDeclared:

    @pytest.mark.parametrize('batch', [1, 4, 16])
    def test_declared_decode_overheads_are_added_exactly(self, batch):
        base = _decode(RAW, batch_size=batch)['Latency']
        hw = _hw(compute_efficiency=1.0, memory_efficiency=1.0,
                 kernel_launch_latency_ms=0.01, step_overhead_ms=2.0, per_sequence_overhead_ms=0.5)
        _, n_ops = _decode_roofline(hw, batch_size=batch)
        got = _decode(hw, batch_size=batch)['Latency']
        assert got == pytest.approx(base + 0.01 * n_ops + 2.0 + 0.5 * batch, rel=1e-9)

    def test_declared_prefill_overheads_are_added_exactly(self):
        kw = dict(model=STABLE_MODEL, batch_size=3, input_tokens=2048, bits='bf16')
        base = prefill_moddeling(system_name=RAW, **kw)['Latency']
        hw = _hw(compute_efficiency=1.0, memory_efficiency=1.0,
                 kernel_launch_latency_ms=0.01, step_overhead_ms=2.0, per_sequence_overhead_ms=0.5)
        model_df, _ = prefill_moddeling(system_name=hw, model_profilling=True, **kw)
        n_ops = count_repeat_aware_ops(model_df)
        got = prefill_moddeling(system_name=hw, **kw)['Latency']
        # prefill is ONE engine step over the prompt: per-op dispatch + one step + one per sequence.
        assert got == pytest.approx(base + 0.01 * n_ops + 2.0 + 0.5 * 3, rel=1e-9)

    def test_the_overhead_is_additive_not_multiplicative(self):
        """Measured Qwen3-0.6B TPOT at batch 1 sits on a ~2 ms floor across a 16x context range
        (2.03 ms at 2k, 2.98 ms at 32k): a fixed cost, not a scale. The term must be independent of how
        big the roofline underneath it is."""
        hw = _hw(compute_efficiency=1.0, memory_efficiency=1.0,
                 kernel_launch_latency_ms=0.0, step_overhead_ms=2.0, per_sequence_overhead_ms=0.0)
        for ctx in (512, 2048, 32768):
            base = _decode(RAW, input_tokens=ctx)['Latency']
            got = _decode(hw, input_tokens=ctx)['Latency']
            assert got - base == pytest.approx(2.0, rel=1e-9), f'ctx={ctx}'

    def test_decode_has_a_floor_the_shape_cannot_go_below(self):
        """However small the step, TPOT cannot fall below the declared fixed host cost."""
        realism = resolve_inference_realism('H100_GPU', phase='decode')
        floor = realism['step_overhead_ms'] + realism['per_sequence_overhead_ms']
        assert floor > 0, 'the fixed per-step cost is still modelled as zero'
        assert _decode('H100_GPU', input_tokens=8, batch_size=1)['Latency'] > floor

    def test_overheads_do_not_scale_with_the_weight_term(self):
        """Guard against double-counting when a model-config fix grows the roofline: the declared
        terms are additive and model-independent, so the DIFFERENCE they make must not change when
        the modelled weights grow."""
        hw = _hw(compute_efficiency=1.0, memory_efficiency=1.0,
                 kernel_launch_latency_ms=0.0, step_overhead_ms=1.0, per_sequence_overhead_ms=0.25)
        deltas = []
        for model in ('llama2_7b', 'llama2_13b'):
            base = decode_moddeling(model=model, batch_size=1, input_tokens=2048, output_tokens=1,
                                    bits='bf16', system_name=RAW)['Latency']
            got = decode_moddeling(model=model, batch_size=1, input_tokens=2048, output_tokens=1,
                                   bits='bf16', system_name=hw)['Latency']
            deltas.append(got - base)
        assert deltas[0] == pytest.approx(deltas[1], rel=1e-9), (
            f'the fixed term moved with the weight term: {deltas}')


# ==================================================================================================
# 4. The declaration layer itself
# ==================================================================================================
class TestRealismDeclarationResolution:

    def test_an_undeclared_gpu_gets_the_documented_class_default(self):
        realism = resolve_inference_realism(_hw(), phase='decode')
        expected = INFERENCE_REALISM_DEFAULTS['gpu']['decode']
        for key in _OVERHEAD_KEYS:
            assert realism[key] == expected[key]
        # efficiencies stay None -> get_inference_system's documented per-technology bands apply,
        # so the principled default for those two keeps living in exactly one place.
        assert realism['compute_efficiency'] is None
        assert realism['memory_efficiency'] is None

    def test_prefill_launches_eagerly_and_therefore_costs_more_per_operator(self):
        """Engines CUDA-graph decode (static shapes) and run prefill eagerly (prompt length varies),
        so the per-operator dispatch default must be larger for prefill."""
        decode = resolve_inference_realism('H100_GPU', phase='decode')
        prefill = resolve_inference_realism('H100_GPU', phase='prefill')
        assert prefill['kernel_launch_latency_ms'] > decode['kernel_launch_latency_ms'] > 0

    def test_a_record_declaration_overrides_the_class_default(self):
        realism = resolve_inference_realism(_hw(step_overhead_ms=7.5, memory_efficiency=0.61),
                                            phase='decode')
        assert realism['step_overhead_ms'] == 7.5
        assert realism['memory_efficiency'] == 0.61

    def test_a_declaration_may_be_per_phase(self):
        hw = _hw(step_overhead_ms={'prefill': 4.0, 'decode': 0.5})
        assert resolve_inference_realism(hw, phase='prefill')['step_overhead_ms'] == 4.0
        assert resolve_inference_realism(hw, phase='decode')['step_overhead_ms'] == 0.5

    def test_a_measured_calibration_block_beats_a_declaration(self):
        hw = _hw(memory_efficiency=0.9, kernel_launch_latency_ms=0.5,
                 inference_calibration={'decode': {'eta_mem': 0.42, 't_launch_ms': 0.001}})
        realism = resolve_inference_realism(hw, phase='decode')
        assert realism['memory_efficiency'] == 0.42
        assert realism['kernel_launch_latency_ms'] == 0.001

    def test_a_calibrated_phase_does_not_also_collect_the_class_priors(self):
        """GB10 carries a MEASURED block fitted end-to-end. Layering the class-default step and
        per-sequence overhead on top of a measured fit would double-count it."""
        realism = resolve_inference_realism('GB10', phase='decode')
        assert realism['kernel_launch_latency_ms'] == pytest.approx(0.00456)
        assert realism['memory_efficiency'] == pytest.approx(0.659)
        assert realism['step_overhead_ms'] == 0.0
        assert realism['per_sequence_overhead_ms'] == 0.0

    def test_cpu_declares_zero_because_its_floor_constant_already_contains_the_overhead(self):
        cpu = [c for c in HARDWARE_CONFIGS.values() if str(c.get('type', '')).lower() == 'cpu']
        assert cpu, 'no CPU record to check'
        realism = resolve_inference_realism(cpu[0], phase='decode')
        for key in _OVERHEAD_KEYS:
            assert realism[key] == 0.0

    def test_every_hardware_record_resolves_to_a_concrete_overhead(self):
        for name, cfg in HARDWARE_CONFIGS.items():
            for phase in INFERENCE_PHASES:
                realism = resolve_inference_realism(cfg, phase=phase)
                for key in _OVERHEAD_KEYS:
                    value = realism[key]
                    assert value is not None and value >= 0 and math.isfinite(value), \
                        f'{name}/{phase}/{key} = {value!r}'

    def test_a_system_object_gets_the_class_default_not_its_constructor_zero(self):
        """A System built directly (BudEvolve's hardware explorer) has kernel_launch_latency_ms=0.0
        and compute_efficiency=1 as CONSTRUCTOR defaults; reading those back as 'declarations' would
        quietly restore the 100%-of-peak, zero-overhead model."""
        from llm_memory_calculator.genz.system import System
        realism = resolve_inference_realism(System(unit=UNIT), phase='decode')
        assert realism['step_overhead_ms'] == INFERENCE_REALISM_DEFAULTS[DEFAULT_DEVICE_CLASS]['decode']['step_overhead_ms']
        assert realism['compute_efficiency'] is None
        assert realism['memory_efficiency'] is None

    def test_a_system_built_with_a_deliberate_efficiency_is_respected(self):
        from llm_memory_calculator.genz.system import System
        realism = resolve_inference_realism(
            System(unit=UNIT, compute_efficiency=0.4, memory_efficiency=0.6), phase='decode')
        assert realism['compute_efficiency'] == pytest.approx(0.4)
        assert realism['memory_efficiency'] == pytest.approx(0.6)

    @pytest.mark.parametrize('bad', [
        {'memory_efficiency': 1.4},      # would claim the device beats its datasheet
        {'compute_efficiency': 0.0},
        {'compute_efficiency': -0.2},
        {'step_overhead_ms': -1.0},
        {'kernel_launch_latency_ms': float('nan')},
        {'per_sequence_overhead_ms': 'fast'},
    ])
    def test_a_nonsensical_declaration_fails_loudly(self, bad):
        with pytest.raises(ValueError):
            resolve_inference_realism(_hw(**bad), phase='decode')

    def test_an_unknown_hardware_name_fails_loudly(self):
        with pytest.raises(ValueError):
            resolve_inference_realism('NO_SUCH_DEVICE_XYZ', phase='decode')

    def test_an_unknown_phase_fails_loudly(self):
        with pytest.raises(ValueError):
            resolve_inference_realism('H100_GPU', phase='training')

    def test_apply_attaches_every_overhead_to_the_system(self):
        from llm_memory_calculator.genz.system import System
        realism = resolve_inference_realism('H100_GPU', phase='decode')
        system = apply_inference_realism(System(unit=UNIT), realism)
        assert system.kernel_launch_latency_ms > 0
        assert system.step_overhead_ms > 0
        # Measured to be ~0 for a vLLM-class engine (see section 6), but still ATTACHED, so a runtime
        # declaring a non-zero value on its record reaches the system the same way.
        assert system.per_sequence_overhead_ms == realism['per_sequence_overhead_ms']
        declared = apply_inference_realism(System(unit=UNIT),
                                           resolve_inference_realism(_hw(per_sequence_overhead_ms=0.4),
                                                                     phase='decode'))
        assert declared.per_sequence_overhead_ms == 0.4

    def test_the_declarable_key_set_is_the_one_the_defaults_cover(self):
        for device_class, phases in INFERENCE_REALISM_DEFAULTS.items():
            assert set(phases) == set(INFERENCE_PHASES), device_class
            for phase, defaults in phases.items():
                assert set(defaults) <= set(INFERENCE_REALISM_KEYS), (device_class, phase)


# ==================================================================================================
# 5. Total_latency = TPOT x output_tokens — examined, and correct as-is
# ==================================================================================================
class TestTotalLatencyIsATrapezoidSum:
    """llm_decode already returns the 50/50 average of the step at context=input and the step at
    context=input+output, and per-step decode latency is affine in the KV length. The mean of the
    endpoints times N therefore IS the trapezoidal integral, i.e. the sum over the generation — so
    `Total_latency = Latency * output_tokens` in performance_estimator.py is correct, not a
    first-step-times-N approximation. No fix applied."""

    N = 8
    CTX = 2048

    def test_total_latency_equals_the_endpoint_trapezoid_exactly(self):
        from llm_memory_calculator import estimate_decode_performance
        result = estimate_decode_performance(model=STABLE_MODEL, batch_size=1, input_tokens=self.CTX,
                                             output_tokens=self.N, system_name=RAW, bits='bf16')
        first = _decode(RAW, input_tokens=self.CTX)['Latency']
        last = _decode(RAW, input_tokens=self.CTX + self.N)['Latency']
        # rel=1e-3, not exact: a separately-invoked reference step sees a marginally different
        # on-chip allocation state (each decode_moddeling call runs a different number of
        # get_model_df passes over the same System). The structural identity is what matters — the
        # result tracks the endpoint mean, not the first step.
        assert result['Total_latency'] == pytest.approx(self.N * (first + last) / 2, rel=1e-3)

    def test_the_generation_is_integrated_not_the_first_step_repeated(self):
        """With enough generated tokens the KV cache growth is visible, and the total must land on
        the trapezoid — strictly ABOVE first_step x N, which is what a flat-rate model would give."""
        from llm_memory_calculator import estimate_decode_performance
        n = 8192
        total = estimate_decode_performance(model=STABLE_MODEL, batch_size=1, input_tokens=self.CTX,
                                            output_tokens=n, system_name=RAW,
                                            bits='bf16')['Total_latency']
        first = _decode(RAW, input_tokens=self.CTX)['Latency']
        last = _decode(RAW, input_tokens=self.CTX + n)['Latency']
        assert last > 1.05 * first, 'KV growth is invisible at this shape; pick a longer generation'
        assert total > n * first, 'total is the first step repeated, not the integral'
        assert total == pytest.approx(n * (first + last) / 2, rel=1e-3)

    def test_the_trapezoid_matches_the_true_per_step_sum(self):
        from llm_memory_calculator import estimate_decode_performance
        total = estimate_decode_performance(model=STABLE_MODEL, batch_size=1, input_tokens=self.CTX,
                                            output_tokens=self.N, system_name=RAW,
                                            bits='bf16')['Total_latency']
        per_step = sum(_decode(RAW, input_tokens=self.CTX + i)['Latency'] for i in range(self.N))
        assert total == pytest.approx(per_step, rel=0.01), (
            f'trapezoid {total} vs true sum {per_step}: per-step latency is not affine in context')

    def test_the_fixed_overhead_is_charged_once_per_generated_token(self):
        """The additive terms are constant across steps, so they average to themselves and must come
        out of Total_latency exactly N times — no more, no less."""
        from llm_memory_calculator import estimate_decode_performance
        hw = _hw(compute_efficiency=1.0, memory_efficiency=1.0,
                 kernel_launch_latency_ms=0.0, step_overhead_ms=3.0, per_sequence_overhead_ms=0.0)
        kw = dict(model=STABLE_MODEL, batch_size=1, input_tokens=self.CTX, output_tokens=self.N,
                  bits='bf16')
        raw = estimate_decode_performance(system_name=RAW, **kw)['Total_latency']
        with_oh = estimate_decode_performance(system_name=hw, **kw)['Total_latency']
        assert with_oh - raw == pytest.approx(3.0 * self.N, rel=1e-9)


# ==================================================================================================
# 6. Against the measured H100 numbers
# ==================================================================================================
REGISTRY = '/data/models-registry'
FIXTURES = os.path.join(os.path.dirname(__file__), 'fixtures', 'measured_checkpoints')
#: GenZ builds the operator graph from config.json alone, so the two public configs are committed as
#: fixtures and these guards run everywhere. The 27B checkpoint is not public; its test still needs
#: the registry mounted.
CHECKPOINTS = {
    'Qwen3-0.6B': os.path.join(FIXTURES, 'Qwen_Qwen3-0.6B'),
    'gpt-oss-20b': os.path.join(FIXTURES, 'openai_gpt-oss-20b'),
    'Qwen3.8-27B': f'{REGISTRY}/qwen_qwen3_8-27b_a849ea74',
}

#: How every point below was measured, and why each rule exists. H100XM-80C vGPU, vLLM V1, tp=1,
#: driven from a pod in the same cluster:
#:   * ONE engine, addressed by pod IP — a gateway in front of N replicas turns batch B into B/N.
#:   * a unique nonce opening every prompt — identical prompts share KV blocks through the prefix
#:     cache, so the batch reads one copy of the context instead of B.
#:   * ignore_eos with min_tokens == max_tokens == 512 — requests stopping at their own EOS drain the
#:     batch mid-decode.
#:   * the median inter-token gap over tokens 256..512 — by then every prefill has finished, so no
#:     step is shared with a prefill chunk.
#:   * batch sizes that are captured CUDA-graph sizes — an odd batch is padded to the next one.
#: The previous corpus (batch 1 and 10) broke the first three rules at batch 10, which is where the
#: 0.25 ms per-sequence default and the "attention over-charged at long context" finding came from.
STEADY_STATE_OFFSET = 384  # the 256..512 window holds prompt + ~384 tokens of KV

#: (batch, prompt tokens) -> steady-state TPOT in ms.
MEASURED_TPOT_QWEN06B = {
    (1, 2037): 2.025, (8, 2037): 2.476, (16, 2037): 3.243, (32, 2037): 4.748,
    (1, 8191): 2.009, (8, 8191): 4.269, (16, 8191): 6.780, (24, 8191): 9.286,
    (1, 16378): 2.321, (8, 16378): 6.680,
    (1, 32757): 2.976, (4, 32757): 6.554,
}
#: gpt-oss-20b: MoE (32 experts, top-4), MXFP4 experts, alternating 128-token sliding window.
#: Repeated in two independent runs; every point reproduced within 2%.
MEASURED_TPOT_GPT_OSS = {
    (1, 2044): 3.228, (8, 2044): 5.083, (16, 2044): 5.842, (32, 2044): 6.863,
    (1, 8153): 3.287, (8, 8153): 5.407, (16, 8153): 6.554, (32, 8153): 8.428,
    (1, 32722): 3.489, (8, 32722): 6.962,
}
MEASURED = {'Qwen3-0.6B': MEASURED_TPOT_QWEN06B, 'gpt-oss-20b': MEASURED_TPOT_GPT_OSS}


def _checkpoint(name):
    path = CHECKPOINTS[name]
    if not os.path.isdir(path):
        pytest.skip(f'measured-reference checkpoint not mounted: {path}')
    return path


def _tpot(model_path, system_name, batch, prompt_tokens, offset=STEADY_STATE_OFFSET):
    # Bb=1: these are greedy serving measurements. decode_moddeling's default is beam search 4.
    return decode_moddeling(model=model_path, batch_size=batch, input_tokens=prompt_tokens + offset,
                            output_tokens=1, Bb=1, system_name=system_name, bits='bf16')['Latency']


def _ratios(name, system_name='H100_GPU'):
    path = _checkpoint(name)
    return {key: _tpot(path, system_name, *key) / measured for key, measured in MEASURED[name].items()}


def _mean_abs_log(ratios):
    return sum(abs(math.log(r)) for r in ratios.values()) / len(ratios)


class TestAgainstMeasuredH100:

    @pytest.mark.parametrize('name', ['Qwen3-0.6B', 'gpt-oss-20b'])
    def test_every_measured_point_is_within_band(self, name):
        """Two unrelated architectures, batch 1..32, context 2k..32k. Before the per-sequence default
        was corrected these ran to 2.69x (Qwen, batch 32) and 3.42x (gpt-oss, batch 32, which also
        timed its MXFP4 experts at bf16)."""
        ratios = _ratios(name)
        outside = {k: round(r, 3) for k, r in ratios.items() if not 0.8 <= r <= 1.25}
        assert not outside, f'{name}: (batch, prompt) -> predicted/measured outside [0.8, 1.25]: {outside}'
        assert _mean_abs_log(ratios) < 0.12, f'{name}: mean |ln| {_mean_abs_log(ratios):.3f}'

    @pytest.mark.parametrize('name', ['Qwen3-0.6B', 'gpt-oss-20b'])
    def test_error_does_not_grow_with_batch(self, name):
        """The signature of the old default: a linear per-sequence charge made error climb with batch
        (Qwen 2k context: 1.08x at batch 1 -> 2.69x at batch 32). The largest-batch point at each
        context must now sit within 20% of that context's batch-1 point."""
        ratios = _ratios(name)
        for ctx in {c for _, c in ratios}:
            by_batch = sorted((b, r) for (b, c), r in ratios.items() if c == ctx)
            (_, r_small), (_, r_large) = by_batch[0], by_batch[-1]
            assert r_large / r_small < 1.2, f'{name} ctx={ctx}: batch-1 {r_small:.2f}x vs top batch {r_large:.2f}x'

    def test_the_default_per_sequence_cost_fits_inside_the_measured_step(self):
        """A physical bound, not a fit: at batch 32 / 2k the WHOLE measured step is 4.748 ms. The
        declared per-step cost plus 32 x the per-sequence cost must leave room for the weights and KV
        the step actually reads. The old 0.25 ms default asserted 1.0 + 8.0 = 9.0 ms here."""
        realism = resolve_inference_realism('H100_GPU', phase='decode')
        measured_step = MEASURED_TPOT_QWEN06B[(32, 2037)]
        declared_fixed = realism['step_overhead_ms'] + 32 * realism['per_sequence_overhead_ms']
        assert declared_fixed < 0.5 * measured_step, (
            f'declared host cost {declared_fixed:.2f} ms leaves no room in a {measured_step} ms step')

    def test_tpot_never_predicts_below_the_measured_floor(self):
        """Measured TPOT never drops below ~2.0 ms for this model at any shape; the raw roofline
        predicted 0.506 ms. A prediction under 1.5 ms means the fixed cost is missing again."""
        path = _checkpoint('Qwen3-0.6B')
        for batch, prompt in ((1, 128), (1, 512), (1, 2048)):
            assert _tpot(path, 'H100_GPU', batch, prompt, offset=0) > 1.5

    def test_the_declared_terms_strictly_improve_the_measured_qwen_set(self):
        """Holistic, not point-fitted: mean |log ratio| over every measured 0.6B point must fall by
        at least half against the same device with its fixed per-step cost switched off."""
        before = _mean_abs_log(_ratios('Qwen3-0.6B', NO_OVERHEAD))
        after = _mean_abs_log(_ratios('Qwen3-0.6B', 'H100_GPU'))
        assert after < 0.5 * before, (
            f'expected the fixed cost to at least halve the log error: {after:.3f} vs {before:.3f}')

    def test_fixed_overhead_cannot_explain_the_27b_gap(self):
        """These terms are NOT the fix for Qwen3.8-27B (measured TPOT 20.197 ms). The declared fixed
        cost for one of its decode steps is a couple of ms at most — the rest is the model-config
        defect owned elsewhere (its dimensions live in a nested `text_config`)."""
        path = _checkpoint('Qwen3.8-27B')
        realism = resolve_inference_realism('H100_GPU', phase='decode')
        model_df, _ = decode_moddeling(model=path, batch_size=1, input_tokens=2048, output_tokens=1,
                                       system_name='H100_GPU', bits='bf16', model_profilling=True)
        fixed = (realism['kernel_launch_latency_ms'] * count_repeat_aware_ops(model_df)
                 + realism['step_overhead_ms'] + realism['per_sequence_overhead_ms'])
        assert fixed < 3.0, f'fixed cost {fixed:.3f} ms is too large to be a per-step host cost'
        assert fixed < 0.25 * 20.197, (
            'the fixed cost must not be able to absorb the 27B error; that is the config defect')
