"""Unit tests for bud_component.py's source-resolution + precision helpers (TDD).

bud_component.py (at the simulator root) is the single estimation primitive Bud Gaia calls. These cover
the pure helpers that decide the perf precision: memory weights always come from the exact on-disk byte
sum, but the GenZ perf `bits` must reflect the model's ACTUAL quant or decode TPOT is mis-predicted
(e.g. an F16 GGUF modelled as int4 under-predicts decode ~2x — the regression these guard).
"""
import os
import sys

# bud_component.py lives at the simulator repo root: this file is <root>/llm-memory-calculator/tests/,
# so the root is three dirnames up.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
import bud_component as bc  # noqa: E402


class TestGgufFileTypePrecision:
    def test_f16_is_fp16_not_int4(self):
        # The regression we fixed: an F16 GGUF (file_type=1) modelled as int4 under-predicts decode ~2x.
        assert bc._precision_from_gguf_file_type(1) == "fp16"

    def test_f32_is_fp32(self):
        assert bc._precision_from_gguf_file_type(0) == "fp32"

    def test_q8_and_q5_q6_round_up_to_int8(self):
        # Round UP (int8) so we never UNDER-predict decode time (the unsafe SLO direction).
        for ft in (7, 8, 9, 16, 17, 18):
            assert bc._precision_from_gguf_file_type(ft) == "int8", ft

    def test_q4_family_is_int4(self):
        # The dominant served case: Q4_0/Q4_K_S/Q4_K_M (2,14,15) and other low-bit quants.
        for ft in (2, 3, 14, 15, 10, 11, 12, 13):
            assert bc._precision_from_gguf_file_type(ft) == "int4", ft

    def test_unknown_or_none_defaults_to_int4(self):
        assert bc._precision_from_gguf_file_type(None) == "int4"
        assert bc._precision_from_gguf_file_type(9999) == "int4"
        assert bc._precision_from_gguf_file_type("garbage") == "int4"


class TestOnnxDirPrecision:
    def test_int4_int8_fp16_by_segment(self):
        assert bc._precision_from_name("/m/cuda-int4-rtn-block-32") == "int4"
        assert bc._precision_from_name("/m/cpu-int8") == "int8"
        assert bc._precision_from_name("/m/llama31-8b-onnx-fp16") == "fp16"

    def test_incidental_substring_does_not_match(self):
        # A word that merely CONTAINS 'int8'/'f16' as a substring must NOT match (segment-exact).
        assert bc._precision_from_name("/m/checkpoint8") == "fp16"  # not int8
        assert bc._precision_from_name("/m/a3f16b9c") == "fp16"  # default, not a quant match


class TestBatchSweep:
    """`--sweep-batch-max` emits the per-batch memory+SLO curve the orchestrator picks from (max batch
    that fits a live memory budget AND meets a TTFT/TPOT SLO). KV ∝ batch; ONNX arena activation also
    ∝ batch, so the curve's memory ceiling is what bounds the dynamic batch size."""

    _CFG = ('{"model_type":"llama","num_hidden_layers":32,"hidden_size":4096,'
            '"num_attention_heads":32,"num_key_value_heads":8,"intermediate_size":14336,'
            '"vocab_size":128256}')

    def _run(self, *extra):
        import json
        import subprocess
        import sys
        proc = subprocess.run(
            [sys.executable, bc.__file__, "--config", self._CFG, "--hardware", "GB10",
             "--seq", "4096", *extra],
            capture_output=True, text=True, timeout=180,
        )
        assert proc.returncode == 0, proc.stderr[-600:]
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def test_sweep_curve_is_monotone_and_kv_linear_in_batch(self):
        d = self._run("--sweep-batch-max", "8")
        sweep = d["batch_sweep"]
        assert [p["batch"] for p in sweep] == [1, 2, 4, 8]
        gb = [p["total_runtime_gb"] for p in sweep]
        kv = [p["kv_cache_bytes"] for p in sweep]
        assert gb == sorted(gb) and gb[-1] > gb[0], gb  # memory grows with batch
        assert abs(kv[3] / kv[0] - 8.0) < 0.01, "KV must scale ~linearly with batch (8× at batch 8)"
        # Throughput rises with batch; TPOT is non-decreasing (the bandwidth knee).
        thr = [p["throughput_tok_s"] for p in sweep]
        tpot = [p["tpot_ms"] for p in sweep]
        assert thr[-1] > thr[0], thr
        assert tpot[-1] >= tpot[0], tpot

    def test_no_sweep_key_without_the_flag(self):
        d = self._run()
        assert "batch_sweep" not in d
        assert "memory" in d  # the normal single-batch result is unchanged


class TestPrefixCacheCredit:
    """`--prefix-cached-tokens` credits KV reuse to the SLO: TTFT prefills only the uncached suffix +
    the (tiny) KV-load cost, so a system-prompt-cached request can meet a tighter TTFT than a cold one."""

    _CFG = TestBatchSweep._CFG

    def _run(self, *extra):
        import json
        import subprocess
        import sys
        proc = subprocess.run(
            [sys.executable, bc.__file__, "--config", self._CFG, "--hardware", "GB10",
             "--seq", "8192", *extra],
            capture_output=True, text=True, timeout=180,
        )
        assert proc.returncode == 0, proc.stderr[-600:]
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def test_cached_prefix_cuts_ttft_with_negligible_load(self):
        full = self._run()
        cached = self._run("--prefix-cached-tokens", "4000")
        s = cached["slo"]
        assert s["prefix_cached_tokens"] == 4000
        # Caching 4000 of the ~6144 prompt tokens cuts TTFT well below the cold prefill.
        assert s["ttft_ms_prefix_cached"] < s["ttft_ms"] * 0.7, (s["ttft_ms_prefix_cached"], s["ttft_ms"])
        # The KV load is ~free on GB10 (273 GB/s): ~0.48 us/token ⇒ < 50 ms for 4000 tokens.
        assert s["prefix_kv_load_ms"] < 50, s["prefix_kv_load_ms"]
        # No flag ⇒ no cached field (cold path unchanged).
        assert "ttft_ms_prefix_cached" not in full["slo"]

    def test_cache_clamped_below_prompt_length(self):
        # Asking to cache more than the prompt must not zero/negative the suffix prefill.
        s = self._run("--prefix-cached-tokens", "999999")["slo"]
        assert s["prefix_cached_tokens"] < s["input_tokens"]
        assert s["ttft_ms_prefix_cached"] > 0


class TestUnknownHardwareFailsClosed:
    """A typo'd / unsupported --hardware must FAIL-CLOSED (exit !=0 + an error JSON), never produce a
    memory-only result with a silently-missing SLO. bud-gaia keeps no fallback, so this is where an
    unrecognized hardware becomes a refused-and-disclosed route instead of a serve-blind."""

    def test_unknown_hardware_exits_nonzero_with_error(self):
        import json
        import subprocess
        import sys

        # A real GGUF isn't needed: the hardware gate is hit before any heavy work. Use a dummy --gguf
        # path; the hardware check must fire first. (If no gguf, argparse still parses; the hw gate runs.)
        proc = subprocess.run(
            [sys.executable, bc.__file__,
             "--config", "{}", "--hardware", "TOTALLY_FAKE_GPU_9000"],
            capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode != 0, "unknown hardware must fail-closed (non-zero exit)"
        # The stdout JSON must name the offending hardware so the disclosure is actionable.
        payload = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "{}"
        try:
            err = json.loads(payload)
            assert "error" in err and "TOTALLY_FAKE_GPU_9000" in err["error"]
        except json.JSONDecodeError:
            assert "TOTALLY_FAKE_GPU_9000" in proc.stdout
