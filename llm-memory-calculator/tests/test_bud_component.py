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
