# tests/budevolve/test_roofline_analyzer.py
"""Tests for RooflineAnalyzer."""
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from llm_memory_calculator.budevolve.types import RooflineReport, OperatorInsight


@pytest.fixture
def mock_model_df():
    return pd.DataFrame({
        "Layer Name": ["Attn_Q_proj", "Attn_KV_proj", "Attn_Logit", "FFN_up", "FFN_down", "AllReduce"],
        "Op Type": ["FC", "FC", "Logit", "FC", "FC", "Sync"],
        "Bound": ["Compute", "Memory", "Memory", "Compute", "Compute", "Collective"],
        "Op Intensity": [120.5, 8.2, 3.1, 95.0, 95.0, 0.0],
        "Latency (ms)": [1.5, 0.8, 0.6, 1.2, 1.2, 0.5],
        "Compute time (ms)": [1.5, 0.3, 0.2, 1.2, 1.2, 0.0],
        "Memory time (ms)": [0.4, 0.8, 0.6, 0.5, 0.5, 0.0],
        "Communication time (ms)": [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
        "Compute Utilization": [0.9, 0.3, 0.2, 0.85, 0.85, 0.0],
        "Memory Utilization": [0.3, 0.9, 0.95, 0.4, 0.4, 0.0],
        "Communication Utilization": [0.0, 0.0, 0.0, 0.0, 0.0, 0.8],
    })


@pytest.fixture
def mock_modelling_output(mock_model_df):
    return SimpleNamespace(
        Latency=5.8, Throughput=40.0, Throughput_tokens_per_sec=5120.0,
        Runtime_breakdown=SimpleNamespace(MHA=2.9, FFN=2.4, Embedding=0.0, Collective=0.5),
        is_offload=False, model_df=mock_model_df, summary_table=MagicMock(),
    )


class TestRooflineAnalyzer:
    def test_analyze_from_model_df(self, mock_model_df):
        from llm_memory_calculator.budevolve.roofline_analyzer import RooflineAnalyzer
        analyzer = RooflineAnalyzer()
        report = analyzer.analyze_from_model_df(mock_model_df)
        assert isinstance(report, RooflineReport)
        assert len(report.per_operator) == 6
        assert report.overall_bottleneck in ("compute", "memory_bandwidth", "interconnect")

    def test_per_operator_pct_sums_to_one(self, mock_model_df):
        from llm_memory_calculator.budevolve.roofline_analyzer import RooflineAnalyzer
        analyzer = RooflineAnalyzer()
        report = analyzer.analyze_from_model_df(mock_model_df)
        total_pct = sum(op.pct_of_total for op in report.per_operator)
        assert abs(total_pct - 1.0) < 0.01

    def test_recommendations_generated(self, mock_model_df):
        from llm_memory_calculator.budevolve.roofline_analyzer import RooflineAnalyzer
        analyzer = RooflineAnalyzer()
        report = analyzer.analyze_from_model_df(mock_model_df)
        assert len(report.recommendations) > 0
        assert all(isinstance(r, str) for r in report.recommendations)

    def test_analyze_config(self, mock_modelling_output):
        from llm_memory_calculator.budevolve.roofline_analyzer import RooflineAnalyzer
        from llm_memory_calculator.budevolve.types import ServingConfig
        with patch("llm_memory_calculator.budevolve.roofline_analyzer.prefill_moddeling",
                   return_value=mock_modelling_output):
            analyzer = RooflineAnalyzer()
            cfg = ServingConfig(model="test", hardware="H100_GPU")
            report = analyzer.analyze_config(cfg, input_tokens=512)
            assert isinstance(report, RooflineReport)

    def test_format_for_prompt(self, mock_model_df):
        from llm_memory_calculator.budevolve.roofline_analyzer import RooflineAnalyzer
        analyzer = RooflineAnalyzer()
        report = analyzer.analyze_from_model_df(mock_model_df)
        prompt_text = analyzer.format_for_prompt(report)
        assert isinstance(prompt_text, str)
        assert "bottleneck" in prompt_text.lower()
        assert len(prompt_text) > 50
