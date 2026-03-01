"""Tests for BudSimEvaluator."""
import pytest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from llm_memory_calculator.budevolve.types import ServingConfig, HardwareSpec, EvalResult


@pytest.fixture
def mock_modelling_output():
    return SimpleNamespace(
        Latency=25.0,
        Throughput=40.0,
        Throughput_tokens_per_sec=5120.0,
        Runtime_breakdown=SimpleNamespace(MHA=10.0, FFN=8.0, Embedding=2.0, Collective=5.0),
        is_offload=False,
        model_df=MagicMock(),
        summary_table=MagicMock(),
    )


class TestBudSimEvaluator:
    def test_evaluate_config_returns_eval_result(self, mock_modelling_output):
        from llm_memory_calculator.budevolve.evaluator import BudSimEvaluator

        with patch("llm_memory_calculator.budevolve.evaluator.prefill_moddeling",
                   return_value=mock_modelling_output) as mock_prefill, \
             patch("llm_memory_calculator.budevolve.evaluator.decode_moddeling",
                   return_value=mock_modelling_output) as mock_decode:
            evaluator = BudSimEvaluator()
            cfg = ServingConfig(model="test-model", hardware="H100_GPU")
            result = evaluator.evaluate_config(cfg)
            assert isinstance(result, EvalResult)
            assert result.throughput_rps == 40.0
            assert result.ttft_ms == 25.0
            assert result.tpot_ms == 25.0
            mock_prefill.assert_called_once()
            mock_decode.assert_called_once()

    def test_evaluate_config_passes_params_correctly(self, mock_modelling_output):
        from llm_memory_calculator.budevolve.evaluator import BudSimEvaluator

        with patch("llm_memory_calculator.budevolve.evaluator.prefill_moddeling",
                   return_value=mock_modelling_output) as mock_prefill, \
             patch("llm_memory_calculator.budevolve.evaluator.decode_moddeling",
                   return_value=mock_modelling_output):
            evaluator = BudSimEvaluator()
            cfg = ServingConfig(
                model="llama-70b", hardware="A100_40GB_GPU",
                tensor_parallel=4, pipeline_parallel=2,
                batch_size=64, precision="fp8",
            )
            evaluator.evaluate_config(cfg, input_tokens=1024, output_tokens=256)
            call_kwargs = mock_prefill.call_args[1]
            assert call_kwargs["model"] == "llama-70b"
            assert call_kwargs["system_name"] == "A100_40GB_GPU"
            assert call_kwargs["tensor_parallel"] == 4
            assert call_kwargs["pipeline_parallel"] == 2
            assert call_kwargs["batch_size"] == 64
            assert call_kwargs["bits"] == "fp8"
            assert call_kwargs["input_tokens"] == 1024

    def test_evaluate_config_handles_failure_gracefully(self):
        from llm_memory_calculator.budevolve.evaluator import BudSimEvaluator

        with patch("llm_memory_calculator.budevolve.evaluator.prefill_moddeling",
                   side_effect=Exception("model not found")), \
             patch("llm_memory_calculator.budevolve.evaluator.decode_moddeling",
                   side_effect=Exception("model not found")):
            evaluator = BudSimEvaluator()
            cfg = ServingConfig(model="nonexistent", hardware="H100_GPU")
            result = evaluator.evaluate_config(cfg)
            assert isinstance(result, EvalResult)
            assert result.feasible is False

    def test_evaluate_hardware_creates_system(self, mock_modelling_output):
        from llm_memory_calculator.budevolve.evaluator import BudSimEvaluator

        with patch("llm_memory_calculator.budevolve.evaluator.prefill_moddeling",
                   return_value=mock_modelling_output), \
             patch("llm_memory_calculator.budevolve.evaluator.decode_moddeling",
                   return_value=mock_modelling_output), \
             patch("llm_memory_calculator.budevolve.evaluator.System") as mock_system:
            mock_system.return_value = MagicMock()
            evaluator = BudSimEvaluator()
            hw = HardwareSpec(flops_tflops=500.0, offchip_mem_bw_gbps=3000.0, off_chip_mem_size_gb=80.0)
            result = evaluator.evaluate_hardware(hw, model="test-model", input_tokens=512, output_tokens=128)
            assert isinstance(result, EvalResult)
