"""Tests for BudEvolve CLI."""
import pytest
from unittest.mock import patch, MagicMock


class TestCLIParsing:
    def test_import(self):
        from llm_memory_calculator.budevolve.cli import create_parser
        parser = create_parser()
        assert parser is not None

    def test_optimize_command(self):
        from llm_memory_calculator.budevolve.cli import create_parser
        parser = create_parser()
        args = parser.parse_args([
            "optimize",
            "--model", "meta-llama/Meta-Llama-3.1-70B",
            "--hardware", "H100_GPU",
            "--objectives", "throughput,latency",
        ])
        assert args.command == "optimize"
        assert args.model == "meta-llama/Meta-Llama-3.1-70B"

    def test_explore_hardware_command(self):
        from llm_memory_calculator.budevolve.cli import create_parser
        parser = create_parser()
        args = parser.parse_args([
            "explore-hardware",
            "--model", "meta-llama/Meta-Llama-3.1-70B",
            "--objectives", "throughput,cost",
        ])
        assert args.command == "explore-hardware"

    def test_what_if_command(self):
        from llm_memory_calculator.budevolve.cli import create_parser
        parser = create_parser()
        args = parser.parse_args([
            "what-if",
            "--base-hardware", "A100_40GB_GPU",
            "--param", "offchip_mem_bw_gbps",
            "--range", "1600,8000",
        ])
        assert args.command == "what-if"
        assert args.param == "offchip_mem_bw_gbps"

    def test_sensitivity_command(self):
        from llm_memory_calculator.budevolve.cli import create_parser
        parser = create_parser()
        args = parser.parse_args([
            "sensitivity",
            "--model", "test",
            "--hardware", "H100_GPU",
            "--target", "throughput",
        ])
        assert args.command == "sensitivity"
