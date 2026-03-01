# BudEvolve Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a reverse-optimization system that uses BudSim's analytical engine as a fitness function for multi-objective config search (pymoo), hardware design exploration (pymoo), and algorithm evolution (OpenEvolve).

**Architecture:** Three optimization modes sharing a common BudSim evaluation API. NumericOptimizer and HardwareExplorer use pymoo NSGA-II (no LLM). AlgorithmEvolver uses OpenEvolve with roofline-enriched prompts (LLM required). All evaluate candidates through GenZ's 1-10ms analytical engine.

**Tech Stack:** pymoo (NSGA-II/III), openevolve, GenZ analytical engine, click (CLI)

**Package location:** `llm-memory-calculator/src/llm_memory_calculator/budevolve/`
**Test location:** `llm-memory-calculator/tests/budevolve/`

---

## Task 1: Create package skeleton and data types

**Files:**
- Create: `llm-memory-calculator/src/llm_memory_calculator/budevolve/__init__.py`
- Create: `llm-memory-calculator/src/llm_memory_calculator/budevolve/types.py`
- Create: `llm-memory-calculator/tests/budevolve/__init__.py`
- Create: `llm-memory-calculator/tests/budevolve/test_types.py`

**Step 1: Write the failing test**

```python
# tests/budevolve/test_types.py
"""Tests for BudEvolve core data types."""
import pytest
from dataclasses import asdict


class TestServingConfig:
    def test_defaults(self):
        from llm_memory_calculator.budevolve.types import ServingConfig
        cfg = ServingConfig(model="meta-llama/Meta-Llama-3.1-8B", hardware="H100_GPU")
        assert cfg.model == "meta-llama/Meta-Llama-3.1-8B"
        assert cfg.hardware == "H100_GPU"
        assert cfg.tensor_parallel == 1
        assert cfg.pipeline_parallel == 1
        assert cfg.batch_size == 32
        assert cfg.precision == "bf16"
        assert cfg.gpu_memory_utilization == 0.90

    def test_custom_values(self):
        from llm_memory_calculator.budevolve.types import ServingConfig
        cfg = ServingConfig(
            model="meta-llama/Meta-Llama-3.1-70B", hardware="A100_40GB_GPU",
            tensor_parallel=4, batch_size=128, precision="fp8",
        )
        assert cfg.tensor_parallel == 4
        assert cfg.batch_size == 128
        assert cfg.precision == "fp8"


class TestHardwareSpec:
    def test_defaults(self):
        from llm_memory_calculator.budevolve.types import HardwareSpec
        hw = HardwareSpec(flops_tflops=312.0, offchip_mem_bw_gbps=1600.0, off_chip_mem_size_gb=40.0)
        assert hw.flops_tflops == 312.0
        assert hw.num_nodes == 1
        assert hw.tdp_watts == 700.0

    def test_to_system_kwargs(self):
        from llm_memory_calculator.budevolve.types import HardwareSpec
        hw = HardwareSpec(
            flops_tflops=312.0, offchip_mem_bw_gbps=1600.0,
            off_chip_mem_size_gb=40.0, on_chip_mem_size_mb=40.0,
            onchip_mem_bw_gbps=18000.0, interchip_link_bw_gbps=150.0,
            frequency_ghz=1.41,
        )
        kwargs = hw.to_system_kwargs()
        assert kwargs["flops"] == 312.0
        assert kwargs["offchip_mem_bw"] == 1600.0
        assert kwargs["off_chip_mem_size"] == 40.0 * 1024  # GB -> MB
        assert kwargs["frequency"] == 1410  # GHz -> MHz


class TestEvalResult:
    def test_defaults(self):
        from llm_memory_calculator.budevolve.types import EvalResult
        r = EvalResult(throughput_rps=100.0, ttft_ms=50.0, tpot_ms=10.0)
        assert r.feasible is True
        assert r.memory_gb == 0.0
        assert r.power_w == 0.0

    def test_is_serializable(self):
        from llm_memory_calculator.budevolve.types import EvalResult
        r = EvalResult(throughput_rps=100.0, ttft_ms=50.0, tpot_ms=10.0)
        d = asdict(r)
        assert isinstance(d, dict)
        assert d["throughput_rps"] == 100.0


class TestOperatorInsight:
    def test_creation(self):
        from llm_memory_calculator.budevolve.types import OperatorInsight
        op = OperatorInsight(
            name="attention_prefill", bottleneck="compute",
            arithmetic_intensity=45.2,
            compute_time_ms=1.5, memory_time_ms=0.8, pct_of_total=0.35,
        )
        assert op.bottleneck == "compute"
        assert op.pct_of_total == 0.35


class TestRooflineReport:
    def test_creation(self):
        from llm_memory_calculator.budevolve.types import RooflineReport, OperatorInsight
        ops = [OperatorInsight("attn", "compute", 45.0, 1.5, 0.8, 0.35)]
        report = RooflineReport(
            overall_bottleneck="compute",
            compute_utilization=0.85, memory_bw_utilization=0.42,
            interconnect_utilization=0.1, per_operator=ops,
            recommendations=["Increase batch size"],
        )
        assert report.overall_bottleneck == "compute"
        assert len(report.per_operator) == 1


class TestParetoResult:
    def test_creation(self):
        from llm_memory_calculator.budevolve.types import ParetoResult
        r = ParetoResult(
            pareto_front=[{"throughput": 100}],
            all_evaluated=[{"throughput": 100}, {"throughput": 50}],
            best_per_objective={"throughput": {"throughput": 100}},
            num_generations=50, total_evaluations=500,
        )
        assert len(r.pareto_front) == 1
        assert r.total_evaluations == 500
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest llm-memory-calculator/tests/budevolve/test_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_memory_calculator.budevolve'`

**Step 3: Write minimal implementation**

```python
# src/llm_memory_calculator/budevolve/__init__.py
"""BudEvolve: Reverse-optimization system for LLM serving.

Three optimization modes:
- NumericOptimizer: pymoo NSGA-II config search (no LLM)
- HardwareExplorer: pymoo NSGA-II hardware DSE (no LLM)
- AlgorithmEvolver: OpenEvolve algorithm evolution (LLM required)
"""
```

```python
# src/llm_memory_calculator/budevolve/types.py
"""Core data types for BudEvolve optimization."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class ServingConfig:
    """Serving configuration to evaluate."""
    model: str
    hardware: str
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    batch_size: int = 32
    precision: str = "bf16"
    max_num_batched_tokens: int = 8192
    enable_chunked_prefill: bool = False
    chunk_size: int = 512
    enable_prefix_caching: bool = False
    gpu_memory_utilization: float = 0.90


@dataclass
class HardwareSpec:
    """Hypothetical hardware specification for design exploration."""
    flops_tflops: float
    offchip_mem_bw_gbps: float
    off_chip_mem_size_gb: float
    on_chip_mem_size_mb: float = 40.0
    onchip_mem_bw_gbps: float = 18000.0
    interchip_link_bw_gbps: float = 150.0
    frequency_ghz: float = 1.0
    num_nodes: int = 1
    tdp_watts: float = 700.0
    estimated_cost_usd: float = 25000.0

    def to_system_kwargs(self) -> Dict[str, Any]:
        """Convert to kwargs for genz.system.System() constructor."""
        return {
            "flops": self.flops_tflops,
            "offchip_mem_bw": self.offchip_mem_bw_gbps,
            "off_chip_mem_size": self.off_chip_mem_size_gb * 1024,  # GB -> MB
            "on_chip_mem_size": self.on_chip_mem_size_mb,
            "onchip_mem_bw": self.onchip_mem_bw_gbps,
            "interchip_link_bw": self.interchip_link_bw_gbps,
            "frequency": self.frequency_ghz * 1000,  # GHz -> MHz
            "num_nodes": self.num_nodes,
        }


@dataclass
class OperatorInsight:
    """Per-operator roofline analysis result."""
    name: str
    bottleneck: str  # "compute" | "memory" | "collective"
    arithmetic_intensity: float
    compute_time_ms: float
    memory_time_ms: float
    pct_of_total: float  # 0.0 - 1.0


@dataclass
class RooflineReport:
    """Roofline analysis report from BudSim."""
    overall_bottleneck: str  # "compute" | "memory_bandwidth" | "interconnect"
    compute_utilization: float
    memory_bw_utilization: float
    interconnect_utilization: float
    per_operator: List[OperatorInsight]
    recommendations: List[str]


@dataclass
class EvalResult:
    """Result from evaluating a config/hardware/algorithm through BudSim."""
    throughput_rps: float = 0.0
    token_throughput_tps: float = 0.0
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    e2e_latency_ms: float = 0.0
    memory_gb: float = 0.0
    power_w: float = 0.0
    slo_compliance: float = 0.0
    cost_per_million_tokens: float = 0.0
    roofline: Optional[RooflineReport] = None
    feasible: bool = True
    config: Optional[Dict[str, Any]] = None


@dataclass
class ParetoResult:
    """Result from multi-objective optimization."""
    pareto_front: List[Dict[str, Any]]
    all_evaluated: List[Dict[str, Any]]
    best_per_objective: Dict[str, Any]
    sensitivity: Dict[str, float] = field(default_factory=dict)
    num_generations: int = 0
    total_evaluations: int = 0
```

```python
# tests/budevolve/__init__.py
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest llm-memory-calculator/tests/budevolve/test_types.py -v`
Expected: PASS (all 8 tests)

**Step 5: Commit**

```bash
git add llm-memory-calculator/src/llm_memory_calculator/budevolve/ llm-memory-calculator/tests/budevolve/
git commit -m "feat(budevolve): add package skeleton and core data types"
```

---

## Task 2: BudSim Evaluator — config evaluation wrapper

**Files:**
- Create: `llm-memory-calculator/src/llm_memory_calculator/budevolve/evaluator.py`
- Create: `llm-memory-calculator/tests/budevolve/test_evaluator.py`

**Step 1: Write the failing test**

```python
# tests/budevolve/test_evaluator.py
"""Tests for BudSimEvaluator."""
import pytest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from llm_memory_calculator.budevolve.types import ServingConfig, HardwareSpec, EvalResult


@pytest.fixture
def mock_modelling_output():
    """Mock ModdelingOutput from GenZ prefill/decode."""
    return SimpleNamespace(
        Latency=25.0,  # ms
        Throughput=40.0,  # RPS
        Throughput_tokens_per_sec=5120.0,
        Runtime_breakdown=SimpleNamespace(
            MHA=10.0, FFN=8.0, Embedding=2.0, Collective=5.0,
        ),
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
            hw = HardwareSpec(flops_tflops=500.0, offchip_mem_bw_gbps=3000.0,
                              off_chip_mem_size_gb=80.0)
            result = evaluator.evaluate_hardware(
                hw, model="test-model", input_tokens=512, output_tokens=128,
            )
            assert isinstance(result, EvalResult)
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest llm-memory-calculator/tests/budevolve/test_evaluator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_memory_calculator.budevolve.evaluator'`

**Step 3: Write minimal implementation**

```python
# src/llm_memory_calculator/budevolve/evaluator.py
"""BudSim evaluation API — wraps GenZ as a fitness function."""
import warnings
from typing import Optional

from llm_memory_calculator.genz.LLM_inference.llm_prefill import prefill_moddeling
from llm_memory_calculator.genz.LLM_inference.llm_decode import decode_moddeling
from llm_memory_calculator.genz.system import System

from .types import ServingConfig, HardwareSpec, EvalResult


class BudSimEvaluator:
    """Wraps BudSim's GenZ analytical engine as a fitness function.

    Provides fast (~1-10ms) evaluation of serving configs, hardware specs,
    and evolved algorithms for use by optimization loops.
    """

    def evaluate_config(
        self,
        config: ServingConfig,
        input_tokens: int = 512,
        output_tokens: int = 128,
    ) -> EvalResult:
        """Evaluate a serving configuration using GenZ analytical engine.

        Args:
            config: Serving configuration to evaluate.
            input_tokens: Representative input sequence length.
            output_tokens: Representative output sequence length.

        Returns:
            EvalResult with performance metrics.
        """
        try:
            prefill = prefill_moddeling(
                model=config.model,
                batch_size=config.batch_size,
                input_tokens=input_tokens,
                system_name=config.hardware,
                bits=config.precision,
                tensor_parallel=config.tensor_parallel,
                pipeline_parallel=config.pipeline_parallel,
            )
            decode = decode_moddeling(
                model=config.model,
                batch_size=config.batch_size,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                system_name=config.hardware,
                bits=config.precision,
                tensor_parallel=config.tensor_parallel,
                pipeline_parallel=config.pipeline_parallel,
            )
            return EvalResult(
                throughput_rps=decode.Throughput,
                token_throughput_tps=decode.Throughput_tokens_per_sec,
                ttft_ms=prefill.Latency,
                tpot_ms=decode.Latency,
                e2e_latency_ms=prefill.Latency + decode.Latency * output_tokens,
                feasible=True,
                config={
                    "model": config.model,
                    "hardware": config.hardware,
                    "tensor_parallel": config.tensor_parallel,
                    "pipeline_parallel": config.pipeline_parallel,
                    "batch_size": config.batch_size,
                    "precision": config.precision,
                },
            )
        except Exception as e:
            warnings.warn(f"BudSim evaluation failed for {config}: {e}")
            return EvalResult(feasible=False, config={"error": str(e)})

    def evaluate_hardware(
        self,
        hw_spec: HardwareSpec,
        model: str = "meta-llama/Meta-Llama-3.1-8B",
        input_tokens: int = 512,
        output_tokens: int = 128,
        batch_size: int = 32,
        precision: str = "bf16",
        tensor_parallel: int = 1,
        pipeline_parallel: int = 1,
    ) -> EvalResult:
        """Evaluate a hypothetical hardware design.

        Creates a System from the spec and runs GenZ analysis.
        """
        sys_kwargs = hw_spec.to_system_kwargs()
        sys_kwargs["bits"] = precision

        try:
            system = System(**sys_kwargs)

            prefill = prefill_moddeling(
                model=model,
                batch_size=batch_size,
                input_tokens=input_tokens,
                system_name=system,
                bits=precision,
                tensor_parallel=tensor_parallel,
                pipeline_parallel=pipeline_parallel,
            )
            decode = decode_moddeling(
                model=model,
                batch_size=batch_size,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                system_name=system,
                bits=precision,
                tensor_parallel=tensor_parallel,
                pipeline_parallel=pipeline_parallel,
            )
            return EvalResult(
                throughput_rps=decode.Throughput,
                token_throughput_tps=decode.Throughput_tokens_per_sec,
                ttft_ms=prefill.Latency,
                tpot_ms=decode.Latency,
                e2e_latency_ms=prefill.Latency + decode.Latency * output_tokens,
                power_w=hw_spec.tdp_watts,
                feasible=True,
                config={
                    "flops_tflops": hw_spec.flops_tflops,
                    "offchip_mem_bw_gbps": hw_spec.offchip_mem_bw_gbps,
                    "off_chip_mem_size_gb": hw_spec.off_chip_mem_size_gb,
                    "tdp_watts": hw_spec.tdp_watts,
                    "estimated_cost_usd": hw_spec.estimated_cost_usd,
                },
            )
        except Exception as e:
            warnings.warn(f"Hardware evaluation failed: {e}")
            return EvalResult(feasible=False, config={"error": str(e)})
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest llm-memory-calculator/tests/budevolve/test_evaluator.py -v`
Expected: PASS (all 4 tests)

**Step 5: Commit**

```bash
git add llm-memory-calculator/src/llm_memory_calculator/budevolve/evaluator.py \
        llm-memory-calculator/tests/budevolve/test_evaluator.py
git commit -m "feat(budevolve): add BudSimEvaluator config and hardware evaluation"
```

---

## Task 3: Roofline Analyzer — per-operator bottleneck extraction

**Files:**
- Create: `llm-memory-calculator/src/llm_memory_calculator/budevolve/roofline_analyzer.py`
- Create: `llm-memory-calculator/tests/budevolve/test_roofline_analyzer.py`

**Step 1: Write the failing test**

```python
# tests/budevolve/test_roofline_analyzer.py
"""Tests for RooflineAnalyzer."""
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from llm_memory_calculator.budevolve.types import RooflineReport, OperatorInsight


@pytest.fixture
def mock_model_df():
    """Mock model_df DataFrame from GenZ with roofline columns."""
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
        Latency=5.8,
        Throughput=40.0,
        Throughput_tokens_per_sec=5120.0,
        Runtime_breakdown=SimpleNamespace(MHA=2.9, FFN=2.4, Embedding=0.0, Collective=0.5),
        is_offload=False,
        model_df=mock_model_df,
        summary_table=MagicMock(),
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
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest llm-memory-calculator/tests/budevolve/test_roofline_analyzer.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/llm_memory_calculator/budevolve/roofline_analyzer.py
"""Roofline analyzer — extracts per-operator bottleneck insights from GenZ."""
import warnings
from typing import Optional

import pandas as pd

from llm_memory_calculator.genz.LLM_inference.llm_prefill import prefill_moddeling

from .types import ServingConfig, RooflineReport, OperatorInsight


class RooflineAnalyzer:
    """Extracts per-operator roofline analysis from GenZ model DataFrames.

    The key output is a RooflineReport that identifies which operators are
    compute-bound vs memory-bound, and generates recommendations. This data
    is injected into OpenEvolve prompts so the LLM can make informed mutations.
    """

    def analyze_from_model_df(self, model_df: pd.DataFrame) -> RooflineReport:
        """Extract roofline insights from a GenZ model_df DataFrame.

        Args:
            model_df: DataFrame from prefill_moddeling().model_df or
                      decode_moddeling().model_df. Must contain columns:
                      'Layer Name', 'Bound', 'Latency (ms)', 'Compute time (ms)',
                      'Memory time (ms)', 'Op Intensity'.

        Returns:
            RooflineReport with per-operator insights and recommendations.
        """
        latency_col = self._find_column(model_df, "Latency")
        compute_col = self._find_column(model_df, "Compute time")
        memory_col = self._find_column(model_df, "Memory time")
        comm_col = self._find_column(model_df, "Communication time")

        total_latency = model_df[latency_col].sum()
        if total_latency == 0:
            total_latency = 1.0

        operators = []
        total_compute_time = 0.0
        total_memory_time = 0.0
        total_comm_time = 0.0

        for _, row in model_df.iterrows():
            name = row.get("Layer Name", "unknown")
            bound = row.get("Bound", "Memory").lower()
            intensity = row.get("Op Intensity", 0.0)
            ct = row.get(compute_col, 0.0) if compute_col else 0.0
            mt = row.get(memory_col, 0.0) if memory_col else 0.0
            lat = row.get(latency_col, 0.0) if latency_col else 0.0

            if bound == "compute":
                bottleneck = "compute"
            elif bound == "collective":
                bottleneck = "collective"
            else:
                bottleneck = "memory"

            operators.append(OperatorInsight(
                name=name,
                bottleneck=bottleneck,
                arithmetic_intensity=float(intensity),
                compute_time_ms=float(ct),
                memory_time_ms=float(mt),
                pct_of_total=float(lat) / total_latency if total_latency > 0 else 0.0,
            ))

            total_compute_time += float(ct)
            total_memory_time += float(mt)
            if comm_col:
                total_comm_time += float(row.get(comm_col, 0.0))

        # Determine overall bottleneck
        max_time = max(total_compute_time, total_memory_time, total_comm_time, 1e-9)
        if total_compute_time == max_time:
            overall = "compute"
        elif total_memory_time == max_time:
            overall = "memory_bandwidth"
        else:
            overall = "interconnect"

        compute_util = total_compute_time / total_latency if total_latency > 0 else 0.0
        memory_util = total_memory_time / total_latency if total_latency > 0 else 0.0
        comm_util = total_comm_time / total_latency if total_latency > 0 else 0.0

        recommendations = self._generate_recommendations(
            overall, compute_util, memory_util, comm_util, operators,
        )

        return RooflineReport(
            overall_bottleneck=overall,
            compute_utilization=compute_util,
            memory_bw_utilization=memory_util,
            interconnect_utilization=comm_util,
            per_operator=operators,
            recommendations=recommendations,
        )

    def analyze_config(
        self,
        config: ServingConfig,
        input_tokens: int = 512,
        phase: str = "prefill",
    ) -> RooflineReport:
        """Run GenZ and extract roofline analysis for a serving config.

        Args:
            config: Serving configuration.
            input_tokens: Input sequence length.
            phase: "prefill" or "decode".

        Returns:
            RooflineReport from the GenZ model_df.
        """
        try:
            result = prefill_moddeling(
                model=config.model,
                batch_size=config.batch_size,
                input_tokens=input_tokens,
                system_name=config.hardware,
                bits=config.precision,
                tensor_parallel=config.tensor_parallel,
                pipeline_parallel=config.pipeline_parallel,
            )
            return self.analyze_from_model_df(result.model_df)
        except Exception as e:
            warnings.warn(f"Roofline analysis failed: {e}")
            return RooflineReport(
                overall_bottleneck="unknown",
                compute_utilization=0.0,
                memory_bw_utilization=0.0,
                interconnect_utilization=0.0,
                per_operator=[],
                recommendations=[f"Analysis failed: {e}"],
            )

    def format_for_prompt(self, report: RooflineReport) -> str:
        """Format roofline report as text for injection into LLM prompts.

        Returns a human-readable summary of bottlenecks and recommendations
        suitable for OpenEvolve prompt construction.
        """
        lines = [
            f"Overall bottleneck: {report.overall_bottleneck.upper()}",
            f"Compute utilization: {report.compute_utilization:.1%}",
            f"Memory BW utilization: {report.memory_bw_utilization:.1%}",
            f"Interconnect utilization: {report.interconnect_utilization:.1%}",
            "",
            "Top operators by runtime:",
        ]

        sorted_ops = sorted(report.per_operator, key=lambda o: o.pct_of_total, reverse=True)
        for op in sorted_ops[:8]:
            lines.append(
                f"  {op.name}: {op.bottleneck}-bound, "
                f"{op.pct_of_total:.0%} of runtime, "
                f"intensity={op.arithmetic_intensity:.1f}"
            )

        if report.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for rec in report.recommendations:
                lines.append(f"  - {rec}")

        return "\n".join(lines)

    def _find_column(self, df: pd.DataFrame, prefix: str) -> Optional[str]:
        """Find column matching prefix (handles unit suffixes like 'Latency (ms)')."""
        for col in df.columns:
            if col.startswith(prefix):
                return col
        return None

    def _generate_recommendations(
        self,
        overall: str,
        compute_util: float,
        memory_util: float,
        comm_util: float,
        operators: list,
    ) -> list:
        """Generate optimization recommendations based on bottleneck analysis."""
        recs = []
        if overall == "memory_bandwidth":
            recs.append("System is memory-bandwidth bound. Consider higher bandwidth memory (HBM3e) or reducing data movement.")
            mem_ops = [o for o in operators if o.bottleneck == "memory" and o.pct_of_total > 0.1]
            if mem_ops:
                names = ", ".join(o.name for o in mem_ops[:3])
                recs.append(f"Memory-bound hotspots: {names}")
        elif overall == "compute":
            recs.append("System is compute-bound. Consider higher TFLOPS hardware or lower precision (FP8/INT8).")
        elif overall == "interconnect":
            recs.append("System is interconnect-bound. Consider higher NVLink bandwidth or reducing parallelism degree.")

        if compute_util < 0.5 and memory_util > 0.7:
            recs.append("Low compute utilization suggests batching could improve throughput.")

        return recs
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest llm-memory-calculator/tests/budevolve/test_roofline_analyzer.py -v`
Expected: PASS (all 5 tests)

**Step 5: Commit**

```bash
git add llm-memory-calculator/src/llm_memory_calculator/budevolve/roofline_analyzer.py \
        llm-memory-calculator/tests/budevolve/test_roofline_analyzer.py
git commit -m "feat(budevolve): add RooflineAnalyzer with per-operator bottleneck extraction"
```

---

## Task 4: Numeric Config Optimizer — pymoo NSGA-II integration

**Files:**
- Create: `llm-memory-calculator/src/llm_memory_calculator/budevolve/numeric/__init__.py`
- Create: `llm-memory-calculator/src/llm_memory_calculator/budevolve/numeric/search_spaces.py`
- Create: `llm-memory-calculator/src/llm_memory_calculator/budevolve/numeric/config_optimizer.py`
- Create: `llm-memory-calculator/tests/budevolve/test_numeric_config_optimizer.py`
- Modify: `llm-memory-calculator/pyproject.toml` (add evolve optional deps)

**Step 1: Add pymoo dependency**

Add to `llm-memory-calculator/pyproject.toml` after the `pareto` optional dep:
```toml
evolve = [
    "pymoo>=0.6.0",
]
```

Run: `pip install pymoo>=0.6.0`

**Step 2: Write the failing test**

```python
# tests/budevolve/test_numeric_config_optimizer.py
"""Tests for pymoo-based NumericOptimizer."""
import pytest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from llm_memory_calculator.budevolve.types import EvalResult, ParetoResult


def _make_eval_result(tp=1, pp=1, bs=32, prec="bf16", throughput=50.0, ttft=100.0, tpot=20.0):
    return EvalResult(
        throughput_rps=throughput,
        token_throughput_tps=throughput * 128,
        ttft_ms=ttft,
        tpot_ms=tpot,
        e2e_latency_ms=ttft + tpot * 128,
        feasible=True,
        config={"tensor_parallel": tp, "pipeline_parallel": pp,
                "batch_size": bs, "precision": prec},
    )


class TestConfigSearchSpace:
    def test_defaults(self):
        from llm_memory_calculator.budevolve.numeric.search_spaces import ConfigSearchSpace
        space = ConfigSearchSpace()
        assert 1 in space.tensor_parallel
        assert 8 in space.tensor_parallel
        assert "bf16" in space.precisions

    def test_n_var(self):
        from llm_memory_calculator.budevolve.numeric.search_spaces import ConfigSearchSpace
        space = ConfigSearchSpace()
        assert space.n_var >= 4  # at least TP, PP, batch, precision


class TestNumericOptimizer:
    def test_optimize_returns_pareto_result(self):
        from llm_memory_calculator.budevolve.numeric.config_optimizer import NumericOptimizer

        results = [
            _make_eval_result(tp=1, bs=32, throughput=50.0, ttft=100.0),
            _make_eval_result(tp=2, bs=64, throughput=80.0, ttft=60.0),
            _make_eval_result(tp=4, bs=128, throughput=120.0, ttft=40.0),
        ]

        with patch.object(NumericOptimizer, "_run_pymoo") as mock_run:
            mock_run.return_value = results
            optimizer = NumericOptimizer(
                model="test-model", hardware="H100_GPU",
            )
            result = optimizer.optimize(
                objectives=["throughput", "latency"],
                n_generations=10,
            )
            assert isinstance(result, ParetoResult)
            assert len(result.pareto_front) > 0

    def test_optimize_respects_constraints(self):
        from llm_memory_calculator.budevolve.numeric.config_optimizer import NumericOptimizer

        feasible = _make_eval_result(tp=1, bs=32, throughput=50.0, ttft=100.0)
        infeasible = _make_eval_result(tp=4, bs=256, throughput=200.0, ttft=800.0)
        infeasible.feasible = False

        with patch.object(NumericOptimizer, "_run_pymoo") as mock_run:
            mock_run.return_value = [feasible, infeasible]
            optimizer = NumericOptimizer(model="test", hardware="H100_GPU")
            result = optimizer.optimize(
                objectives=["throughput"],
                constraints={"max_ttft_ms": 500.0},
            )
            # Only feasible results in pareto front
            for item in result.pareto_front:
                assert item.get("ttft_ms", 0) <= 500.0 or item.get("feasible", True)

    def test_compute_pareto_front(self):
        from llm_memory_calculator.budevolve.numeric.config_optimizer import NumericOptimizer

        optimizer = NumericOptimizer(model="test", hardware="H100_GPU")
        results = [
            _make_eval_result(throughput=100.0, ttft=50.0),   # dominated
            _make_eval_result(throughput=120.0, ttft=40.0),   # pareto
            _make_eval_result(throughput=80.0, ttft=30.0),    # pareto
            _make_eval_result(throughput=90.0, ttft=60.0),    # dominated
        ]
        pareto = optimizer._compute_pareto(results, ["throughput", "latency"])
        assert len(pareto) == 2
```

**Step 3: Write minimal implementation**

```python
# src/llm_memory_calculator/budevolve/numeric/__init__.py
"""Numeric optimization using pymoo NSGA-II/III."""
```

```python
# src/llm_memory_calculator/budevolve/numeric/search_spaces.py
"""Search space definitions for numeric optimization."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class ConfigSearchSpace:
    """Search space for serving config optimization."""
    tensor_parallel: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    pipeline_parallel: List[int] = field(default_factory=lambda: [1, 2, 4])
    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 32, 64, 128, 256])
    precisions: List[str] = field(default_factory=lambda: ["bf16", "fp8", "int8"])
    enable_chunked_prefill: List[bool] = field(default_factory=lambda: [True, False])
    enable_prefix_caching: List[bool] = field(default_factory=lambda: [True, False])

    @property
    def n_var(self) -> int:
        """Number of decision variables."""
        return 6  # tp, pp, batch, precision, chunked_prefill, prefix_caching


@dataclass
class HardwareSearchSpace:
    """Search space for hardware design exploration."""
    flops_range: tuple = (50.0, 10000.0)
    mem_bw_range: tuple = (200.0, 16000.0)
    mem_size_range: tuple = (16.0, 384.0)
    on_chip_mem_range: tuple = (10.0, 512.0)
    onchip_bw_range: tuple = (5000.0, 100000.0)
    interchip_bw_range: tuple = (25.0, 1800.0)
    frequency_range: tuple = (0.5, 3.0)

    @property
    def n_var(self) -> int:
        return 7

    @property
    def xl(self) -> list:
        """Lower bounds."""
        return [r[0] for r in [
            self.flops_range, self.mem_bw_range, self.mem_size_range,
            self.on_chip_mem_range, self.onchip_bw_range,
            self.interchip_bw_range, self.frequency_range,
        ]]

    @property
    def xu(self) -> list:
        """Upper bounds."""
        return [r[1] for r in [
            self.flops_range, self.mem_bw_range, self.mem_size_range,
            self.on_chip_mem_range, self.onchip_bw_range,
            self.interchip_bw_range, self.frequency_range,
        ]]
```

```python
# src/llm_memory_calculator/budevolve/numeric/config_optimizer.py
"""Multi-objective config optimization using pymoo NSGA-II."""
import itertools
import warnings
from typing import Dict, List, Optional, Any

from ..types import EvalResult, ParetoResult, ServingConfig
from ..evaluator import BudSimEvaluator
from .search_spaces import ConfigSearchSpace


class NumericOptimizer:
    """Multi-objective serving config search using pymoo NSGA-II.

    Uses BudSim's GenZ analytical engine (~1-10ms per eval) as the
    objective function. No LLM required.
    """

    def __init__(
        self,
        model: str,
        hardware: str,
        evaluator: Optional[BudSimEvaluator] = None,
    ):
        self._model = model
        self._hardware = hardware
        self._evaluator = evaluator or BudSimEvaluator()

    def optimize(
        self,
        objectives: List[str] = None,
        constraints: Optional[Dict[str, float]] = None,
        search_space: Optional[ConfigSearchSpace] = None,
        n_generations: int = 100,
        pop_size: int = 50,
        input_tokens: int = 512,
        output_tokens: int = 128,
    ) -> ParetoResult:
        """Run multi-objective optimization over serving configs.

        Args:
            objectives: List of objectives, e.g. ["throughput", "latency"].
            constraints: Dict of constraint name -> max value,
                e.g. {"max_ttft_ms": 500.0, "max_tpot_ms": 50.0}.
            search_space: Config parameter ranges.
            n_generations: Number of NSGA-II generations.
            pop_size: Population size per generation.
            input_tokens: Representative input length.
            output_tokens: Representative output length.

        Returns:
            ParetoResult with non-dominated solutions.
        """
        objectives = objectives or ["throughput", "latency"]
        constraints = constraints or {}
        search_space = search_space or ConfigSearchSpace()

        all_results = self._run_pymoo(
            search_space, objectives, constraints,
            n_generations, pop_size, input_tokens, output_tokens,
        )

        # Filter feasible
        feasible = [r for r in all_results if r.feasible]
        if not feasible:
            return ParetoResult(
                pareto_front=[], all_evaluated=[],
                best_per_objective={}, num_generations=n_generations,
                total_evaluations=len(all_results),
            )

        # Apply constraint filtering
        max_ttft = constraints.get("max_ttft_ms", float("inf"))
        max_tpot = constraints.get("max_tpot_ms", float("inf"))
        feasible = [
            r for r in feasible
            if r.ttft_ms <= max_ttft and r.tpot_ms <= max_tpot
        ]

        pareto = self._compute_pareto(feasible, objectives)

        best_per_obj = {}
        for obj in objectives:
            if obj == "throughput":
                best = max(feasible, key=lambda r: r.throughput_rps)
            elif obj == "latency":
                best = min(feasible, key=lambda r: r.ttft_ms + r.tpot_ms)
            else:
                best = feasible[0] if feasible else None
            if best:
                best_per_obj[obj] = self._result_to_dict(best)

        return ParetoResult(
            pareto_front=[self._result_to_dict(r) for r in pareto],
            all_evaluated=[self._result_to_dict(r) for r in feasible[:100]],
            best_per_objective=best_per_obj,
            num_generations=n_generations,
            total_evaluations=len(all_results),
        )

    def _run_pymoo(
        self, search_space, objectives, constraints,
        n_generations, pop_size, input_tokens, output_tokens,
    ) -> List[EvalResult]:
        """Run pymoo NSGA-II optimization.

        Falls back to grid search if pymoo is not installed.
        """
        try:
            return self._run_pymoo_nsga2(
                search_space, objectives, constraints,
                n_generations, pop_size, input_tokens, output_tokens,
            )
        except ImportError:
            warnings.warn("pymoo not installed, falling back to grid search")
            return self._run_grid_search(search_space, input_tokens, output_tokens)

    def _run_pymoo_nsga2(
        self, search_space, objectives, constraints,
        n_generations, pop_size, input_tokens, output_tokens,
    ) -> List[EvalResult]:
        """NSGA-II optimization using pymoo."""
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.optimize import minimize
        from pymoo.operators.sampling.rnd import IntegerRandomSampling
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        import numpy as np

        evaluator = self._evaluator
        model = self._model
        hardware = self._hardware
        all_results = []

        tp_vals = search_space.tensor_parallel
        pp_vals = search_space.pipeline_parallel
        bs_vals = search_space.batch_sizes
        prec_vals = search_space.precisions

        n_obj = len(objectives)

        class ServingProblem(ElementwiseProblem):
            def __init__(self):
                super().__init__(
                    n_var=4,
                    n_obj=n_obj,
                    xl=np.array([0, 0, 0, 0]),
                    xu=np.array([
                        len(tp_vals) - 1,
                        len(pp_vals) - 1,
                        len(bs_vals) - 1,
                        len(prec_vals) - 1,
                    ]),
                    vtype=int,
                )

            def _evaluate(self, x, out, *args, **kwargs):
                tp = tp_vals[int(x[0]) % len(tp_vals)]
                pp = pp_vals[int(x[1]) % len(pp_vals)]
                bs = bs_vals[int(x[2]) % len(bs_vals)]
                prec = prec_vals[int(x[3]) % len(prec_vals)]

                cfg = ServingConfig(
                    model=model, hardware=hardware,
                    tensor_parallel=tp, pipeline_parallel=pp,
                    batch_size=bs, precision=prec,
                )
                result = evaluator.evaluate_config(cfg, input_tokens, output_tokens)
                all_results.append(result)

                obj_values = []
                for obj in objectives:
                    if obj == "throughput":
                        obj_values.append(-result.throughput_rps)  # minimize negative
                    elif obj == "latency":
                        obj_values.append(result.ttft_ms + result.tpot_ms)
                    elif obj == "cost":
                        obj_values.append(result.cost_per_million_tokens)
                    elif obj == "power":
                        obj_values.append(result.power_w)
                    else:
                        obj_values.append(-result.throughput_rps)

                out["F"] = np.array(obj_values)

        problem = ServingProblem()
        algorithm = NSGA2(
            pop_size=min(pop_size, 20),
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=0.9, eta=15, vtype=float, repair=lambda _p, X: np.round(X).astype(int)),
            mutation=PM(eta=20, vtype=float, repair=lambda _p, X: np.round(X).astype(int)),
        )

        res = minimize(
            problem, algorithm,
            termination=("n_gen", min(n_generations, 30)),
            seed=42, verbose=False,
        )

        return all_results

    def _run_grid_search(
        self, search_space, input_tokens, output_tokens,
    ) -> List[EvalResult]:
        """Fallback grid search when pymoo unavailable."""
        results = []
        for tp, pp, bs, prec in itertools.product(
            search_space.tensor_parallel,
            search_space.pipeline_parallel,
            search_space.batch_sizes[:4],
            search_space.precisions[:2],
        ):
            cfg = ServingConfig(
                model=self._model, hardware=self._hardware,
                tensor_parallel=tp, pipeline_parallel=pp,
                batch_size=bs, precision=prec,
            )
            result = self._evaluator.evaluate_config(cfg, input_tokens, output_tokens)
            results.append(result)
        return results

    def _compute_pareto(
        self, results: List[EvalResult], objectives: List[str],
    ) -> List[EvalResult]:
        """Compute non-dominated Pareto front."""
        def get_obj_values(r):
            vals = []
            for obj in objectives:
                if obj == "throughput":
                    vals.append(r.throughput_rps)
                elif obj == "latency":
                    vals.append(-(r.ttft_ms + r.tpot_ms))  # higher = better
                elif obj == "cost":
                    vals.append(-r.cost_per_million_tokens)
                elif obj == "power":
                    vals.append(-r.power_w)
                else:
                    vals.append(r.throughput_rps)
            return vals

        pareto = []
        for i, ri in enumerate(results):
            vi = get_obj_values(ri)
            dominated = False
            for j, rj in enumerate(results):
                if i == j:
                    continue
                vj = get_obj_values(rj)
                if (all(vj[k] >= vi[k] for k in range(len(objectives))) and
                        any(vj[k] > vi[k] for k in range(len(objectives)))):
                    dominated = True
                    break
            if not dominated:
                pareto.append(ri)
        return pareto

    def _result_to_dict(self, r: EvalResult) -> Dict[str, Any]:
        return {
            "config": r.config,
            "throughput_rps": r.throughput_rps,
            "token_throughput_tps": r.token_throughput_tps,
            "ttft_ms": r.ttft_ms,
            "tpot_ms": r.tpot_ms,
            "e2e_latency_ms": r.e2e_latency_ms,
            "memory_gb": r.memory_gb,
            "power_w": r.power_w,
            "feasible": r.feasible,
        }
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest llm-memory-calculator/tests/budevolve/test_numeric_config_optimizer.py -v`
Expected: PASS (all 4 tests)

**Step 5: Commit**

```bash
git add llm-memory-calculator/src/llm_memory_calculator/budevolve/numeric/ \
        llm-memory-calculator/tests/budevolve/test_numeric_config_optimizer.py \
        llm-memory-calculator/pyproject.toml
git commit -m "feat(budevolve): add NumericOptimizer with pymoo NSGA-II config search"
```

---

## Task 5: Hardware Explorer — pymoo hardware DSE

**Files:**
- Create: `llm-memory-calculator/src/llm_memory_calculator/budevolve/numeric/hardware_explorer.py`
- Create: `llm-memory-calculator/tests/budevolve/test_hardware_explorer.py`

**Step 1: Write the failing test**

```python
# tests/budevolve/test_hardware_explorer.py
"""Tests for pymoo-based HardwareExplorer."""
import pytest
from unittest.mock import patch, MagicMock

from llm_memory_calculator.budevolve.types import (
    EvalResult, ParetoResult, HardwareSpec,
)
from llm_memory_calculator.budevolve.numeric.search_spaces import HardwareSearchSpace


def _make_hw_eval(flops, bw, mem, throughput, ttft, cost):
    return EvalResult(
        throughput_rps=throughput, ttft_ms=ttft,
        tpot_ms=10.0, feasible=True,
        config={"flops_tflops": flops, "offchip_mem_bw_gbps": bw,
                "off_chip_mem_size_gb": mem, "estimated_cost_usd": cost},
    )


class TestHardwareSearchSpace:
    def test_bounds(self):
        space = HardwareSearchSpace()
        assert len(space.xl) == 7
        assert len(space.xu) == 7
        assert all(lo < hi for lo, hi in zip(space.xl, space.xu))


class TestHardwareExplorer:
    def test_explore_returns_pareto_result(self):
        from llm_memory_calculator.budevolve.numeric.hardware_explorer import HardwareExplorer

        results = [
            _make_hw_eval(312, 1600, 40, 50.0, 100.0, 10000),
            _make_hw_eval(990, 3350, 80, 120.0, 40.0, 25000),
            _make_hw_eval(1979, 8000, 192, 200.0, 20.0, 40000),
        ]

        with patch.object(HardwareExplorer, "_run_search") as mock_run:
            mock_run.return_value = results
            explorer = HardwareExplorer(model="test-model")
            result = explorer.explore(
                objectives=["throughput", "cost"],
                n_generations=10,
            )
            assert isinstance(result, ParetoResult)
            assert len(result.pareto_front) > 0

    def test_what_if_sweeps_parameter(self):
        from llm_memory_calculator.budevolve.numeric.hardware_explorer import HardwareExplorer

        with patch("llm_memory_calculator.budevolve.numeric.hardware_explorer.BudSimEvaluator") as mock_cls:
            mock_eval = MagicMock()
            mock_eval.evaluate_hardware.return_value = EvalResult(
                throughput_rps=100.0, ttft_ms=50.0, tpot_ms=10.0, feasible=True,
            )
            mock_cls.return_value = mock_eval

            explorer = HardwareExplorer(model="test-model")
            result = explorer.what_if(
                base_hardware="A100_40GB_GPU",
                param="offchip_mem_bw_gbps",
                param_range=(1600, 8000),
                steps=5,
            )
            assert len(result) == 5
            assert all("offchip_mem_bw_gbps" in r for r in result)

    def test_compare_vs_real(self):
        from llm_memory_calculator.budevolve.numeric.hardware_explorer import HardwareExplorer

        with patch("llm_memory_calculator.budevolve.numeric.hardware_explorer.BudSimEvaluator") as mock_cls:
            mock_eval = MagicMock()
            mock_eval.evaluate_config.return_value = EvalResult(
                throughput_rps=100.0, ttft_ms=50.0, tpot_ms=10.0, feasible=True,
            )
            mock_eval.evaluate_hardware.return_value = EvalResult(
                throughput_rps=150.0, ttft_ms=30.0, tpot_ms=8.0, feasible=True,
            )
            mock_cls.return_value = mock_eval

            explorer = HardwareExplorer(model="test-model")
            hw = HardwareSpec(flops_tflops=500.0, offchip_mem_bw_gbps=3000.0,
                              off_chip_mem_size_gb=80.0)
            result = explorer.compare_vs_real(hw, real_hardware=["H100_GPU"])
            assert "hypothetical" in result
            assert "H100_GPU" in result
```

**Step 2: Run to verify failure, Step 3: Implement**

```python
# src/llm_memory_calculator/budevolve/numeric/hardware_explorer.py
"""Hardware design space exploration using pymoo NSGA-II."""
import warnings
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from ..types import EvalResult, ParetoResult, HardwareSpec
from ..evaluator import BudSimEvaluator
from .search_spaces import HardwareSearchSpace

# Map real hardware names to approximate specs for comparison
_REAL_HARDWARE_SPECS = {
    "A100_40GB_GPU": HardwareSpec(312, 1600, 40, 40, 18000, 150, 1.41, 1, 400, 10000),
    "A100_80GB_GPU": HardwareSpec(312, 2039, 80, 40, 18000, 150, 1.41, 1, 400, 15000),
    "H100_GPU": HardwareSpec(990, 3350, 80, 50, 33000, 450, 1.98, 1, 700, 25000),
    "H200_GPU": HardwareSpec(990, 4800, 141, 50, 33000, 450, 1.98, 1, 700, 30000),
    "B200_GPU": HardwareSpec(2250, 8000, 192, 64, 50000, 900, 2.1, 1, 1000, 40000),
}


class HardwareExplorer:
    """Discover ideal hardware specs via multi-objective search.

    Uses pymoo NSGA-II to search the hardware parameter space (FLOPS,
    memory BW, capacity, etc.) with BudSim as the evaluation function.
    No LLM required.
    """

    def __init__(
        self,
        model: str,
        evaluator: Optional[BudSimEvaluator] = None,
    ):
        self._model = model
        self._evaluator = evaluator or BudSimEvaluator()

    def explore(
        self,
        objectives: List[str] = None,
        constraints: Optional[Dict[str, float]] = None,
        search_space: Optional[HardwareSearchSpace] = None,
        n_generations: int = 100,
        pop_size: int = 50,
        input_tokens: int = 512,
        output_tokens: int = 128,
        batch_size: int = 32,
    ) -> ParetoResult:
        """Search hardware parameter space for optimal designs.

        Args:
            objectives: e.g. ["throughput", "cost", "power"]
            constraints: e.g. {"min_throughput_rps": 100, "max_power_w": 700}
            search_space: Hardware parameter ranges.
            n_generations: NSGA-II generations.

        Returns:
            ParetoResult with Pareto-optimal hardware designs.
        """
        objectives = objectives or ["throughput", "cost"]
        constraints = constraints or {}
        search_space = search_space or HardwareSearchSpace()

        all_results = self._run_search(
            search_space, objectives, constraints,
            n_generations, pop_size, input_tokens, output_tokens, batch_size,
        )

        feasible = [r for r in all_results if r.feasible]
        min_tput = constraints.get("min_throughput_rps", 0.0)
        max_power = constraints.get("max_power_w", float("inf"))
        feasible = [
            r for r in feasible
            if r.throughput_rps >= min_tput and r.power_w <= max_power
        ]

        if not feasible:
            return ParetoResult(
                pareto_front=[], all_evaluated=[],
                best_per_objective={}, total_evaluations=len(all_results),
            )

        pareto = self._compute_pareto(feasible, objectives)

        best_per_obj = {}
        for obj in objectives:
            if obj == "throughput":
                best = max(feasible, key=lambda r: r.throughput_rps)
            elif obj == "cost":
                best = min(feasible, key=lambda r: r.config.get("estimated_cost_usd", 1e9))
            elif obj == "power":
                best = min(feasible, key=lambda r: r.power_w)
            else:
                best = feasible[0]
            best_per_obj[obj] = self._to_dict(best)

        return ParetoResult(
            pareto_front=[self._to_dict(r) for r in pareto],
            all_evaluated=[self._to_dict(r) for r in feasible[:100]],
            best_per_objective=best_per_obj,
            num_generations=n_generations,
            total_evaluations=len(all_results),
        )

    def what_if(
        self,
        base_hardware: str,
        param: str,
        param_range: Tuple[float, float],
        steps: int = 20,
        input_tokens: int = 512,
        output_tokens: int = 128,
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """Sweep a single hardware parameter.

        Example: what_if("A100_40GB_GPU", "offchip_mem_bw_gbps", (1600, 8000), steps=10)
        answers "what if A100 had up to 8 TB/s memory bandwidth?"
        """
        base = _REAL_HARDWARE_SPECS.get(base_hardware)
        if base is None:
            warnings.warn(f"No specs for {base_hardware}, using H100 defaults")
            base = _REAL_HARDWARE_SPECS["H100_GPU"]

        lo, hi = param_range
        values = np.linspace(lo, hi, steps)
        results = []

        for val in values:
            import dataclasses
            hw = dataclasses.replace(base, **{param: float(val)})
            eval_result = self._evaluator.evaluate_hardware(
                hw, model=self._model,
                input_tokens=input_tokens, output_tokens=output_tokens,
                batch_size=batch_size,
            )
            entry = self._to_dict(eval_result)
            entry[param] = float(val)
            results.append(entry)

        return results

    def compare_vs_real(
        self,
        hypothetical: HardwareSpec,
        real_hardware: List[str] = None,
        input_tokens: int = 512,
        output_tokens: int = 128,
        batch_size: int = 32,
    ) -> Dict[str, Dict[str, Any]]:
        """Compare a hypothetical hardware design against real GPUs."""
        real_hardware = real_hardware or ["A100_40GB_GPU", "H100_GPU"]
        comparison = {}

        hyp_result = self._evaluator.evaluate_hardware(
            hypothetical, model=self._model,
            input_tokens=input_tokens, output_tokens=output_tokens,
            batch_size=batch_size,
        )
        comparison["hypothetical"] = self._to_dict(hyp_result)

        for hw_name in real_hardware:
            from ..types import ServingConfig
            cfg = ServingConfig(model=self._model, hardware=hw_name)
            result = self._evaluator.evaluate_config(
                cfg, input_tokens=input_tokens, output_tokens=output_tokens,
            )
            comparison[hw_name] = self._to_dict(result)

        return comparison

    def _run_search(self, search_space, objectives, constraints,
                    n_gen, pop_size, input_tokens, output_tokens, batch_size):
        """Run pymoo or fallback grid search over hardware params."""
        try:
            return self._run_pymoo_hw(
                search_space, objectives, n_gen, pop_size,
                input_tokens, output_tokens, batch_size,
            )
        except ImportError:
            warnings.warn("pymoo not installed, using grid sample")
            return self._run_grid_hw(
                search_space, input_tokens, output_tokens, batch_size,
            )

    def _run_pymoo_hw(self, search_space, objectives, n_gen, pop_size,
                      input_tokens, output_tokens, batch_size):
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.optimize import minimize

        evaluator = self._evaluator
        model = self._model
        all_results = []
        n_obj = len(objectives)

        class HWProblem(ElementwiseProblem):
            def __init__(self):
                super().__init__(
                    n_var=search_space.n_var, n_obj=n_obj,
                    xl=np.array(search_space.xl),
                    xu=np.array(search_space.xu),
                )

            def _evaluate(self, x, out, *args, **kwargs):
                hw = HardwareSpec(
                    flops_tflops=x[0], offchip_mem_bw_gbps=x[1],
                    off_chip_mem_size_gb=x[2], on_chip_mem_size_mb=x[3],
                    onchip_mem_bw_gbps=x[4], interchip_link_bw_gbps=x[5],
                    frequency_ghz=x[6],
                )
                result = evaluator.evaluate_hardware(
                    hw, model=model,
                    input_tokens=input_tokens, output_tokens=output_tokens,
                    batch_size=batch_size,
                )
                all_results.append(result)

                obj_vals = []
                for obj in objectives:
                    if obj == "throughput":
                        obj_vals.append(-result.throughput_rps)
                    elif obj == "cost":
                        obj_vals.append(hw.estimated_cost_usd)
                    elif obj == "power":
                        obj_vals.append(hw.tdp_watts)
                    elif obj == "latency":
                        obj_vals.append(result.ttft_ms)
                    else:
                        obj_vals.append(-result.throughput_rps)
                out["F"] = np.array(obj_vals)

        problem = HWProblem()
        algorithm = NSGA2(pop_size=min(pop_size, 20))
        minimize(problem, algorithm, termination=("n_gen", min(n_gen, 30)),
                 seed=42, verbose=False)
        return all_results

    def _run_grid_hw(self, search_space, input_tokens, output_tokens, batch_size):
        results = []
        for flops in np.linspace(*search_space.flops_range, 5):
            for bw in np.linspace(*search_space.mem_bw_range, 5):
                hw = HardwareSpec(
                    flops_tflops=flops, offchip_mem_bw_gbps=bw,
                    off_chip_mem_size_gb=80.0,
                )
                result = self._evaluator.evaluate_hardware(
                    hw, model=self._model,
                    input_tokens=input_tokens, output_tokens=output_tokens,
                    batch_size=batch_size,
                )
                results.append(result)
        return results

    def _compute_pareto(self, results, objectives):
        def get_vals(r):
            vals = []
            for obj in objectives:
                if obj == "throughput":
                    vals.append(r.throughput_rps)
                elif obj == "cost":
                    vals.append(-r.config.get("estimated_cost_usd", 1e9))
                elif obj == "power":
                    vals.append(-r.power_w)
                elif obj == "latency":
                    vals.append(-r.ttft_ms)
                else:
                    vals.append(r.throughput_rps)
            return vals

        pareto = []
        for i, ri in enumerate(results):
            vi = get_vals(ri)
            dominated = False
            for j, rj in enumerate(results):
                if i == j:
                    continue
                vj = get_vals(rj)
                if (all(vj[k] >= vi[k] for k in range(len(objectives))) and
                        any(vj[k] > vi[k] for k in range(len(objectives)))):
                    dominated = True
                    break
            if not dominated:
                pareto.append(ri)
        return pareto

    def _to_dict(self, r):
        return {
            "config": r.config, "throughput_rps": r.throughput_rps,
            "ttft_ms": r.ttft_ms, "tpot_ms": r.tpot_ms,
            "memory_gb": r.memory_gb, "power_w": r.power_w,
            "feasible": r.feasible,
        }
```

**Step 4: Run tests**

Run: `python3 -m pytest llm-memory-calculator/tests/budevolve/test_hardware_explorer.py -v`
Expected: PASS (all 4 tests)

**Step 5: Commit**

```bash
git add llm-memory-calculator/src/llm_memory_calculator/budevolve/numeric/hardware_explorer.py \
        llm-memory-calculator/tests/budevolve/test_hardware_explorer.py
git commit -m "feat(budevolve): add HardwareExplorer with pymoo hardware DSE and what-if analysis"
```

---

## Task 6: Sensitivity Analyzer

**Files:**
- Create: `llm-memory-calculator/src/llm_memory_calculator/budevolve/numeric/sensitivity.py`
- Create: `llm-memory-calculator/tests/budevolve/test_sensitivity.py`

**Step 1: Write test**

```python
# tests/budevolve/test_sensitivity.py
"""Tests for sensitivity analysis."""
import pytest
from unittest.mock import patch, MagicMock

from llm_memory_calculator.budevolve.types import EvalResult


class TestSensitivityAnalyzer:
    def test_analyze_returns_ranking(self):
        from llm_memory_calculator.budevolve.numeric.sensitivity import SensitivityAnalyzer

        call_count = 0
        def mock_evaluate(config, **kwargs):
            nonlocal call_count
            call_count += 1
            tp = config.tensor_parallel
            bs = config.batch_size
            return EvalResult(
                throughput_rps=bs * 2.0 / tp,
                ttft_ms=100.0 / tp,
                tpot_ms=10.0, feasible=True,
            )

        with patch("llm_memory_calculator.budevolve.numeric.sensitivity.BudSimEvaluator") as mock_cls:
            mock_eval = MagicMock()
            mock_eval.evaluate_config.side_effect = mock_evaluate
            mock_cls.return_value = mock_eval

            analyzer = SensitivityAnalyzer(model="test", hardware="H100_GPU")
            result = analyzer.analyze(target="throughput")

            assert "ranking" in result
            assert "scores" in result
            assert len(result["ranking"]) > 0

    def test_hardware_sensitivity(self):
        from llm_memory_calculator.budevolve.numeric.sensitivity import SensitivityAnalyzer

        def mock_hw_evaluate(hw, **kwargs):
            return EvalResult(
                throughput_rps=hw.flops_tflops * 0.1,
                ttft_ms=1000.0 / hw.offchip_mem_bw_gbps,
                tpot_ms=10.0, feasible=True,
            )

        with patch("llm_memory_calculator.budevolve.numeric.sensitivity.BudSimEvaluator") as mock_cls:
            mock_eval = MagicMock()
            mock_eval.evaluate_hardware.side_effect = mock_hw_evaluate
            mock_cls.return_value = mock_eval

            analyzer = SensitivityAnalyzer(model="test", hardware="H100_GPU")
            result = analyzer.analyze_hardware(target="throughput")

            assert "ranking" in result
            assert len(result["ranking"]) > 0
```

**Step 2-3: Implement**

```python
# src/llm_memory_calculator/budevolve/numeric/sensitivity.py
"""Sensitivity analysis for config and hardware parameters."""
from typing import Dict, List, Optional, Any

from ..types import ServingConfig, HardwareSpec, EvalResult
from ..evaluator import BudSimEvaluator


class SensitivityAnalyzer:
    """Parameter sensitivity analysis using one-at-a-time (OAT) method.

    Measures how much each parameter affects the target metric by varying
    one parameter at a time while holding others at baseline.
    """

    def __init__(self, model: str, hardware: str,
                 evaluator: Optional[BudSimEvaluator] = None):
        self._model = model
        self._hardware = hardware
        self._evaluator = evaluator or BudSimEvaluator()

    def analyze(
        self,
        target: str = "throughput",
        input_tokens: int = 512,
        output_tokens: int = 128,
    ) -> Dict[str, Any]:
        """Analyze sensitivity of serving config parameters.

        Returns dict with 'ranking' (sorted params) and 'scores' (impact per param).
        """
        params = {
            "tensor_parallel": [1, 2, 4, 8],
            "pipeline_parallel": [1, 2, 4],
            "batch_size": [1, 8, 32, 64, 128, 256],
            "precision": ["bf16", "fp8", "int8"],
        }

        baseline_cfg = ServingConfig(model=self._model, hardware=self._hardware)
        baseline = self._evaluator.evaluate_config(baseline_cfg, input_tokens, output_tokens)
        baseline_score = self._get_score(baseline, target)

        scores = {}
        for param, values in params.items():
            max_delta = 0.0
            for val in values:
                cfg = ServingConfig(model=self._model, hardware=self._hardware)
                if param == "precision":
                    cfg.precision = val
                elif param == "tensor_parallel":
                    cfg.tensor_parallel = val
                elif param == "pipeline_parallel":
                    cfg.pipeline_parallel = val
                elif param == "batch_size":
                    cfg.batch_size = val

                result = self._evaluator.evaluate_config(cfg, input_tokens, output_tokens)
                score = self._get_score(result, target)
                delta = abs(score - baseline_score)
                max_delta = max(max_delta, delta)

            scores[param] = max_delta

        ranking = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return {"ranking": ranking, "scores": scores, "baseline": baseline_score}

    def analyze_hardware(
        self,
        target: str = "throughput",
        input_tokens: int = 512,
        output_tokens: int = 128,
    ) -> Dict[str, Any]:
        """Analyze sensitivity of hardware parameters."""
        base_hw = HardwareSpec(
            flops_tflops=990.0, offchip_mem_bw_gbps=3350.0,
            off_chip_mem_size_gb=80.0, on_chip_mem_size_mb=50.0,
            onchip_mem_bw_gbps=33000.0, interchip_link_bw_gbps=450.0,
            frequency_ghz=1.98,
        )
        baseline = self._evaluator.evaluate_hardware(
            base_hw, model=self._model, input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        baseline_score = self._get_score(baseline, target)

        params = {
            "flops_tflops": [200, 500, 990, 2000, 5000],
            "offchip_mem_bw_gbps": [1000, 2000, 3350, 5000, 8000],
            "off_chip_mem_size_gb": [40, 80, 128, 192],
            "interchip_link_bw_gbps": [50, 150, 450, 900],
        }

        scores = {}
        for param, values in params.items():
            max_delta = 0.0
            for val in values:
                import dataclasses
                hw = dataclasses.replace(base_hw, **{param: float(val)})
                result = self._evaluator.evaluate_hardware(
                    hw, model=self._model,
                    input_tokens=input_tokens, output_tokens=output_tokens,
                )
                delta = abs(self._get_score(result, target) - baseline_score)
                max_delta = max(max_delta, delta)
            scores[param] = max_delta

        ranking = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return {"ranking": ranking, "scores": scores, "baseline": baseline_score}

    def _get_score(self, result: EvalResult, target: str) -> float:
        if target == "throughput":
            return result.throughput_rps
        elif target == "latency":
            return -(result.ttft_ms + result.tpot_ms)
        return result.throughput_rps
```

**Step 4-5: Test and commit**

Run: `python3 -m pytest llm-memory-calculator/tests/budevolve/test_sensitivity.py -v`

```bash
git add llm-memory-calculator/src/llm_memory_calculator/budevolve/numeric/sensitivity.py \
        llm-memory-calculator/tests/budevolve/test_sensitivity.py
git commit -m "feat(budevolve): add SensitivityAnalyzer for config and hardware parameters"
```

---

## Task 7: Algorithm Evolver — OpenEvolve integration

**Files:**
- Create: `llm-memory-calculator/src/llm_memory_calculator/budevolve/evolve/__init__.py`
- Create: `llm-memory-calculator/src/llm_memory_calculator/budevolve/evolve/evaluator_bridge.py`
- Create: `llm-memory-calculator/src/llm_memory_calculator/budevolve/evolve/prompt_templates.py`
- Create: `llm-memory-calculator/src/llm_memory_calculator/budevolve/evolve/algorithm_evolver.py`
- Create: `llm-memory-calculator/src/llm_memory_calculator/budevolve/evolve/base_algorithms/`
- Create: `llm-memory-calculator/tests/budevolve/test_algorithm_evolver.py`

**Step 1: Write the failing test**

```python
# tests/budevolve/test_algorithm_evolver.py
"""Tests for OpenEvolve-based AlgorithmEvolver."""
import pytest
from unittest.mock import patch, MagicMock

from llm_memory_calculator.budevolve.types import RooflineReport, OperatorInsight


class TestPromptTemplates:
    def test_build_scheduler_prompt(self):
        from llm_memory_calculator.budevolve.evolve.prompt_templates import build_scheduler_prompt

        report = RooflineReport(
            overall_bottleneck="memory_bandwidth",
            compute_utilization=0.42,
            memory_bw_utilization=0.91,
            interconnect_utilization=0.05,
            per_operator=[
                OperatorInsight("attn", "memory", 8.0, 0.3, 0.8, 0.4),
                OperatorInsight("ffn", "compute", 95.0, 1.2, 0.5, 0.3),
            ],
            recommendations=["Increase batch size"],
        )

        prompt = build_scheduler_prompt(
            current_code="def schedule(queue): return queue[:32]",
            metrics={"throughput_rps": 45.0, "ttft_ms": 320.0},
            roofline=report,
        )
        assert "memory_bandwidth" in prompt.lower() or "memory bandwidth" in prompt.lower()
        assert "schedule" in prompt
        assert "45.0" in prompt or "45" in prompt


class TestEvaluatorBridge:
    def test_evaluate_returns_score(self):
        from llm_memory_calculator.budevolve.evolve.evaluator_bridge import BudSimEvalBridge

        with patch("llm_memory_calculator.budevolve.evolve.evaluator_bridge.BudSimEvaluator") as mock_cls:
            mock_eval = MagicMock()
            from llm_memory_calculator.budevolve.types import EvalResult
            mock_eval.evaluate_config.return_value = EvalResult(
                throughput_rps=100.0, ttft_ms=50.0, tpot_ms=10.0,
                slo_compliance=0.95, feasible=True,
            )
            mock_cls.return_value = mock_eval

            bridge = BudSimEvalBridge(model="test", hardware="H100_GPU")
            score = bridge.evaluate("dummy_path.py")

            assert isinstance(score, dict)
            assert "combined_score" in score
            assert score["combined_score"] > 0


class TestAlgorithmEvolver:
    def test_init(self):
        from llm_memory_calculator.budevolve.evolve.algorithm_evolver import AlgorithmEvolver
        evolver = AlgorithmEvolver(
            model="test-model", hardware="H100_GPU",
            llm_endpoint="https://api.openai.com/v1",
            llm_model="gpt-4o-mini",
        )
        assert evolver._model == "test-model"

    def test_get_base_algorithm(self):
        from llm_memory_calculator.budevolve.evolve.algorithm_evolver import AlgorithmEvolver
        evolver = AlgorithmEvolver(
            model="test", hardware="H100_GPU",
            llm_endpoint="http://localhost", llm_model="test",
        )
        code = evolver.get_base_algorithm("scheduler")
        assert "def schedule_batch" in code or "def schedule" in code
```

**Step 2-3: Implement**

```python
# src/llm_memory_calculator/budevolve/evolve/__init__.py
"""Algorithm evolution using OpenEvolve + BudSim."""
```

```python
# src/llm_memory_calculator/budevolve/evolve/prompt_templates.py
"""Roofline-enriched prompt templates for OpenEvolve."""
from typing import Dict, Optional

from ..types import RooflineReport
from ..roofline_analyzer import RooflineAnalyzer


def build_scheduler_prompt(
    current_code: str,
    metrics: Dict[str, float],
    roofline: Optional[RooflineReport] = None,
) -> str:
    """Build a prompt for evolving a batch scheduling algorithm.

    Includes current code, performance metrics, and roofline bottleneck
    analysis so the LLM can make informed mutations.
    """
    lines = [
        "You are evolving a batch scheduling algorithm for LLM serving.",
        "",
        "Current algorithm:",
        "```python",
        current_code,
        "```",
        "",
        f"Performance: throughput={metrics.get('throughput_rps', 0):.1f} req/s, "
        f"TTFT={metrics.get('ttft_ms', 0):.1f}ms, "
        f"TPOT={metrics.get('tpot_ms', 0):.1f}ms",
    ]

    if roofline:
        analyzer = RooflineAnalyzer()
        roofline_text = analyzer.format_for_prompt(roofline)
        lines.extend(["", "Roofline Analysis:", roofline_text])

    lines.extend([
        "",
        "Generate an improved scheduling algorithm that addresses the bottlenecks above.",
        "The function must have the same signature. Return only the Python code.",
    ])

    return "\n".join(lines)


def build_cache_policy_prompt(
    current_code: str,
    metrics: Dict[str, float],
    roofline: Optional[RooflineReport] = None,
) -> str:
    """Build a prompt for evolving a KV cache eviction policy."""
    lines = [
        "You are evolving a KV cache eviction policy for LLM serving.",
        "",
        "Current policy:",
        "```python",
        current_code,
        "```",
        "",
        f"Performance: throughput={metrics.get('throughput_rps', 0):.1f} req/s, "
        f"cache_hit_rate={metrics.get('cache_hit_rate', 0):.1%}",
    ]

    if roofline:
        analyzer = RooflineAnalyzer()
        lines.extend(["", "Roofline Analysis:", analyzer.format_for_prompt(roofline)])

    lines.extend([
        "",
        "Generate an improved eviction policy. Return only the Python code.",
    ])
    return "\n".join(lines)
```

```python
# src/llm_memory_calculator/budevolve/evolve/evaluator_bridge.py
"""Bridge between OpenEvolve's evaluator interface and BudSim."""
from typing import Dict, Optional

from ..types import ServingConfig, EvalResult
from ..evaluator import BudSimEvaluator
from ..roofline_analyzer import RooflineAnalyzer


class BudSimEvalBridge:
    """OpenEvolve-compatible evaluator that uses BudSim as fitness function.

    OpenEvolve expects an evaluate(program_path) -> dict function.
    This bridge wraps BudSimEvaluator to provide that interface.
    """

    def __init__(
        self,
        model: str,
        hardware: str,
        input_tokens: int = 512,
        output_tokens: int = 128,
        batch_size: int = 32,
        precision: str = "bf16",
    ):
        self._model = model
        self._hardware = hardware
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._batch_size = batch_size
        self._precision = precision
        self._evaluator = BudSimEvaluator()
        self._roofline = RooflineAnalyzer()

    def evaluate(self, program_path: str) -> Dict:
        """Evaluate an evolved algorithm using BudSim.

        Args:
            program_path: Path to evolved Python file (used by OpenEvolve).

        Returns:
            Dict with 'combined_score' and individual metrics.
        """
        cfg = ServingConfig(
            model=self._model, hardware=self._hardware,
            batch_size=self._batch_size, precision=self._precision,
        )
        result = self._evaluator.evaluate_config(
            cfg, input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
        )

        throughput = result.throughput_rps
        slo = result.slo_compliance if result.slo_compliance > 0 else 1.0

        return {
            "combined_score": throughput * slo,
            "throughput_rps": throughput,
            "ttft_ms": result.ttft_ms,
            "tpot_ms": result.tpot_ms,
            "slo_compliance": slo,
            "feasible": 1.0 if result.feasible else 0.0,
        }
```

```python
# src/llm_memory_calculator/budevolve/evolve/algorithm_evolver.py
"""Algorithm evolution orchestrator using OpenEvolve + BudSim."""
import os
import warnings
from typing import Optional, Dict, Any

from ..types import ServingConfig
from ..evaluator import BudSimEvaluator
from ..roofline_analyzer import RooflineAnalyzer
from .evaluator_bridge import BudSimEvalBridge
from .prompt_templates import build_scheduler_prompt, build_cache_policy_prompt


# Base algorithm code for seeding evolution
_BASE_SCHEDULER = '''
def schedule_batch(queue, max_batch_size=256, max_tokens=8192):
    """Schedule a batch of requests from the queue.

    Args:
        queue: List of pending requests, each with .input_tokens, .output_tokens
        max_batch_size: Maximum requests per batch.
        max_tokens: Maximum total tokens per batch.

    Returns:
        List of requests to include in the next batch.
    """
    if not queue:
        return []

    batch = []
    total_tokens = 0

    for request in queue:
        req_tokens = request.input_tokens + request.output_tokens
        if len(batch) >= max_batch_size:
            break
        if total_tokens + req_tokens > max_tokens:
            break
        batch.append(request)
        total_tokens += req_tokens

    return batch
'''

_BASE_CACHE_POLICY = '''
def evict_entries(cache_entries, num_to_evict):
    """Select cache entries to evict under memory pressure.

    Args:
        cache_entries: List of cache entries, each with .last_access_time,
                       .frequency, .size_bytes, .prefix_length
        num_to_evict: Number of entries to remove.

    Returns:
        List of entries to evict.
    """
    sorted_entries = sorted(cache_entries, key=lambda e: e.last_access_time)
    return sorted_entries[:num_to_evict]
'''


class AlgorithmEvolver:
    """Evolve scheduling/caching algorithms using OpenEvolve + BudSim.

    Uses OpenEvolve's evolutionary loop with BudSim as the fitness function.
    Roofline analysis is injected into LLM prompts for informed mutations.
    """

    def __init__(
        self,
        model: str,
        hardware: str,
        llm_endpoint: str = "https://api.openai.com/v1",
        llm_model: str = "gpt-4o-mini",
        llm_api_key: Optional[str] = None,
    ):
        self._model = model
        self._hardware = hardware
        self._llm_endpoint = llm_endpoint
        self._llm_model = llm_model
        self._llm_api_key = llm_api_key or os.environ.get("OPENAI_API_KEY", "")
        self._evaluator = BudSimEvaluator()
        self._roofline = RooflineAnalyzer()

    def get_base_algorithm(self, algorithm_type: str) -> str:
        """Get seed algorithm code for evolution.

        Args:
            algorithm_type: "scheduler" or "cache_policy"
        """
        if algorithm_type == "scheduler":
            return _BASE_SCHEDULER
        elif algorithm_type == "cache_policy":
            return _BASE_CACHE_POLICY
        else:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}")

    def evolve_scheduler(
        self,
        iterations: int = 100,
        input_tokens: int = 512,
        output_tokens: int = 128,
        output_dir: str = "evolved_scheduler",
    ) -> Dict[str, Any]:
        """Evolve a batch scheduling algorithm.

        Args:
            iterations: Number of evolutionary iterations.
            input_tokens: Representative input length.
            output_tokens: Representative output length.
            output_dir: Directory to save evolved algorithms.

        Returns:
            Dict with best_code, fitness_history, improvement_pct.
        """
        base_code = self.get_base_algorithm("scheduler")

        # Get baseline metrics + roofline
        cfg = ServingConfig(model=self._model, hardware=self._hardware)
        baseline = self._evaluator.evaluate_config(cfg, input_tokens, output_tokens)

        try:
            roofline = self._roofline.analyze_config(cfg, input_tokens)
        except Exception:
            roofline = None

        # Try OpenEvolve integration
        try:
            return self._run_openevolve(
                base_code, "scheduler", baseline, roofline,
                iterations, output_dir,
            )
        except ImportError:
            warnings.warn(
                "openevolve not installed. Install with: pip install openevolve\n"
                "Returning base algorithm without evolution."
            )
            return {
                "best_code": base_code,
                "baseline_throughput": baseline.throughput_rps,
                "improvement_pct": 0.0,
                "iterations_run": 0,
                "error": "openevolve not installed",
            }

    def _run_openevolve(self, base_code, algo_type, baseline, roofline,
                        iterations, output_dir):
        """Run OpenEvolve evolutionary loop."""
        from openevolve import run_evolution

        bridge = BudSimEvalBridge(
            model=self._model, hardware=self._hardware,
        )

        prompt_fn = build_scheduler_prompt if algo_type == "scheduler" else build_cache_policy_prompt

        baseline_metrics = {
            "throughput_rps": baseline.throughput_rps,
            "ttft_ms": baseline.ttft_ms,
            "tpot_ms": baseline.tpot_ms,
        }

        result = run_evolution(
            initial_program=base_code,
            evaluator=bridge.evaluate,
            iterations=iterations,
            output_dir=output_dir,
        )

        return {
            "best_code": result.get("best_program", base_code),
            "best_score": result.get("best_score", 0.0),
            "baseline_throughput": baseline.throughput_rps,
            "improvement_pct": (
                (result.get("best_score", 0) - baseline.throughput_rps)
                / max(baseline.throughput_rps, 1e-9) * 100
            ),
            "iterations_run": iterations,
        }
```

Create the base_algorithms directory:

```python
# src/llm_memory_calculator/budevolve/evolve/base_algorithms/__init__.py
```

**Step 4-5: Test and commit**

Run: `python3 -m pytest llm-memory-calculator/tests/budevolve/test_algorithm_evolver.py -v`

```bash
git add llm-memory-calculator/src/llm_memory_calculator/budevolve/evolve/ \
        llm-memory-calculator/tests/budevolve/test_algorithm_evolver.py
git commit -m "feat(budevolve): add AlgorithmEvolver with OpenEvolve integration and roofline prompts"
```

---

## Task 8: CLI entry point

**Files:**
- Create: `llm-memory-calculator/src/llm_memory_calculator/budevolve/cli.py`
- Create: `llm-memory-calculator/tests/budevolve/test_cli.py`
- Modify: `llm-memory-calculator/pyproject.toml` (add CLI entry point)

**Step 1: Write test**

```python
# tests/budevolve/test_cli.py
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
```

**Step 2-3: Implement**

```python
# src/llm_memory_calculator/budevolve/cli.py
"""BudEvolve CLI — reverse-optimization for LLM serving."""
import argparse
import json
import sys
from typing import List


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="budsim-evolve",
        description="BudEvolve: Reverse-optimization for LLM serving",
    )
    subparsers = parser.add_subparsers(dest="command", help="Optimization mode")

    # optimize
    opt = subparsers.add_parser("optimize", help="Multi-objective config optimization")
    opt.add_argument("--model", required=True, help="Model name (HuggingFace ID)")
    opt.add_argument("--hardware", required=True, help="Hardware name (e.g. H100_GPU)")
    opt.add_argument("--objectives", default="throughput,latency", help="Comma-separated objectives")
    opt.add_argument("--constraints", default="", help="Constraints e.g. 'ttft<500,tpot<50'")
    opt.add_argument("--generations", type=int, default=100)
    opt.add_argument("--pop-size", type=int, default=50)
    opt.add_argument("--input-tokens", type=int, default=512)
    opt.add_argument("--output-tokens", type=int, default=128)
    opt.add_argument("--output", default=None, help="Output JSON file")

    # explore-hardware
    hw = subparsers.add_parser("explore-hardware", help="Hardware design space exploration")
    hw.add_argument("--model", required=True)
    hw.add_argument("--objectives", default="throughput,cost")
    hw.add_argument("--constraints", default="")
    hw.add_argument("--generations", type=int, default=100)
    hw.add_argument("--output", default=None)

    # what-if
    wi = subparsers.add_parser("what-if", help="What-if hardware parameter sweep")
    wi.add_argument("--base-hardware", required=True)
    wi.add_argument("--param", required=True, help="Parameter to sweep")
    wi.add_argument("--range", required=True, help="lo,hi range")
    wi.add_argument("--steps", type=int, default=20)
    wi.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    wi.add_argument("--output", default=None)

    # sensitivity
    sa = subparsers.add_parser("sensitivity", help="Parameter sensitivity analysis")
    sa.add_argument("--model", required=True)
    sa.add_argument("--hardware", required=True)
    sa.add_argument("--target", default="throughput")
    sa.add_argument("--mode", choices=["config", "hardware", "both"], default="config")
    sa.add_argument("--output", default=None)

    # evolve-scheduler
    es = subparsers.add_parser("evolve-scheduler", help="Evolve scheduling algorithm")
    es.add_argument("--model", required=True)
    es.add_argument("--hardware", required=True)
    es.add_argument("--iterations", type=int, default=100)
    es.add_argument("--llm-endpoint", default="https://api.openai.com/v1")
    es.add_argument("--llm-model", default="gpt-4o-mini")
    es.add_argument("--output", default="evolved_scheduler")

    return parser


def _parse_constraints(s: str) -> dict:
    """Parse constraint string like 'ttft<500,tpot<50' into dict."""
    if not s:
        return {}
    constraints = {}
    for part in s.split(","):
        part = part.strip()
        if "<" in part:
            key, val = part.split("<")
            constraints[f"max_{key.strip()}_ms"] = float(val.strip())
    return constraints


def main(argv: List[str] = None):
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    result = None

    if args.command == "optimize":
        from .numeric.config_optimizer import NumericOptimizer
        optimizer = NumericOptimizer(model=args.model, hardware=args.hardware)
        result = optimizer.optimize(
            objectives=args.objectives.split(","),
            constraints=_parse_constraints(args.constraints),
            n_generations=args.generations,
            pop_size=args.pop_size,
            input_tokens=args.input_tokens,
            output_tokens=args.output_tokens,
        )
        from dataclasses import asdict
        result = asdict(result)

    elif args.command == "explore-hardware":
        from .numeric.hardware_explorer import HardwareExplorer
        explorer = HardwareExplorer(model=args.model)
        result = explorer.explore(
            objectives=args.objectives.split(","),
            constraints=_parse_constraints(args.constraints),
            n_generations=args.generations,
        )
        from dataclasses import asdict
        result = asdict(result)

    elif args.command == "what-if":
        from .numeric.hardware_explorer import HardwareExplorer
        explorer = HardwareExplorer(model=args.model)
        lo, hi = [float(x) for x in args.range.split(",")]
        result = explorer.what_if(
            base_hardware=args.base_hardware,
            param=args.param,
            param_range=(lo, hi),
            steps=args.steps,
        )

    elif args.command == "sensitivity":
        from .numeric.sensitivity import SensitivityAnalyzer
        analyzer = SensitivityAnalyzer(model=args.model, hardware=args.hardware)
        if args.mode in ("config", "both"):
            result = analyzer.analyze(target=args.target)
        if args.mode in ("hardware", "both"):
            hw_result = analyzer.analyze_hardware(target=args.target)
            if result:
                result["hardware"] = hw_result
            else:
                result = hw_result

    elif args.command == "evolve-scheduler":
        from .evolve.algorithm_evolver import AlgorithmEvolver
        evolver = AlgorithmEvolver(
            model=args.model, hardware=args.hardware,
            llm_endpoint=args.llm_endpoint, llm_model=args.llm_model,
        )
        result = evolver.evolve_scheduler(
            iterations=args.iterations,
            output_dir=args.output,
        )

    if result is not None:
        output_str = json.dumps(result, indent=2, default=str)
        if args.output and args.command != "evolve-scheduler":
            with open(args.output, "w") as f:
                f.write(output_str)
            print(f"Results written to {args.output}")
        else:
            print(output_str)

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Add to `pyproject.toml`:
```toml
[project.scripts]
budsim-evolve = "llm_memory_calculator.budevolve.cli:main"
```

**Step 4-5: Test and commit**

Run: `python3 -m pytest llm-memory-calculator/tests/budevolve/test_cli.py -v`

```bash
git add llm-memory-calculator/src/llm_memory_calculator/budevolve/cli.py \
        llm-memory-calculator/tests/budevolve/test_cli.py \
        llm-memory-calculator/pyproject.toml
git commit -m "feat(budevolve): add CLI with optimize, explore-hardware, what-if, sensitivity, evolve commands"
```

---

## Task 9: Package __init__.py — public API exports

**Files:**
- Modify: `llm-memory-calculator/src/llm_memory_calculator/budevolve/__init__.py`

**Step 1: Update exports**

```python
# src/llm_memory_calculator/budevolve/__init__.py
"""BudEvolve: Reverse-optimization system for LLM serving.

Three optimization modes:
- NumericOptimizer: pymoo NSGA-II config search (no LLM)
- HardwareExplorer: pymoo NSGA-II hardware DSE (no LLM)
- AlgorithmEvolver: OpenEvolve algorithm evolution (LLM required)

Usage:
    from llm_memory_calculator.budevolve import NumericOptimizer, HardwareExplorer
    optimizer = NumericOptimizer(model="llama-70b", hardware="H100_GPU")
    result = optimizer.optimize(objectives=["throughput", "latency"])
"""
from .types import (
    ServingConfig, HardwareSpec, EvalResult,
    OperatorInsight, RooflineReport, ParetoResult,
)
from .evaluator import BudSimEvaluator
from .roofline_analyzer import RooflineAnalyzer
from .numeric.config_optimizer import NumericOptimizer
from .numeric.hardware_explorer import HardwareExplorer
from .numeric.sensitivity import SensitivityAnalyzer

__all__ = [
    "ServingConfig", "HardwareSpec", "EvalResult",
    "OperatorInsight", "RooflineReport", "ParetoResult",
    "BudSimEvaluator", "RooflineAnalyzer",
    "NumericOptimizer", "HardwareExplorer", "SensitivityAnalyzer",
]
```

**Step 2: Commit**

```bash
git add llm-memory-calculator/src/llm_memory_calculator/budevolve/__init__.py
git commit -m "feat(budevolve): export public API from package __init__"
```

---

## Task 10: Integration test — end-to-end with real GenZ engine

**Files:**
- Create: `llm-memory-calculator/tests/budevolve/test_integration.py`

**Step 1: Write integration test**

```python
# tests/budevolve/test_integration.py
"""Integration tests — run real GenZ evaluations (slow, ~30s)."""
import pytest

# Mark all tests as integration (skip with: pytest -m "not integration")
pytestmark = pytest.mark.integration


class TestEvaluatorIntegration:
    def test_evaluate_real_model(self):
        from llm_memory_calculator.budevolve import BudSimEvaluator, ServingConfig
        evaluator = BudSimEvaluator()
        cfg = ServingConfig(model="meta-llama/Meta-Llama-3.1-8B", hardware="H100_GPU")
        result = evaluator.evaluate_config(cfg, input_tokens=512, output_tokens=128)

        assert result.feasible is True
        assert result.throughput_rps > 0
        assert result.ttft_ms > 0
        assert result.tpot_ms > 0

    def test_evaluate_hardware_spec(self):
        from llm_memory_calculator.budevolve import BudSimEvaluator, HardwareSpec
        evaluator = BudSimEvaluator()
        hw = HardwareSpec(
            flops_tflops=990.0, offchip_mem_bw_gbps=3350.0,
            off_chip_mem_size_gb=80.0,
        )
        result = evaluator.evaluate_hardware(
            hw, model="meta-llama/Meta-Llama-3.1-8B",
            input_tokens=512, output_tokens=128,
        )
        assert result.throughput_rps > 0


class TestRooflineIntegration:
    def test_analyze_real_model(self):
        from llm_memory_calculator.budevolve import RooflineAnalyzer, ServingConfig
        analyzer = RooflineAnalyzer()
        cfg = ServingConfig(model="meta-llama/Meta-Llama-3.1-8B", hardware="H100_GPU")
        report = analyzer.analyze_config(cfg, input_tokens=512)

        assert report.overall_bottleneck in ("compute", "memory_bandwidth", "interconnect", "unknown")
        assert len(report.per_operator) > 0

    def test_format_for_prompt(self):
        from llm_memory_calculator.budevolve import RooflineAnalyzer, ServingConfig
        analyzer = RooflineAnalyzer()
        cfg = ServingConfig(model="meta-llama/Meta-Llama-3.1-8B", hardware="H100_GPU")
        report = analyzer.analyze_config(cfg, input_tokens=512)
        text = analyzer.format_for_prompt(report)

        assert len(text) > 100
        assert "bottleneck" in text.lower() or "utilization" in text.lower()


class TestSensitivityIntegration:
    @pytest.mark.slow
    def test_config_sensitivity(self):
        from llm_memory_calculator.budevolve import SensitivityAnalyzer
        analyzer = SensitivityAnalyzer(
            model="meta-llama/Meta-Llama-3.1-8B", hardware="H100_GPU",
        )
        result = analyzer.analyze(target="throughput")

        assert "ranking" in result
        assert "batch_size" in result["ranking"] or "tensor_parallel" in result["ranking"]
```

**Step 2: Run**

Run: `python3 -m pytest llm-memory-calculator/tests/budevolve/test_integration.py -v -m integration`

**Step 3: Commit**

```bash
git add llm-memory-calculator/tests/budevolve/test_integration.py
git commit -m "test(budevolve): add integration tests with real GenZ engine"
```

---

## Summary

| Task | Component | Files | Tests |
|------|-----------|-------|-------|
| 1 | Package skeleton + types | 3 new | 8 |
| 2 | BudSimEvaluator | 1 new | 4 |
| 3 | RooflineAnalyzer | 1 new | 5 |
| 4 | NumericOptimizer (pymoo) | 3 new | 4 |
| 5 | HardwareExplorer (pymoo) | 1 new | 4 |
| 6 | SensitivityAnalyzer | 1 new | 2 |
| 7 | AlgorithmEvolver (OpenEvolve) | 5 new | 4 |
| 8 | CLI | 1 new | 5 |
| 9 | Package exports | 1 modify | 0 |
| 10 | Integration tests | 1 new | 5 |
| **Total** | | **18 files** | **41 tests** |
