"""Roofline analyzer -- extracts per-operator bottleneck insights from GenZ.

Consumes the model_df DataFrame produced by GenZ's prefill_moddeling /
decode_moddeling and converts it into a structured RooflineReport with
per-operator bottleneck classification, utilization metrics, and
actionable recommendations suitable for LLM-driven optimization loops.
"""
import warnings
from typing import List, Optional

import pandas as pd

from llm_memory_calculator.genz.LLM_inference.llm_prefill import prefill_moddeling

from .types import ServingConfig, RooflineReport, OperatorInsight


class RooflineAnalyzer:
    """Extracts per-operator roofline analysis from GenZ model DataFrames.

    The GenZ engine produces a model_df DataFrame where each row represents
    one computational operator (FC, Logit, Attend, Sync, etc.) with columns
    for latency, compute/memory/communication time, utilization, and the
    bound classification (Compute / Memory / Collective).

    This analyzer:
      1. Iterates over each operator row and classifies its bottleneck.
      2. Computes each operator's fraction of total runtime.
      3. Aggregates compute/memory/communication time to determine the
         overall system bottleneck.
      4. Generates human-readable recommendations based on the profile.
    """

    def analyze_from_model_df(self, model_df: pd.DataFrame) -> RooflineReport:
        """Build a RooflineReport from a GenZ model_df DataFrame.

        Args:
            model_df: DataFrame with columns produced by GenZ operator_base.py
                get_roofline(). Expected columns include 'Layer Name',
                'Op Type', 'Bound', 'Op Intensity', 'Latency (ms)',
                'Compute time (ms)', 'Memory time (ms)',
                'Communication time (ms)', and utilization columns.

        Returns:
            RooflineReport with per-operator insights, overall bottleneck,
            utilization ratios, and recommendations.
        """
        latency_col = self._find_column(model_df, "Latency")
        compute_col = self._find_column(model_df, "Compute time")
        memory_col = self._find_column(model_df, "Memory time")
        comm_col = self._find_column(model_df, "Communication time")

        total_latency = model_df[latency_col].sum() if latency_col else 1.0
        if total_latency == 0:
            total_latency = 1.0

        operators: List[OperatorInsight] = []
        total_compute_time = 0.0
        total_memory_time = 0.0
        total_comm_time = 0.0

        for _, row in model_df.iterrows():
            name = row.get("Layer Name", "unknown")
            bound = str(row.get("Bound", "Memory")).lower()
            intensity = float(row.get("Op Intensity", 0.0))
            ct = float(row.get(compute_col, 0.0)) if compute_col else 0.0
            mt = float(row.get(memory_col, 0.0)) if memory_col else 0.0
            lat = float(row.get(latency_col, 0.0)) if latency_col else 0.0

            if bound == "compute":
                bottleneck = "compute"
            elif bound == "collective":
                bottleneck = "collective"
            else:
                bottleneck = "memory"

            operators.append(OperatorInsight(
                name=name,
                bottleneck=bottleneck,
                arithmetic_intensity=intensity,
                compute_time_ms=ct,
                memory_time_ms=mt,
                pct_of_total=lat / total_latency,
            ))
            total_compute_time += ct
            total_memory_time += mt
            if comm_col:
                total_comm_time += float(row.get(comm_col, 0.0))

        # Determine the overall system bottleneck by dominant time category
        max_time = max(total_compute_time, total_memory_time, total_comm_time, 1e-9)
        if total_compute_time == max_time:
            overall = "compute"
        elif total_memory_time == max_time:
            overall = "memory_bandwidth"
        else:
            overall = "interconnect"

        # Utilization as fraction of total latency spent in each category
        compute_util = total_compute_time / total_latency
        memory_util = total_memory_time / total_latency
        comm_util = total_comm_time / total_latency

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
        """Run GenZ prefill modeling and analyze the resulting model_df.

        This is a convenience method that calls prefill_moddeling with
        the given ServingConfig parameters, then passes the resulting
        model_df through analyze_from_model_df.

        Args:
            config: ServingConfig specifying model, hardware, parallelism, etc.
            input_tokens: Number of input tokens for prefill modeling.
            phase: Analysis phase (currently only "prefill" is supported).

        Returns:
            RooflineReport from the GenZ analysis. If GenZ raises an exception,
            returns a fallback report with overall_bottleneck="unknown".
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
        """Format a RooflineReport as text suitable for LLM prompts.

        Produces a compact multi-line summary with overall bottleneck,
        utilization percentages, top operators by runtime share, and
        recommendations. This text is designed to be injected into an
        LLM prompt for BudEvolve's optimization loops.

        Args:
            report: RooflineReport to format.

        Returns:
            Human-readable string summarizing the roofline analysis.
        """
        lines = [
            f"Overall bottleneck: {report.overall_bottleneck.upper()}",
            f"Compute utilization: {report.compute_utilization:.1%}",
            f"Memory BW utilization: {report.memory_bw_utilization:.1%}",
            f"Interconnect utilization: {report.interconnect_utilization:.1%}",
            "",
            "Top operators by runtime:",
        ]
        sorted_ops = sorted(
            report.per_operator,
            key=lambda o: o.pct_of_total,
            reverse=True,
        )
        for op in sorted_ops[:8]:
            lines.append(
                f"  {op.name}: {op.bottleneck}-bound, "
                f"{op.pct_of_total:.0%} of runtime, "
                f"intensity={op.arithmetic_intensity:.1f}"
            )
        if report.recommendations:
            lines.extend(["", "Recommendations:"])
            for rec in report.recommendations:
                lines.append(f"  - {rec}")
        return "\n".join(lines)

    def _find_column(self, df: pd.DataFrame, prefix: str) -> Optional[str]:
        """Find the first DataFrame column whose name starts with *prefix*.

        GenZ column names sometimes include units in parentheses
        (e.g. "Latency (ms)") so we match by prefix rather than exact name.

        Args:
            df: DataFrame to search.
            prefix: Column name prefix to match.

        Returns:
            Matching column name, or None if no column matches.
        """
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
        operators: List[OperatorInsight],
    ) -> List[str]:
        """Generate actionable optimization recommendations.

        Recommendations are based on the overall bottleneck classification,
        utilization ratios, and per-operator bottleneck distribution.

        Args:
            overall: Overall bottleneck category string.
            compute_util: Fraction of total latency from compute.
            memory_util: Fraction of total latency from memory access.
            comm_util: Fraction of total latency from communication.
            operators: List of per-operator insights.

        Returns:
            List of human-readable recommendation strings.
        """
        recs: List[str] = []

        if overall == "memory_bandwidth":
            recs.append(
                "System is memory-bandwidth bound. Consider higher bandwidth "
                "memory (HBM3e) or reducing data movement."
            )
            mem_ops = [
                o for o in operators
                if o.bottleneck == "memory" and o.pct_of_total > 0.1
            ]
            if mem_ops:
                names = ", ".join(o.name for o in mem_ops[:3])
                recs.append(f"Memory-bound hotspots: {names}")
        elif overall == "compute":
            recs.append(
                "System is compute-bound. Consider higher TFLOPS hardware "
                "or lower precision (FP8/INT8)."
            )
        elif overall == "interconnect":
            recs.append(
                "System is interconnect-bound. Consider higher NVLink "
                "bandwidth or reducing parallelism degree."
            )

        if compute_util < 0.5 and memory_util > 0.7:
            recs.append(
                "Low compute utilization suggests batching could improve "
                "throughput."
            )

        return recs
