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
        """Number of decision variables used in pymoo NSGA-II."""
        return 4  # tp, pp, batch, precision


@dataclass
class HardwareSearchSpace:
    """Search space for hardware design exploration.

    R2-BE3: tdp_range and cost_range make TDP and price SWEPT decision variables
    (hardware/cost co-design) rather than leaving them at the HardwareSpec dataclass
    defaults (700 W / $25000), which made the cost and power objectives CONSTANT across
    the search and produced a degenerate Pareto front. Bounds are the observed min/max of
    the in-repo _REAL_HARDWARE_SPECS anchor table (tdp: A100=400 .. B200=1000 W;
    cost: A100_40GB=10000 .. B200=40000 USD) -- not invented.
    """
    flops_range: tuple = (50.0, 10000.0)
    mem_bw_range: tuple = (200.0, 16000.0)
    mem_size_range: tuple = (16.0, 384.0)
    on_chip_mem_range: tuple = (10.0, 512.0)
    onchip_bw_range: tuple = (5000.0, 100000.0)
    interchip_bw_range: tuple = (25.0, 1800.0)
    frequency_range: tuple = (0.5, 3.0)
    tdp_range: tuple = (400.0, 1000.0)
    cost_range: tuple = (10000.0, 40000.0)

    @property
    def _ranges(self) -> list:
        """Ordered list of (lo, hi) bounds; defines the decision-variable layout."""
        return [
            self.flops_range, self.mem_bw_range, self.mem_size_range,
            self.on_chip_mem_range, self.onchip_bw_range,
            self.interchip_bw_range, self.frequency_range,
            self.tdp_range, self.cost_range,
        ]

    @property
    def n_var(self) -> int:
        return len(self._ranges)  # 9: 7 physical params + tdp + cost

    @property
    def xl(self) -> list:
        """Lower bounds."""
        return [r[0] for r in self._ranges]

    @property
    def xu(self) -> list:
        """Upper bounds."""
        return [r[1] for r in self._ranges]
