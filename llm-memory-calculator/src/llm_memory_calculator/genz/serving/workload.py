"""Workload generation for LLM serving simulation."""
from dataclasses import dataclass, field
from typing import Dict, List, Any
import numpy as np

from .constants import NS_PER_S
from .request import Request


@dataclass
class WorkloadConfig:
    """Configuration for workload generation."""
    arrival_rate_rps: float = 10.0
    arrival_pattern: str = "poisson"  # poisson, gamma, bursty, constant
    num_requests: int = 300
    input_length_distribution: Dict[str, Any] = field(default_factory=lambda: {
        "dist": "lognormal", "mean": 512, "std": 200, "min": 64, "max": 4096,
    })
    output_length_distribution: Dict[str, Any] = field(default_factory=lambda: {
        "dist": "lognormal", "mean": 128, "std": 64, "min": 16, "max": 2048,
    })
    model: str = "default"
    random_seed: int = 42
    # For gamma pattern
    gamma_shape: float = 2.0
    # For bursty pattern
    burst_period_s: float = 10.0
    burst_amplitude: float = 3.0


class WorkloadGenerator:
    """Generate realistic LLM inference workloads."""

    def __init__(self, config: WorkloadConfig):
        self.config = config
        self._rng = np.random.default_rng(config.random_seed)

    def generate(self) -> List[Request]:
        """Generate workload as a sorted list of Request objects."""
        arrivals = self._generate_arrivals()
        input_lengths = self._sample_lengths(self.config.input_length_distribution)
        output_lengths = self._sample_lengths(self.config.output_length_distribution)

        requests = []
        for i in range(self.config.num_requests):
            requests.append(Request(
                request_id=i,
                model=self.config.model,
                input_tokens=int(input_lengths[i]),
                max_output_tokens=int(output_lengths[i]),
                arrival_time_ns=int(arrivals[i]),
            ))

        return sorted(requests, key=lambda r: r.arrival_time_ns)

    def _generate_arrivals(self) -> np.ndarray:
        """Generate arrival times in nanoseconds based on the configured pattern."""
        n = self.config.num_requests
        rate = self.config.arrival_rate_rps

        if self.config.arrival_pattern == "constant":
            inter_arrivals = np.full(n, 1.0 / rate)
        elif self.config.arrival_pattern == "poisson":
            inter_arrivals = self._rng.exponential(1.0 / rate, size=n)
        elif self.config.arrival_pattern == "gamma":
            shape = self.config.gamma_shape
            scale = 1.0 / (rate * shape)
            inter_arrivals = self._rng.gamma(shape, scale, size=n)
        elif self.config.arrival_pattern == "bursty":
            # Poisson with sinusoidal rate modulation
            base_inter = self._rng.exponential(1.0 / rate, size=n)
            cumulative = np.cumsum(base_inter)
            period = self.config.burst_period_s
            amplitude = self.config.burst_amplitude
            modulation = 1.0 + amplitude * np.sin(2 * np.pi * cumulative / period)
            modulation = np.clip(modulation, 0.2, amplitude + 1.0)
            inter_arrivals = base_inter / modulation
        else:
            raise ValueError(f"Unknown arrival pattern: {self.config.arrival_pattern}")

        # Convert to cumulative nanoseconds
        cumulative_s = np.cumsum(inter_arrivals)
        return (cumulative_s * NS_PER_S).astype(np.int64)

    def _sample_lengths(self, dist_config: Dict[str, Any]) -> np.ndarray:
        """Sample sequence lengths from the configured distribution."""
        n = self.config.num_requests
        dist = dist_config.get("dist", "lognormal")
        min_val = dist_config.get("min", 1)
        max_val = dist_config.get("max", 100000)
        mean = dist_config.get("mean", 512)
        std = dist_config.get("std", 200)

        if dist == "lognormal":
            variance = std ** 2
            mu = np.log(mean ** 2 / np.sqrt(variance + mean ** 2))
            sigma = np.sqrt(np.log(1 + variance / mean ** 2))
            samples = self._rng.lognormal(mu, sigma, size=n)
        elif dist == "uniform":
            samples = self._rng.uniform(min_val, max_val, size=n)
        elif dist == "normal":
            samples = self._rng.normal(mean, std, size=n)
        elif dist == "fixed":
            samples = np.full(n, mean)
        else:
            raise ValueError(f"Unknown distribution: {dist}")

        return np.clip(np.round(samples), min_val, max_val)

    @classmethod
    def preset(cls, name: str, arrival_rate: float = 10.0,
               num_requests: int = 300, seed: int = 42) -> 'WorkloadGenerator':
        """Create WorkloadGenerator from a named preset."""
        presets = {
            "chat": WorkloadConfig(
                arrival_rate_rps=arrival_rate,
                arrival_pattern="poisson",
                num_requests=num_requests,
                input_length_distribution={
                    "dist": "lognormal", "mean": 256, "std": 150, "min": 32, "max": 2048,
                },
                output_length_distribution={
                    "dist": "lognormal", "mean": 256, "std": 128, "min": 32, "max": 2048,
                },
                random_seed=seed,
            ),
            "rag": WorkloadConfig(
                arrival_rate_rps=arrival_rate,
                arrival_pattern="poisson",
                num_requests=num_requests,
                input_length_distribution={
                    "dist": "lognormal", "mean": 2048, "std": 1024, "min": 512, "max": 8192,
                },
                output_length_distribution={
                    "dist": "lognormal", "mean": 128, "std": 64, "min": 32, "max": 512,
                },
                random_seed=seed,
            ),
            "batch": WorkloadConfig(
                arrival_rate_rps=arrival_rate,
                arrival_pattern="constant",
                num_requests=num_requests,
                input_length_distribution={
                    "dist": "lognormal", "mean": 512, "std": 100, "min": 256, "max": 1024,
                },
                output_length_distribution={
                    "dist": "lognormal", "mean": 64, "std": 32, "min": 16, "max": 256,
                },
                random_seed=seed,
            ),
            "coding": WorkloadConfig(
                arrival_rate_rps=arrival_rate,
                arrival_pattern="bursty",
                num_requests=num_requests,
                input_length_distribution={
                    "dist": "lognormal", "mean": 1024, "std": 512, "min": 128, "max": 4096,
                },
                output_length_distribution={
                    "dist": "lognormal", "mean": 512, "std": 256, "min": 64, "max": 4096,
                },
                random_seed=seed,
            ),
            "classification": WorkloadConfig(
                arrival_rate_rps=arrival_rate,
                arrival_pattern="poisson",
                num_requests=num_requests,
                input_length_distribution={
                    "dist": "lognormal", "mean": 128, "std": 64, "min": 16, "max": 512,
                },
                output_length_distribution={
                    "dist": "fixed", "mean": 5, "min": 1, "max": 20,
                },
                random_seed=seed,
            ),
        }
        if name not in presets:
            raise ValueError(
                f"Unknown preset: {name}. Available: {list(presets.keys())}"
            )
        return cls(presets[name])
