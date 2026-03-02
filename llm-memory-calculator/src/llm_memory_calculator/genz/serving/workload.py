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
        "dist": "exponential", "mean": 128, "std": 64, "min": 16, "max": 2048,
    })
    model: str = "default"
    random_seed: int = 42
    # For gamma pattern
    gamma_shape: float = 2.0
    # For bursty pattern
    burst_period_s: float = 10.0
    burst_amplitude: float = 3.0
    # For diurnal pattern
    diurnal_period_s: float = 86400.0  # 24 hours default
    diurnal_peak_to_valley: float = 10.0  # R = peak/valley ratio
    # For weibull pattern
    weibull_shape: float = 0.8  # shape < 1 = bursty, > 1 = regular


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
        elif self.config.arrival_pattern == "diurnal":
            # Sinusoidal rate modulation with peak-to-valley ratio R
            # A = (R-1)/(R+1) derives amplitude from ratio
            R = max(self.config.diurnal_peak_to_valley, 1.01)
            A = (R - 1.0) / (R + 1.0)
            period = self.config.diurnal_period_s
            # Generate base Poisson arrivals, then modulate
            base_inter = self._rng.exponential(1.0 / rate, size=n)
            cumulative = np.cumsum(base_inter)
            modulation = 1.0 + A * np.sin(2 * np.pi * cumulative / period)
            modulation = np.clip(modulation, 0.1, 1.0 + A)
            inter_arrivals = base_inter / modulation
        elif self.config.arrival_pattern == "weibull":
            # Weibull inter-arrivals (best fit for medium-sized models, ServeGen)
            shape = self.config.weibull_shape
            # E[X] = scale * Gamma(1 + 1/shape), solve for scale to get E[X] = 1/rate
            from math import gamma as math_gamma
            scale = (1.0 / rate) / math_gamma(1.0 + 1.0 / shape)
            inter_arrivals = self._rng.weibull(shape, size=n) * scale
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
        elif dist == "exponential":
            samples = self._rng.exponential(scale=mean, size=n)
        elif dist == "zipf":
            a = std if std > 1.0 else 2.0  # Zipf shape parameter (must be > 1)
            raw = self._rng.zipf(a, size=n).astype(float)
            # Rescale to target mean
            samples = raw * (mean / raw.mean()) if raw.mean() > 0 else np.full(n, mean)
        elif dist == "bimodal":
            # Mixture of two lognormals (70/30 split) - BurstGPT chat output pattern
            n1 = int(n * 0.7)
            n2 = n - n1
            mean1 = mean * 0.6
            mean2 = mean * 2.0
            std1 = std * 0.5
            std2 = std * 1.5
            var1 = std1 ** 2
            mu1 = np.log(mean1 ** 2 / np.sqrt(var1 + mean1 ** 2))
            sigma1 = np.sqrt(np.log(1 + var1 / mean1 ** 2))
            var2 = std2 ** 2
            mu2 = np.log(mean2 ** 2 / np.sqrt(var2 + mean2 ** 2))
            sigma2 = np.sqrt(np.log(1 + var2 / mean2 ** 2))
            s1 = self._rng.lognormal(mu1, sigma1, size=n1)
            s2 = self._rng.lognormal(mu2, sigma2, size=n2)
            samples = np.concatenate([s1, s2])
            self._rng.shuffle(samples)
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
                    "dist": "lognormal", "mean": 1024, "std": 512, "min": 32, "max": 8192,
                },
                output_length_distribution={
                    "dist": "exponential", "mean": 128, "std": 64, "min": 16, "max": 2048,
                },
                random_seed=seed,
            ),
            "general_chat": WorkloadConfig(
                arrival_rate_rps=arrival_rate,
                arrival_pattern="poisson",
                num_requests=num_requests,
                input_length_distribution={
                    "dist": "lognormal", "mean": 215, "std": 150, "min": 16, "max": 4096,
                },
                output_length_distribution={
                    "dist": "lognormal", "mean": 215, "std": 150, "min": 16, "max": 4096,
                },
                random_seed=seed,
            ),
            "multi_turn_chat": WorkloadConfig(
                arrival_rate_rps=arrival_rate,
                arrival_pattern="poisson",
                num_requests=num_requests,
                input_length_distribution={
                    "dist": "lognormal", "mean": 1024, "std": 512, "min": 64, "max": 16384,
                },
                output_length_distribution={
                    "dist": "lognormal", "mean": 415, "std": 256, "min": 32, "max": 4096,
                },
                random_seed=seed,
            ),
            "rag": WorkloadConfig(
                arrival_rate_rps=arrival_rate,
                arrival_pattern="poisson",
                num_requests=num_requests,
                input_length_distribution={
                    "dist": "lognormal", "mean": 8000, "std": 4000, "min": 1024, "max": 32768,
                },
                output_length_distribution={
                    "dist": "exponential", "mean": 200, "std": 100, "min": 32, "max": 2048,
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
                    "dist": "exponential", "mean": 64, "std": 32, "min": 16, "max": 256,
                },
                random_seed=seed,
            ),
            "coding": WorkloadConfig(
                arrival_rate_rps=arrival_rate,
                arrival_pattern="bursty",
                num_requests=num_requests,
                input_length_distribution={
                    "dist": "lognormal", "mean": 1500, "std": 750, "min": 128, "max": 8192,
                },
                output_length_distribution={
                    "dist": "exponential", "mean": 16, "std": 8, "min": 1, "max": 512,
                },
                random_seed=seed,
            ),
            "inline_completion": WorkloadConfig(
                arrival_rate_rps=arrival_rate,
                arrival_pattern="bursty",
                num_requests=num_requests,
                input_length_distribution={
                    "dist": "lognormal", "mean": 1500, "std": 750, "min": 128, "max": 8192,
                },
                output_length_distribution={
                    "dist": "exponential", "mean": 16, "std": 8, "min": 1, "max": 256,
                },
                random_seed=seed,
            ),
            "long_context": WorkloadConfig(
                arrival_rate_rps=arrival_rate,
                arrival_pattern="poisson",
                num_requests=num_requests,
                input_length_distribution={
                    "dist": "lognormal", "mean": 8000, "std": 4000, "min": 2048, "max": 65536,
                },
                output_length_distribution={
                    "dist": "exponential", "mean": 200, "std": 100, "min": 32, "max": 2048,
                },
                random_seed=seed,
            ),
            "summarization": WorkloadConfig(
                arrival_rate_rps=arrival_rate,
                arrival_pattern="poisson",
                num_requests=num_requests,
                input_length_distribution={
                    "dist": "lognormal", "mean": 4096, "std": 2048, "min": 512, "max": 32768,
                },
                output_length_distribution={
                    "dist": "exponential", "mean": 256, "std": 128, "min": 32, "max": 2048,
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
