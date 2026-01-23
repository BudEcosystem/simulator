"""
Total Cost of Ownership (TCO) Calculator for Training.

This module provides comprehensive cost estimation for LLM training including:
- Cloud provider pricing (AWS, GCP, Azure, Lambda Labs, CoreWeave, RunPod, Vast.ai)
- Power consumption costs
- Network and storage costs
- Operations overhead

Static pricing database approach - no external API dependencies.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import math

from .cluster_optimizer_types import TCOBreakdown, PricingTier


@dataclass
class GPUPricing:
    """GPU pricing information by provider."""
    gpu_name: str

    # Cloud provider hourly rates (USD per GPU-hour)
    aws_on_demand: float = 0.0
    aws_spot: float = 0.0
    aws_reserved_1yr: float = 0.0
    aws_reserved_3yr: float = 0.0

    gcp_on_demand: float = 0.0
    gcp_preemptible: float = 0.0
    gcp_committed_1yr: float = 0.0
    gcp_committed_3yr: float = 0.0

    azure_on_demand: float = 0.0
    azure_spot: float = 0.0

    # Specialized providers
    lambda_labs: float = 0.0
    coreweave: float = 0.0
    runpod: float = 0.0
    vast_ai: float = 0.0

    # On-premise (for reference)
    purchase_price_usd: float = 0.0
    tdp_watts: int = 400

    def get_best_rate(
        self,
        allow_spot: bool = True,
        allow_preemptible: bool = True,
        preferred_provider: Optional[str] = None,
    ) -> Tuple[float, str, str]:
        """
        Get the best available rate.

        Returns:
            Tuple of (hourly_rate, provider, tier)
        """
        rates = []

        # Preferred provider first
        if preferred_provider:
            provider_rates = self._get_provider_rates(preferred_provider, allow_spot, allow_preemptible)
            if provider_rates:
                return min(provider_rates, key=lambda x: x[0])

        # Specialized providers (often cheapest)
        if self.lambda_labs > 0:
            rates.append((self.lambda_labs, "Lambda Labs", "on_demand"))
        if self.coreweave > 0:
            rates.append((self.coreweave, "CoreWeave", "on_demand"))
        if self.runpod > 0:
            rates.append((self.runpod, "RunPod", "on_demand"))
        if self.vast_ai > 0:
            rates.append((self.vast_ai, "Vast.ai", "on_demand"))

        # Spot/preemptible instances
        if allow_spot or allow_preemptible:
            if allow_spot and self.aws_spot > 0:
                rates.append((self.aws_spot, "AWS", "spot"))
            if allow_preemptible and self.gcp_preemptible > 0:
                rates.append((self.gcp_preemptible, "GCP", "preemptible"))
            if allow_spot and self.azure_spot > 0:
                rates.append((self.azure_spot, "Azure", "spot"))

        # On-demand instances
        if self.aws_on_demand > 0:
            rates.append((self.aws_on_demand, "AWS", "on_demand"))
        if self.gcp_on_demand > 0:
            rates.append((self.gcp_on_demand, "GCP", "on_demand"))
        if self.azure_on_demand > 0:
            rates.append((self.azure_on_demand, "Azure", "on_demand"))

        if not rates:
            return (5.0, "Unknown", "on_demand")  # Default fallback

        return min(rates, key=lambda x: x[0])

    def _get_provider_rates(
        self,
        provider: str,
        allow_spot: bool,
        allow_preemptible: bool,
    ) -> list:
        """Get rates for a specific provider."""
        rates = []
        provider = provider.lower()

        if provider == "aws":
            if self.aws_on_demand > 0:
                rates.append((self.aws_on_demand, "AWS", "on_demand"))
            if allow_spot and self.aws_spot > 0:
                rates.append((self.aws_spot, "AWS", "spot"))
        elif provider == "gcp":
            if self.gcp_on_demand > 0:
                rates.append((self.gcp_on_demand, "GCP", "on_demand"))
            if allow_preemptible and self.gcp_preemptible > 0:
                rates.append((self.gcp_preemptible, "GCP", "preemptible"))
        elif provider == "azure":
            if self.azure_on_demand > 0:
                rates.append((self.azure_on_demand, "Azure", "on_demand"))
            if allow_spot and self.azure_spot > 0:
                rates.append((self.azure_spot, "Azure", "spot"))
        elif provider == "lambda":
            if self.lambda_labs > 0:
                rates.append((self.lambda_labs, "Lambda Labs", "on_demand"))
        elif provider == "coreweave":
            if self.coreweave > 0:
                rates.append((self.coreweave, "CoreWeave", "on_demand"))
        elif provider == "runpod":
            if self.runpod > 0:
                rates.append((self.runpod, "RunPod", "on_demand"))
        elif provider == "vast":
            if self.vast_ai > 0:
                rates.append((self.vast_ai, "Vast.ai", "on_demand"))

        return rates


# Static GPU pricing database (updated periodically)
# Prices in USD per GPU-hour as of late 2024
GPU_PRICING: Dict[str, GPUPricing] = {
    # NVIDIA V100
    "v100_16gb": GPUPricing(
        gpu_name="v100_16gb",
        aws_on_demand=0.90,
        aws_spot=0.27,
        gcp_on_demand=0.74,
        gcp_preemptible=0.22,
        lambda_labs=0.50,
        vast_ai=0.25,
        runpod=0.39,
        tdp_watts=250,
        purchase_price_usd=3000,
    ),
    "v100_32gb": GPUPricing(
        gpu_name="v100_32gb",
        aws_on_demand=1.21,
        aws_spot=0.36,
        gcp_on_demand=0.99,
        gcp_preemptible=0.30,
        lambda_labs=0.60,
        vast_ai=0.35,
        runpod=0.49,
        tdp_watts=250,
        purchase_price_usd=5000,
    ),

    # NVIDIA A100
    "a100_40gb": GPUPricing(
        gpu_name="a100_40gb",
        aws_on_demand=2.21,
        aws_spot=0.80,
        gcp_on_demand=2.06,
        gcp_preemptible=0.72,
        azure_on_demand=2.42,
        azure_spot=0.73,
        lambda_labs=1.10,
        coreweave=1.28,
        vast_ai=0.80,
        runpod=0.79,
        tdp_watts=400,
        purchase_price_usd=10000,
    ),
    "a100_80gb": GPUPricing(
        gpu_name="a100_80gb",
        aws_on_demand=3.67,
        aws_spot=1.40,
        gcp_on_demand=3.67,
        gcp_preemptible=1.10,
        azure_on_demand=3.67,
        azure_spot=1.28,
        lambda_labs=1.29,
        coreweave=1.85,
        vast_ai=1.10,
        runpod=1.19,
        tdp_watts=400,
        purchase_price_usd=15000,
    ),

    # NVIDIA L40S
    "l40s": GPUPricing(
        gpu_name="l40s",
        aws_on_demand=1.98,
        gcp_on_demand=1.70,
        lambda_labs=0.99,
        coreweave=1.14,
        runpod=0.99,
        tdp_watts=350,
        purchase_price_usd=8000,
    ),

    # NVIDIA H100
    "h100_sxm": GPUPricing(
        gpu_name="h100_sxm",
        aws_on_demand=4.76,
        aws_spot=2.00,
        gcp_on_demand=4.76,
        gcp_preemptible=1.90,
        azure_on_demand=5.12,
        azure_spot=2.05,
        lambda_labs=2.49,
        coreweave=2.85,
        vast_ai=2.00,
        runpod=2.39,
        tdp_watts=700,
        purchase_price_usd=30000,
    ),
    "h100_pcie": GPUPricing(
        gpu_name="h100_pcie",
        aws_on_demand=3.98,
        gcp_on_demand=3.98,
        lambda_labs=1.99,
        coreweave=2.23,
        runpod=1.99,
        tdp_watts=350,
        purchase_price_usd=25000,
    ),

    # NVIDIA H200
    "h200": GPUPricing(
        gpu_name="h200",
        aws_on_demand=6.50,
        gcp_on_demand=6.50,
        lambda_labs=3.99,
        coreweave=3.85,
        tdp_watts=700,
        purchase_price_usd=40000,
    ),

    # NVIDIA Blackwell (estimated pricing)
    "b100": GPUPricing(
        gpu_name="b100",
        lambda_labs=5.00,
        coreweave=5.50,
        tdp_watts=700,
        purchase_price_usd=50000,
    ),
    "b200": GPUPricing(
        gpu_name="b200",
        lambda_labs=7.00,
        coreweave=7.50,
        tdp_watts=1000,
        purchase_price_usd=70000,
    ),

    # AMD MI300X
    "mi300x": GPUPricing(
        gpu_name="mi300x",
        aws_on_demand=3.50,
        gcp_on_demand=3.50,
        coreweave=2.50,
        tdp_watts=750,
        purchase_price_usd=20000,
    ),
}

# Alias mapping for different naming conventions
GPU_PRICING_ALIASES: Dict[str, str] = {
    # Standard naming
    "A100_40GB_GPU": "a100_40gb",
    "A100_80GB_GPU": "a100_80gb",
    "H100_GPU": "h100_sxm",
    "H100_80GB_GPU": "h100_sxm",
    "H100_SXM_GPU": "h100_sxm",
    "H100_PCIe_GPU": "h100_pcie",
    "H200_GPU": "h200",
    "B100_GPU": "b100",
    "B200_GPU": "b200",
    "V100_16GB_GPU": "v100_16gb",
    "V100_32GB_GPU": "v100_32gb",
    "L40S_48GB_GPU": "l40s",
    "MI300X_GPU": "mi300x",

    # Lowercase variants
    "a100": "a100_80gb",
    "h100": "h100_sxm",
    "h200": "h200",
    "b100": "b100",
    "b200": "b200",
    "l40s": "l40s",
    "mi300x": "mi300x",
}


def get_gpu_pricing(gpu_type: str) -> GPUPricing:
    """
    Get pricing information for a GPU type.

    Args:
        gpu_type: GPU type name (various formats supported)

    Returns:
        GPUPricing object

    Raises:
        ValueError: If GPU type is not found
    """
    # Try direct lookup
    normalized = gpu_type.lower().replace("-", "_").replace(" ", "_")
    if normalized in GPU_PRICING:
        return GPU_PRICING[normalized]

    # Try alias lookup
    if gpu_type in GPU_PRICING_ALIASES:
        return GPU_PRICING[GPU_PRICING_ALIASES[gpu_type]]

    # Try to find partial match
    for alias, canonical in GPU_PRICING_ALIASES.items():
        if alias.lower() in gpu_type.lower() or gpu_type.lower() in alias.lower():
            return GPU_PRICING[canonical]

    # Default to H100 pricing as fallback
    return GPU_PRICING["h100_sxm"]


def calculate_tco(
    gpu_type: str,
    num_gpus: int,
    training_hours: float,
    dataset_tokens: int = 0,
    dataset_samples: int = 0,
    avg_sequence_length: int = 2048,
    pricing_tier: str = "on_demand",
    provider: Optional[str] = None,
    allow_spot: bool = False,
    electricity_rate: float = 0.10,  # USD per kWh
    pue: float = 1.2,                # Power Usage Effectiveness
    include_operations_overhead: bool = True,
) -> TCOBreakdown:
    """
    Calculate comprehensive TCO for a training job.

    Args:
        gpu_type: GPU type name
        num_gpus: Number of GPUs
        training_hours: Total training duration in hours
        dataset_tokens: Total tokens (for cost-per-token calculation)
        dataset_samples: Total samples (alternative to tokens)
        avg_sequence_length: Average sequence length (for sample to token conversion)
        pricing_tier: Pricing tier (on_demand, spot, reserved_1yr, reserved_3yr)
        provider: Preferred cloud provider (aws, gcp, azure, lambda, coreweave, runpod, vast)
        allow_spot: Allow spot/preemptible instances
        electricity_rate: Electricity cost in USD per kWh
        pue: Data center Power Usage Effectiveness (1.0-2.0)
        include_operations_overhead: Include 15% operations overhead

    Returns:
        TCOBreakdown with detailed cost analysis
    """
    pricing = get_gpu_pricing(gpu_type)

    # Get hourly rate
    hourly_rate, actual_provider, actual_tier = pricing.get_best_rate(
        allow_spot=allow_spot,
        allow_preemptible=allow_spot,
        preferred_provider=provider,
    )

    # Override with specific tier if requested
    if pricing_tier == "spot" and allow_spot:
        if pricing.aws_spot > 0:
            hourly_rate = pricing.aws_spot
            actual_provider = "AWS"
            actual_tier = "spot"
        elif pricing.gcp_preemptible > 0:
            hourly_rate = pricing.gcp_preemptible
            actual_provider = "GCP"
            actual_tier = "preemptible"

    # 1. GPU compute cost
    gpu_compute_cost = hourly_rate * num_gpus * training_hours

    # 2. Power cost
    tdp_watts = pricing.tdp_watts
    power_kw = (tdp_watts * num_gpus) / 1000
    power_cost = power_kw * training_hours * electricity_rate * pue

    # 3. Network cost (InfiniBand for multi-node)
    num_nodes = math.ceil(num_gpus / 8)
    network_cost = 0.0
    if num_nodes > 1:
        # Estimate network egress for gradient sync
        # Roughly 0.50 per node-hour for InfiniBand
        network_cost = 0.50 * num_nodes * training_hours

    # 4. Storage cost (checkpoints, logs)
    # Estimate ~50GB per GPU for checkpoints over training
    storage_gb = num_gpus * 50
    # S3/GCS pricing: ~$0.023 per GB-month
    storage_hours_to_months = training_hours / 720
    storage_cost = storage_gb * 0.023 * storage_hours_to_months

    # 5. Operations overhead (15% of compute cost)
    operations_overhead = gpu_compute_cost * 0.15 if include_operations_overhead else 0.0

    # Calculate savings from spot (compared to on-demand baseline)
    spot_savings = 0.0
    if actual_tier in ("spot", "preemptible"):
        on_demand_rate = pricing.aws_on_demand or pricing.gcp_on_demand or hourly_rate * 2
        baseline_cost = on_demand_rate * num_gpus * training_hours
        spot_savings = baseline_cost - gpu_compute_cost

    # Total cost
    total_cost = gpu_compute_cost + power_cost + network_cost + storage_cost + operations_overhead

    # Calculate per-token and per-sample costs
    total_tokens = dataset_tokens if dataset_tokens > 0 else dataset_samples * avg_sequence_length
    cost_per_million_tokens = (total_cost / total_tokens * 1_000_000) if total_tokens > 0 else 0.0
    cost_per_sample = (total_cost / dataset_samples) if dataset_samples > 0 else 0.0

    return TCOBreakdown(
        gpu_compute_cost=gpu_compute_cost,
        power_cost=power_cost,
        network_cost=network_cost,
        storage_cost=storage_cost,
        operations_overhead=operations_overhead,
        spot_savings=spot_savings,
        total_cost=total_cost,
        cost_per_million_tokens=cost_per_million_tokens,
        cost_per_sample=cost_per_sample,
        pricing_tier=actual_tier,
        provider=actual_provider,
    )


def estimate_training_cost(
    gpu_type: str,
    num_gpus: int,
    tokens_per_second: float,
    dataset_tokens: int,
    allow_spot: bool = True,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Quick estimate of training cost given throughput.

    Args:
        gpu_type: GPU type
        num_gpus: Number of GPUs
        tokens_per_second: Training throughput
        dataset_tokens: Total tokens to process
        allow_spot: Allow spot instances
        provider: Preferred provider

    Returns:
        Dictionary with cost estimate
    """
    training_hours = dataset_tokens / (tokens_per_second * 3600) if tokens_per_second > 0 else 0

    tco = calculate_tco(
        gpu_type=gpu_type,
        num_gpus=num_gpus,
        training_hours=training_hours,
        dataset_tokens=dataset_tokens,
        allow_spot=allow_spot,
        provider=provider,
    )

    return {
        'training_hours': training_hours,
        'total_cost_usd': tco.total_cost,
        'cost_per_million_tokens': tco.cost_per_million_tokens,
        'provider': tco.provider,
        'pricing_tier': tco.pricing_tier,
    }


def compare_provider_costs(
    gpu_type: str,
    num_gpus: int,
    training_hours: float,
    dataset_tokens: int = 0,
) -> Dict[str, Dict[str, float]]:
    """
    Compare costs across different providers.

    Returns:
        Dictionary mapping provider to cost breakdown
    """
    providers = ["aws", "gcp", "azure", "lambda", "coreweave", "runpod", "vast"]
    results = {}

    for provider in providers:
        try:
            tco = calculate_tco(
                gpu_type=gpu_type,
                num_gpus=num_gpus,
                training_hours=training_hours,
                dataset_tokens=dataset_tokens,
                provider=provider,
                allow_spot=False,
            )
            if tco.gpu_compute_cost > 0:
                results[provider] = {
                    'total_cost': tco.total_cost,
                    'hourly_rate': tco.gpu_compute_cost / (num_gpus * training_hours),
                    'cost_per_million_tokens': tco.cost_per_million_tokens,
                }
        except Exception:
            continue

    return results
