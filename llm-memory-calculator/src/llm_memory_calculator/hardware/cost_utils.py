"""Cost utility functions for hardware configurations."""

from typing import Dict, Any, Optional, Tuple


def get_best_rate(
    cost_data: Optional[Dict[str, Any]],
    allow_spot: bool = True,
    preferred_provider: Optional[str] = None,
) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """Get best available rate from cost data.

    Args:
        cost_data: Cost dictionary from hardware config containing pricing fields.
        allow_spot: Whether to include spot/preemptible instances in consideration.
        preferred_provider: If specified, only consider rates from this provider.

    Returns:
        Tuple of (hourly_rate, provider, tier) or (None, None, None) if no rates available.

    Example:
        >>> cost_data = {'aws_on_demand': 2.21, 'aws_spot': 0.80, 'lambda_labs': 1.10}
        >>> rate, provider, tier = get_best_rate(cost_data)
        >>> print(f"${rate}/hr from {provider} ({tier})")
        $0.80/hr from AWS (spot)
    """
    if not cost_data:
        return (None, None, None)

    # If preferred provider specified, filter to only that provider
    if preferred_provider:
        rates = _get_provider_rates(cost_data, preferred_provider, allow_spot)
        if rates:
            return min(rates, key=lambda x: x[0])
        # Fall through to all providers if preferred has no rates

    rates = []

    # Specialized providers (often cheapest)
    for provider, key in [
        ('Lambda Labs', 'lambda_labs'),
        ('CoreWeave', 'coreweave'),
        ('RunPod', 'runpod'),
        ('Vast.ai', 'vast_ai'),
    ]:
        if cost_data.get(key, 0) > 0:
            rates.append((cost_data[key], provider, 'on_demand'))

    # Spot/preemptible instances
    if allow_spot:
        if cost_data.get('aws_spot', 0) > 0:
            rates.append((cost_data['aws_spot'], 'AWS', 'spot'))
        if cost_data.get('gcp_preemptible', 0) > 0:
            rates.append((cost_data['gcp_preemptible'], 'GCP', 'preemptible'))
        if cost_data.get('azure_spot', 0) > 0:
            rates.append((cost_data['azure_spot'], 'Azure', 'spot'))

    # On-demand instances (hyperscalers)
    for provider, key in [
        ('AWS', 'aws_on_demand'),
        ('GCP', 'gcp_on_demand'),
        ('Azure', 'azure_on_demand'),
    ]:
        if cost_data.get(key, 0) > 0:
            rates.append((cost_data[key], provider, 'on_demand'))

    if not rates:
        return (None, None, None)

    return min(rates, key=lambda x: x[0])


def _get_provider_rates(
    cost_data: Dict[str, Any],
    provider: str,
    allow_spot: bool,
) -> list:
    """Get rates for a specific provider."""
    rates = []
    provider_lower = provider.lower()

    if provider_lower == "aws":
        if cost_data.get('aws_on_demand', 0) > 0:
            rates.append((cost_data['aws_on_demand'], 'AWS', 'on_demand'))
        if allow_spot and cost_data.get('aws_spot', 0) > 0:
            rates.append((cost_data['aws_spot'], 'AWS', 'spot'))
    elif provider_lower == "gcp":
        if cost_data.get('gcp_on_demand', 0) > 0:
            rates.append((cost_data['gcp_on_demand'], 'GCP', 'on_demand'))
        if allow_spot and cost_data.get('gcp_preemptible', 0) > 0:
            rates.append((cost_data['gcp_preemptible'], 'GCP', 'preemptible'))
    elif provider_lower == "azure":
        if cost_data.get('azure_on_demand', 0) > 0:
            rates.append((cost_data['azure_on_demand'], 'Azure', 'on_demand'))
        if allow_spot and cost_data.get('azure_spot', 0) > 0:
            rates.append((cost_data['azure_spot'], 'Azure', 'spot'))
    elif provider_lower in ("lambda", "lambda labs"):
        if cost_data.get('lambda_labs', 0) > 0:
            rates.append((cost_data['lambda_labs'], 'Lambda Labs', 'on_demand'))
    elif provider_lower == "coreweave":
        if cost_data.get('coreweave', 0) > 0:
            rates.append((cost_data['coreweave'], 'CoreWeave', 'on_demand'))
    elif provider_lower == "runpod":
        if cost_data.get('runpod', 0) > 0:
            rates.append((cost_data['runpod'], 'RunPod', 'on_demand'))
    elif provider_lower in ("vast", "vast.ai"):
        if cost_data.get('vast_ai', 0) > 0:
            rates.append((cost_data['vast_ai'], 'Vast.ai', 'on_demand'))

    return rates


def has_cost_data(config: Dict[str, Any]) -> bool:
    """Check if hardware config has cost data.

    Args:
        config: Hardware configuration dictionary.

    Returns:
        True if config has at least one non-zero cost field.
    """
    cost = config.get('cost')
    if not cost:
        return False

    cost_keys = [
        'aws_on_demand',
        'aws_spot',
        'gcp_on_demand',
        'gcp_preemptible',
        'azure_on_demand',
        'azure_spot',
        'lambda_labs',
        'coreweave',
        'runpod',
        'vast_ai',
    ]
    return any(cost.get(k, 0) > 0 for k in cost_keys)


def get_all_provider_rates(cost_data: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Get all available rates organized by provider.

    Args:
        cost_data: Cost dictionary from hardware config.

    Returns:
        Dictionary mapping provider names to their available tiers and rates.

    Example:
        >>> rates = get_all_provider_rates(cost_data)
        >>> print(rates)
        {'AWS': {'on_demand': 2.21, 'spot': 0.80}, 'Lambda Labs': {'on_demand': 1.10}}
    """
    if not cost_data:
        return {}

    result = {}

    # AWS
    aws_rates = {}
    if cost_data.get('aws_on_demand', 0) > 0:
        aws_rates['on_demand'] = cost_data['aws_on_demand']
    if cost_data.get('aws_spot', 0) > 0:
        aws_rates['spot'] = cost_data['aws_spot']
    if aws_rates:
        result['AWS'] = aws_rates

    # GCP
    gcp_rates = {}
    if cost_data.get('gcp_on_demand', 0) > 0:
        gcp_rates['on_demand'] = cost_data['gcp_on_demand']
    if cost_data.get('gcp_preemptible', 0) > 0:
        gcp_rates['preemptible'] = cost_data['gcp_preemptible']
    if gcp_rates:
        result['GCP'] = gcp_rates

    # Azure
    azure_rates = {}
    if cost_data.get('azure_on_demand', 0) > 0:
        azure_rates['on_demand'] = cost_data['azure_on_demand']
    if cost_data.get('azure_spot', 0) > 0:
        azure_rates['spot'] = cost_data['azure_spot']
    if azure_rates:
        result['Azure'] = azure_rates

    # Specialized providers
    if cost_data.get('lambda_labs', 0) > 0:
        result['Lambda Labs'] = {'on_demand': cost_data['lambda_labs']}
    if cost_data.get('coreweave', 0) > 0:
        result['CoreWeave'] = {'on_demand': cost_data['coreweave']}
    if cost_data.get('runpod', 0) > 0:
        result['RunPod'] = {'on_demand': cost_data['runpod']}
    if cost_data.get('vast_ai', 0) > 0:
        result['Vast.ai'] = {'on_demand': cost_data['vast_ai']}

    return result
