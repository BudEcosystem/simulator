"""Shared Pareto front computation for numeric optimizers."""
from typing import Callable, List, TypeVar

T = TypeVar("T")


def compute_pareto_front(
    results: List[T],
    get_obj_values: Callable[[T], List[float]],
) -> List[T]:
    """Compute the non-dominated Pareto front from a list of results.

    For each result, checks whether any other result dominates it
    (i.e., is at least as good on all objectives and strictly better
    on at least one). Non-dominated results form the Pareto front.

    All objective values should be oriented so that *higher is better*.

    Args:
        results: List of candidate solutions.
        get_obj_values: Callable that extracts a list of objective values
            from a result.  Higher values are better for all objectives.

    Returns:
        List of non-dominated results.
    """
    if not results:
        return []

    pareto: List[T] = []
    for i, ri in enumerate(results):
        vi = get_obj_values(ri)
        dominated = False
        for j, rj in enumerate(results):
            if i == j:
                continue
            vj = get_obj_values(rj)
            if (all(vj[k] >= vi[k] for k in range(len(vi))) and
                    any(vj[k] > vi[k] for k in range(len(vi)))):
                dominated = True
                break
        if not dominated:
            pareto.append(ri)
    return pareto
