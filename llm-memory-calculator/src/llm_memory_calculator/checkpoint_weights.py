"""Weight bytes taken from a checkpoint, rather than estimated from its architecture.

Determining a model's weight memory has two methods, and this module is the FIRST
one: if the checkpoint says how much it weighs, that IS the answer. Counting
parameters from ``config.json`` is the fallback for when it does not.

Why the checkpoint comes first: the estimator has to know the architecture, and it can
be wrong about it. A heuristic once assumed 256 experts shared a down-projection and
under-counted Qwen3.5-35B-A3B by 47% -- 49.08 GB reported for a checkpoint whose own
index declares 71.90 GB -- and a pod sized from that OOMKilled while loading weights.
After fixing it the estimate is still ~3.4% low. The checkpoint cannot be wrong about
the architecture, because it does not model it.

That also makes this architecture-agnostic by construction: it branches on file
layout, never on ``model_type``. Dense, MoE, hybrid linear-attention, multimodal and
quantized exports all answer the same way. A quantized export is the sharpest case --
same parameter count, a quarter of the bytes, so any counting rule must know the
scheme while the checkpoint simply states the answer.

**A declared total is not a measurement of local disk, and that distinction is the
whole point of this module.** ``metadata.total_size`` in a shard index is a property
of the model: it is equally true before the shards are fetched, while they are being
fetched, and after. Callers routinely size a model *before* transferring it -- a
control plane picking a pod memory limit has the index (a few hundred KB, pulled with
``config.json``) and none of the shards. Refusing to answer there would push exactly
that caller onto the estimator this module exists to backstop, which is what happened:
an earlier version required the shards to be present, so the one caller that most
needed the true number was the one guaranteed not to get it.

Completeness therefore matters only where a partial download can actually *understate*
the answer -- summing the files. A declared total does not shrink when shards are
missing; a sum does, and would size a pod into a confident OOMKill carrying the
authority of a measurement. So the shard-count check lives on the summing path.

Kept dependency-free (stdlib only) so a control plane can import it without pulling
the plotting and dataframe stack the analysis modules use.
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple


# Ordered by preference. Safetensors wins over .bin because repos commonly ship both
# and summing both would double the model.
WEIGHT_SUFFIXES: Tuple[str, ...] = (".safetensors", ".bin", ".pth", ".pt")

# Sharded checkpoints name themselves: ``model-00003-of-00014.safetensors``. The tail
# of that name is the repo telling us how many shards there are, which is the only way
# to notice a truncated download once there is no index to compare against.
SHARD_NAME = re.compile(r"-(\d+)-of-(\d+)\.[A-Za-z]+$")


def _sum_by_suffix(directory: str, names: List[str], suffix: str) -> int:
    total = 0
    for name in names:
        if name.endswith(suffix):
            try:
                total += os.path.getsize(os.path.join(directory, name))
            except OSError:
                continue
    return total


def _missing_shards(names: List[str], suffix: str) -> Tuple[int, int]:
    """``(missing, expected)`` from the ``-of-NNNNN`` in the filenames, ``(0, 0)`` if unsharded."""
    expected = 0
    seen: Dict[int, bool] = {}
    for name in names:
        if not name.endswith(suffix):
            continue
        match = SHARD_NAME.search(name)
        if match:
            seen[int(match.group(1))] = True
            expected = max(expected, int(match.group(2)))
    if not expected:
        return 0, 0
    return max(0, expected - len(seen)), expected


def weights_from_checkpoint(model_dir: Optional[str]) -> Tuple[Optional[int], str]:
    """Weight bytes for a checkpoint directory, or ``(None, why-not)``.

    Tried in order, stopping at the first that answers:

    1. ``*.index.json`` -> ``metadata.total_size``. Authoritative, free, and available
       before the weights are. The file exists precisely because the model is sharded,
       and it states the total for all of them whether or not they are on disk yet.
    2. Sum of ``*.safetensors`` -- single-file, or sharded without an index.
    3. Sum of ``*.bin`` / ``*.pth`` / ``*.pt`` for pre-safetensors checkpoints.

    Methods 2 and 3 are measurements of local disk, so they refuse when the filenames
    show shards are missing: a truncated checkpoint would otherwise read as a small
    model. Method 1 has no such requirement -- see the module docstring.

    Returns ``(None, reason)`` when nothing is readable, so the caller falls back to
    counting parameters. That is the normal case for a model with neither an index nor
    a local copy, and it is why the estimator remains load-bearing rather than dead code.
    """
    try:
        if not model_dir or not os.path.isdir(model_dir):
            return None, "checkpoint directory not readable"

        names = os.listdir(model_dir)

        for name in names:
            if not name.endswith(".index.json"):
                continue
            try:
                with open(os.path.join(model_dir, name)) as fh:
                    index = json.load(fh)
                declared = int((index.get("metadata") or {}).get("total_size") or 0)
            except (OSError, ValueError, TypeError):
                # A corrupt index must not veto a perfectly readable set of shards.
                continue
            if declared > 0:
                return declared, f"{name}:metadata.total_size"

        for suffix in WEIGHT_SUFFIXES:
            total = _sum_by_suffix(model_dir, names, suffix)
            if total <= 0:
                continue
            missing, expected = _missing_shards(names, suffix)
            if missing:
                return None, (
                    f"checkpoint incomplete: {missing} of {expected} *{suffix} shards are "
                    f"absent, so summing the {total / 1024**3:.2f} GiB present would "
                    f"understate the model"
                )
            return total, f"sum of *{suffix}"

        return None, "no weight files in the checkpoint directory"
    except Exception as e:  # noqa: BLE001
        return None, f"could not measure checkpoint: {e}"
