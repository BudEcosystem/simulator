"""Weight bytes measured from a checkpoint on disk, rather than estimated.

Determining a model's weight memory has two methods, and this module is the FIRST
one: if the checkpoint is readable, its files ARE the answer. Counting parameters
from ``config.json`` is the fallback for when it is not -- which is common, since
callers often size a model before its weights have been fetched.

Why measurement comes first: the estimator has to know the architecture, and it can
be wrong about it. A heuristic once assumed 256 experts shared a down-projection and
under-counted Qwen3.5-35B-A3B by 47% -- 49.08 GB reported for a checkpoint whose own
index declares 71.90 GB -- and a pod sized from that OOMKilled while loading weights.
After fixing it the estimate is still ~3.4% low. File sizes cannot be wrong about the
architecture, because they do not model it.

That also makes this architecture-agnostic by construction: it branches on file
layout, never on ``model_type``. Dense, MoE, hybrid linear-attention, multimodal and
quantized exports all answer the same way. A quantized export is the sharpest case --
same parameter count, a quarter of the bytes, so any counting rule must know the
scheme while the file size simply is the answer.

Kept dependency-free (stdlib only) so a control plane can import it without pulling
the plotting and dataframe stack the analysis modules use.
"""

import json
import os
from typing import List, Optional, Tuple


# Ordered by preference. Safetensors wins over .bin because repos commonly ship both
# and summing both would double the model.
WEIGHT_SUFFIXES: Tuple[str, ...] = (".safetensors", ".bin", ".pth", ".pt")

# An index that declares more than the shards on disk means a partial download. Read
# it as unusable rather than measuring it: a truncated checkpoint reads as a small
# model and would size a pod into a confident OOMKill -- worse than the estimate it
# replaced, because it carries the authority of a measurement.
SHARD_COMPLETENESS = 0.99


def _sum_by_suffix(directory: str, names: List[str], suffix: str) -> int:
    total = 0
    for name in names:
        if name.endswith(suffix):
            try:
                total += os.path.getsize(os.path.join(directory, name))
            except OSError:
                continue
    return total


def weights_from_checkpoint(model_dir: Optional[str]) -> Tuple[Optional[int], str]:
    """Exact weight bytes for a checkpoint directory, or ``(None, why-not)``.

    Tried in order, stopping at the first that answers:

    1. ``*.index.json`` -> ``metadata.total_size``. Authoritative and free; the file
       exists precisely because the weights are sharded.
    2. Sum of ``*.safetensors`` -- single-file, or sharded without an index.
    3. Sum of ``*.bin`` / ``*.pth`` / ``*.pt`` for pre-safetensors checkpoints.

    Returns ``(None, reason)`` when nothing is readable, so the caller falls back to
    counting parameters. That is the normal case when a model has not been fetched
    yet, and it is why the estimator remains load-bearing rather than dead code.
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
            if declared <= 0:
                continue
            shard_suffix = ".safetensors" if "safetensors" in name else ".bin"
            on_disk = _sum_by_suffix(model_dir, names, shard_suffix)
            if on_disk < declared * SHARD_COMPLETENESS:
                return None, (
                    f"checkpoint incomplete: {name} declares {declared / 1024**3:.2f} GiB "
                    f"but only {on_disk / 1024**3:.2f} GiB of {shard_suffix} is present"
                )
            return declared, f"{name}:metadata.total_size"

        for suffix in WEIGHT_SUFFIXES:
            total = _sum_by_suffix(model_dir, names, suffix)
            if total > 0:
                return total, f"sum of *{suffix}"

        return None, "no weight files in the checkpoint directory"
    except Exception as e:  # noqa: BLE001
        return None, f"could not measure checkpoint: {e}"
