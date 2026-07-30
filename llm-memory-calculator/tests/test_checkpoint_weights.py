"""Measuring a checkpoint's weight bytes, the FIRST method for weight memory.

Determining weight memory has two methods and this is the primary one; counting
parameters from config.json is the fallback for when the checkpoint is not readable.
This lived in budcluster because that is where the files happen to be at deploy time
-- placing logic by who holds the input rather than by what owns the method, the same
error that had the LoRA scratch duplicated there.

budsim estimates weights from config.json by counting parameters. That estimate was
47% low for Qwen3.5-35B-A3B (a heuristic assumed 256 experts shared one
down-projection) and is still ~3.4% low after the fix, which is the dangerous
direction -- budcluster sizes the pod from it and a short pod OOMKills while loading.

Where the checkpoint is on disk the number does not have to be estimated at all. The
file sizes ARE the weights, and they are architecture-agnostic: a dense 7B, a
256-expert MoE, a hybrid linear-attention stack, a multimodal wrapper and a quantized
export all answer the same way, and none can be got wrong by a counting rule.

But measurement is not always possible, so this is a fallback chain, not a
replacement. The model registry is a metadata cache -- of five real checkpoints in it,
one had a shard index, two had a bare `model.safetensors`, one had `pytorch_model.bin`
and one had no weight files at all. Weights arrive from object storage per deployment,
so at simulation time there is usually nothing to measure and the estimator remains
the only answer. It has to stay correct; this narrows where it is trusted, it does not
retire it.
"""

import json

import pytest

from llm_memory_calculator.checkpoint_weights import weights_from_checkpoint


measure = weights_from_checkpoint


def write(tmp_path, name, size=0, content=None):
    p = tmp_path / name
    if content is not None:
        p.write_text(content)
    else:
        with open(p, "wb") as fh:
            fh.truncate(size)
    return p


# ------------------------------------------------------------------ the layouts that exist


def test_sharded_with_index_uses_the_declared_total(tmp_path):
    """The Qwen3.5-35B layout. `metadata.total_size` is authoritative and free."""
    total = 71_903_655_008
    write(
        tmp_path,
        "model.safetensors.index.json",
        content=json.dumps({"metadata": {"total_size": total}, "weight_map": {}}),
    )
    for i in range(14):
        write(tmp_path, f"model-{i:05d}-of-00014.safetensors", size=total // 14)

    got, source = measure(str(tmp_path))
    assert got == total
    assert "total_size" in source


def test_single_safetensors_without_an_index(tmp_path):
    """The lfm2/gte layout: one `model.safetensors`, no index anywhere."""
    write(tmp_path, "model.safetensors", size=2_000_000_000)
    got, source = measure(str(tmp_path))
    assert got == 2_000_000_000
    assert "safetensors" in source


def test_sharded_safetensors_without_an_index(tmp_path):
    """Shards but no index -- sum them rather than giving up."""
    for i in range(3):
        write(tmp_path, f"model-{i:05d}-of-00003.safetensors", size=1_000_000_000)
    got, _ = measure(str(tmp_path))
    assert got == 3_000_000_000


def test_pytorch_bin_checkpoint(tmp_path):
    """The bert-tiny layout: pre-safetensors, `pytorch_model.bin` only."""
    write(tmp_path, "pytorch_model.bin", size=17_000_000)
    got, source = measure(str(tmp_path))
    assert got == 17_000_000
    assert ".bin" in source


def test_safetensors_wins_over_bin_when_both_exist(tmp_path):
    """Many repos ship both formats; counting both would double the model."""
    write(tmp_path, "model.safetensors", size=2_000_000_000)
    write(tmp_path, "pytorch_model.bin", size=2_000_000_000)
    got, _ = measure(str(tmp_path))
    assert got == 2_000_000_000


# ------------------------------------------------------------------ when NOT to measure


def test_a_config_only_registry_entry_returns_none(tmp_path):
    """The qwen3-0.6B case: config and tokenizer, no weights.

    This is the common case at simulation time, and it must fall back to the
    estimator rather than reporting a model with no weight.
    """
    write(tmp_path, "config.json", content="{}")
    write(tmp_path, "tokenizer.json", content="{}")
    got, why = measure(str(tmp_path))
    assert got is None
    assert "no weight files" in why


def test_a_truncated_download_is_refused_not_measured(tmp_path):
    """The exact stub that sat in the registry: index present, shards missing.

    Measuring here would report a 67 GiB model as ~0 and size a pod into a confident
    OOMKill -- worse than the estimate it replaced, because it looks authoritative.
    """
    total = 71_903_655_008
    write(
        tmp_path,
        "model.safetensors.index.json",
        content=json.dumps({"metadata": {"total_size": total}, "weight_map": {}}),
    )
    write(tmp_path, "model.safetensors-00009-of-00014.safetensors.aria2", size=0)

    got, why = measure(str(tmp_path))
    assert got is None
    assert "incomplete" in why


def test_a_partially_downloaded_shard_set_is_refused(tmp_path):
    """Half the shards present is still not a model."""
    total = 14_000_000_000
    write(
        tmp_path,
        "model.safetensors.index.json",
        content=json.dumps({"metadata": {"total_size": total}, "weight_map": {}}),
    )
    for i in range(7):
        write(tmp_path, f"model-{i:05d}-of-00014.safetensors", size=1_000_000_000)
    got, why = measure(str(tmp_path))
    assert got is None
    assert "incomplete" in why


def test_a_missing_directory_returns_none(tmp_path):
    got, why = measure(str(tmp_path / "nope"))
    assert got is None and "not readable" in why


def test_an_unparseable_index_falls_through_to_summing(tmp_path):
    """A corrupt index must not veto a perfectly readable set of shards."""
    write(tmp_path, "model.safetensors.index.json", content="{ this is not json")
    write(tmp_path, "model.safetensors", size=5_000_000_000)
    got, source = measure(str(tmp_path))
    assert got == 5_000_000_000
    assert "safetensors" in source


# ------------------------------------------------------------------ architecture-agnostic


@pytest.mark.parametrize(
    "label, files",
    [
        ("dense", {"model.safetensors": 16_000_000_000}),
        ("MoE", {f"model-{i:05d}-of-00014.safetensors": 5_000_000_000 for i in range(14)}),
        ("multimodal", {"model.safetensors": 71_900_000_000}),
        ("quantized AWQ", {"model.safetensors": 4_000_000_000}),
    ],
)
def test_the_same_rule_serves_every_architecture(tmp_path, label, files):
    """No branch on model_type anywhere -- that is the point.

    A quantized export is the sharpest case: its parameter count is unchanged but its
    weights are a quarter of the size, so any counting rule needs to know the scheme
    while the file size simply is the answer.
    """
    for name, size in files.items():
        write(tmp_path, name, size=size)
    got, _ = measure(str(tmp_path))
    assert got == sum(files.values()), label
