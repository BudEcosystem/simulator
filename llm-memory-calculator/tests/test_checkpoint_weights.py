"""Taking a checkpoint's weight bytes, the FIRST method for weight memory.

Determining weight memory has two methods and this is the primary one; counting
parameters from config.json is the fallback for when the checkpoint does not say.

budsim estimates weights from config.json by counting parameters. That estimate was
47% low for Qwen3.5-35B-A3B (a heuristic assumed 256 experts shared one
down-projection) and is still ~3.4% low after the fix, which is the dangerous
direction -- the pod is sized from it and a short pod OOMKills while loading.

Where the checkpoint states its size the number does not have to be estimated at all,
and the statement is architecture-agnostic: a dense 7B, a 256-expert MoE, a hybrid
linear-attention stack, a multimodal wrapper and a quantized export all answer the
same way, and none can be got wrong by a counting rule.

The sharpest distinction these tests draw is between the two ways a checkpoint answers:

* A shard **index** DECLARES the total. That is model metadata -- a few hundred KB
  fetched alongside config.json -- and it is equally true before, during and after the
  shards are transferred. The registry is a metadata cache, so this is precisely the
  state a caller sizing a model *before* deployment finds it in.
* Summing **files** MEASURES local disk. A partial download understates it.

An earlier version required the shards to be present for both, which meant the one
caller that most needed the true number -- sizing a pod before the 67 GiB transfer --
was the one guaranteed to be refused and pushed back onto the estimator. Completeness
is now checked only on the summing path, where a partial download can actually lie.
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


def test_an_index_answers_before_the_shards_arrive(tmp_path):
    """The exact stub that sat in the registry: index present, not one shard fetched.

    This is the whole reason the module exists. A control plane sizes the pod from the
    metadata cache, *then* the 67 GiB transfers -- so every real sizing call sees this
    directory, not a complete one. `metadata.total_size` is a property of the model and
    is already correct here; refusing it (as an earlier version did, on the grounds that
    the bytes were not on disk) sent the caller to the estimator that under-counts this
    exact model, and the pod OOMKilled at 78.96 GiB.
    """
    total = 71_903_655_008
    write(
        tmp_path,
        "model.safetensors.index.json",
        content=json.dumps({"metadata": {"total_size": total}, "weight_map": {}}),
    )
    write(tmp_path, "config.json", content="{}")
    write(tmp_path, "model.safetensors-00009-of-00014.safetensors.aria2", size=0)

    got, source = measure(str(tmp_path))
    assert got == total
    assert "total_size" in source


def test_a_declared_total_does_not_track_the_bytes_on_disk(tmp_path):
    """Half the shards present, and the answer is unchanged.

    A declared total cannot shrink with the download; a sum would. That asymmetry is
    why completeness is checked on one path and not the other.
    """
    total = 14_000_000_000
    write(
        tmp_path,
        "model.safetensors.index.json",
        content=json.dumps({"metadata": {"total_size": total}, "weight_map": {}}),
    )
    for i in range(7):
        write(tmp_path, f"model-{i:05d}-of-00014.safetensors", size=1_000_000_000)
    got, _ = measure(str(tmp_path))
    assert got == total


def test_a_truncated_shard_set_with_no_index_is_refused(tmp_path):
    """No index to declare the total, so summing is all there is -- and it would lie.

    The filenames still say how many shards belong here, which is the only signal left
    that 7 GiB is not the model. Reporting it would size a 14 GiB model into a
    confident OOMKill, worse than the estimate it replaced because it looks measured.
    """
    for i in range(1, 8):
        write(tmp_path, f"model-{i:05d}-of-00014.safetensors", size=1_000_000_000)
    got, why = measure(str(tmp_path))
    assert got is None
    assert "incomplete" in why and "7 of 14" in why


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
        (
            "MoE",
            {f"model-{i:05d}-of-00014.safetensors": 5_000_000_000 for i in range(14)},
        ),
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
