"""MoE parameter-count regression tests.

A heuristic in ``_is_down_shared`` inferred that experts share a single
down-projection whenever ``n_routed_experts > 32`` and
``moe_intermediate_size / hidden_size < 0.6``, on the reasoning that "small experts
often share down projection".

That is wrong, and wrong in the direction that under-counts a model into an
OOMKill. A small intermediate/hidden ratio is the *signature of fine-grained MoE*
-- many narrow experts -- not evidence of weight sharing. Every mainstream MoE
gives each expert its own gate, up and down matrix.

It fired on precisely the architectures that do not share, and Mixtral escaped only
by having 8 experts:

    Qwen3.5-35B-A3B   256 experts, ratio 0.250  -> fired
    DeepSeek-V3       256 experts, ratio 0.286  -> fired
    Qwen3-235B-A22B   128 experts, ratio 0.375  -> fired
    Mixtral-8x7B        8 experts, ratio 3.500  -> did not

Downstream, on Qwen3.5-35B-A3B it dropped 10.70B parameters (21.39 GB at bf16), so
budsim reported 49.08 GB for a checkpoint whose own safetensors index says 71.90 GB,
budcluster sized a 59 GiB pod, and the deployment OOMKilled loading weights.
"""

import pytest

from llm_memory_calculator.parameter_counter import UniversalParameterCounter


def counter():
    return UniversalParameterCounter()


# Published parameter counts. These are the whole point of the file: an estimator
# that cannot reproduce a known model's size cannot be trusted to size a pod.
MIXTRAL_8X7B = dict(
    model_type="mixtral",
    num_hidden_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    vocab_size=32000,
    intermediate_size=14336,
    moe_intermediate_size=14336,
    num_experts=8,
    n_routed_experts=8,
    num_experts_per_tok=2,
    torch_dtype="bfloat16",
)
QWEN3_235B_A22B = dict(
    model_type="qwen3_moe",
    num_hidden_layers=94,
    hidden_size=4096,
    num_attention_heads=64,
    num_key_value_heads=4,
    head_dim=128,
    vocab_size=151936,
    moe_intermediate_size=1536,
    num_experts=128,
    n_routed_experts=128,
    num_experts_per_tok=8,
    torch_dtype="bfloat16",
)
# The deployment that OOMKilled. 256 fine-grained experts, ratio 0.25.
QWEN3_5_35B_A3B_TEXT = dict(
    model_type="qwen3_5_moe_text",
    num_hidden_layers=40,
    hidden_size=2048,
    num_attention_heads=16,
    num_key_value_heads=2,
    head_dim=256,
    vocab_size=248320,
    moe_intermediate_size=512,
    shared_expert_intermediate_size=512,
    num_experts=256,
    n_routed_experts=256,
    num_experts_per_tok=8,
    torch_dtype="bfloat16",
)


@pytest.mark.parametrize(
    "name, config, published_billions",
    [
        ("Mixtral-8x7B", MIXTRAL_8X7B, 46.7),
        ("Qwen3-235B-A22B", QWEN3_235B_A22B, 235.0),
    ],
)
def test_reproduces_published_parameter_counts(name, config, published_billions):
    """Within a few percent of the published size for both expert granularities.

    Mixtral has 8 wide experts, Qwen3-235B has 128 narrow ones. The retired
    heuristic put the second ~32% low while leaving the first untouched, so a suite
    that only covered Mixtral would have called the estimator healthy.
    """
    counted = counter().count_parameters(config) / 1e9
    assert counted == pytest.approx(published_billions, rel=0.03), (
        f"{name}: counted {counted:.1f}B vs published {published_billions}B"
    )


def test_experts_own_their_down_projection_unless_told_otherwise():
    """Each expert must be charged gate + up + down: three matrices, not two.

    With sharing assumed, Qwen3.5-35B-A3B's experts come to 21.52B parameters; with
    each expert owning its down-projection, 32.21B. That 10.70B gap is 21.39 GB of
    weights, and it is the whole reason a pod was sized 21 GiB short.
    """
    c = counter()
    assert not c._is_down_shared(QWEN3_5_35B_A3B_TEXT)

    layers, experts, hidden, inter = 40, 256, 2048, 512
    moe = c._calculate_moe_params(QWEN3_5_35B_A3B_TEXT, layers, hidden)
    per_expert_three_matrices = layers * experts * 3 * hidden * inter
    # the term is experts + router, so it must at least cover the expert matrices
    assert moe >= per_expert_three_matrices
    # and it must NOT be the shared-projection figure
    shared = layers * (experts * 2 * hidden * inter + hidden * inter)
    assert moe > shared * 1.4


@pytest.mark.parametrize(
    "name, config",
    [
        ("Qwen3.5-35B-A3B", QWEN3_5_35B_A3B_TEXT),
        ("Qwen3-235B-A22B", QWEN3_235B_A22B),
    ],
)
def test_the_retired_ratio_heuristic_does_not_come_back(name, config):
    """Guard the specific inference that caused the OOMKill.

    Both of these have >32 experts and an intermediate/hidden ratio below 0.6 --
    exactly the trigger condition. Sharing must now require an explicit config key.
    """
    c = counter()
    experts = config["n_routed_experts"]
    ratio = config["moe_intermediate_size"] / config["hidden_size"]
    assert experts > 32 and ratio < 0.6, "fixture no longer exercises the trigger"
    assert not c._is_down_shared(config), f"{name}: ratio heuristic reintroduced"


def test_an_explicit_sharing_key_is_still_honoured():
    """Removing the guess must not remove the ability to state the fact."""
    c = counter()
    for key in c.shared_down_keys:
        assert c._is_down_shared({**QWEN3_5_35B_A3B_TEXT, key: True}), key


def test_gated_ffn_is_known_from_model_type_not_just_the_activation_string():
    """A 3-matrix expert must not be charged as 2 when `hidden_act` is missing.

    Detection falls back to the activation string, and dropping to 2 matrices is a
    33% under-count of every expert. The model_type set exists precisely so an
    architectural fact does not depend on an optional field being populated.
    """
    c = counter()
    without_act = {k: v for k, v in QWEN3_5_35B_A3B_TEXT.items() if k != "hidden_act"}
    assert c._is_gated_ffn(without_act), "model_type alone must establish gated FFN"


def test_a_dense_model_is_untouched_by_moe_logic():
    """No experts means no MoE term at all, whatever the ratios look like."""
    dense = dict(
        model_type="llama",
        num_hidden_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=128256,
        intermediate_size=14336,
        torch_dtype="bfloat16",
    )
    assert counter()._calculate_moe_params(dense, 32, 4096) == 0
