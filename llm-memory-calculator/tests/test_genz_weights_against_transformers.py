"""GenZ's graph weights for MoE models, against transformers' own parameter count.

The truth figures are transformers 5.12's meta-device parameter counts of the text model
(``AutoModelForCausalLM.from_config``) for each config, at 2 bytes a parameter. A graph that
leaves out a shared expert -- the MLP every MoE layer runs beside its routed experts (DeepSeek's
and Kimi's `n_shared_experts` x `moe_intermediate_size`, Qwen MoE's
`shared_expert_intermediate_size`, Granite 4's `shared_intermediate_size`, Llama 4's one
`intermediate_size`-wide `shared_expert`) -- sizes the model below what the GPU must hold. So does
one that graphs a gated FFN as two matrices because the config leaves the activation at its
config class's default (Llama 4's `hidden_act="silu"`).

Each graph must hold at least the model's weights, and not grossly more (an embedding or a norm
GenZ sizes its own way is a rounding error; a whole stack graphed twice is not).
"""

import copy
import json
import os
import sys

import pytest

from llm_memory_calculator.genz.Models.get_language_model import (
    create_full_decode_model,
    create_full_prefill_model,
    get_configs,
)
from llm_memory_calculator.genz.analyse_model import get_model_df, get_summary_table

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
import kv_corpus  # noqa: E402

MIB = 2**20

#: transformers 5.12 text-model parameter counts (meta device).
TRUTH = {
    # DeepseekV32ForCausalLM: 58 MoE layers, each with one 2048-wide shared expert
    "planner/models/deepseek-v3.json": 671_877_929_216,
    # DeepseekV2ForCausalLM: 26 MoE layers x 2 shared experts x 1408
    "deepseek_v2_lite_chat_config.json": 15_706_484_224,
    # GraniteMoeHybridForCausalLM: a 1536-wide shared MLP on each of 40 layers
    "hybrid_configs/ibm-granite_granite-4.0-h-small.json": 32_207_337_984,
    # Qwen2MoeForCausalLM: one 5632-wide shared expert (and its gate) on each of 24 layers
    "planner/models/qwen1.5-moe-a2.7b.json": 14_315_784_192,
    # Qwen3NextForCausalLM: one 512-wide shared expert on each of 48 layers
    "hybrid_configs/Qwen_Qwen3-Next-80B-A3B-Instruct.json": 79_674_391_296,
    # Llama4ForCausalLM: the synthetic text configs as committed (no `hidden_act`: the class's
    # "silu"), one shared expert on each MoE layer (48 for Scout, 24 for Maverick)
    "synth:llama4-scout-text": 107_769_861_120,
    "synth:llama4-maverick-text": 400_711_848_960,
}

#: How far above the truth a graph may size the weights.
SLACK = 1.02
#: What GenZ graphs no operator for: the norm and bias vectors (a (2L + 1) x hidden handful of
#: parameters; DeepSeek-V3's are 1.5 parts in a million, Qwen1.5-MoE's QKV biases 22). Matrices
#: only are asserted against the truth.
VECTORS = 5e-5


def _config(name):
    if name.startswith("synth:"):
        return copy.deepcopy(kv_corpus.synthetic_configs(FIXTURES)[name[6:]])
    with open(os.path.join(FIXTURES, name)) as f:
        return json.load(f)


def _weights_mib(builder, cfg, tmp_path, **kw):
    path = builder(name=get_configs(cfg), data_path=str(tmp_path), **kw)
    df = get_model_df(path, batch_size=1, data_path=str(tmp_path), model_characterstics=True)
    return float(get_summary_table(df, model_characterstics=True)["Total Weights (MB)"].values[0])


@pytest.fixture
def genz_dir(tmp_path):
    for sub in ("model", "sparsity"):
        (tmp_path / sub).mkdir()
    return tmp_path


@pytest.mark.parametrize("name", sorted(TRUTH))
@pytest.mark.parametrize("graph", ["prefill", "decode"])
def test_graph_weights_hold_the_model(name, graph, genz_dir):
    cfg = _config(name)
    truth = TRUTH[name] * 2 / MIB
    floor = truth * (1 - VECTORS)
    if graph == "prefill":
        got = _weights_mib(create_full_prefill_model, cfg, genz_dir, input_sequence_length=512)
    else:
        got = _weights_mib(create_full_decode_model, cfg, genz_dir, input_sequence_length=512,
                           output_gen_tokens=0)
        # A decode step looks its token up in the input embedding (a gather, which GenZ graphs
        # no operator for, on origin/main as here): the decode graph holds every other matrix.
        tc = cfg.get("text_config") or cfg
        floor -= tc["vocab_size"] * tc["hidden_size"] * 2 / MIB
    assert got >= floor, f"{name} {graph}: {got:,.1f} MiB < transformers' {truth:,.1f} MiB"
    assert got <= truth * SLACK, f"{name} {graph}: {got:,.1f} MiB > {SLACK} x {truth:,.1f} MiB"


def test_shared_expert_width_is_read_from_the_config():
    """One reader for every spelling (layer_plan.shared_expert_width), the total width of the MLP
    an MoE layer runs beside its routed experts."""
    from llm_memory_calculator.layer_plan import shared_expert_width

    assert shared_expert_width(_config("planner/models/deepseek-v3.json")) == 2048
    assert shared_expert_width(_config("deepseek_v2_lite_chat_config.json")) == 2 * 1408
    assert shared_expert_width(
        _config("hybrid_configs/ibm-granite_granite-4.0-h-small.json")) == 1536
    assert shared_expert_width(_config("planner/models/qwen1.5-moe-a2.7b.json")) == 5632
    assert shared_expert_width(_config("synth:llama4-scout-text")) == 8192
    # no shared expert: Mixtral-style and Qwen3-MoE
    assert shared_expert_width(_config("planner/models/qwen3-30b-a3b-fp8.json")) == 0
    assert shared_expert_width(_config("hybrid_configs/ai21labs_Jamba-v0.1.json")) == 0
