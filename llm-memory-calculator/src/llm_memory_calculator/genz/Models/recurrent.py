"""Operators for a recurrent token mixer: Gated DeltaNet, Mamba-1/2, KDA, plain linear
attention, short convolution, RWKV, xLSTM, RG-LRU.

What every one of these shares, and what separates them from attention, is that the state a
layer carries forward is FIXED in size: it is read and rewritten once per token and does not
grow with the context. So the graph charges

* ``Rec Proj``  -- the layer's projections, as one GEMM over the mixer's own parameter count
  (``mixer_params.recurrent_mixer_params``): the FLOPs and the weight bytes a decode step streams.
* ``Rec State`` -- the recurrent matrix (``heads x key_dim x value_dim``), as an Attend over a
  "context" of ``key_dim`` rows. Its weight-side input is the state, so the summary table counts
  it with the KV cache (GenZ's "KV Cache" column has always carried Mamba's state too), it scales
  with the batch, and it does NOT scale with the sequence.
* ``Rec Conv``  -- the short convolution's ring buffer, the same way, when the mechanism has one.

The shapes come from ``state_memory.recurrent_state_shape``, the formula the memory calculator
uses, so the graph and the calculator cannot disagree about a layer's state.
"""

from math import ceil

from llm_memory_calculator.genz.Models import ModelConfig, ResidencyInfo, OpType, CollectiveType
from llm_memory_calculator.genz.parallelism import ParallelismConfig


def _ops(lc, model_config: ModelConfig, parallelism_config: ParallelismConfig, tokens: int,
         decode: bool):
    tp = parallelism_config.tensor_parallel
    sp = parallelism_config.sequence_parallel
    D = model_config.hidden_size
    m = 1 if decode else max(tokens // sp, 1)
    residency_w = ResidencyInfo.AC_onchip if decode else ResidencyInfo.All_offchip
    residency_s = ResidencyInfo.AC_onchip if decode else ResidencyInfo.A_onchip
    # Not the causal-prefill kind: a scan touches the whole state once per token,
    # it does not attend to a triangle of earlier tokens.
    attend = OpType.Attend

    shape = lc.recurrent_state or {}
    params = int(lc.recurrent_params or 0)
    layers = []
    if params > 0:
        width = max(int(ceil(params / D / tp)), 1)
        layers += [["Rec Proj", width, m, D, 1, 1, residency_w, OpType.GEMM]]
    heads = int(shape.get("heads") or 0)
    kd = int(shape.get("key_dim") or 0)
    vd = int(shape.get("value_dim") or 0)
    if heads and kd and vd:
        h = max(int(ceil(heads / tp)), 1)
        layers += [["Rec State", h, m, kd, vd, h, residency_s, attend]]
    conv = int(shape.get("conv") or 0) + int(shape.get("extra") or 0)
    if conv > 0:
        layers += [["Rec Conv", 1, m, 1, max(int(ceil(conv / tp)), 1), 1, residency_s, attend]]
    if tp > 1:
        layers += [["Rec AR", m, D, 1, 1, tp, CollectiveType.AllReduce, OpType.Sync]]
    return layers


def recurrent_prefill(lc, model_config: ModelConfig, parallelism_config: ParallelismConfig,
                      input_sequence_length: int):
    return _ops(lc, model_config, parallelism_config, input_sequence_length, decode=False)


def recurrent_decode(lc, model_config: ModelConfig, parallelism_config: ParallelismConfig):
    return _ops(lc, model_config, parallelism_config, 1, decode=True)
