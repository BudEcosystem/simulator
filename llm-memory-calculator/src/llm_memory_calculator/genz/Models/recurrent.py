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

The shapes come from ``state_memory.recurrent_state_shape`` and the bytes per element from the
same module: the recurrent matrix is held at the state dtype (fp32 unless the config says
otherwise, as vLLM and llama.cpp hold it), the conv buffer at the activation precision. So the
graph and the memory calculator agree on a layer's state in shape and in bytes.
"""

from math import ceil

from llm_memory_calculator.genz.Models import ModelConfig, ResidencyInfo, OpType, CollectiveType
from llm_memory_calculator.genz.parallelism import ParallelismConfig

#: The state operator's name per bytes-per-element of the recurrent matrix. The graph is saved
#: as rows of (name, dims), so the dtype travels in the name and analysis reads it back.
STATE_OP_BY_BYTES = {4: "Rec State", 2: "Rec State16"}
CONV_OP = "Rec Conv"
RECURRENT_OPS = ("Rec Proj", "Rec AR", CONV_OP) + tuple(STATE_OP_BY_BYTES.values())


def op_state_bytes(layer_name):
    """Bytes per element of the state an operator holds: a number for the recurrent matrix,
    ``"act"`` for the conv buffer (activation precision), None for any other operator."""
    for nbytes, name in STATE_OP_BY_BYTES.items():
        if layer_name == name:
            return nbytes
    if layer_name == CONV_OP:
        return "act"
    return None


def _state_op_name(shape):
    return STATE_OP_BY_BYTES.get(int(shape.get("state_bytes") or 4), STATE_OP_BY_BYTES[4])


def _proj(lc, model_config, parallelism_config, m, decode):
    tp = parallelism_config.tensor_parallel
    D = model_config.hidden_size
    params = int(lc.recurrent_params or 0)
    residency_w = ResidencyInfo.AC_onchip if decode else ResidencyInfo.All_offchip
    if params <= 0:
        return []
    width = max(int(ceil(params / D / tp)), 1)
    return [["Rec Proj", width, m, D, 1, 1, residency_w, OpType.GEMM]]


def _state(lc, parallelism_config, m, sequences, decode):
    """The state ops for ``sequences`` sequences advancing ``m`` tokens each."""
    tp = parallelism_config.tensor_parallel
    residency_s = ResidencyInfo.AC_onchip if decode else ResidencyInfo.A_onchip
    # Not the causal-prefill kind: a scan touches the whole state once per token,
    # it does not attend to a triangle of earlier tokens.
    attend = OpType.Attend
    shape = lc.recurrent_state or {}
    heads = int(shape.get("heads") or 0)
    kd = int(shape.get("key_dim") or 0)
    vd = int(shape.get("value_dim") or 0)
    layers = []
    if heads and kd and vd:
        h = max(int(ceil(heads / tp)), 1) * sequences
        layers += [[_state_op_name(shape), h, m, kd, vd, h, residency_s, attend]]
    conv = int(shape.get("conv") or 0) + int(shape.get("extra") or 0)
    if conv > 0:
        c = max(int(ceil(conv / tp)), 1)
        layers += [[CONV_OP, sequences, m, 1, c, sequences, residency_s, attend]]
    return layers


def _sync(model_config, parallelism_config, m):
    tp = parallelism_config.tensor_parallel
    if tp > 1:
        return [["Rec AR", m, model_config.hidden_size, 1, 1, tp, CollectiveType.AllReduce, OpType.Sync]]
    return []


def recurrent_prefill(lc, model_config: ModelConfig, parallelism_config: ParallelismConfig,
                      input_sequence_length: int):
    m = max(input_sequence_length // parallelism_config.sequence_parallel, 1)
    return (_proj(lc, model_config, parallelism_config, m, decode=False)
            + _state(lc, parallelism_config, m, 1, decode=False)
            + _sync(model_config, parallelism_config, m))


def recurrent_decode(lc, model_config: ModelConfig, parallelism_config: ParallelismConfig):
    return (_proj(lc, model_config, parallelism_config, 1, decode=True)
            + _state(lc, parallelism_config, 1, 1, decode=True)
            + _sync(model_config, parallelism_config, 1))


def decode_sequences(decode_kv_sizes):
    """Sequences a chunked step decodes, by the same reading the chunk size uses."""
    n = 0
    for kv in decode_kv_sizes:
        if isinstance(kv, tuple) and len(kv) == 4:
            n += kv[1]
        elif isinstance(kv, tuple) and len(kv) == 2:
            n += kv[0]
        else:
            n += 1
    return n


def recurrent_chunked(lc, model_config: ModelConfig, parallelism_config: ParallelismConfig,
                      chunk_size: int, prefill_kv_sizes, decode_kv_sizes):
    """A chunked step: the projections over every token of the chunk, and ONE state per
    sequence in it -- each prefill request advancing its chunk of tokens, each decoding
    sequence one token. The state does not depend on how much each has already cached."""
    layers = _proj(lc, model_config, parallelism_config, chunk_size, decode=False)
    for _, tokens in prefill_kv_sizes:
        layers += _state(lc, parallelism_config, max(int(tokens), 1), 1, decode=False)
    n_decode = decode_sequences(decode_kv_sizes)
    if n_decode:
        layers += _state(lc, parallelism_config, 1, n_decode, decode=True)
    return layers + _sync(model_config, parallelism_config, chunk_size)
