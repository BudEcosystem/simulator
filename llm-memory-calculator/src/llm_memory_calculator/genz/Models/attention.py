from llm_memory_calculator.genz.Models import ModelConfig, ResidencyInfo, OpType, CollectiveType
from llm_memory_calculator.genz.parallelism import ParallelismConfig
from math import ceil

def qkv_rows(model_config: ModelConfig, per_node_H: int, per_node_Hkv: int, Dq: int) -> int:
    """The fused Q / K / V projection's output rows: the query heads, doubled where the attention
    gates its output from a second query-width projection (``attention_output_gate``: Qwen3-Next
    and Qwen3.5 build `q_proj` at 2 x heads x head_dim), and the K and V heads."""
    gate = 2 if getattr(model_config, "attention_output_gate", False) else 1
    return per_node_H * Dq * gate + 2 * per_node_Hkv * Dq


def mha_flash_attention_prefill(model_config:ModelConfig, parallelism_config:ParallelismConfig, input_sequence_length:int):
    H = model_config.num_attention_heads
    Hkv = model_config.num_key_value_heads
    D = model_config.hidden_size
    Dq = model_config.head_dim

    tp = parallelism_config.tensor_parallel
    sp = parallelism_config.sequence_parallel
    per_node_H = max(ceil(H / tp), 1)
    per_node_Hkv = max(ceil(Hkv / tp), 1)

    # MLA (Multi-head Latent Attention) is implemented in mla_attention_prefill/decode below.
    # Reference: https://arxiv.org/pdf/2405.04434

    ## [Batch/dp, Seq/sp, Dmodel] * [2, Dmodel, Dq, Hkv/tp] + [Dmodel, Dq, Head/tp]= [Batch/dp, Seq/sp, 3, Dq, Head/tp]
    QKV =           [["QKV", qkv_rows(model_config, per_node_H, per_node_Hkv, Dq), input_sequence_length//sp, D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]

    ## [Batch/dp, Seq, Dq, Head/tp] * [Batch/dp, Seq/sp, Dq, Head/tp] = [Batch/dp, Seq, Seq/sp, Head/tp]
    logit =         [["Logit",per_node_H, input_sequence_length, input_sequence_length//sp, Dq, per_node_Hkv, ResidencyInfo.C_onchip, OpType.Logit_Causal_PREFILL]]

    ## [Batch/dp, Seq, Seq/sp, Head/tp] * [Batch/dp, Seq/sp, Dq, Head/tp] = [Batch/dp, Seq, Dq, Head/tp]
    attend =        [["Attend",per_node_H, input_sequence_length, input_sequence_length//sp, Dq, per_node_Hkv, ResidencyInfo.A_onchip, OpType.Attend_Causal_PREFILL]]

    ## [Batch/dp, Seq, Dq, Head/tp] * [Dq, Head/tp,  Dmodel] = [Batch/dp, Seq, Dmodel]
    output =        [["Out Proj", D, input_sequence_length//sp, (per_node_H) * Dq, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]

    if tp > 1:
        sync =          [["MHA AR", input_sequence_length//sp, D, 1, 1, tp, CollectiveType.AllReduce, OpType.Sync]]
    else:
        sync = []
    return QKV + logit + attend + output + sync

def mha_flash_attention_prefill_local(model_config:ModelConfig, parallelism_config:ParallelismConfig, input_sequence_length:int):
    """Local (sliding window) attention for prefill. KV context is capped at sliding_window."""
    H = model_config.num_attention_heads
    Hkv = model_config.num_key_value_heads
    D = model_config.hidden_size
    Dq = model_config.head_dim
    sw = model_config.sliding_window or input_sequence_length

    tp = parallelism_config.tensor_parallel
    sp = parallelism_config.sequence_parallel
    per_node_H = max(ceil(H / tp), 1)
    per_node_Hkv = max(ceil(Hkv / tp), 1)

    # Local attention: each query token attends to at most sliding_window KV tokens
    kv_len = min(sw, input_sequence_length)

    QKV =           [["QKV", qkv_rows(model_config, per_node_H, per_node_Hkv, Dq), input_sequence_length//sp, D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
    logit =         [["Logit",per_node_H, input_sequence_length, kv_len//sp, Dq, per_node_Hkv, ResidencyInfo.C_onchip, OpType.Logit_Causal_PREFILL]]
    attend =        [["Attend",per_node_H, input_sequence_length, kv_len//sp, Dq, per_node_Hkv, ResidencyInfo.A_onchip, OpType.Attend_Causal_PREFILL]]
    output =        [["Out Proj", D, input_sequence_length//sp, (per_node_H) * Dq, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]

    if tp > 1:
        sync =          [["MHA AR", input_sequence_length//sp, D, 1, 1, tp, CollectiveType.AllReduce, OpType.Sync]]
    else:
        sync = []
    return QKV + logit + attend + output + sync

def mha_flash_attention_decode_local(model_config:ModelConfig, parallelism_config:ParallelismConfig, input_sequence_length:int, output_gen_tokens:int):
    """Local (sliding window) attention for decode. KV context is capped at sliding_window."""
    H = model_config.num_attention_heads
    Hkv = model_config.num_key_value_heads
    D = model_config.hidden_size
    Dq = model_config.head_dim
    sw = model_config.sliding_window or input_sequence_length

    tp = parallelism_config.tensor_parallel
    sp = parallelism_config.sequence_parallel

    per_node_H = max(ceil(H / tp), 1)
    per_node_Hkv = max(ceil(Hkv / tp), 1)

    # Local attention: KV cache capped at sliding_window
    kv_len = min(sw, input_sequence_length)

    query =         [["QKV", qkv_rows(model_config, per_node_H, per_node_Hkv, Dq), 1, D, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]
    logit_pre =     [["Logit Pre",per_node_H, 1, kv_len//sp, Dq, per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Logit_BM_PREFILL]]
    attend_pre =    [["Attend Pre",per_node_H, 1, kv_len//sp, Dq, per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Attend_BM_PREFILL]]
    logit_suf =     [["Logit Suf",per_node_H, 1, min(output_gen_tokens, sw), Dq, per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Logit]]
    attend_suf =    [["Attend Suf",per_node_H, 1, min(output_gen_tokens, sw), Dq, per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Attend]]
    output =        [["Out Proj",D, 1, (per_node_H) * Dq, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]
    if tp > 1:
        sync =          [["MHA AR",1, D, 1, 1, tp, CollectiveType.AllReduce, OpType.Sync]]
    else:
        sync = []

    return query + logit_pre + logit_suf + attend_pre + attend_suf + output + sync

def mha_flash_attention_decode(model_config:ModelConfig, parallelism_config:ParallelismConfig, input_sequence_length:int, output_gen_tokens:int):
    H = model_config.num_attention_heads
    Hkv = model_config.num_key_value_heads
    D = model_config.hidden_size
    Dq = model_config.head_dim

    tp = parallelism_config.tensor_parallel
    sp = parallelism_config.sequence_parallel
    dp = parallelism_config.data_parallel

    per_node_H = max(ceil(H / tp), 1)
    per_node_Hkv = max(ceil(Hkv / tp), 1)

    query =         [["QKV", qkv_rows(model_config, per_node_H, per_node_Hkv, Dq), 1, D, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]
    logit_pre =     [["Logit Pre",per_node_H, 1, input_sequence_length//sp, Dq, per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Logit_BM_PREFILL]]
    attend_pre =    [["Attend Pre",per_node_H, 1, input_sequence_length//sp, Dq, per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Attend_BM_PREFILL]]
    logit_suf =     [["Logit Suf",per_node_H, 1, output_gen_tokens, Dq, per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Logit]]
    attend_suf =    [["Attend Suf",per_node_H, 1, output_gen_tokens, Dq, per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Attend]]
    output =        [["Out Proj",D, 1, (per_node_H) * Dq, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]
    if tp > 1:
        sync =          [["MHA AR",1, D, 1, 1, tp, CollectiveType.AllReduce, OpType.Sync]]
    else:
        sync = []

    return query + logit_pre + logit_suf + attend_pre + attend_suf + output + sync



def window_kv_sizes(prefill_kv_sizes, decode_kv_sizes, window):
    """A chunked step's KV sizes as a windowed layer sees them: no context beyond ``window``.

    A prefill chunk of ``cur`` tokens after ``past`` cached ones reads at most the last
    ``window`` of them; a decoding sequence reads at most ``window`` cached tokens. With no
    window the sizes are returned unchanged.
    """
    if not window:
        return prefill_kv_sizes, decode_kv_sizes
    w = int(window)
    prefill = [(min(past, w), cur) for past, cur in prefill_kv_sizes]
    decode = []
    for kv in decode_kv_sizes:
        if isinstance(kv, tuple) and len(kv) == 4:
            decode.append((kv[0], kv[1], min(kv[2], w), min(kv[3], w)))
        elif isinstance(kv, tuple) and len(kv) == 2:
            decode.append((kv[0], min(kv[1], w)))
        else:
            decode.append(min(kv, w))
    return prefill, decode


def mla_indexer_ops(model_config: ModelConfig, tokens: int, residency):
    """DeepSeek Sparse Attention's lightning indexer beside MLA (V3.2's `index_n_heads` x
    `index_head_dim`): its query up-projection from the query latent, its key projection and its
    per-head weights, as transformers' `DeepseekV32Indexer` builds them (`wq_b`, `wk`,
    `weights_proj`). None for an MLA without one. Leaving it out sized V3.2's graph 852M
    parameters below the checkpoint."""
    n = getattr(model_config, "index_n_heads", None)
    d = getattr(model_config, "index_head_dim", None)
    if not n or not d:
        return []
    D = model_config.hidden_size
    q_in = model_config.q_lora_rank or D
    return [["Index Q", n * d, tokens, q_in, 1, 1, residency, OpType.GEMM],
            ["Index K", d, tokens, D, 1, 1, residency, OpType.GEMM],
            ["Index W", n, tokens, D, 1, 1, residency, OpType.GEMM]]


def mla_latent_op(model_config: ModelConfig, tokens: int, sequences: int = 1, op_type=OpType.Logit):
    """What MLA caches: one latent of `kv_lora_rank + qk_rope_head_dim` per token, read once a step
    for all heads (vLLM's and FlashMLA's absorbed attention). A score operator without queries
    (no compute), so the graph's `KV Cache` is the latent; the expanded score operators beside it
    (named "(expanded)") give the compute, and their per-head K and V are not a cache. Sized from
    those, DeepSeek-V3's cache was 320 elements a token, not 576."""
    width = (model_config.kv_lora_rank or 0) + (model_config.qk_rope_head_dim or 0)
    if not width or tokens <= 0:
        return []
    return [["MLA Latent", max(int(sequences), 1), 0, int(tokens), width, max(int(sequences), 1),
             ResidencyInfo.AC_onchip, op_type]]


def mla_attention_chunked(model_config:ModelConfig, parallelism_config:ParallelismConfig,
                          chunk_size: int, prefill_kv_sizes, decode_kv_sizes):
    """Multi-head Latent Attention over a chunked step: MLA's projections over the chunk's
    tokens, and its single shared latent head (``Hkv = 1``) read over each request's cache,
    as `mla_attention_prefill` / `mla_attention_decode` read it."""
    H = model_config.num_attention_heads
    D = model_config.hidden_size
    kv_lora_rank = model_config.kv_lora_rank
    q_lora_rank = model_config.q_lora_rank
    rope = model_config.qk_rope_head_dim or 0
    nope = model_config.qk_nope_head_dim or 0
    v_head_dim = model_config.v_head_dim

    tp = parallelism_config.tensor_parallel
    sp = parallelism_config.sequence_parallel
    per_node_H = max(ceil(H / tp), 1)
    Dq = nope + rope

    layers = []
    q_out_dim = per_node_H * Dq
    if q_lora_rank:
        layers += [["Q Down", q_lora_rank, chunk_size, D, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]
        layers += [["Q Up", q_out_dim, chunk_size, q_lora_rank, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]
    else:
        layers += [["Q Proj", q_out_dim, chunk_size, D, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]
    layers += [["KV Compress", kv_lora_rank + rope, chunk_size, D, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]
    layers += mla_indexer_ops(model_config, chunk_size, ResidencyInfo.AC_onchip)
    kv_out_dim = per_node_H * (nope + v_head_dim)
    context = sum(p + c for p, c in prefill_kv_sizes)
    for kv in decode_kv_sizes:
        if isinstance(kv, tuple) and len(kv) == 4:
            context += kv[2] + kv[3]
        elif isinstance(kv, tuple) and len(kv) == 2:
            context += kv[0] * kv[1]
        else:
            context += kv
    layers += [["KV Up", kv_out_dim, max(context // sp, 1), kv_lora_rank, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]

    for past, cur in prefill_kv_sizes:
        layers += [["Logit Pre (expanded)", per_node_H, cur, past + cur, Dq, 1, ResidencyInfo.C_onchip, OpType.Logit_Causal_PREFILL]]
        layers += [["Attend Pre (expanded)", per_node_H, cur, past + cur, v_head_dim, 1, ResidencyInfo.A_onchip, OpType.Attend_Causal_PREFILL]]
    for kv in decode_kv_sizes:
        if isinstance(kv, tuple) and len(kv) == 4:
            pb, db, pctx, dctx = kv
            layers += [["Logit Pre (expanded)", pb * per_node_H, 1, pctx // sp, Dq, 1, ResidencyInfo.AC_onchip, OpType.Logit]]
            layers += [["Attend Pre (expanded)", pb * per_node_H, 1, pctx // sp, v_head_dim, 1, ResidencyInfo.AC_onchip, OpType.Attend]]
            layers += [["Logit Suf (expanded)", db * per_node_H, 1, dctx, Dq, 1, ResidencyInfo.AC_onchip, OpType.Logit]]
            layers += [["Attend Suf (expanded)", db * per_node_H, 1, dctx, v_head_dim, 1, ResidencyInfo.AC_onchip, OpType.Attend]]
        elif isinstance(kv, tuple) and len(kv) == 2:
            nb, ctx = kv
            layers += [["Logit Dec (expanded)", nb * per_node_H, 1, ctx, Dq, nb, ResidencyInfo.AC_onchip, OpType.Logit]]
            layers += [["Attend Dec (expanded)", nb * per_node_H, 1, ctx, v_head_dim, nb, ResidencyInfo.AC_onchip, OpType.Attend]]
        else:
            layers += [["Logit Dec (expanded)", per_node_H, 1, kv, Dq, 1, ResidencyInfo.AC_onchip, OpType.Logit]]
            layers += [["Attend Dec (expanded)", per_node_H, 1, kv, v_head_dim, 1, ResidencyInfo.AC_onchip, OpType.Attend]]
    # The cache MLA keeps: one latent per token of every request's context (mla_latent_op)
    for past, cur in prefill_kv_sizes:
        layers += mla_latent_op(model_config, past + cur)
    for kv in decode_kv_sizes:
        if isinstance(kv, tuple) and len(kv) == 4:
            pb, db, pctx, dctx = kv
            layers += mla_latent_op(model_config, pctx // sp, pb) + mla_latent_op(model_config, dctx, db)
        elif isinstance(kv, tuple) and len(kv) == 2:
            layers += mla_latent_op(model_config, kv[1], kv[0])
        else:
            layers += mla_latent_op(model_config, kv)
    layers += [["Out Proj", D, chunk_size, per_node_H * v_head_dim, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]
    if tp > 1:
        layers += [["MLA AR", chunk_size, D, 1, 1, tp, CollectiveType.AllReduce, OpType.Sync]]
    return layers


def mha_flash_attention_chunked(model_config:ModelConfig, parallelism_config:ParallelismConfig,
                                chunk_size: int, prefill_kv_sizes: list[int,int], decode_kv_sizes: list[int]):
    '''
        Generates a list of operators for multi-head attention (MHA) with flash attention,
        chunked processing, and parallelism configurations.
        Args:
            model_config (ModelConfig): Configuration object containing model parameters such as
                                        number of attention heads, key-value heads, hidden size, and head dimension.
            parallelism_config (ParallelismConfig): Configuration object containing parallelism parameters
                                                    such as tensor parallelism, expert parallelism, sequence parallelism, and data parallelism.
            chunk_size (int): Maximum chunk size of the values to be processed.
            prefill_kv_sizes (int): List of sizes of the prefill
                                    First value of tuple is the tokens processed till now and second value is the tokens processed in the current chunk.
            decode_kv_sizes (list[int]): List of sizes for the key-value pairs during the decode stage.

            The call to this function should handle the prefill_kv_sizes calculation.
        Returns:
            list: A list of layers with their respective configurations for the MHA with flash attention.

    '''
    H = model_config.num_attention_heads
    Hkv = model_config.num_key_value_heads
    D = model_config.hidden_size
    Dq = model_config.head_dim

    tp = parallelism_config.tensor_parallel
    sp = parallelism_config.sequence_parallel
    dp = parallelism_config.data_parallel

    per_node_H = max(ceil(H / tp), 1)
    per_node_Hkv = max(ceil(Hkv / tp), 1)


    layers = []
    query =      [["QKV", qkv_rows(model_config, per_node_H, per_node_Hkv, Dq), chunk_size, D, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]
    layers += query

    ## Prefill LA layers (causal: chunk of kv_size[1] queries attends causally to kv_size[0]+kv_size[1] keys)
    for kv_size in prefill_kv_sizes:
        layers +=    [["Logit Pre",per_node_H, kv_size[1], kv_size[0]+kv_size[1], Dq, per_node_Hkv, ResidencyInfo.C_onchip, OpType.Logit_Causal_PREFILL]]
        layers +=    [["Attend Pre",per_node_H, kv_size[1], kv_size[0]+kv_size[1], Dq, per_node_Hkv, ResidencyInfo.A_onchip, OpType.Attend_Causal_PREFILL]]

    ## Decode LA layers
    for kv_size in decode_kv_sizes:
        if isinstance(kv_size, tuple) and len(kv_size) == 4:
            prefill_num_beams = kv_size[0]
            decode_num_beams = kv_size[1]
            prefill_context = kv_size[2]//sp
            decode_context = kv_size[3]
            # layers +=    [["Logit Dec",per_node_H, 1, kv_size[1], Dq, per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Logit]]
            # layers +=    [["Attend Dec",per_node_H, 1, kv_size[1], Dq, per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Attend]]
            layers +=     [["Logit Pre",(prefill_num_beams*per_node_H), 1, prefill_context, Dq, per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Logit]]
            layers +=    [["Attend Pre",(prefill_num_beams*per_node_H), 1, prefill_context, Dq, per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Attend]]
            layers +=     [["Logit Suf",(decode_num_beams*per_node_H), 1, decode_context, Dq, per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Logit]]
            layers +=    [["Attend Suf",(decode_num_beams*per_node_H), 1, decode_context, Dq, per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Attend]]
        elif isinstance(kv_size, tuple) and len(kv_size) == 2:
            num_batches = kv_size[0]
            past_context = kv_size[1]
            layers +=     [["Logit Dec",num_batches*per_node_H, 1, past_context, Dq, num_batches*per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Logit]]
            layers +=    [["Attend Dec",num_batches*per_node_H, 1, past_context, Dq, num_batches*per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Attend]]
        else:
            layers +=     [["Logit Dec",per_node_H, 1, kv_size, Dq, per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Logit]]
            layers +=    [["Attend Dec",per_node_H, 1, kv_size, Dq, per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Attend]]


    layers +=        [["Out Proj",D, chunk_size, (per_node_H) * Dq, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]
    if tp > 1:
        layers +=          [["MHA AR",chunk_size, D, 1, 1, tp, CollectiveType.AllReduce, OpType.Sync]]
    else:
        sync = []

    return layers


def linear_attention_prefill(model_config:ModelConfig, parallelism_config:ParallelismConfig, input_sequence_length:int):
    """Linear replacement for attention (DeciLM-style). Single GEMM [D, S, D]."""
    D = model_config.hidden_size
    tp = parallelism_config.tensor_parallel
    sp = parallelism_config.sequence_parallel

    layers = [["Attn Linear", D, input_sequence_length//sp, D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
    if tp > 1:
        layers += [["Attn Linear AR", input_sequence_length//sp, D, 1, 1, tp, CollectiveType.AllReduce, OpType.Sync]]
    return layers


def linear_attention_decode(model_config:ModelConfig, parallelism_config:ParallelismConfig, input_sequence_length:int, output_gen_tokens:int):
    """Linear replacement for attention (DeciLM-style) in decode phase."""
    D = model_config.hidden_size
    tp = parallelism_config.tensor_parallel

    layers = [["Attn Linear", D, 1, D, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]
    if tp > 1:
        layers += [["Attn Linear AR", 1, D, 1, 1, tp, CollectiveType.AllReduce, OpType.Sync]]
    return layers


def mla_attention_prefill(model_config:ModelConfig, parallelism_config:ParallelismConfig, input_sequence_length:int):
    """Multi-head Latent Attention (MLA) for prefill phase.
    Used by DeepSeek V2/V3. Compresses KV cache via low-rank projections.
    Reference: https://arxiv.org/pdf/2405.04434
    """
    H = model_config.num_attention_heads
    D = model_config.hidden_size
    kv_lora_rank = model_config.kv_lora_rank
    q_lora_rank = model_config.q_lora_rank
    qk_rope_head_dim = model_config.qk_rope_head_dim
    qk_nope_head_dim = model_config.qk_nope_head_dim
    v_head_dim = model_config.v_head_dim

    tp = parallelism_config.tensor_parallel
    sp = parallelism_config.sequence_parallel
    per_node_H = max(ceil(H / tp), 1)

    layers = []

    # Query projection. A low-rank down/up pair when q_lora_rank is set; a single
    # full-rank projection when the config ships it null (DeepSeek-V2-Lite,
    # Kimi Linear) -- building the pair from None put a NaN in the operator dims.
    q_out_dim = per_node_H * (qk_nope_head_dim + qk_rope_head_dim)
    if q_lora_rank:
        # Q down-projection: [B, S, D] -> [B, S, q_lora_rank]
        layers += [["Q Down", q_lora_rank, input_sequence_length//sp, D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
        # Q up-projection: [B, S, q_lora_rank] -> [B, S, H*(qk_nope_head_dim + qk_rope_head_dim)]
        layers += [["Q Up", q_out_dim, input_sequence_length//sp, q_lora_rank, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
    else:
        layers += [["Q Proj", q_out_dim, input_sequence_length//sp, D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]

    # KV compression: [B, S, D] -> [B, S, kv_lora_rank + qk_rope_head_dim]
    kv_compressed_dim = kv_lora_rank + qk_rope_head_dim
    layers += [["KV Compress", kv_compressed_dim, input_sequence_length//sp, D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
    layers += mla_indexer_ops(model_config, input_sequence_length//sp, ResidencyInfo.All_offchip)

    # KV up-projection: [B, S, kv_lora_rank] -> [B, S, H*(qk_nope_head_dim + v_head_dim)]
    kv_out_dim = per_node_H * (qk_nope_head_dim + v_head_dim)
    layers += [["KV Up", kv_out_dim, input_sequence_length//sp, kv_lora_rank, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]

    # Logit: Q * K^T
    Dq = qk_nope_head_dim + qk_rope_head_dim
    layers += [["Logit (expanded)", per_node_H, input_sequence_length, input_sequence_length//sp, Dq, 1, ResidencyInfo.C_onchip, OpType.Logit_Causal_PREFILL]]

    # Attend: softmax(QK^T) * V
    layers += [["Attend (expanded)", per_node_H, input_sequence_length, input_sequence_length//sp, v_head_dim, 1, ResidencyInfo.A_onchip, OpType.Attend_Causal_PREFILL]]

    # The cache MLA keeps: one latent per token (mla_latent_op)
    layers += mla_latent_op(model_config, input_sequence_length//sp)

    # Output projection: [B, S, H*v_head_dim] -> [B, S, D]
    layers += [["Out Proj", D, input_sequence_length//sp, per_node_H * v_head_dim, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]

    if tp > 1:
        layers += [["MLA AR", input_sequence_length//sp, D, 1, 1, tp, CollectiveType.AllReduce, OpType.Sync]]

    return layers


def mla_attention_decode(model_config:ModelConfig, parallelism_config:ParallelismConfig, input_sequence_length:int, output_gen_tokens:int):
    """Multi-head Latent Attention (MLA) for decode phase.
    KV cache stores compressed representations (kv_lora_rank + qk_rope_head_dim per token).
    """
    H = model_config.num_attention_heads
    D = model_config.hidden_size
    kv_lora_rank = model_config.kv_lora_rank
    q_lora_rank = model_config.q_lora_rank
    qk_rope_head_dim = model_config.qk_rope_head_dim
    qk_nope_head_dim = model_config.qk_nope_head_dim
    v_head_dim = model_config.v_head_dim

    tp = parallelism_config.tensor_parallel
    sp = parallelism_config.sequence_parallel
    per_node_H = max(ceil(H / tp), 1)

    layers = []

    # Query projection: the low-rank pair, or one full-rank projection when
    # q_lora_rank is null (see prefill).
    q_out_dim = per_node_H * (qk_nope_head_dim + qk_rope_head_dim)
    if q_lora_rank:
        layers += [["Q Down", q_lora_rank, 1, D, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]
        layers += [["Q Up", q_out_dim, 1, q_lora_rank, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]
    else:
        layers += [["Q Proj", q_out_dim, 1, D, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]

    # KV compression for the new token
    kv_compressed_dim = kv_lora_rank + qk_rope_head_dim
    layers += [["KV Compress", kv_compressed_dim, 1, D, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]
    layers += mla_indexer_ops(model_config, 1, ResidencyInfo.AC_onchip)

    # KV up-projection (applied to cached compressed KV for attention computation)
    kv_out_dim = per_node_H * (qk_nope_head_dim + v_head_dim)
    layers += [["KV Up", kv_out_dim, input_sequence_length//sp, kv_lora_rank, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]

    # Logit and Attend over prefill context
    Dq = qk_nope_head_dim + qk_rope_head_dim
    layers += [["Logit Pre (expanded)", per_node_H, 1, input_sequence_length//sp, Dq, 1, ResidencyInfo.AC_onchip, OpType.Logit_BM_PREFILL]]
    layers += [["Attend Pre (expanded)", per_node_H, 1, input_sequence_length//sp, v_head_dim, 1, ResidencyInfo.AC_onchip, OpType.Attend_BM_PREFILL]]

    # Logit and Attend over generated tokens
    layers += [["Logit Suf (expanded)", per_node_H, 1, output_gen_tokens, Dq, 1, ResidencyInfo.AC_onchip, OpType.Logit]]
    layers += [["Attend Suf (expanded)", per_node_H, 1, output_gen_tokens, v_head_dim, 1, ResidencyInfo.AC_onchip, OpType.Attend]]

    # The cache MLA keeps: one latent per token of the context (mla_latent_op), the prompt's
    # shared by every beam as the expanded "Pre" operators read it
    layers += mla_latent_op(model_config, input_sequence_length//sp, op_type=OpType.Logit_BM_PREFILL)
    layers += mla_latent_op(model_config, output_gen_tokens)

    # Output projection
    layers += [["Out Proj", D, 1, per_node_H * v_head_dim, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]

    if tp > 1:
        layers += [["MLA AR", 1, D, 1, 1, tp, CollectiveType.AllReduce, OpType.Sync]]

    return layers



#: The operators the cross-attention and encoder graphs emit, by what `get_runtime_breakdown`
#: books them as: projections (MHA, QKVO), scores and values (MHA, the attention kernels), and
#: the tensor-parallel all-reduce (MHA, collective).
CROSS_PROJECTION_OPS = ("Cross KV", "Cross Q", "Cross Out", "Enc QKV", "Enc Out Proj", "Enc Merge",
                        "Enc Proj", "Enc Embed")
CROSS_SCORE_OPS = ("Cross Logit", "Cross Attend", "Enc Logit", "Enc Attend")
CROSS_COLLECTIVE_OPS = ("Cross AR", "Enc MHA AR")


def cross_attention(model_config: ModelConfig, parallelism_config: ParallelismConfig,
                    query_tokens: int, reads, project_encoder: bool = False,
                    own_layer: bool = False):
    """A decoder layer's cross-attention (``model_config.cross_attention``,
    layer_plan.cross_attention_cache): its query and output projections over ``query_tokens``
    decoder tokens, and, for each sequence in ``reads`` -- ``(queries, encoder tokens if the
    config states none)`` --, its scores and values over that sequence's cached K / V of the
    encoder's output (the cache the decode graph reads every step). ``project_encoder`` runs the
    K / V projection of the encoder's output, once a request, at prefill; otherwise its weights
    are held at no time (a GEMM over no tokens), so every graph weighs the resident layer.

    An encoder-decoder's cross-attention sits in every decoder layer, beside its self-attention;
    a decoder-only stack's own cross-attention layers (Mllama, ``own_layers``) are layers of their
    own, graphed with ``own_layer``. Empty for a call that does not match the model's kind, and
    for any model without cross-attention."""
    cross = getattr(model_config, "cross_attention", None)
    if not cross or bool(cross.get("own_layers")) != bool(own_layer):
        return []
    H = model_config.num_attention_heads
    Hkv, Dq = cross["kv_heads"], cross["head_dim"]
    D = model_config.hidden_size
    encoder = getattr(model_config, "encoder", None) or {}
    De = encoder.get("projected_to") or encoder.get("hidden_size") or D
    tp = parallelism_config.tensor_parallel
    per_node_H = max(ceil(H / tp), 1)
    per_node_Hkv = max(ceil(Hkv / tp), 1)
    layers = []
    tokens = (cross.get("tokens") or max((t for _, t in reads), default=0)) if project_encoder else 0
    layers += [["Cross KV", 2 * per_node_Hkv * Dq, tokens, De, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
    layers += [["Cross Q", per_node_H * Dq, query_tokens, D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
    for queries, fallback in reads:
        kv = cross.get("tokens") or fallback
        layers += [["Cross Logit", per_node_H, queries, kv, Dq, per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Logit]]
        layers += [["Cross Attend", per_node_H, queries, kv, Dq, per_node_Hkv, ResidencyInfo.AC_onchip, OpType.Attend]]
    layers += [["Cross Out", D, query_tokens, per_node_H * Dq, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
    if tp > 1:
        layers += [["Cross AR", query_tokens, D, 1, 1, tp, CollectiveType.AllReduce, OpType.Sync]]
    return layers


def encoder_attention(encoder_config: ModelConfig, parallelism_config: ParallelismConfig, tokens: int):
    """One encoder layer's bidirectional self-attention over the encoder's ``tokens`` (every
    query sees every key: plain Logit / Attend, not the causal prefill kind)."""
    H = encoder_config.num_attention_heads
    Hkv = encoder_config.num_key_value_heads
    D = encoder_config.hidden_size
    Dq = encoder_config.head_dim
    tp = parallelism_config.tensor_parallel
    per_node_H = max(ceil(H / tp), 1)
    per_node_Hkv = max(ceil(Hkv / tp), 1)
    layers = [["Enc QKV", (per_node_H * Dq + 2 * per_node_Hkv * Dq), tokens, D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM],
              ["Enc Logit", per_node_H, tokens, tokens, Dq, per_node_Hkv, ResidencyInfo.C_onchip, OpType.Logit],
              ["Enc Attend", per_node_H, tokens, tokens, Dq, per_node_Hkv, ResidencyInfo.A_onchip, OpType.Attend],
              ["Enc Out Proj", D, tokens, per_node_H * Dq, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
    if tp > 1:
        layers += [["Enc MHA AR", tokens, D, 1, 1, tp, CollectiveType.AllReduce, OpType.Sync]]
    return layers
