from llm_memory_calculator.genz.Models import ModelConfig, ResidencyInfo, OpType, CollectiveType
from llm_memory_calculator.genz.parallelism import ParallelismConfig
from math import ceil

def mha_flash_attention_prefill(model_config:ModelConfig, parallelism_config:ParallelismConfig, input_sequence_length:int):
    H = model_config.num_attention_heads
    Hkv = model_config.num_key_value_heads
    D = model_config.hidden_size
    Dq = model_config.head_dim

    tp = parallelism_config.tensor_parallel * parallelism_config.expert_parallel
    sp = parallelism_config.sequence_parallel
    per_node_H = max(ceil(H / tp), 1)
    per_node_Hkv = max(ceil(Hkv / tp), 1)

    # MLA (Multi-head Latent Attention) is implemented in mla_attention_prefill/decode below.
    # Reference: https://arxiv.org/pdf/2405.04434

    ## [Batch/dp, Seq/sp, Dmodel] * [2, Dmodel, Dq, Hkv/tp] + [Dmodel, Dq, Head/tp]= [Batch/dp, Seq/sp, 3, Dq, Head/tp]
    QKV =           [["QKV", (per_node_H*Dq + 2*per_node_Hkv*Dq), input_sequence_length//sp, D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]

    ## [Batch/dp, Seq, Dq, Head/tp] * [Batch/dp, Seq/sp, Dq, Head/tp] = [Batch/dp, Seq, Seq/sp, Head/tp]
    logit =         [["Logit",per_node_H, input_sequence_length, input_sequence_length//sp, Dq, per_node_Hkv, ResidencyInfo.C_onchip, OpType.Logit]]

    ## [Batch/dp, Seq, Seq/sp, Head/tp] * [Batch/dp, Seq/sp, Dq, Head/tp] = [Batch/dp, Seq, Dq, Head/tp]
    attend =        [["Attend",per_node_H, input_sequence_length, input_sequence_length//sp, Dq, per_node_Hkv, ResidencyInfo.A_onchip, OpType.Attend]]

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

    tp = parallelism_config.tensor_parallel * parallelism_config.expert_parallel
    sp = parallelism_config.sequence_parallel
    per_node_H = max(ceil(H / tp), 1)
    per_node_Hkv = max(ceil(Hkv / tp), 1)

    # Local attention: each query token attends to at most sliding_window KV tokens
    kv_len = min(sw, input_sequence_length)

    QKV =           [["QKV", (per_node_H*Dq + 2*per_node_Hkv*Dq), input_sequence_length//sp, D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
    logit =         [["Logit",per_node_H, input_sequence_length, kv_len//sp, Dq, per_node_Hkv, ResidencyInfo.C_onchip, OpType.Logit]]
    attend =        [["Attend",per_node_H, input_sequence_length, kv_len//sp, Dq, per_node_Hkv, ResidencyInfo.A_onchip, OpType.Attend]]
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

    tp = parallelism_config.tensor_parallel * parallelism_config.expert_parallel
    sp = parallelism_config.sequence_parallel

    per_node_H = max(ceil(H / tp), 1)
    per_node_Hkv = max(ceil(Hkv / tp), 1)

    # Local attention: KV cache capped at sliding_window
    kv_len = min(sw, input_sequence_length)

    query =         [["QKV", (per_node_H*Dq + 2*per_node_Hkv*Dq), 1, D, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]
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

    tp = parallelism_config.tensor_parallel * parallelism_config.expert_parallel
    sp = parallelism_config.sequence_parallel
    dp = parallelism_config.data_parallel

    per_node_H = max(ceil(H / tp), 1)
    per_node_Hkv = max(ceil(Hkv / tp), 1)

    query =         [["QKV", (per_node_H*Dq + 2*per_node_Hkv*Dq), 1, D, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]
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

    tp = parallelism_config.tensor_parallel * parallelism_config.expert_parallel
    sp = parallelism_config.sequence_parallel
    dp = parallelism_config.data_parallel

    per_node_H = max(ceil(H / tp), 1)
    per_node_Hkv = max(ceil(Hkv / tp), 1)


    layers = []
    query =      [["QKV", (per_node_H*Dq + 2*per_node_Hkv*Dq), chunk_size, D, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]
    layers += query

    ## Prefill LA layers
    for kv_size in prefill_kv_sizes:
        layers +=    [["Logit Pre",per_node_H, kv_size[1], kv_size[0]+kv_size[1], Dq, per_node_Hkv, ResidencyInfo.C_onchip, OpType.Logit]]
        layers +=    [["Attend Pre",per_node_H, kv_size[1], kv_size[0]+kv_size[1], Dq, per_node_Hkv, ResidencyInfo.A_onchip, OpType.Attend]]

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
    tp = parallelism_config.tensor_parallel * parallelism_config.expert_parallel
    sp = parallelism_config.sequence_parallel

    layers = [["Attn Linear", D, input_sequence_length//sp, D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]
    if tp > 1:
        layers += [["Attn Linear AR", input_sequence_length//sp, D, 1, 1, tp, CollectiveType.AllReduce, OpType.Sync]]
    return layers


def linear_attention_decode(model_config:ModelConfig, parallelism_config:ParallelismConfig, input_sequence_length:int, output_gen_tokens:int):
    """Linear replacement for attention (DeciLM-style) in decode phase."""
    D = model_config.hidden_size
    tp = parallelism_config.tensor_parallel * parallelism_config.expert_parallel

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

    tp = parallelism_config.tensor_parallel * parallelism_config.expert_parallel
    sp = parallelism_config.sequence_parallel
    per_node_H = max(ceil(H / tp), 1)

    layers = []

    # Q down-projection: [B, S, D] -> [B, S, q_lora_rank]
    layers += [["Q Down", q_lora_rank, input_sequence_length//sp, D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]

    # Q up-projection: [B, S, q_lora_rank] -> [B, S, H*(qk_nope_head_dim + qk_rope_head_dim)]
    q_out_dim = per_node_H * (qk_nope_head_dim + qk_rope_head_dim)
    layers += [["Q Up", q_out_dim, input_sequence_length//sp, q_lora_rank, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]

    # KV compression: [B, S, D] -> [B, S, kv_lora_rank + qk_rope_head_dim]
    kv_compressed_dim = kv_lora_rank + qk_rope_head_dim
    layers += [["KV Compress", kv_compressed_dim, input_sequence_length//sp, D, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]

    # KV up-projection: [B, S, kv_lora_rank] -> [B, S, H*(qk_nope_head_dim + v_head_dim)]
    kv_out_dim = per_node_H * (qk_nope_head_dim + v_head_dim)
    layers += [["KV Up", kv_out_dim, input_sequence_length//sp, kv_lora_rank, 1, 1, ResidencyInfo.All_offchip, OpType.GEMM]]

    # Logit: Q * K^T
    Dq = qk_nope_head_dim + qk_rope_head_dim
    layers += [["Logit", per_node_H, input_sequence_length, input_sequence_length//sp, Dq, 1, ResidencyInfo.C_onchip, OpType.Logit]]

    # Attend: softmax(QK^T) * V
    layers += [["Attend", per_node_H, input_sequence_length, input_sequence_length//sp, v_head_dim, 1, ResidencyInfo.A_onchip, OpType.Attend]]

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

    tp = parallelism_config.tensor_parallel * parallelism_config.expert_parallel
    sp = parallelism_config.sequence_parallel
    per_node_H = max(ceil(H / tp), 1)

    layers = []

    # Q down-projection + up-projection
    layers += [["Q Down", q_lora_rank, 1, D, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]
    q_out_dim = per_node_H * (qk_nope_head_dim + qk_rope_head_dim)
    layers += [["Q Up", q_out_dim, 1, q_lora_rank, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]

    # KV compression for the new token
    kv_compressed_dim = kv_lora_rank + qk_rope_head_dim
    layers += [["KV Compress", kv_compressed_dim, 1, D, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]

    # KV up-projection (applied to cached compressed KV for attention computation)
    kv_out_dim = per_node_H * (qk_nope_head_dim + v_head_dim)
    layers += [["KV Up", kv_out_dim, input_sequence_length//sp, kv_lora_rank, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]

    # Logit and Attend over prefill context
    Dq = qk_nope_head_dim + qk_rope_head_dim
    layers += [["Logit Pre", per_node_H, 1, input_sequence_length//sp, Dq, 1, ResidencyInfo.AC_onchip, OpType.Logit_BM_PREFILL]]
    layers += [["Attend Pre", per_node_H, 1, input_sequence_length//sp, v_head_dim, 1, ResidencyInfo.AC_onchip, OpType.Attend_BM_PREFILL]]

    # Logit and Attend over generated tokens
    layers += [["Logit Suf", per_node_H, 1, output_gen_tokens, Dq, 1, ResidencyInfo.AC_onchip, OpType.Logit]]
    layers += [["Attend Suf", per_node_H, 1, output_gen_tokens, v_head_dim, 1, ResidencyInfo.AC_onchip, OpType.Attend]]

    # Output projection
    layers += [["Out Proj", D, 1, per_node_H * v_head_dim, 1, 1, ResidencyInfo.AC_onchip, OpType.GEMM]]

    if tp > 1:
        layers += [["MLA AR", 1, D, 1, 1, tp, CollectiveType.AllReduce, OpType.Sync]]

    return layers

