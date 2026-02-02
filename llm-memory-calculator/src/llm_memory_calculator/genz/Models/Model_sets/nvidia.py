from ..default_models import ModelConfig, LayerConfig, get_all_model_configs
from ..model_quality import QualityMetricsCollection, MMLU, MATH, GSM8K,  IFEval,  GPQA, Hellaswag, TLDR, TriviaQA, BIG_Bench

##### Nvidia Models ########
# https://huggingface.co/nvidia/Nemotron-4-340B-Instruct/blob/main/model_config.yaml
nemotron_340b_config = ModelConfig(model='nvidia/Nemotron-4-340B-Instruct',
    hidden_size=18432, num_attention_heads=96,
    num_key_value_heads=8, num_ffi = 1,
    intermediate_size=73728, num_decoder_layers=96,
    vocab_size=256000, max_model_len=4*1024, hidden_act="silu",
    model_quality=QualityMetricsCollection([MMLU(accuracy=78.7, shots=0), IFEval(accuracy=86.1), GSM8K(accuracy=92.3, shots=0)]),
)

# Nemotron-51B (DeciLM architecture with heterogeneous per-layer configs)
# https://huggingface.co/nvidia/Llama-3_1-Nemotron-51B-Instruct/blob/main/config.json
# Per-layer tuples: (attn_no_op, attn_linear, n_heads_in_group, ffn_no_op, ffn_linear, ffn_mult)
_nemotron_51b_block_tuples = [
    (False, False, 8,  False, False, 1.3125),  # 0
    (False, False, 16, False, False, 2.625),   # 1
    (False, False, 8,  False, False, 5.25),    # 2
    (False, False, 8,  False, False, 5.25),    # 3
    (False, False, 8,  False, False, 5.25),    # 4
    (False, False, 32, False, False, 2.625),   # 5
    (False, False, 32, False, False, 2.625),   # 6
    (False, False, 64, False, False, 2.625),   # 7
    (False, False, 64, False, False, 2.625),   # 8
    (False, False, 32, False, False, 2.625),   # 9
    (False, False, 32, False, False, 2.625),   # 10
    (False, True,  None, False, False, 2.625), # 11
    (False, False, 64, False, False, 2.625),   # 12
    (False, False, 32, False, False, 2.625),   # 13
    (False, False, 32, False, False, 2.625),   # 14
    (False, True,  None, False, False, 1.3125),# 15
    (False, False, 8,  False, False, 5.25),    # 16
    (False, False, 8,  False, False, 5.25),    # 17
    (False, False, 8,  False, False, 5.25),    # 18
    (False, False, 8,  False, False, 5.25),    # 19
    (False, False, 8,  False, False, 5.25),    # 20
    (False, False, 8,  False, False, 5.25),    # 21
    (False, False, 8,  False, False, 5.25),    # 22
    (False, False, 8,  False, False, 5.25),    # 23
    (False, False, 8,  False, False, 5.25),    # 24
    (False, False, 8,  False, False, 5.25),    # 25
    (False, False, 8,  False, False, 5.25),    # 26
    (False, False, 8,  False, False, 5.25),    # 27
    (False, False, 8,  False, False, 5.25),    # 28
    (False, False, 8,  False, False, 5.25),    # 29
    (False, False, 8,  False, False, 5.25),    # 30
    (False, False, 8,  False, False, 5.25),    # 31
    (False, False, 8,  False, False, 5.25),    # 32
    (False, False, 8,  False, False, 5.25),    # 33
    (False, False, 8,  False, False, 5.25),    # 34
    (False, True,  None, False, False, 2.625), # 35
    (False, False, 8,  False, False, 5.25),    # 36
    (False, False, 8,  False, False, 5.25),    # 37
    (False, True,  None, False, False, 2.625), # 38
    (False, True,  None, False, False, 5.25),  # 39
    (False, True,  None, False, False, 2.625), # 40
    (False, True,  None, False, False, 2.625), # 41
    (False, True,  None, False, False, 2.625), # 42
    (True,  False, None, False, False, 1.3125),# 43
    (False, True,  None, False, False, 1.3125),# 44
    (False, False, 8,  False, False, 5.25),    # 45
    (True,  False, None, False, False, 1.3125),# 46
    (False, True,  None, False, False, 1.3125),# 47
    (True,  False, None, False, False, 1.3125),# 48
    (False, False, 8,  False, False, 5.25),    # 49
    (False, True,  None, False, False, 1.3125),# 50
    (True,  False, None, False, False, 1.3125),# 51
    (False, True,  None, False, False, 1.3125),# 52
    (False, True,  None, False, False, 1.3125),# 53
    (True,  False, None, False, False, 1.3125),# 54
    (True,  False, None, False, False, 1.3125),# 55
    (False, True,  None, False, False, 1.3125),# 56
    (True,  False, None, False, False, 1.3125),# 57
    (True,  False, None, False, False, 1.3125),# 58
    (False, True,  None, False, False, 1.3125),# 59
    (False, True,  None, False, False, 1.3125),# 60
    (False, True,  None, False, False, 1.3125),# 61
    (False, True,  None, False, False, 1.3125),# 62
    (False, False, 8,  False, False, 5.25),    # 63
    (False, False, 8,  False, False, 5.25),    # 64
    (False, False, 8,  False, False, 5.25),    # 65
    (False, False, 8,  False, False, 5.25),    # 66
    (False, False, 8,  False, False, 5.25),    # 67
    (False, False, 8,  False, False, 5.25),    # 68
    (False, False, 8,  False, False, 5.25),    # 69
    (False, False, 8,  False, False, 5.25),    # 70
    (False, False, 8,  False, False, 5.25),    # 71
    (False, False, 8,  False, False, 5.25),    # 72
    (False, False, 8,  False, False, 5.25),    # 73
    (False, False, 8,  False, False, 5.25),    # 74
    (False, False, 8,  False, False, 5.25),    # 75
    (False, False, 8,  False, False, 5.25),    # 76
    (False, False, 8,  False, False, 5.25),    # 77
    (False, False, 8,  False, False, 5.25),    # 78
    (False, False, 8,  False, False, 5.25),    # 79
]

def _build_nemotron_51b_layer_configs():
    """Build LayerConfig list from Nemotron-51B block config tuples."""
    num_heads = 64
    configs = []
    for attn_no_op, attn_linear, n_heads_in_group, ffn_no_op, ffn_linear, ffn_mult in _nemotron_51b_block_tuples:
        if attn_no_op:
            attn_type = "no_op"
        elif attn_linear:
            attn_type = "linear"
        else:
            attn_type = "MHA-global"
        if ffn_no_op:
            ffn_type = "no_op"
        elif ffn_linear:
            ffn_type = "linear"
        else:
            ffn_type = "Dense"
        num_kv_heads = None
        if n_heads_in_group is not None and attn_type == "MHA-global":
            num_kv_heads = num_heads // n_heads_in_group
        configs.append(LayerConfig(
            attention_type=attn_type,
            ffn_type=ffn_type,
            num_key_value_heads=num_kv_heads,
            ffn_mult=ffn_mult,
        ))
    return configs

nemotron_51b_config = ModelConfig(model='nvidia/Llama-3_1-Nemotron-51B-Instruct',
    hidden_size=8192, num_attention_heads=64,
    num_key_value_heads=8, num_ffi=2,
    intermediate_size=28672, num_decoder_layers=80,
    vocab_size=128256, max_model_len=128*1024, hidden_act="silu",
    layer_configs=_build_nemotron_51b_layer_configs(),
    model_quality=QualityMetricsCollection([MMLU(accuracy=80.2, shots=0), IFEval(accuracy=88.5), GPQA(accuracy=46.7)]),
)


# https://arxiv.org/pdf/2402.16819
nemotron_15b_config = ModelConfig(model='nvidia/Nemotron-4-15B',
    hidden_size=6144, num_attention_heads=48,
    num_key_value_heads=8, num_ffi = 1,
    intermediate_size=4*6144, num_decoder_layers=32,
    vocab_size=256000, max_model_len=4*1024, hidden_act="relu",
    model_quality=QualityMetricsCollection([MMLU(accuracy=64.2), GSM8K(accuracy=46.0), MATH(accuracy=47.0)]),
)

# https://huggingface.co/nvidia/Mistral-NeMo-Minitron-8B-Base/blob/main/config.json
minitron_8b_config = ModelConfig(model='nvidia/Mistral-NeMo-Minitron-8B-Base',
    hidden_size=4096, num_attention_heads=32,
    num_key_value_heads=8, num_ffi = 2,
    intermediate_size=11520, num_decoder_layers=40,
    vocab_size=131072, max_model_len=8*1024, hidden_act="silu",
    model_quality=QualityMetricsCollection([MMLU(accuracy=69.5), GSM8K(accuracy=58.5), Hellaswag(accuracy=83.0)]),
)

nvidia_models = get_all_model_configs(__name__)
nvidia_models.update({
    'nvidia/Llama-3_1-Nemotron-51B-Instruct': nemotron_51b_config,
    'nvidia/Mistral-NeMo-Minitron-8B-Base': minitron_8b_config,
    'nvidia/Nemotron-4-15B': nemotron_15b_config,
    'nvidia/Nemotron-4-340B-Instruct': nemotron_340b_config,
    'nvidia/Nemotron-4-340B': nemotron_340b_config,

})
