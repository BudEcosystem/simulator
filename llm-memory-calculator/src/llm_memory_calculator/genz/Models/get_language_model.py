import pandas as pd
import os
from math import ceil
import numpy as np
from datetime import datetime
from types import SimpleNamespace
from typing import Union
from llm_memory_calculator.genz.parallelism import ParallelismConfig

from llm_memory_calculator.genz.Models.default_models import ModelConfig, MODEL_DICT
from llm_memory_calculator.huggingface_loader import HuggingFaceConfigLoader

from llm_memory_calculator.genz.Models.utils import OpType, ResidencyInfo, CollectiveType, parse_einsum_expression
from llm_memory_calculator.genz.Models.attention import mha_flash_attention_prefill, mha_flash_attention_decode, mha_flash_attention_chunked
from llm_memory_calculator.genz.Models.ffn import ffn_prefill, ffn_decode, deepseek_ffn_prefill
from llm_memory_calculator.genz.Models.mamba import mamba_prefill, mamba_decode
from llm_memory_calculator.genz.Models.embedding import input_embedding, output_embedding
from difflib import get_close_matches
from uuid import uuid4
try:
    from BudSimulator.lora.injection import inject_lora_ops
except ImportError:
    # Fallback import
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from lora.injection import inject_lora_ops

def huggingface_config_to_model_config(hf_config: dict, model_name: str) -> ModelConfig:
    """Convert HuggingFace config to ModelConfig format.

    Supports:
    - Standard transformer models (LLaMA, GPT, etc.)
    - MoE models (Mixtral, DeepSeek, Qwen MoE, etc.)
    - Mamba/SSM models (Mamba, Jamba, etc.)
    - Hybrid models (Jamba with Mamba + MoE)
    """
    # Extract architecture type
    architectures = hf_config.get('architectures', [])
    model_type = hf_config.get('model_type', '').lower()
    is_mamba = any('mamba' in arch.lower() for arch in architectures) or 'mamba' in model_type
    is_jamba = 'jamba' in model_type or any('jamba' in arch.lower() for arch in architectures)

    # Extract key parameters with proper defaults
    vocab_size = hf_config.get('vocab_size', 32000)
    hidden_size = hf_config.get('hidden_size', 4096)
    num_layers = hf_config.get('num_hidden_layers', 32)
    num_heads = hf_config.get('num_attention_heads', 32)

    # Handle different naming conventions for intermediate size
    intermediate_size = hf_config.get('intermediate_size',
                                     hf_config.get('ffn_dim',
                                     hf_config.get('mlp_dim', 11008)))

    # Handle KV heads
    num_kv_heads = hf_config.get('num_key_value_heads', num_heads)

    # Handle head dim
    head_dim = hf_config.get('head_dim', hidden_size // num_heads if num_heads > 0 else 128)

    # Get max position embeddings
    max_seq_len = hf_config.get('max_position_embeddings',
                               hf_config.get('max_sequence_length', 128000))

    # ============================================================
    # MoE (Mixture of Experts) Attribute Extraction
    # ============================================================
    # Different HF configs use different naming conventions
    num_experts = hf_config.get('num_local_experts',  # Mixtral, Qwen
                               hf_config.get('num_experts',  # DeepSeek
                               hf_config.get('n_routed_experts',  # DeepSeek V2/V3
                               1)))

    expert_top_k = hf_config.get('num_experts_per_tok',  # Mixtral, DeepSeek
                                hf_config.get('num_experts_per_token',
                                hf_config.get('expert_top_k',
                                hf_config.get('top_k', 1 if num_experts <= 1 else 2))))

    # MoE intermediate size (per expert)
    moe_intermediate_size = hf_config.get('moe_intermediate_size',  # DeepSeek V2/V3
                                         hf_config.get('expert_intermediate_size',
                                         intermediate_size))

    # Shared experts (DeepSeek V2/V3 specific)
    n_shared_experts = hf_config.get('n_shared_experts',
                                    hf_config.get('num_shared_experts', 0))

    shared_expert_intermediate_size = hf_config.get('shared_expert_intermediate_size',
                                                   intermediate_size if n_shared_experts > 0 else None)

    # MoE layer frequency (which layers have MoE)
    moe_layer_freq = hf_config.get('moe_layer_freq',
                                  hf_config.get('expert_layer_freq'))

    # First K dense layers before MoE kicks in (DeepSeek)
    first_k_dense_replace = hf_config.get('first_k_dense_replace',
                                         hf_config.get('num_dense_layers'))

    # Expert layer period and offset
    expert_layer_period = hf_config.get('expert_layer_period', 1 if num_experts > 1 else 1)
    expert_layer_offset = hf_config.get('expert_layer_offset', 0)

    # ============================================================
    # Mamba/SSM Attribute Extraction
    # ============================================================
    mamba_d_state = hf_config.get('mamba_d_state',
                                 hf_config.get('state_size',
                                 hf_config.get('ssm_state_size', None)))

    mamba_d_conv = hf_config.get('mamba_d_conv',
                                hf_config.get('conv_kernel',
                                hf_config.get('ssm_conv_kernel', None)))

    mamba_expand = hf_config.get('mamba_expand',
                                hf_config.get('expand',
                                hf_config.get('ssm_expand', None)))

    mamba_dt_rank = hf_config.get('mamba_dt_rank',
                                 hf_config.get('time_step_rank', 'auto'))

    mamba_conv_bias = hf_config.get('mamba_conv_bias',
                                   hf_config.get('use_conv_bias', True))

    mamba_proj_bias = hf_config.get('mamba_proj_bias',
                                   hf_config.get('use_bias', False))

    # Jamba-specific: attention layer period
    attn_layer_period = hf_config.get('attn_layer_period', 1)
    attn_layer_offset = hf_config.get('attn_layer_offset', 0)
    mamba_layer_period = hf_config.get('mamba_layer_period', 1)

    # ============================================================
    # FFN Implementation Selection
    # ============================================================
    # DeepSeek models use a special FFN implementation with shared experts
    ffn_implementation = "default"
    if n_shared_experts > 0 or 'deepseek' in model_name.lower():
        ffn_implementation = "deepseek"

    # Create ModelConfig with all extracted attributes
    return ModelConfig(
        model=model_name,
        vocab_size=vocab_size,
        max_model_len=max_seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_decoder_layers=num_layers,
        num_attention_heads=num_heads,
        head_dim=head_dim,
        num_key_value_heads=num_kv_heads,
        hidden_act=hf_config.get('hidden_act', hf_config.get('activation_function', 'silu')),
        sliding_window=hf_config.get('sliding_window', None),
        ffn_implementation=ffn_implementation,
        # MoE parameters
        moe_layer_freq=moe_layer_freq,
        num_experts=num_experts,
        expert_top_k=expert_top_k,
        moe_intermediate_size=moe_intermediate_size,
        n_shared_experts=n_shared_experts,
        shared_expert_intermediate_size=shared_expert_intermediate_size,
        first_k_dense_replace=first_k_dense_replace,
        # Mamba parameters
        mamba_d_state=mamba_d_state,
        mamba_d_conv=mamba_d_conv,
        mamba_expand=mamba_expand,
        mamba_dt_rank=mamba_dt_rank,
        mamba_conv_bias=mamba_conv_bias,
        mamba_proj_bias=mamba_proj_bias,
        # Multi-type model parameters
        mamba_layer_period=mamba_layer_period,
        attn_layer_period=attn_layer_period,
        attn_layer_offset=attn_layer_offset,
        expert_layer_period=expert_layer_period,
        expert_layer_offset=expert_layer_offset,
    )

def get_configs(name: Union[str, dict, 'ModelConfig', SimpleNamespace]) -> ModelConfig:
    """Get model configuration from various input types.

    Args:
        name: Can be one of:
            - str: Model name (from MODEL_DICT or HuggingFace model ID)
            - dict: HuggingFace config dict (e.g., from config.json)
            - ModelConfig: Already a ModelConfig object (passed through)
            - SimpleNamespace: Config object with attributes (converted to dict then ModelConfig)

    Returns:
        ModelConfig object with the model's configuration
    """
    if isinstance(name, ModelConfig):
        return name
    elif isinstance(name, SimpleNamespace):
        # Convert SimpleNamespace back to dict, then to ModelConfig
        config_dict = vars(name)
        model_name = getattr(name, 'name', getattr(name, 'model', getattr(name, '_name_or_path', 'custom')))
        return huggingface_config_to_model_config(config_dict, model_name)
    elif isinstance(name, dict):
        # Handle HuggingFace config dicts
        model_name = name.get('name', name.get('model', name.get('_name_or_path', 'custom')))
        return huggingface_config_to_model_config(name, model_name)
    elif isinstance(name, str):
        # First try the static MODEL_DICT with lowercase
        name_lower = name.lower()
        if model := MODEL_DICT.get_model(name_lower):
            model_config = model
            return model_config

        # If not found in MODEL_DICT, try loading from HuggingFace
        try:
            loader = HuggingFaceConfigLoader()
            hf_config = loader.fetch_model_config(name)
            return huggingface_config_to_model_config(hf_config, name)
        except Exception as e:
            # If HuggingFace loading fails, show suggestions from MODEL_DICT
            model_list = MODEL_DICT.list_models()
            close_matches = get_close_matches(name_lower, model_list, cutoff=0.4)
            if close_matches:
                print("Did you mean one of these models?")
                for match in close_matches:
                    print(f" - {match}")
            raise ValueError(f"ERROR, model '{name}' not found in MODEL_DICT and failed to load from HuggingFace: {str(e)}")

    else:
        raise ValueError("ERROR, model name parsed incorrect, please check!!! Model Name:",name)

def get_ffn_implementation(model_config:ModelConfig):
    if model_config.ffn_implementation == "default":
        return ffn_prefill
    elif model_config.ffn_implementation == "deepseek":
        return deepseek_ffn_prefill
    else:
        raise ValueError("FFN implementation not supported")


def save_layers(layers:list, data_path:str, name:str):
    model_path = os.path.join(data_path,"model")
    df = pd.DataFrame(layers, columns=['Name', 'M', 'N', 'D', 'H', 'Z', 'Z', 'T'])
    file_name = name.replace("/", "_") + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + str(uuid4()) +'.csv'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    df.to_csv(os.path.join(model_path, file_name),  header=True, index=None)
    return file_name

def repeat_layers(num_repeat:int):
    return [["Repeat", num_repeat, 1, 1, 1, 1, 1, OpType.REPEAT]]

def end_repeat_layers(num_repeat:int):
    return [["End Repeat", num_repeat, 1, 1, 1, 1, 1, OpType.ENDREPEAT]]

DATA_PATH = "/tmp/genz/data/"

def create_inference_moe_prefill_layer(input_sequence_length, name='GPT-2', data_path=DATA_PATH,
                         **args):
    model_config = get_configs(name)

    parallelism_config = ParallelismConfig(
        tensor_parallel=args.get('tensor_parallel',1),
        expert_parallel=args.get('expert_parallel',1),
        sequence_parallel=args.get('sequence_parallel',1),
        data_parallel=args.get('data_parallel',1)
        )

    layers = mha_flash_attention_prefill(model_config, parallelism_config, input_sequence_length) + ffn_prefill(model_config, parallelism_config, input_sequence_length)
    if isinstance(name, ModelConfig):
        name = name.model
    return save_layers(layers=layers, data_path=data_path, name=name+"_prefix_")

def create_inference_moe_decode_layer(input_sequence_length, name='GPT-2', data_path=DATA_PATH,
                         output_gen_tokens=32, **args):

    model_config = get_configs(name)
    parallelism_config = ParallelismConfig(
        tensor_parallel=args.get('tensor_parallel',1),
        expert_parallel=args.get('expert_parallel',1),
        sequence_parallel=args.get('sequence_parallel',1),
        data_parallel=args.get('data_parallel',1)
        )
    layers = mha_flash_attention_decode(model_config, parallelism_config, input_sequence_length, output_gen_tokens) + ffn_decode(model_config, parallelism_config)
    if isinstance(name, ModelConfig):
        name = name.model
    return save_layers(layers=layers, data_path=data_path, name=name+"_decode_")

def create_full_prefill_model(
    name: Union[str, ModelConfig] ='GPT-2', 
    input_sequence_length: int=1024,
    data_path:str=DATA_PATH,
    **args) -> str:
    """
    The function `create_full_prefill_model` constructs a model with specified configurations and
    parallelism settings, saving the layers to a specified data path.
    
    name: The `name` parameter in the `create_full_prefill_model` function is used to specify the
    model configuration to be used. It can either be a string representing the name of the model
    (default is 'GPT-2') or an instance of `ModelConfig` class, defaults to GPT-2
    
    input_sequence_length: The `input_sequence_length` parameter specifies the length of the
    input sequence for the model. In this function, it is set to a default value of 1024. This parameter
    determines how many tokens or elements can be processed in a single input sequence, defaults to 1024
    
    data_path: The `data_path` parameter in the `create_full_prefill_model` function is a string
    that represents the path where the data will be saved or loaded from. It is a default parameter with
    a value of `DATA_PATH`, which is likely a constant or variable defined elsewhere in your codebase

    tensor_parallel: The `tensor_parallel` to define the degree of tensor parallelism, defaults to 1
    expert_parallel: The `expert_parallel` to define the degree of expert parallelism, defaults to 1
    sequence_parallel: The `sequence_parallel` to define the degree of sequence parallelism, defaults to 1
    data_parallel: The `data_parallel` to define the degree of data parallelism, defaults to 1
    pipeline_parallel: The `pipeline_parallel` to define the degree of pipeline parallelism, defaults to 1 
    
    return: The function `create_full_prefill_model` returns a string, which is the result of calling
    the `save_layers` function with the `full_model`, `data_path`, and a modified `name` as arguments.
    """
    model_config = get_configs(name)
    pipeline_stages = args.get('pipeline_parallel',1)

    parallelism_config = ParallelismConfig(
        tensor_parallel=args.get('tensor_parallel',1),
        expert_parallel=args.get('expert_parallel',1),
        sequence_parallel=args.get('sequence_parallel',1),
        data_parallel=args.get('data_parallel',1),
        )

    def add_layers(layers, num_layers):
        layers += repeat_layers(num_layers)
        if model_config.unique_layers == 1:
            if model_config.layer_type[0][0] == "MHA-global":
                attn_ops = mha_flash_attention_prefill(model_config, parallelism_config, input_sequence_length)
                # Inject LoRA ops for attention if configured
                attn_ops = inject_lora_ops(attn_ops, model_config, parallelism_config, input_sequence_length)
                layers += attn_ops
            elif model_config.layer_type[0][0] == "Mamba":
                layers += mamba_prefill(model_config, parallelism_config, input_sequence_length)
        else:
            raise ValueError("More then 1 unique layers not supported. Work in progress")
        
        # Get FFN operations and inject LoRA if configured
        ffn_ops = ffn_prefill(model_config, parallelism_config, input_sequence_length)
        ffn_ops = inject_lora_ops(ffn_ops, model_config, parallelism_config, input_sequence_length)
        layers += ffn_ops
        
        layers += end_repeat_layers(num_layers)
        return layers

    full_model = []
    full_model += input_embedding(model_config, parallelism_config, input_sequence_length)
    if pipeline_stages > 1:
        layers_per_stage = ceil(model_config.num_decoder_layers / pipeline_stages)
        layers_last_stage = model_config.num_decoder_layers - layers_per_stage * (pipeline_stages - 1)

        ## For PP stages
        ## First PP-1 stages will have layers_per_stage layers and message pass at the end
        full_model += repeat_layers(pipeline_stages - 1)
        ## Single stage will have layers_per_stage layers
        full_model = add_layers(full_model, layers_per_stage)
        ## Single stage layers end and message pass at the end
        full_model += [["Message Pass", input_sequence_length // args.get('sequence_parallel', 1), model_config.hidden_size, 1, 1, 1, CollectiveType.MessagePass, OpType.Sync]]
        full_model += end_repeat_layers(pipeline_stages - 1)
        ## Last stage will have layers_last_stage layers and no message pass at the end
        full_model = add_layers(full_model, layers_last_stage)
    else:
        full_model = add_layers(full_model, model_config.num_decoder_layers)

    full_model += output_embedding(model_config, parallelism_config, input_sequence_length)
    if isinstance(name, ModelConfig):
        name = name.model
    return save_layers(layers=full_model, data_path=data_path, name=name + "_prefix_")


def create_full_decode_model(
    name: Union[str, ModelConfig] ='GPT-2',
    input_sequence_length: int = 1024,
    output_gen_tokens: int = 0,
    data_path: str=DATA_PATH,
    **args) -> str:
    """
    The function `create_full_decode_model` constructs a decode model with specified configurations and
    parallelism settings, saving the layers to a specified data path.

    name: The `name` parameter in the `create_full_decode_model` function is used to specify the
    model configuration to be used. It can either be a string representing the name of the model
    (default is 'GPT-2') or an instance of `ModelConfig` class, defaults to GPT-2

    input_sequence_length: The `input_sequence_length` parameter specifies the length of the
    input sequence for the model. In this function, it is set to a default value of 1024. This parameter
    determines how many tokens or elements can be processed in a single input sequence, defaults to 1024

    output_gen_tokens: The `output_gen_tokens` parameter specifies the number of tokens to generated since the prefill.
                        This is to keep a track of multiple beams. Defaults to 1

    data_path: The `data_path` parameter in the `create_full_decode_model` function is a string
    that represents the path where the data will be saved or loaded from. It is a default parameter with
    a value of `DATA_PATH`, which is likely a constant or variable defined elsewhere in your codebase

    tensor_parallel: The `tensor_parallel` to define the degree of tensor parallelism, defaults to 1
    expert_parallel: The `expert_parallel` to define the degree of expert parallelism, defaults to 1
    sequence_parallel: The `sequence_parallel` to define the degree of sequence parallelism, defaults to 1
    data_parallel: The `data_parallel` to define the degree of data parallelism, defaults to 1
    pipeline_parallel: The `pipeline_parallel` to define the degree of pipeline parallelism, defaults to 1

    return: The function `create_full_decode_model` returns a string, which is the result of calling
    the `save_layers` function with the `full_model`, `data_path`, and a modified `name` as arguments.
    """
    model_config = get_configs(name)
    pipeline_stages = args.get('pipeline_parallel', 1)

    parallelism_config = ParallelismConfig(
        tensor_parallel=args.get('tensor_parallel', 1),
        expert_parallel=args.get('expert_parallel', 1),
        sequence_parallel=args.get('sequence_parallel', 1),
        data_parallel=args.get('data_parallel', 1),
    )

    def add_layers(layers, num_layers):
        layers += repeat_layers(num_layers)
        if model_config.unique_layers == 1:
            if model_config.layer_type[0][0] == "MHA-global":
                attn_ops = mha_flash_attention_decode(model_config, parallelism_config, input_sequence_length, output_gen_tokens)
                # Inject LoRA ops for attention if configured
                attn_ops = inject_lora_ops(attn_ops, model_config, parallelism_config, 1)  # decode processes 1 token at a time
                layers += attn_ops
            elif model_config.layer_type[0][0] == "Mamba":
                layers += mamba_decode(model_config, parallelism_config, input_sequence_length)
        else:
            raise ValueError("More then 1 unique layers not supported. Work in progress")
        
        # Get FFN operations and inject LoRA if configured
        ffn_ops = ffn_decode(model_config, parallelism_config)
        ffn_ops = inject_lora_ops(ffn_ops, model_config, parallelism_config, 1)  # decode processes 1 token at a time
        layers += ffn_ops
        
        layers += end_repeat_layers(num_layers)
        return layers

    full_model = []

    if pipeline_stages > 1:
        layers_per_stage = ceil(model_config.num_decoder_layers / pipeline_stages)
        layers_last_stage = model_config.num_decoder_layers - layers_per_stage * (pipeline_stages - 1)

        full_model += repeat_layers(pipeline_stages - 1)
        full_model = add_layers(full_model, layers_per_stage)
        full_model += [["Message Pass", 1, model_config.hidden_size, 1, 1, 1, CollectiveType.MessagePass, OpType.Sync]]
        full_model += end_repeat_layers(pipeline_stages - 1)
        full_model = add_layers(full_model, layers_last_stage)
    else:
        full_model = add_layers(full_model, model_config.num_decoder_layers)
    full_model += output_embedding(model_config, parallelism_config, 1)
    if isinstance(name, ModelConfig):
        name = name.model
    return save_layers(layers=full_model, data_path=data_path, name=name + "_decode_")

def create_full_chunked_model(name:str ='GPT-2',
                            prefill_kv_sizes:list[(int,int)] =[], decode_kv_sizes: list[int]=[],
                            data_path:str = DATA_PATH, **args):
    ## Prefill KV sizes is a list of request by request, num tokens calculated and to be calculated.
    model_config = get_configs(name)
    pipeline_stages = args.get('pipeline_parallel',1)

    parallelism_config = ParallelismConfig(
        tensor_parallel=args.get('tensor_parallel',1),
        expert_parallel=args.get('expert_parallel',1),
        sequence_parallel=args.get('sequence_parallel',1),
        data_parallel=args.get('data_parallel',1),
        )

    ## Calculate the chunk size
    prefill_length = sum([i[1] for i in prefill_kv_sizes])
    decode_length = 0
    for kv_size in decode_kv_sizes:
        if isinstance(kv_size, tuple) and len(kv_size) == 4:
            decode_num_beams = kv_size[1]
            decode_length += decode_num_beams
        elif isinstance(kv_size, tuple) and len(kv_size) == 2:
            num_beams = kv_size[0]
            decode_length += num_beams
        else:
            decode_length += 1
    chunk_size = prefill_length + decode_length

    def add_layers(layers, num_layers):
        layers += repeat_layers(num_layers)
        
        # Get attention operations and inject LoRA if configured
        attn_ops = mha_flash_attention_chunked(  model_config=model_config,
                                                parallelism_config=parallelism_config,
                                                chunk_size=chunk_size,
                                                prefill_kv_sizes=prefill_kv_sizes,
                                                decode_kv_sizes=decode_kv_sizes)
        attn_ops = inject_lora_ops(attn_ops, model_config, parallelism_config, chunk_size)
        layers += attn_ops
        
        # Get FFN operations and inject LoRA if configured
        ffn_ops = get_ffn_implementation(model_config)(model_config, parallelism_config, chunk_size)
        ffn_ops = inject_lora_ops(ffn_ops, model_config, parallelism_config, chunk_size)
        layers += ffn_ops
        
        layers += end_repeat_layers(num_layers)
        return layers

    # assert prefill_length > 0, "Chunk size should be greater than the decode batches"
    full_model = []
    full_model += input_embedding(model_config, parallelism_config, prefill_length)
    if pipeline_stages > 1:
        layers_per_stage = ceil(model_config.num_decoder_layers / pipeline_stages)
        layers_last_stage = model_config.num_decoder_layers - layers_per_stage * (pipeline_stages - 1)

        ## For PP stages
        ## First PP-1 stages will have layers_per_stage layers and message pass at the end
        full_model += repeat_layers(pipeline_stages - 1)
        ## Single stage will have layers_per_stage layers
        full_model = add_layers(full_model, layers_per_stage)
        ## Single stage layers end and message pass at the end
        full_model += [["Message Pass", chunk_size // args.get('sequence_parallel', 1), model_config.hidden_size, 1, 1, 1, CollectiveType.MessagePass, OpType.Sync]]
        full_model += end_repeat_layers(pipeline_stages - 1)
        ## Last stage will have layers_last_stage layers and no message pass at the end
        full_model = add_layers(full_model, layers_last_stage)
    else:
        full_model = add_layers(full_model, model_config.num_decoder_layers)

    full_model += output_embedding(model_config, parallelism_config, chunk_size)
    if isinstance(name, ModelConfig):
        name = name.model
    return save_layers(layers=full_model, data_path=data_path, name=name+"_chunked_")


def create_inference_mamba_prefix_model(input_sequence_length, name='jamba', data_path=DATA_PATH,
                         **args):

    model_config = get_configs(name)

    parallelism_config = ParallelismConfig(
        tensor_parallel=args.get('tensor_parallel',1),
        expert_parallel=args.get('expert_parallel',1),
        sequence_parallel=args.get('sequence_parallel',1),
        data_parallel=args.get('data_parallel',1)
        )

    layers = mamba_prefill(model_config, parallelism_config, input_sequence_length) + ffn_prefill(model_config, parallelism_config, input_sequence_length)
    if isinstance(name, ModelConfig):
        name = name.model
    return save_layers(layers=layers, data_path=data_path, name=name+"_prefix_")


def create_inference_mamba_decode_model(input_sequence_length, name='jamba', data_path=DATA_PATH,
                         **args):

    model_config = get_configs(name)
    parallelism_config = ParallelismConfig(
        tensor_parallel=args.get('tensor_parallel',1),
        expert_parallel=args.get('expert_parallel',1),
        sequence_parallel=args.get('sequence_parallel',1),
        data_parallel=args.get('data_parallel',1)
        )
    layers = mamba_decode(model_config, parallelism_config, input_sequence_length) + ffn_decode(model_config, parallelism_config)
    if isinstance(name, ModelConfig):
        name = name.model
    return save_layers(layers=layers, data_path=data_path, name=name+"_decode_")



def einsum_test(equation=None, einsum_vars=None):

    if equation is None:
        A = (2, 3, 4)
        B = (2, 4, 5)
        C = (5, 6)
        equation = 'ijk,ikl,lm->ijm'
        einsum_vars = parse_einsum_expression(equation, A, B, C)

    layers = [["test", equation, einsum_vars, 1, 1, 1, ResidencyInfo.All_offchip, OpType.EINSUM]]

    return save_layers(layers=layers, data_path=DATA_PATH, name="einsum_")
