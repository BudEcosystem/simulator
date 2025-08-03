import json
import os
import requests
from typing import Dict, Union, Tuple, Optional
import pandas as pd
from pathlib import Path

from llm_memory_calculator.genz.Models.get_language_model import get_configs
from llm_memory_calculator.genz.LLM_inference import (
    prefill_moddeling,
    decode_moddeling
)
from llm_memory_calculator.genz.analyse_model import (
    get_model_df,
    get_summary_table
)
from llm_memory_calculator.genz.unit import Unit
from llm_memory_calculator.genz.system import System
from llm_memory_calculator.genz.Models.default_models import ModelConfig, MODEL_DICT

def calculate_model_parameters_from_huggingface(
    model_input: Union[str, Dict, Path],
    bits: str = 'bf16',
    return_breakdown: bool = False,
    custom_config: Dict = None,
    cache_dir: str = "/tmp/genz/configs"
) -> Union[float, Tuple[float, pd.DataFrame]]:
    """
    Calculate total parameters in a model using GenZ functions.
    
    Args:
        model_input: Can be:
            - HuggingFace model name (e.g., 'meta-llama/Llama-2-70B')
            - Path to local config.json file
            - URL to config.json file
            - Dictionary containing the config
        bits: Data type for weights ('bf16', 'int8', 'fp32', 'int4', etc.)
        return_breakdown: If True, returns detailed breakdown of parameters
        custom_config: Optional dict to override config values
        cache_dir: Directory to cache downloaded configs
    
    Returns:
        Total number of parameters (in billions if > 1B, millions otherwise)
        If return_breakdown=True, also returns a DataFrame with layer-wise breakdown
    
    Examples:
        # From HuggingFace model ID
        params = calculate_model_parameters_from_huggingface('meta-llama/Llama-2-7B')
        
        # From local file
        params = calculate_model_parameters_from_huggingface('/path/to/config.json')
        
        # From URL
        params = calculate_model_parameters_from_huggingface('https://huggingface.co/meta-llama/Llama-2-7B/raw/main/config.json')
        
        # From dict
        config_dict = {"hidden_size": 4096, "num_hidden_layers": 32, ...}
        params = calculate_model_parameters_from_huggingface(config_dict)
    """
    
    # Step 1: Load configuration
    model_config = None
    model_name = "custom_model"
    
    # Try to load from GenZ's predefined models first (if string)
    if isinstance(model_input, str) and not (
        model_input.endswith('.json') or 
        model_input.startswith('http') or 
        os.path.exists(model_input)
    ):
        try:
            model_config = get_configs(model_input)
            model_name = model_config.model
            print(f"Found model in GenZ database: {model_name}")
        except:
            # Not in GenZ database, try to download from HuggingFace
            hf_config = load_huggingface_config(model_input, cache_dir)
            if hf_config:
                model_config = create_model_config_from_hf(hf_config, model_input)
                model_name = model_input
            else:
                raise ValueError(f"Could not load model configuration for: {model_input}")
    
    # Handle different input types
    if model_config is None:
        if isinstance(model_input, dict):
            # Direct dictionary input
            hf_config = model_input
            model_name = hf_config.get('_name_or_path', 'custom_model')
        elif isinstance(model_input, (str, Path)):
            # File path or URL
            hf_config = load_config_from_source(model_input, cache_dir)
            model_name = extract_model_name(model_input, hf_config)
        else:
            raise ValueError(f"Invalid model_input type: {type(model_input)}")
        
        # Create GenZ ModelConfig
        model_config = create_model_config_from_hf(hf_config, model_name)
    
    # Apply custom config overrides if provided
    if custom_config:
        for key, value in custom_config.items():
            if hasattr(model_config, key):
                setattr(model_config, key, value)
    
    # Step 2: Use GenZ to get model weights
    print(f"Analyzing model architecture...")
    try:
        model_df, summary_table = prefill_moddeling(
            model=model_config,
            batch_size=1,
            input_tokens=1,
            model_profilling=True,
            bits=bits,
            tensor_parallel=1,
            pipeline_parallel=1
        )
    except Exception as e:
        print(f"Error in prefill_moddeling: {e}")
        # Fallback to decode modeling
        model_df, summary_table = decode_moddeling(
            model=model_config,
            batch_size=1,
            input_tokens=1,
            output_tokens=0,
            Bb=1,
            model_profilling=True,
            bits=bits,
            tensor_parallel=1,
            pipeline_parallel=1
        )
    
    # Step 3: Extract total weights and convert to parameters
    total_weights_mb = summary_table['Total Weights (MB)'].values[0]
    unused_weights_mb = summary_table['Unused Weights (MB)'].values[0]
    active_weights_mb = total_weights_mb - unused_weights_mb
    
    # Convert MB to number of parameters based on data type
    bytes_per_param = System.mem_multiplier.get(bits, 2)
    total_weights_bytes = total_weights_mb * 1024 * 1024
    total_params = total_weights_bytes / bytes_per_param
    
    active_weights_bytes = active_weights_mb * 1024 * 1024
    active_params = active_weights_bytes / bytes_per_param
    
    # Step 4: Format the output
    if total_params >= 1e9:
        param_str = f"{total_params/1e9:.2f}B"
        active_param_str = f"{active_params/1e9:.2f}B"
    else:
        param_str = f"{total_params/1e6:.2f}M"
        active_param_str = f"{active_params/1e6:.2f}M"
    
    print(f"\nModel: {model_name}")
    print(f"Architecture: {model_config.model}")
    print(f"Total Parameters: {param_str}")
    if unused_weights_mb > 0:
        print(f"Active Parameters: {active_param_str} (MoE model)")
    print(f"Total Weights: {total_weights_mb:.2f} MB (stored as {bits})")
    print(f"Configuration:")
    print(f"  - Hidden size: {model_config.hidden_size}")
    print(f"  - Layers: {model_config.num_decoder_layers}")
    print(f"  - Attention heads: {model_config.num_attention_heads}")
    if hasattr(model_config, 'num_experts') and model_config.num_experts > 1:
        print(f"  - Experts: {model_config.num_experts} (top-k: {model_config.expert_top_k})")
    
    if return_breakdown:
        # Create detailed breakdown
        breakdown_df = create_parameter_breakdown(model_df, summary_table, bytes_per_param)
        return total_params, breakdown_df
    
    return total_params


def load_huggingface_config(model_id: str, cache_dir: str) -> Optional[Dict]:
    """
    Download config.json from HuggingFace Hub
    """
    # Construct URL for config.json
    url = f"https://huggingface.co/{model_id}/raw/main/config.json"
    
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{model_id.replace('/', '_')}_config.json")
    
    # Check cache first
    if os.path.exists(cache_path):
        print(f"Loading config from cache: {cache_path}")
        with open(cache_path, 'r') as f:
            return json.load(f)
    
    # Download from HuggingFace
    try:
        print(f"Downloading config from: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        config = response.json()
        
        # Cache the config
        with open(cache_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config
    except Exception as e:
        print(f"Failed to download config: {e}")
        return None


def load_config_from_source(source: Union[str, Path], cache_dir: str) -> Dict:
    """
    Load config from file path or URL
    """
    source = str(source)
    
    # Check if it's a URL
    if source.startswith(('http://', 'https://')):
        try:
            response = requests.get(source, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise ValueError(f"Failed to download config from URL: {e}")
    
    # Check if it's a local file
    elif os.path.exists(source):
        with open(source, 'r') as f:
            return json.load(f)
    
    # Try as HuggingFace model ID
    else:
        config = load_huggingface_config(source, cache_dir)
        if config:
            return config
        else:
            raise ValueError(f"Could not load config from: {source}")


def extract_model_name(source: Union[str, Path], config: Dict) -> str:
    """
    Extract a reasonable model name from the source or config
    """
    # Try from config first
    if '_name_or_path' in config:
        return config['_name_or_path']
    
    # Extract from file path
    source = str(source)
    if '/' in source:
        parts = source.split('/')
        # Look for something that looks like a model name
        for i, part in enumerate(parts):
            if 'huggingface.co' in part and i + 1 < len(parts):
                # HuggingFace URL format
                return f"{parts[i+1]}/{parts[i+2]}"
            elif part.endswith('.json'):
                # Local file - use parent directory name
                if i > 0:
                    return parts[i-1]
    
    # Fallback
    return "custom_model"


def create_model_config_from_hf(hf_config: Dict, model_name: str) -> ModelConfig:
    """
    Create a GenZ ModelConfig from HuggingFace config.json
    
    This handles the mapping between HF config keys and GenZ ModelConfig
    """
    # Start with defaults
    genz_config = {
        'model': model_name,
        'vocab_size': 32000,
        'hidden_size': 4096,
        'intermediate_size': 11008,
        'num_decoder_layers': 32,
        'num_attention_heads': 32,
        'hidden_act': 'silu',
        'max_model_len': 2048,
    }
    
    # Direct mappings
    direct_mappings = {
        'vocab_size': 'vocab_size',
        'hidden_size': 'hidden_size',
        'intermediate_size': 'intermediate_size',
        'num_hidden_layers': 'num_decoder_layers',
        'num_attention_heads': 'num_attention_heads',
        'num_key_value_heads': 'num_key_value_heads',
        'hidden_act': 'hidden_act',
        'max_position_embeddings': 'max_model_len',
        'sliding_window': 'sliding_window',
        'rms_norm_eps': 'rms_norm_eps',
        'rope_theta': 'rope_theta',
    }
    
    # Apply direct mappings
    for hf_key, genz_key in direct_mappings.items():
        if hf_key in hf_config:
            genz_config[genz_key] = hf_config[hf_key]
    
    # Handle model type specific configurations
    model_type = hf_config.get('model_type', '').lower()
    architectures = hf_config.get('architectures', [])
    
    # Architecture-specific handling
    if model_type == 'llama' or any('llama' in arch.lower() for arch in architectures):
        genz_config['num_ffi'] = 2
        genz_config['hidden_act'] = hf_config.get('hidden_act', 'silu')
        
    elif model_type == 'mistral' or any('mistral' in arch.lower() for arch in architectures):
        genz_config['num_ffi'] = 2
        genz_config['hidden_act'] = 'silu'
        
    elif model_type == 'mixtral' or any('mixtral' in arch.lower() for arch in architectures):
        genz_config['num_ffi'] = 2
        genz_config['num_experts'] = hf_config.get('num_local_experts', 8)
        genz_config['expert_top_k'] = hf_config.get('num_experts_per_tok', 2)
        genz_config['hidden_act'] = 'silu'
        
    elif model_type == 'gpt2' or any('gpt' in arch.lower() for arch in architectures):
        genz_config['num_ffi'] = 1
        genz_config['hidden_act'] = hf_config.get('activation_function', 'gelu_new')
        
    elif model_type == 'gptj':
        genz_config['num_ffi'] = 1
        genz_config['hidden_act'] = 'gelu_new'
        genz_config['rotary_dim'] = hf_config.get('rotary_dim', 64)
        
    elif model_type == 'opt':
        genz_config['num_ffi'] = 1
        genz_config['hidden_act'] = 'relu'
        
    elif model_type == 'gemma':
        genz_config['num_ffi'] = 2
        genz_config['hidden_act'] = hf_config.get('hidden_act', 'gelu')
        genz_config['head_dim'] = hf_config.get('head_dim', 256)
        
    elif model_type == 'phi':
        genz_config['num_ffi'] = 2
        genz_config['hidden_act'] = 'gelu_new'
        if 'partial_rotary_factor' in hf_config:
            genz_config['partial_rotary_factor'] = hf_config['partial_rotary_factor']
    
    elif model_type == 'qwen' or model_type == 'qwen2':
        genz_config['num_ffi'] = 2
        genz_config['hidden_act'] = 'silu'
        
    elif model_type == 'dbrx':
        # DBRX specific MoE configuration
        genz_config['num_experts'] = hf_config.get('ffn_config', {}).get('moe_num_experts', 16)
        genz_config['expert_top_k'] = hf_config.get('ffn_config', {}).get('moe_top_k', 4)
        genz_config['num_ffi'] = 2
        
    elif model_type == 'deepseek_v2':
        # DeepSeek V2 MoE
        genz_config['num_experts'] = hf_config.get('n_routed_experts', 160)
        genz_config['expert_top_k'] = hf_config.get('num_experts_per_tok', 6)
        genz_config['n_shared_experts'] = hf_config.get('n_shared_experts', 2)
        genz_config['moe_intermediate_size'] = hf_config.get('moe_intermediate_size')
        
    # Handle MoE models
    if 'num_local_experts' in hf_config or 'num_experts' in hf_config:
        genz_config['num_experts'] = hf_config.get('num_local_experts', 
                                                   hf_config.get('num_experts', 1))
        genz_config['expert_top_k'] = hf_config.get('num_experts_per_tok', 
                                                    hf_config.get('top_k_experts', 2))
    
    # Calculate derived values
    if 'head_dim' in hf_config:
        genz_config['head_dim'] = hf_config['head_dim']
    elif genz_config.get('hidden_size') and genz_config.get('num_attention_heads'):
        genz_config['head_dim'] = genz_config['hidden_size'] // genz_config['num_attention_heads']
    
    # Handle special activation functions
    if 'hidden_activation' in hf_config:
        genz_config['hidden_act'] = hf_config['hidden_activation']
    elif 'act_fn' in hf_config:
        genz_config['hidden_act'] = hf_config['act_fn']
    
    # Create and return ModelConfig
    return ModelConfig(**genz_config)


def create_parameter_breakdown(model_df: pd.DataFrame, summary_table: pd.DataFrame, 
                              bytes_per_param: float) -> pd.DataFrame:
    """
    Create a detailed breakdown of parameters by layer type
    """
    unit = Unit()
    breakdown_data = []
    
    # Process each row in the model dataframe
    for idx, row in model_df.iterrows():
        layer_name = row['Layer Name']
        op_type = row['Op Type']
        
        if op_type in ['GEMM', 'Logit', 'Attend', 'CONV1D']:
            weights_mb = row[f'Input_w ({unit.unit_mem})']
            if weights_mb > 0:
                params = (weights_mb * 1024 * 1024) / bytes_per_param
                
                breakdown_data.append({
                    'Layer Type': layer_name,
                    'Operation': op_type,
                    'Parameters (M)': params / 1e6,
                    'Weight Size (MB)': weights_mb,
                    'Dimension': str(row['Dimension'])
                })
    
    breakdown_df = pd.DataFrame(breakdown_data)
    
    if len(breakdown_df) > 0:
        # Aggregate by layer type
        layer_summary = breakdown_df.groupby('Layer Type')['Parameters (M)'].sum().sort_values(ascending=False)
        
        print("\nParameter Distribution by Layer:")
        total_params_m = layer_summary.sum()
        for layer, params in layer_summary.items():
            percentage = (params / total_params_m) * 100
            print(f"  {layer}: {params:.2f}M ({percentage:.1f}%)")
        
        # Component summary
        component_mapping = {
            'Embeddings': ['embeddings', 'embed', 'word_embeddings'],
            'Attention': ['QKV', 'Out Proj', 'Logit', 'Attend', 'query', 'key', 'value', 'dense'],
            'FFN': ['up+gate', 'down', 'Gate', 'mlp', 'fc1', 'fc2', 'dense_h_to_4h', 'dense_4h_to_h'],
            'Output': ['classifier', 'lm_head', 'output'],
        }
        
        print("\nParameter Distribution by Component:")
        for component, keywords in component_mapping.items():
            mask = breakdown_df['Layer Type'].str.lower().str.contains('|'.join(keywords), case=False, na=False)
            component_params = breakdown_df[mask]['Parameters (M)'].sum()
            if component_params > 0:
                percentage = (component_params / total_params_m) * 100
                print(f"  {component}: {component_params:.2f}M ({percentage:.1f}%)")
    
    return breakdown_df


# Additional utility functions
def download_and_analyze_model(model_id: str, save_config: bool = False):
    """
    Download a model config from HuggingFace and analyze it
    """
    print(f"Downloading and analyzing: {model_id}")
    
    # Download config
    config = load_huggingface_config(model_id, "/tmp/genz/configs")
    
    if save_config:
        # Save for inspection
        save_path = f"{model_id.replace('/', '_')}_config.json"
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Config saved to: {save_path}")
    
    # Analyze
    total_params, breakdown = calculate_model_parameters_from_huggingface(
        config,
        bits='bf16',
        return_breakdown=True
    )
    
    return total_params, breakdown, config


def batch_analyze_configs(config_dir: str, output_file: str = "model_analysis.csv"):
    """
    Analyze all config.json files in a directory
    """
    results = []
    
    for file in Path(config_dir).glob("*.json"):
        print(f"\nAnalyzing: {file.name}")
        try:
            total_params = calculate_model_parameters_from_huggingface(
                str(file),
                bits='bf16'
            )
            
            results.append({
                'Config File': file.name,
                'Parameters': total_params,
                'Size': f"{total_params/1e9:.2f}B" if total_params >= 1e9 else f"{total_params/1e6:.0f}M"
            })
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                'Config File': file.name,
                'Parameters': None,
                'Size': 'Error'
            })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return df


# Example usage
if __name__ == "__main__":
    # Example 1: From HuggingFace model ID (will download config)
    print("Example 1: HuggingFace model ID")
    
    
    # Example 2: From local config.json file
    print("\n\nExample 2: Local config.json file")
    # Save a sample config for testing
    sample_config =  {
  "architectures": [
    "DeepseekV2ForCausalLM"
  ],
  "attention_bias": False,
  "attention_dropout": 0.0,
  "auto_map": {
    "AutoConfig": "configuration_deepseek.DeepseekV2Config",
    "AutoModel": "modeling_deepseek.DeepseekV2Model",
    "AutoModelForCausalLM": "modeling_deepseek.DeepseekV2ForCausalLM"
  },
  "aux_loss_alpha": 0.001,
  "bos_token_id": 100000,
  "eos_token_id": 100001,
  "ep_size": 1,
  "first_k_dense_replace": 1,
  "hidden_act": "silu",
  "hidden_size": 5120,
  "initializer_range": 0.02,
  "intermediate_size": 12288,
  "kv_lora_rank": 512,
  "max_position_embeddings": 163840,
  "model_type": "deepseek_v2",
  "moe_intermediate_size": 1536,
  "moe_layer_freq": 1,
  "n_group": 8,
  "n_routed_experts": 160,
  "n_shared_experts": 2,
  "norm_topk_prob": False,
  "num_attention_heads": 128,
  "num_experts_per_tok": 6,
  "num_hidden_layers": 60,
  "num_key_value_heads": 128,
  "pretraining_tp": 1,
  "q_lora_rank": 1536,
  "qk_nope_head_dim": 128,
  "qk_rope_head_dim": 64,
  "rms_norm_eps": 1e-06,
  "rope_scaling": {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 40,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 4096,
    "type": "yarn"
  },
  "rope_theta": 10000,
  "routed_scaling_factor": 16.0,
  "scoring_func": "softmax",
  "seq_aux": True,
  "tie_word_embeddings": False,
  "topk_group": 3,
  "topk_method": "group_limited_greedy",
  "torch_dtype": "bfloat16",
  "transformers_version": "4.39.3",
  "use_cache": True,
  "v_head_dim": 128,
  "vocab_size": 102400
}
    
    with open("/tmp/test_config.json", "w") as f:
        json.dump(sample_config, f)
    
    params = calculate_model_parameters_from_huggingface("/tmp/test_config.json")
    print(params)