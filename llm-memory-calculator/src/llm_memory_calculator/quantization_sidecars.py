"""Quantization metadata that lives beside config.json instead of inside it.

Most quantizers write `quantization_config` into config.json, but three widely used toolchains write it to a
sidecar file, and a checkpoint that only has the sidecar looked unquantized — so it was timed and sized at bf16:

  hf_quant_config.json   NVIDIA ModelOpt (FP8, NVFP4, INT4/W4A8 AWQ, INT8 SmoothQuant exports)
  quantize_config.json   AutoGPTQ (and GPTQ exports that predate transformers' quantization_config)
  quant_config.json      AutoAWQ's legacy layout

`merge_quantization_sidecar` normalises whichever is present into the HuggingFace `quantization_config`
shape the rest of the package already reads. config.json's own block always wins: a sidecar is only a
fallback for a checkpoint that declares nothing. Standard library only, so the config loader can use it
without importing the performance model.
"""
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

#: Sidecar files in the order they are consulted.
SIDECAR_FILES = ('hf_quant_config.json', 'quantize_config.json', 'quant_config.json')


def _from_modelopt(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    quant = data.get('quantization') if isinstance(data.get('quantization'), dict) else data
    algo = quant.get('quant_algo')
    if not algo:
        return None
    out = {'quant_method': 'modelopt', 'quant_algo': algo}
    for key in ('group_size', 'kv_cache_quant_algo', 'exclude_modules'):
        if quant.get(key) is not None:
            out[key] = quant[key]
    return out


def _from_autogptq(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if 'bits' not in data:
        return None
    out = dict(data)
    out.setdefault('quant_method', 'gptq')
    return out


def _from_autoawq(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    bits = data.get('w_bit', data.get('bits'))
    if bits is None:
        return None
    return {'quant_method': 'awq', 'bits': bits,
            'group_size': data.get('q_group_size', data.get('group_size', 128)),
            'zero_point': data.get('zero_point', True)}


_NORMALISERS = {
    'hf_quant_config.json': _from_modelopt,
    'quantize_config.json': _from_autogptq,
    'quant_config.json': _from_autoawq,
}


def merge_quantization_sidecar(config: Dict[str, Any],
                               read_json: Callable[[str], Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """Return `config` with a sidecar's quantization folded in, when config.json declares none.

    Args:
        config: the parsed config.json.
        read_json: returns the parsed JSON of a sibling file by name, or None when it does not exist. Passing
            a reader rather than a path lets the same logic serve a local directory and a Hub repository.
    """
    if not isinstance(config, dict):
        return config
    if isinstance(config.get('quantization_config'), dict) and config['quantization_config']:
        return config
    if isinstance(config.get('compression_config'), dict) and config['compression_config']:
        return config
    for filename in SIDECAR_FILES:
        data = read_json(filename)
        if not isinstance(data, dict):
            continue
        quantization = _NORMALISERS[filename](data)
        if quantization:
            merged = dict(config)
            merged['quantization_config'] = quantization
            merged['_quantization_config_source'] = filename
            return merged
    return config


def local_reader(directory: str) -> Callable[[str], Optional[Dict[str, Any]]]:
    """A `read_json` for a checkpoint directory on disk."""
    root = Path(directory)

    def read(filename: str) -> Optional[Dict[str, Any]]:
        path = root / filename
        if not path.is_file():
            return None
        try:
            with open(path, 'r') as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return None

    return read
