"""Stub chakra package for testing purposes."""
import sys as _sys

# Make this package importable as top-level "chakra"
_sys.modules.setdefault('chakra', _sys.modules[__name__])

# Ensure submodules mapped
from types import ModuleType as _Mod

for sub in ['schema', 'schema.protobuf', 'schema.protobuf.et_def_pb2', 'et_def', 'et_def.et_def_pb2']:
    full_name = f'chakra.{sub}'
    if full_name not in _sys.modules:
        module = _Mod(full_name)
        _sys.modules[full_name] = module

# Provide minimal stubs
class GlobalMetadata:  # type: ignore
    pass

_sys.modules['chakra.schema.protobuf.et_def_pb2'].GlobalMetadata = GlobalMetadata
_sys.modules['chakra.et_def.et_def_pb2'] = _Mod('chakra.et_def.et_def_pb2')
_sys.modules['chakra.et_def.et_def_pb2'].GlobalMetadata = GlobalMetadata 