"""Top-level stub chakra package for tests."""
import sys as _sys, types as _types

# dynamically create submodules chain used in tests
for sub in [
    'schema',
    'schema.protobuf',
    'schema.protobuf.et_def_pb2',
    'et_def',
    'et_def.et_def_pb2']:
    mod = _types.ModuleType(f'chakra.{sub}')
    _sys.modules[f'chakra.{sub}'] = mod

class GlobalMetadata:  # type: ignore
    pass

_sys.modules['chakra.schema.protobuf.et_def_pb2'].GlobalMetadata = GlobalMetadata
_sys.modules['chakra.et_def.et_def_pb2'].GlobalMetadata = GlobalMetadata

# Provide minimal Node, NodeType, AttributeProto classes
class _StubEnum(int):
    def __new__(cls, value):
        obj = int.__new__(cls, value)
        return obj

class NodeType(_StubEnum):
    COMM_COLL_NODE = 0
    COMM_SEND_NODE = 1
    COMM_RECV_NODE = 2

class AttributeProto:  # type: ignore
    def __init__(self, name: str = '', **kwargs):
        self.name = name
        # allow dynamic attributes
        for k, v in kwargs.items():
            setattr(self, k, v)
    def WhichOneof(self, _):
        return 'int_val'

class Node:  # type: ignore
    def __init__(self):
        self.attr = []
        self.type = 0

# attach to modules
for modname in ['chakra.schema.protobuf.et_def_pb2', 'chakra.et_def.et_def_pb2']:
    mod = _sys.modules[modname]
    mod.Node = Node
    mod.NodeType = NodeType
    mod.AttributeProto = AttributeProto

# Stub protolib functions
proto_mods = ['chakra.third_party.utils.protolib', 'chakra.src.third_party.utils.protolib']
for modname in proto_mods:
    base_mod = _types.ModuleType(modname)
    def _noop(*args, **kwargs):
        return None
    base_mod.encodeMessage = lambda *args, **kwargs: None
    base_mod.decodeMessage = lambda *args, **kwargs: False
    base_mod.openFileRd = lambda *args, **kwargs: None
    _sys.modules[modname] = base_mod 