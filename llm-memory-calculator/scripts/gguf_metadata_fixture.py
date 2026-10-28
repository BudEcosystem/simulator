#!/usr/bin/env python3
"""A GGUF's metadata, kept as a small fixture the KV gate's committed corpus can read.

The GGUF reader (``bud_component.gguf_to_hf_config``, with ``gguf_layers.hf_layer_keys_from_gguf``)
reads a file's metadata -- its architecture's keys -- and its tensor NAMES (a separate output head
means untied, an ``ffn_gate`` a gated FFN); it never reads a tensor's data. This copies every
metadata key of a real GGUF verbatim and every tensor under its own name with a one-element
placeholder, so the fixture resolves to the same config as the source file at a few hundred KB.
Two things are dropped and stated: the tokenizer's vocabulary arrays (megabytes of strings), whose
length is written as ``{arch}.vocab_size`` (the key llama.cpp reads first), and the tensors' data.

    python scripts/gguf_metadata_fixture.py /home/bud/models/qwen3.5-2b-gguf/Qwen3.5-2B-Q4_K_M.gguf \
        tests/fixtures/gguf/Qwen3.5-2B-Q4_K_M.metadata.gguf
"""

import sys

#: Tokenizer arrays the reader never needs (their length is the vocabulary size).
_DROPPED = ("tokenizer.ggml.tokens", "tokenizer.ggml.scores", "tokenizer.ggml.token_type",
            "tokenizer.ggml.merges", "tokenizer.chat_template")


def copy_metadata(src, out):
    import gguf
    import numpy as np

    reader = gguf.GGUFReader(src)
    arch = reader.fields["general.architecture"].contents()
    writer = gguf.GGUFWriter(out, arch)
    vocab = None
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        if key == "tokenizer.ggml.tokens":
            vocab = len(field.data)
        if key in _DROPPED:
            continue
        vtype = field.types[0]
        sub = field.types[1] if vtype == gguf.GGUFValueType.ARRAY and len(field.types) > 1 \
            else None
        writer.add_key_value(key, field.contents(), vtype, sub)
    if vocab is not None and f"{arch}.vocab_size" not in reader.fields:
        writer.add_uint32(f"{arch}.vocab_size", vocab)
    writer.add_string("general.description",
                      "metadata-only fixture (scripts/gguf_metadata_fixture.py): tensor data "
                      "replaced by one-element placeholders, tokenizer arrays dropped")
    for t in reader.tensors:
        writer.add_tensor(t.name, np.zeros(1, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


if __name__ == "__main__":
    copy_metadata(sys.argv[1], sys.argv[2])
    print(f"{sys.argv[1]} -> {sys.argv[2]}")
