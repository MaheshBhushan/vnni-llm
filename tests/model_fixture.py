"""Deterministic tiny BF16 model shared by native smoke and MCP tests."""

import json
import struct


def write_model(directory):
    """One-layer, 32-wide fixture whose greedy output is always ASCII A."""
    config = {"model_type": "llama", "hidden_size": 32, "intermediate_size": 32,
              "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1,
              "vocab_size": 259, "max_position_embeddings": 4096, "rms_norm_eps": 1e-6,
              "tie_word_embeddings": True, "eos_token_id": 256}
    (directory / "config.json").write_text(json.dumps(config))
    extra, vocab = 0, {}
    for byte in range(256):
        printable = 33 <= byte <= 126 or 161 <= byte <= 172 or byte >= 174
        vocab[chr(byte if printable else 256 + extra)] = byte
        if not printable:
            extra += 1
    specials = [{"id": 256, "content": "<|endoftext|>"},
                {"id": 257, "content": "<|im_start|>"},
                {"id": 258, "content": "<|im_end|>"}]
    (directory / "tokenizer.json").write_text(json.dumps({
        "model": {"vocab": vocab, "merges": []}, "added_tokens": specials}))
    tensors, payload = {}, bytearray()

    def add(name, shape, values):
        start = len(payload)
        for value in values:
            payload.extend(struct.pack("<f", value)[2:])  # BF16
        tensors[name] = {"dtype": "BF16", "shape": shape,
                         "data_offsets": [start, len(payload)]}

    add("model.embed_tokens.weight", [259, 32],
        [(2.0 if row == 65 else 1.0) if col == 0 else 0.0
         for row in range(259) for col in range(32)])
    add("model.norm.weight", [32], [1.0] * 32)
    for name in ("input_layernorm", "post_attention_layernorm"):
        add(f"model.layers.0.{name}.weight", [32], [1.0] * 32)
    for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                 "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
        add(f"model.layers.0.{name}.weight", [32, 32], [0.0] * 1024)
    header = json.dumps(tensors).encode()
    header += b" " * (-len(header) % 8)
    (directory / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + payload)
