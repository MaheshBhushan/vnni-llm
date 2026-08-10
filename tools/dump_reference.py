#!/usr/bin/env python3
"""Dump HF reference activations for the golden-logit test.

Usage: dump_reference.py <model_dir> <out_dir> [prompt]

Writes (raw little-endian f32, last prompt token only):
  tokens.txt, layer_<i>.bin, final_norm.bin, logits.bin
"""
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir, out_dir = sys.argv[1], sys.argv[2]
prompt = sys.argv[3] if len(sys.argv) > 3 else (
    "The quick brown fox jumps over the lazy dog. "
    "In a distant galaxy, scientists discovered"
)
os.makedirs(out_dir, exist_ok=True)

tok = AutoTokenizer.from_pretrained(model_dir)
ids = tok(prompt, return_tensors="pt").input_ids
with open(f"{out_dir}/tokens.txt", "w") as f:
    for t in ids[0].tolist():
        f.write(f"{t}\n")
print(f"{ids.shape[1]} tokens")

# float32 so the reference itself has no bf16 accumulation error; weights are
# upcast bf16->f32 exactly as the C++ side reads them.
model = AutoModelForCausalLM.from_pretrained(
    model_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True
)
model.eval()

captured = {}


def hook(idx):
    def fn(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured[f"layer_{idx}"] = h[0, -1].detach().float()
    return fn


for i, layer in enumerate(model.model.layers):
    layer.register_forward_hook(hook(i))
model.model.norm.register_forward_hook(
    lambda _m, _i, out: captured.__setitem__("final_norm", out[0, -1].detach().float())
)

with torch.no_grad():
    logits = model(ids).logits[0, -1].float()

for name, t in captured.items():
    t.numpy().tofile(f"{out_dir}/{name}.bin")
logits.numpy().tofile(f"{out_dir}/logits.bin")
print(f"argmax: {int(logits.argmax())} ({tok.decode([int(logits.argmax())])!r})")
print(f"wrote {len(captured) + 1} arrays to {out_dir}")
