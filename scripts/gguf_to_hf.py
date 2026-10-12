"""Turns a llama.cpp GGUF quant back into an fp16 Hugging Face model, so it can be scored by exactly the same eval as
LittleBit (same perplexity code, same zero-shot harness). Each weight is dequantized with gguf-py's reference code, so
the model carries precisely the rounding error of that quant type.
usage: gguf_to_hf.py <quant.gguf> <HF model dir (config + tokenizer)> <out dir>"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from accelerate import init_empty_weights
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM

import gguf
from gguf import GGUFReader, GGMLQuantizationType as Q
from gguf.quants import dequantize

src, hf_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
reader = GGUFReader(str(src))
tensors = {t.name: t for t in reader.tensors}

config = AutoConfig.from_pretrained(hf_dir)
with init_empty_weights():                       # (torch.device("meta") breaks the model module's import)
    skeleton = AutoModelForCausalLM.from_config(config)
arch = {"qwen3": gguf.MODEL_ARCH.QWEN3, "llama": gguf.MODEL_ARCH.LLAMA, "qwen2": gguf.MODEL_ARCH.QWEN2}[config.model_type]
names = gguf.get_tensor_name_map(arch, config.num_hidden_layers)

state, body_bits, body_weights, types = {}, 0, 0, {}
for hf_name, p in skeleton.state_dict().items():
    if hf_name == "lm_head.weight" and config.tie_word_embeddings:
        continue
    g = names.get_name(hf_name, try_suffixes=(".weight", ".bias"))
    t = tensors[g]
    w = dequantize(t.data, t.tensor_type).reshape(tuple(int(x) for x in reversed(t.shape)))
    assert tuple(w.shape) == tuple(p.shape), (hf_name, w.shape, p.shape)
    state[hf_name] = torch.from_numpy(np.ascontiguousarray(w)).to(torch.float16)
    if w.ndim == 2 and g not in ("token_embd.weight", "output.weight"):
        body_bits += int(t.n_bytes) * 8
        body_weights += int(t.n_elements)
        types[t.tensor_type.name] = types.get(t.tensor_type.name, 0) + int(t.n_elements)

out.mkdir(parents=True, exist_ok=True)
save_file(state, str(out / "model.safetensors"), metadata={"format": "pt"})
for f in hf_dir.iterdir():
    if f.suffix in (".json", ".txt", ".model") and f.name != "model.safetensors.index.json":
        shutil.copy(f, out / f.name)
info = {"gguf": src.name, "body_bits_per_weight": body_bits / body_weights, "body_weights": body_weights,
        "gguf_bytes": src.stat().st_size, "types_by_weights": types,
        "embeddings": tensors["token_embd.weight"].tensor_type.name}
(out / "gguf_info.json").write_text(json.dumps(info, indent=2))
print(json.dumps(info))
