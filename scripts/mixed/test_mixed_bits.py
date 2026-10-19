#!/usr/bin/env python3
"""--eff_bit_layers: the named decoder blocks get their own eff_bit, every other block keeps --eff_bit, and the saved
config rebuilds the same shapes. A tiny random Qwen3 on the CPU (28 blocks like the 1.7B), no download.

    cd LittleBit && ../venv/bin/python ../bench/scripts/mixed/test_mixed_bits.py
"""
import argparse, json, os, re, sys, tempfile

sys.path.insert(0, os.getcwd())
import torch
from transformers import Qwen3Config, AutoModelForCausalLM
from quantization.utils import apply_littlebit_patch, parse_eff_bit_layers   # utils before modules (circular import)
from quantization.modules import LittleBitLinear

assert parse_eff_bit_layers("0-7,27:2.0") == [([0, 1, 2, 3, 4, 5, 6, 7, 27], 2.0)]
assert parse_eff_bit_layers("1:1.5; 3-4:0.5") == [([1], 1.5), ([3, 4], 0.5)]
assert parse_eff_bit_layers(None) == [] and parse_eff_bit_layers("") == []

cfg = Qwen3Config(vocab_size=512, hidden_size=256, intermediate_size=768, num_hidden_layers=28, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=64, tie_word_embeddings=True)


def build(spec):
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(cfg)
    args = argparse.Namespace(quant_func="SmoothSign", quant_mod="LittleBitLinear", residual=True, split_dim=1024,
                              eff_bit=1.0, eff_bit_layers=spec, kv_factor=1.0, min_split_dim=8, use_itq=False,
                              itq_n_iter=50, model_id="tiny-qwen3")
    return apply_littlebit_patch(m, args, do_train=True)


def per_block(m):
    out = {}
    for name, mod in m.named_modules():
        if isinstance(mod, LittleBitLinear):
            b = int(re.search(r"\.layers\.(\d+)\.", name).group(1))
            out.setdefault(b, set()).add(round(mod.eff_bit_target, 4))
    return out


plain, mixed = per_block(build(None)), per_block(build("0-7,27:2.0"))
assert all(v == {1.0} for v in plain.values()), plain
for b, v in mixed.items():
    assert v == ({2.0} if b <= 7 or b == 27 else {1.0}), (b, v)
n_plain = sum(p.numel() for n, p in build(None).named_parameters() if ".layers." in n)
n_mixed = sum(p.numel() for n, p in build("0-7,27:2.0").named_parameters() if ".layers." in n)
assert n_mixed > n_plain, (n_mixed, n_plain)
print(f"OK: blocks 0-7 + 27 at 2.0 bits, the other 19 at 1.0; block parameters {n_plain:,} -> {n_mixed:,} "
      f"(x{n_mixed / n_plain:.2f})")
