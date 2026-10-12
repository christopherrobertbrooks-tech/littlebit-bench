"""Where does a LittleBit training checkpoint overflow fp16? Loads checkpoint-N (latent factors, as trained), runs one
WikiText-2 window through it twice -- fp16 autocast as in training, then fp32 -- and reports the largest value coming
out of every decoder layer and every LittleBit layer, and the first one that is not finite.
usage (from the LittleBit checkout): find_overflow.py <checkpoint dir> <HF model id>"""
import argparse
import glob
import sys

import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, ".")
from quantization.utils import apply_littlebit_patch
from quantization.modules import LittleBitLinear
from utils.datautils import get_wikitext2, load_tokenizer

ck, model_id = sys.argv[1], sys.argv[2]
cfg = AutoConfig.from_pretrained(model_id)
args = argparse.Namespace(quant_func="SmoothSign", quant_mod="LittleBitLinear", residual=True, split_dim=1024,
                          eff_bit=1.0, kv_factor=1.0, min_split_dim=8, use_itq=True, itq_n_iter=50, model_id=model_id)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16)
model = apply_littlebit_patch(model, args, do_train=False)
sd = {}
for f in sorted(glob.glob(f"{ck}/model*.safetensors")):
    sd.update(load_file(f, device="cuda:0"))
res = model.load_state_dict(sd, strict=False, assign=True)
model.tie_weights()
model.to("cuda:0")
meta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.device.type == "meta"]
print("missing:", [k for k in res.missing_keys if "lm_head" not in k][:5], "| still on meta:", meta[:5])
model.eval()

ids = get_wikitext2(load_tokenizer(model_id)).input_ids[:, :2048].to("cuda:0")
stats = {}
def hook(name):
    def f(mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        fin = torch.isfinite(t)
        stats[name] = (t[fin].abs().max().item() if fin.any() else float("nan"), (~fin).sum().item())
    return f
names = []
for n, m in model.named_modules():
    if isinstance(m, LittleBitLinear) or n.endswith(tuple(f"layers.{i}" for i in range(cfg.num_hidden_layers))):
        m.register_forward_hook(hook(n)); names.append(n)

def run(label, autocast):
    stats.clear()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
        logits = model(ids, use_cache=False).logits.float()
    loss = torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:]).item()
    first_bad = next((n for n in names if stats.get(n, (0, 0))[1] > 0), None)
    print(f"\n== {label}: loss {loss:.4f}, first non-finite output: {first_bad}")
    top = sorted(((v[0], n) for n, v in stats.items() if v[0] == v[0]), reverse=True)[:8]
    for v, n in top:
        print(f"   {v:12.1f}  {n}")
    return stats.copy()

run("fp16 autocast (as in training)", True)
model.float()
run("fp32", False)
