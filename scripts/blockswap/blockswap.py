#!/usr/bin/env python3
"""Block swap (2026-10-18, Chris: "yes, try the block swap"): where does 1-bit LittleBit do its damage?

The 1-bit Qwen3-1.7B (A100 continuation) keeps embeddings, lm_head and norms in fp16 already; only the 7 linear layers
of each of the 28 blocks are 1-bit. Here chosen layers are swapped back to the ORIGINAL fp16 weights at inference time
(no training), and each configuration gets WikiText-2 perplexity on the first N windows of 2048 tokens (LittleBit's own
eval code and data) plus three greedy sample paragraphs to read. Then a single-block sweep: each block swapped alone.
Writes results.json + samples.md next to --out.
"""
import argparse, json, os, sys, time
sys.path.insert(0, "/mnt/steam/littlebit/LittleBit")
import quantization.utils  # noqa: F401  (must come before quantization.modules: circular import)
import torch, torch.nn as nn
import torch._dynamo  # noqa: F401  (as eval.py: imported before LittleBit builds the model on the meta device)
import transformers.models.qwen3.modeling_qwen3  # noqa: F401
from safetensors import safe_open
from quantization.hub import LittleBitModel
from utils.datautils import get_eval_loaders, load_tokenizer

ATTN = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
MLP = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
PROMPTS = ["The history of the city of Rome", "To make a good cup of coffee, you should", "Once upon a time, a small robot"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onebit", default="/mnt/steam/littlebit-final-models/qwen3-1.7b-b1.0-a100cont2")
    ap.add_argument("--orig", default="/mnt/nvme/models/phone/Qwen3-1.7B")
    ap.add_argument("--windows", type=int, default=40); ap.add_argument("--out", required=True)
    ap.add_argument("--sweep", action="store_true", help="also every block alone (perplexity only)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda:0"
    model = LittleBitModel.from_pretrained(a.onebit, torch_dtype=torch.float16, device=dev)
    model.eval()
    tok = load_tokenizer(a.onebit)
    n_layers = model.config.num_hidden_layers
    orig = {}
    for f in sorted(os.listdir(a.orig)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(a.orig, f), framework="pt", device=dev) as s:   # straight to the GPU (16 GB RAM)
                for k in s.keys():
                    if ".layers." in k and k.endswith("_proj.weight"):
                        orig[k] = s.get_tensor(k).to(torch.float16)
    enc = get_eval_loaders("wikitext2", tok).input_ids
    seq = 2048; nwin = min(a.windows, enc.numel() // seq)

    @torch.no_grad()
    def ppl():
        nll = 0.0
        for i in range(nwin):
            b = enc[:, i * seq:(i + 1) * seq].to(dev)
            lg = model(b, use_cache=False).logits[:, :-1].float()
            nll += nn.functional.cross_entropy(lg.reshape(-1, lg.size(-1)), b[:, 1:].reshape(-1), reduction="sum").item()
        return float(torch.exp(torch.tensor(nll / (nwin * (seq - 1)))))

    @torch.no_grad()
    def samples():
        out = []
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt").input_ids.to(dev)
            g = model.generate(ids, max_new_tokens=60, do_sample=False, repetition_penalty=1.0)
            out.append(tok.decode(g[0], skip_special_tokens=True))
        return out

    try:
        model.get_submodule("model.layers"); root = model
    except AttributeError:
        root = model.model   # LittleBitModel wraps the transformers model one level deeper

    def swap(names):
        saved = []
        for n in names:
            parent, attr = n.rsplit(".", 1)
            pm = root.get_submodule(parent); old = getattr(pm, attr); w = orig[n + ".weight"]
            lin = nn.Linear(w.shape[1], w.shape[0], bias=False, device=dev, dtype=torch.float16); lin.weight.data = w
            setattr(pm, attr, lin); saved.append((pm, attr, old))
        return saved

    def restore(saved):
        for pm, attr, old in saved: setattr(pm, attr, old)

    def layers(idx, parts=ATTN + MLP): return [f"model.layers.{i}.{p}" for i in idx for p in parts]
    L = n_layers - 1
    configs = [("1-bit as trained (nothing swapped)", []), ("everything back to fp16 (the original)", layers(range(n_layers))),
               ("first block (0)", layers([0])), (f"last block ({L})", layers([L])), (f"first + last (0, {L})", layers([0, L])),
               ("first 4 blocks (0-3)", layers(range(4))), (f"last 4 blocks ({L - 3}-{L})", layers(range(L - 3, n_layers))),
               ("all attention layers", layers(range(n_layers), ATTN)), ("all MLP layers", layers(range(n_layers), MLP))]
    res = []; md = ["# Block swap samples (greedy, 60 tokens)\n"]
    per_layer_params = sum(orig[f"model.layers.0.{p}.weight"].numel() for p in ATTN + MLP)
    for name, names in configs:
        t0 = time.time(); saved = swap(names); p = ppl(); smp = samples(); restore(saved)
        extra = sum(orig[n + ".weight"].numel() for n in names) * 2 / 2**20
        res.append({"config": name, "swapped": len(names), "ppl": p, "extra_fp16_MiB": round(extra), "samples": smp})
        print(f"{name:42s} ppl {p:8.2f}   +{extra:6.0f} MiB fp16   ({time.time() - t0:.0f} s)", flush=True)
        md += [f"\n## {name} -- perplexity {p:.2f}\n"] + [f"- {s!r}\n" for s in smp]
        json.dump(res, open(f"{a.out}/results.json", "w"), indent=1); open(f"{a.out}/samples.md", "w").writelines(md)
    if a.sweep:
        sw = []
        for i in range(n_layers):
            saved = swap(layers([i])); p = ppl(); restore(saved); sw.append({"block": i, "ppl": p})
            print(f"block {i:2d} alone: ppl {p:8.2f}", flush=True)
            json.dump({"configs": res, "sweep": sw}, open(f"{a.out}/results.json", "w"), indent=1)
    print(f"per block: {per_layer_params / 1e6:.1f}M weights = {per_layer_params * 2 / 2**20:.0f} MiB in fp16; {nwin} windows of {seq}")


if __name__ == "__main__":
    main()
