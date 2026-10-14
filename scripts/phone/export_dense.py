"""Writes a finished LittleBit model out as an ordinary Hugging Face model with dense fp16 weights (2026-10-13), so
llama.cpp can run it (convert_hf_to_gguf.py -> f16 GGUF -> llama-server with the action grammar) and the phone
assistant's own quizzes score it. Nothing is re-quantized: each dense matrix is exactly what the 1-bit layer computes.

How: a LittleBit layer is linear in its input (weights only are binarized), so feeding it the identity matrix returns
its whole weight, W^T. That is done in fp32 on a float copy of the layer, then rounded to fp16 once. Every other tensor
(embeddings, norms, the head) is copied as it is.

Checked before saving: the dense model's logits against the LittleBit model's own on a few phone commands.

Run from the LittleBit checkout:  export_dense.py <LittleBit model dir> <base HF model dir> <out dir>
"""
import copy
import sys

import torch

sys.path.insert(0, ".")
import quantization.utils  # noqa: E402,F401  (must come before quantization.modules: circular import)
from quantization.hub import LittleBitModel  # noqa: E402
from quantization.modules import LittleBitLinear  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
# LittleBit builds the model skeleton on the meta device; importing Qwen3's code for the first time inside that runs
# a module-level .item() and fails ("cannot be called on meta tensors"), so import it here first.
import transformers.models.qwen3.modeling_qwen3  # noqa: E402,F401

src, base, out = sys.argv[1:4]
lb = LittleBitModel.from_pretrained(src, torch_dtype=torch.float16, device="cuda").model.eval()
dense = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16).cuda().eval()

with torch.no_grad():
    lb_mods = dict(lb.named_modules())
    quantized = {name for name, m in lb_mods.items() if isinstance(m, LittleBitLinear)}
    for name in sorted(quantized):
        mod, q = dense.get_submodule(name), lb_mods[name]
        f = copy.deepcopy(q).float()
        eye = torch.eye(f.in_features, device="cuda", dtype=torch.float32)
        mod.weight.copy_(f(eye).T.to(torch.float16))       # f(I) = I @ W^T
        assert mod.bias is None and q.bias is None
        del f, eye
    # everything else (embeddings, norms, an untied head) as the LittleBit model has it
    dsd = dense.state_dict()
    rest = {k: v for k, v in lb.state_dict().items() if k in dsd and k.rsplit(".", 1)[0] not in quantized}
    bad = [k for k in rest if dsd[k].shape != rest[k].shape]
    assert not bad, bad
    dense.load_state_dict(rest, strict=False)
    left = [k for k in dsd if k.rsplit(".", 1)[0] not in quantized and k not in rest]
    print(f"{len(quantized)} LittleBit layers made dense; {len(rest)} other tensors copied; not in the LittleBit "
          f"model (kept from the base): {left or 'none'}")

    tok = AutoTokenizer.from_pretrained(base)
    worst, agree, total = 0.0, 0, 0
    for say in ["set a timer for ten minutes", "text mom that I'm running late", "what's the weather like tomorrow"]:
        ids = tok.apply_chat_template([{"role": "user", "content": say}], add_generation_prompt=True,
                                      enable_thinking=False, return_tensors="pt").cuda()
        a = lb(ids).logits.float()
        b = dense(ids).logits.float()
        worst = max(worst, (a - b).abs().max().item())
        agree += (a.argmax(-1) == b.argmax(-1)).sum().item()
        total += ids.shape[1]
    print(f"check: top token agrees on {agree}/{total} positions, largest logit difference {worst:.3f}")
    assert agree >= 0.98 * total, "the dense copy does not reproduce the LittleBit model"

dense.save_pretrained(out, safe_serialization=True)
tok.save_pretrained(out)
print(f"saved {out}")
