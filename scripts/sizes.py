"""What a saved LittleBit model is made of: bytes per part, and the real bits per weight of the compressed layers.
usage: sizes.py <model dir> [--json out.json]"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open


def part(name):
    if "embed_tokens" in name:
        return "embeddings"
    if name.startswith("lm_head"):
        return "lm_head"
    if name.endswith("_packed"):
        return "binary factors (packed)"
    if name.endswith("_shape") or "_eff_bit" in name or "_split_dim" in name:
        return "bookkeeping"
    if any(name.endswith(s) for s in (".u1", ".u2", ".v1", ".v2", ".u1_R", ".u2_R", ".v1_R", ".v2_R")):
        return "scales"
    if "norm" in name:
        return "norms"
    return "other"


def main(model_dir, out=None):
    sizes, weights = defaultdict(int), 0
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(str(f), "pt") as st:
            for k in st.keys():
                t = st.get_tensor(k)
                sizes[part(k)] += t.numel() * t.element_size()
                if k.endswith(".U_shape"):              # U is [out, split]; V_shape gives [split, in]
                    v = st.get_tensor(k.replace(".U_shape", ".V_shape"))
                    weights += int(t[0]) * int(v[1])
    total = sum(sizes.values())
    body = sizes["binary factors (packed)"] + sizes["scales"]
    res = {"bytes": dict(sizes), "total_bytes": total, "compressed_layer_weights": weights,
           "body_bits_per_weight": body * 8 / weights if weights else None}
    for k, v in sorted(sizes.items(), key=lambda kv: -kv[1]):
        print(f"{k:26s} {v / 2**20:9.1f} MiB")
    print(f"{'total':26s} {total / 2**20:9.1f} MiB")
    if weights:
        print(f"compressed layers: {weights / 1e6:.1f}M weights at {res['body_bits_per_weight']:.3f} bits each "
              f"(binary factors + scales)")
    if out:
        Path(out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--json" else None)
