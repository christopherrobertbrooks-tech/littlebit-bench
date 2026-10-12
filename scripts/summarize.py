"""Results table from results/*/eval-*.json: perplexity and zero-shot accuracy per model, with bits per weight.
Zero-shot average = mean plain accuracy ("acc") over the 7 tasks. usage: summarize.py [results dir]"""
import json
import math
import sys
from pathlib import Path

TASKS = ["boolq", "piqa", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]
root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")


def best_eval(d):
    """fp16 result unless it overflowed (NaN), then fp32."""
    for dtype in ("fp16", "fp32"):
        f = d / f"eval-{dtype}.json"
        if f.exists():
            j = json.loads(f.read_text())
            r = j["results"]
            if not any(isinstance(v.get("ppl"), float) and math.isnan(v["ppl"]) for v in r.values()):
                limit = j["args"].get("limit", -1)
                return dtype + (f", {limit} per task" if limit not in (-1, None) else ""), r
    return None, None


def bits(d):
    g = d / "gguf_info.json"
    if g.exists():
        return f"{json.loads(g.read_text())['body_bits_per_weight']:.2f}"
    c = d / "littlebit_config.json"
    if c.exists():
        return f"{json.loads(c.read_text())['eff_bit']:.2f}"
    return "16"


rows = []
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    dtype, r = best_eval(d)
    if not r:
        continue
    zs = [r[t]["acc,none"] for t in TASKS if t in r]
    rows.append((d.name, bits(d), r.get("wikitext2", {}).get("ppl"), r.get("c4", {}).get("ppl"),
                 sum(zs) / len(zs) * 100 if len(zs) == len(TASKS) else None, dtype))

fmt = lambda x, p=2: "--" if x is None else f"{x:,.{p}f}"
print("| model | bits/weight (body) | wikitext2 PPL | c4 PPL | zero-shot avg % | eval dtype |")
print("| :--- | ---: | ---: | ---: | ---: | :--- |")
for name, b, w, c, z, dt in sorted(rows, key=lambda r: -float(r[1])):
    print(f"| {name} | {b} | {fmt(w)} | {fmt(c)} | {fmt(z, 1)} | {dt} |")
