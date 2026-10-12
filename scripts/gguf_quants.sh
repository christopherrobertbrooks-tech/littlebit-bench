#!/usr/bin/env bash
# gguf_quants.sh <tag> <HF model id> [types...]  -- llama.cpp quants of the same model, for comparison. CPU only.
# f16 GGUF -> importance matrix from WikiText-2 *train* (LittleBit also trains on it) -> each quant type, with the
# embeddings and output kept f16 like LittleBit's -> dequantized back to an HF model in $RUNS/<tag>-<type>.
set -euo pipefail
source "$(dirname "$0")/env.sh"
tag=$1; model=$2; shift 2
types=("$@"); [ ${#types[@]} -eq 0 ] && types=(Q4_K_M Q3_K_M Q2_K IQ2_XXS IQ1_M IQ1_S)
here=$(cd "$(dirname "$0")" && pwd)
LCPP=${LCPP:-/home/chris/llama.cpp-mainline}
BIN=$LB_ROOT/llama-build/bin
work=$LB_ROOT/gguf/$tag; mkdir -p "$work"
hf_dir=$("$PY" -c "from huggingface_hub import snapshot_download as s; print(s('$model'))")
export PYTHONPATH=$LCPP/gguf-py

[ -f "$work/f16.gguf" ] || "$PY" "$LCPP/convert_hf_to_gguf.py" "$hf_dir" --outtype f16 --outfile "$work/f16.gguf"
[ -f "$work/calib.txt" ] || "$PY" -c "
from datasets import load_dataset
t = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')['text']
open('$work/calib.txt', 'w').write(''.join(t))"
[ -f "$work/imatrix.gguf" ] || nice -n 10 "$BIN/llama-imatrix" -m "$work/f16.gguf" -f "$work/calib.txt" \
    -o "$work/imatrix.gguf" -c 512 --chunks 200 -t 8
# the f16 round trip first: it must score exactly like the original model, or the converter is wrong
"$PY" "$here/gguf_to_hf.py" "$work/f16.gguf" "$hf_dir" "$RUNS/$tag-gguf-F16"
for t in "${types[@]}"; do
    [ -f "$work/$t.gguf" ] || nice -n 10 "$BIN/llama-quantize" --imatrix "$work/imatrix.gguf" \
        --token-embedding-type f16 --output-tensor-type f16 "$work/f16.gguf" "$work/$t.gguf" "$t" 8
    "$PY" "$here/gguf_to_hf.py" "$work/$t.gguf" "$hf_dir" "$RUNS/$tag-gguf-$t"
done
echo GGUF-DONE
