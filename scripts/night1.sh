#!/usr/bin/env bash
# Night 1 (2026-10-11): one long LittleBit run on Qwen3-0.6B at 1.0 bit with a perplexity curve, then the full evals:
# that model, the fp16 baseline's zero-shot, and the llama.cpp quants of the same model.
source "$(dirname "$0")/env.sh"
here=$(cd "$(dirname "$0")" && pwd)
status=$RUNS/night1.status
say() { echo "$(date '+%F %T') $*" | tee -a "$status"; }
name=qwen3-0.6b-b1.0-long
say "training $name"
SAMPLES=${SAMPLES:-24000} LR=${LR:-1.6e-04} "$here/train.sh" "$name" Qwen/Qwen3-0.6B 1.0 --eval_every 500 > /dev/null 2>&1 \
    && say "trained" || say "training FAILED"
if [ -f "$RUNS/$name/model.safetensors" ]; then
    "$here/eval.sh" "$name" "$RUNS/$name" fp16 > /dev/null 2>&1 && say "$name eval done" || say "$name eval FAILED"
    grep -qi nan "$RUNS/$name/eval-fp16.json" 2>/dev/null && { "$here/eval.sh" "$name" "$RUNS/$name" fp32 > /dev/null 2>&1; say "$name fp32 eval done"; }
fi
"$here/eval.sh" qwen3-0.6b-fp Qwen/Qwen3-0.6B fp16 fp > /dev/null 2>&1 && say "baseline eval done" || say "baseline eval FAILED"
# A broken-enough model overflows fp16 (IQ1_S gave NaN perplexity); score those in fp32 instead.
for t in Q4_K_M Q3_K_M Q2_K IQ2_XXS IQ1_M IQ1_S; do
    d=qwen3-0.6b-gguf-$t
    "$here/eval.sh" "$d" "$RUNS/$d" fp16 fp > /dev/null 2>&1 && say "$t eval done" || say "$t eval FAILED"
    if grep -qi nan "$RUNS/$d/eval-fp16.json" 2>/dev/null; then
        "$here/eval.sh" "$d" "$RUNS/$d" fp32 fp > /dev/null 2>&1 && say "$t fp32 eval done" || say "$t fp32 eval FAILED"
    fi
done
say "night1 finished"
