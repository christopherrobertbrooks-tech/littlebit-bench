#!/usr/bin/env bash
# Qwen3-1.7B at 1.0 bit (started 2026-10-12): same recipe and token budget as night 1's 0.6B run (24,000 sequences,
# 49M tokens, effective batch 4, lr 1.6e-4), batch 1 x 4 accumulation and 8-bit AdamW to fit in 32 GB (plain AdamW's
# state for 1.36B latent weights is 11 GB: it ran out of memory on step 2), resumable checkpoints every 500 steps.
# First: the fp16 teacher must score like fp32. Last: the full evals, with the llama.cpp quants. RESUME=1 continues.
source "$(dirname "$0")/env.sh"
here=$(cd "$(dirname "$0")" && pwd)
status=$RUNS/run-1.7b.status
say() { echo "$(date '+%F %T') $*" | tee -a "$status"; }
m=Qwen/Qwen3-1.7B; tag=qwen3-1.7b; name=$tag-b1.0
[ -f "$RUNS/$tag-fp/eval-fp32.json" ] || for dt in fp16 fp32; do ZS=0 "$here/eval.sh" "$tag-fp" "$m" $dt fp > /dev/null 2>&1; done
say "teacher check: $(grep -h 'PPL =' "$RUNS/$tag-fp/eval-fp16.log" | tr '\n' ' ') vs fp32 $(grep -h 'PPL =' "$RUNS/$tag-fp/eval-fp32.log" | tr '\n' ' ')"
say "training $name"
SAMPLES=24000 BS=1 ACC=4 DIRECT=True LR=1.6e-04 "$here/train.sh" "$name" "$m" 1.0 \
    --eval_every 500 --save_every 500 --fp16_init_scale 0.03125 --optim adamw_bnb_8bit ${RESUME:+--resume_from latest} > /dev/null 2>&1 \
    && say "trained" || say "training FAILED"
if [ -f "$RUNS/$name/model.safetensors" ]; then
    "$here/eval.sh" "$name" "$RUNS/$name" fp16 > /dev/null 2>&1 && say "$name eval done" || say "$name eval FAILED"
    grep -qi nan "$RUNS/$name/eval-fp16.json" && { "$here/eval.sh" "$name" "$RUNS/$name" fp32 > /dev/null 2>&1; say "$name fp32 eval done"; }
fi
"$here/eval.sh" "$tag-fp" "$m" fp16 fp > /dev/null 2>&1 && say "baseline eval done" || say "baseline eval FAILED"
ZS=0 "$here/eval.sh" "$tag-gguf-F16" "$RUNS/$tag-gguf-F16" fp16 fp > /dev/null 2>&1 && say "f16 round trip: $(grep -h 'PPL =' "$RUNS/$tag-gguf-F16/eval-fp16.log" | tr '\n' ' ')"
for t in Q4_K_M Q3_K_M Q2_K IQ2_XXS IQ1_M IQ1_S; do
    d=$tag-gguf-$t
    [ -d "$RUNS/$d" ] || { say "$t missing"; continue; }
    "$here/eval.sh" "$d" "$RUNS/$d" fp16 fp > /dev/null 2>&1 && say "$t eval done" || say "$t eval FAILED"
    grep -qi nan "$RUNS/$d/eval-fp16.json" 2>/dev/null && { "$here/eval.sh" "$d" "$RUNS/$d" fp32 > /dev/null 2>&1; say "$t fp32 eval done"; }
done
say "run-1.7b finished"
