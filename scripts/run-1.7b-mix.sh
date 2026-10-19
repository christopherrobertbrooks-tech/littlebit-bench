#!/usr/bin/env bash
# Qwen3-1.7B, 1-bit with blocks 0-7 and 27 at 2 bits from the start (2026-10-18, Chris: "run the bitnet training").
# Why: the block swap (results/blockswap-1.7b) showed those are the blocks 1-bit training rewrites hardest. Everything
# else is run-1.7b.sh's recipe (24,000 sequences, effective batch 4, lr 1.6e-4, 8-bit AdamW, evals and checkpoints
# every 500 steps), so its ppl_curve.jsonl compares step for step with results/qwen3-1.7b-b1.0. RESUME=1 continues.
source "$(dirname "$0")/env.sh"
here=$(cd "$(dirname "$0")" && pwd)
status=$RUNS/run-1.7b-mix.status
say() { echo "$(date '+%F %T') $*" | tee -a "$status"; }
[ -n "${LB_REPAIR_RANK:-}" ] && [ "$LB_REPAIR_RANK" != 0 ] && { say "refusing: LB_REPAIR_RANK is set"; exit 1; }
pgrep -f "[m]ain.py .*--f_name" > /dev/null && { say "refusing: a LittleBit training is already running"; exit 1; }
m=Qwen/Qwen3-1.7B; name=qwen3-1.7b-b1.0-mix2
say "training $name (blocks 0-7,27 at 2.0 bits, the rest 1.0)"
SAMPLES=24000 BS=1 ACC=4 DIRECT=True LR=1.6e-04 "$here/train.sh" "$name" "$m" 1.0 --eff_bit_layers "0-7,27:2.0" \
    --eval_every 500 --save_every 500 --fp16_init_scale 0.03125 --optim adamw_bnb_8bit ${RESUME:+--resume_from latest} > /dev/null 2>&1 \
    && say "trained" || say "training stopped or FAILED (see $RUNS/$name/train.log)"
if [ -f "$RUNS/$name/model.safetensors" ]; then
    "$here/eval.sh" "$name" "$RUNS/$name" fp16 > /dev/null 2>&1 && say "$name eval done" || say "$name eval FAILED"
fi
say "run-1.7b-mix finished"
