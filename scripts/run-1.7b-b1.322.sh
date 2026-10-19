#!/usr/bin/env bash
# Qwen3-1.7B at 1.322 bits EVERYWHERE (2026-10-19, Chris): the fair control for run-1.7b-mix.sh, the same size as the mixed
# model (19 blocks at 1 bit + 9 at 2 = 1.322 bits on average), so it says whether WHERE the extra bits go matters.
# Same recipe, data and seed as run-1.7b.sh and run-1.7b-mix.sh.
source "$(dirname "$0")/env.sh"
here=$(cd "$(dirname "$0")" && pwd)
status=$RUNS/run-1.7b-b1.322.status
say() { echo "$(date '+%F %T') $*" | tee -a "$status"; }
[ -n "${LB_REPAIR_RANK:-}" ] && [ "$LB_REPAIR_RANK" != 0 ] && { say "refusing: LB_REPAIR_RANK is set"; exit 1; }
pgrep -f "[m]ain.py .*--f_name" > /dev/null && { say "refusing: a LittleBit training is already running"; exit 1; }
m=Qwen/Qwen3-1.7B; name=qwen3-1.7b-b1.322
say "training $name (1.322 bits in every block)"
SAMPLES=24000 BS=1 ACC=4 DIRECT=True LR=1.6e-04 "$here/train.sh" "$name" "$m" 1.322 \
    --eval_every 500 --save_every 500 --fp16_init_scale 0.03125 --optim adamw_bnb_8bit ${RESUME:+--resume_from latest} > /dev/null 2>&1 \
    && say "trained" || say "training stopped or FAILED (see $RUNS/$name/train.log)"
if [ -f "$RUNS/$name/model.safetensors" ]; then
    "$here/eval.sh" "$name" "$RUNS/$name" fp16 > /dev/null 2>&1 && say "$name eval done" || say "$name eval FAILED"
fi
say "run-1.7b-b1.322 finished"
