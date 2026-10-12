#!/usr/bin/env bash
# Before the 1.7B run: (1) does 8-bit AdamW learn like plain AdamW? The 0.6B lr probe again (400 sequences, lr 1.6e-4,
# same seed) with adamw_bnb_8bit, against results/probe-lr1.6e-4. (2) The 1.7B's real memory peak, with the
# optimizer actually stepping (finite gradients from the first step), 8 steps.
source "$(dirname "$0")/env.sh"
here=$(cd "$(dirname "$0")" && pwd)
SAMPLES=400 LR=1.6e-04 "$here/train.sh" probe-lr1.6e-4-adam8 Qwen/Qwen3-0.6B 1.0 --optim adamw_bnb_8bit > /dev/null 2>&1
ZS=0 "$here/eval.sh" probe-lr1.6e-4-adam8 "$RUNS/probe-lr1.6e-4-adam8" fp16 > /dev/null 2>&1
rm -f "$RUNS/smoke-1.7b-mem.txt"
( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES" >> "$RUNS/smoke-1.7b-mem.txt"; sleep 1; done ) &
mon=$!
rm -rf "$RUNS/smoke-1.7b-adam8"
SAMPLES=32 BS=1 ACC=4 DIRECT=True LR=1.6e-04 "$here/train.sh" smoke-1.7b-adam8 Qwen/Qwen3-1.7B 1.0 \
    --optim adamw_bnb_8bit --fp16_init_scale 0.03125 > /dev/null 2>&1
echo "1.7B smoke exit $?"
kill $mon
echo "1.7B peak MiB: $(sort -n "$RUNS/smoke-1.7b-mem.txt" | tail -1)"
tr "\r" "\n" < "$RUNS/smoke-1.7b-adam8/train.log" | grep -a "grad_norm\|OutOfMemory" | cut -c1-120
echo CHECK8-DONE
