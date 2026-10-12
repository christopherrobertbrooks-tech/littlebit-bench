#!/usr/bin/env bash
# Can a finished (packed) LittleBit model be trained further? Warm start from night 1's model: its signs and scales,
# latent magnitude 0.03, fresh optimizer, lr 4e-5, 1,600 new sequences (seed 43). Step 0 must reproduce its
# perplexity (37.64); the curve then shows whether it holds and keeps improving. Checkpoints every 200 steps.
source "$(dirname "$0")/env.sh"
here=$(cd "$(dirname "$0")" && pwd)
name=qwen3-0.6b-b1.0-warm
SAMPLES=1600 LR=4e-05 "$here/train.sh" "$name" Qwen/Qwen3-0.6B 1.0 --warm_start_from "$RUNS/qwen3-0.6b-b1.0-long" \
    --eval_every 100 --save_every 200 --seed 43 > /dev/null 2>&1 && echo "trained" || echo "training FAILED"
[ -f "$RUNS/$name/model.safetensors" ] && ZS=0 "$here/eval.sh" "$name" "$RUNS/$name" fp16 > /dev/null 2>&1
cat "$RUNS/$name/ppl_curve.jsonl"; grep -a "PPL =" "$RUNS/$name/eval-fp16.log"
echo WARM-DONE
