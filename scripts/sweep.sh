#!/usr/bin/env bash
# sweep.sh <tag> <HF model id> <samples> [bits...]  -- one model, every bit level: train, then full eval.
# A failed level is logged and skipped; the next one still runs. Progress goes to $RUNS/<tag>-sweep.status.
source "$(dirname "$0")/env.sh"
tag=$1; model=$2; samples=$3; shift 3
bits=("$@"); [ ${#bits[@]} -eq 0 ] && bits=(1.0 0.55 0.3 0.1)
here=$(cd "$(dirname "$0")" && pwd)
status=$RUNS/$tag-sweep.status
say() { echo "$(date '+%F %T') $*" | tee -a "$status"; }

say "start: $model, $samples samples per level, bits ${bits[*]}"
if [ ! -f "$RUNS/$tag-fp/eval-fp16.json" ] || ! grep -q hellaswag "$RUNS/$tag-fp/eval-fp16.json"; then
    "$here/eval.sh" "$tag-fp" "$model" fp16 fp > /dev/null 2>&1 && say "baseline eval done" || say "baseline eval FAILED"
fi
for b in "${bits[@]}"; do
    name=$tag-b$b-s$samples
    say "$name: training"
    if SAMPLES=$samples "$here/train.sh" "$name" "$model" "$b" > /dev/null 2>&1 && [ -f "$RUNS/$name/model.safetensors" ]; then
        say "$name: evaluating"
        "$here/eval.sh" "$name" "$RUNS/$name" fp16 > /dev/null 2>&1 && say "$name: done" || say "$name: eval FAILED"
    else
        say "$name: training FAILED"
    fi
done
say "sweep finished"
