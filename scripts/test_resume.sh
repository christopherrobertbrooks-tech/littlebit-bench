#!/usr/bin/env bash
# Resumable checkpoints work: an 8-step run saves at steps 4 and 8; a second run resumed from step 4 must log the same
# losses for steps 5-8 as the first (same data order, weights, optimizer and loss scale). Prints PASS or FAIL.
set -uo pipefail
source "$(dirname "$0")/env.sh"
here=$(cd "$(dirname "$0")" && pwd)
rm -rf "$RUNS/test-resume-a" "$RUNS/test-resume-b"
common=(--save_every 4 --save_total_limit 3 --fp16_init_scale 0.125)
SAMPLES=32 "$here/train.sh" test-resume-a Qwen/Qwen3-0.6B 1.0 "${common[@]}" > /dev/null 2>&1
SAMPLES=32 "$here/train.sh" test-resume-b Qwen/Qwen3-0.6B 1.0 "${common[@]}" \
    --resume_from "$RUNS/test-resume-a/checkpoint-4" > /dev/null 2>&1
"$PY" - "$RUNS/test-resume-a/train.log" "$RUNS/test-resume-b/train.log" <<'PY'
import ast, sys
def losses(f):
    rows = [ast.literal_eval(l.strip()) for l in open(f).read().replace("\r", "\n").splitlines()
            if l.strip().startswith("{'loss'")]
    return [r["loss"] for r in rows]
a, b = losses(sys.argv[1]), losses(sys.argv[2])
print("run A steps 1-8:", a)
print("run B (resumed at 4):", b)
ok = len(a) == 8 and len(b) == 4 and all(abs(x - y) / x < 0.01 for x, y in zip(a[4:], b))
print("PASS" if ok else "FAIL")
PY
du -sh "$RUNS/test-resume-a/checkpoint-4" 2>/dev/null
