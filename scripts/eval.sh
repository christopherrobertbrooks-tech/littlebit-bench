#!/usr/bin/env bash
# eval.sh <name> <model dir or HF id> <dtype> [fp]   -- perplexity (wikitext2, c4 @ 2048) + the paper's 7 zero-shot tasks.
# Pass "fp" as the 4th argument for an uncompressed baseline. ZS=0 skips the zero-shot tasks.
set -euo pipefail
source "$(dirname "$0")/env.sh"
name=$1; model=$2; dtype=$3; fp=${4:-}
out=$RUNS/$name; mkdir -p "$out"
tasks="boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa"
[ "${ZS:-1}" = 0 ] && tasks=""
cd "$UP"
"$PY" eval.py --model_id "$model" --fp_eval "$([ "$fp" = fp ] && echo True || echo False)" --dtype "$dtype" \
    --ppl_task wikitext2,c4 --zeroshot_task "$tasks" --seqlen 2048 --batch_size "${BS:-8}" \
    --limit "${LIMIT:--1}" --results_json "$out/eval-$dtype.json" 2>&1 | tee "$out/eval-$dtype.log"
