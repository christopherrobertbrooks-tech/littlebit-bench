#!/usr/bin/env bash
# The phone assistant's trained model at 1 bit (2026-10-13, Chris: "lets try our trained phone model with little bit
# and see what happens"). Teacher = q17-v3 (Qwen3-1.7B + its LoRA, merged, fp16). Student = warm start from the
# general 1-bit Qwen3-1.7B (the A100 continuation), distilled from the teacher on the phone model's own 4,315 training
# commands. Scored like every phone model: dense f16 GGUF -> llama-server with the action grammar -> the four quizzes
# (main 333, Chris 31, fresh 40, Muse 50) via phone-assistant scripts/baseline.sh --gguf.
# Stages: the teacher's own f16 score; step 0 (the general 1-bit model, no phone training); 3 epochs; 3 more.
# Refuses to start while the V100 is in use. Stop: kill the process group of this script (pgrep -f "[r]un-phone.sh").
source "$(dirname "$0")/../env.sh"
here=$(cd "$(dirname "$0")" && pwd)
PA=/mnt/steam/phone-assistant; TEACHER=$PA/runs/q17-v3/merged; GEN=$RUNS/qwen3-1.7b-b1.0-a100cont2
LCPP=/home/chris/llama.cpp-mainline
status=$RUNS/phone.status
say() { echo "$(date '+%F %T') $*" | tee -a "$status"; }
used=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$used" -gt 1000 ] && { say "refusing: the V100 has $used MiB in use"; exit 1; }

quiz() {  # quiz <LittleBit model dir> <tag>: dense export (checked against the 1-bit model), f16 GGUF, the quizzes
  local d=$RUNS/$2-dense
  if [ ! -f "$d/f16.gguf" ]; then
    (cd "$UP" && "$PY" "$here/export_dense.py" "$1" "$TEACHER" "$d") > "$RUNS/$2.export.log" 2>&1 \
      && "$PY" "$LCPP/convert_hf_to_gguf.py" "$d" --outtype f16 --outfile "$d/f16.gguf" >> "$RUNS/$2.export.log" 2>&1 \
      || { say "$2: export FAILED (see $2.export.log)"; return 1; }
    rm -f "$d/model.safetensors"                          # the GGUF is what gets scored
  fi
  say "$2: $(grep -h '^check:' "$RUNS/$2.export.log")"
  (cd "$PA" && scripts/baseline.sh --gguf "$d/f16.gguf" "$2") > "$RUNS/$2.quiz.log" 2>&1 \
    && say "$2: quizzed" || { say "$2: quiz FAILED (see $2.quiz.log)"; return 1; }
}

train() {  # train <name> <warm start dir> <epochs> <lr>: 16 commands per step, 8-bit AdamW
  EPOCHS=$3 BS=16 ACC=1 DIRECT=True LR=$4 ITQ=False "$here/../train.sh" "$1" "$TEACHER" 1.0 --dataset phone \
    --warm_start_from "$2" --fp16_init_scale 0.03125 --optim adamw_bnb_8bit > /dev/null 2>&1 \
    && [ -f "$RUNS/$1/model.safetensors" ] && say "$1: trained" || { say "$1: training FAILED"; return 1; }
}

say "START"
(cd "$UP" && "$PY" "$here/prep_phone.py" --model_id "$TEACHER" --data_root "$LB_ROOT/data" --pa_root "$PA") \
  > "$RUNS/phone-prep.log" 2>&1 && say "data: $(grep -h 'commands, padded' "$RUNS/phone-prep.log")" || { say "data prep FAILED"; exit 1; }
(cd "$PA" && scripts/baseline.sh --gguf "$PA/runs/q17-v3/f16.gguf" q17-v3-f16) > "$RUNS/q17-v3-f16.quiz.log" 2>&1 \
  && say "teacher f16: quizzed" || say "teacher f16: quiz FAILED"
quiz "$GEN" phone1bit-step0 || exit 1
train phone1bit-e3 "$GEN" 3 1e-4 && quiz "$RUNS/phone1bit-e3" phone1bit-e3 || exit 1
train phone1bit-e6 "$RUNS/phone1bit-e3" 3 5e-5 && quiz "$RUNS/phone1bit-e6" phone1bit-e6
say "PHONE DONE; V100 now $(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader)"
