#!/usr/bin/env bash
# train.sh <name> <HF model id> <eff_bit> [extra main.py args...]
# LittleBit QAT on one V100 in fp16. The defaults are the README's single-GPU recipe (SmoothSign, residual, KD + 10x
# layer-to-layer loss, lr 4e-5, warmup 2%) with LittleBit-2's Joint-ITQ init; SAMPLES/EPOCHS/BS/ACC size the run.
set -euo pipefail
source "$(dirname "$0")/env.sh"
name=$1; model=$2; bits=$3; shift 3
out=$RUNS/$name; mkdir -p "$out"
cd "$UP"
"$PY" main.py --model_id "$model" --dataset c4_wiki --data_root "$LB_ROOT/data" \
    --save_dir "$RUNS" --f_name "$name" \
    --num_train_epochs "${EPOCHS:-1}" --per_device_train_batch_size "${BS:-2}" --gradient_accumulation_steps "${ACC:-2}" \
    --lr "${LR:-4e-05}" --warmup_ratio 0.02 --report none \
    --quant_func SmoothSign --quant_mod LittleBitLinear --residual True --eff_bit "$bits" --kv_factor 1.0 \
    --min_split_dim 8 --l2l_loss_scale 10.0 --use_itq "${ITQ:-True}" \
    --precision "${PREC:-fp16}" --teacher_dtype "${TEACHER:-auto}" --max_train_samples "${SAMPLES:--1}" \
    --load_direct_to_gpu "${DIRECT:-False}" "$@" 2>&1 | tee "$out/train.log"
