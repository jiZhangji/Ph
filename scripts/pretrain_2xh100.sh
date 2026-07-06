#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BATCH_SIZE="${BATCH_SIZE:-512}"
DATA_PATH="${DATA_PATH:-dataset/modelscope/extracted/Pretraining_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/pretrain_2xh100}"
GPUS="${GPUS:-2}"
ACCUM_ITER="${ACCUM_ITER:-1}"
MODEL="${MODEL:-mae_vit_base_patch16}"
NUM_WORKERS="${NUM_WORKERS:-16}"
EPOCHS="${EPOCHS:-300}"
SAVE_FREQ="${SAVE_FREQ:-50}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
BLR="${BLR:-5e-5}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-20}"
GRAD_LOSS_WEIGHT="${GRAD_LOSS_WEIGHT:-1.0}"
LFST_LOSS_WEIGHT="${LFST_LOSS_WEIGHT:-1.0}"
TARGET_NORM="${TARGET_NORM:-patch}"
TARGET_MODE="${TARGET_MODE:-sasgt}"
INIT_CKPT="${INIT_CKPT:-}"
INIT_CKPT_SCOPE="${INIT_CKPT_SCOPE:-encoder}"
RESUME="${RESUME:-}"
ENCODER_LR_SCALE="${ENCODER_LR_SCALE:-1.0}"
SASGT_SCALES="${SASGT_SCALES:-0.8,1.6,3.2,6.4}"
SASGT_TEMPERATURE="${SASGT_TEMPERATURE:-1.0}"
SASGT_GAMMA="${SASGT_GAMMA:-1.0}"
SASGT_RELIABILITY_WINDOW="${SASGT_RELIABILITY_WINDOW:-7}"

torchrun --standalone --nproc_per_node="$GPUS" Pretraining/main_pretrain.py \
  --model "$MODEL" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --log_dir "$OUTPUT_DIR" \
  --device cuda \
  --amp_dtype "$AMP_DTYPE" \
  --batch_size "$BATCH_SIZE" \
  --accum_iter "$ACCUM_ITER" \
  --epochs "$EPOCHS" \
  --blr "$BLR" \
  --warmup_epochs "$WARMUP_EPOCHS" \
  --encoder_lr_scale "$ENCODER_LR_SCALE" \
  --grad_loss_weight "$GRAD_LOSS_WEIGHT" \
  --lfst_loss_weight "$LFST_LOSS_WEIGHT" \
  --target_norm "$TARGET_NORM" \
  --target_mode "$TARGET_MODE" \
  --sasgt_scales "$SASGT_SCALES" \
  --sasgt_temperature "$SASGT_TEMPERATURE" \
  --sasgt_gamma "$SASGT_GAMMA" \
  --sasgt_reliability_window "$SASGT_RELIABILITY_WINDOW" \
  --num_workers "$NUM_WORKERS" \
  --pin_mem \
  --window_size 7 \
  --num_window 4 \
  --mask_ratio 0.8 \
  --save_freq "$SAVE_FREQ" \
  --init_ckpt "$INIT_CKPT" \
  --init_ckpt_scope "$INIT_CKPT_SCOPE" \
  --resume "$RESUME"
