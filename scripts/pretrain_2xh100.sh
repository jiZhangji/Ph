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
BLR="${BLR:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-20}"
GRAD_LOSS_WEIGHT="${GRAD_LOSS_WEIGHT:-1.0}"
LFST_LOSS_WEIGHT="${LFST_LOSS_WEIGHT:-10.0}"
INIT_CKPT="${INIT_CKPT:-}"
INIT_CKPT_SCOPE="${INIT_CKPT_SCOPE:-encoder}"

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
  --grad_loss_weight "$GRAD_LOSS_WEIGHT" \
  --lfst_loss_weight "$LFST_LOSS_WEIGHT" \
  --num_workers "$NUM_WORKERS" \
  --pin_mem \
  --window_size 7 \
  --num_window 4 \
  --mask_ratio 0.8 \
  --save_freq "$SAVE_FREQ" \
  --init_ckpt "$INIT_CKPT" \
  --init_ckpt_scope "$INIT_CKPT_SCOPE"
