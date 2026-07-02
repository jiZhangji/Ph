#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GPUS="${GPUS:-2}"
BATCH_SIZE="${BATCH_SIZE:-512}"
ACCUM_ITER="${ACCUM_ITER:-1}"
DATA_PATH="${DATA_PATH:-dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/pretrain_2xa100}"
MODEL="${MODEL:-mae_vit_base_patch16}"
NUM_WORKERS="${NUM_WORKERS:-12}"
EPOCHS="${EPOCHS:-300}"
SAVE_FREQ="${SAVE_FREQ:-50}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
BLR="${BLR:-1e-3}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-20}"

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
  --num_workers "$NUM_WORKERS" \
  --pin_mem \
  --window_size 7 \
  --num_window 4 \
  --mask_ratio 0.8 \
  --save_freq "$SAVE_FREQ" \
  --init_ckpt ''
