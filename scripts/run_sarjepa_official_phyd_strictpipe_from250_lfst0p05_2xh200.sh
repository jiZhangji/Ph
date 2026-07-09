#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN_NAME="${RUN_NAME:-sarjepa_official_phyd_strictpipe_from250_bs1024_lfst0p05_image_2xh200}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
GPUS="${GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-25658}"

DATA_PATH="${DATA_PATH:-$ROOT/dataset/modelscope/extracted/Pretraining_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR}"
RESUME="${RESUME:-$ROOT/runs/sarjepa_official_phyd_2xh100/checkpoint-250.pth}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}"

# Two H200s: 512 samples per GPU is the largest configuration already
# observed to run reliably. The effective batch size is 1024.
BATCH_SIZE="${BATCH_SIZE:-512}"
ACCUM_ITER="${ACCUM_ITER:-1}"
EPOCHS="${EPOCHS:-400}"
BLR="${BLR:-3e-5}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-20}"
NUM_WORKERS="${NUM_WORKERS:-16}"

# Keep the spatial target dominant while retaining LFST as a weak physical
# regularizer. Image normalization was the strongest historical setting.
GRAD_LOSS_WEIGHT="${GRAD_LOSS_WEIGHT:-1.0}"
LFST_LOSS_WEIGHT="${LFST_LOSS_WEIGHT:-0.05}"
TARGET_NORM="${TARGET_NORM:-image}"

# Save a rolling checkpoint every epoch and fixed snapshots every 10 epochs
# from epoch 260 onward.
SAVE_EVERY_AFTER_EPOCH="${SAVE_EVERY_AFTER_EPOCH:-260}"
SAVE_INTERVAL_AFTER_EPOCH="${SAVE_INTERVAL_AFTER_EPOCH:-10}"

if [[ ! -f "$RESUME" ]]; then
  echo "Resume checkpoint not found: $RESUME"
  exit 1
fi

if [[ "$ALLOW_EXISTING_OUTPUT" != "1" ]] && {
  [[ -f "$OUTPUT_DIR/log.txt" ]] ||
  [[ -f "$OUTPUT_DIR/checkpoint-last.pth" ]];
}; then
  echo "Refusing to reuse non-empty training output: $OUTPUT_DIR"
  echo "Choose a new RUN_NAME, or set ALLOW_EXISTING_OUTPUT=1 only when intentionally resuming this new run."
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$ROOT/logs"

echo "Strict official-pipeline PhyD continuation"
echo "RUN_NAME=$RUN_NAME"
echo "RESUME=$RESUME"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "GPUS=$GPUS"
echo "BATCH_SIZE_PER_GPU=$BATCH_SIZE"
echo "EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * ACCUM_ITER * GPUS))"
echo "EPOCHS=$EPOCHS"
echo "BLR=$BLR"
echo "GRAD_LOSS_WEIGHT=$GRAD_LOSS_WEIGHT"
echo "LFST_LOSS_WEIGHT=$LFST_LOSS_WEIGHT"
echo "TARGET_NORM=$TARGET_NORM"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export RUN_NAME CUDA_VISIBLE_DEVICES GPUS MASTER_PORT
export DATA_PATH OUTPUT_DIR LOG_DIR RESUME
export BATCH_SIZE ACCUM_ITER EPOCHS BLR WARMUP_EPOCHS NUM_WORKERS
export GRAD_LOSS_WEIGHT LFST_LOSS_WEIGHT TARGET_NORM
export SAVE_EVERY_AFTER_EPOCH SAVE_INTERVAL_AFTER_EPOCH

bash "$ROOT/scripts/run_sarjepa_official_phyd_pretrain_2xh100.sh"
