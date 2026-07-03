#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p logs

RUN_NAME="${RUN_NAME:-pretrain_2xh100_stable_rerun_bs256_lr1e-4}"
LOG_FILE="${LOG_FILE:-logs/${RUN_NAME}.log}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export DATA_PATH="${DATA_PATH:-dataset/modelscope/extracted/Pretraining_dataset}"
export OUTPUT_DIR="${OUTPUT_DIR:-runs/${RUN_NAME}}"

# Conservative defaults for the two-target physical pretraining objective.
# effective batch = 256 * 2 GPUs = 512, actual lr = 5e-5 * 512 / 256 = 1e-4.
export BATCH_SIZE="${BATCH_SIZE:-256}"
export BLR="${BLR:-5e-5}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-20}"
export EPOCHS="${EPOCHS:-300}"
export SAVE_FREQ="${SAVE_FREQ:-25}"
export NUM_WORKERS="${NUM_WORKERS:-16}"

export GRAD_LOSS_WEIGHT="${GRAD_LOSS_WEIGHT:-1.0}"
export LFST_LOSS_WEIGHT="${LFST_LOSS_WEIGHT:-1.0}"
export TARGET_NORM="${TARGET_NORM:-patch}"
export AMP_DTYPE="${AMP_DTYPE:-bf16}"
export INIT_CKPT="${INIT_CKPT:-}"
export INIT_CKPT_SCOPE="${INIT_CKPT_SCOPE:-encoder}"

{
  echo "RUN_NAME=$RUN_NAME"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "DATA_PATH=$DATA_PATH"
  echo "OUTPUT_DIR=$OUTPUT_DIR"
  echo "BATCH_SIZE=$BATCH_SIZE"
  echo "BLR=$BLR"
  echo "WARMUP_EPOCHS=$WARMUP_EPOCHS"
  echo "EPOCHS=$EPOCHS"
  echo "SAVE_FREQ=$SAVE_FREQ"
  echo "NUM_WORKERS=$NUM_WORKERS"
  echo "GRAD_LOSS_WEIGHT=$GRAD_LOSS_WEIGHT"
  echo "LFST_LOSS_WEIGHT=$LFST_LOSS_WEIGHT"
  echo "TARGET_NORM=$TARGET_NORM"
  echo "AMP_DTYPE=$AMP_DTYPE"
  echo "INIT_CKPT=$INIT_CKPT"
  echo "INIT_CKPT_SCOPE=$INIT_CKPT_SCOPE"
} > "$LOG_FILE"

nohup bash scripts/pretrain_2xh100.sh >> "$LOG_FILE" 2>&1 &
PID="$!"

echo "Started stable 2xH100 pretraining."
echo "PID: $PID"
echo "Log: $LOG_FILE"
echo "Tail: tail -f $LOG_FILE"
echo "Monitor: python scripts/monitor_pretrain_log.py --log $OUTPUT_DIR/log.txt"
