#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p logs

RUN_NAME="${RUN_NAME:-pretrain_2xh100_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-logs/${RUN_NAME}.log}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export DATA_PATH="${DATA_PATH:-dataset/modelscope/extracted/Pretraining_dataset}"
export OUTPUT_DIR="${OUTPUT_DIR:-runs/${RUN_NAME}}"
export BATCH_SIZE="${BATCH_SIZE:-512}"
export BLR="${BLR:-5e-5}"
export GRAD_LOSS_WEIGHT="${GRAD_LOSS_WEIGHT:-1.0}"
export LFST_LOSS_WEIGHT="${LFST_LOSS_WEIGHT:-1.0}"
export TARGET_NORM="${TARGET_NORM:-patch}"
export INIT_CKPT="${INIT_CKPT:-}"
export INIT_CKPT_SCOPE="${INIT_CKPT_SCOPE:-encoder}"
export SAVE_FREQ="${SAVE_FREQ:-50}"

{
  echo "RUN_NAME=$RUN_NAME"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "DATA_PATH=$DATA_PATH"
  echo "OUTPUT_DIR=$OUTPUT_DIR"
  echo "BATCH_SIZE=$BATCH_SIZE"
  echo "BLR=$BLR"
  echo "GRAD_LOSS_WEIGHT=$GRAD_LOSS_WEIGHT"
  echo "LFST_LOSS_WEIGHT=$LFST_LOSS_WEIGHT"
  echo "TARGET_NORM=$TARGET_NORM"
  echo "INIT_CKPT=$INIT_CKPT"
  echo "INIT_CKPT_SCOPE=$INIT_CKPT_SCOPE"
  echo "SAVE_FREQ=$SAVE_FREQ"
} > "$LOG_FILE"

nohup bash scripts/pretrain_2xh100.sh >> "$LOG_FILE" 2>&1 &
PID="$!"

echo "Started 2xH100 pretraining."
echo "PID: $PID"
echo "Log: $LOG_FILE"
echo "Tail: tail -f $LOG_FILE"
