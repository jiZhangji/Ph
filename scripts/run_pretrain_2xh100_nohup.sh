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
export TARGET_MODE="${TARGET_MODE:-sasgt}"
export INIT_CKPT="${INIT_CKPT:-}"
export INIT_CKPT_SCOPE="${INIT_CKPT_SCOPE:-encoder}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export ENCODER_LR_SCALE="${ENCODER_LR_SCALE:-1.0}"
export SASGT_SCALES="${SASGT_SCALES:-0.8,1.6,3.2,6.4}"
export SASGT_TEMPERATURE="${SASGT_TEMPERATURE:-1.0}"
export SASGT_GAMMA="${SASGT_GAMMA:-1.0}"
export SASGT_RELIABILITY_WINDOW="${SASGT_RELIABILITY_WINDOW:-7}"

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
  echo "TARGET_MODE=$TARGET_MODE"
  echo "INIT_CKPT=$INIT_CKPT"
  echo "INIT_CKPT_SCOPE=$INIT_CKPT_SCOPE"
  echo "SAVE_FREQ=$SAVE_FREQ"
  echo "ENCODER_LR_SCALE=$ENCODER_LR_SCALE"
  echo "SASGT_SCALES=$SASGT_SCALES"
  echo "SASGT_TEMPERATURE=$SASGT_TEMPERATURE"
  echo "SASGT_GAMMA=$SASGT_GAMMA"
  echo "SASGT_RELIABILITY_WINDOW=$SASGT_RELIABILITY_WINDOW"
} > "$LOG_FILE"

nohup bash scripts/pretrain_2xh100.sh >> "$LOG_FILE" 2>&1 &
PID="$!"

echo "Started 2xH100 pretraining."
echo "PID: $PID"
echo "Log: $LOG_FILE"
echo "Tail: tail -f $LOG_FILE"
