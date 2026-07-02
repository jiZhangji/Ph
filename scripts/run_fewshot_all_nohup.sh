#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p logs

RUN_NAME="${RUN_NAME:-downstream_fewshot_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-logs/${RUN_NAME}.log}"
export RUN_NAME

{
  echo "RUN_NAME=$RUN_NAME"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
  echo "DATA_ROOT=${DATA_ROOT:-dataset/modelscope/extracted/classification_dataset}"
  echo "CHECKPOINT=${CHECKPOINT:-runs/pretrain_2xh100/checkpoint-299.pth}"
  echo "OUTPUT_DIR=${DOWNSTREAM_OUTPUT_DIR:-runs/${RUN_NAME}}"
  echo "DATASETS=${DATASETS:-mstar fusar_ship sar_acd}"
  echo "PROTOCOLS=${PROTOCOLS:-finetune linear}"
  echo "SHOTS=${SHOTS:-10 20 40}"
  echo "SEEDS=${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
  echo "EPOCHS=${EPOCHS:-40}"
  echo "BATCH_SIZE=${DOWNSTREAM_BATCH_SIZE:-50}"
  echo "LR=${DOWNSTREAM_LR:-1e-3}"
  echo "WEIGHT_DECAY=${DOWNSTREAM_WEIGHT_DECAY:-5e-4}"
  echo "ADAM_BETA1=${DOWNSTREAM_ADAM_BETA1:-0.9}"
  echo "ADAM_BETA2=${DOWNSTREAM_ADAM_BETA2:-0.999}"
  echo "LR_SCHEDULER=${DOWNSTREAM_LR_SCHEDULER:-cosine}"
  echo "WARMUP_EPOCHS=${DOWNSTREAM_WARMUP_EPOCHS:-2}"
  echo "WARMUP_CONS_LR=${DOWNSTREAM_WARMUP_CONS_LR:-1e-5}"
  echo "TRAIN_AUG=${DOWNSTREAM_TRAIN_AUG:-none}"
  echo "NUM_WORKERS=${DOWNSTREAM_NUM_WORKERS:-0}"
  echo "EVAL_TRAIN=${DOWNSTREAM_EVAL_TRAIN:-0}"
} > "$LOG_FILE"

nohup bash scripts/run_fewshot_all.sh >> "$LOG_FILE" 2>&1 &
PID="$!"

echo "Started downstream few-shot evaluation."
echo "PID: $PID"
echo "Log: $LOG_FILE"
echo "Tail: tail -f $LOG_FILE"
