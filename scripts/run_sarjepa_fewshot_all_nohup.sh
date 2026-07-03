#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p logs

RUN_NAME="${RUN_NAME:-sarjepa_fewshot_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-logs/${RUN_NAME}.log}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CHECKPOINT="${CHECKPOINT:-runs/pretrain_2xh100_stable_full_bs512/checkpoint-299.pth}"
export DATA_ROOT="${DATA_ROOT:-dataset/modelscope/extracted/classification_dataset/few_shot_classification}"
export OUTPUT_DIR="${OUTPUT_DIR:-few_shot_classification/finetune/output_${RUN_NAME}}"
export DATASETS="${DATASETS:-MSTAR_SOC New_FUSAR SAR_ACD}"
export PROTOCOLS="${PROTOCOLS:-MIM_finetune MIM_linear}"
export SHOTS="${SHOTS:-10 20 40}"
export SEEDS="${SEEDS:-0 1 2 3 4}"
export FORCE="${FORCE:-0}"

{
  echo "RUN_NAME=$RUN_NAME"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "CHECKPOINT=$CHECKPOINT"
  echo "DATA_ROOT=$DATA_ROOT"
  echo "OUTPUT_DIR=$OUTPUT_DIR"
  echo "DATASETS=$DATASETS"
  echo "PROTOCOLS=$PROTOCOLS"
  echo "SHOTS=$SHOTS"
  echo "SEEDS=$SEEDS"
  echo "FORCE=$FORCE"
} > "$LOG_FILE"

nohup bash scripts/run_sarjepa_fewshot_all.sh >> "$LOG_FILE" 2>&1 &
PID="$!"

echo "Started SAR-JEPA-style few-shot evaluation."
echo "PID: $PID"
echo "Log: $LOG_FILE"
echo "Tail: tail -f $LOG_FILE"
