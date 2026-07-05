#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CHECKPOINT="${CHECKPOINT:-$ROOT/runs/sarjepa_pretrain_2xh100/checkpoint-299.pth}"
RUN_NAME="${RUN_NAME:-sarjepa_baseline_ckpt299}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/few_shot_classification/finetune/output_${RUN_NAME}}"
DATA_ROOT="${DATA_ROOT:-$ROOT/dataset/modelscope/extracted/classification_dataset/few_shot_classification}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DATASETS="${DATASETS:-mstar}"
PROTOCOLS="${PROTOCOLS:-finetune}"
SHOTS="${SHOTS:-10}"
SEEDS="${SEEDS:-0}"
EPOCHS="${EPOCHS:-40}"
LR="${LR:-1e-4}"
FORCE="${FORCE:-1}"

echo "SAR-JEPA baseline few-shot"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CHECKPOINT=$CHECKPOINT"
echo "DATA_ROOT=$DATA_ROOT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "DATASETS=$DATASETS"
echo "PROTOCOLS=$PROTOCOLS"
echo "SHOTS=$SHOTS"
echo "SEEDS=$SEEDS"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
CHECKPOINT="$CHECKPOINT" \
DATA_ROOT="$DATA_ROOT" \
OUTPUT_DIR="$OUTPUT_DIR" \
DATASETS="$DATASETS" \
PROTOCOLS="$PROTOCOLS" \
SHOTS="$SHOTS" \
SEEDS="$SEEDS" \
EPOCHS="$EPOCHS" \
LR="$LR" \
FORCE="$FORCE" \
USE_SFAFM=0 \
bash scripts/run_sarjepa_fewshot_all.sh
