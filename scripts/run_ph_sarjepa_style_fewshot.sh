#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CHECKPOINT="${CHECKPOINT:-runs/ph_sarjepa_style_2xh100/checkpoint-299.pth}"
RUN_NAME="${RUN_NAME:-ph_sarjepa_style_ckpt299}"
OUTPUT_DIR="${OUTPUT_DIR:-few_shot_classification/finetune/output_${RUN_NAME}}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DATASETS="${DATASETS:-mstar}"
PROTOCOLS="${PROTOCOLS:-finetune}"
SHOTS="${SHOTS:-10}"
SEEDS="${SEEDS:-0}"
EPOCHS="${EPOCHS:-40}"
LR="${LR:-1e-4}"
FORCE="${FORCE:-1}"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
CHECKPOINT="$CHECKPOINT" \
RUN_NAME="$RUN_NAME" \
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
