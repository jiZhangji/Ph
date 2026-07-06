#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN_NAME="${RUN_NAME:-ph_target_sarjepa_2xh100}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/$RUN_NAME}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
RUN_NAME="$RUN_NAME" \
OUTPUT_DIR="$OUTPUT_DIR" \
TARGET_MODE=sarjepa \
GRAD_LOSS_WEIGHT="${GRAD_LOSS_WEIGHT:-1.0}" \
LFST_LOSS_WEIGHT="${LFST_LOSS_WEIGHT:-0.0}" \
TARGET_NORM="${TARGET_NORM:-patch}" \
BATCH_SIZE="${BATCH_SIZE:-256}" \
BLR="${BLR:-5e-5}" \
WARMUP_EPOCHS="${WARMUP_EPOCHS:-20}" \
EPOCHS="${EPOCHS:-300}" \
SAVE_FREQ="${SAVE_FREQ:-50}" \
INIT_CKPT="${INIT_CKPT:-weights/mae_pretrain_vit_base.pth}" \
INIT_CKPT_SCOPE="${INIT_CKPT_SCOPE:-encoder}" \
bash scripts/pretrain_2xh100.sh
