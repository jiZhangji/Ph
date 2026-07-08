#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN_NAME="${RUN_NAME:-sarjepa_official_phyd_ft250_bs1024_lfst0p1_image_2xh200}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
GPUS="${GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-25655}"

DATA_PATH="${DATA_PATH:-$ROOT/dataset/modelscope/extracted/Pretraining_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR}"
RESUME="${RESUME:-$ROOT/runs/sarjepa_official_phyd_2xh100/checkpoint-250.pth}"

# H200-safe large-batch continuation. BATCH_SIZE is per GPU.
BATCH_SIZE="${BATCH_SIZE:-512}"
ACCUM_ITER="${ACCUM_ITER:-1}"
EPOCHS="${EPOCHS:-420}"
BLR="${BLR:-3e-5}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-0}"
NUM_WORKERS="${NUM_WORKERS:-16}"

GRAD_LOSS_WEIGHT="${GRAD_LOSS_WEIGHT:-1.0}"
LFST_LOSS_WEIGHT="${LFST_LOSS_WEIGHT:-0.1}"
TARGET_NORM="${TARGET_NORM:-image}"

# Keep fixed checkpoints after 300 so long continuation runs are easy to evaluate.
SAVE_EVERY_AFTER_EPOCH="${SAVE_EVERY_AFTER_EPOCH:-300}"
SAVE_INTERVAL_AFTER_EPOCH="${SAVE_INTERVAL_AFTER_EPOCH:-10}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export RUN_NAME CUDA_VISIBLE_DEVICES GPUS MASTER_PORT
export DATA_PATH OUTPUT_DIR LOG_DIR RESUME
export BATCH_SIZE ACCUM_ITER EPOCHS BLR WARMUP_EPOCHS NUM_WORKERS
export GRAD_LOSS_WEIGHT LFST_LOSS_WEIGHT TARGET_NORM
export SAVE_EVERY_AFTER_EPOCH SAVE_INTERVAL_AFTER_EPOCH

bash "$ROOT/scripts/run_sarjepa_official_phyd_pretrain_2xh100.sh"
