#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BEST_CHECKPOINT="${BEST_CHECKPOINT:-$ROOT/runs/sarjepa_official_phyd_ft250_bs1024_lfst0p1_image_2xh200/checkpoint-300.pth}"
RUN_NAME="${RUN_NAME:-sarjepa_official_phyd_sfafm_from_best300_g1_l1_bs1024_2xh200}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/$RUN_NAME}"

if [[ ! -f "$BEST_CHECKPOINT" ]]; then
  echo "Missing initialization checkpoint: $BEST_CHECKPOINT" >&2
  exit 1
fi

if [[ -e "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Output directory is not empty: $OUTPUT_DIR" >&2
  echo "Choose a new RUN_NAME, or use RESUME with the generic runner." >&2
  exit 1
fi

export RUN_NAME
export OUTPUT_DIR
export LOG_DIR="${LOG_DIR:-$OUTPUT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export GPUS="${GPUS:-2}"
export MASTER_PORT="${MASTER_PORT:-25731}"
export DATA_PATH="${DATA_PATH:-$ROOT/dataset/modelscope/extracted/Pretraining_dataset}"
export INIT_CHECKPOINT="$BEST_CHECKPOINT"
export RESUME=""

# 512 samples per H200 gives an effective batch size of 1024.
export BATCH_SIZE="${BATCH_SIZE:-512}"
export ACCUM_ITER="${ACCUM_ITER:-1}"
export EPOCHS="${EPOCHS:-50}"
export BLR="${BLR:-1e-5}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
export NUM_WORKERS="${NUM_WORKERS:-16}"

export GRAD_LOSS_WEIGHT="${GRAD_LOSS_WEIGHT:-1.0}"
export LFST_LOSS_WEIGHT="${LFST_LOSS_WEIGHT:-1.0}"
export TARGET_NORM="${TARGET_NORM:-image}"
export USE_SFAFM=1
export SFAFM_REDUCTION="${SFAFM_REDUCTION:-4}"

export SAVE_EVERY_AFTER_EPOCH="${SAVE_EVERY_AFTER_EPOCH:-0}"
export SAVE_INTERVAL_AFTER_EPOCH="${SAVE_INTERVAL_AFTER_EPOCH:-10}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

exec bash scripts/run_sarjepa_official_phyd_pretrain_2xh100.sh
