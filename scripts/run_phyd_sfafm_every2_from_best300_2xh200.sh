#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BEST_CHECKPOINT="${BEST_CHECKPOINT:-$ROOT/runs/sarjepa_official_phyd_ft250_bs1024_lfst0p1_image_2xh200/checkpoint-300.pth}"
RUN_NAME="${RUN_NAME:-phyd_sfafm_every2end_from_best300_g1_lfst0p1_image_300e_2xh200}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/$RUN_NAME}"

if [[ ! -f "$BEST_CHECKPOINT" ]]; then
  echo "Missing initialization checkpoint: $BEST_CHECKPOINT" >&2
  exit 1
fi

if [[ -e "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Output directory is not empty: $OUTPUT_DIR" >&2
  echo "Use a new RUN_NAME, or resume it with the generic runner." >&2
  exit 1
fi

export RUN_NAME
export OUTPUT_DIR
export LOG_DIR="${LOG_DIR:-$OUTPUT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export GPUS="${GPUS:-2}"
export MASTER_PORT="${MASTER_PORT:-25841}"
export DATA_PATH="${DATA_PATH:-$ROOT/dataset/modelscope/extracted/Pretraining_dataset}"
export INIT_CHECKPOINT="$BEST_CHECKPOINT"
export RESUME=""

# Seven SFAFM modules add activation memory; start conservatively at 384 samples per H200.
export BATCH_SIZE="${BATCH_SIZE:-384}"
export ACCUM_ITER="${ACCUM_ITER:-1}"
export EPOCHS="${EPOCHS:-300}"

# Protect the pretrained backbone while allowing the identity-initialized SFAFM modules to learn faster.
export BLR="${BLR:-1e-6}"
export SFAFM_LR_SCALE="${SFAFM_LR_SCALE:-10.0}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
export NUM_WORKERS="${NUM_WORKERS:-16}"

export GRAD_LOSS_WEIGHT="${GRAD_LOSS_WEIGHT:-1.0}"
export LFST_LOSS_WEIGHT="${LFST_LOSS_WEIGHT:-0.1}"
export TARGET_NORM="${TARGET_NORM:-image}"
export USE_SFAFM=1
export SFAFM_LAYOUT=every2_end
export SFAFM_REDUCTION="${SFAFM_REDUCTION:-4}"

export SAVE_EVERY_AFTER_EPOCH="${SAVE_EVERY_AFTER_EPOCH:-0}"
export SAVE_INTERVAL_AFTER_EPOCH="${SAVE_INTERVAL_AFTER_EPOCH:-10}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

exec bash scripts/run_sarjepa_official_phyd_pretrain_2xh100.sh
