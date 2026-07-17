#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA_PATH="${DATA_PATH:-dataset/modelscope/extracted/Pretraining_dataset}"
RUN_NAME="${RUN_NAME:-sarjepa_official_phyd_2xh100}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
GPUS="${GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-25645}"

# Keep SAR-JEPA official training settings by default.
BATCH_SIZE="${BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-300}"
ACCUM_ITER="${ACCUM_ITER:-1}"
BLR="${BLR:-1e-3}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-20}"
NUM_WORKERS="${NUM_WORKERS:-16}"
MODEL="${MODEL:-mae_vit_base_patch16}"
MASK_RATIO="${MASK_RATIO:-0.8}"
WINDOW_SIZE="${WINDOW_SIZE:-7}"
NUM_WINDOW="${NUM_WINDOW:-4}"
RESUME="${RESUME:-}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
INIT_SCOPE="${INIT_SCOPE:-full}"
SAVE_EVERY_AFTER_EPOCH="${SAVE_EVERY_AFTER_EPOCH:--1}"
SAVE_INTERVAL_AFTER_EPOCH="${SAVE_INTERVAL_AFTER_EPOCH:-10}"

# PhyD-only target/loss switches.
GRAD_LOSS_WEIGHT="${GRAD_LOSS_WEIGHT:-1.0}"
LFST_LOSS_WEIGHT="${LFST_LOSS_WEIGHT:-1.0}"
TARGET_NORM="${TARGET_NORM:-patch}"
LFST_CUTOFF="${LFST_CUTOFF:-30}"
LFST_INPUT_MODE="${LFST_INPUT_MODE:-raw}"
LFST_TARGET_TYPE="${LFST_TARGET_TYPE:-lfst}"
SASGT_SCALES="${SASGT_SCALES:-0.8,1.6,3.2,6.4}"
SASGT_TEMPERATURE="${SASGT_TEMPERATURE:-1.0}"
SASGT_GAMMA="${SASGT_GAMMA:-1.0}"
SASGT_RELIABILITY_WINDOW="${SASGT_RELIABILITY_WINDOW:-7}"
USE_SFAFM="${USE_SFAFM:-0}"
SFAFM_REDUCTION="${SFAFM_REDUCTION:-4}"
SFAFM_LAYOUT="${SFAFM_LAYOUT:-late}"
SFAFM_LR_SCALE="${SFAFM_LR_SCALE:-1.0}"
CLIP_GRAD="${CLIP_GRAD:-}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "SAR-JEPA official framework + PhyD targets"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "DATA_PATH=$DATA_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "EPOCHS=$EPOCHS"
echo "BLR=$BLR"
echo "MASK_RATIO=$MASK_RATIO"
echo "WINDOW_SIZE=$WINDOW_SIZE"
echo "NUM_WINDOW=$NUM_WINDOW"
echo "GRAD_LOSS_WEIGHT=$GRAD_LOSS_WEIGHT"
echo "LFST_LOSS_WEIGHT=$LFST_LOSS_WEIGHT"
echo "TARGET_NORM=$TARGET_NORM"
echo "LFST_CUTOFF=$LFST_CUTOFF"
echo "LFST_INPUT_MODE=$LFST_INPUT_MODE"
echo "LFST_TARGET_TYPE=$LFST_TARGET_TYPE"
echo "USE_SFAFM=$USE_SFAFM"
echo "SFAFM_REDUCTION=$SFAFM_REDUCTION"
echo "SFAFM_LAYOUT=$SFAFM_LAYOUT"
echo "SFAFM_LR_SCALE=$SFAFM_LR_SCALE"
echo "CLIP_GRAD=${CLIP_GRAD:-disabled}"
echo "INIT_CHECKPOINT=$INIT_CHECKPOINT"
echo "INIT_SCOPE=$INIT_SCOPE"
echo "SAVE_EVERY_AFTER_EPOCH=$SAVE_EVERY_AFTER_EPOCH"
echo "SAVE_INTERVAL_AFTER_EPOCH=$SAVE_INTERVAL_AFTER_EPOCH"

extra_args=()
if [[ "$USE_SFAFM" == "1" ]]; then
  extra_args+=(--use_sfafm)
fi
if [[ -n "$CLIP_GRAD" ]]; then
  extra_args+=(--clip_grad "$CLIP_GRAD")
fi

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
python -m torch.distributed.launch \
  --nproc_per_node="$GPUS" \
  --master_port="$MASTER_PORT" \
  Pretraining_sarjepa_official_phyd/main_pretrain.py \
  --model "$MODEL" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --log_dir "$LOG_DIR" \
  --save_every_after_epoch "$SAVE_EVERY_AFTER_EPOCH" \
  --save_interval_after_epoch "$SAVE_INTERVAL_AFTER_EPOCH" \
  --device cuda \
  --batch_size "$BATCH_SIZE" \
  --accum_iter "$ACCUM_ITER" \
  --epochs "$EPOCHS" \
  --blr "$BLR" \
  --warmup_epochs "$WARMUP_EPOCHS" \
  --num_workers "$NUM_WORKERS" \
  --pin_mem \
  --window_size "$WINDOW_SIZE" \
  --num_window "$NUM_WINDOW" \
  --mask_ratio "$MASK_RATIO" \
  --resume "$RESUME" \
  --init_checkpoint "$INIT_CHECKPOINT" \
  --init_scope "$INIT_SCOPE" \
  --lfst_cutoff "$LFST_CUTOFF" \
  --lfst_input_mode "$LFST_INPUT_MODE" \
  --lfst_target_type "$LFST_TARGET_TYPE" \
  --grad_loss_weight "$GRAD_LOSS_WEIGHT" \
  --lfst_loss_weight "$LFST_LOSS_WEIGHT" \
  --target_norm "$TARGET_NORM" \
  --sasgt_scales "$SASGT_SCALES" \
  --sasgt_temperature "$SASGT_TEMPERATURE" \
  --sasgt_gamma "$SASGT_GAMMA" \
  --sasgt_reliability_window "$SASGT_RELIABILITY_WINDOW" \
  --sfafm_reduction "$SFAFM_REDUCTION" \
  --sfafm_layout "$SFAFM_LAYOUT" \
  --sfafm_lr_scale "$SFAFM_LR_SCALE" \
  "${extra_args[@]}"
