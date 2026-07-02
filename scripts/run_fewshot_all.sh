#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA_ROOT="${DATA_ROOT:-dataset/modelscope/extracted/classification_dataset}"
CHECKPOINT="${CHECKPOINT:-latest}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-runs/pretrain_2xh100}"
RUN_NAME="${RUN_NAME:-downstream_fewshot}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/${RUN_NAME}}"

DATASETS="${DATASETS:-mstar fusar_ship sar_acd}"
PROTOCOLS="${PROTOCOLS:-finetune linear}"
SHOTS="${SHOTS:-10 20 40}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"

EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-50}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
SUMMARY_METRIC="${SUMMARY_METRIC:-final_acc}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DEVICE="${DEVICE:-cuda}"

python Downstream/main_fewshot.py \
  --data_root "$DATA_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint_root "$CHECKPOINT_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --datasets $DATASETS \
  --protocols $PROTOCOLS \
  --shots $SHOTS \
  --seeds $SEEDS \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --summary_metric "$SUMMARY_METRIC" \
  --num_workers "$NUM_WORKERS" \
  --device "$DEVICE"
