#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

INHERITED_BATCH_SIZE="${BATCH_SIZE:-}"
INHERITED_OUTPUT_DIR="${OUTPUT_DIR:-}"

DATA_ROOT="${DATA_ROOT:-dataset/modelscope/extracted/classification_dataset}"
CHECKPOINT="${CHECKPOINT:-runs/pretrain_2xh100/checkpoint-299.pth}"
RUN_NAME="${RUN_NAME:-downstream_fewshot}"
OUTPUT_DIR="${DOWNSTREAM_OUTPUT_DIR:-runs/${RUN_NAME}}"

DATASETS="${DATASETS:-mstar fusar_ship sar_acd}"
PROTOCOLS="${PROTOCOLS:-finetune linear}"
SHOTS="${SHOTS:-10 20 40}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"

EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${DOWNSTREAM_BATCH_SIZE:-50}"
LR="${DOWNSTREAM_LR:-1e-3}"
WEIGHT_DECAY="${DOWNSTREAM_WEIGHT_DECAY:-5e-4}"
ADAM_BETA1="${DOWNSTREAM_ADAM_BETA1:-0.9}"
ADAM_BETA2="${DOWNSTREAM_ADAM_BETA2:-0.999}"
LR_SCHEDULER="${DOWNSTREAM_LR_SCHEDULER:-cosine}"
WARMUP_EPOCHS="${DOWNSTREAM_WARMUP_EPOCHS:-2}"
WARMUP_CONS_LR="${DOWNSTREAM_WARMUP_CONS_LR:-1e-5}"
TRAIN_AUG="${DOWNSTREAM_TRAIN_AUG:-none}"
NUM_WORKERS="${DOWNSTREAM_NUM_WORKERS:-0}"
DEVICE="${DEVICE:-cuda}"
EXTRA_ARGS=()

if [[ "${DOWNSTREAM_EVAL_TRAIN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--eval_train)
fi

if [[ -n "$INHERITED_BATCH_SIZE" && -z "${DOWNSTREAM_BATCH_SIZE:-}" ]]; then
  echo "WARNING: ignoring inherited BATCH_SIZE=$INHERITED_BATCH_SIZE; downstream uses BATCH_SIZE=$BATCH_SIZE"
fi
if [[ -n "$INHERITED_OUTPUT_DIR" && -z "${DOWNSTREAM_OUTPUT_DIR:-}" ]]; then
  echo "WARNING: ignoring inherited OUTPUT_DIR=$INHERITED_OUTPUT_DIR; downstream uses OUTPUT_DIR=$OUTPUT_DIR"
fi

python Downstream/main_fewshot.py \
  --data_root "$DATA_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$OUTPUT_DIR" \
  --datasets $DATASETS \
  --protocols $PROTOCOLS \
  --shots $SHOTS \
  --seeds $SEEDS \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --adam_beta1 "$ADAM_BETA1" \
  --adam_beta2 "$ADAM_BETA2" \
  --lr_scheduler "$LR_SCHEDULER" \
  --warmup_epochs "$WARMUP_EPOCHS" \
  --warmup_cons_lr "$WARMUP_CONS_LR" \
  --train_aug "$TRAIN_AUG" \
  --num_workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  "${EXTRA_ARGS[@]}"
