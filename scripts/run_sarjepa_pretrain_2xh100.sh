#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASELINE_DIR="${BASELINE_DIR:-$ROOT/baselines/SAR-JEPA}"

if [[ ! -d "$BASELINE_DIR/Pretraining" ]]; then
  bash "$ROOT/scripts/setup_sarjepa_baseline.sh"
fi

DATA_PATH="${DATA_PATH:-$ROOT/dataset/modelscope/extracted/Pretraining_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/sarjepa_pretrain_2xh100}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
GPUS="${GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-25642}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-300}"
BLR="${BLR:-1e-3}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-20}"
NUM_WORKERS="${NUM_WORKERS:-16}"
MODEL="${MODEL:-mae_vit_base_patch16}"
MASK_RATIO="${MASK_RATIO:-0.8}"
WINDOW_SIZE="${WINDOW_SIZE:-7}"
NUM_WINDOW="${NUM_WINDOW:-4}"

DATA_PATH="$(cd "$DATA_PATH" && pwd)"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
LOG_DIR="$(cd "$LOG_DIR" && pwd)"

echo "SAR-JEPA pretraining"
echo "BASELINE_DIR=$BASELINE_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "DATA_PATH=$DATA_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "EPOCHS=$EPOCHS"
echo "BLR=$BLR"

cd "$BASELINE_DIR/Pretraining"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
python -m torch.distributed.launch \
  --nproc_per_node="$GPUS" \
  --master_port="$MASTER_PORT" \
  main_pretrain.py \
  --model "$MODEL" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --log_dir "$LOG_DIR" \
  --device cuda \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --blr "$BLR" \
  --warmup_epochs "$WARMUP_EPOCHS" \
  --num_workers "$NUM_WORKERS" \
  --pin_mem \
  --window_size "$WINDOW_SIZE" \
  --num_window "$NUM_WINDOW" \
  --mask_ratio "$MASK_RATIO"
