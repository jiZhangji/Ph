#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PROTOCOL="${PROTOCOL:?Set PROTOCOL to MIM_finetune or MIM_linear}"
SHARD_ID="${SHARD_ID:?Set SHARD_ID}"
NUM_SHARDS="${NUM_SHARDS:?Set NUM_SHARDS}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-$ROOT/weights/sar-ssl-paper-baseline-weights-v1}"
OUTPUT_ROOT="${SPECKLE_OUTPUT_ROOT:-$ROOT/few_shot_classification/finetune/output_speckle_robustness_10seeds}"
PHYD_CHECKPOINT="${PHYD_CHECKPOINT:-$ROOT/runs/sarjepa_official_phyd_ft250_bs1024_lfst0p1_image_2xh200/checkpoint-300.pth}"

case "$PROTOCOL" in
  MIM_finetune)
    baseline_lr="${BASELINE_FINETUNE_LR:-1e-4}"
    phyd_lr="${PHYD_FINETUNE_LR:-1e-4}"
    ;;
  MIM_linear)
    baseline_lr="${BASELINE_LINEAR_LR:-1e-4}"
    phyd_lr="${PHYD_LINEAR_LR:-1e-3}"
    ;;
  *)
    echo "Unsupported protocol: $PROTOCOL"
    exit 2
    ;;
esac

echo "Protocol=$PROTOCOL shard=$SHARD_ID/$NUM_SHARDS"
echo "Baseline LR=$baseline_lr; PhyD-MAE LR=$phyd_lr"

env \
  WEIGHTS_ROOT="$WEIGHTS_ROOT" \
  PHYD_CHECKPOINT="$PHYD_CHECKPOINT" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  METHODS="mae lomar fg_mae i_jepa sar_jepa phyd_mae" \
  DATASETS="MSTAR_SOC New_FUSAR SAR_ACD" \
  PROTOCOLS="$PROTOCOL" \
  SHOTS="10 20 40" \
  SEEDS="0 1 2 3 4 5 6 7 8 9" \
  SPECKLE_LOOKS="clean 8 4 2 1" \
  NOISE_SEEDS="0 1 2" \
  LR="$baseline_lr" \
  PHYD_LR="$phyd_lr" \
  EPOCHS=40 \
  BATCH_SIZE=50 \
  FORCE=0 \
  NUM_SHARDS="$NUM_SHARDS" \
  SHARD_ID="$SHARD_ID" \
  SUMMARIZE=0 \
  bash scripts/run_speckle_robustness.sh

echo "Full controlled-speckle protocol worker complete"
