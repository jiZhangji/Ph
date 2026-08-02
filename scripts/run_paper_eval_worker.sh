#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

: "${BASELINE_SHARD_ID:?Set BASELINE_SHARD_ID}"

BASELINE_NUM_SHARDS="${BASELINE_NUM_SHARDS:-26}"
SPECKLE_SHARD_ID="${SPECKLE_SHARD_ID:-}"
SPECKLE_NUM_SHARDS="${SPECKLE_NUM_SHARDS:-10}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-$ROOT/weights/sar-ssl-paper-baseline-weights-v1}"
BASELINE_OUTPUT_ROOT="${BASELINE_OUTPUT_ROOT:-$ROOT/few_shot_classification/finetune/output_paper_baselines_10seeds}"
SPECKLE_OUTPUT_ROOT="${SPECKLE_OUTPUT_ROOT:-$ROOT/few_shot_classification/finetune/output_speckle_robustness_10seeds}"
PHYD_CHECKPOINT="${PHYD_CHECKPOINT:-$ROOT/runs/sarjepa_official_phyd_ft250_bs1024_lfst0p1_image_2xh200/checkpoint-300.pth}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

echo "Paper evaluation worker"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Baseline shard=$BASELINE_SHARD_ID/$BASELINE_NUM_SHARDS"
echo "Speckle shard=${SPECKLE_SHARD_ID:-none}/$SPECKLE_NUM_SHARDS"

env \
  WEIGHTS_ROOT="$WEIGHTS_ROOT" \
  OUTPUT_ROOT="$BASELINE_OUTPUT_ROOT" \
  METHODS="mae lomar fg_mae i_jepa sar_jepa" \
  DATASETS="MSTAR_SOC New_FUSAR SAR_ACD" \
  PROTOCOLS="MIM_finetune MIM_linear" \
  SHOTS="10 20 40" \
  SEEDS="0 1 2 3 4 5 6 7 8 9" \
  LR=1e-4 \
  EPOCHS=40 \
  BATCH_SIZE=50 \
  FORCE=0 \
  NUM_SHARDS="$BASELINE_NUM_SHARDS" \
  SHARD_ID="$BASELINE_SHARD_ID" \
  bash scripts/run_paper_baseline_fewshot.sh

if [[ -n "$SPECKLE_SHARD_ID" ]]; then
  env \
    WEIGHTS_ROOT="$WEIGHTS_ROOT" \
    PHYD_CHECKPOINT="$PHYD_CHECKPOINT" \
    OUTPUT_ROOT="$SPECKLE_OUTPUT_ROOT" \
    METHODS="mae lomar fg_mae i_jepa sar_jepa phyd_mae" \
    DATASETS="MSTAR_SOC" \
    PROTOCOLS="MIM_linear" \
    SHOTS="20" \
    SEEDS="0 1 2 3 4 5 6 7 8 9" \
    SPECKLE_LOOKS="clean 8 4 2 1" \
    NOISE_SEEDS="0 1 2" \
    LR=1e-4 \
    EPOCHS=40 \
    BATCH_SIZE=50 \
    FORCE=0 \
    NUM_SHARDS="$SPECKLE_NUM_SHARDS" \
    SHARD_ID="$SPECKLE_SHARD_ID" \
    bash scripts/run_speckle_robustness.sh
fi

echo "Paper evaluation worker complete"
