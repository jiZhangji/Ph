#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

FINE_TUNE_SHARD_ID="${FINE_TUNE_SHARD_ID:-}"
LINEAR_SHARD_ID="${LINEAR_SHARD_ID:-}"
FINE_TUNE_NUM_SHARDS="${FINE_TUNE_NUM_SHARDS:-38}"
LINEAR_NUM_SHARDS="${LINEAR_NUM_SHARDS:-52}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-$ROOT/weights/sar-ssl-paper-baseline-weights-v1}"
OUTPUT_ROOT="${BASELINE_OUTPUT_ROOT:-$ROOT/few_shot_classification/finetune/output_paper_baselines_10seeds}"

if [[ -z "$FINE_TUNE_SHARD_ID" && -z "$LINEAR_SHARD_ID" ]]; then
  echo "Set FINE_TUNE_SHARD_ID and/or LINEAR_SHARD_ID"
  exit 2
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

run_protocol() {
  local protocol="$1"
  local shard_id="$2"
  local num_shards="$3"

  echo "Protocol worker: protocol=$protocol shard=$shard_id/$num_shards"
  env \
    WEIGHTS_ROOT="$WEIGHTS_ROOT" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    METHODS="mae lomar fg_mae i_jepa sar_jepa" \
    DATASETS="MSTAR_SOC New_FUSAR SAR_ACD" \
    PROTOCOLS="$protocol" \
    SHOTS="10 20 40" \
    SEEDS="0 1 2 3 4 5 6 7 8 9" \
    LR=1e-4 \
    EPOCHS=40 \
    BATCH_SIZE=50 \
    FORCE=0 \
    NUM_SHARDS="$num_shards" \
    SHARD_ID="$shard_id" \
    bash scripts/run_paper_baseline_fewshot.sh
}

if [[ -n "$FINE_TUNE_SHARD_ID" ]]; then
  run_protocol MIM_finetune "$FINE_TUNE_SHARD_ID" "$FINE_TUNE_NUM_SHARDS"
fi

if [[ -n "$LINEAR_SHARD_ID" ]]; then
  run_protocol MIM_linear "$LINEAR_SHARD_ID" "$LINEAR_NUM_SHARDS"
fi

echo "Protocol worker complete"
