#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SHARD_ID="${SHARD_ID:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-$ROOT/weights/sar-ssl-paper-baseline-weights-v1}"
FEATURE_ROOT="${FEATURE_ROOT:-$ROOT/few_shot_classification/finetune/pretrained_tsne_features}"
RUN_ROOT="${RUN_ROOT:-$ROOT/few_shot_classification/finetune/output_pretrained_tsne}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

if [[ ! "$NUM_SHARDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_SHARDS must be positive" >&2
  exit 2
fi
if [[ ! "$SHARD_ID" =~ ^[0-9]+$ ]] || (( SHARD_ID >= NUM_SHARDS )); then
  echo "SHARD_ID must be in [0, $((NUM_SHARDS - 1))]" >&2
  exit 2
fi

resolve_checkpoint() {
  case "$1" in
    mae) echo "$WEIGHTS_ROOT/weights/mae/checkpoint-200.pth" ;;
    lomar) echo "$WEIGHTS_ROOT/weights/lomar/checkpoint-200.pth" ;;
    fg_mae) echo "$WEIGHTS_ROOT/weights/fg_mae/checkpoint-200.pth" ;;
    i_jepa) echo "$WEIGHTS_ROOT/weights/i_jepa/jepa-ep200.pth.tar" ;;
    sar_jepa) echo "$WEIGHTS_ROOT/weights/sar_jepa/checkpoint-200.pth" ;;
    phyd_mae)
      echo "$ROOT/runs/sarjepa_official_phyd_ft250_bs1024_lfst0p1_image_2xh200/checkpoint-300.pth"
      ;;
    *) echo "Unknown method: $1" >&2; return 2 ;;
  esac
}

methods=(mae lomar fg_mae i_jepa sar_jepa phyd_mae)
datasets=(MSTAR_SOC New_FUSAR SAR_ACD)
mkdir -p "$FEATURE_ROOT" "$RUN_ROOT"

job_index=0
selected_jobs=0
for dataset in "${datasets[@]}"; do
  for method in "${methods[@]}"; do
    current_job=$job_index
    job_index=$((job_index + 1))
    if (( current_job % NUM_SHARDS != SHARD_ID )); then
      continue
    fi
    selected_jobs=$((selected_jobs + 1))

    feature_file="$FEATURE_ROOT/$dataset/$method.npz"
    if [[ -s "$feature_file" ]]; then
      echo "Skip completed feature: $feature_file"
      continue
    fi

    checkpoint="$(resolve_checkpoint "$method")"
    if [[ ! -f "$checkpoint" ]]; then
      echo "Missing checkpoint: $checkpoint" >&2
      exit 1
    fi

    echo "============================================================"
    echo "Frozen pretrained feature: method=$method dataset=$dataset"
    echo "checkpoint=$checkpoint"
    echo "output=$feature_file"
    echo "============================================================"

    mkdir -p "$(dirname "$feature_file")"
    env \
      PYTHONUNBUFFERED=1 \
      SHARD_ID=0 \
      NUM_SHARDS=1 \
      CHECKPOINT="$checkpoint" \
      OUTPUT_DIR="$RUN_ROOT/$method" \
      MODEL_FAMILY="$method" \
      DATASETS="$dataset" \
      PROTOCOLS="MIM_linear" \
      SHOTS="10" \
      SEEDS="0" \
      BATCH_SIZE="50" \
      USE_SFAFM=0 \
      FEATURE_POOL=cls \
      FORCE=1 \
      MIM_FEATURE_ONLY=1 \
      MIM_TEST_SPECKLE_LOOKS=clean \
      MIM_FEATURE_OUTPUT="$feature_file" \
      bash scripts/run_sarjepa_fewshot_all.sh
  done
done

echo "Pretrained feature worker complete: shard=$SHARD_ID/$NUM_SHARDS selected=$selected_jobs total=$job_index"
