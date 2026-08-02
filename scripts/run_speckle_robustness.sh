#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

WEIGHTS_ROOT="${WEIGHTS_ROOT:-/mnt/data5/zhangji/SAR-JEPA/weights}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/few_shot_classification/finetune/output_speckle_robustness}"
METHODS="${METHODS:-mae lomar fg_mae i_jepa sar_jepa phyd_mae}"
DATASETS="${DATASETS:-MSTAR_SOC}"
PROTOCOLS="${PROTOCOLS:-MIM_linear}"
SHOTS="${SHOTS:-20}"
SEEDS="${SEEDS:-0 1 2 3 4}"
SPECKLE_LOOKS="${SPECKLE_LOOKS:-clean 8 4 2 1}"
NOISE_SEEDS="${NOISE_SEEDS:-0 1 2}"
LR="${LR:-1e-4}"
PHYD_LR="${PHYD_LR:-$LR}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-50}"
FORCE="${FORCE:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_ID="${SHARD_ID:-0}"
SUMMARIZE="${SUMMARIZE:-1}"

first_checkpoint() {
  local candidate
  for candidate in "$@"; do
    if [[ -n "$candidate" && -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

resolve_checkpoint() {
  case "$1" in
    mae) first_checkpoint "$WEIGHTS_ROOT/MAE/checkpoint-200.pth" "$WEIGHTS_ROOT/weights/mae/checkpoint-200.pth" ;;
    lomar) first_checkpoint "$WEIGHTS_ROOT/LoMaR/checkpoint-200.pth" "$WEIGHTS_ROOT/weights/lomar/checkpoint-200.pth" ;;
    fg_mae) first_checkpoint "$WEIGHTS_ROOT/FG-MAE/checkpoint-200.pth" "$WEIGHTS_ROOT/weights/fg_mae/checkpoint-200.pth" ;;
    i_jepa) first_checkpoint "$WEIGHTS_ROOT/ijepa/jepa-ep200.pth.tar" "$WEIGHTS_ROOT/weights/i_jepa/jepa-ep200.pth.tar" ;;
    sar_jepa) first_checkpoint "$WEIGHTS_ROOT/SAR-JEPA/checkpoint-200.pth" "$WEIGHTS_ROOT/weights/sar_jepa/checkpoint-200.pth" ;;
    phyd_mae)
      first_checkpoint \
        "${PHYD_CHECKPOINT:-}" \
        "$ROOT/runs/sarjepa_official_phyd_ft250_bs1024_lfst0p1_image_2xh200/checkpoint-300.pth"
      ;;
    *) echo "Unknown method: $1" >&2; return 2 ;;
  esac
}

echo "Controlled additional-speckle robustness evaluation"
echo "L_add values: $SPECKLE_LOOKS"
echo "Noise seeds: $NOISE_SEEDS"
echo "The original dataset look number is not inferred or modified."

for method in $METHODS; do
  checkpoint="$(resolve_checkpoint "$method" || true)"
  if [[ -z "$checkpoint" ]]; then
    echo "Checkpoint not found for $method under $WEIGHTS_ROOT" >&2
    exit 1
  fi

  method_lr="$LR"
  if [[ "$method" == "phyd_mae" ]]; then
    method_lr="$PHYD_LR"
  fi

  echo "============================================================"
  echo "Method: $method"
  echo "Checkpoint: $checkpoint"
  echo "Downstream learning rate: $method_lr"
  echo "============================================================"

  env \
    CHECKPOINT="$checkpoint" \
    OUTPUT_DIR="$OUTPUT_ROOT/$method" \
    MODEL_FAMILY="$method" \
    DATASETS="$DATASETS" \
    PROTOCOLS="$PROTOCOLS" \
    SHOTS="$SHOTS" \
    SEEDS="$SEEDS" \
    LR="$method_lr" \
    EPOCHS="$EPOCHS" \
    BATCH_SIZE="$BATCH_SIZE" \
    USE_SFAFM=0 \
    FEATURE_POOL=cls \
    FORCE="$FORCE" \
    COMPLETION_MARKER='SPECKLE_ROBUSTNESS_COMPLETE' \
    MIM_TEST_SPECKLE_LOOKS_LIST="$SPECKLE_LOOKS" \
    MIM_TEST_SPECKLE_NOISE_SEEDS="$NOISE_SEEDS" \
    MIM_DOWNSTREAM_LR="$method_lr" \
    NUM_SHARDS="$NUM_SHARDS" \
    SHARD_ID="$SHARD_ID" \
    bash scripts/run_sarjepa_fewshot_all.sh
done

if [[ "$SUMMARIZE" == "1" ]]; then
  expected_downstream_seeds="$(wc -w <<< "$SEEDS" | tr -d ' ')"
  python scripts/summarize_speckle_robustness.py \
    "$OUTPUT_ROOT" \
    --output "$OUTPUT_ROOT/speckle_robustness_summary.csv" \
    --expected-downstream-seeds "$expected_downstream_seeds"
fi

echo "All requested controlled-speckle evaluations completed."
