#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

WEIGHTS_ROOT="${WEIGHTS_ROOT:-/mnt/data5/zhangji/SAR-JEPA/weights}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/few_shot_classification/finetune/output_paper_baselines_5seeds}"
METHODS="${METHODS:-mae lomar fg_mae i_jepa sar_jepa}"
DATASETS="${DATASETS:-MSTAR_SOC New_FUSAR SAR_ACD}"
PROTOCOLS="${PROTOCOLS:-MIM_finetune MIM_linear}"
SHOTS="${SHOTS:-10 20 40}"
SEEDS="${SEEDS:-0 1 2 3 4}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-50}"
FORCE="${FORCE:-0}"

first_checkpoint() {
  local candidate
  for candidate in "$@"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

resolve_checkpoint() {
  case "$1" in
    mae)
      first_checkpoint \
        "$WEIGHTS_ROOT/MAE/checkpoint-200.pth" \
        "$WEIGHTS_ROOT/weights/mae/checkpoint-200.pth"
      ;;
    lomar)
      first_checkpoint \
        "$WEIGHTS_ROOT/LoMaR/checkpoint-200.pth" \
        "$WEIGHTS_ROOT/weights/lomar/checkpoint-200.pth"
      ;;
    fg_mae)
      first_checkpoint \
        "$WEIGHTS_ROOT/FG-MAE/checkpoint-200.pth" \
        "$WEIGHTS_ROOT/weights/fg_mae/checkpoint-200.pth"
      ;;
    i_jepa)
      first_checkpoint \
        "$WEIGHTS_ROOT/ijepa/jepa-ep200.pth.tar" \
        "$WEIGHTS_ROOT/weights/i_jepa/jepa-ep200.pth.tar"
      ;;
    sar_jepa)
      first_checkpoint \
        "$WEIGHTS_ROOT/SAR-JEPA/checkpoint-200.pth" \
        "$WEIGHTS_ROOT/weights/sar_jepa/checkpoint-200.pth"
      ;;
    phyd_mae)
      first_checkpoint \
        "${PHYD_CHECKPOINT:-}" \
        "$ROOT/runs/sarjepa_official_phyd_ft250_bs1024_lfst0p1_image_2xh200/checkpoint-300.pth"
      ;;
    *)
      echo "Unknown method: $1" >&2
      return 2
      ;;
  esac
}

for method in $METHODS; do
  checkpoint="$(resolve_checkpoint "$method" || true)"
  if [[ -z "$checkpoint" ]]; then
    echo "Checkpoint not found for $method under $WEIGHTS_ROOT" >&2
    exit 1
  fi

  echo "============================================================"
  echo "Method: $method"
  echo "Checkpoint: $checkpoint"
  echo "============================================================"

  env \
    CHECKPOINT="$checkpoint" \
    OUTPUT_DIR="$OUTPUT_ROOT/$method" \
    MODEL_FAMILY="$method" \
    DATASETS="$DATASETS" \
    PROTOCOLS="$PROTOCOLS" \
    SHOTS="$SHOTS" \
    SEEDS="$SEEDS" \
    LR="$LR" \
    EPOCHS="$EPOCHS" \
    BATCH_SIZE="$BATCH_SIZE" \
    USE_SFAFM=0 \
    FEATURE_POOL=cls \
    FORCE="$FORCE" \
    bash scripts/run_sarjepa_fewshot_all.sh
done

echo "All requested paper baseline evaluations completed."
