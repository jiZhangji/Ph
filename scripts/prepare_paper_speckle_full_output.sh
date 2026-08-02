#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUTPUT_ROOT="${SPECKLE_OUTPUT_ROOT:-$ROOT/few_shot_classification/finetune/output_speckle_robustness_10seeds}"
ARCHIVE_ROOT="${SPECKLE_ARCHIVE_ROOT:-$ROOT/few_shot_classification/finetune/output_speckle_robustness_archives}"
PHYD_DIR="$OUTPUT_ROOT/phyd_mae"
READY_MARKER="$OUTPUT_ROOT/.phyd_main_lr1e3_ready"

mkdir -p "$OUTPUT_ROOT" "$ARCHIVE_ROOT"

if [[ -f "$READY_MARKER" ]]; then
  echo "Full-output preparation already completed: $READY_MARKER"
  exit 0
fi

if [[ -d "$PHYD_DIR" ]]; then
  timestamp="$(date '+%Y%m%d-%H%M%S')"
  archive_dir="$ARCHIVE_ROOT/phyd_mae_lr1e4_linear20_${timestamp}"
  echo "Archiving the completed LR=1e-4 PhyD-MAE pilot:"
  echo "  source:  $PHYD_DIR"
  echo "  archive: $archive_dir"
  mv -- "$PHYD_DIR" "$archive_dir"
fi

printf '%s\n' \
  "PhyD-MAE linear probing must use LR=1e-3 in the full robustness run." \
  > "$READY_MARKER"

echo "Preparation complete. Baseline outputs were left in place."
echo "Marker: $READY_MARKER"
