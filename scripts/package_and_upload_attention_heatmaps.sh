#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ATTENTION_ROOT="${ATTENTION_ROOT:-$ROOT/paper_visualizations/attention_heatmaps_40shot}"
PACKAGE_DIR="${PACKAGE_DIR:-$ROOT/hf_release/phyd-mae-paper-attention-heatmaps-v1}"
HF_REPO_ID="${HF_REPO_ID:-shimiandeshu/phyd-mae-paper-attention-heatmaps-v1}"
HF_PRIVATE="${HF_PRIVATE:-1}"
METHODS=(mae lomar fg_mae i_jepa sar_jepa phyd_mae)
DATASETS=(MSTAR_SOC New_FUSAR SAR_ACD)

for dataset in "${DATASETS[@]}"; do
  for method in "${METHODS[@]}"; do
    marker="$ATTENTION_ROOT/methods/$dataset/$method/ATTENTION_EXPORT_COMPLETE.json"
    if [[ ! -s "$marker" ]]; then
      echo "Missing completion marker: $marker" >&2
      exit 1
    fi
  done
done
if [[ ! -s "$ATTENTION_ROOT/MERGED_EXPORT_COMPLETE.json" ]]; then
  echo "Missing merged output marker" >&2
  exit 1
fi

rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR"

for dataset in "${DATASETS[@]}"; do
  archive="$PACKAGE_DIR/${dataset}_40shot_all_methods_attention.tar.gz"
  tar -C "$ATTENTION_ROOT" -czf "$archive" "methods/$dataset"
  echo "Created $archive"
done
tar -C "$ATTENTION_ROOT" -czf \
  "$PACKAGE_DIR/originals_3datasets_50each.tar.gz" originals
tar -C "$ATTENTION_ROOT" -czf \
  "$PACKAGE_DIR/merged_3datasets_6methods_50each.tar.gz" \
  merged merged_index.csv MERGED_EXPORT_COMPLETE.json

cp -f "$ATTENTION_ROOT/selected_seeds.csv" "$PACKAGE_DIR/selected_seeds.csv"
printf '%s\n' \
  '# Three-dataset 40-shot fine-tuning attention heatmaps' \
  '' \
  'Datasets: MSTAR, FUSAR-Ship, and SAR-ACD; 50 fixed random test images each.' \
  'Methods: MAE, LoMaR, FG-MAE, I-JEPA, SAR-JEPA, and PhyD-MAE.' \
  'Dataset archives contain overlays, index CSV files, compressed 14x14 CAM' \
  'arrays, and run metadata for all six methods.' \
  'Separate archives contain original images and compact merged comparisons.' \
  'CAMs target the ground-truth class and use the final ViT block norm1 tokens.' \
  > "$PACKAGE_DIR/README.md"

python - "$ATTENTION_ROOT" "$PACKAGE_DIR/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
output = Path(sys.argv[2])
records = []
for marker in sorted((root / "methods").glob("*/*/ATTENTION_EXPORT_COMPLETE.json")):
    records.append(json.loads(marker.read_text(encoding="utf-8")))
merged = json.loads(
    (root / "MERGED_EXPORT_COMPLETE.json").read_text(encoding="utf-8")
)
output.write_text(
    json.dumps(
        {"exports": records, "merged": merged},
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY

(
  cd "$PACKAGE_DIR"
  sha256sum *.tar.gz selected_seeds.csv manifest.json README.md > SHA256SUMS
)

export HF_REPO_ID HF_PRIVATE
python - <<'PY'
import os
from huggingface_hub import HfApi

HfApi().create_repo(
    repo_id=os.environ["HF_REPO_ID"],
    repo_type="dataset",
    private=os.environ.get("HF_PRIVATE", "1") != "0",
    exist_ok=True,
)
PY

hf upload "$HF_REPO_ID" \
  --repo-type=dataset \
  "$PACKAGE_DIR" \
  . \
  --commit-message "Upload complete three-dataset attention heatmaps"

echo "Upload complete: https://huggingface.co/datasets/$HF_REPO_ID"
