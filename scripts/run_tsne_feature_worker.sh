#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MANIFEST="${MANIFEST:-$ROOT/logs/tsne_selected_seeds.csv}"
SHARD_ID="${SHARD_ID:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-$ROOT/weights/sar-ssl-paper-baseline-weights-v1}"
FEATURE_ROOT="${FEATURE_ROOT:-$ROOT/few_shot_classification/finetune/tsne_features}"
RUN_ROOT="${RUN_ROOT:-$ROOT/few_shot_classification/finetune/output_tsne_selected}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing seed manifest: $MANIFEST" >&2
  exit 1
fi
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

mkdir -p "$FEATURE_ROOT" "$RUN_ROOT"

python - "$MANIFEST" "$SHARD_ID" "$NUM_SHARDS" <<'PY' |
import csv
import sys

manifest, shard_id, num_shards = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
jobs = []
with open(manifest, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        # Encoder features in linear probing are independent of shot and seed.
        if row["protocol"] == "MIM_linear" and int(row["shots"]) != 10:
            continue
        jobs.append(row)
for index, row in enumerate(jobs):
    if index % num_shards != shard_id:
        continue
    print("|".join(row[key] for key in ("method", "dataset", "protocol", "shots", "seed", "accuracy")))
PY
while IFS='|' read -r method dataset protocol shots seed accuracy; do
  if [[ "$protocol" == "MIM_linear" ]]; then
    feature_dir="$FEATURE_ROOT/$dataset/$protocol/encoder"
  else
    feature_dir="$FEATURE_ROOT/$dataset/$protocol/${shots}shot"
  fi
  feature_file="$feature_dir/$method.npz"
  if [[ -s "$feature_file" ]]; then
    echo "Skip completed feature: $feature_file"
    continue
  fi

  checkpoint="$(resolve_checkpoint "$method")"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint: $checkpoint" >&2
    exit 1
  fi
  lr="1e-4"
  if [[ "$method" == "phyd_mae" && "$protocol" == "MIM_linear" ]]; then
    lr="1e-3"
  fi

  echo "============================================================"
  echo "t-SNE feature: method=$method dataset=$dataset protocol=$protocol"
  echo "shot=$shots seed=$seed prior_accuracy=$accuracy lr=$lr"
  echo "output=$feature_file"
  echo "============================================================"

  mkdir -p "$feature_dir"
  env \
    PYTHONUNBUFFERED=1 \
    CHECKPOINT="$checkpoint" \
    OUTPUT_DIR="$RUN_ROOT/$method" \
    MODEL_FAMILY="$method" \
    DATASETS="$dataset" \
    PROTOCOLS="$protocol" \
    SHOTS="$shots" \
    SEEDS="$seed" \
    LR="$lr" \
    EPOCHS=40 \
    BATCH_SIZE=50 \
    USE_SFAFM=0 \
    FEATURE_POOL=cls \
    FORCE=1 \
    MIM_TEST_SPECKLE_LOOKS=clean \
    MIM_FEATURE_OUTPUT="$feature_file" \
    bash scripts/run_sarjepa_fewshot_all.sh
done

echo "Feature worker complete: shard=$SHARD_ID/$NUM_SHARDS"
