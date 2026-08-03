#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MANIFEST="${MANIFEST:-$ROOT/logs/attention_selected_seeds.csv}"
SHARD_ID="${SHARD_ID:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
METHODS="${METHODS:-mae lomar fg_mae i_jepa sar_jepa phyd_mae}"
DATASETS="${DATASETS:-MSTAR_SOC New_FUSAR SAR_ACD}"
PROTOCOL="${PROTOCOL:-MIM_finetune}"
SHOTS="${SHOTS:-40}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-$ROOT/weights/sar-ssl-paper-baseline-weights-v1}"
ATTENTION_ROOT="${ATTENTION_ROOT:-$ROOT/paper_visualizations/attention_heatmaps_40shot}"
RUN_ROOT="${RUN_ROOT:-$ROOT/few_shot_classification/finetune/output_attention_heatmaps_40shot}"
SAMPLES_PER_DATASET="${SAMPLES_PER_DATASET:-50}"
SAMPLE_SEED="${SAMPLE_SEED:-20260804}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-50}"

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

mkdir -p "$ATTENTION_ROOT" "$RUN_ROOT"
cp -f "$MANIFEST" "$ATTENTION_ROOT/selected_seeds.csv"

python - "$MANIFEST" "$METHODS" "$DATASETS" "$PROTOCOL" "$SHOTS" \
  "$SHARD_ID" "$NUM_SHARDS" <<'PY' |
import csv
import sys

manifest, methods_raw, datasets_raw, protocol, shots = sys.argv[1:6]
shard_id, num_shards = int(sys.argv[6]), int(sys.argv[7])
methods = methods_raw.split()
datasets = datasets_raw.split()
rows = []
with open(manifest, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        if (
            row["method"] in methods
            and row["dataset"] in datasets
            and row["protocol"] == protocol
            and int(row["shots"]) == int(shots)
        ):
            rows.append(row)
rows.sort(
    key=lambda row: (
        datasets.index(row["dataset"]),
        methods.index(row["method"]),
    )
)
expected = {(dataset, method) for dataset in datasets for method in methods}
found = {(row["dataset"], row["method"]) for row in rows}
if found != expected:
    raise RuntimeError(f"Missing selected jobs: {sorted(expected - found)}")
for index, row in enumerate(rows):
    if index % num_shards == shard_id:
        print(
            "|".join(
                row[key]
                for key in ("dataset", "method", "seed", "accuracy")
            )
        )
PY
while IFS='|' read -r dataset method seed prior_accuracy; do
  output_dir="$ATTENTION_ROOT/methods/$dataset/$method"
  original_dir="$ATTENTION_ROOT/originals/$dataset"
  marker="$output_dir/ATTENTION_EXPORT_COMPLETE.json"
  if [[ -s "$marker" && -s "$output_dir/index.csv" \
        && -s "$output_dir/attention_maps.npz" ]]; then
    echo "Skip completed attention export: $dataset $method"
    continue
  fi

  checkpoint="$(resolve_checkpoint "$method")"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint: $checkpoint" >&2
    exit 1
  fi

  echo "============================================================"
  echo "Attention heatmaps: method=$method dataset=$dataset"
  echo "protocol=$PROTOCOL shots=$SHOTS seed=$seed"
  echo "selected_seed_accuracy=$prior_accuracy checkpoint=$checkpoint"
  echo "output=$output_dir"
  echo "============================================================"

  rm -rf "$output_dir"
  mkdir -p "$output_dir"
  env \
    PYTHONUNBUFFERED=1 \
    CHECKPOINT="$checkpoint" \
    OUTPUT_DIR="$RUN_ROOT/$method" \
    MODEL_FAMILY="$method" \
    DATASETS="$dataset" \
    PROTOCOLS="$PROTOCOL" \
    SHOTS="$SHOTS" \
    SEEDS="$seed" \
    LR="$LR" \
    EPOCHS="$EPOCHS" \
    BATCH_SIZE="$BATCH_SIZE" \
    USE_SFAFM=0 \
    FEATURE_POOL=cls \
    FORCE=1 \
    MIM_TEST_SPECKLE_LOOKS=clean \
    MIM_ATTENTION_TARGET=ground_truth \
    MIM_ATTENTION_SAMPLES_PER_DATASET="$SAMPLES_PER_DATASET" \
    MIM_ATTENTION_SAMPLE_SEED="$SAMPLE_SEED" \
    MIM_ATTENTION_ORIGINAL_DIR="$original_dir" \
    MIM_ATTENTION_OUTPUT_DIR="$output_dir" \
    bash scripts/run_sarjepa_fewshot_all.sh
done

echo "Attention worker complete: shard=$SHARD_ID/$NUM_SHARDS"
