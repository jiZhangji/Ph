#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ACTION="${ACTION:-all}"
case "$ACTION" in
  all|pretrain|eval|summary) ;;
  *) echo "ACTION must be all, pretrain, eval, or summary; got: $ACTION" >&2; exit 2 ;;
esac

# All branches resume the same stage-I state and run epochs 251--300. This
# keeps initialization, optimizer state, schedule, and target normalization
# identical while changing only the LFST loss coefficient.
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-$ROOT/runs/sarjepa_official_phyd_2xh100/checkpoint-250.pth}"
DATA_PATH="${DATA_PATH:-$ROOT/dataset/modelscope/extracted/Pretraining_dataset}"
LFST_WEIGHTS="${LFST_WEIGHTS:-0.05 0.1 0.2 0.5 1.0}"

SUITE_NAME="${SUITE_NAME:-lfst_weight_sensitivity_fusar_10seeds}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/few_shot_classification/finetune/output_${SUITE_NAME}}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/${SUITE_NAME}}"

TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1}"
TRAIN_GPUS="${TRAIN_GPUS:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
TRAIN_ACCUM_ITER="${TRAIN_ACCUM_ITER:-1}"
TRAIN_END_EPOCH="${TRAIN_END_EPOCH:-300}"
TRAIN_TOTAL_EPOCHS="$((TRAIN_END_EPOCH + 1))"
TRAIN_BLR="${TRAIN_BLR:-3e-5}"
TRAIN_WARMUP_EPOCHS="${TRAIN_WARMUP_EPOCHS:-0}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-16}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-27831}"

# Space-separated physical GPU IDs. Downstream jobs are deterministically
# sharded over these devices after each checkpoint has finished pre-training.
EVAL_FT_CUDA_DEVICES="${EVAL_FT_CUDA_DEVICES:-0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1}"
EVAL_LP_CUDA_DEVICES="${EVAL_LP_CUDA_DEVICES:-0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1}"
EVAL_DATASETS="${EVAL_DATASETS:-New_FUSAR}"
EVAL_PROTOCOLS="${EVAL_PROTOCOLS:-MIM_finetune MIM_linear}"
EVAL_SHOTS="${EVAL_SHOTS:-10 20 40}"
EVAL_SEEDS="${EVAL_SEEDS:-0 1 2 3 4 5 6 7 8 9}"
EVAL_LR="${EVAL_LR:-1e-3}"
EVAL_EPOCHS="${EVAL_EPOCHS:-40}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-50}"
REUSE_EXISTING_RESULTS="${REUSE_EXISTING_RESULTS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi

mkdir -p "$RUN_ROOT" "$OUTPUT_ROOT" "$LOG_ROOT"

log() {
  echo "[$(date '+%F %T')] $*"
}

weight_tag() {
  local weight="$1"
  echo "${weight//./p}"
}

run_name_for_weight() {
  local weight="$1"
  local tag
  tag="$(weight_tag "$weight")"
  echo "phyd_lfst_weight_${tag}_ft250_to300_bs$((TRAIN_BATCH_SIZE * TRAIN_GPUS * TRAIN_ACCUM_ITER))"
}

checkpoint_for_weight() {
  local weight="$1"
  echo "$RUN_ROOT/$(run_name_for_weight "$weight")/checkpoint-${TRAIN_END_EPOCH}.pth"
}

validate_inputs() {
  if [[ "$TRAIN_END_EPOCH" -le 250 ]]; then
    echo "TRAIN_END_EPOCH must be greater than 250; got: $TRAIN_END_EPOCH" >&2
    exit 2
  fi
  local weight
  for weight in $LFST_WEIGHTS; do
    if ! "$PYTHON_BIN" - "$weight" <<'PY'
import sys
value = float(sys.argv[1])
raise SystemExit(0 if value > 0 else 1)
PY
    then
      echo "LFST weights must be positive numbers; got: $weight" >&2
      exit 2
    fi
  done
  if [[ "$ACTION" == "all" || "$ACTION" == "pretrain" ]]; then
    if [[ ! -f "$SOURCE_CHECKPOINT" ]]; then
      echo "Missing shared stage-I checkpoint: $SOURCE_CHECKPOINT" >&2
      exit 1
    fi
    if [[ ! -d "$DATA_PATH" ]]; then
      echo "Missing pre-training dataset: $DATA_PATH" >&2
      exit 1
    fi
  fi
}

train_weight() {
  local weight="$1"
  local index="$2"
  local run_name output_dir final_checkpoint run_log resume
  run_name="$(run_name_for_weight "$weight")"
  output_dir="$RUN_ROOT/$run_name"
  final_checkpoint="$(checkpoint_for_weight "$weight")"
  run_log="$LOG_ROOT/${run_name}.pretrain.log"
  resume="$SOURCE_CHECKPOINT"

  if [[ -f "$final_checkpoint" ]]; then
    log "Skip completed pre-training: lambda_LFST=$weight ($final_checkpoint)"
    return
  fi
  if [[ -f "$output_dir/checkpoint-last.pth" ]]; then
    resume="$output_dir/checkpoint-last.pth"
    log "Resume lambda_LFST=$weight from $resume"
  elif [[ -d "$output_dir" ]] && find "$output_dir" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to overwrite non-empty run without checkpoint-last: $output_dir" >&2
    exit 1
  else
    log "Start lambda_LFST=$weight from the shared epoch-250 checkpoint"
  fi

  env \
    RUN_NAME="$run_name" \
    OUTPUT_DIR="$output_dir" \
    LOG_DIR="$output_dir" \
    DATA_PATH="$DATA_PATH" \
    CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" \
    GPUS="$TRAIN_GPUS" \
    MASTER_PORT="$((MASTER_PORT_BASE + index))" \
    BATCH_SIZE="$TRAIN_BATCH_SIZE" \
    ACCUM_ITER="$TRAIN_ACCUM_ITER" \
    EPOCHS="$TRAIN_TOTAL_EPOCHS" \
    BLR="$TRAIN_BLR" \
    WARMUP_EPOCHS="$TRAIN_WARMUP_EPOCHS" \
    NUM_WORKERS="$TRAIN_NUM_WORKERS" \
    RESUME="$resume" \
    INIT_CHECKPOINT= \
    GRAD_LOSS_WEIGHT=1.0 \
    LFST_LOSS_WEIGHT="$weight" \
    TARGET_NORM=image \
    LFST_CUTOFF=30 \
    LFST_INPUT_MODE=raw \
    LFST_TARGET_TYPE=lfst \
    SASGT_SCALES=0.8,1.6,3.2,6.4 \
    SASGT_TEMPERATURE=1.0 \
    SASGT_GAMMA=1.0 \
    SASGT_RELIABILITY_WINDOW=7 \
    SASGT_MODE=complete \
    USE_SFAFM=0 \
    SAVE_EVERY_AFTER_EPOCH="$TRAIN_END_EPOCH" \
    SAVE_INTERVAL_AFTER_EPOCH=1 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    bash scripts/run_sarjepa_official_phyd_pretrain_2xh100.sh \
    2>&1 | tee -a "$run_log"

  if [[ ! -f "$final_checkpoint" ]]; then
    echo "Expected checkpoint was not produced: $final_checkpoint" >&2
    exit 1
  fi
  log "Finished pre-training: lambda_LFST=$weight"
}

build_manifest() {
  local manifest="$OUTPUT_ROOT/expected_matrix.tsv"
  printf 'model\tlfst_weight\tcheckpoint\tdatasets\tprotocols\tshots\tseeds\n' > "$manifest"
  local weight tag checkpoint
  for weight in $LFST_WEIGHTS; do
    tag="lfst_weight_$(weight_tag "$weight")"
    checkpoint="$(checkpoint_for_weight "$weight")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$tag" "$weight" "$checkpoint" "$EVAL_DATASETS" \
      "$EVAL_PROTOCOLS" "$EVAL_SHOTS" "$EVAL_SEEDS" >> "$manifest"
  done
  echo "$manifest"
}

build_generic_manifest() {
  local source_manifest="$1"
  local generic_manifest="$OUTPUT_ROOT/expected_matrix_generic.tsv"
  "$PYTHON_BIN" - "$source_manifest" "$generic_manifest" <<'PY'
import csv
import sys

source, destination = sys.argv[1:]
with open(source, newline="", encoding="utf-8") as src:
    rows = list(csv.DictReader(src, delimiter="\t"))
fields = ("model", "checkpoint", "datasets", "protocols", "shots", "seeds")
with open(destination, "w", newline="", encoding="utf-8") as dst:
    writer = csv.DictWriter(dst, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row[key] for key in fields})
PY
  echo "$generic_manifest"
}

evaluate_checkpoint() {
  local tag="$1"
  local checkpoint="$2"
  local protocol="$3"
  local device_list="$4"
  local output_dir="$OUTPUT_ROOT/$tag"
  local -a devices pids
  read -r -a devices <<< "$device_list"
  if [[ ${#devices[@]} -eq 0 ]]; then
    echo "EVAL_CUDA_DEVICES must contain at least one GPU ID" >&2
    exit 2
  fi

  local num_shards="${#devices[@]}"
  local shard worker_pid failed=0
  pids=()
  log "Evaluate $tag: protocol=$protocol, shards=$num_shards"
  for ((shard = 0; shard < num_shards; shard++)); do
    env \
      CUDA_VISIBLE_DEVICES="${devices[$shard]}" \
      CHECKPOINT="$checkpoint" \
      OUTPUT_DIR="$output_dir" \
      DATASETS="$EVAL_DATASETS" \
      PROTOCOLS="$protocol" \
      SHOTS="$EVAL_SHOTS" \
      SEEDS="$EVAL_SEEDS" \
      LR="$EVAL_LR" \
      EPOCHS="$EVAL_EPOCHS" \
      BATCH_SIZE="$EVAL_BATCH_SIZE" \
      USE_SFAFM=0 \
      FEATURE_POOL=cls \
      MODEL_FAMILY=phyd_mae \
      FORCE=0 \
      NUM_SHARDS="$num_shards" \
      SHARD_ID="$shard" \
      bash scripts/run_sarjepa_fewshot_all.sh \
      > "$LOG_ROOT/${tag}.${protocol}.shard${shard}.log" 2>&1 &
    worker_pid="$!"
    pids+=("$worker_pid")
    log "Started $tag shard=$shard/$num_shards GPU=${devices[$shard]} PID=$worker_pid"
  done

  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" == "1" ]]; then
    echo "Downstream evaluation failed for $tag $protocol; inspect $LOG_ROOT/${tag}.${protocol}.shard*.log" >&2
    exit 1
  fi
  log "Finished downstream evaluation: $tag $protocol"
}

validate_inputs

if [[ "$ACTION" == "all" || "$ACTION" == "pretrain" ]]; then
  index=0
  for weight in $LFST_WEIGHTS; do
    train_weight "$weight" "$index"
    index=$((index + 1))
  done
fi

manifest="$(build_manifest)"
generic_manifest="$(build_generic_manifest "$manifest")"

if [[ "$ACTION" == "all" || "$ACTION" == "eval" ]]; then
  while IFS=$'\t' read -r tag weight checkpoint datasets protocols shots seeds; do
    [[ "$tag" == "model" ]] && continue
    if [[ ! -f "$checkpoint" ]]; then
      echo "Missing evaluation checkpoint for lambda_LFST=$weight: $checkpoint" >&2
      echo "Run ACTION=pretrain first." >&2
      exit 1
    fi
  done < "$manifest"

  if [[ "$REUSE_EXISTING_RESULTS" == "1" ]]; then
    "$PYTHON_BIN" scripts/reuse_completed_downstream_results.py \
      --search-root "$ROOT/few_shot_classification/finetune" \
      --output-root "$OUTPUT_ROOT" \
      --manifest "$generic_manifest" \
      --lr "$EVAL_LR" \
      --epochs "$EVAL_EPOCHS" \
      --batch-size "$EVAL_BATCH_SIZE"
  fi

  while IFS=$'\t' read -r tag weight checkpoint datasets protocols shots seeds; do
    [[ "$tag" == "model" ]] && continue
    for protocol in $EVAL_PROTOCOLS; do
      if [[ "$protocol" == "MIM_linear" || "$protocol" == "linear" ]]; then
        evaluate_checkpoint "$tag" "$checkpoint" "$protocol" "$EVAL_LP_CUDA_DEVICES"
      else
        evaluate_checkpoint "$tag" "$checkpoint" "$protocol" "$EVAL_FT_CUDA_DEVICES"
      fi
    done
  done < "$manifest"
fi

if [[ "$ACTION" == "all" || "$ACTION" == "eval" || "$ACTION" == "summary" ]]; then
  "$PYTHON_BIN" scripts/summarize_lfst_weight_sensitivity.py \
    --root "$OUTPUT_ROOT" \
    --manifest "$manifest"
fi

log "Done."
log "Manifest: $manifest"
log "Per-seed CSV: $OUTPUT_ROOT/results_per_seed.csv"
log "Summary CSV: $OUTPUT_ROOT/results_mean_std_max.csv"
log "Readable summary: $OUTPUT_ROOT/results_summary.txt"
log "Logs: $LOG_ROOT"
