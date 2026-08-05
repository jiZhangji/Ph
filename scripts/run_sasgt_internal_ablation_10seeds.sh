#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ACTION="${ACTION:-all}"
case "$ACTION" in
  all|pretrain|eval|summary) ;;
  *) echo "ACTION must be all, pretrain, eval, or summary; got: $ACTION" >&2; exit 2 ;;
esac

SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-$ROOT/runs/sarjepa_official_phyd_ft250_bs1024_lfst0p1_image_2xh200/checkpoint-300.pth}"
FULL_SASGT_CHECKPOINT="${FULL_SASGT_CHECKPOINT:-$ROOT/runs/phyd_ckpt300_target_pilot_30e_bs1088_msgt_only/checkpoint-29.pth}"
DATA_PATH="${DATA_PATH:-$ROOT/dataset/modelscope/extracted/Pretraining_dataset}"

SUITE_NAME="${SUITE_NAME:-sasgt_internal_ablation_fusar_10seeds}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/few_shot_classification/finetune/output_${SUITE_NAME}}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/${SUITE_NAME}}"

TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1}"
TRAIN_GPUS="${TRAIN_GPUS:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-544}"
TRAIN_ACCUM_ITER="${TRAIN_ACCUM_ITER:-1}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-30}"
TRAIN_BLR="${TRAIN_BLR:-5e-6}"
TRAIN_WARMUP_EPOCHS="${TRAIN_WARMUP_EPOCHS:-3}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-16}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-27131}"

# Space-separated physical GPU IDs. Each device runs one downstream shard.
EVAL_CUDA_DEVICES="${EVAL_CUDA_DEVICES:-0 1}"
EVAL_PROTOCOLS="${EVAL_PROTOCOLS:-MIM_finetune MIM_linear}"
EVAL_SHOTS="${EVAL_SHOTS:-10 20 40}"
EVAL_SEEDS="${EVAL_SEEDS:-0 1 2 3 4 5 6 7 8 9}"
EVAL_LR="${EVAL_LR:-1e-4}"
EVAL_EPOCHS="${EVAL_EPOCHS:-40}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-50}"
REUSE_EXISTING_RESULTS="${REUSE_EXISTING_RESULTS:-1}"
INCLUDE_FULL_SASGT="${INCLUDE_FULL_SASGT:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi

mkdir -p "$RUN_ROOT" "$OUTPUT_ROOT" "$LOG_ROOT"

effective_batch=$((TRAIN_BATCH_SIZE * TRAIN_GPUS * TRAIN_ACCUM_ITER))
declare -a VARIANTS=(
  "sasgt_uniform|uniform|phyd_sasgt_uniform_${TRAIN_EPOCHS}e_bs${effective_batch}_v2"
  "sasgt_no_scale|adaptive|phyd_sasgt_no_scale_${TRAIN_EPOCHS}e_bs${effective_batch}_v2"
  "sasgt_no_log|no_log|phyd_sasgt_no_log_${TRAIN_EPOCHS}e_bs${effective_batch}_v1"
)

log() {
  echo "[$(date '+%F %T')] $*"
}

checkpoint_for_run() {
  local run_name="$1"
  echo "$RUN_ROOT/$run_name/checkpoint-$((TRAIN_EPOCHS - 1)).pth"
}

train_variant() {
  local tag="$1"
  local mode="$2"
  local run_name="$3"
  local output_dir="$RUN_ROOT/$run_name"
  local final_checkpoint
  final_checkpoint="$(checkpoint_for_run "$run_name")"
  local run_log="$LOG_ROOT/${run_name}.pretrain.log"
  local resume=""
  local init_checkpoint="$SOURCE_CHECKPOINT"
  local port="$4"

  if [[ -f "$final_checkpoint" ]]; then
    log "Skip completed pretraining: $tag ($final_checkpoint)"
    return
  fi
  if [[ -f "$output_dir/checkpoint-last.pth" ]]; then
    resume="$output_dir/checkpoint-last.pth"
    init_checkpoint=""
    log "Resume pretraining: $tag from $resume"
  elif [[ -d "$output_dir" ]] && find "$output_dir" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to overwrite non-empty run without checkpoint-last: $output_dir" >&2
    exit 1
  else
    log "Start pretraining: $tag (SASGT_MODE=$mode)"
  fi

  env \
    RUN_NAME="$run_name" \
    OUTPUT_DIR="$output_dir" \
    LOG_DIR="$output_dir" \
    DATA_PATH="$DATA_PATH" \
    CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" \
    GPUS="$TRAIN_GPUS" \
    MASTER_PORT="$port" \
    BATCH_SIZE="$TRAIN_BATCH_SIZE" \
    ACCUM_ITER="$TRAIN_ACCUM_ITER" \
    EPOCHS="$TRAIN_EPOCHS" \
    BLR="$TRAIN_BLR" \
    WARMUP_EPOCHS="$TRAIN_WARMUP_EPOCHS" \
    NUM_WORKERS="$TRAIN_NUM_WORKERS" \
    RESUME="$resume" \
    INIT_CHECKPOINT="$init_checkpoint" \
    INIT_SCOPE=encoder \
    GRAD_LOSS_WEIGHT=1.0 \
    LFST_LOSS_WEIGHT=0.0 \
    TARGET_NORM=image \
    SASGT_SCALES=0.8,1.6,3.2,6.4 \
    SASGT_TEMPERATURE=1.0 \
    SASGT_GAMMA=1.0 \
    SASGT_RELIABILITY_WINDOW=7 \
    SASGT_MODE="$mode" \
    USE_SFAFM=0 \
    CLIP_GRAD=1.0 \
    SAVE_EVERY_AFTER_EPOCH=0 \
    SAVE_INTERVAL_AFTER_EPOCH=5 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    bash scripts/run_sarjepa_official_phyd_pretrain_2xh100.sh \
    2>&1 | tee -a "$run_log"

  if [[ ! -f "$final_checkpoint" ]]; then
    echo "Expected final checkpoint was not produced: $final_checkpoint" >&2
    exit 1
  fi
  log "Finished pretraining: $tag"
}

build_manifest() {
  local manifest="$OUTPUT_ROOT/expected_matrix.tsv"
  printf 'model\tcheckpoint\tdatasets\tprotocols\tshots\tseeds\n' > "$manifest"
  local spec tag mode run_name checkpoint
  for spec in "${VARIANTS[@]}"; do
    IFS='|' read -r tag mode run_name <<< "$spec"
    checkpoint="$(checkpoint_for_run "$run_name")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$tag" "$checkpoint" "New_FUSAR" "$EVAL_PROTOCOLS" "$EVAL_SHOTS" "$EVAL_SEEDS" \
      >> "$manifest"
  done
  if [[ "$INCLUDE_FULL_SASGT" == "1" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "sasgt_complete" "$FULL_SASGT_CHECKPOINT" "New_FUSAR" \
      "$EVAL_PROTOCOLS" "$EVAL_SHOTS" "$EVAL_SEEDS" >> "$manifest"
  fi
  echo "$manifest"
}

evaluate_checkpoint() {
  local tag="$1"
  local checkpoint="$2"
  local output_dir="$OUTPUT_ROOT/$tag"
  local -a devices
  read -r -a devices <<< "$EVAL_CUDA_DEVICES"
  if [[ ${#devices[@]} -eq 0 ]]; then
    echo "EVAL_CUDA_DEVICES must contain at least one device ID" >&2
    exit 2
  fi

  local num_shards="${#devices[@]}"
  local -a pids=()
  local shard
  local worker_pid
  log "Evaluate $tag with $num_shards shard(s): checkpoint=$checkpoint"
  for ((shard = 0; shard < num_shards; shard++)); do
    env \
      CUDA_VISIBLE_DEVICES="${devices[$shard]}" \
      CHECKPOINT="$checkpoint" \
      OUTPUT_DIR="$output_dir" \
      DATASETS=New_FUSAR \
      PROTOCOLS="$EVAL_PROTOCOLS" \
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
      > "$LOG_ROOT/${tag}.eval.shard${shard}.log" 2>&1 &
    worker_pid="$!"
    pids+=("$worker_pid")
    log "Started eval shard $shard/$num_shards on GPU ${devices[$shard]}: PID=$worker_pid"
  done

  local failed=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" == "1" ]]; then
    echo "At least one downstream shard failed for $tag; inspect $LOG_ROOT/${tag}.eval.shard*.log" >&2
    exit 1
  fi
  log "Finished downstream evaluation: $tag"
}

if [[ "$ACTION" == "all" || "$ACTION" == "pretrain" ]]; then
  if [[ ! -f "$SOURCE_CHECKPOINT" ]]; then
    echo "Missing encoder initialization checkpoint: $SOURCE_CHECKPOINT" >&2
    exit 1
  fi
  if [[ ! -d "$DATA_PATH" ]]; then
    echo "Missing pretraining dataset: $DATA_PATH" >&2
    exit 1
  fi
  "$PYTHON_BIN" scripts/smoke_test_sasgt_ablation_modes.py

  train_index=0
  for spec in "${VARIANTS[@]}"; do
    IFS='|' read -r tag mode run_name <<< "$spec"
    train_variant "$tag" "$mode" "$run_name" "$((MASTER_PORT_BASE + train_index))"
    train_index=$((train_index + 1))
  done
fi

manifest="$(build_manifest)"

if [[ "$ACTION" == "all" || "$ACTION" == "eval" ]]; then
  while IFS=$'\t' read -r tag checkpoint datasets protocols shots seeds; do
    [[ "$tag" == "model" ]] && continue
    if [[ ! -f "$checkpoint" ]]; then
      echo "Missing evaluation checkpoint for $tag: $checkpoint" >&2
      exit 1
    fi
  done < "$manifest"

  if [[ "$REUSE_EXISTING_RESULTS" == "1" ]]; then
    "$PYTHON_BIN" scripts/reuse_completed_downstream_results.py \
      --search-root "$ROOT/few_shot_classification/finetune" \
      --output-root "$OUTPUT_ROOT" \
      --manifest "$manifest" \
      --lr "$EVAL_LR" \
      --epochs "$EVAL_EPOCHS" \
      --batch-size "$EVAL_BATCH_SIZE"
  fi

  while IFS=$'\t' read -r tag checkpoint datasets protocols shots seeds; do
    [[ "$tag" == "model" ]] && continue
    evaluate_checkpoint "$tag" "$checkpoint"
  done < "$manifest"
fi

if [[ "$ACTION" == "all" || "$ACTION" == "eval" || "$ACTION" == "summary" ]]; then
  "$PYTHON_BIN" scripts/summarize_paper_ablation_results.py \
    --root "$OUTPUT_ROOT" \
    --manifest "$manifest"
fi

log "Done."
log "Manifest: $manifest"
log "Per-seed results: $OUTPUT_ROOT/results_per_seed.csv"
log "Mean/std results: $OUTPUT_ROOT/results_mean_std.csv"
log "Logs: $LOG_ROOT"
