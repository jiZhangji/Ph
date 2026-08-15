#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ACTION="${ACTION:-all}"
case "$ACTION" in
  all|pretrain|finetune|linear|eval|lfst|summary) ;;
  *)
    echo "ACTION must be all, pretrain, finetune, linear, eval, lfst, or summary; got: $ACTION" >&2
    exit 2
    ;;
esac

SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-$ROOT/runs/sarjepa_official_phyd_ft250_bs1024_lfst0p1_image_2xh200/checkpoint-300.pth}"
DEFAULT_CHECKPOINT="${DEFAULT_CHECKPOINT:-$ROOT/runs/phyd_ckpt300_target_pilot_30e_bs1088_msgt_only/checkpoint-29.pth}"
DATA_PATH="${DATA_PATH:-$ROOT/dataset/modelscope/extracted/Pretraining_dataset}"

SUITE_NAME="${SUITE_NAME:-sasgt_parameter_sensitivity_lr1e3_10seeds}"
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
MASTER_PORT_BASE="${MASTER_PORT_BASE:-27431}"

# Downstream protocol used by the final paper experiments.
# Twenty interleaved shards follow the final downstream launcher used in the
# paper experiments. On two H200 GPUs this starts ten workers per GPU while
# distributing consecutive jobs across different devices.
EVAL_CUDA_DEVICES="${EVAL_CUDA_DEVICES:-0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1}"
EVAL_DATASET="${EVAL_DATASET:-New_FUSAR}"
EVAL_SHOTS="${EVAL_SHOTS:-10 20 40}"
EVAL_SEEDS="${EVAL_SEEDS:-0 1 2 3 4 5 6 7 8 9}"
EVAL_LR="${EVAL_LR:-1e-3}"
EVAL_EPOCHS="${EVAL_EPOCHS:-40}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-50}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# The LFST loss-weight evaluation runs after the SASGT parameter suite.
LFST_WEIGHTS="${LFST_WEIGHTS:-0.05 0.1 0.2 0.5 1.0}"
LFST_SUITE_NAME="${LFST_SUITE_NAME:-lfst_weight_sensitivity_official_lr1e3_10seeds}"
LFST_EVAL_DATASETS="${LFST_EVAL_DATASETS:-New_FUSAR MSTAR_SOC SAR_ACD}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi

mkdir -p "$RUN_ROOT" "$OUTPUT_ROOT" "$LOG_ROOT"

effective_batch=$((TRAIN_BATCH_SIZE * TRAIN_GPUS * TRAIN_ACCUM_ITER))

# tag | gamma | reliability window | temperature | run name
# The default (gamma=1, w=7, tau=1) checkpoint already exists and is not
# retrained. Every new run changes exactly one SASGT parameter.
declare -a TRAIN_VARIANTS=(
  "gamma0|0|7|1.0|phyd_sasgt_gamma0_${TRAIN_EPOCHS}e_bs${effective_batch}"
  "gamma2|2|7|1.0|phyd_sasgt_gamma2_${TRAIN_EPOCHS}e_bs${effective_batch}"
  "w3|1|3|1.0|phyd_sasgt_w3_${TRAIN_EPOCHS}e_bs${effective_batch}"
  "w11|1|11|1.0|phyd_sasgt_w11_${TRAIN_EPOCHS}e_bs${effective_batch}"
  "tau0p5|1|7|0.5|phyd_sasgt_tau0p5_${TRAIN_EPOCHS}e_bs${effective_batch}"
  "tau2|1|7|2.0|phyd_sasgt_tau2_${TRAIN_EPOCHS}e_bs${effective_batch}"
)

log() {
  echo "[$(date '+%F %T')] $*"
}

checkpoint_for_run() {
  local run_name="$1"
  echo "$RUN_ROOT/$run_name/checkpoint-$((TRAIN_EPOCHS - 1)).pth"
}

validate_common_inputs() {
  if [[ ! -f "$SOURCE_CHECKPOINT" ]]; then
    echo "Missing encoder initialization checkpoint: $SOURCE_CHECKPOINT" >&2
    exit 1
  fi
  if [[ ! -f "$DEFAULT_CHECKPOINT" ]]; then
    echo "Missing default SASGT checkpoint: $DEFAULT_CHECKPOINT" >&2
    exit 1
  fi
  if [[ ! -d "$DATA_PATH" ]]; then
    echo "Missing pre-training dataset: $DATA_PATH" >&2
    exit 1
  fi
}

train_variant() {
  local tag="$1"
  local gamma="$2"
  local window="$3"
  local temperature="$4"
  local run_name="$5"
  local port="$6"
  local output_dir="$RUN_ROOT/$run_name"
  local final_checkpoint
  final_checkpoint="$(checkpoint_for_run "$run_name")"
  local run_log="$LOG_ROOT/${run_name}.pretrain.log"
  local resume=""
  local init_checkpoint="$SOURCE_CHECKPOINT"

  if [[ -f "$final_checkpoint" ]]; then
    log "Skip completed pre-training: $tag ($final_checkpoint)"
    return
  fi

  if [[ -f "$output_dir/checkpoint-last.pth" ]]; then
    resume="$output_dir/checkpoint-last.pth"
    init_checkpoint=""
    log "Resume pre-training: $tag from $resume"
  elif [[ -d "$output_dir" ]] && find "$output_dir" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to overwrite non-empty run without checkpoint-last: $output_dir" >&2
    exit 1
  else
    log "Start pre-training: $tag (gamma=$gamma, w=$window, tau=$temperature)"
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
    SASGT_GAMMA="$gamma" \
    SASGT_RELIABILITY_WINDOW="$window" \
    SASGT_TEMPERATURE="$temperature" \
    SASGT_MODE=complete \
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
  log "Finished pre-training: $tag"
}

build_eval_specs() {
  EVAL_TAGS=("default")
  EVAL_CHECKPOINTS=("$DEFAULT_CHECKPOINT")

  local spec tag gamma window temperature run_name
  for spec in "${TRAIN_VARIANTS[@]}"; do
    IFS='|' read -r tag gamma window temperature run_name <<< "$spec"
    EVAL_TAGS+=("$tag")
    EVAL_CHECKPOINTS+=("$(checkpoint_for_run "$run_name")")
  done
}

build_manifest() {
  local manifest="$OUTPUT_ROOT/expected_matrix.tsv"
  printf 'model\tcheckpoint\tdatasets\tprotocols\tshots\tseeds\n' > "$manifest"
  local index
  for index in "${!EVAL_TAGS[@]}"; do
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${EVAL_TAGS[$index]}" \
      "${EVAL_CHECKPOINTS[$index]}" \
      "$EVAL_DATASET" \
      "MIM_finetune MIM_linear" \
      "$EVAL_SHOTS" \
      "$EVAL_SEEDS" \
      >> "$manifest"
  done
  echo "$manifest"
}

validate_eval_checkpoints() {
  local index checkpoint
  for index in "${!EVAL_TAGS[@]}"; do
    checkpoint="${EVAL_CHECKPOINTS[$index]}"
    if [[ ! -f "$checkpoint" ]]; then
      echo "Missing evaluation checkpoint for ${EVAL_TAGS[$index]}: $checkpoint" >&2
      echo "Run ACTION=pretrain first." >&2
      exit 1
    fi
  done
}

evaluate_checkpoint_protocol() {
  local tag="$1"
  local checkpoint="$2"
  local protocol="$3"
  local output_dir="$OUTPUT_ROOT/$tag"
  local -a devices
  read -r -a devices <<< "$EVAL_CUDA_DEVICES"
  if [[ ${#devices[@]} -eq 0 ]]; then
    echo "EVAL_CUDA_DEVICES must contain at least one device ID" >&2
    exit 2
  fi

  local num_shards="${#devices[@]}"
  local -a pids=()
  local shard pid failed=0
  log "Evaluate $tag: protocol=$protocol, checkpoint=$checkpoint, lr=$EVAL_LR"

  for ((shard = 0; shard < num_shards; shard++)); do
    env \
      CUDA_VISIBLE_DEVICES="${devices[$shard]}" \
      CHECKPOINT="$checkpoint" \
      OUTPUT_DIR="$output_dir" \
      DATASETS="$EVAL_DATASET" \
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
    pid="$!"
    pids+=("$pid")
    log "Started $tag $protocol shard $shard/$num_shards on GPU ${devices[$shard]}: PID=$pid"
  done

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

evaluate_protocol_all() {
  local protocol="$1"
  local index
  for index in "${!EVAL_TAGS[@]}"; do
    evaluate_checkpoint_protocol \
      "${EVAL_TAGS[$index]}" \
      "${EVAL_CHECKPOINTS[$index]}" \
      "$protocol"
  done
}

summarize_results() {
  local manifest="$1"
  "$PYTHON_BIN" scripts/summarize_paper_ablation_results.py \
    --root "$OUTPUT_ROOT" \
    --manifest "$manifest"
}

run_lfst_weight_stage() {
  local lfst_action="$1"
  log "Start LFST loss-weight stage: action=$lfst_action, weights=$LFST_WEIGHTS"
  env \
    ACTION="$lfst_action" \
    LFST_WEIGHTS="$LFST_WEIGHTS" \
    INCLUDE_REFERENCE=0 \
    SUITE_NAME="$LFST_SUITE_NAME" \
    EVAL_DATASETS="$LFST_EVAL_DATASETS" \
    EVAL_PROTOCOLS="MIM_finetune MIM_linear" \
    EVAL_SHOTS="$EVAL_SHOTS" \
    EVAL_SEEDS="$EVAL_SEEDS" \
    EVAL_LR="$EVAL_LR" \
    EVAL_EPOCHS="$EVAL_EPOCHS" \
    EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
    EVAL_CUDA_DEVICES="$EVAL_CUDA_DEVICES" \
    REUSE_EXISTING_RESULTS=0 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    bash scripts/run_lfst_weight_sensitivity_10seeds.sh
  log "Finished LFST loss-weight stage: action=$lfst_action"
}

validate_common_inputs
build_eval_specs
manifest="$(build_manifest)"

if [[ "$ACTION" == "all" || "$ACTION" == "pretrain" ]]; then
  train_index=0
  for spec in "${TRAIN_VARIANTS[@]}"; do
    IFS='|' read -r tag gamma window temperature run_name <<< "$spec"
    train_variant \
      "$tag" "$gamma" "$window" "$temperature" "$run_name" \
      "$((MASTER_PORT_BASE + train_index))"
    train_index=$((train_index + 1))
  done
fi

if [[ "$ACTION" == "all" || "$ACTION" == "finetune" || "$ACTION" == "linear" || "$ACTION" == "eval" ]]; then
  validate_eval_checkpoints
fi

if [[ "$ACTION" == "all" || "$ACTION" == "finetune" || "$ACTION" == "eval" ]]; then
  evaluate_protocol_all MIM_finetune
fi

if [[ "$ACTION" == "all" || "$ACTION" == "linear" || "$ACTION" == "eval" ]]; then
  evaluate_protocol_all MIM_linear
fi

if [[ "$ACTION" == "all" || "$ACTION" == "eval" || "$ACTION" == "summary" ]]; then
  summarize_results "$manifest"
fi

if [[ "$ACTION" == "all" || "$ACTION" == "lfst" ]]; then
  run_lfst_weight_stage eval
fi

if [[ "$ACTION" == "summary" ]]; then
  run_lfst_weight_stage summary
fi

log "Done."
log "Manifest: $manifest"
log "Per-seed results: $OUTPUT_ROOT/results_per_seed.csv"
log "Mean/std results: $OUTPUT_ROOT/results_mean_std.csv"
log "LFST results: $ROOT/few_shot_classification/finetune/output_${LFST_SUITE_NAME}"
log "Logs: $LOG_ROOT"
