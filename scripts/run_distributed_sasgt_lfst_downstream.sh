#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FINETUNE_DIR="$ROOT/few_shot_classification/finetune"
cd "$ROOT"

ACTION="${ACTION:-status}"
case "$ACTION" in
  launch|worker|dynamic-launch|dynamic-worker|clear-locks|status|summary) ;;
  *) echo "ACTION must be launch, worker, dynamic-launch, dynamic-worker, clear-locks, status, or summary; got: $ACTION" >&2; exit 2 ;;
esac

PROTOCOL="${PROTOCOL:-MIM_finetune}"
case "$PROTOCOL" in
  finetune|MIM_finetune) PROTOCOL=MIM_finetune ;;
  linear|MIM_linear) PROTOCOL=MIM_linear ;;
  *) echo "PROTOCOL must be MIM_finetune or MIM_linear; got: $PROTOCOL" >&2; exit 2 ;;
esac

TOTAL_SHARDS="${TOTAL_SHARDS:-1}"
SHARD_OFFSET="${SHARD_OFFSET:-0}"
GLOBAL_SHARD_ID="${GLOBAL_SHARD_ID:-0}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
HOST_TAG="${HOST_TAG:-$(hostname -s)}"
WORKER_OFFSET="${WORKER_OFFSET:-0}"
DYNAMIC_WORKER_ID="${DYNAMIC_WORKER_ID:-0}"

EVAL_SHOTS="${EVAL_SHOTS:-10 20 40}"
EVAL_SEEDS="${EVAL_SEEDS:-0 1 2 3 4 5 6 7 8 9}"
EVAL_LR="${EVAL_LR:-1e-3}"
EVAL_EPOCHS="${EVAL_EPOCHS:-40}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-50}"

SASGT_SUITE="${SASGT_SUITE:-sasgt_parameter_sensitivity_lr1e3_10seeds}"
LFST_SUITE="${LFST_SUITE:-lfst_weight_sensitivity_official_lr1e3_10seeds}"
SASGT_OUTPUT="$FINETUNE_DIR/output_${SASGT_SUITE}"
LFST_OUTPUT="$FINETUNE_DIR/output_${LFST_SUITE}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/distributed_sasgt_lfst_downstream}"
LOCK_ROOT="${LOCK_ROOT:-$ROOT/few_shot_classification/finetune/.dynamic_sasgt_lfst_locks}"

DATA_ROOT="${DATA_ROOT:-$ROOT/dataset/modelscope/extracted/classification_dataset/few_shot_classification}"
if [[ ! -d "$DATA_ROOT" ]]; then
  DATA_ROOT="${DATA_ROOT_FALLBACK:-$ROOT/dataset/modelscope/extracted/classification_dataset}"
fi
if [[ -d "$DATA_ROOT" ]]; then
  DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"
fi

declare -a SASGT_SPECS=(
  "default|$ROOT/runs/phyd_ckpt300_target_pilot_30e_bs1088_msgt_only/checkpoint-29.pth"
  "gamma0|$ROOT/runs/phyd_sasgt_gamma0_30e_bs1088/checkpoint-29.pth"
  "gamma2|$ROOT/runs/phyd_sasgt_gamma2_30e_bs1088/checkpoint-29.pth"
  "w3|$ROOT/runs/phyd_sasgt_w3_30e_bs1088/checkpoint-29.pth"
  "w11|$ROOT/runs/phyd_sasgt_w11_30e_bs1088/checkpoint-29.pth"
  "tau0p5|$ROOT/runs/phyd_sasgt_tau0p5_30e_bs1088/checkpoint-29.pth"
  "tau2|$ROOT/runs/phyd_sasgt_tau2_30e_bs1088/checkpoint-29.pth"
)

declare -a LFST_SPECS=(
  "lfst_weight_0p05|0.05|$ROOT/runs/phyd_lfst_weight_0p05_ft250_to300_bs1024/checkpoint-300.pth"
  "lfst_weight_0p1|0.1|$ROOT/runs/phyd_lfst_weight_0p1_ft250_to300_bs1024/checkpoint-300.pth"
  "lfst_weight_0p2|0.2|$ROOT/runs/phyd_lfst_weight_0p2_ft250_to300_bs1024/checkpoint-300.pth"
  "lfst_weight_0p5|0.5|$ROOT/runs/phyd_lfst_weight_0p5_ft250_to300_bs1024/checkpoint-300.pth"
  "lfst_weight_1p0|1.0|$ROOT/runs/phyd_lfst_weight_1p0_ft250_to300_bs1024/checkpoint-300.pth"
)

log() {
  echo "[$(date '+%F %T')] $*"
}

validate_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "$name must be a non-negative integer; got: $value" >&2
    exit 2
  fi
}

validate_shards() {
  validate_integer TOTAL_SHARDS "$TOTAL_SHARDS"
  validate_integer SHARD_OFFSET "$SHARD_OFFSET"
  validate_integer GLOBAL_SHARD_ID "$GLOBAL_SHARD_ID"
  if (( TOTAL_SHARDS < 1 )); then
    echo "TOTAL_SHARDS must be positive" >&2
    exit 2
  fi
}

validate_checkpoints() {
  local spec tag weight checkpoint
  for spec in "${SASGT_SPECS[@]}"; do
    IFS='|' read -r tag checkpoint <<< "$spec"
    [[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }
  done
  for spec in "${LFST_SPECS[@]}"; do
    IFS='|' read -r tag weight checkpoint <<< "$spec"
    [[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }
  done
}

ensure_environment() {
  if [[ ! -d "$FINETUNE_DIR" ]]; then
    echo "Missing downstream project: $FINETUNE_DIR" >&2
    exit 1
  fi
  if [[ ! -d "$DATA_ROOT" ]]; then
    echo "Missing classification dataset: $DATA_ROOT" >&2
    exit 1
  fi
  if ! python - <<'PY' >/dev/null 2>&1
import dassl
PY
  then
    local dassl_dir="$ROOT/few_shot_classification/Dassl.pytorch"
    if [[ -d "$dassl_dir/dassl" ]]; then
      export PYTHONPATH="$dassl_dir:${PYTHONPATH:-}"
    else
      echo "Dassl is unavailable in the active environment" >&2
      exit 1
    fi
  fi
}

resolve_dataset_name() {
  case "$1" in
    mstar|MSTAR|MSTAR_SOC) echo MSTAR_SOC ;;
    fusar|fusar_ship|New_FUSAR) echo New_FUSAR ;;
    sar_acd|SAR_ACD) echo SAR_ACD ;;
    *) echo "$1" ;;
  esac
}

link_dataset() {
  local name="$1"
  local target=""
  local alias candidate
  local -a aliases=("$name")
  case "$name" in
    MSTAR_SOC) aliases+=(mstar MSTAR) ;;
    New_FUSAR) aliases+=(fusar_ship fusar FUSAR) ;;
    SAR_ACD) aliases+=(sar_acd SAR-ACD) ;;
  esac

  for alias in "${aliases[@]}"; do
    for candidate in \
      "$DATA_ROOT/$alias" \
      "$DATA_ROOT/data/$alias" \
      "$DATA_ROOT/finetune/data/$alias" \
      "$DATA_ROOT/few_shot_classification/$alias" \
      "$DATA_ROOT/few_shot_classification/data/$alias" \
      "$DATA_ROOT/few_shot_classification/finetune/data/$alias"
    do
      if [[ -d "$candidate" ]]; then
        target="$(cd "$candidate" && pwd)"
        break 2
      fi
    done
  done

  if [[ -z "$target" ]]; then
    echo "Dataset $name not found under $DATA_ROOT" >&2
    exit 1
  fi
  mkdir -p "$FINETUNE_DIR/data"
  ln -sfn "$target" "$FINETUNE_DIR/data/$name"
}

is_complete() {
  local log_path="$1"
  local seed="$2"
  [[ -f "$log_path" ]] \
    && grep -qE '^\* accuracy:' "$log_path" \
    && grep -qE "^SEED:[[:space:]]*${seed}[[:space:]]*$" "$log_path"
}

run_job() {
  local group="$1"
  local tag="$2"
  local checkpoint="$3"
  local raw_dataset="$4"
  local shots="$5"
  local seed="$6"
  local output_root dataset run_dir result_log

  if [[ "$group" == sasgt ]]; then
    output_root="$SASGT_OUTPUT"
  else
    output_root="$LFST_OUTPUT"
  fi
  dataset="$(resolve_dataset_name "$raw_dataset")"
  link_dataset "$dataset"
  run_dir="$output_root/$tag/$dataset/$PROTOCOL/vit_b16_${shots}shots/seed${seed}"
  result_log="$run_dir/log.txt"

  if is_complete "$result_log" "$seed"; then
    log "Skip completed: $group/$tag/$dataset/$PROTOCOL/${shots}shot/seed${seed}"
    return
  fi
  if [[ -d "$run_dir" ]]; then
    rm -rf -- "$run_dir"
  fi

  export MIM_CKPT="$checkpoint"
  export MIM_USE_SFAFM=0
  export MIM_FEATURE_POOL=cls
  export MIM_SFAFM_LAYOUT=late
  export MIM_MODEL_FAMILY=phyd_mae

  log "Run: $group/$tag/$dataset/$PROTOCOL/${shots}shot/seed${seed}"
  cd "$FINETUNE_DIR"
  python train.py \
    --root "$FINETUNE_DIR/data" \
    --seed "$seed" \
    --trainer "$PROTOCOL" \
    --dataset-config-file "configs/datasets/${dataset}.yaml" \
    --config-file "configs/trainers/${PROTOCOL}/vit_b16.yaml" \
    --output-dir "$run_dir" \
    DATASET.NUM_SHOTS "$shots" \
    OPTIM.LR "$EVAL_LR" \
    OPTIM.MAX_EPOCH "$EVAL_EPOCHS" \
    DATALOADER.TRAIN_X.BATCH_SIZE "$EVAL_BATCH_SIZE" \
    DATALOADER.TEST.BATCH_SIZE "$EVAL_BATCH_SIZE"
  cd "$ROOT"
}

consider_job() {
  local group="$1"
  local tag="$2"
  local checkpoint="$3"
  local dataset="$4"
  local shots="$5"
  local seed="$6"
  local current_job="$job_index"
  job_index=$((job_index + 1))

  if (( current_job % TOTAL_SHARDS != GLOBAL_SHARD_ID )); then
    return
  fi
  selected_jobs=$((selected_jobs + 1))
  run_job "$group" "$tag" "$checkpoint" "$dataset" "$shots" "$seed"
}

declare -a DYNAMIC_JOBS=()
CURRENT_LOCK=""

build_dynamic_jobs() {
  DYNAMIC_JOBS=()
  local spec tag weight checkpoint dataset shots seed
  for spec in "${SASGT_SPECS[@]}"; do
    IFS='|' read -r tag checkpoint <<< "$spec"
    for shots in $EVAL_SHOTS; do
      for seed in $EVAL_SEEDS; do
        DYNAMIC_JOBS+=("sasgt|$tag|$checkpoint|New_FUSAR|$shots|$seed")
      done
    done
  done
  for spec in "${LFST_SPECS[@]}"; do
    IFS='|' read -r tag weight checkpoint <<< "$spec"
    for dataset in New_FUSAR MSTAR_SOC SAR_ACD; do
      for shots in $EVAL_SHOTS; do
        for seed in $EVAL_SEEDS; do
          DYNAMIC_JOBS+=("lfst|$tag|$checkpoint|$dataset|$shots|$seed")
        done
      done
    done
  done
}

job_result_log() {
  local group="$1"
  local tag="$2"
  local dataset="$3"
  local shots="$4"
  local seed="$5"
  local output_root
  if [[ "$group" == sasgt ]]; then
    output_root="$SASGT_OUTPUT"
  else
    output_root="$LFST_OUTPUT"
  fi
  dataset="$(resolve_dataset_name "$dataset")"
  echo "$output_root/$tag/$dataset/$PROTOCOL/vit_b16_${shots}shots/seed${seed}/log.txt"
}

cleanup_dynamic_lock() {
  if [[ -n "$CURRENT_LOCK" && -d "$CURRENT_LOCK" ]]; then
    rm -rf -- "$CURRENT_LOCK"
  fi
  CURRENT_LOCK=""
}

dynamic_signal_exit() {
  cleanup_dynamic_lock
  exit 143
}

try_dynamic_job() {
  local record="$1"
  local group tag checkpoint dataset shots seed result_log key lock_dir status
  IFS='|' read -r group tag checkpoint dataset shots seed <<< "$record"
  result_log="$(job_result_log "$group" "$tag" "$dataset" "$shots" "$seed")"
  if is_complete "$result_log" "$seed"; then
    return 1
  fi

  key="${group}__${tag}__${dataset}__${PROTOCOL}__${shots}shot__seed${seed}"
  lock_dir="$LOCK_ROOT/$PROTOCOL/${key}.lock"
  mkdir -p "$LOCK_ROOT/$PROTOCOL"
  if ! mkdir "$lock_dir" 2>/dev/null; then
    return 1
  fi
  CURRENT_LOCK="$lock_dir"
  printf 'host=%s\npid=%s\nworker=%s\ntime=%s\n' \
    "$HOST_TAG" "$$" "$DYNAMIC_WORKER_ID" "$(date -Iseconds)" \
    > "$lock_dir/owner.txt"

  if is_complete "$result_log" "$seed"; then
    cleanup_dynamic_lock
    return 1
  fi

  status=0
  run_job "$group" "$tag" "$checkpoint" "$dataset" "$shots" "$seed" || status=$?
  cleanup_dynamic_lock
  return "$status"
}

run_dynamic_worker() {
  validate_integer DYNAMIC_WORKER_ID "$DYNAMIC_WORKER_ID"
  validate_checkpoints
  ensure_environment
  build_dynamic_jobs
  trap dynamic_signal_exit INT TERM
  trap cleanup_dynamic_lock EXIT

  local total_jobs="${#DYNAMIC_JOBS[@]}"
  local iteration=0 start step index claimed remaining status
  while true; do
    claimed=0
    start=$(( (DYNAMIC_WORKER_ID * 97 + iteration * 31) % total_jobs ))
    for ((step = 0; step < total_jobs; step++)); do
      index=$(( (start + step) % total_jobs ))
      if try_dynamic_job "${DYNAMIC_JOBS[$index]}"; then
        claimed=1
        break
      else
        status=$?
        if (( status > 1 )); then
          log "Job failed with status=$status; worker=$DYNAMIC_WORKER_ID will continue with the shared queue"
        fi
      fi
    done

    if (( claimed )); then
      iteration=$((iteration + 1))
      continue
    fi

    if [[ "$PROTOCOL" == MIM_finetune ]]; then
      remaining=$((660 - $(count_complete "$SASGT_OUTPUT" MIM_finetune) - $(count_complete "$LFST_OUTPUT" MIM_finetune)))
    else
      remaining=$((660 - $(count_complete "$SASGT_OUTPUT" MIM_linear) - $(count_complete "$LFST_OUTPUT" MIM_linear)))
    fi
    if (( remaining <= 0 )); then
      log "Dynamic worker complete: protocol=$PROTOCOL worker=$DYNAMIC_WORKER_ID"
      return
    fi
    sleep 10
    iteration=$((iteration + 1))
  done
}

run_worker() {
  validate_shards
  if (( GLOBAL_SHARD_ID >= TOTAL_SHARDS )); then
    echo "GLOBAL_SHARD_ID must be smaller than TOTAL_SHARDS" >&2
    exit 2
  fi
  validate_checkpoints
  ensure_environment

  local spec tag weight checkpoint dataset shots seed
  job_index=0
  selected_jobs=0

  for spec in "${SASGT_SPECS[@]}"; do
    IFS='|' read -r tag checkpoint <<< "$spec"
    for shots in $EVAL_SHOTS; do
      for seed in $EVAL_SEEDS; do
        consider_job sasgt "$tag" "$checkpoint" New_FUSAR "$shots" "$seed"
      done
    done
  done

  for spec in "${LFST_SPECS[@]}"; do
    IFS='|' read -r tag weight checkpoint <<< "$spec"
    for dataset in New_FUSAR MSTAR_SOC SAR_ACD; do
      for shots in $EVAL_SHOTS; do
        for seed in $EVAL_SEEDS; do
          consider_job lfst "$tag" "$checkpoint" "$dataset" "$shots" "$seed"
        done
      done
    done
  done

  log "Worker complete: protocol=$PROTOCOL shard=$GLOBAL_SHARD_ID/$TOTAL_SHARDS selected=$selected_jobs total=$job_index"
}

launch_workers() {
  validate_shards
  local -a devices pids
  read -r -a devices <<< "$CUDA_DEVICES"
  if [[ ${#devices[@]} -eq 0 ]]; then
    echo "CUDA_DEVICES must list at least one device" >&2
    exit 2
  fi
  if (( SHARD_OFFSET + ${#devices[@]} > TOTAL_SHARDS )); then
    echo "Local shard range exceeds TOTAL_SHARDS" >&2
    exit 2
  fi

  local log_dir="$LOG_ROOT/$PROTOCOL/$HOST_TAG"
  mkdir -p "$log_dir"
  pids=()
  local local_id global_id device pid failed=0
  for local_id in "${!devices[@]}"; do
    global_id=$((SHARD_OFFSET + local_id))
    device="${devices[$local_id]}"
    env \
      ACTION=worker \
      PROTOCOL="$PROTOCOL" \
      TOTAL_SHARDS="$TOTAL_SHARDS" \
      GLOBAL_SHARD_ID="$global_id" \
      CUDA_VISIBLE_DEVICES="$device" \
      EVAL_LR="$EVAL_LR" \
      EVAL_EPOCHS="$EVAL_EPOCHS" \
      EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
      OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 \
      bash "$0" \
      > "$log_dir/worker-${global_id}.log" 2>&1 &
    pid="$!"
    pids+=("$pid")
    log "Started protocol=$PROTOCOL global_shard=$global_id/$TOTAL_SHARDS GPU=$device PID=$pid"
  done

  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if (( failed )); then
    echo "One or more workers failed; inspect $log_dir" >&2
    exit 1
  fi
  log "Host launch complete: protocol=$PROTOCOL host=$HOST_TAG"
}

launch_dynamic_workers() {
  local -a devices pids
  read -r -a devices <<< "$CUDA_DEVICES"
  if [[ ${#devices[@]} -eq 0 ]]; then
    echo "CUDA_DEVICES must list at least one device" >&2
    exit 2
  fi

  local log_dir="$LOG_ROOT/dynamic-$PROTOCOL/$HOST_TAG"
  mkdir -p "$log_dir"
  pids=()
  local local_id global_id device pid failed=0
  for local_id in "${!devices[@]}"; do
    global_id=$((WORKER_OFFSET + local_id))
    device="${devices[$local_id]}"
    env \
      ACTION=dynamic-worker \
      PROTOCOL="$PROTOCOL" \
      DYNAMIC_WORKER_ID="$global_id" \
      CUDA_VISIBLE_DEVICES="$device" \
      EVAL_LR="$EVAL_LR" \
      EVAL_EPOCHS="$EVAL_EPOCHS" \
      EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
      HOST_TAG="$HOST_TAG" \
      OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 \
      bash "$0" \
      > "$log_dir/worker-${global_id}.log" 2>&1 &
    pid="$!"
    pids+=("$pid")
    log "Started dynamic protocol=$PROTOCOL worker=$global_id GPU=$device PID=$pid"
  done

  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if (( failed )); then
    echo "One or more dynamic workers failed; inspect $log_dir" >&2
    exit 1
  fi
  log "Dynamic host launch complete: protocol=$PROTOCOL host=$HOST_TAG"
}

clear_dynamic_locks() {
  local active_pids
  active_pids="$({
    ps -eo pid=,args= \
      | awk -v self="$$" \
          '$1 != self && $0 ~ /bash .*run_distributed_sasgt_lfst_downstream\.sh/ {print $1}'
  } || true)"
  if [[ -n "$active_pids" && "${FORCE_CLEAR_LOCKS:-0}" != 1 ]]; then
    echo "Refusing to clear locks while local distributed workers are running: $active_pids" >&2
    echo "Stop workers on every instance first, or set FORCE_CLEAR_LOCKS=1 after verifying they are stopped." >&2
    exit 1
  fi
  rm -rf -- "$LOCK_ROOT/$PROTOCOL"
  echo "Cleared dynamic locks for $PROTOCOL: $LOCK_ROOT/$PROTOCOL"
}

count_complete() {
  local root="$1"
  local protocol="$2"
  if [[ ! -d "$root" ]]; then
    echo 0
    return
  fi
  {
    find "$root" -type f \
      -path "*/${protocol}/vit_b16_*shots/seed*/log.txt" \
      -exec grep -l '^\* accuracy:' {} + 2>/dev/null || true
  } | wc -l
}

show_status() {
  local sasgt_ft sasgt_lp lfst_ft lfst_lp
  sasgt_ft="$(count_complete "$SASGT_OUTPUT" MIM_finetune)"
  sasgt_lp="$(count_complete "$SASGT_OUTPUT" MIM_linear)"
  lfst_ft="$(count_complete "$LFST_OUTPUT" MIM_finetune)"
  lfst_lp="$(count_complete "$LFST_OUTPUT" MIM_linear)"
  echo "SASGT fine-tuning:    $sasgt_ft/210"
  echo "SASGT linear probing: $sasgt_lp/210"
  echo "LFST fine-tuning:     $lfst_ft/450"
  echo "LFST linear probing:  $lfst_lp/450"
  echo "Fine-tuning total:    $((sasgt_ft + lfst_ft))/660"
  echo "Linear total:         $((sasgt_lp + lfst_lp))/660"
  echo "Overall total:        $((sasgt_ft + sasgt_lp + lfst_ft + lfst_lp))/1320"
}

write_manifests() {
  mkdir -p "$SASGT_OUTPUT" "$LFST_OUTPUT"
  local sasgt_manifest="$SASGT_OUTPUT/expected_matrix.tsv"
  local lfst_manifest="$LFST_OUTPUT/expected_matrix.tsv"
  local spec tag weight checkpoint

  printf 'model\tcheckpoint\tdatasets\tprotocols\tshots\tseeds\n' > "$sasgt_manifest"
  for spec in "${SASGT_SPECS[@]}"; do
    IFS='|' read -r tag checkpoint <<< "$spec"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$tag" "$checkpoint" New_FUSAR "MIM_finetune MIM_linear" \
      "$EVAL_SHOTS" "$EVAL_SEEDS" >> "$sasgt_manifest"
  done

  printf 'model\tlfst_weight\tcheckpoint\tdatasets\tprotocols\tshots\tseeds\n' > "$lfst_manifest"
  for spec in "${LFST_SPECS[@]}"; do
    IFS='|' read -r tag weight checkpoint <<< "$spec"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$tag" "$weight" "$checkpoint" "New_FUSAR MSTAR_SOC SAR_ACD" \
      "MIM_finetune MIM_linear" "$EVAL_SHOTS" "$EVAL_SEEDS" >> "$lfst_manifest"
  done
}

summarize_all() {
  write_manifests
  python scripts/summarize_paper_ablation_results.py \
    --root "$SASGT_OUTPUT" \
    --manifest "$SASGT_OUTPUT/expected_matrix.tsv"
  python scripts/summarize_lfst_weight_sensitivity.py \
    --root "$LFST_OUTPUT" \
    --manifest "$LFST_OUTPUT/expected_matrix.tsv"
  show_status
}

case "$ACTION" in
  launch) launch_workers ;;
  worker) run_worker ;;
  dynamic-launch) launch_dynamic_workers ;;
  dynamic-worker) run_dynamic_worker ;;
  clear-locks) clear_dynamic_locks ;;
  status) show_status ;;
  summary) summarize_all ;;
esac
