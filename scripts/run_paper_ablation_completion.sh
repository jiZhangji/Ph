#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ACTION="${ACTION:-all}"
# Run the complete requested paper matrix by default. Individual groups remain
# selectable through EXPERIMENT_GROUPS when a smaller follow-up is needed.
EXPERIMENT_GROUPS="${EXPERIMENT_GROUPS:-all}"

SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-$ROOT/runs/sarjepa_official_phyd_ft250_bs1024_lfst0p1_image_2xh200/checkpoint-300.pth}"
DATA_PATH="${DATA_PATH:-$ROOT/dataset/modelscope/extracted/Pretraining_dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/few_shot_classification/finetune/output_paper_ablation_completion_5seeds}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/paper_ablation_completion_5seeds}"

TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1}"
TRAIN_GPUS="${TRAIN_GPUS:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-544}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-30}"
TRAIN_BLR="${TRAIN_BLR:-5e-6}"
TRAIN_WARMUP_EPOCHS="${TRAIN_WARMUP_EPOCHS:-3}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-16}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-26731}"

EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}"
EVAL_PROTOCOLS="${EVAL_PROTOCOLS:-MIM_finetune MIM_linear}"
EVAL_SHOTS="${EVAL_SHOTS:-10 20 40}"
EVAL_SEEDS="${EVAL_SEEDS:-0 1 2 3 4}"
EVAL_LR="${EVAL_LR:-1e-4}"
EVAL_EPOCHS="${EVAL_EPOCHS:-40}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-50}"
REUSE_EXISTING_RESULTS="${REUSE_EXISTING_RESULTS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi

case "$ACTION" in
  all|pretrain|eval|summary) ;;
  *) echo "ACTION must be all, pretrain, eval, or summary; got: $ACTION" >&2; exit 2 ;;
esac

has_group() {
  [[ "$EXPERIMENT_GROUPS" == "all" || " $EXPERIMENT_GROUPS " == *" $1 "* ]]
}

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" runs

declare -a eval_specs=()
declare -A eval_tags=()

add_eval() {
  local tag="$1"
  local checkpoint="$2"
  local datasets="$3"
  if [[ -n "${eval_tags[$tag]:-}" ]]; then
    return
  fi
  eval_tags[$tag]=1
  eval_specs+=("$tag|$checkpoint|$datasets")
}

# Core target ablation uses the pilot-selected checkpoints already recorded in
# release/experiment_records/PHYD_KEY_MODELS.txt.
if has_group core; then
  add_eval \
    "core_lfst_raw_c30_ckpt15" \
    "$ROOT/runs/phyd_ckpt300_target_pilot_30e_bs1088_lfst_raw_c30/checkpoint-15.pth" \
    "MSTAR_SOC New_FUSAR SAR_ACD"
  add_eval \
    "sasgt_complete_ckpt29" \
    "$ROOT/runs/phyd_ckpt300_target_pilot_30e_bs1088_msgt_only/checkpoint-29.pth" \
    "MSTAR_SOC New_FUSAR SAR_ACD"
fi

if has_group sasgt_internal; then
  add_eval \
    "sasgt_uniform_ckpt29" \
    "$ROOT/runs/phyd_ckpt300_sasgt_internal_uniform_30e_bs1088/checkpoint-29.pth" \
    "New_FUSAR"
  add_eval \
    "sasgt_adaptive_ckpt29" \
    "$ROOT/runs/phyd_ckpt300_sasgt_internal_adaptive_30e_bs1088/checkpoint-29.pth" \
    "New_FUSAR"
  add_eval \
    "sasgt_complete_ckpt29" \
    "$ROOT/runs/phyd_ckpt300_target_pilot_30e_bs1088_msgt_only/checkpoint-29.pth" \
    "New_FUSAR"
fi

if has_group lfst_sensitivity; then
  add_eval \
    "lfst_raw_c20_ckpt29" \
    "$ROOT/runs/phyd_ckpt300_lfst_raw_c20_30e_bs1088/checkpoint-29.pth" \
    "New_FUSAR"
  add_eval \
    "lfst_raw_c30_ckpt29" \
    "$ROOT/runs/phyd_ckpt300_target_pilot_30e_bs1088_lfst_raw_c30/checkpoint-29.pth" \
    "New_FUSAR"
  add_eval \
    "lfst_raw_c40_ckpt29" \
    "$ROOT/runs/phyd_ckpt300_lfst_raw_c40_30e_bs1088/checkpoint-29.pth" \
    "New_FUSAR"
  add_eval \
    "lfst_log_c20_ckpt29" \
    "$ROOT/runs/phyd_ckpt300_target_pilot_30e_bs1088_lfst_log_c20/checkpoint-29.pth" \
    "New_FUSAR"
  add_eval \
    "lfst_log_c30_ckpt29" \
    "$ROOT/runs/phyd_ckpt300_target_pilot_30e_bs1088_lfst_log_c30/checkpoint-29.pth" \
    "New_FUSAR"
  add_eval \
    "lfst_log_c40_ckpt29" \
    "$ROOT/runs/phyd_ckpt300_target_pilot_30e_bs1088_lfst_log_c40/checkpoint-29.pth" \
    "New_FUSAR"
fi

if has_group filter_comparison; then
  add_eval \
    "lfst_raw_c40_ckpt29" \
    "$ROOT/runs/phyd_ckpt300_lfst_raw_c40_30e_bs1088/checkpoint-29.pth" \
    "New_FUSAR"
  add_eval \
    "spatial_lpf_c40_ckpt29" \
    "$ROOT/runs/phyd_pixmim_lpf_only_from_best300_c40_30e_bs1088_2xh200/checkpoint-29.pth" \
    "New_FUSAR"
fi

if [[ ${#eval_specs[@]} -eq 0 ]]; then
  echo "No evaluation jobs selected by EXPERIMENT_GROUPS=$EXPERIMENT_GROUPS" >&2
  exit 2
fi

train_missing_variant() {
  local run_name="$1"
  local sasgt_mode="$2"
  local grad_weight="$3"
  local lfst_weight="$4"
  local lfst_mode="$5"
  local lfst_cutoff="$6"
  local output_dir="$ROOT/runs/$run_name"
  local final_checkpoint="$output_dir/checkpoint-29.pth"
  local run_log="$LOG_ROOT/${run_name}.pretrain.log"
  local resume=""
  local init_checkpoint="$SOURCE_CHECKPOINT"

  if [[ -f "$final_checkpoint" ]]; then
    echo "Skip completed pretraining: $final_checkpoint"
    return
  fi
  if [[ -f "$output_dir/checkpoint-last.pth" ]]; then
    resume="$output_dir/checkpoint-last.pth"
    init_checkpoint=""
    echo "Resume pretraining: $run_name from $resume"
  elif [[ -d "$output_dir" ]] && find "$output_dir" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to overwrite non-empty run without checkpoint-last: $output_dir" >&2
    exit 1
  else
    echo "Start pretraining: $run_name"
  fi

  local port=$((MASTER_PORT_BASE + train_index))
  train_index=$((train_index + 1))
  env \
    RUN_NAME="$run_name" \
    OUTPUT_DIR="$output_dir" \
    LOG_DIR="$output_dir" \
    DATA_PATH="$DATA_PATH" \
    CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" \
    GPUS="$TRAIN_GPUS" \
    MASTER_PORT="$port" \
    BATCH_SIZE="$TRAIN_BATCH_SIZE" \
    ACCUM_ITER=1 \
    EPOCHS="$TRAIN_EPOCHS" \
    BLR="$TRAIN_BLR" \
    WARMUP_EPOCHS="$TRAIN_WARMUP_EPOCHS" \
    NUM_WORKERS="$TRAIN_NUM_WORKERS" \
    RESUME="$resume" \
    INIT_CHECKPOINT="$init_checkpoint" \
    INIT_SCOPE=encoder \
    GRAD_LOSS_WEIGHT="$grad_weight" \
    LFST_LOSS_WEIGHT="$lfst_weight" \
    LFST_INPUT_MODE="$lfst_mode" \
    LFST_CUTOFF="$lfst_cutoff" \
    LFST_TARGET_TYPE=lfst \
    TARGET_NORM=image \
    SASGT_MODE="$sasgt_mode" \
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
}

if [[ "$ACTION" == "all" || "$ACTION" == "pretrain" ]]; then
  if [[ ! -f "$SOURCE_CHECKPOINT" ]]; then
    echo "Missing source checkpoint: $SOURCE_CHECKPOINT" >&2
    exit 1
  fi
  if [[ ! -d "$DATA_PATH" ]]; then
    echo "Missing pretraining dataset: $DATA_PATH" >&2
    exit 1
  fi

  train_index=0
  if has_group sasgt_internal; then
    train_missing_variant \
      "phyd_ckpt300_sasgt_internal_uniform_30e_bs1088" \
      uniform 1.0 0.0 raw 30
  fi
  if has_group lfst_sensitivity; then
    train_missing_variant \
      "phyd_ckpt300_lfst_raw_c20_30e_bs1088" \
      complete 0.0 1.0 raw 20
  fi
  if has_group lfst_sensitivity || has_group filter_comparison; then
    train_missing_variant \
      "phyd_ckpt300_lfst_raw_c40_30e_bs1088" \
      complete 0.0 1.0 raw 40
  fi
fi

manifest="$OUTPUT_ROOT/expected_matrix.tsv"
printf 'model\tcheckpoint\tdatasets\tprotocols\tshots\tseeds\n' > "$manifest"
for spec in "${eval_specs[@]}"; do
  IFS='|' read -r tag checkpoint datasets <<< "$spec"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$tag" "$checkpoint" "$datasets" "$EVAL_PROTOCOLS" "$EVAL_SHOTS" "$EVAL_SEEDS" \
    >> "$manifest"
done

if [[ "$ACTION" == "all" || "$ACTION" == "eval" ]]; then
  if [[ "$REUSE_EXISTING_RESULTS" == "1" ]]; then
    "$PYTHON_BIN" scripts/reuse_completed_downstream_results.py \
      --search-root "$ROOT/few_shot_classification/finetune" \
      --output-root "$OUTPUT_ROOT" \
      --manifest "$manifest" \
      --lr "$EVAL_LR" \
      --epochs "$EVAL_EPOCHS" \
      --batch-size "$EVAL_BATCH_SIZE"
  fi

  for spec in "${eval_specs[@]}"; do
    IFS='|' read -r tag checkpoint datasets <<< "$spec"
    if [[ ! -f "$checkpoint" ]]; then
      echo "Missing evaluation checkpoint: $checkpoint" >&2
      echo "Run ACTION=pretrain first if this is a newly added variant." >&2
      exit 1
    fi

    echo "========================================================================"
    echo "Evaluate: $tag"
    echo "Checkpoint: $checkpoint"
    echo "Datasets: $datasets"
    echo "========================================================================"
    env \
      CUDA_VISIBLE_DEVICES="$EVAL_CUDA_VISIBLE_DEVICES" \
      CHECKPOINT="$checkpoint" \
      OUTPUT_DIR="$OUTPUT_ROOT/$tag" \
      DATASETS="$datasets" \
      PROTOCOLS="$EVAL_PROTOCOLS" \
      SHOTS="$EVAL_SHOTS" \
      SEEDS="$EVAL_SEEDS" \
      LR="$EVAL_LR" \
      EPOCHS="$EVAL_EPOCHS" \
      BATCH_SIZE="$EVAL_BATCH_SIZE" \
      USE_SFAFM=0 \
      FEATURE_POOL=cls \
      FORCE=0 \
      bash scripts/run_sarjepa_fewshot_all.sh \
      2>&1 | tee -a "$LOG_ROOT/${tag}.eval.log"
  done
fi

if [[ "$ACTION" == "all" || "$ACTION" == "eval" || "$ACTION" == "summary" ]]; then
  "$PYTHON_BIN" scripts/summarize_paper_ablation_results.py \
    --root "$OUTPUT_ROOT" \
    --manifest "$manifest"
fi

echo "Done."
echo "Outputs: $OUTPUT_ROOT"
echo "Logs: $LOG_ROOT"
