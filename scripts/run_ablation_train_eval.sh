#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p logs runs

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SUITE_NAME="${SUITE_NAME:-ablation_${STAMP}}"
SUITE_LOG="${SUITE_LOG:-logs/${SUITE_NAME}.log}"
SUMMARY_CSV="${SUMMARY_CSV:-logs/${SUITE_NAME}_summary.csv}"

TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}"

DATA_PATH="${DATA_PATH:-dataset/modelscope/extracted/Pretraining_dataset}"
INIT_CKPT="${INIT_CKPT:-weights/mae_pretrain_vit_base.pth}"
INIT_CKPT_SCOPE="${INIT_CKPT_SCOPE:-encoder}"

PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-256}"
PRETRAIN_BLR="${PRETRAIN_BLR:-5e-5}"
PRETRAIN_WARMUP_EPOCHS="${PRETRAIN_WARMUP_EPOCHS:-20}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-100}"
PRETRAIN_SAVE_FREQ="${PRETRAIN_SAVE_FREQ:-25}"
TARGET_NORM="${TARGET_NORM:-patch}"
CHECKPOINT_EPOCHS="${CHECKPOINT_EPOCHS:-49 99}"
SKIP_PRETRAIN_IF_DONE="${SKIP_PRETRAIN_IF_DONE:-1}"

DOWNSTREAM_DATASETS="${DOWNSTREAM_DATASETS:-mstar}"
DOWNSTREAM_PROTOCOLS="${DOWNSTREAM_PROTOCOLS:-finetune}"
DOWNSTREAM_SHOTS="${DOWNSTREAM_SHOTS:-10}"
DOWNSTREAM_SEEDS="${DOWNSTREAM_SEEDS:-0}"
DOWNSTREAM_LR="${DOWNSTREAM_LR:-1e-4}"
DOWNSTREAM_EPOCHS="${DOWNSTREAM_EPOCHS:-40}"
DOWNSTREAM_BATCH_SIZE="${DOWNSTREAM_BATCH_SIZE:-}"
USE_SFAFM="${USE_SFAFM:-1}"
FORCE_EVAL="${FORCE_EVAL:-1}"

# Format: tag:grad_loss_weight:lfst_loss_weight:encoder_lr_scale[:target_norm[:sasgt_scales]]
# Override EXPERIMENTS to run a smaller/larger grid, for example:
# EXPERIMENTS="lfst1_enc1:1.0:1.0:1.0 sasgt_image:1.0:0.0:1.0:image:0.8,1.6,3.2,6.4"
EXPERIMENTS="${EXPERIMENTS:-lfst1_enc1:1.0:1.0:1.0 lfst0p3_enc1:1.0:0.3:1.0 lfst1_enc0p1:1.0:1.0:0.1 sasgt_only_enc1:1.0:0.0:1.0}"

echo "suite,experiment,checkpoint,dataset,protocol,shots,seed,accuracy,macro_f1,log" > "$SUMMARY_CSV"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$SUITE_LOG"
}

extract_metric() {
  local pattern="$1"
  local file="$2"
  grep -E "$pattern" "$file" | tail -n 1 | sed -E 's/.*: *([0-9.]+)%.*/\1/' || true
}

append_eval_summary() {
  local experiment="$1"
  local ckpt_epoch="$2"
  local eval_log="$3"
  local accuracy macro_f1
  accuracy="$(extract_metric 'accuracy:' "$eval_log")"
  macro_f1="$(extract_metric 'macro_f1:' "$eval_log")"
  if [[ -z "$accuracy" ]]; then
    accuracy="NA"
  fi
  if [[ -z "$macro_f1" ]]; then
    macro_f1="NA"
  fi
  echo "${SUITE_NAME},${experiment},${ckpt_epoch},${DOWNSTREAM_DATASETS},${DOWNSTREAM_PROTOCOLS},${DOWNSTREAM_SHOTS},${DOWNSTREAM_SEEDS},${accuracy},${macro_f1},${eval_log}" >> "$SUMMARY_CSV"
}

log "Suite: $SUITE_NAME"
log "Experiments: $EXPERIMENTS"
log "Checkpoint epochs: $CHECKPOINT_EPOCHS"
log "Summary CSV: $SUMMARY_CSV"

for spec in $EXPERIMENTS; do
  IFS=':' read -r tag grad_weight lfst_weight encoder_lr_scale experiment_target_norm experiment_sasgt_scales <<< "$spec"
  experiment_target_norm="${experiment_target_norm:-$TARGET_NORM}"
  experiment_sasgt_scales="${experiment_sasgt_scales:-${SASGT_SCALES:-0.8,1.6,3.2,6.4}}"
  run_name="${SUITE_NAME}_${tag}"
  output_dir="runs/${run_name}"
  pretrain_log="logs/${run_name}_pretrain.log"

  log "==== Pretrain $tag ===="
  log "grad=$grad_weight lfst=$lfst_weight encoder_lr_scale=$encoder_lr_scale target_norm=$experiment_target_norm sasgt_scales=$experiment_sasgt_scales output=$output_dir"

  final_ckpt="${output_dir}/checkpoint-$((PRETRAIN_EPOCHS - 1)).pth"
  if [[ "$SKIP_PRETRAIN_IF_DONE" == "1" && -f "$final_ckpt" ]]; then
    log "Skip pretrain because final checkpoint exists: $final_ckpt"
  else
    RUN_NAME="$run_name" \
    CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" \
    DATA_PATH="$DATA_PATH" \
    OUTPUT_DIR="$output_dir" \
    BATCH_SIZE="$PRETRAIN_BATCH_SIZE" \
    BLR="$PRETRAIN_BLR" \
    WARMUP_EPOCHS="$PRETRAIN_WARMUP_EPOCHS" \
    EPOCHS="$PRETRAIN_EPOCHS" \
    SAVE_FREQ="$PRETRAIN_SAVE_FREQ" \
    GRAD_LOSS_WEIGHT="$grad_weight" \
    LFST_LOSS_WEIGHT="$lfst_weight" \
    TARGET_NORM="$experiment_target_norm" \
    SASGT_SCALES="$experiment_sasgt_scales" \
    ENCODER_LR_SCALE="$encoder_lr_scale" \
    INIT_CKPT="$INIT_CKPT" \
    INIT_CKPT_SCOPE="$INIT_CKPT_SCOPE" \
    bash scripts/pretrain_2xh100.sh 2>&1 | tee "$pretrain_log"
  fi

  for ckpt_epoch in $CHECKPOINT_EPOCHS; do
    ckpt_path="${output_dir}/checkpoint-${ckpt_epoch}.pth"
    if [[ ! -f "$ckpt_path" ]]; then
      log "Skip eval; checkpoint not found: $ckpt_path"
      continue
    fi

    eval_name="${run_name}_ckpt${ckpt_epoch}_${DOWNSTREAM_DATASETS}_${DOWNSTREAM_PROTOCOLS}_${DOWNSTREAM_SHOTS}shot"
    eval_log="logs/${eval_name}.log"
    eval_output="few_shot_classification/finetune/output_${eval_name}"

    log "==== Eval $tag checkpoint-$ckpt_epoch ===="
    log "checkpoint=$ckpt_path output=$eval_output"

    export CUDA_VISIBLE_DEVICES="$EVAL_CUDA_VISIBLE_DEVICES"
    export RUN_NAME="$eval_name"
    export CHECKPOINT="$ckpt_path"
    export OUTPUT_DIR="$eval_output"
    export DATASETS="$DOWNSTREAM_DATASETS"
    export PROTOCOLS="$DOWNSTREAM_PROTOCOLS"
    export SHOTS="$DOWNSTREAM_SHOTS"
    export SEEDS="$DOWNSTREAM_SEEDS"
    export FORCE="$FORCE_EVAL"
    export USE_SFAFM="$USE_SFAFM"
    export LR="$DOWNSTREAM_LR"
    export EPOCHS="$DOWNSTREAM_EPOCHS"
    if [[ -n "$DOWNSTREAM_BATCH_SIZE" ]]; then
      export BATCH_SIZE="$DOWNSTREAM_BATCH_SIZE"
    else
      unset BATCH_SIZE || true
    fi

    bash scripts/run_sarjepa_fewshot_all.sh 2>&1 | tee "$eval_log"
    append_eval_summary "$tag" "$ckpt_epoch" "$eval_log"
  done
done

log "All experiments finished."
log "Summary CSV: $SUMMARY_CSV"
cat "$SUMMARY_CSV"
