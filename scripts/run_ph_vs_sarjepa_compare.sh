#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p logs runs

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SUITE_NAME="${SUITE_NAME:-ph_vs_sarjepa_${STAMP}}"
SUITE_LOG="${SUITE_LOG:-logs/${SUITE_NAME}.log}"
SUMMARY_CSV="${SUMMARY_CSV:-logs/${SUITE_NAME}_summary.csv}"

TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1}"
EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}"
GPUS="${GPUS:-2}"

DATA_PATH="${DATA_PATH:-dataset/modelscope/extracted/Pretraining_dataset}"
DATA_ROOT="${DATA_ROOT:-$ROOT/dataset/modelscope/extracted/classification_dataset/few_shot_classification}"

PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-300}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-256}"
PRETRAIN_WARMUP_EPOCHS="${PRETRAIN_WARMUP_EPOCHS:-20}"
PRETRAIN_SAVE_FREQ="${PRETRAIN_SAVE_FREQ:-50}"
CHECKPOINT_EPOCHS="${CHECKPOINT_EPOCHS:-$((PRETRAIN_EPOCHS - 1))}"

PH_RUN_NAME="${PH_RUN_NAME:-${SUITE_NAME}_ph}"
PH_OUTPUT_DIR="${PH_OUTPUT_DIR:-runs/${PH_RUN_NAME}}"
PH_BLR="${PH_BLR:-5e-5}"
PH_INIT_CKPT="${PH_INIT_CKPT:-weights/mae_pretrain_vit_base.pth}"
PH_INIT_CKPT_SCOPE="${PH_INIT_CKPT_SCOPE:-encoder}"
PH_GRAD_LOSS_WEIGHT="${PH_GRAD_LOSS_WEIGHT:-1.0}"
PH_LFST_LOSS_WEIGHT="${PH_LFST_LOSS_WEIGHT:-1.0}"
PH_TARGET_NORM="${PH_TARGET_NORM:-patch}"
PH_ENCODER_LR_SCALE="${PH_ENCODER_LR_SCALE:-1.0}"
PH_SASGT_SCALES="${PH_SASGT_SCALES:-0.8,1.6,3.2,6.4}"

SARJEPA_RUN_NAME="${SARJEPA_RUN_NAME:-${SUITE_NAME}_sarjepa}"
SARJEPA_OUTPUT_DIR="${SARJEPA_OUTPUT_DIR:-runs/${SARJEPA_RUN_NAME}}"
SARJEPA_BLR="${SARJEPA_BLR:-1e-3}"

DOWNSTREAM_DATASETS="${DOWNSTREAM_DATASETS:-mstar}"
DOWNSTREAM_PROTOCOLS="${DOWNSTREAM_PROTOCOLS:-finetune}"
DOWNSTREAM_SHOTS="${DOWNSTREAM_SHOTS:-10}"
DOWNSTREAM_SEEDS="${DOWNSTREAM_SEEDS:-0}"
DOWNSTREAM_EPOCHS="${DOWNSTREAM_EPOCHS:-40}"
DOWNSTREAM_LR="${DOWNSTREAM_LR:-1e-4}"
DOWNSTREAM_BATCH_SIZE="${DOWNSTREAM_BATCH_SIZE:-}"
FORCE_EVAL="${FORCE_EVAL:-1}"

RUN_PH="${RUN_PH:-1}"
RUN_SARJEPA="${RUN_SARJEPA:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
SKIP_PRETRAIN_IF_DONE="${SKIP_PRETRAIN_IF_DONE:-1}"

echo "suite,method,checkpoint,dataset,protocol,shots,seeds,accuracy,macro_f1,log" > "$SUMMARY_CSV"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$SUITE_LOG"
}

extract_metric() {
  local pattern="$1"
  local file="$2"
  grep -E "$pattern" "$file" | tail -n 1 | sed -E 's/.*: *([0-9.]+)%.*/\1/' || true
}

append_summary() {
  local method="$1"
  local ckpt_epoch="$2"
  local eval_log="$3"
  local accuracy macro_f1
  accuracy="$(extract_metric 'accuracy:' "$eval_log")"
  macro_f1="$(extract_metric 'macro_f1:' "$eval_log")"
  [[ -n "$accuracy" ]] || accuracy="NA"
  [[ -n "$macro_f1" ]] || macro_f1="NA"
  echo "${SUITE_NAME},${method},${ckpt_epoch},${DOWNSTREAM_DATASETS},${DOWNSTREAM_PROTOCOLS},${DOWNSTREAM_SHOTS},${DOWNSTREAM_SEEDS},${accuracy},${macro_f1},${eval_log}" >> "$SUMMARY_CSV"
}

run_ph_pretrain() {
  local final_ckpt="${PH_OUTPUT_DIR}/checkpoint-$((PRETRAIN_EPOCHS - 1)).pth"
  log "==== PH pretrain ===="
  log "output=$PH_OUTPUT_DIR blr=$PH_BLR batch=$PRETRAIN_BATCH_SIZE epochs=$PRETRAIN_EPOCHS"
  if [[ "$SKIP_PRETRAIN_IF_DONE" == "1" && -f "$final_ckpt" ]]; then
    log "Skip PH pretrain because final checkpoint exists: $final_ckpt"
    return
  fi

  RUN_NAME="$PH_RUN_NAME" \
  CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" \
  GPUS="$GPUS" \
  DATA_PATH="$DATA_PATH" \
  OUTPUT_DIR="$PH_OUTPUT_DIR" \
  BATCH_SIZE="$PRETRAIN_BATCH_SIZE" \
  EPOCHS="$PRETRAIN_EPOCHS" \
  SAVE_FREQ="$PRETRAIN_SAVE_FREQ" \
  BLR="$PH_BLR" \
  WARMUP_EPOCHS="$PRETRAIN_WARMUP_EPOCHS" \
  INIT_CKPT="$PH_INIT_CKPT" \
  INIT_CKPT_SCOPE="$PH_INIT_CKPT_SCOPE" \
  GRAD_LOSS_WEIGHT="$PH_GRAD_LOSS_WEIGHT" \
  LFST_LOSS_WEIGHT="$PH_LFST_LOSS_WEIGHT" \
  TARGET_NORM="$PH_TARGET_NORM" \
  ENCODER_LR_SCALE="$PH_ENCODER_LR_SCALE" \
  SASGT_SCALES="$PH_SASGT_SCALES" \
  bash scripts/pretrain_2xh100.sh 2>&1 | tee "logs/${PH_RUN_NAME}_pretrain.log"
}

run_sarjepa_pretrain() {
  local final_ckpt="${SARJEPA_OUTPUT_DIR}/checkpoint-$((PRETRAIN_EPOCHS - 1)).pth"
  log "==== SAR-JEPA pretrain ===="
  log "output=$SARJEPA_OUTPUT_DIR blr=$SARJEPA_BLR batch=$PRETRAIN_BATCH_SIZE epochs=$PRETRAIN_EPOCHS"
  if [[ "$SKIP_PRETRAIN_IF_DONE" == "1" && -f "$final_ckpt" ]]; then
    log "Skip SAR-JEPA pretrain because final checkpoint exists: $final_ckpt"
    return
  fi

  CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" \
  GPUS="$GPUS" \
  DATA_PATH="$DATA_PATH" \
  OUTPUT_DIR="$SARJEPA_OUTPUT_DIR" \
  BATCH_SIZE="$PRETRAIN_BATCH_SIZE" \
  EPOCHS="$PRETRAIN_EPOCHS" \
  BLR="$SARJEPA_BLR" \
  WARMUP_EPOCHS="$PRETRAIN_WARMUP_EPOCHS" \
  bash scripts/run_sarjepa_pretrain_2xh100.sh 2>&1 | tee "logs/${SARJEPA_RUN_NAME}_pretrain.log"
}

run_eval_one() {
  local method="$1"
  local output_dir="$2"
  local use_sfafm="$3"
  local ckpt_epoch="$4"
  local ckpt_path="${output_dir}/checkpoint-${ckpt_epoch}.pth"
  if [[ ! -f "$ckpt_path" ]]; then
    log "Skip $method eval; checkpoint not found: $ckpt_path"
    return
  fi

  local eval_name="${SUITE_NAME}_${method}_ckpt${ckpt_epoch}_${DOWNSTREAM_DATASETS}_${DOWNSTREAM_PROTOCOLS}_${DOWNSTREAM_SHOTS}shot"
  local eval_log="logs/${eval_name}.log"
  local eval_output="few_shot_classification/finetune/output_${eval_name}"

  log "==== Eval $method checkpoint-$ckpt_epoch ===="
  log "checkpoint=$ckpt_path output=$eval_output use_sfafm=$use_sfafm"

  export CUDA_VISIBLE_DEVICES="$EVAL_CUDA_VISIBLE_DEVICES"
  export CHECKPOINT="$ckpt_path"
  export DATA_ROOT="$DATA_ROOT"
  export OUTPUT_DIR="$eval_output"
  export DATASETS="$DOWNSTREAM_DATASETS"
  export PROTOCOLS="$DOWNSTREAM_PROTOCOLS"
  export SHOTS="$DOWNSTREAM_SHOTS"
  export SEEDS="$DOWNSTREAM_SEEDS"
  export EPOCHS="$DOWNSTREAM_EPOCHS"
  export LR="$DOWNSTREAM_LR"
  export FORCE="$FORCE_EVAL"
  export USE_SFAFM="$use_sfafm"
  if [[ -n "$DOWNSTREAM_BATCH_SIZE" ]]; then
    export BATCH_SIZE="$DOWNSTREAM_BATCH_SIZE"
  else
    unset BATCH_SIZE || true
  fi

  bash scripts/run_sarjepa_fewshot_all.sh 2>&1 | tee "$eval_log"
  append_summary "$method" "$ckpt_epoch" "$eval_log"
}

log "Suite: $SUITE_NAME"
log "Summary CSV: $SUMMARY_CSV"
log "Checkpoint epochs: $CHECKPOINT_EPOCHS"

if [[ "$RUN_PH" == "1" ]]; then
  run_ph_pretrain
fi

if [[ "$RUN_SARJEPA" == "1" ]]; then
  run_sarjepa_pretrain
fi

if [[ "$RUN_EVAL" == "1" ]]; then
  for ckpt_epoch in $CHECKPOINT_EPOCHS; do
    if [[ "$RUN_PH" == "1" ]]; then
      run_eval_one "ph" "$PH_OUTPUT_DIR" "1" "$ckpt_epoch"
    fi
    if [[ "$RUN_SARJEPA" == "1" ]]; then
      run_eval_one "sarjepa" "$SARJEPA_OUTPUT_DIR" "0" "$ckpt_epoch"
    fi
  done
fi

log "All comparisons finished."
log "Summary CSV: $SUMMARY_CSV"
cat "$SUMMARY_CSV"
