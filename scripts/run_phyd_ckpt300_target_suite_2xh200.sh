#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-$ROOT/runs/sarjepa_official_phyd_ft250_bs1024_lfst0p1_image_2xh200/checkpoint-300.pth}"
DATA_PATH="${DATA_PATH:-$ROOT/dataset/modelscope/extracted/Pretraining_dataset}"
SUITE_NAME="${SUITE_NAME:-phyd_ckpt300_target_pilot_30e_bs1088}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
GPUS="${GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-25921}"

# 544 samples per GPU is the largest configuration already verified on 140 GB H200s.
BATCH_SIZE="${BATCH_SIZE:-544}"
ACCUM_ITER="${ACCUM_ITER:-1}"
EPOCHS="${EPOCHS:-30}"
BLR="${BLR:-5e-6}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
NUM_WORKERS="${NUM_WORKERS:-16}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"

if [[ ! -f "$SOURCE_CHECKPOINT" ]]; then
  echo "Missing checkpoint-300 initialization: $SOURCE_CHECKPOINT" >&2
  exit 1
fi
if [[ ! -d "$DATA_PATH" ]]; then
  echo "Missing pretraining data: $DATA_PATH" >&2
  exit 1
fi

mkdir -p "$ROOT/logs"

experiments=(
  "msgt_only:1.0:0.0:raw:30"
  "lfst_raw_c30:0.0:1.0:raw:30"
  "lfst_log_c20:0.0:1.0:log:20"
  "lfst_log_c30:0.0:1.0:log:30"
  "lfst_log_c40:0.0:1.0:log:40"
  "dual_log_c30:1.0:0.1:log:30"
)

echo "PhyD checkpoint-300 target pilot suite"
echo "source checkpoint: $SOURCE_CHECKPOINT"
echo "experiments: ${#experiments[@]}"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "per-GPU batch: $BATCH_SIZE"
echo "effective batch: $((BATCH_SIZE * GPUS * ACCUM_ITER))"
echo "epochs per experiment: $EPOCHS"
echo "base learning rate: $BLR"
echo "SFAFM: disabled"

for spec in "${experiments[@]}"; do
  IFS=: read -r experiment grad_weight lfst_weight lfst_mode cutoff <<< "$spec"
  run_name="${SUITE_NAME}_${experiment}"
  output_dir="$ROOT/runs/$run_name"
  run_log="$ROOT/logs/${run_name}.nohup.log"
  resume=""
  init_checkpoint="$SOURCE_CHECKPOINT"

  if [[ -f "$output_dir/log.txt" ]] \
      && grep -Eq '"epoch"[[:space:]]*:[[:space:]]*'"$((EPOCHS - 1))"'([},])' "$output_dir/log.txt"; then
    echo "Skip completed experiment: $experiment"
    continue
  fi

  if [[ -f "$output_dir/checkpoint-last.pth" ]]; then
    resume="$output_dir/checkpoint-last.pth"
    init_checkpoint=""
    echo "Resume incomplete experiment: $experiment from $resume"
  else
    if [[ -e "$output_dir" ]] && find "$output_dir" -mindepth 1 -print -quit | grep -q .; then
      echo "Output exists without checkpoint-last; refusing to overwrite: $output_dir" >&2
      exit 1
    fi
    echo "Start experiment from checkpoint-300 encoder: $experiment"
  fi

  echo "  grad=$grad_weight lfst=$lfst_weight mode=$lfst_mode cutoff=$cutoff"
  echo "  output=$output_dir"

  env \
    RUN_NAME="$run_name" \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    GPUS="$GPUS" \
    MASTER_PORT="$MASTER_PORT" \
    DATA_PATH="$DATA_PATH" \
    OUTPUT_DIR="$output_dir" \
    LOG_DIR="$output_dir" \
    RESUME="$resume" \
    INIT_CHECKPOINT="$init_checkpoint" \
    INIT_SCOPE=encoder \
    BATCH_SIZE="$BATCH_SIZE" \
    ACCUM_ITER="$ACCUM_ITER" \
    EPOCHS="$EPOCHS" \
    BLR="$BLR" \
    WARMUP_EPOCHS="$WARMUP_EPOCHS" \
    NUM_WORKERS="$NUM_WORKERS" \
    GRAD_LOSS_WEIGHT="$grad_weight" \
    LFST_LOSS_WEIGHT="$lfst_weight" \
    LFST_INPUT_MODE="$lfst_mode" \
    LFST_CUTOFF="$cutoff" \
    TARGET_NORM=image \
    USE_SFAFM=0 \
    CLIP_GRAD="$CLIP_GRAD" \
    SAVE_EVERY_AFTER_EPOCH=0 \
    SAVE_INTERVAL_AFTER_EPOCH="$SAVE_INTERVAL" \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    bash scripts/run_sarjepa_official_phyd_pretrain_2xh100.sh \
    2>&1 | tee -a "$run_log"

  echo "Finished experiment: $experiment"
done

echo "All target pilot experiments finished."
