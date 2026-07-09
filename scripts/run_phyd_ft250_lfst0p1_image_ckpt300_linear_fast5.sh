#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CHECKPOINT="${CHECKPOINT:-$ROOT/runs/sarjepa_official_phyd_ft250_bs1024_lfst0p1_image_2xh200/checkpoint-300.pth}"
BASE_OUTPUT="${BASE_OUTPUT:-$ROOT/few_shot_classification/finetune/output_phyd_ft250_lfst0p1_image_ckpt300_linear_fast5}"
SEEDS="${SEEDS:-0 1 2 3 4}"
SHOTS="${SHOTS:-10 20 40}"
EPOCHS="${EPOCHS:-40}"
LR="${LR:-1e-3}"
USE_SFAFM="${USE_SFAFM:-0}"
FORCE="${FORCE:-0}"

mkdir -p logs

run_one() {
  local gpu="$1"
  local dataset="$2"
  local tag="$3"

  CUDA_VISIBLE_DEVICES="$gpu" \
  CHECKPOINT="$CHECKPOINT" \
  OUTPUT_DIR="$BASE_OUTPUT" \
  DATASETS="$dataset" \
  PROTOCOLS=MIM_linear \
  SHOTS="$SHOTS" \
  SEEDS="$SEEDS" \
  EPOCHS="$EPOCHS" \
  LR="$LR" \
  USE_SFAFM="$USE_SFAFM" \
  FORCE="$FORCE" \
  bash scripts/run_sarjepa_fewshot_all.sh \
    >> "logs/phyd_ft250_lfst0p1_image_ckpt300_linear_fast5_${tag}.nohup.log" 2>&1
}

run_one 0 MSTAR_SOC mstar &
run_one 1 New_FUSAR fusar &
run_one 2 SAR_ACD saracd &

wait
echo "All linear fast5 jobs finished."
