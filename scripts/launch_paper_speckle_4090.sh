#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

WEIGHTS_ROOT="${WEIGHTS_ROOT:-$ROOT/weights/sar-ssl-paper-baseline-weights-v1}"
OUTPUT_ROOT="${SPECKLE_OUTPUT_ROOT:-$ROOT/few_shot_classification/finetune/output_speckle_robustness_10seeds}"
PHYD_CHECKPOINT="${PHYD_CHECKPOINT:-$ROOT/runs/sarjepa_official_phyd_ft250_bs1024_lfst0p1_image_2xh200/checkpoint-300.pth}"
NUM_WORKERS="${NUM_WORKERS:-4}"

if ! command -v flock >/dev/null 2>&1; then
  echo "flock is required to prevent duplicate workers"
  exit 1
fi
if [[ ! "$NUM_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_WORKERS must be a positive integer, got: $NUM_WORKERS"
  exit 2
fi
if [[ ! -f "$PHYD_CHECKPOINT" ]]; then
  echo "PhyD-MAE checkpoint not found: $PHYD_CHECKPOINT"
  exit 1
fi

mkdir -p "$ROOT/logs"
pid_file="$ROOT/logs/paper_speckle_4090.pids"
: > "$pid_file"

for ((shard = 0; shard < NUM_WORKERS; shard++)); do
  name="paper_speckle_4090_gpu0_shard${shard}"
  log_file="$ROOT/logs/${name}.log"
  lock_file="$ROOT/logs/${name}.lock"

  nohup setsid flock -n "$lock_file" \
    env \
      CUDA_VISIBLE_DEVICES=0 \
      CUDA_MODULE_LOADING=LAZY \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      WEIGHTS_ROOT="$WEIGHTS_ROOT" \
      PHYD_CHECKPOINT="$PHYD_CHECKPOINT" \
      OUTPUT_ROOT="$OUTPUT_ROOT" \
      METHODS="mae lomar fg_mae i_jepa sar_jepa phyd_mae" \
      DATASETS="MSTAR_SOC New_FUSAR SAR_ACD" \
      PROTOCOLS="MIM_linear" \
      SHOTS="20" \
      SEEDS="0 1 2 3 4 5 6 7 8 9" \
      SPECKLE_LOOKS="clean 8 4 2 1" \
      NOISE_SEEDS="0 1 2" \
      LR=1e-4 \
      EPOCHS=40 \
      BATCH_SIZE=50 \
      FORCE=0 \
      NUM_SHARDS="$NUM_WORKERS" \
      SHARD_ID="$shard" \
      SUMMARIZE=0 \
      bash scripts/run_speckle_robustness.sh \
    > "$log_file" 2>&1 &

  pid=$!
  printf '%s\t%s\t%s\n' "$pid" 0 "$name" | tee -a "$pid_file"
done

echo "Started $NUM_WORKERS controlled-speckle workers on GPU 0"
echo "Datasets: MSTAR_SOC New_FUSAR SAR_ACD"
echo "Seeds: 0 1 2 3 4 5 6 7 8 9"
echo "PID file: $pid_file"
echo "Completed results are skipped because FORCE=0."
