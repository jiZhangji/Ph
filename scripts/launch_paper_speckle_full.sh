#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

profile="${1:-${INSTANCE_PROFILE:-}}"
if [[ -z "$profile" ]]; then
  echo "Usage: bash scripts/launch_paper_speckle_full.sh {4090|2h100|1h100|2h200}"
  exit 2
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "flock is required to prevent duplicate workers"
  exit 1
fi

# Format: GPU:protocol:shard. The 4090 receives only frozen-encoder linear
# jobs; high-memory fine-tuning is concentrated on H100/H200 GPUs.
case "$profile" in
  4090)
    assignments=(
      "0:MIM_linear:0" "0:MIM_linear:1"
      "0:MIM_linear:2" "0:MIM_linear:3"
    )
    ;;
  2h100)
    assignments=(
      "0:MIM_finetune:0" "0:MIM_finetune:1"
      "0:MIM_finetune:2" "0:MIM_finetune:3"
      "0:MIM_linear:4" "0:MIM_linear:5"
      "1:MIM_finetune:4" "1:MIM_finetune:5"
      "1:MIM_finetune:6" "1:MIM_finetune:7"
      "1:MIM_linear:6" "1:MIM_linear:7"
    )
    ;;
  1h100)
    assignments=(
      "0:MIM_finetune:8" "0:MIM_finetune:9"
      "0:MIM_finetune:10" "0:MIM_finetune:11"
      "0:MIM_linear:8" "0:MIM_linear:9"
    )
    ;;
  2h200)
    assignments=(
      "0:MIM_finetune:12" "0:MIM_finetune:13"
      "0:MIM_finetune:14" "0:MIM_finetune:15"
      "0:MIM_finetune:16" "0:MIM_finetune:17"
      "0:MIM_finetune:18" "0:MIM_finetune:19"
      "0:MIM_linear:10" "0:MIM_linear:11"
      "0:MIM_linear:12" "0:MIM_linear:13"
      "1:MIM_finetune:20" "1:MIM_finetune:21"
      "1:MIM_finetune:22" "1:MIM_finetune:23"
      "1:MIM_finetune:24" "1:MIM_finetune:25"
      "1:MIM_finetune:26" "1:MIM_finetune:27"
      "1:MIM_linear:14" "1:MIM_linear:15"
      "1:MIM_linear:16" "1:MIM_linear:17"
    )
    ;;
  *)
    echo "Unknown profile: $profile"
    exit 2
    ;;
esac

WEIGHTS_ROOT="${WEIGHTS_ROOT:-$ROOT/weights/sar-ssl-paper-baseline-weights-v1}"
SPECKLE_OUTPUT_ROOT="${SPECKLE_OUTPUT_ROOT:-$ROOT/few_shot_classification/finetune/output_speckle_robustness_10seeds}"
PHYD_CHECKPOINT="${PHYD_CHECKPOINT:-$ROOT/runs/sarjepa_official_phyd_ft250_bs1024_lfst0p1_image_2xh200/checkpoint-300.pth}"
READY_MARKER="$SPECKLE_OUTPUT_ROOT/.phyd_main_lr1e3_ready"

if [[ ! -f "$PHYD_CHECKPOINT" ]]; then
  echo "PhyD-MAE checkpoint not found: $PHYD_CHECKPOINT"
  exit 1
fi
if [[ ! -f "$READY_MARKER" ]]; then
  echo "Output is not prepared for the corrected PhyD-MAE LR."
  echo "Run once: bash scripts/prepare_paper_speckle_full_output.sh"
  exit 1
fi

mkdir -p "$ROOT/logs"
pid_file="$ROOT/logs/paper_speckle_full_${profile}.pids"
: > "$pid_file"

for assignment in "${assignments[@]}"; do
  IFS=: read -r gpu protocol shard_id <<< "$assignment"
  if [[ "$protocol" == "MIM_finetune" ]]; then
    num_shards=28
    short_protocol="ft"
  else
    num_shards=18
    short_protocol="lin"
  fi

  name="paper_speckle_full_${profile}_gpu${gpu}_${short_protocol}${shard_id}"
  log_file="$ROOT/logs/${name}.log"
  lock_file="$ROOT/logs/${name}.lock"

  nohup setsid flock -n "$lock_file" \
    env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      CUDA_MODULE_LOADING=LAZY \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      PYTHONUNBUFFERED=1 \
      OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      PROTOCOL="$protocol" \
      SHARD_ID="$shard_id" \
      NUM_SHARDS="$num_shards" \
      WEIGHTS_ROOT="$WEIGHTS_ROOT" \
      SPECKLE_OUTPUT_ROOT="$SPECKLE_OUTPUT_ROOT" \
      PHYD_CHECKPOINT="$PHYD_CHECKPOINT" \
      bash scripts/run_paper_speckle_protocol_worker.sh \
    > "$log_file" 2>&1 &

  pid=$!
  printf '%s\t%s\t%s\n' "$pid" "$gpu" "$name" | tee -a "$pid_file"
done

echo "Started ${#assignments[@]} full controlled-speckle workers for $profile"
echo "PID file: $pid_file"
echo "Completed matching results are skipped because FORCE=0."
