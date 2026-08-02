#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

profile="${1:-${INSTANCE_PROFILE:-}}"
RUN_SPECKLE="${RUN_SPECKLE:-0}"
if [[ -z "$profile" ]]; then
  echo "Usage: bash scripts/launch_paper_eval_instance.sh {4090|2h100|1h100|2h200}"
  exit 2
fi

if ! command -v flock >/dev/null 2>&1; then
  echo "flock is required to prevent duplicate workers"
  exit 1
fi

case "$profile" in
  4090)
    assignments=("0:0:0" "0:1:")
    ;;
  2h100)
    assignments=(
      "0:2:1" "0:3:" "0:4:" "0:5:"
      "1:6:2" "1:7:" "1:8:" "1:9:"
    )
    ;;
  1h100)
    assignments=("0:10:3" "0:11:" "0:12:" "0:13:")
    ;;
  2h200)
    assignments=(
      "0:14:4" "0:15:5" "0:16:6" "0:17:" "0:18:" "0:19:"
      "1:20:7" "1:21:8" "1:22:9" "1:23:" "1:24:" "1:25:"
    )
    ;;
  *)
    echo "Unknown profile: $profile"
    exit 2
    ;;
esac

WEIGHTS_ROOT="${WEIGHTS_ROOT:-$ROOT/weights/sar-ssl-paper-baseline-weights-v1}"
BASELINE_OUTPUT_ROOT="${BASELINE_OUTPUT_ROOT:-$ROOT/few_shot_classification/finetune/output_paper_baselines_10seeds}"
SPECKLE_OUTPUT_ROOT="${SPECKLE_OUTPUT_ROOT:-$ROOT/few_shot_classification/finetune/output_speckle_robustness_10seeds}"
PHYD_CHECKPOINT="${PHYD_CHECKPOINT:-$ROOT/runs/sarjepa_official_phyd_ft250_bs1024_lfst0p1_image_2xh200/checkpoint-300.pth}"

mkdir -p "$ROOT/logs"
pid_file="$ROOT/logs/paper_eval_${profile}.pids"
: > "$pid_file"

for assignment in "${assignments[@]}"; do
  IFS=: read -r gpu baseline_shard speckle_shard <<< "$assignment"
  if [[ "$RUN_SPECKLE" != "1" ]]; then
    speckle_shard=""
  fi
  speckle_label="${speckle_shard:-none}"
  name="paper_eval_${profile}_gpu${gpu}_b${baseline_shard}_s${speckle_label}"
  log_file="$ROOT/logs/${name}.log"
  lock_file="$ROOT/logs/${name}.lock"

  nohup setsid flock -n "$lock_file" \
    env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      OMP_NUM_THREADS=2 \
      MKL_NUM_THREADS=2 \
      BASELINE_SHARD_ID="$baseline_shard" \
      BASELINE_NUM_SHARDS=26 \
      SPECKLE_SHARD_ID="$speckle_shard" \
      SPECKLE_NUM_SHARDS=10 \
      WEIGHTS_ROOT="$WEIGHTS_ROOT" \
      BASELINE_OUTPUT_ROOT="$BASELINE_OUTPUT_ROOT" \
      SPECKLE_OUTPUT_ROOT="$SPECKLE_OUTPUT_ROOT" \
      PHYD_CHECKPOINT="$PHYD_CHECKPOINT" \
      bash scripts/run_paper_eval_worker.sh \
    > "$log_file" 2>&1 &

  pid=$!
  printf '%s\t%s\t%s\n' "$pid" "$gpu" "$name" | tee -a "$pid_file"
done

echo "Started ${#assignments[@]} workers for profile=$profile"
echo "Controlled speckle enabled: $RUN_SPECKLE"
echo "PID file: $pid_file"
