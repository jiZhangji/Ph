#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

profile="${1:-${INSTANCE_PROFILE:-}}"
if [[ -z "$profile" ]]; then
  echo "Usage: bash scripts/launch_paper_eval_balanced.sh {4090|2h100|1h100|2h200}"
  exit 2
fi

if ! command -v flock >/dev/null 2>&1; then
  echo "flock is required to prevent duplicate workers"
  exit 1
fi

case "$profile" in
  4090)
    assignments=("0::0" "0::1" "0::2" "0::3")
    ;;
  2h100)
    assignments=(
      "0:0:14" "0:1:15" "0:2:16" "0:3:17" "0:4:18" "0:5:19" "0::4" "0::5"
      "1:6:20" "1:7:21" "1:8:22" "1:9:23" "1:10:24" "1:11:25" "1::6" "1::7"
    )
    ;;
  1h100)
    assignments=(
      "0:12:26" "0:13:27" "0:14:28" "0:15:29" "0:16:30" "0:17:31" "0::8" "0::9"
    )
    ;;
  2h200)
    assignments=(
      "0:18:32" "0:19:33" "0:20:34" "0:21:35" "0:22:36" "0:23:37"
      "0:24:38" "0:25:39" "0:26:40" "0:27:41" "0::10" "0::11"
      "1:28:42" "1:29:43" "1:30:44" "1:31:45" "1:32:46" "1:33:47"
      "1:34:48" "1:35:49" "1:36:50" "1:37:51" "1::12" "1::13"
    )
    ;;
  *)
    echo "Unknown profile: $profile"
    exit 2
    ;;
esac

WEIGHTS_ROOT="${WEIGHTS_ROOT:-$ROOT/weights/sar-ssl-paper-baseline-weights-v1}"
BASELINE_OUTPUT_ROOT="${BASELINE_OUTPUT_ROOT:-$ROOT/few_shot_classification/finetune/output_paper_baselines_10seeds}"

mkdir -p "$ROOT/logs"
pid_file="$ROOT/logs/paper_balanced_${profile}.pids"
: > "$pid_file"

for assignment in "${assignments[@]}"; do
  IFS=: read -r gpu fine_tune_shard linear_shard <<< "$assignment"
  ft_label="${fine_tune_shard:-none}"
  linear_label="${linear_shard:-none}"
  name="paper_balanced_${profile}_gpu${gpu}_ft${ft_label}_lin${linear_label}"
  log_file="$ROOT/logs/${name}.log"
  lock_file="$ROOT/logs/${name}.lock"

  nohup setsid flock -n "$lock_file" \
    env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      FINE_TUNE_SHARD_ID="$fine_tune_shard" \
      FINE_TUNE_NUM_SHARDS=38 \
      LINEAR_SHARD_ID="$linear_shard" \
      LINEAR_NUM_SHARDS=52 \
      WEIGHTS_ROOT="$WEIGHTS_ROOT" \
      BASELINE_OUTPUT_ROOT="$BASELINE_OUTPUT_ROOT" \
      bash scripts/run_paper_protocol_worker.sh \
    > "$log_file" 2>&1 &

  pid=$!
  printf '%s\t%s\t%s\n' "$pid" "$gpu" "$name" | tee -a "$pid_file"
done

echo "Started ${#assignments[@]} balanced workers for profile=$profile"
echo "PID file: $pid_file"
