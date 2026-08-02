#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

profile="${1:?Usage: bash scripts/collect_eval_telemetry.sh PROFILE [INTERVAL_SECONDS]}"
interval="${2:-30}"
telemetry_dir="$ROOT/logs/cluster_telemetry"
output="$telemetry_dir/${profile}.txt"
tmp="$telemetry_dir/.${profile}.$$.tmp"
pid_file="$ROOT/logs/paper_eval_${profile}.pids"

mkdir -p "$telemetry_dir"
trap 'rm -f "$tmp"' EXIT

while true; do
  running=0
  exited=0
  {
    echo "profile=$profile"
    echo "timestamp=$(date '+%Y-%m-%d %H:%M:%S %z')"
    echo "hostname=$(hostname)"
    echo "collector_pid=$$"

    echo "[gpus]"
    nvidia-smi \
      --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
      --format=csv,noheader,nounits || true

    echo "[workers]"
    if [[ -f "$pid_file" ]]; then
      while IFS=$'\t' read -r pid gpu name; do
        [[ -n "$pid" ]] || continue
        if kill -0 "$pid" 2>/dev/null; then
          state="RUNNING"
          running=$((running + 1))
        else
          state="EXITED"
          exited=$((exited + 1))
        fi
        printf '%s\tpid=%s\tgpu=%s\t%s\n' "$state" "$pid" "$gpu" "$name"
      done < "$pid_file"
    else
      echo "PID_FILE_MISSING $pid_file"
    fi
    echo "worker_summary running=$running exited=$exited"

    echo "[compute_processes]"
    nvidia-smi \
      --query-compute-apps=gpu_uuid,pid,used_memory \
      --format=csv,noheader,nounits || true
  } > "$tmp"

  mv -f "$tmp" "$output"
  sleep "$interval"
done
