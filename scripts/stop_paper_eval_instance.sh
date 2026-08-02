#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

profile="${1:?Usage: bash scripts/stop_paper_eval_instance.sh PROFILE}"
pid_file="$ROOT/logs/paper_eval_${profile}.pids"

if [[ ! -f "$pid_file" ]]; then
  echo "No previous PID file: $pid_file"
  exit 0
fi

while IFS=$'\t' read -r pid gpu name; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "Already exited: pid=$pid name=$name"
    continue
  fi

  pgid="$(ps -o pgid= -p "$pid" | tr -d ' ')"
  if [[ "$pgid" =~ ^[0-9]+$ ]] && (( pgid > 1 )); then
    kill -TERM -- "-$pgid" 2>/dev/null || true
    echo "Stopped process group: pgid=$pgid name=$name"
  else
    kill -TERM "$pid" 2>/dev/null || true
    echo "Stopped process: pid=$pid name=$name"
  fi
done < "$pid_file"

echo "Completed outputs are preserved; incomplete outputs will be rerun."
