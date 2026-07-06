#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASELINE_DIR="${BASELINE_DIR:-$ROOT/baselines/SAR-JEPA}"
SARJEPA_REPO="${SARJEPA_REPO:-https://github.com/waterdisappear/SAR-JEPA.git}"
SARJEPA_CLONE_DEPTH="${SARJEPA_CLONE_DEPTH:-1}"
SARJEPA_CLONE_RETRIES="${SARJEPA_CLONE_RETRIES:-3}"

mkdir -p "$(dirname "$BASELINE_DIR")"

if [[ -d "$BASELINE_DIR/.git" ]]; then
  echo "Updating SAR-JEPA baseline: $BASELINE_DIR"
  git -C "$BASELINE_DIR" pull --ff-only
else
  echo "Cloning SAR-JEPA baseline to: $BASELINE_DIR"
  for attempt in $(seq 1 "$SARJEPA_CLONE_RETRIES"); do
    rm -rf "$BASELINE_DIR"
    if git clone --depth "$SARJEPA_CLONE_DEPTH" "$SARJEPA_REPO" "$BASELINE_DIR"; then
      break
    fi
    if [[ "$attempt" == "$SARJEPA_CLONE_RETRIES" ]]; then
      echo "Failed to clone SAR-JEPA after $SARJEPA_CLONE_RETRIES attempts."
      exit 1
    fi
    echo "Clone failed, retrying in 15 seconds... ($attempt/$SARJEPA_CLONE_RETRIES)"
    sleep 15
  done
fi

echo "SAR-JEPA baseline is ready: $BASELINE_DIR"
