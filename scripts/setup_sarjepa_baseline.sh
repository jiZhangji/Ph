#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASELINE_DIR="${BASELINE_DIR:-$ROOT/baselines/SAR-JEPA}"
SARJEPA_REPO="${SARJEPA_REPO:-https://github.com/waterdisappear/SAR-JEPA.git}"

mkdir -p "$(dirname "$BASELINE_DIR")"

if [[ -d "$BASELINE_DIR/.git" ]]; then
  echo "Updating SAR-JEPA baseline: $BASELINE_DIR"
  git -C "$BASELINE_DIR" pull --ff-only
else
  echo "Cloning SAR-JEPA baseline to: $BASELINE_DIR"
  git clone "$SARJEPA_REPO" "$BASELINE_DIR"
fi

echo "SAR-JEPA baseline is ready: $BASELINE_DIR"
