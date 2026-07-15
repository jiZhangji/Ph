#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PACKAGE_DIR="${PACKAGE_DIR:-$ROOT/hf_release/phyd-sar-release}"
INCLUDE_FULL_CHECKPOINTS="${INCLUDE_FULL_CHECKPOINTS:-0}"
INCLUDE_RAW_LOGS="${INCLUDE_RAW_LOGS:-0}"
INCLUDE_HISTORICAL="${INCLUDE_HISTORICAL:-0}"
OVERWRITE="${OVERWRITE:-0}"

args=(
  --root "$ROOT"
  --package-dir "$PACKAGE_DIR"
)

if [[ "$INCLUDE_FULL_CHECKPOINTS" == "1" ]]; then
  args+=(--include-full-checkpoints)
fi
if [[ "$INCLUDE_RAW_LOGS" == "1" ]]; then
  args+=(--include-raw-logs)
fi
if [[ "$INCLUDE_HISTORICAL" == "1" ]]; then
  args+=(--include-historical)
fi
if [[ "$OVERWRITE" == "1" ]]; then
  args+=(--overwrite)
fi

python scripts/prepare_hf_release.py "${args[@]}"

echo
du -sh "$PACKAGE_DIR"
echo "Package directory: $PACKAGE_DIR"
