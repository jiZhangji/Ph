#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

HF_REPO_ID="${HF_REPO_ID:-shimiandeshu/phyd-mae-paper-non-speckle-results-v1}"
HF_PRIVATE="${HF_PRIVATE:-1}"
PACKAGE_DIR="${PACKAGE_DIR:-$ROOT/hf_release/phyd-mae-paper-non-speckle-results-v1}"

if ! command -v hf >/dev/null 2>&1; then
  echo "Missing Hugging Face CLI. Install: python -m pip install -U huggingface_hub" >&2
  exit 1
fi
if ! hf auth whoami >/dev/null 2>&1; then
  echo "Hugging Face authentication is missing. Run: hf auth login" >&2
  exit 1
fi

python scripts/export_non_speckle_results.py \
  --root "$ROOT" \
  --output-dir "$PACKAGE_DIR"

export HF_REPO_ID HF_PRIVATE
python - <<'PY'
import os
from huggingface_hub import HfApi

private = os.environ["HF_PRIVATE"].lower() not in {"0", "false", "no"}
repo_id = os.environ["HF_REPO_ID"]
url = HfApi().create_repo(
    repo_id=repo_id,
    repo_type="dataset",
    private=private,
    exist_ok=True,
)
print(f"Hugging Face dataset repository: {url}")
print(f"Private: {private}")
PY

export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
echo "Uploading $PACKAGE_DIR"
hf upload "$HF_REPO_ID" \
  --repo-type=dataset \
  "$PACKAGE_DIR" . \
  --commit-message "Upload complete non-speckle per-seed results and statistics"

echo "Upload complete: https://huggingface.co/datasets/$HF_REPO_ID"
