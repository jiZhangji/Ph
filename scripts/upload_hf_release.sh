#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

: "${HF_REPO_ID:?Set HF_REPO_ID, for example your-name/phyd-sar-release}"

PACKAGE_DIR="${PACKAGE_DIR:-$ROOT/hf_release/phyd-sar-release}"
HF_PRIVATE="${HF_PRIVATE:-1}"
HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

if [[ ! -f "$PACKAGE_DIR/manifest.json" || ! -f "$PACKAGE_DIR/README.md" ]]; then
  echo "Invalid or incomplete package directory: $PACKAGE_DIR" >&2
  echo "Run scripts/package_hf_release.sh first." >&2
  exit 1
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "Missing Hugging Face CLI. Install it with:" >&2
  echo "  python -m pip install -U huggingface_hub" >&2
  exit 1
fi

if ! hf auth whoami >/dev/null 2>&1; then
  echo "Not authenticated with Hugging Face. Run: hf auth login" >&2
  exit 1
fi

export HF_REPO_ID HF_PRIVATE HF_XET_HIGH_PERFORMANCE
python - <<'PY'
import os
from huggingface_hub import HfApi

private = os.environ["HF_PRIVATE"].lower() not in {"0", "false", "no"}
repo_id = os.environ["HF_REPO_ID"]
url = HfApi().create_repo(
    repo_id=repo_id,
    repo_type="model",
    private=private,
    exist_ok=True,
)
print(f"Hugging Face repository: {url}")
print(f"Private: {private}")
PY

echo "Uploading $PACKAGE_DIR to $HF_REPO_ID"
echo "This command is resumable; rerun it if the connection is interrupted."
hf upload "$HF_REPO_ID" --repo-type=model "$PACKAGE_DIR" .

echo "Upload complete: https://huggingface.co/$HF_REPO_ID"
