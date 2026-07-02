#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "========== Current Directory =========="
pwd
echo

check_file() {
  if [ -f "$1" ]; then
    echo "[OK] $1"
  else
    echo "[MISSING] $1"
  fi
}

check_dir() {
  if [ -d "$1" ]; then
    echo "[OK] $1"
  else
    echo "[MISSING] $1"
  fi
}

echo "========== Code Files =========="
check_file "README.md"
check_file "requirements.txt"
check_file "environment.yml"
check_dir "Pretraining"
check_file "Pretraining/main_pretrain.py"
check_file "Pretraining/util/datasets.py"
check_dir "scripts"
check_file "scripts/download_modelscope_data.py"
check_file "scripts/pretrain_4xa100.sh"
check_file "scripts/pretrain_2xh100.sh"
echo

echo "========== Dataset Zip Files =========="
check_file "dataset/modelscope/zips/Pretraining_dataset.zip"
check_file "dataset/modelscope/zips/classification_dataset.zip"
echo

echo "========== Zip Sizes =========="
if [ -f "dataset/modelscope/zips/Pretraining_dataset.zip" ]; then
  ls -lh "dataset/modelscope/zips/Pretraining_dataset.zip"
fi
if [ -f "dataset/modelscope/zips/classification_dataset.zip" ]; then
  ls -lh "dataset/modelscope/zips/classification_dataset.zip"
fi
echo

echo "========== Extracted Dataset Directories =========="
check_dir "dataset/modelscope/extracted/Pretraining_dataset"
check_dir "dataset/modelscope/extracted/classification_dataset"
echo

echo "========== Zip Integrity =========="
python - <<'PY'
from pathlib import Path
from zipfile import BadZipFile, ZipFile

for name in (
    "dataset/modelscope/zips/Pretraining_dataset.zip",
    "dataset/modelscope/zips/classification_dataset.zip",
):
    path = Path(name)
    if not path.exists():
        continue
    try:
        with ZipFile(path) as archive:
            bad = archive.testzip()
        if bad is None:
            print(f"[OK] {name}")
        else:
            print(f"[BAD] {name}: first bad file is {bad}")
    except BadZipFile:
        print(f"[BAD] {name}: not a valid zip file")
PY
echo

echo "========== Image Counts =========="
count_images() {
  if [ -d "$1" ]; then
    find "$1" -type f \( \
      -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o \
      -iname "*.tif" -o -iname "*.tiff" -o -iname "*.bmp" \
    \) | wc -l
  else
    echo 0
  fi
}

echo "Pretraining images: $(count_images dataset/modelscope/extracted/Pretraining_dataset)"
echo "Classification images: $(count_images dataset/modelscope/extracted/classification_dataset)"
echo

echo "========== Python Imports =========="
python - <<'PY'
import sys

print("Python:", sys.version.replace("\n", " "))
for module in ("torch", "torchvision", "PIL"):
    try:
        imported = __import__(module)
        version = getattr(imported, "__version__", "ok")
        print(f"[OK] {module}: {version}")
    except Exception as exc:
        print(f"[WARN] {module} import failed: {exc}")
PY
echo

echo "========== Done =========="
