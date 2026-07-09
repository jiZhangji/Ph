#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASELINE_DIR="${BASELINE_DIR:-$ROOT/baselines/SAR-JEPA}"
DATA_PATH="${DATA_PATH:-$ROOT/dataset/modelscope/extracted/Pretraining_dataset}"
CLS_ROOT="${CLS_ROOT:-$ROOT/dataset/modelscope/extracted/classification_dataset/few_shot_classification}"
OFFICIAL_REPO="${OFFICIAL_REPO:-https://github.com/waterdisappear/SAR-JEPA.git}"

echo "========== SAR-JEPA Reproduction Audit =========="
echo "ROOT=$ROOT"
echo "BASELINE_DIR=$BASELINE_DIR"
echo "DATA_PATH=$DATA_PATH"
echo "CLS_ROOT=$CLS_ROOT"
echo

echo "========== 1. Git and Script State =========="
if [[ -d "$BASELINE_DIR/.git" ]]; then
  echo "[baseline git]"
  git -C "$BASELINE_DIR" remote -v || true
  git -C "$BASELINE_DIR" rev-parse HEAD || true
  git -C "$BASELINE_DIR" status --short || true
else
  echo "Missing baseline git repo: $BASELINE_DIR"
fi
echo

echo "[Ph wrapper relevant defaults]"
grep -nE 'BATCH_SIZE=|EPOCHS=|BLR=|WARMUP_EPOCHS=|MASK_RATIO=|WINDOW_SIZE=|NUM_WINDOW=|cp "\$ROOT/Pretraining/util/datasets.py"' \
  "$ROOT/scripts/run_sarjepa_pretrain_2xh100.sh" || true
echo

echo "========== 2. Official File Drift =========="
tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

if command -v git >/dev/null 2>&1; then
  echo "Cloning official SAR-JEPA to temporary directory for diff..."
  if git clone --depth 1 "$OFFICIAL_REPO" "$tmp_dir/SAR-JEPA-official" >/dev/null 2>&1; then
    for rel in \
      Pretraining/main_pretrain.py \
      Pretraining/models_lomar.py \
      Pretraining/util/datasets.py \
      Pretraining/util/misc.py
    do
      echo
      echo "--- diff summary: $rel ---"
      if [[ -f "$BASELINE_DIR/$rel" && -f "$tmp_dir/SAR-JEPA-official/$rel" ]]; then
        diff -u "$tmp_dir/SAR-JEPA-official/$rel" "$BASELINE_DIR/$rel" | sed -n '1,120p' || true
      else
        echo "missing one side: official=$tmp_dir/SAR-JEPA-official/$rel baseline=$BASELINE_DIR/$rel"
      fi
    done
  else
    echo "Could not clone official repo. Network may be unavailable; skip diff."
  fi
fi
echo

echo "========== 3. Pretraining Dataset Count =========="
python - "$DATA_PATH" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
print(f"exists={root.exists()} path={root}")
if root.exists():
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    print(f"image_count={len(files)}")
    top = {}
    for p in files:
        try:
            key = p.relative_to(root).parts[0]
        except Exception:
            key = "."
        top[key] = top.get(key, 0) + 1
    for key, value in sorted(top.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {key}: {value}")
PY
echo

echo "========== 4. Pretraining Transform / Augmentation =========="
for f in \
  "$BASELINE_DIR/Pretraining/main_pretrain.py" \
  "$ROOT/Pretraining_sarjepa_official_phyd/main_pretrain.py" \
  "$ROOT/Pretraining/main_pretrain.py"
do
  echo
  echo "--- $f ---"
  if [[ -f "$f" ]]; then
    grep -nE 'RandomResizedCrop|RandomHorizontalFlip|ColorJitter|load_data|batch_size|epochs|blr|warmup_epochs' "$f" | sed -n '1,120p' || true
  else
    echo "missing"
  fi
done
echo

echo "========== 5. Downstream Config =========="
for f in \
  "$ROOT/few_shot_classification/finetune/configs/trainers/MIM_finetune/vit_b16.yaml" \
  "$ROOT/few_shot_classification/finetune/configs/trainers/MIM_linear/vit_b16.yaml" \
  "$ROOT/few_shot_classification/finetune/configs/datasets/MSTAR_SOC.yaml"
do
  echo
  echo "--- $f ---"
  if [[ -f "$f" ]]; then
    sed -n '1,120p' "$f"
  else
    echo "missing"
  fi
done
echo

echo "========== 6. Downstream Dataset Count =========="
python - "$CLS_ROOT" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
print(f"exists={root.exists()} path={root}")
if root.exists():
    for ds in ["MSTAR_SOC", "mstar", "MSTAR", "New_FUSAR", "fusar", "fusar_ship", "SAR_ACD", "sar_acd"]:
        p = root / ds
        if not p.exists():
            continue
        files = [x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in exts]
        print(f"{ds}: image_count={len(files)} path={p}")
        split = {}
        for x in files:
            rel = x.relative_to(p).parts
            key = rel[0] if rel else "."
            split[key] = split.get(key, 0) + 1
        for k, v in sorted(split.items()):
            print(f"  {k}: {v}")
PY
echo

echo "========== 7. Existing Official SAR-JEPA Results =========="
python - "$ROOT" <<'PY'
from pathlib import Path
import re
import sys
root = Path(sys.argv[1])
targets = [
    "sarjepa_pretrain_2xh100/checkpoint-200.pth",
    "sarjepa_pretrain_2xh100/checkpoint-299.pth",
]
for target in targets:
    print()
    print(f"target={target}")
    rows = []
    for log in root.glob("**/MSTAR_SOC/MIM_finetune/vit_b16_10shots/seed*/log.txt"):
        text = log.read_text(errors="ignore")
        if target not in text:
            continue
        acc = re.findall(r"\* accuracy:\s*([0-9.]+)%", text)
        f1 = re.findall(r"\* macro_f1:\s*([0-9.]+)%", text)
        seed = re.search(r"/seed(\d+)/", log.as_posix())
        if acc and seed:
            rows.append((int(seed.group(1)), float(acc[-1]), float(f1[-1]) if f1 else None, log))
    for seed, acc, f1, log in sorted(rows):
        print(f"seed={seed} acc={acc:.2f} f1={f1 if f1 is not None else 'NA'} log={log.relative_to(root)}")
    print(f"n={len(rows)}")
PY

echo
echo "Audit finished."
