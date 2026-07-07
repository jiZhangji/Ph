#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN_NAME="${RUN_NAME:-sarjepa_official_phyd_ckpt250_all_20seeds}"
CHECKPOINT="${CHECKPOINT:-$ROOT/runs/sarjepa_official_phyd_2xh100/checkpoint-250.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/few_shot_classification/finetune/output_${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-$ROOT/logs/${RUN_NAME}.log}"
SUMMARY_CSV="${SUMMARY_CSV:-$ROOT/logs/${RUN_NAME}_summary.csv}"

DATASETS="${DATASETS:-MSTAR_SOC New_FUSAR SAR_ACD}"
PROTOCOLS="${PROTOCOLS:-MIM_finetune MIM_linear}"
SHOTS="${SHOTS:-10 20 40}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19}"

EPOCHS="${EPOCHS:-40}"
FINETUNE_LR="${FINETUNE_LR:-1e-4}"
LINEAR_LR="${LINEAR_LR:-1e-3}"
USE_SFAFM="${USE_SFAFM:-0}"
FORCE="${FORCE:-0}"

mkdir -p "$ROOT/logs" "$OUTPUT_DIR"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" | tee -a "$LOG_FILE"
  exit 1
fi

{
  echo "[$(date '+%F %T')] RUN_NAME=$RUN_NAME"
  echo "[$(date '+%F %T')] CHECKPOINT=$CHECKPOINT"
  echo "[$(date '+%F %T')] OUTPUT_DIR=$OUTPUT_DIR"
  echo "[$(date '+%F %T')] DATASETS=$DATASETS"
  echo "[$(date '+%F %T')] PROTOCOLS=$PROTOCOLS"
  echo "[$(date '+%F %T')] SHOTS=$SHOTS"
  echo "[$(date '+%F %T')] SEEDS=$SEEDS"
  echo "[$(date '+%F %T')] EPOCHS=$EPOCHS USE_SFAFM=$USE_SFAFM FORCE=$FORCE"
} | tee -a "$LOG_FILE"

for protocol in $PROTOCOLS; do
  case "$protocol" in
    finetune|MIM_finetune) lr="$FINETUNE_LR" ;;
    linear|MIM_linear) lr="$LINEAR_LR" ;;
    *)
      echo "Unknown protocol: $protocol" | tee -a "$LOG_FILE"
      exit 1
      ;;
  esac

  {
    echo
    echo "[$(date '+%F %T')] ==== Protocol: $protocol lr=$lr ===="
  } | tee -a "$LOG_FILE"

  CHECKPOINT="$CHECKPOINT" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  DATASETS="$DATASETS" \
  PROTOCOLS="$protocol" \
  SHOTS="$SHOTS" \
  SEEDS="$SEEDS" \
  EPOCHS="$EPOCHS" \
  LR="$lr" \
  USE_SFAFM="$USE_SFAFM" \
  FORCE="$FORCE" \
  bash "$ROOT/scripts/run_sarjepa_fewshot_all.sh" 2>&1 | tee -a "$LOG_FILE"
done

python - <<PY | tee -a "$LOG_FILE"
import csv
import re
import statistics
from pathlib import Path

root = Path(r"$ROOT")
output_dir = Path(r"$OUTPUT_DIR")
summary_csv = Path(r"$SUMMARY_CSV")
checkpoint = str(Path(r"$CHECKPOINT"))

rows = []
for log in sorted(output_dir.glob("*/MIM_*/vit_b16_*shots/seed*/log.txt")):
    text = log.read_text(errors="ignore")
    if checkpoint not in text:
        continue
    m = re.search(r"([^/]+)/(?P<protocol>MIM_[^/]+)/vit_b16_(?P<shots>\\d+)shots/seed(?P<seed>\\d+)/log\\.txt$", log.as_posix())
    if not m:
        continue
    parts = log.parts
    dataset = parts[-5]
    protocol = parts[-4]
    shots = int(parts[-3].split("_")[-1].replace("shots", ""))
    seed = int(parts[-2].replace("seed", ""))
    acc = re.findall(r"\\* accuracy:\\s*([0-9.]+)%", text)
    f1 = re.findall(r"\\* macro_f1:\\s*([0-9.]+)%", text)
    if acc:
        rows.append({
            "dataset": dataset,
            "protocol": protocol,
            "shots": shots,
            "seed": seed,
            "accuracy": float(acc[-1]),
            "macro_f1": float(f1[-1]) if f1 else "",
            "log": str(log),
        })

summary_csv.parent.mkdir(parents=True, exist_ok=True)
with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["dataset", "protocol", "shots", "seed", "accuracy", "macro_f1", "log"])
    writer.writeheader()
    writer.writerows(rows)

print()
print(f"Saved per-seed summary: {summary_csv}")
print(f"Parsed runs: {len(rows)}")
print()

groups = {}
for r in rows:
    groups.setdefault((r["dataset"], r["protocol"], r["shots"]), []).append(r)

for key in sorted(groups):
    vals = groups[key]
    accs = [v["accuracy"] for v in vals]
    f1s = [v["macro_f1"] for v in vals if v["macro_f1"] != ""]
    dataset, protocol, shots = key
    line = (
        f"{dataset:9s} {protocol:12s} {shots:2d}-shot "
        f"n={len(vals):2d} acc={statistics.mean(accs):.2f}+-{statistics.pstdev(accs):.2f}"
    )
    if f1s:
        line += f" f1={statistics.mean(f1s):.2f}+-{statistics.pstdev(f1s):.2f}"
    print(line)
PY

echo "[$(date '+%F %T')] Done." | tee -a "$LOG_FILE"
