#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REPRO_COMMIT="${REPRO_COMMIT:-1b86fee}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-0}"
DOWNSTREAM_LR="${DOWNSTREAM_LR:-1e-4}"
DOWNSTREAM_EPOCHS="${DOWNSTREAM_EPOCHS:-40}"
DOWNSTREAM_BATCH_SIZE="${DOWNSTREAM_BATCH_SIZE:-50}"
FORCE="${FORCE:-0}"

DATA_ROOT="${DATA_ROOT:-$ROOT/dataset/modelscope/extracted/classification_dataset/few_shot_classification}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/few_shot_classification/finetune/output_sfafm7_ab_mstar_repro_lr1e4}"

REFERENCE_CHECKPOINT="${REFERENCE_CHECKPOINT:-$ROOT/runs/phyd_sfafm7_every2end_from_best300_g1_lfst0p1_image_bs768_300e_2xh200/checkpoint-20.pth}"
RUN_A="${RUN_A:-phyd_sfafm7_from_ckpt20_stable_lfst0p1_40e}"
RUN_B="${RUN_B:-phyd_sfafm7_from_ckpt20_stable_lfst0p05_40e}"

critical_files=(
  few_shot_classification/finetune/train.py
  few_shot_classification/finetune/configs/trainers/MIM_finetune/vit_b16.yaml
  few_shot_classification/finetune/trainers/mim_sar_encoder.py
  scripts/run_sarjepa_fewshot_all.sh
  Pretraining_sarjepa_official_phyd/models_lomar.py
)

if ! git cat-file -e "${REPRO_COMMIT}^{commit}" 2>/dev/null; then
  echo "Missing reproduction commit: $REPRO_COMMIT" >&2
  exit 1
fi

if ! git diff --quiet "$REPRO_COMMIT" -- "${critical_files[@]}"; then
  echo "Downstream reproduction files differ from $REPRO_COMMIT:" >&2
  git diff --stat "$REPRO_COMMIT" -- "${critical_files[@]}" >&2
  exit 1
fi

if [[ ! -f "$REFERENCE_CHECKPOINT" ]]; then
  echo "Missing reference checkpoint: $REFERENCE_CHECKPOINT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

echo "Historical-compatible SFAFM7 downstream reproduction"
echo "code commit baseline: $REPRO_COMMIT"
echo "GPU: $GPU"
echo "seeds: $SEEDS"
echo "learning rate: $DOWNSTREAM_LR"
echo "epochs: $DOWNSTREAM_EPOCHS"
echo "batch size: $DOWNSTREAM_BATCH_SIZE"
echo "output root: $OUTPUT_ROOT"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader 2>/dev/null || true

evaluate_checkpoint() {
  local label="$1"
  local checkpoint="$2"
  local output_dir="$OUTPUT_ROOT/$label"

  if [[ ! -f "$checkpoint" ]]; then
    echo "Skip missing checkpoint: $checkpoint"
    return
  fi

  echo
  echo "============================================================"
  echo "label: $label"
  echo "checkpoint: $checkpoint"
  echo "started: $(date '+%F %T')"
  echo "============================================================"

  env \
    -u LR \
    -u EPOCHS \
    -u BATCH_SIZE \
    -u MIM_CKPT \
    -u MIM_USE_SFAFM \
    -u MIM_SFAFM_LAYOUT \
    -u MIM_FEATURE_POOL \
    CUDA_VISIBLE_DEVICES="$GPU" \
    DATA_ROOT="$DATA_ROOT" \
    CHECKPOINT="$checkpoint" \
    OUTPUT_DIR="$output_dir" \
    DATASETS="MSTAR_SOC" \
    PROTOCOLS="MIM_finetune" \
    SHOTS="10" \
    SEEDS="$SEEDS" \
    LR="$DOWNSTREAM_LR" \
    EPOCHS="$DOWNSTREAM_EPOCHS" \
    BATCH_SIZE="$DOWNSTREAM_BATCH_SIZE" \
    USE_SFAFM=1 \
    SFAFM_LAYOUT=every2_end \
    FEATURE_POOL=cls \
    FORCE="$FORCE" \
    bash scripts/run_sarjepa_fewshot_all.sh
}

evaluate_checkpoint "reference_checkpoint-20" "$REFERENCE_CHECKPOINT"

for spec in "A:$RUN_A" "B:$RUN_B"; do
  IFS=: read -r tag run_name <<< "$spec"
  run_dir="$ROOT/runs/$run_name"

  if [[ ! -d "$run_dir" ]]; then
    echo "Skip missing run directory: $run_dir"
    continue
  fi

  mapfile -t checkpoints < <(
    find "$run_dir" -maxdepth 1 -type f -name 'checkpoint-[0-9]*.pth' | sort -V
  )
  if [[ -f "$run_dir/checkpoint-last.pth" ]]; then
    checkpoints+=("$run_dir/checkpoint-last.pth")
  fi

  for checkpoint in "${checkpoints[@]}"; do
    checkpoint_name="$(basename "$checkpoint" .pth)"
    evaluate_checkpoint "${tag}_${checkpoint_name}" "$checkpoint"
  done
done

OUTPUT_ROOT="$OUTPUT_ROOT" python - <<'PY'
import os
import re
from pathlib import Path

root = Path(os.environ["OUTPUT_ROOT"])
rows = []
for log in root.glob(
    "*/MSTAR_SOC/MIM_finetune/vit_b16_10shots/seed*/log.txt"
):
    text = log.read_text(errors="ignore")
    accuracy = re.findall(r"\* accuracy:\s*([0-9.]+)%", text)
    macro_f1 = re.findall(r"\* macro_f1:\s*([0-9.]+)%", text)
    if accuracy:
        rows.append((
            log.parents[4].name,
            log.parent.name,
            float(accuracy[-1]),
            float(macro_f1[-1]) if macro_f1 else None,
        ))

print("\n===== SFAFM7 reference/A/B reproduction results =====")
print(f"{'checkpoint':>28} {'seed':>8} {'accuracy':>10} {'macro_f1':>10}")
for label, seed, accuracy, macro_f1 in sorted(rows):
    f1_text = f"{macro_f1:.2f}%" if macro_f1 is not None else "N/A"
    print(f"{label:>28} {seed:>8} {accuracy:9.2f}% {f1_text:>10}")
PY

echo "All historical-compatible reference/A/B evaluations finished."
