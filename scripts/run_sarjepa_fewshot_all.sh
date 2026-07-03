#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FINETUNE_DIR="$ROOT/few_shot_classification/finetune"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-$ROOT/dataset/modelscope/extracted/classification_dataset/few_shot_classification}"
if [[ ! -d "$DATA_ROOT" ]]; then
  DATA_ROOT="${DATA_ROOT_FALLBACK:-$ROOT/dataset/modelscope/extracted/classification_dataset}"
fi
if [[ -d "$DATA_ROOT" ]]; then
  DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"
fi

CHECKPOINT="${CHECKPOINT:-$ROOT/runs/pretrain_2xh100_stable_full_bs512/checkpoint-299.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-$FINETUNE_DIR/output_sarjepa}"
DATASETS="${DATASETS:-MSTAR_SOC New_FUSAR SAR_ACD}"
PROTOCOLS="${PROTOCOLS:-MIM_finetune MIM_linear}"
SHOTS="${SHOTS:-10 20 40}"
SEEDS="${SEEDS:-}"
CFG="${CFG:-vit_b16}"
FORCE="${FORCE:-0}"

if [[ ! -d "$FINETUNE_DIR" ]]; then
  echo "Missing $FINETUNE_DIR"
  exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT"
  exit 1
fi

ensure_dassl() {
  if python - <<'PY' >/dev/null 2>&1
import dassl
PY
  then
    return
  fi

  local zip_path="$ROOT/few_shot_classification/Dassl.pytorch.zip"
  local dassl_dir="$ROOT/few_shot_classification/Dassl.pytorch"
  if [[ -d "$dassl_dir/dassl" ]]; then
    export PYTHONPATH="$dassl_dir:${PYTHONPATH:-}"
    return
  fi

  if [[ -f "$zip_path" ]]; then
    unzip -q -o "$zip_path" -d "$ROOT/few_shot_classification"
    export PYTHONPATH="$dassl_dir:${PYTHONPATH:-}"
    if pip install -e "$dassl_dir" --no-build-isolation --no-deps; then
      return
    fi

    echo "Editable Dassl install failed; falling back to PYTHONPATH."
    python - <<'PY'
import dassl
PY
    return
  fi

  echo "Dassl is not installed and $zip_path is missing."
  echo "Install Dassl first, then rerun this script."
  exit 1
}

resolve_dataset_name() {
  case "$1" in
    mstar|MSTAR|MSTAR_SOC) echo "MSTAR_SOC" ;;
    fusar|fusar_ship|New_FUSAR) echo "New_FUSAR" ;;
    sar_acd|SAR_ACD) echo "SAR_ACD" ;;
    *) echo "$1" ;;
  esac
}

resolve_trainer_name() {
  case "$1" in
    finetune|MIM_finetune) echo "MIM_finetune" ;;
    linear|MIM_linear) echo "MIM_linear" ;;
    *) echo "$1" ;;
  esac
}

link_dataset() {
  local name="$1"
  local target=""
  local aliases=("$name")
  case "$name" in
    MSTAR_SOC) aliases+=("mstar" "MSTAR") ;;
    New_FUSAR) aliases+=("fusar_ship" "fusar" "FUSAR") ;;
    SAR_ACD) aliases+=("sar_acd" "SAR-ACD") ;;
  esac

  for alias in "${aliases[@]}"; do
    for candidate in \
      "$DATA_ROOT/$alias" \
      "$DATA_ROOT/data/$alias" \
      "$DATA_ROOT/finetune/data/$alias" \
      "$DATA_ROOT/few_shot_classification/$alias" \
      "$DATA_ROOT/few_shot_classification/data/$alias" \
      "$DATA_ROOT/few_shot_classification/finetune/data/$alias"
    do
      if [[ -d "$candidate" ]]; then
        target="$(cd "$candidate" && pwd)"
        break 2
      fi
    done
  done

  if [[ -z "$target" ]]; then
    echo "Dataset $name not found under $DATA_ROOT"
    echo "Available directories:"
    find "$DATA_ROOT" -maxdepth 4 -type d | sed -n '1,120p'
    exit 1
  fi

  mkdir -p "$FINETUNE_DIR/data"
  ln -sfn "$target" "$FINETUNE_DIR/data/$name"
}

ensure_dassl

CHECKPOINT="$(cd "$(dirname "$CHECKPOINT")" && pwd)/$(basename "$CHECKPOINT")"
OUTPUT_DIR="$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)"
export MIM_CKPT="$CHECKPOINT"

cd "$FINETUNE_DIR"

for raw_dataset in $DATASETS; do
  dataset="$(resolve_dataset_name "$raw_dataset")"
  link_dataset "$dataset"

  for raw_protocol in $PROTOCOLS; do
    trainer="$(resolve_trainer_name "$raw_protocol")"
    if [[ -n "$SEEDS" ]]; then
      run_seeds="$SEEDS"
    elif [[ "$trainer" == "MIM_linear" ]]; then
      run_seeds="0 1 2 3 4 5"
    else
      run_seeds="0 1 2 3 4"
    fi

    for shots in $SHOTS; do
      for seed in $run_seeds; do
        run_dir="$OUTPUT_DIR/${dataset}/${trainer}/${CFG}_${shots}shots/seed${seed}"
        if [[ -d "$run_dir" && "$FORCE" != "1" ]]; then
          echo "Skip existing: $run_dir"
          continue
        fi
        if [[ -d "$run_dir" && "$FORCE" == "1" ]]; then
          rm -rf "$run_dir"
        fi

        echo "Running ${dataset} ${trainer} ${shots}-shot seed=${seed}"
        python train.py \
          --root "$FINETUNE_DIR/data" \
          --seed "$seed" \
          --trainer "$trainer" \
          --dataset-config-file "configs/datasets/${dataset}.yaml" \
          --config-file "configs/trainers/${trainer}/${CFG}.yaml" \
          --output-dir "$run_dir" \
          DATASET.NUM_SHOTS "$shots"
      done
    done
  done
done

find "$OUTPUT_DIR" -mindepth 3 -maxdepth 3 -type d | sort | while read -r result_dir; do
  python parse_test_res.py "$result_dir" --test-log --keyword accuracy || true
done
