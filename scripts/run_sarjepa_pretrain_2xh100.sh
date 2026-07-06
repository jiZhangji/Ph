#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASELINE_DIR="${BASELINE_DIR:-$ROOT/baselines/SAR-JEPA}"

if [[ ! -d "$BASELINE_DIR/Pretraining" ]]; then
  bash "$ROOT/scripts/setup_sarjepa_baseline.sh"
fi

# SAR-JEPA ships a Pretraining/profile.py script. With newer PyTorch/torchvision,
# cProfile imports the stdlib "profile" module during startup; when we run from
# Pretraining/, that local file shadows stdlib profile and triggers its timm
# version assertion. The file is only a profiling helper, so move it aside.
if [[ -f "$BASELINE_DIR/Pretraining/profile.py" ]]; then
  mv "$BASELINE_DIR/Pretraining/profile.py" "$BASELINE_DIR/Pretraining/profile_sarjepa.py"
fi

# Official SAR-JEPA targets older PyTorch versions where torch._six existed.
# PyTorch 2.x removed torch._six, and only math.inf is needed here.
if grep -q "from torch._six import inf" "$BASELINE_DIR/Pretraining/util/misc.py"; then
  sed -i 's/from torch\._six import inf/from math import inf/' "$BASELINE_DIR/Pretraining/util/misc.py"
fi

# Newer PyTorch launchers pass --local-rank, while the official SAR-JEPA
# parser only defines --local_rank. Accept both spellings.
if grep -q "parser.add_argument('--local_rank', default=-1, type=int)" "$BASELINE_DIR/Pretraining/main_pretrain.py"; then
  sed -i "s/parser.add_argument('--local_rank', default=-1, type=int)/parser.add_argument('--local_rank', '--local-rank', default=-1, type=int)/" "$BASELINE_DIR/Pretraining/main_pretrain.py"
fi

# NumPy 1.24 removed deprecated aliases used by older MAE/SAR-JEPA code.
# Patch only exact aliases, not valid scalar types such as np.float32.
python - "$BASELINE_DIR/Pretraining" <<'PY'
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
alias_pattern = re.compile(r"\bnp\.(float|int|bool)\b")
repair_pattern = re.compile(r"\b(float32|float64|int32|int64|bool_)\b")
repairs = {
    "float32": "np.float32",
    "float64": "np.float64",
    "int32": "np.int32",
    "int64": "np.int64",
    "bool_": "np.bool_",
}

for path in root.rglob("*.py"):
    text = path.read_text()
    patched = alias_pattern.sub(lambda m: {"float": "float", "int": "int", "bool": "bool"}[m.group(1)], text)
    # Repair files touched by an earlier broad sed patch.
    patched = repair_pattern.sub(lambda m: repairs[m.group(1)], patched)
    if patched != text:
        path.write_text(patched)
PY

# timm versions differ: some expose param_groups_weight_decay, while the
# SAR-JEPA/MAE code path also works with add_weight_decay.
if grep -q "optim_factory.param_groups_weight_decay" "$BASELINE_DIR/Pretraining/main_pretrain.py"; then
  sed -i 's/optim_factory\.param_groups_weight_decay/optim_factory.add_weight_decay/g' "$BASELINE_DIR/Pretraining/main_pretrain.py"
fi

DATA_PATH="${DATA_PATH:-$ROOT/dataset/modelscope/extracted/Pretraining_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/sarjepa_pretrain_2xh100}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
GPUS="${GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-25642}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-300}"
BLR="${BLR:-1e-3}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-20}"
NUM_WORKERS="${NUM_WORKERS:-16}"
MODEL="${MODEL:-mae_vit_base_patch16}"
MASK_RATIO="${MASK_RATIO:-0.8}"
WINDOW_SIZE="${WINDOW_SIZE:-7}"
NUM_WINDOW="${NUM_WINDOW:-4}"

DATA_PATH="$(cd "$DATA_PATH" && pwd)"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
LOG_DIR="$(cd "$LOG_DIR" && pwd)"

echo "SAR-JEPA pretraining"
echo "BASELINE_DIR=$BASELINE_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "DATA_PATH=$DATA_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "EPOCHS=$EPOCHS"
echo "BLR=$BLR"

cd "$BASELINE_DIR/Pretraining"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
python -m torch.distributed.launch \
  --nproc_per_node="$GPUS" \
  --master_port="$MASTER_PORT" \
  main_pretrain.py \
  --model "$MODEL" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --log_dir "$LOG_DIR" \
  --device cuda \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --blr "$BLR" \
  --warmup_epochs "$WARMUP_EPOCHS" \
  --num_workers "$NUM_WORKERS" \
  --pin_mem \
  --window_size "$WINDOW_SIZE" \
  --num_window "$NUM_WINDOW" \
  --mask_ratio "$MASK_RATIO"
