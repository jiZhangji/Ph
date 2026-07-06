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
repair_pattern = re.compile(r"(?<![\w.])\b(float32|float64|int32|int64|bool_)\b")
repairs = {
    "float32": "np.float32",
    "float64": "np.float64",
    "int32": "np.int32",
    "int64": "np.int64",
    "bool_": "np.bool_",
}

for path in root.rglob("*.py"):
    text = path.read_text()
    text = text.replace("np.np.", "np.")
    text = text.replace("torch.np.np.", "torch.")
    text = text.replace("torch.np.", "torch.")
    patched = alias_pattern.sub(lambda m: {"float": "float", "int": "int", "bool": "bool"}[m.group(1)], text)
    # Repair files touched by an earlier broad sed patch.
    patched = repair_pattern.sub(lambda m: repairs[m.group(1)], patched)
    if patched != text:
        path.write_text(patched)
PY

# timm optimizer helper names differ across versions, and some installations
# expose neither helper. Inject a local fallback so the official code can run
# without pinning the whole environment to an old timm release.
python - "$BASELINE_DIR/Pretraining/main_pretrain.py" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()

text = text.replace("optim_factory.param_groups_weight_decay", "optim_factory.add_weight_decay")

fallback = '''\

if not hasattr(optim_factory, "add_weight_decay"):
    def _sarjepa_add_weight_decay(model, weight_decay=1e-5, skip_list=()):
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {"params": no_decay, "weight_decay": 0.},
            {"params": decay, "weight_decay": weight_decay},
        ]
    optim_factory.add_weight_decay = _sarjepa_add_weight_decay
'''

marker = "import timm.optim.optim_factory as optim_factory\n"
if "_sarjepa_add_weight_decay" not in text:
    if marker not in text:
        raise SystemExit(f"Could not find optimizer import marker in {path}")
    text = text.replace(marker, marker + fallback, 1)

path.write_text(text)
PY

# Keep a rolling checkpoint-last.pth for robust resume after interruptions while
# preserving the official 50-epoch checkpoint cadence.
python - "$BASELINE_DIR/Pretraining/main_pretrain.py" "$BASELINE_DIR/Pretraining/util/misc.py" <<'PY'
import pathlib
import sys

main_path = pathlib.Path(sys.argv[1])
misc_path = pathlib.Path(sys.argv[2])

misc_text = misc_path.read_text()
old_sig = "def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):"
new_sig = "def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, checkpoint_name=None):"
if old_sig in misc_text:
    misc_text = misc_text.replace(old_sig, new_sig, 1)
    misc_text = misc_text.replace(
        "    epoch_name = str(epoch)\n",
        "    epoch_name = str(epoch) if checkpoint_name is None else str(checkpoint_name)\n",
        1,
    )
misc_path.write_text(misc_text)

main_text = main_path.read_text()
needle = """        if args.output_dir and (epoch % 50 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
"""
replacement = """        if args.output_dir:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, checkpoint_name="last")

        if args.output_dir and (epoch % 50 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
"""
if "checkpoint_name=\"last\"" not in main_text:
    if needle not in main_text:
        raise SystemExit(f"Could not find SAR-JEPA save block in {main_path}")
    main_text = main_text.replace(needle, replacement, 1)
main_path.write_text(main_text)
PY

# timm==0.3.x also imports torch._six on old installations. Patch the active
# environment in-place before SAR-JEPA imports timm.
python - <<'PY'
import pathlib
import site
import sys

roots = []
for getter in (site.getsitepackages,):
    try:
        roots.extend(getter())
    except Exception:
        pass
try:
    roots.append(site.getusersitepackages())
except Exception:
    pass
roots.extend(sys.path)

seen = set()
for root in roots:
    if not root or root in seen:
        continue
    seen.add(root)
    helper = pathlib.Path(root) / "timm" / "models" / "layers" / "helpers.py"
    if not helper.exists():
        continue
    text = helper.read_text()
    patched = text.replace("from torch._six import container_abcs", "import collections.abc as container_abcs")
    if patched != text:
        helper.write_text(patched)
        print(f"Patched timm torch._six compatibility: {helper}")
PY

# The official loader walks every file and cv2.imread can return None for
# unsupported/corrupt entries. Use the robust single-channel PIL loader from
# this repo so SAR-JEPA and PH see the same image set.
cp "$ROOT/Pretraining/util/datasets.py" "$BASELINE_DIR/Pretraining/util/datasets.py"

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
RESUME="${RESUME:-}"

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
  --mask_ratio "$MASK_RATIO" \
  --resume "$RESUME"
