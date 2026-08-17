#!/usr/bin/env bash
set -Eeuo pipefail

# Build and upload a paper-facing incremental archive. The archive deliberately
# excludes the previously released main PhyD, stage-I, and SAR-JEPA checkpoints.

ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
MS_REPO_ID="${MS_REPO_ID:-shimian123/PhyD-MAE}"
MODELSCOPE_ENDPOINT="${MODELSCOPE_ENDPOINT:-https://modelscope.cn}"
MAX_WORKERS="${MAX_WORKERS:-4}"
DRY_RUN="${DRY_RUN:-0}"
STAMP="${STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
RELEASE_PARENT="${RELEASE_PARENT:-$(dirname "$ROOT")}" 
RELEASE="${RELEASE:-$RELEASE_PARENT/PhyD-MAE-paper-increment-$STAMP}"
REMOTE_PREFIX="${REMOTE_PREFIX:-paper_archive_20260818}"
CONTENT="$RELEASE/$REMOTE_PREFIX"

SASGT_OUTPUT="$ROOT/few_shot_classification/finetune/output_sasgt_parameter_sensitivity_lr1e3_10seeds"
LFST_OUTPUT="$ROOT/few_shot_classification/finetune/output_lfst_weight_sensitivity_official_lr1e3_10seeds"
FINETUNE_ROOT="$ROOT/few_shot_classification/finetune"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

for command_name in python find sort sha256sum rsync git; do
  require_command "$command_name"
done

ROOT="$(readlink -f "$ROOT")"
mkdir -p "$RELEASE_PARENT"
RELEASE_PARENT="$(readlink -f "$RELEASE_PARENT")"
RELEASE="$RELEASE_PARENT/$(basename "$RELEASE")"
CONTENT="$RELEASE/$REMOTE_PREFIX"

[[ -d "$ROOT" ]] || die "Missing project root: $ROOT"
[[ ! -e "$RELEASE" ]] || die "Release already exists: $RELEASE"
case "$RELEASE" in
  "$ROOT"|"$ROOT"/*) die "Release must be outside the project root" ;;
esac

mkdir -p \
  "$CONTENT/weights/sasgt_parameter_sensitivity" \
  "$CONTENT/weights/lfst_weight_sensitivity" \
  "$CONTENT/weights/manifest_referenced" \
  "$CONTENT/results" \
  "$CONTENT/code_snapshot" \
  "$CONTENT/reproducibility/environment" \
  "$CONTENT/reproducibility/data_manifests" \
  "$CONTENT/reproducibility/few_shot_splits" \
  "$CONTENT/reproducibility/git" \
  "$CONTENT/manuscript"

copy_required() {
  local source="$1"
  local destination="$2"
  [[ -f "$source" ]] || die "Missing required file: $source"
  mkdir -p "$(dirname "$destination")"
  cp -f "$source" "$destination"
}

copy_optional() {
  local source="$1"
  local destination="$2"
  [[ -f "$source" ]] || return 0
  mkdir -p "$(dirname "$destination")"
  cp -f "$source" "$destination"
}

copy_csv_without_server_path() {
  local source="$1"
  local destination="$2"
  [[ -f "$source" ]] || return 0
  python - "$source" "$destination" <<'PY'
import csv
import sys
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
destination.parent.mkdir(parents=True, exist_ok=True)
with source.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if not rows:
    raise RuntimeError(f"Empty CSV: {source}")
fields = [field for field in rows[0] if field != "log"]
with destination.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
PY
}

copy_weight() {
  local source="$1"
  local destination="$2"
  copy_required "$source" "$CONTENT/$destination"
  echo "Included weight: $destination"
}

echo "Generating complete SASGT/LFST summaries ..."
ACTION=summary bash "$ROOT/scripts/run_distributed_sasgt_lfst_downstream.sh"

copy_weight "$ROOT/runs/phyd_sasgt_gamma0_30e_bs1088/checkpoint-29.pth" \
  "weights/sasgt_parameter_sensitivity/gamma0/checkpoint-29.pth"
copy_weight "$ROOT/runs/phyd_sasgt_gamma2_30e_bs1088/checkpoint-29.pth" \
  "weights/sasgt_parameter_sensitivity/gamma2/checkpoint-29.pth"
copy_weight "$ROOT/runs/phyd_sasgt_w3_30e_bs1088/checkpoint-29.pth" \
  "weights/sasgt_parameter_sensitivity/w3/checkpoint-29.pth"
copy_weight "$ROOT/runs/phyd_sasgt_w11_30e_bs1088/checkpoint-29.pth" \
  "weights/sasgt_parameter_sensitivity/w11/checkpoint-29.pth"
copy_weight "$ROOT/runs/phyd_sasgt_tau0p5_30e_bs1088/checkpoint-29.pth" \
  "weights/sasgt_parameter_sensitivity/tau0p5/checkpoint-29.pth"
copy_weight "$ROOT/runs/phyd_sasgt_tau2_30e_bs1088/checkpoint-29.pth" \
  "weights/sasgt_parameter_sensitivity/tau2/checkpoint-29.pth"

for spec in \
  "0p05:0.05" \
  "0p1:0.1" \
  "0p2:0.2" \
  "0p5:0.5" \
  "1p0:1.0"
do
  tag="${spec%%:*}"
  label="${spec#*:}"
  copy_weight \
    "$ROOT/runs/phyd_lfst_weight_${tag}_ft250_to300_bs1024/checkpoint-300.pth" \
    "weights/lfst_weight_sensitivity/weight_${label}/checkpoint-300.pth"
done

copy_light_result_tree() {
  local source="$1"
  local destination="$2"
  [[ -d "$source" ]] || return 0
  mkdir -p "$destination"
  (
    cd "$source"
    find . -type f \
      \( -name 'log.txt' \
         -o -name '*.csv' \
         -o -name '*.tsv' \
         -o -name '*.json' \
         -o -name '*.txt' \
         -o -name '*.yaml' \
         -o -name '*.yml' \) \
      -print0 \
      | rsync -a --from0 --files-from=- ./ "$destination/"
  )
}

copy_light_result_tree "$SASGT_OUTPUT" "$CONTENT/results/sasgt_parameter_sensitivity"
copy_light_result_tree "$LFST_OUTPUT" "$CONTENT/results/lfst_weight_sensitivity"

# Replace copied summary CSVs with sanitized versions that do not expose the
# server's absolute log paths.
copy_csv_without_server_path \
  "$SASGT_OUTPUT/results_per_seed.csv" \
  "$CONTENT/results/sasgt_parameter_sensitivity/results_per_seed.csv"
copy_csv_without_server_path \
  "$SASGT_OUTPUT/results_mean_std.csv" \
  "$CONTENT/results/sasgt_parameter_sensitivity/results_mean_std.csv"
copy_csv_without_server_path \
  "$LFST_OUTPUT/results_per_seed.csv" \
  "$CONTENT/results/lfst_weight_sensitivity/results_per_seed.csv"
copy_csv_without_server_path \
  "$LFST_OUTPUT/results_mean_std_max.csv" \
  "$CONTENT/results/lfst_weight_sensitivity/results_mean_std_max.csv"

for result_dir in "$FINETUNE_ROOT"/output*; do
  [[ -d "$result_dir" ]] || continue
  name="$(basename "$result_dir")"
  case "$name" in
    *sasgt_internal*|*target_ablation*|*paper_ablation*|*controlled_speckle*|\
    *additional_speckle*|*speckle_stress*|*lfst_cutoff*|*sasgt_scale*)
      echo "Including paper result tree: $name"
      copy_light_result_tree "$result_dir" "$CONTENT/results/$name"
      ;;
  esac
done

# Copy checkpoints referenced by selected result manifests. Previously released
# base checkpoints are explicitly excluded to keep this archive incremental.
python - "$ROOT" "$CONTENT" <<'PY'
import csv
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
content = Path(sys.argv[2]).resolve()
finetune = root / "few_shot_classification" / "finetune"
excluded = {
    (root / "runs/sarjepa_official_phyd_ft250_bs1024_lfst0p1_image_2xh200/checkpoint-300.pth").resolve(),
    (root / "runs/sarjepa_official_phyd_2xh100/checkpoint-250.pth").resolve(),
    (root / "runs/sarjepa_pretrain_2xh100/checkpoint-200.pth").resolve(),
    (root / "runs/phyd_ckpt300_target_pilot_30e_bs1088_msgt_only/checkpoint-29.pth").resolve(),
}
patterns = (
    "sasgt_internal",
    "target_ablation",
    "paper_ablation",
    "controlled_speckle",
    "additional_speckle",
    "speckle_stress",
    "lfst_cutoff",
    "sasgt_scale",
)
copied = set()
for manifest in sorted(finetune.glob("output*/expected_matrix.tsv")):
    if not any(pattern in manifest.parent.name for pattern in patterns):
        continue
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            checkpoint_text = row.get("checkpoint", "").strip()
            if not checkpoint_text:
                continue
            checkpoint = Path(checkpoint_text)
            if not checkpoint.is_absolute():
                checkpoint = root / checkpoint
            checkpoint = checkpoint.resolve()
            if checkpoint in excluded or checkpoint in copied or not checkpoint.is_file():
                continue
            model = row.get("model", checkpoint.parent.name).replace("/", "_")
            destination = content / "weights" / "manifest_referenced" / model / checkpoint.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checkpoint, destination)
            copied.add(checkpoint)
            print(f"Included manifest checkpoint: {checkpoint} -> {destination}")
PY

echo "Capturing exact code snapshot ..."
for relative in \
  Pretraining_sarjepa_official_phyd \
  few_shot_classification/finetune/trainers \
  few_shot_classification/finetune/configs \
  scripts \
  release/experiment_records \
  release/paper_revisions
do
  [[ -d "$ROOT/$relative" ]] || continue
  mkdir -p "$CONTENT/code_snapshot/$relative"
  rsync -a \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='tmp/' \
    "$ROOT/$relative/" "$CONTENT/code_snapshot/$relative/"
done

copy_optional "$ROOT/README.md" "$CONTENT/code_snapshot/README.md"
copy_optional "$ROOT/requirements.txt" "$CONTENT/reproducibility/environment/requirements.txt"
copy_optional "$ROOT/environment.yml" "$CONTENT/reproducibility/environment/environment.yml"

git -C "$ROOT" rev-parse HEAD > "$CONTENT/reproducibility/git/GIT_COMMIT.txt"
git -C "$ROOT" status --short > "$CONTENT/reproducibility/git/GIT_STATUS.txt"
git -C "$ROOT" diff --binary HEAD > "$CONTENT/reproducibility/git/WORKTREE.patch"
git -C "$ROOT" ls-files --others --exclude-standard \
  > "$CONTENT/reproducibility/git/UNTRACKED_FILES.txt"

python --version > "$CONTENT/reproducibility/environment/python-version.txt" 2>&1
python - <<'PY' > "$CONTENT/reproducibility/environment/framework-versions.txt" 2>&1
import platform
print("platform:", platform.platform())
for name in ("torch", "torchvision", "timm", "numpy", "scipy"):
    try:
        module = __import__(name)
        print(f"{name}:", getattr(module, "__version__", "unknown"))
    except Exception as exc:
        print(f"{name}: unavailable ({exc})")
try:
    import torch
    print("cuda:", torch.version.cuda)
    print("cudnn:", torch.backends.cudnn.version())
except Exception:
    pass
PY
python -m pip freeze > "$CONTENT/reproducibility/environment/pip-freeze.txt"
if command -v conda >/dev/null 2>&1; then
  conda env export --no-builds > "$CONTENT/reproducibility/environment/conda-environment-no-builds.yml" || true
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -q > "$CONTENT/reproducibility/environment/nvidia-smi.txt" || true
fi

write_data_manifest() {
  local source="$1"
  local output="$2"
  [[ -d "$source" ]] || return 0
  (
    cd "$source"
    find . -type f -printf '%s\t%P\n' | sort -k2,2
  ) > "$output"
}

write_data_manifest \
  "$ROOT/dataset/modelscope/extracted/Pretraining_dataset" \
  "$CONTENT/reproducibility/data_manifests/pretraining_files_size.tsv"
write_data_manifest \
  "$ROOT/dataset/modelscope/extracted/classification_dataset" \
  "$CONTENT/reproducibility/data_manifests/downstream_files_size.tsv"

for data_root in \
  "$ROOT/dataset/modelscope/extracted/classification_dataset" \
  "$ROOT/dataset/modelscope/extracted/classification_dataset/few_shot_classification"
do
  [[ -d "$data_root" ]] || continue
  (
    cd "$data_root"
    find . -type f \
      \( -name '*.pkl' \
         -o -name '*.json' \
         -o -name '*.csv' \
         -o -name '*.txt' \
         -o -name '*.yaml' \
         -o -name '*.yml' \) \
      -print0 \
      | rsync -a --ignore-existing --from0 --files-from=- ./ \
          "$CONTENT/reproducibility/few_shot_splits/"
  )
done

copy_optional "$ROOT/scripts/download_modelscope_data.py" \
  "$CONTENT/reproducibility/data_manifests/download_modelscope_data.py"

manuscript_found=0
for relative in paper-稿件 paper supplementary manuscript; do
  [[ -d "$ROOT/$relative" ]] || continue
  manuscript_found=1
  mkdir -p "$CONTENT/manuscript/$relative"
  rsync -a \
    --exclude='*.aux' \
    --exclude='*.bbl' \
    --exclude='*.blg' \
    --exclude='*.fls' \
    --exclude='*.fdb_latexmk' \
    --exclude='*.synctex.gz' \
    "$ROOT/$relative/" "$CONTENT/manuscript/$relative/"
done
if [[ "$manuscript_found" == 0 ]]; then
  cat > "$CONTENT/manuscript/MANUSCRIPT_NOT_ON_SERVER.txt" <<'EOF'
The complete main-paper and supplementary LaTeX sources were not found under
the server project root. Synchronize the local paper directory to the server
and create a later manuscript-only update if archival of the submission source
is required.
EOF
fi

cat > "$CONTENT/PAPER_ARTIFACT_MAP.csv" <<'EOF'
paper_item,artifact_directory,purpose
SASGT parameter sensitivity,weights/sasgt_parameter_sensitivity;results/sasgt_parameter_sensitivity,Seven configurations and 420 downstream runs
LFST weight sensitivity,weights/lfst_weight_sensitivity;results/lfst_weight_sensitivity,Five weights and 900 downstream runs
SASGT internal ablation,weights/manifest_referenced;results/output_sasgt_internal_ablation_fusar_10seeds,Uniform/no-scale/no-log/full comparison
Dual-target ablation,weights/manifest_referenced;results,Pixel/LFST/SASGT/dual supervision comparison
Additional speckle stress test,results,Controlled robustness evidence and fixed noise realizations
Exact source snapshot,code_snapshot;reproducibility/git,Training/evaluation implementation and working-tree patch
Data provenance,reproducibility/data_manifests;reproducibility/few_shot_splits,Dataset inventory and exact few-shot split metadata
EOF

cat > "$CONTENT/README.md" <<EOF
# PhyD-MAE paper incremental archive

This archive adds paper-facing artifacts that are not part of the previous
base release. It contains sensitivity and ablation checkpoints, complete
per-seed downstream records, exact source snapshots, environment metadata,
dataset inventories, and few-shot split files.

Previously released artifacts intentionally excluded here:

- Main PhyD checkpoint-300
- Stage-I PhyD checkpoint-250
- Official SAR-JEPA reproduction checkpoint-200
- Default SASGT-only checkpoint-29
- Previously released attention heatmap archives
- Raw SAR image payloads
- Intermediate and failed/collapsed checkpoints

Source Git commit: $(git -C "$ROOT" rev-parse HEAD)
Build time UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

python - "$CONTENT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
weights = sorted(path.relative_to(root).as_posix() for path in root.rglob("*.pth"))
result_logs = sorted(path.relative_to(root).as_posix() for path in root.rglob("log.txt"))
manifest = {
    "archive": "PhyD-MAE paper incremental archive",
    "weight_files": weights,
    "weight_count": len(weights),
    "downstream_log_count": len(result_logs),
    "expected_core_sensitivity_logs": 1320,
    "raw_images_included": False,
}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
if len(result_logs) < 1320:
    raise RuntimeError(f"Expected at least 1320 downstream logs, found {len(result_logs)}")
print(json.dumps(manifest, indent=2))
PY

(
  cd "$RELEASE"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    > SHA256SUMS
)

echo
echo "Release directory: $RELEASE"
echo "Release size: $(du -sh "$RELEASE" | awk '{print $1}')"
echo "Release files: $(find "$RELEASE" -type f | wc -l)"
echo "Weight files: $(find "$RELEASE" -type f -name '*.pth' | wc -l)"
echo "Result logs: $(find "$RELEASE" -type f -name log.txt | wc -l)"

if [[ "$DRY_RUN" == 1 ]]; then
  echo "DRY_RUN=1: package created without upload."
  exit 0
fi

require_command ms-hub
ms-hub --endpoint "$MODELSCOPE_ENDPOINT" create "$MS_REPO_ID" \
  --repo-type model \
  --visibility public \
  --license apache-2.0 \
  --description "Official implementation and paper artifacts of PhyD-MAE" \
  --exist-ok

ms-hub --endpoint "$MODELSCOPE_ENDPOINT" upload \
  "$MS_REPO_ID" "$RELEASE" \
  --repo-type model \
  --commit-message "Add complete paper sensitivity and reproducibility archive" \
  --max-workers "$MAX_WORKERS" \
  --use-cache

echo "Upload complete: $MODELSCOPE_ENDPOINT/models/$MS_REPO_ID"
echo "Uploaded release: $RELEASE"
