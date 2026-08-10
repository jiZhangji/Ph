#!/usr/bin/env bash
set -Eeuo pipefail

# Build a publication-ready snapshot of the server project and upload it to
# ModelScope. Raw datasets, caches, temporary files, intermediate checkpoints,
# and large feature arrays are excluded. Git-tracked files are always retained.

SRC="${SRC:-/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/Ph}"
MS_REPO_ID="${MS_REPO_ID:-shimian123/PhyD-MAE}"
MODELSCOPE_ENDPOINT="${MODELSCOPE_ENDPOINT:-https://modelscope.cn}"
MAX_FILES="${MAX_FILES:-95000}"
MAX_WORKERS="${MAX_WORKERS:-8}"
INCLUDE_RESULTS="${INCLUDE_RESULTS:-1}"
INCLUDE_FINAL_RUN_WEIGHTS="${INCLUDE_FINAL_RUN_WEIGHTS:-1}"
DRY_RUN="${DRY_RUN:-0}"

STAMP="$(date +%Y%m%d-%H%M%S)"
RELEASE_PARENT="${RELEASE_PARENT:-$(dirname "$SRC")}"
RELEASE="${RELEASE:-$RELEASE_PARENT/PhyD-MAE-modelscope-release-$STAMP}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

for command_name in git rsync find sort awk numfmt ms-hub; do
    require_command "$command_name"
done

SRC="$(readlink -f "$SRC")"
mkdir -p "$RELEASE_PARENT"
RELEASE_PARENT="$(readlink -f "$RELEASE_PARENT")"
RELEASE="$RELEASE_PARENT/$(basename "$RELEASE")"

[[ -d "$SRC" ]] || die "Source directory does not exist: $SRC"
case "$RELEASE" in
    "$SRC"|"$SRC"/*)
        die "Release directory must be outside the source tree: $RELEASE"
        ;;
esac
[[ ! -e "$RELEASE" ]] || die "Release directory already exists: $RELEASE"

copy_git_tracked_files() {
    local destination="$1"
    if git -C "$SRC" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "Copying all Git-tracked files ..."
        git -C "$SRC" ls-files -z \
            | rsync -a --ignore-missing-args --from0 --files-from=- \
                "$SRC/" "$destination/"
    else
        echo "WARNING: $SRC is not a Git worktree; tracked-file copy skipped."
    fi
}

copy_project_tree() {
    local destination="$1"
    local mode="$2"
    local -a excludes=(
        "--exclude=.git/"
        "--exclude=__pycache__/"
        "--exclude=.pytest_cache/"
        "--exclude=.mypy_cache/"
        "--exclude=.ruff_cache/"
        "--exclude=.cache/"
        "--exclude=tmp/"
        "--exclude=downloads/"
        "--exclude=dataset/"
        "--exclude=datasets/"
        "--exclude=data/"
        "--exclude=tsne_features*/"
        "--exclude=*.pyc"
        "--exclude=*.pyo"
        "--exclude=*.npy"
        "--exclude=*.npz"
        "--exclude=*.h5"
        "--exclude=*.hdf5"
        "--exclude=*.lmdb"
        "--exclude=*.pth"
        "--exclude=*.pth.tar"
        "--exclude=*.pt"
        "--exclude=*.ckpt"
        "--exclude=*.safetensors"
        "--exclude=*.tmp"
        "--exclude=*.part"
    )

    # The complete mode keeps lightweight logs, result tables, paper figures,
    # and visualization outputs. The lean fallback drops generated run output.
    if [[ "$mode" == "lean" || "$INCLUDE_RESULTS" == "0" ]]; then
        excludes+=(
            "--exclude=logs/"
            "--exclude=output*/"
            "--exclude=paper-results/"
        )
    fi

    # Runs are copied selectively below so that final checkpoints are retained
    # without publishing every intermediate training checkpoint.
    excludes+=("--exclude=runs/")

    echo "Copying project tree in $mode mode ..."
    rsync -a --prune-empty-dirs "${excludes[@]}" "$SRC/" "$destination/"
}

copy_weight_directory() {
    local relative_path="$1"
    local source_path="$SRC/$relative_path"
    local destination_path="$RELEASE/$relative_path"

    [[ -d "$source_path" ]] || return 0
    echo "Copying weight directory: $relative_path"
    mkdir -p "$destination_path"
    rsync -a \
        --exclude='.cache/' \
        --exclude='*.tmp' \
        --exclude='*.part' \
        "$source_path/" "$destination_path/"
}

copy_final_run_weights() {
    [[ "$INCLUDE_FINAL_RUN_WEIGHTS" == "1" ]] || return 0
    [[ -d "$SRC/runs" ]] || return 0

    echo "Copying final and paper-used run checkpoints ..."
    (
        cd "$SRC"
        find runs -type f \
            \( -name 'checkpoint-last.pth' \
               -o -name 'checkpoint-29.pth' \
               -o -name 'checkpoint-best.pth' \
               -o -name 'best.pth' \
               -o -name 'best_model.pth' \) \
            -print0 \
            | rsync -a --from0 --files-from=- "$SRC/" "$RELEASE/"
    )
}

write_release_metadata() {
    local mode="$1"
    {
        printf 'ModelScope repository: %s\n' "$MS_REPO_ID"
        printf 'Source directory: %s\n' "$SRC"
        printf 'Release directory: %s\n' "$RELEASE"
        printf 'Build mode: %s\n' "$mode"
        printf 'Build time (UTC): %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf '\nExcluded content:\n'
        printf '%s\n' \
            '- Raw dataset directories and dataset payloads' \
            '- Git metadata and Python/tool caches' \
            '- Temporary and downloaded files' \
            '- Large feature arrays (NPY, NPZ, HDF5, LMDB)' \
            '- Intermediate run checkpoints' \
            '- Generated run outputs in lean fallback mode'
    } > "$RELEASE/MODELSCOPE_RELEASE_INFO.txt"

    find "$RELEASE" -type f \
        ! -name 'MODELSCOPE_UPLOAD_MANIFEST.tsv' \
        -printf '%s\t%P\n' \
        | sort -k2,2 \
        > "$RELEASE/MODELSCOPE_UPLOAD_MANIFEST.tsv"
}

release_file_count() {
    find "$RELEASE" -type f | wc -l | awk '{print $1}'
}

release_summary() {
    local count
    count="$(release_file_count)"
    echo
    echo "Release directory: $RELEASE"
    echo "Release size: $(du -sh "$RELEASE" | awk '{print $1}')"
    echo "Release files: $count"
    echo "Largest files:"
    find "$RELEASE" -type f -printf '%s\t%p\n' \
        | sort -nr \
        | head -n 20 \
        | numfmt --field=1 --to=iec || true
}

build_release() {
    local mode="$1"
    mkdir -p "$RELEASE"
    copy_git_tracked_files "$RELEASE"
    copy_project_tree "$RELEASE" "$mode"

    # These directories commonly contain intentionally published weights.
    copy_weight_directory "weights"
    copy_weight_directory "hf_release"
    copy_final_run_weights
    write_release_metadata "$mode"
}

echo "Building complete ModelScope release ..."
build_release "complete"
release_summary

FILE_COUNT="$(release_file_count)"
if (( FILE_COUNT > MAX_FILES )); then
    echo
    echo "Complete release has $FILE_COUNT files, above safety limit $MAX_FILES."
    echo "Building a lean fallback without generated logs and result outputs."

    RELEASE="${RELEASE}-lean"
    [[ ! -e "$RELEASE" ]] || die "Lean release directory already exists: $RELEASE"
    build_release "lean"
    release_summary
    FILE_COUNT="$(release_file_count)"
fi

(( FILE_COUNT <= MAX_FILES )) \
    || die "Release still contains $FILE_COUNT files; inspect $RELEASE before upload."

OVERSIZED_FILE="$(find "$RELEASE" -type f -size +102400M -print -quit)"
[[ -z "$OVERSIZED_FILE" ]] \
    || die "File exceeds ModelScope 100 GB limit: $OVERSIZED_FILE"

echo
echo "Checking ModelScope repository ..."
ms-hub --endpoint "$MODELSCOPE_ENDPOINT" create "$MS_REPO_ID" \
    --repo-type model \
    --visibility public \
    --license apache-2.0 \
    --description "Official implementation and model weights of PhyD-MAE" \
    --exist-ok

if [[ "$DRY_RUN" == "1" ]]; then
    echo
    echo "DRY_RUN=1: release prepared but not uploaded."
    echo "Upload source: $RELEASE"
    exit 0
fi

echo
echo "Uploading release to $MS_REPO_ID ..."
ms-hub --endpoint "$MODELSCOPE_ENDPOINT" upload \
    "$MS_REPO_ID" "$RELEASE" \
    --repo-type model \
    --commit-message "Release PhyD-MAE source, experiments, and model weights" \
    --max-workers "$MAX_WORKERS" \
    --use-cache

echo
echo "Upload complete: $MODELSCOPE_ENDPOINT/models/$MS_REPO_ID"
echo "Uploaded release: $RELEASE"

