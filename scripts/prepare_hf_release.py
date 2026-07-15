#!/usr/bin/env python3
"""Build a reproducible Hugging Face release folder from server checkpoints."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


MODEL_SPECS = [
    {
        "id": "phyd-best-ckpt300",
        "tier": "main",
        "required": True,
        "source": "runs/sarjepa_official_phyd_ft250_bs1024_lfst0p1_image_2xh200/checkpoint-300.pth",
        "destination": "models/main/phyd-best-ckpt300/model.pth",
        "key_checkpoints": ["checkpoint-300.pth"],
        "configuration": {
            "architecture": "SAR-JEPA official framework with PhyD SASGT and LFST targets",
            "grad_loss_weight": 1.0,
            "lfst_loss_weight": 0.1,
            "target_norm": "image",
            "sfafm": False,
            "training_history": "continued from phyd-stage1-ckpt250",
        },
        "result_summary": "Best confirmed paper checkpoint; MSTAR 10-shot finetune 70.22 +- 3.00 accuracy over 20 seeds.",
    },
    {
        "id": "phyd-stage1-ckpt250",
        "tier": "reproduction",
        "required": False,
        "source": "runs/sarjepa_official_phyd_2xh100/checkpoint-250.pth",
        "destination": "models/reproduction/phyd-stage1-ckpt250/model.pth",
        "key_checkpoints": ["checkpoint-250.pth"],
        "configuration": {
            "architecture": "SAR-JEPA official framework with PhyD SASGT and LFST targets",
            "grad_loss_weight": 1.0,
            "lfst_loss_weight": 1.0,
            "target_norm": "patch",
            "sfafm": False,
        },
        "result_summary": "Stage-I checkpoint used to initialize the best checkpoint-300 continuation.",
    },
    {
        "id": "sarjepa-official-reproduction-ckpt200",
        "tier": "baseline",
        "required": False,
        "source": "runs/sarjepa_pretrain_2xh100/checkpoint-200.pth",
        "destination": "models/baseline/sarjepa-official-reproduction-ckpt200/model.pth",
        "key_checkpoints": ["checkpoint-200.pth"],
        "configuration": {
            "architecture": "Official SAR-JEPA local reproduction",
            "sfafm": False,
        },
        "result_summary": "MSTAR 10-shot finetune 58.74 +- 3.05 accuracy and 57.50 +- 2.62 macro-F1 over 5 seeds.",
    },
    {
        "id": "phyd-warmstart-drift-ckpt299",
        "tier": "analysis",
        "required": False,
        "source": "runs/sarjepa_official_phyd_warmstart_bestckpt300_bs1088_lfst0p1_image_20260709_174032/checkpoint-299.pth",
        "destination": "models/analysis/phyd-warmstart-drift-ckpt299/model.pth",
        "key_checkpoints": ["checkpoint-220.pth", "checkpoint-299.pth"],
        "configuration": {
            "architecture": "PhyD model-only warm start from the best checkpoint-300",
            "grad_loss_weight": 1.0,
            "lfst_loss_weight": 0.1,
            "target_norm": "image",
            "sfafm": False,
            "effective_batch_size": 1088,
        },
        "result_summary": "Complete strict-official 360-run result set; useful for representation-drift analysis, not the main model.",
    },
    {
        "id": "phyd-sfafm7-ckpt20",
        "tier": "experimental",
        "required": False,
        "source": "runs/phyd_sfafm7_every2end_from_best300_g1_lfst0p1_image_bs768_300e_2xh200/checkpoint-20.pth",
        "destination": "models/experimental/phyd-sfafm7-ckpt20/model.pth",
        "key_checkpoints": [
            "checkpoint-0.pth",
            "checkpoint-10.pth",
            "checkpoint-20.pth",
            "checkpoint-30.pth",
        ],
        "configuration": {
            "architecture": "PhyD with seven SFAFM modules after encoder blocks 2/4/6/8/10/12 and at encoder end",
            "grad_loss_weight": 1.0,
            "lfst_loss_weight": 0.1,
            "target_norm": "image",
            "sfafm": True,
            "sfafm_layout": "every2_end",
        },
        "result_summary": "Experimental candidate; MSTAR 10-shot finetune seed0 67.50 accuracy/63.90 macro-F1, New_FUSAR 20/40-shot finetune 84.34/87.08 accuracy over 5 seeds.",
    },
]


HISTORICAL_MODEL_SPECS = [
    {
        "id": "legacy-sasgt-image-ckpt24",
        "tier": "historical",
        "required": False,
        "source": "runs/overnight_ablation_8runs_sasgt_only_image/checkpoint-24.pth",
        "destination": "models/historical/legacy-sasgt-image-ckpt24/model.pth",
        "key_checkpoints": ["checkpoint-24.pth"],
        "configuration": {
            "architecture": "Legacy SASGT-only image-normalized ablation",
            "sfafm": False,
        },
        "result_summary": "Legacy-pipeline MSTAR 10-shot seed0 57.60 accuracy and 54.30 macro-F1.",
    },
    {
        "id": "legacy-ph-ckpt99",
        "tier": "historical",
        "required": False,
        "source": "runs/pretrain_2xh100_rerun_bs256_lr1e-4/checkpoint-99.pth",
        "destination": "models/historical/legacy-ph-ckpt99/model.pth",
        "key_checkpoints": ["checkpoint-99.pth"],
        "configuration": {
            "architecture": "Legacy Ph checkpoint",
            "sfafm": "downstream-only in the recorded test",
        },
        "result_summary": "Legacy-pipeline MSTAR 10-shot seed0 54.80 accuracy and 53.60 macro-F1 with downstream SFAFM.",
    },
]


RAW_RESULT_DIRS = [
    "few_shot_classification/finetune/output_phyd_ft250_ckpt300_official_all10_mstar",
    "few_shot_classification/finetune/output_phyd_ft250_ckpt300_official_all10_fusar",
    "few_shot_classification/finetune/output_phyd_ft250_ckpt300_official_all10_saracd",
    "few_shot_classification/finetune/output_warmstart_bestckpt300_ckpt299_official_all_fast3",
    "few_shot_classification/finetune/output_phyd_sfafm7_every2_sweep",
    "few_shot_classification/finetune/output_phyd_sfafm7_ckpt20_fusar_saracd_finetune_5seeds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--include-full-checkpoints", action="store_true")
    parser.add_argument("--include-full-runs", action="store_true")
    parser.add_argument(
        "--full-run-archive-format",
        choices=("directory", "zip"),
        default="directory",
    )
    parser.add_argument(
        "--run-checkpoint-policy",
        choices=("all", "key"),
        default="all",
    )
    parser.add_argument("--skip-model-only", action="store_true")
    parser.add_argument("--include-raw-logs", action="store_true")
    parser.add_argument("--include-historical", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def ensure_empty_package_dir(package_dir: Path, root: Path, overwrite: bool) -> None:
    package_dir = package_dir.resolve()
    root = root.resolve()
    if package_dir == root or root not in package_dir.parents:
        raise ValueError("package-dir must be a child of the repository root")
    if package_dir.exists() and any(package_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Package directory is not empty: {package_dir}. "
                "Choose another PACKAGE_DIR or set OVERWRITE=1."
            )
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)


def copy_tree_if_present(source: Path, destination: Path, label: str) -> bool:
    if not source.exists():
        print(f"WARNING: missing {label}: {source}", file=sys.stderr)
        return False
    shutil.copytree(source, destination, dirs_exist_ok=True)
    return True


def load_and_save_model_only(source: Path, destination: Path, spec: dict) -> dict:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to prepare model-only checkpoints") from exc

    print(f"Converting model-only checkpoint: {source}")
    try:
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:
        # PyTorch releases before the weights_only argument still use this path.
        checkpoint = torch.load(source, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type in {source}: {type(checkpoint)!r}")
    model_state = checkpoint.get("model", checkpoint)
    if not isinstance(model_state, dict):
        raise TypeError(f"Checkpoint model state is not a mapping: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_state,
        "metadata": {
            "id": spec["id"],
            "tier": spec["tier"],
            "source_checkpoint": spec["source"],
            "configuration": spec["configuration"],
            "result_summary": spec["result_summary"],
        },
    }
    if "epoch" in checkpoint:
        payload["epoch"] = checkpoint["epoch"]
    torch.save(payload, destination)
    epoch = checkpoint.get("epoch")
    del payload, model_state, checkpoint
    gc.collect()
    return {"epoch": epoch, "size_bytes": destination.stat().st_size}


def copy_full_checkpoint(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        print(f"Hard-linked full checkpoint: {destination}")
    except OSError:
        shutil.copy2(source, destination)
        print(f"Copied full checkpoint: {destination}")


def link_or_copy_file(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def is_weight_artifact(path: Path) -> bool:
    return path.suffix.lower() in {".pth", ".pt", ".ckpt", ".onnx"}


def should_include_run_file(path: Path, key_checkpoints: set[str] | None) -> bool:
    if key_checkpoints is None or not is_weight_artifact(path):
        return True
    return path.name in key_checkpoints


def link_or_copy_tree(
    source: Path,
    destination: Path,
    key_checkpoints: set[str] | None = None,
) -> dict:
    linked = 0
    copied = 0
    files = 0
    size_bytes = 0
    skipped_weight_files = []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(os.readlink(path))
            continue
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not path.is_file():
            continue
        if not should_include_run_file(path, key_checkpoints):
            skipped_weight_files.append(path.relative_to(source).as_posix())
            continue
        method = link_or_copy_file(path, target)
        files += 1
        size_bytes += path.stat().st_size
        if method == "hardlink":
            linked += 1
        else:
            copied += 1
    return {
        "files": files,
        "size_bytes": size_bytes,
        "hardlinked_files": linked,
        "copied_files": copied,
        "skipped_weight_files": skipped_weight_files,
    }


def find_external_run_logs(root: Path, run_name: str) -> list[Path]:
    log_root = root / "logs"
    if not log_root.is_dir():
        return []
    return [path for path in sorted(log_root.glob(f"{run_name}*")) if path.is_file()]


def collect_external_run_logs(root: Path, package_dir: Path, run_name: str) -> list[str]:
    copied = []
    destination_root = package_dir / "external_logs" / run_name
    for source in find_external_run_logs(root, run_name):
        destination = destination_root / source.name
        link_or_copy_file(source, destination)
        copied.append(source.relative_to(root).as_posix())
    return copied


def add_tree_to_zip(
    archive: zipfile.ZipFile,
    source: Path,
    archive_root: Path,
    key_checkpoints: set[str] | None = None,
) -> tuple[int, int, list[str]]:
    files = 0
    size_bytes = 0
    skipped_weight_files = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        if not should_include_run_file(path, key_checkpoints):
            skipped_weight_files.append(path.relative_to(source).as_posix())
            continue
        relative = path.relative_to(source)
        archive_name = (archive_root / relative).as_posix()
        archive.write(path, archive_name, compress_type=zipfile.ZIP_STORED)
        files += 1
        size_bytes += path.stat().st_size
    return files, size_bytes, skipped_weight_files


def archive_run_to_zip(
    root: Path,
    run_dir: Path,
    destination: Path,
    key_checkpoints: set[str] | None = None,
) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    external_logs = find_external_run_logs(root, run_dir.name)
    with zipfile.ZipFile(
        destination,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        files, size_bytes, skipped_weight_files = add_tree_to_zip(
            archive,
            run_dir,
            Path("runs") / run_dir.name,
            key_checkpoints,
        )
        for source in external_logs:
            archive_name = (
                Path("external_logs") / run_dir.name / source.name
            ).as_posix()
            archive.write(source, archive_name, compress_type=zipfile.ZIP_STORED)
            files += 1
            size_bytes += source.stat().st_size
    return {
        "files": files,
        "source_size_bytes": size_bytes,
        "archive_size_bytes": destination.stat().st_size,
        "archive_format": "zip-store-zip64",
        "external_logs": [path.relative_to(root).as_posix() for path in external_logs],
        "included_key_checkpoints": sorted(key_checkpoints or []),
        "skipped_weight_files": skipped_weight_files,
    }


def archive_result_tree_to_zip(source: Path, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        destination,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        files, size_bytes, _ = add_tree_to_zip(
            archive,
            source,
            Path("results") / "raw_server_logs" / source.name,
        )
    return {
        "source": source.as_posix(),
        "package_path": destination.as_posix(),
        "files": files,
        "source_size_bytes": size_bytes,
        "archive_size_bytes": destination.stat().st_size,
        "archive_format": "zip-store-zip64",
    }


def write_sha256(package_dir: Path) -> None:
    checksum_path = package_dir / "SHA256SUMS"
    rows = []
    for path in sorted(package_dir.rglob("*")):
        if not path.is_file() or path == checksum_path:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        rows.append(f"{digest.hexdigest()}  {path.relative_to(package_dir).as_posix()}")
    checksum_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    package_dir = args.package_dir.resolve()
    ensure_empty_package_dir(package_dir, root, args.overwrite)

    model_card = root / "release" / "hf" / "README.md"
    if not model_card.is_file():
        raise FileNotFoundError(f"Missing HF model card template: {model_card}")
    shutil.copy2(model_card, package_dir / "README.md")
    artifact_index = root / "release" / "hf" / "SERVER_ARTIFACT_INDEX.md"
    if artifact_index.is_file():
        shutil.copy2(artifact_index, package_dir / "SERVER_ARTIFACT_INDEX.md")

    record_source = root / "release" / "experiment_records" / "20260713_downstream_multiseed"
    copy_tree_if_present(
        record_source,
        package_dir / "results" / "confirmed_multiseed",
        "confirmed result ledger",
    )

    environment_dir = package_dir / "code_environment"
    environment_dir.mkdir(parents=True, exist_ok=True)
    for name in ("environment.yml", "requirements.txt"):
        source = root / name
        if source.is_file():
            shutil.copy2(source, environment_dir / name)

    specs = list(MODEL_SPECS)
    if args.include_historical:
        specs.extend(HISTORICAL_MODEL_SPECS)

    manifest_models = []
    packaged_runs = {}
    required_missing = []
    for spec in specs:
        source = root / spec["source"]
        entry = dict(spec)
        entry["included"] = False
        if not source.is_file():
            message = f"Missing checkpoint: {source}"
            if spec["required"]:
                required_missing.append(message)
            else:
                print(f"WARNING: {message}", file=sys.stderr)
            manifest_models.append(entry)
            continue

        entry["included"] = True
        if not args.skip_model_only:
            destination = package_dir / spec["destination"]
            details = load_and_save_model_only(source, destination, spec)
            entry.update(details)
            entry["package_path"] = spec["destination"]
            entry["model_only_included"] = True
        else:
            entry["epoch"] = None
            entry["size_bytes"] = source.stat().st_size
            entry["model_only_included"] = False

        if args.include_full_checkpoints:
            full_destination = (
                package_dir
                / "training_checkpoints"
                / spec["tier"]
                / spec["id"]
                / source.name
            )
            copy_full_checkpoint(source, full_destination)
            entry["full_checkpoint_path"] = full_destination.relative_to(
                package_dir
            ).as_posix()

        if args.include_full_runs:
            run_dir = source.parent
            run_relative = run_dir.relative_to(root).as_posix()
            key_checkpoints = None
            if args.run_checkpoint_policy == "key":
                key_checkpoints = set(spec.get("key_checkpoints", [source.name]))
            if run_relative not in packaged_runs:
                print(f"Packaging complete run: {run_dir}")
                if args.full_run_archive_format == "zip":
                    run_destination = (
                        package_dir / "run_archives" / f"{run_dir.name}.zip"
                    )
                    run_details = archive_run_to_zip(
                        root,
                        run_dir,
                        run_destination,
                        key_checkpoints,
                    )
                    package_path = run_destination.relative_to(package_dir).as_posix()
                else:
                    run_destination = package_dir / run_relative
                    run_details = link_or_copy_tree(
                        run_dir,
                        run_destination,
                        key_checkpoints,
                    )
                    run_details["external_logs"] = collect_external_run_logs(
                        root, package_dir, run_dir.name
                    )
                    run_details["archive_format"] = "directory"
                    package_path = run_relative
                run_details["source"] = run_relative
                run_details["package_path"] = package_path
                packaged_runs[run_relative] = run_details
            entry["full_run_path"] = packaged_runs[run_relative]["package_path"]
        manifest_models.append(entry)

    if required_missing:
        raise FileNotFoundError("\n".join(required_missing))

    copied_raw_dirs = []
    if args.include_raw_logs:
        for relative in RAW_RESULT_DIRS:
            source = root / relative
            if not source.exists():
                print(f"WARNING: missing raw result directory: {source}", file=sys.stderr)
                continue
            if args.full_run_archive_format == "zip":
                destination = (
                    package_dir / "result_archives" / f"{source.name}.zip"
                )
                details = archive_result_tree_to_zip(source, destination)
                details["source"] = relative
                details["package_path"] = destination.relative_to(package_dir).as_posix()
                copied_raw_dirs.append(details)
            else:
                destination = (
                    package_dir / "results" / "raw_server_logs" / source.name
                )
                if copy_tree_if_present(source, destination, "raw result directory"):
                    copied_raw_dirs.append(
                        {
                            "source": relative,
                            "package_path": destination.relative_to(package_dir).as_posix(),
                            "archive_format": "directory",
                        }
                    )

    commit = git_commit(root)
    manifest = {
        "format_version": 1,
        "source_repository": "https://github.com/jiZhangji/Ph",
        "source_git_commit": commit,
        "package_policy": {
            "model_only_checkpoints": not args.skip_model_only,
            "full_training_checkpoints_included": args.include_full_checkpoints,
            "full_run_directories_included": args.include_full_runs,
            "full_run_archive_format": args.full_run_archive_format,
            "run_checkpoint_policy": args.run_checkpoint_policy,
            "raw_downstream_logs_included": args.include_raw_logs,
            "historical_models_included": args.include_historical,
            "datasets_included": False,
        },
        "models": manifest_models,
        "run_directories": list(packaged_runs.values()),
        "raw_result_directories": copied_raw_dirs,
        "comparability_warning": (
            "Only results marked strict official downstream are suitable for paper-facing "
            "controlled comparisons. Historical pipeline results are diagnostic only."
        ),
    }
    (package_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (package_dir / "SOURCE_GIT_COMMIT.txt").write_text(
        (commit or "unknown") + "\n", encoding="utf-8"
    )
    write_sha256(package_dir)

    included = [item for item in manifest_models if item["included"]]
    print()
    print(f"Package ready: {package_dir}")
    print(f"Included model entries: {len(included)}")
    for item in included:
        package_path = item.get("package_path") or item.get("full_run_path")
        print(f"  - {item['id']}: {package_path}")
    print("Verify checksums with: sha256sum -c SHA256SUMS")


if __name__ == "__main__":
    main()
