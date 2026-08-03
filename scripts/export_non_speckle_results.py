#!/usr/bin/env python3
"""Export complete non-speckle paper results and per-seed statistics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import statistics
import subprocess
import tarfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


METHODS = ("mae", "lomar", "fg_mae", "i_jepa", "sar_jepa")
DATASETS = ("MSTAR_SOC", "New_FUSAR", "SAR_ACD")
PROTOCOLS = ("MIM_finetune", "MIM_linear")
SHOTS = (10, 20, 40)
SEEDS = tuple(range(10))

ACCURACY_RE = re.compile(r"^\* accuracy:\s*([0-9.]+)%", re.MULTILINE)
MACRO_F1_RE = re.compile(r"^\* macro_f1:\s*([0-9.]+)%", re.MULTILINE)
SEED_RE = re.compile(r"^SEED:\s*(\d+)\s*$", re.MULTILINE)
CONFIG_RE = re.compile(r"^vit_b16_(\d+)shots$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def last_float(pattern: re.Pattern[str], text: str, label: str, log: Path) -> float:
    values = pattern.findall(text)
    if not values:
        raise ValueError(f"Missing {label} in {log}")
    return float(values[-1])


def parse_baseline_logs(root: Path) -> tuple[list[dict], list[tuple[Path, Path]]]:
    rows: list[dict] = []
    raw_logs: list[tuple[Path, Path]] = []
    seen = set()

    for log in sorted(root.rglob("log.txt")):
        relative = log.relative_to(root)
        if len(relative.parts) != 6:
            continue
        method, dataset, protocol, config, seed_dir, filename = relative.parts
        if filename != "log.txt" or method not in METHODS:
            continue
        config_match = CONFIG_RE.fullmatch(config)
        seed_match = re.fullmatch(r"seed(\d+)", seed_dir)
        if not config_match or not seed_match:
            continue

        shots = int(config_match.group(1))
        seed = int(seed_match.group(1))
        key = (method, dataset, protocol, shots, seed)
        if key in seen:
            raise ValueError(f"Duplicate baseline result: {key}")

        text = log.read_text(encoding="utf-8", errors="replace")
        logged_seeds = SEED_RE.findall(text)
        if not logged_seeds or int(logged_seeds[-1]) != seed:
            raise ValueError(f"Seed mismatch for {log}: path seed={seed}")

        rows.append(
            {
                "method": method,
                "dataset": dataset,
                "protocol": protocol,
                "shots": shots,
                "seed": seed,
                "accuracy": last_float(ACCURACY_RE, text, "accuracy", log),
                "macro_f1": last_float(MACRO_F1_RE, text, "macro_f1", log),
                "source_log": relative.as_posix(),
            }
        )
        raw_logs.append((log, Path("raw_logs") / relative))
        seen.add(key)

    expected = {
        (method, dataset, protocol, shots, seed)
        for method in METHODS
        for dataset in DATASETS
        for protocol in PROTOCOLS
        for shots in SHOTS
        for seed in SEEDS
    }
    actual = {
        (row["method"], row["dataset"], row["protocol"], row["shots"], row["seed"])
        for row in rows
    }
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected or len(rows) != 900:
        raise ValueError(
            "Baseline result set is incomplete or unexpected: "
            f"rows={len(rows)}, missing={len(missing)}, unexpected={len(unexpected)}; "
            f"first_missing={missing[:5]}, first_unexpected={unexpected[:5]}"
        )
    return sorted(
        rows,
        key=lambda row: (
            METHODS.index(row["method"]),
            DATASETS.index(row["dataset"]),
            PROTOCOLS.index(row["protocol"]),
            row["shots"],
            row["seed"],
        ),
    ), raw_logs


def aggregate(
    rows: list[dict], required_seeds: tuple[int, ...] | None = SEEDS
) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["dataset"], row["protocol"], row["shots"])].append(row)

    output = []
    for key in sorted(groups):
        values = sorted(groups[key], key=lambda row: row["seed"])
        seeds = [row["seed"] for row in values]
        if required_seeds is not None and seeds != list(required_seeds):
            raise ValueError(
                f"Expected seeds {list(required_seeds)} for {key}, got {seeds}"
            )
        accuracy = [row["accuracy"] for row in values]
        macro_f1 = [row["macro_f1"] for row in values]
        best = max(values, key=lambda row: (row["accuracy"], -row["seed"]))
        output.append(
            {
                "method": key[0],
                "dataset": key[1],
                "protocol": key[2],
                "shots": key[3],
                "n": len(values),
                "accuracy_mean": statistics.fmean(accuracy),
                "accuracy_std_sample": statistics.stdev(accuracy),
                "accuracy_std_population": statistics.pstdev(accuracy),
                "accuracy_median": statistics.median(accuracy),
                "accuracy_min": min(accuracy),
                "accuracy_max": max(accuracy),
                "best_accuracy_seed": best["seed"],
                "macro_f1_mean": statistics.fmean(macro_f1),
                "macro_f1_std_sample": statistics.stdev(macro_f1),
                "macro_f1_std_population": statistics.pstdev(macro_f1),
                "macro_f1_min": min(macro_f1),
                "macro_f1_max": max(macro_f1),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            formatted = {
                key: f"{value:.6f}" if isinstance(value, float) else value
                for key, value in row.items()
            }
            writer.writerow(formatted)


def copy_phyd_reference(project_root: Path, output_dir: Path) -> dict:
    source_root = (
        project_root
        / "release"
        / "experiment_records"
        / "20260713_downstream_multiseed"
    )
    source = source_root / "best_ckpt300_seed_results.csv"
    if not source.is_file():
        return {"included": False, "reason": f"missing {source}"}

    target_dir = output_dir / "phyd_main_reference"
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target_dir / source.name)
    for name in ("raw_best_ckpt300_official_stats.txt", "README.md"):
        candidate = source_root / name
        if candidate.is_file():
            shutil.copy2(candidate, target_dir / name)

    with source.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    normalized = [
        {
            "method": "phyd_mae",
            "dataset": row["dataset"],
            "protocol": f"MIM_{row['protocol']}",
            "shots": int(row["shots"]),
            "seed": int(row["seed"]),
            "accuracy": float(row["accuracy"]),
            "macro_f1": float(row["macro_f1"]),
            "source_log": "historical strict-official result ledger",
        }
        for row in rows
    ]
    write_csv(
        target_dir / "best_ckpt300_all_available_statistics.csv",
        aggregate(normalized, required_seeds=None),
    )
    first_ten = [row for row in normalized if row["seed"] in SEEDS]
    if first_ten:
        write_csv(target_dir / "best_ckpt300_first10_available.csv", first_ten)
        write_csv(
            target_dir / "best_ckpt300_first10_available_statistics.csv",
            aggregate(first_ten, required_seeds=None),
        )
    return {
        "included": True,
        "all_available_rows": len(rows),
        "first10_available_rows": len(first_ten),
        "note": "PhyD-MAE records use their historical strict-official downstream protocol and are not merged into the 900-row baseline statistics.",
    }


def git_revision(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    project_root = args.root.resolve()
    baseline_root = (
        project_root
        / "few_shot_classification"
        / "finetune"
        / "output_paper_baselines_10seeds"
    )
    output_dir = args.output_dir.resolve()
    allowed_root = (project_root / "hf_release").resolve()
    if not output_dir.is_relative_to(allowed_root) or output_dir == allowed_root:
        raise ValueError(
            f"Output directory must be a child of {allowed_root}, got {output_dir}"
        )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    rows, raw_logs = parse_baseline_logs(baseline_root)
    stats = aggregate(rows)
    write_csv(output_dir / "baseline_all_900_seed_results.csv", rows)
    write_csv(output_dir / "baseline_90_group_statistics.csv", stats)

    archive = output_dir / "baseline_raw_900_logs.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for source, target in raw_logs:
            tar.add(source, arcname=target.as_posix(), recursive=False)

    phyd_reference = copy_phyd_reference(project_root, output_dir)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(project_root),
        "scope": "non-speckle downstream classification only",
        "excluded": "controlled-speckle/L_add results",
        "baseline_root": str(baseline_root),
        "baseline_methods": list(METHODS),
        "datasets": list(DATASETS),
        "protocols": list(PROTOCOLS),
        "shots": list(SHOTS),
        "seeds": list(SEEDS),
        "per_seed_rows": len(rows),
        "aggregate_groups": len(stats),
        "raw_logs": len(raw_logs),
        "phyd_main_reference": phyd_reference,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        """---
license: other
pretty_name: PhyD-MAE Non-Speckle Paper Results
---

# PhyD-MAE non-speckle downstream results

This package contains the complete ordinary downstream evaluation for MAE,
LoMaR, FG-MAE, I-JEPA, and SAR-JEPA: three datasets, fine-tuning and linear
probing, 10/20/40 shots, and seeds 0-9. Controlled-speckle (`L_add`) results
are intentionally excluded.

- `baseline_all_900_seed_results.csv`: every per-seed Accuracy and Macro-F1.
- `baseline_90_group_statistics.csv`: mean, sample/population standard
  deviation, median, range, maximum, and best seed for every configuration.
- `baseline_raw_900_logs.tar.gz`: the corresponding 900 raw `log.txt` files.
- `phyd_main_reference/`: all available historical per-seed records for the
  main PhyD-MAE checkpoint, kept separate because that evaluation used its
  established strict-official downstream protocol.
- `manifest.json`: provenance and completeness metadata.

All metric values are percentages.
""",
        encoding="utf-8",
    )

    checksums = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            checksums.append(f"{sha256(path)}  {path.relative_to(output_dir).as_posix()}")
    (output_dir / "SHA256SUMS").write_text(
        "\n".join(checksums) + "\n", encoding="ascii"
    )

    print(f"Validated baseline rows: {len(rows)}/900")
    print(f"Aggregate groups: {len(stats)}/90")
    print(f"Raw logs archived: {len(raw_logs)}/900")
    print(f"PhyD-MAE reference: {phyd_reference}")
    print(f"Package: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
