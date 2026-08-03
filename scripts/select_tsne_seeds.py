#!/usr/bin/env python3
"""Select fixed, representative, or best downstream seeds for t-SNE runs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


METHODS = ("mae", "lomar", "fg_mae", "i_jepa", "sar_jepa")
DATASETS = ("MSTAR_SOC", "New_FUSAR", "SAR_ACD")
PROTOCOLS = ("MIM_finetune", "MIM_linear")
SHOTS = (10, 20, 40)
SEEDS = set(range(10))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--policy", choices=("best", "representative", "fixed"), default="representative"
    )
    parser.add_argument("--fixed-seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("logs/tsne_selected_seeds.csv")
    )
    return parser.parse_args()


def baseline_rows(root):
    result_root = (
        root
        / "few_shot_classification"
        / "finetune"
        / "output_paper_baselines_10seeds"
    )
    pattern = re.compile(
        r"^(?P<method>[^/]+)/(?P<dataset>[^/]+)/(?P<protocol>MIM_[^/]+)/"
        r"vit_b16_(?P<shots>\d+)shots/seed(?P<seed>\d+)/log\.txt$"
    )
    accuracy_pattern = re.compile(r"^\* accuracy:\s*([0-9.]+)%", re.MULTILINE)
    rows = []
    for log in result_root.rglob("log.txt"):
        relative = log.relative_to(result_root).as_posix()
        match = pattern.match(relative)
        if not match:
            continue
        values = accuracy_pattern.findall(log.read_text(errors="ignore"))
        if not values:
            continue
        record = match.groupdict()
        seed = int(record["seed"])
        if seed not in SEEDS:
            continue
        rows.append(
            {
                "method": record["method"],
                "dataset": record["dataset"],
                "protocol": record["protocol"],
                "shots": int(record["shots"]),
                "seed": seed,
                "accuracy": float(values[-1]),
                "source": relative,
            }
        )
    return rows


def phyd_rows(root):
    source = (
        root
        / "release"
        / "experiment_records"
        / "20260713_downstream_multiseed"
        / "best_ckpt300_seed_results.csv"
    )
    rows = []
    with source.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            seed = int(row["seed"])
            if seed not in SEEDS:
                continue
            protocol = {
                "finetune": "MIM_finetune",
                "linear": "MIM_linear",
            }.get(row["protocol"], row["protocol"])
            rows.append(
                {
                    "method": "phyd_mae",
                    "dataset": row["dataset"],
                    "protocol": protocol,
                    "shots": int(row["shots"]),
                    "seed": seed,
                    "accuracy": float(row["accuracy"]),
                    "source": source.relative_to(root).as_posix(),
                }
            )
    return rows


def select(rows, policy, fixed_seed):
    groups = {}
    for row in rows:
        key = (row["method"], row["dataset"], row["protocol"], row["shots"])
        groups.setdefault(key, []).append(row)

    expected = {
        (method, dataset, protocol, shots)
        for method in (*METHODS, "phyd_mae")
        for dataset in DATASETS
        for protocol in PROTOCOLS
        for shots in SHOTS
    }
    missing = sorted(expected - set(groups))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} result groups; first={missing[:5]}")

    selected = []
    for key in sorted(expected):
        candidates = sorted(groups[key], key=lambda row: row["seed"])
        if policy == "best":
            chosen = max(candidates, key=lambda row: (row["accuracy"], -row["seed"]))
        elif policy == "representative":
            mean = sum(row["accuracy"] for row in candidates) / len(candidates)
            chosen = min(
                candidates,
                key=lambda row: (abs(row["accuracy"] - mean), row["seed"]),
            )
        else:
            matches = [row for row in candidates if row["seed"] == fixed_seed]
            if not matches:
                raise RuntimeError(f"Fixed seed {fixed_seed} unavailable for {key}")
            chosen = matches[0]
        selected.append({**chosen, "policy": policy, "available_seeds": len(candidates)})
    return selected


def main():
    args = parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    rows = baseline_rows(root) + phyd_rows(root)
    selected = select(rows, args.policy, args.fixed_seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "method",
        "dataset",
        "protocol",
        "shots",
        "seed",
        "accuracy",
        "policy",
        "available_seeds",
        "source",
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    print(f"Selected groups: {len(selected)}/108")
    print(f"Policy: {args.policy}")
    print(f"Manifest: {output}")


if __name__ == "__main__":
    main()
