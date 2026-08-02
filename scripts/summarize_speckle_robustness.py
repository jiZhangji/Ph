#!/usr/bin/env python3
"""Aggregate controlled-speckle results across noise and downstream seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


PREFIX = "SPECKLE_RESULT "


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-downstream-seeds", type=int, default=5)
    parser.add_argument("--expected-noise-seeds", type=int, default=3)
    return parser.parse_args()


def load_records(root):
    records = []
    for log in root.rglob("log.txt"):
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(PREFIX):
                record = json.loads(line[len(PREFIX):])
                record["log"] = str(log)
                records.append(record)
    return records


def metric(record, name):
    metrics = record["metrics"]
    if name in metrics:
        return float(metrics[name])
    aliases = {
        "accuracy": ("acc",),
        "macro_f1": ("macro-f1", "f1_macro"),
    }
    for alias in aliases.get(name, ()):
        if alias in metrics:
            return float(metrics[alias])
    return None


def main():
    args = parse_args()
    records = load_records(args.root)
    if not records:
        raise SystemExit(f"No {PREFIX.strip()} records found under {args.root}")

    per_downstream_seed = defaultdict(list)
    noise_seed_sets = defaultdict(set)
    seen_evaluations = set()
    for record in records:
        key = (
            record["method"],
            record["dataset"],
            record["protocol"],
            int(record["shots"]),
            str(record["L_add"]),
            int(record["downstream_seed"]),
        )
        accuracy = metric(record, "accuracy")
        if accuracy is None:
            raise ValueError(f"Missing accuracy in {record['log']}")
        noise_seed = record["noise_seed"]
        evaluation_key = (*key, None if noise_seed is None else int(noise_seed))
        if evaluation_key in seen_evaluations:
            raise ValueError(
                "Duplicate robustness evaluation for "
                f"{evaluation_key}; remove stale duplicate logs"
            )
        seen_evaluations.add(evaluation_key)
        per_downstream_seed[key].append(accuracy)
        if noise_seed is not None:
            noise_seed_sets[key].add(int(noise_seed))

    aggregated_noise = {
        key: statistics.fmean(values)
        for key, values in per_downstream_seed.items()
    }
    groups = defaultdict(list)
    for key, value in aggregated_noise.items():
        groups[key[:-1]].append((key[-1], value))

    clean_means = {}
    for key, seed_values in groups.items():
        method, dataset, protocol, shots, looks = key
        if looks == "clean":
            clean_means[(method, dataset, protocol, shots)] = statistics.fmean(
                value for _, value in seed_values
            )

    rows = []
    incomplete = False
    for key in sorted(groups):
        method, dataset, protocol, shots, looks = key
        seed_values = sorted(groups[key])
        values = [value for _, value in seed_values]
        clean_mean = clean_means.get((method, dataset, protocol, shots))
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        drop = None if clean_mean is None else clean_mean - mean
        retention = None if not clean_mean else 100.0 * mean / clean_mean
        expected_noise = 1 if looks == "clean" else args.expected_noise_seeds
        noise_counts = [
            len(noise_seed_sets[(*key, downstream_seed)])
            if looks != "clean" else 1
            for downstream_seed, _ in seed_values
        ]
        complete = (
            len(seed_values) == args.expected_downstream_seeds
            and all(count == expected_noise for count in noise_counts)
        )
        incomplete |= not complete
        rows.append(
            {
                "method": method,
                "dataset": dataset,
                "protocol": protocol,
                "shots": shots,
                "L_add": looks,
                "downstream_seeds": len(seed_values),
                "noise_records": sum(len(per_downstream_seed[(*key, seed)]) for seed, _ in seed_values),
                "accuracy_mean": f"{mean:.4f}",
                "accuracy_std": f"{std:.4f}",
                "drop_from_clean": "" if drop is None else f"{drop:.4f}",
                "retention_percent": "" if retention is None else f"{retention:.4f}",
                "complete": complete,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"{'method':12} {'dataset':12} {'protocol':13} {'shot':>4} "
        f"{'L_add':>6} {'accuracy':>15} {'drop':>8} {'keep':>8}"
    )
    for row in rows:
        accuracy = f"{float(row['accuracy_mean']):.2f} +/- {float(row['accuracy_std']):.2f}"
        drop = "-" if not row["drop_from_clean"] else f"{float(row['drop_from_clean']):.2f}"
        keep = "-" if not row["retention_percent"] else f"{float(row['retention_percent']):.1f}%"
        print(
            f"{row['method']:12} {row['dataset']:12} {row['protocol']:13} "
            f"{row['shots']:4} {row['L_add']:>6} {accuracy:>15} {drop:>8} {keep:>8}"
        )
    print(f"CSV saved to: {args.output}")
    if incomplete:
        raise SystemExit("Some robustness groups are incomplete; see complete=false in the CSV")


if __name__ == "__main__":
    main()
