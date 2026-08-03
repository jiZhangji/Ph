#!/usr/bin/env python3
"""Report three-dataset attention heatmap export progress."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METHODS = ("mae", "lomar", "fg_mae", "i_jepa", "sar_jepa", "phyd_mae")
DATASETS = ("MSTAR_SOC", "New_FUSAR", "SAR_ACD")


def row_count(path):
    if not path.is_file():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=root / "paper_visualizations" / "attention_heatmaps_40shot",
    )
    parser.add_argument("--expected-per-job", type=int, default=50)
    args = parser.parse_args()
    output_root = args.root.resolve()

    complete = 0
    samples = 0
    print("dataset      method       status  samples")
    print("-------------------------------------------")
    for dataset in DATASETS:
        for method in METHODS:
            directory = output_root / "methods" / dataset / method
            marker = directory / "ATTENTION_EXPORT_COMPLETE.json"
            count = row_count(directory / "index.csv")
            done = marker.is_file() and count == args.expected_per_job
            complete += int(done)
            samples += count
            print(
                f"{dataset:<12} {method:<12} "
                f"{'done' if done else 'pending':<7} {count:>3}"
            )

    merged_marker = output_root / "MERGED_EXPORT_COMPLETE.json"
    merged_count = row_count(output_root / "merged_index.csv")
    print("-------------------------------------------")
    print(
        f"METHOD EXPORTS {complete}/18, "
        f"SAMPLES {samples}/{18 * args.expected_per_job}"
    )
    print(
        f"MERGED {'done' if merged_marker.is_file() else 'pending'} "
        f"{merged_count}/{len(DATASETS) * args.expected_per_job}"
    )
    if merged_marker.is_file():
        payload = json.loads(merged_marker.read_text(encoding="utf-8"))
        print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
