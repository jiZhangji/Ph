#!/usr/bin/env python3
"""Report progress for the sharded paper baseline and speckle evaluations."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


METHODS = ("mae", "lomar", "fg_mae", "i_jepa", "sar_jepa")
SPECKLE_METHODS = METHODS + ("phyd_mae",)
BASELINE_EXPECTED_PER_METHOD = 3 * 2 * 3 * 10
SPECKLE_EXPECTED_PER_METHOD = 10
SPECKLE_RESULTS_PER_LOG = 1 + 4 * 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--baseline-only", action="store_true")
    return parser.parse_args()


def valid_baseline_logs(root: Path) -> int:
    count = 0
    for log in root.rglob("log.txt") if root.is_dir() else ():
        text = log.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"^\* accuracy:", text, re.MULTILINE) and re.search(
            r"^SEED:\s*\d+\s*$", text, re.MULTILINE
        ):
            count += 1
    return count


def valid_speckle_logs(root: Path) -> tuple[int, int]:
    completed = 0
    results = 0
    marker = re.compile(
        rf"SPECKLE_ROBUSTNESS_COMPLETE completed={SPECKLE_RESULTS_PER_LOG} "
        rf"expected={SPECKLE_RESULTS_PER_LOG}"
    )
    for log in root.rglob("log.txt") if root.is_dir() else ():
        text = log.read_text(encoding="utf-8", errors="ignore")
        if marker.search(text):
            completed += 1
            results += len(re.findall(r"^SPECKLE_RESULT ", text, re.MULTILINE))
    return completed, results


def main() -> int:
    args = parse_args()
    baseline_root = (
        args.root
        / "few_shot_classification"
        / "finetune"
        / "output_paper_baselines_10seeds"
    )
    speckle_root = (
        args.root
        / "few_shot_classification"
        / "finetune"
        / "output_speckle_robustness_10seeds"
    )

    baseline_total = 0
    print("Baseline progress")
    print(f"{'method':12} {'complete':>10} {'remain':>8}")
    print("-" * 34)
    for method in METHODS:
        complete = valid_baseline_logs(baseline_root / method)
        baseline_total += complete
        print(
            f"{method:12} {complete:3}/{BASELINE_EXPECTED_PER_METHOD:<3} "
            f"{BASELINE_EXPECTED_PER_METHOD - complete:8}"
        )
    baseline_expected = BASELINE_EXPECTED_PER_METHOD * len(METHODS)
    print(
        f"TOTAL        {baseline_total:3}/{baseline_expected:<3} "
        f"{baseline_expected - baseline_total:8} "
        f"({100 * baseline_total / baseline_expected:.1f}%)"
    )

    if args.baseline_only:
        return 0

    print("\nControlled-speckle progress")
    print(f"{'method':12} {'jobs':>10} {'tests':>10}")
    print("-" * 36)
    speckle_jobs = 0
    speckle_tests = 0
    for method in SPECKLE_METHODS:
        jobs, tests = valid_speckle_logs(speckle_root / method)
        speckle_jobs += jobs
        speckle_tests += tests
        print(
            f"{method:12} {jobs:3}/{SPECKLE_EXPECTED_PER_METHOD:<3} "
            f"{tests:4}/{SPECKLE_EXPECTED_PER_METHOD * SPECKLE_RESULTS_PER_LOG:<4}"
        )
    expected_jobs = SPECKLE_EXPECTED_PER_METHOD * len(SPECKLE_METHODS)
    expected_tests = expected_jobs * SPECKLE_RESULTS_PER_LOG
    print(
        f"TOTAL        {speckle_jobs:3}/{expected_jobs:<3} "
        f"{speckle_tests:4}/{expected_tests:<4}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
