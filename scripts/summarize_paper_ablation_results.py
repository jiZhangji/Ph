#!/usr/bin/env python3
import argparse
import csv
import re
import statistics
from pathlib import Path


ACCURACY_RE = re.compile(r"^\* accuracy:\s*([0-9.]+)%", re.MULTILINE)
MACRO_F1_RE = re.compile(r"^\* macro_f1:\s*([0-9.]+)%", re.MULTILINE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize the resumable paper-ablation downstream matrix."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def parse_metric(log_path, pattern):
    if not log_path.is_file():
        return None
    matches = pattern.findall(log_path.read_text(encoding="utf-8", errors="replace"))
    return float(matches[-1]) if matches else None


def trainer_name(protocol):
    return {
        "finetune": "MIM_finetune",
        "MIM_finetune": "MIM_finetune",
        "linear": "MIM_linear",
        "MIM_linear": "MIM_linear",
    }.get(protocol, protocol)


def mean_std(values):
    if not values:
        return "", ""
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mean:.2f}", f"{std:.2f}"


def main():
    args = parse_args()
    root = args.root.resolve()
    detail_rows = []
    aggregate_rows = []
    expected = 0
    completed = 0

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        specs = list(csv.DictReader(handle, delimiter="\t"))
    if not specs:
        raise ValueError(f"No experiment rows found in manifest: {args.manifest}")

    for spec in specs:
        model = spec["model"]
        for dataset in spec["datasets"].split():
            for protocol in spec["protocols"].split():
                trainer = trainer_name(protocol)
                for shots in spec["shots"].split():
                    group_rows = []
                    for seed in spec["seeds"].split():
                        expected += 1
                        log_path = (
                            root
                            / model
                            / dataset
                            / trainer
                            / f"vit_b16_{shots}shots"
                            / f"seed{seed}"
                            / "log.txt"
                        )
                        accuracy = parse_metric(log_path, ACCURACY_RE)
                        macro_f1 = parse_metric(log_path, MACRO_F1_RE)
                        status = "completed" if accuracy is not None else "missing"
                        if accuracy is not None:
                            completed += 1
                            group_rows.append((accuracy, macro_f1))
                        detail_rows.append(
                            {
                                "model": model,
                                "checkpoint": spec["checkpoint"],
                                "dataset": dataset,
                                "protocol": trainer,
                                "shots": shots,
                                "seed": seed,
                                "accuracy": "" if accuracy is None else f"{accuracy:.2f}",
                                "macro_f1": "" if macro_f1 is None else f"{macro_f1:.2f}",
                                "status": status,
                                "log": str(log_path),
                            }
                        )

                    accuracies = [row[0] for row in group_rows]
                    macro_f1_values = [row[1] for row in group_rows if row[1] is not None]
                    acc_mean, acc_std = mean_std(accuracies)
                    f1_mean, f1_std = mean_std(macro_f1_values)
                    aggregate_rows.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "protocol": trainer,
                            "shots": shots,
                            "completed_seeds": len(accuracies),
                            "expected_seeds": len(spec["seeds"].split()),
                            "accuracy_mean": acc_mean,
                            "accuracy_sample_std": acc_std,
                            "macro_f1_mean": f1_mean,
                            "macro_f1_sample_std": f1_std,
                        }
                    )

    detail_path = root / "results_per_seed.csv"
    aggregate_path = root / "results_mean_std.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_rows[0].keys())
        writer.writeheader()
        writer.writerows(detail_rows)
    with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_rows[0].keys())
        writer.writeheader()
        writer.writerows(aggregate_rows)

    print(f"Completed downstream runs: {completed}/{expected}")
    print(f"Per-seed CSV: {detail_path}")
    print(f"Mean/std CSV: {aggregate_path}")
    if completed != expected:
        print("Rerun the shell script; completed log files will be skipped.")


if __name__ == "__main__":
    main()
