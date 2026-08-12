#!/usr/bin/env python3
import argparse
import csv
import re
import statistics
from pathlib import Path


ACCURACY_RE = re.compile(r"^\* accuracy:\s*([0-9.]+)%", re.MULTILINE)
MACRO_F1_RE = re.compile(r"^\* macro_f1:\s*([0-9.]+)%", re.MULTILINE)
SEED_RE = re.compile(r"^SEED:\s*(-?[0-9]+)\s*$", re.MULTILINE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize the FUSAR-Ship LFST-loss-weight sensitivity study."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def trainer_name(protocol):
    return {
        "finetune": "MIM_finetune",
        "MIM_finetune": "MIM_finetune",
        "linear": "MIM_linear",
        "MIM_linear": "MIM_linear",
    }.get(protocol, protocol)


def parse_result(log_path, expected_seed):
    if not log_path.is_file():
        return None, None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    configured_seeds = [int(value) for value in SEED_RE.findall(text)]
    if not configured_seeds or configured_seeds[-1] != expected_seed:
        return None, None
    accuracy = ACCURACY_RE.findall(text)
    macro_f1 = MACRO_F1_RE.findall(text)
    return (
        float(accuracy[-1]) if accuracy else None,
        float(macro_f1[-1]) if macro_f1 else None,
    )


def stats(values):
    if not values:
        return "", "", ""
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{statistics.fmean(values):.2f}", f"{std:.2f}", f"{max(values):.2f}"


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    root = args.root.resolve()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        specs = list(csv.DictReader(handle, delimiter="\t"))
    if not specs:
        raise ValueError(f"No experiments found in {args.manifest}")

    detail_rows = []
    summary_rows = []
    expected = 0
    completed = 0

    for spec in specs:
        for dataset in spec["datasets"].split():
            for protocol in spec["protocols"].split():
                trainer = trainer_name(protocol)
                for shots in spec["shots"].split():
                    group = []
                    for seed_text in spec["seeds"].split():
                        seed = int(seed_text)
                        expected += 1
                        log_path = (
                            root
                            / spec["model"]
                            / dataset
                            / trainer
                            / f"vit_b16_{shots}shots"
                            / f"seed{seed}"
                            / "log.txt"
                        )
                        accuracy, macro_f1 = parse_result(log_path, seed)
                        status = "completed" if accuracy is not None else "missing"
                        if accuracy is not None:
                            completed += 1
                            group.append((accuracy, macro_f1))
                        detail_rows.append(
                            {
                                "lfst_weight": spec["lfst_weight"],
                                "model": spec["model"],
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

                    accuracies = [item[0] for item in group]
                    f1_values = [item[1] for item in group if item[1] is not None]
                    acc_mean, acc_std, acc_max = stats(accuracies)
                    f1_mean, f1_std, f1_max = stats(f1_values)
                    summary_rows.append(
                        {
                            "lfst_weight": spec["lfst_weight"],
                            "dataset": dataset,
                            "protocol": trainer,
                            "shots": shots,
                            "completed_seeds": len(accuracies),
                            "expected_seeds": len(spec["seeds"].split()),
                            "accuracy_mean": acc_mean,
                            "accuracy_sample_std": acc_std,
                            "accuracy_max": acc_max,
                            "macro_f1_mean": f1_mean,
                            "macro_f1_sample_std": f1_std,
                            "macro_f1_max": f1_max,
                        }
                    )

    detail_rows.sort(
        key=lambda row: (
            float(row["lfst_weight"]),
            row["protocol"],
            int(row["shots"]),
            int(row["seed"]),
        )
    )
    summary_rows.sort(
        key=lambda row: (
            float(row["lfst_weight"]),
            row["protocol"],
            int(row["shots"]),
        )
    )
    write_csv(root / "results_per_seed.csv", detail_rows)
    write_csv(root / "results_mean_std_max.csv", summary_rows)

    lines = [
        "LFST Loss-Weight Sensitivity on FUSAR-Ship",
        "=" * 76,
        f"Completed downstream runs: {completed}/{expected}",
        "Values are mean +/- sample standard deviation; max is also reported.",
        "",
        (
            f"{'lambda':>7}  {'protocol':<12} {'shot':>5} {'n':>5} "
            f"{'accuracy':>17} {'max':>7} {'macro-F1':>17} {'max':>7}"
        ),
        "-" * 86,
    ]
    for row in summary_rows:
        accuracy = "missing"
        macro_f1 = "missing"
        if row["accuracy_mean"]:
            accuracy = f"{row['accuracy_mean']} +/- {row['accuracy_sample_std']}"
        if row["macro_f1_mean"]:
            macro_f1 = f"{row['macro_f1_mean']} +/- {row['macro_f1_sample_std']}"
        lines.append(
            f"{float(row['lfst_weight']):7.2f}  {row['protocol']:<12} "
            f"{int(row['shots']):5d} "
            f"{int(row['completed_seeds']):2d}/{int(row['expected_seeds']):<2d} "
            f"{accuracy:>17} {row['accuracy_max'] or '-':>7} "
            f"{macro_f1:>17} {row['macro_f1_max'] or '-':>7}"
        )
    (root / "results_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Completed downstream runs: {completed}/{expected}")
    print(f"Per-seed CSV: {root / 'results_per_seed.csv'}")
    print(f"Mean/std/max CSV: {root / 'results_mean_std_max.csv'}")
    print(f"Readable summary: {root / 'results_summary.txt'}")
    if completed != expected:
        print("Rerun ACTION=eval; completed logs will be skipped.")


if __name__ == "__main__":
    main()
