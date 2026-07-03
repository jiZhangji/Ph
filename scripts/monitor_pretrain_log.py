#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


METRICS = (
    "train_loss",
    "train_loss_grad",
    "train_loss_lfst",
    "train_loss_grad_weighted",
    "train_loss_lfst_weighted",
)


def read_rows(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def mean(values):
    return sum(values) / max(len(values), 1)


def summarize(rows, window):
    if not rows:
        print("No JSON training rows found yet.")
        return

    recent = rows[-window:]
    previous = rows[-2 * window:-window]
    first = rows[0]
    last = rows[-1]

    print(f"epochs: {first.get('epoch')} -> {last.get('epoch')}  rows={len(rows)}")
    for metric in METRICS:
        values = [row[metric] for row in recent if metric in row]
        if not values:
            continue
        recent_mean = mean(values)
        first_value = first.get(metric)
        last_value = last.get(metric)
        msg = f"{metric}: first={first_value:.6f} last={last_value:.6f} recent{len(values)}_mean={recent_mean:.6f}"
        prev_values = [row[metric] for row in previous if metric in row]
        if prev_values:
            prev_mean = mean(prev_values)
            msg += f" delta_vs_prev={recent_mean - prev_mean:+.6f}"
        print(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path, help="path to pretraining output log.txt")
    parser.add_argument("--window", default=5, type=int, help="recent epoch window")
    args = parser.parse_args()

    if not args.log.exists():
        raise FileNotFoundError(args.log)
    summarize(read_rows(args.log), args.window)


if __name__ == "__main__":
    main()
