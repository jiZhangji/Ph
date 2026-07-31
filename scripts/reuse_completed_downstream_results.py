#!/usr/bin/env python3
import argparse
import csv
import re
import shutil
from pathlib import Path


ACCURACY_RE = re.compile(r"^\* accuracy:\s*[0-9.]+%", re.MULTILINE)
CHECKPOINT_RE = re.compile(r"^Loaded checkpoint:\s*(.+?)\s*$", re.MULTILINE)
LR_RE = re.compile(r"^\s+LR:\s*([0-9.eE+-]+)\s*$", re.MULTILINE)
EPOCH_RE = re.compile(r"^\s+MAX_EPOCH:\s*([0-9]+)\s*$", re.MULTILINE)
BATCH_RE = re.compile(r"^\s+BATCH_SIZE:\s*([0-9]+)\s*$", re.MULTILINE)
SFAFM_RE = re.compile(r"^Use downstream SFAFM:\s*(.+?)\s*$", re.MULTILINE)
POOL_RE = re.compile(r"^Downstream feature pool:\s*(.+?)\s*$", re.MULTILINE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reuse completed downstream logs that match an ablation manifest."
    )
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    return parser.parse_args()


def checkpoint_key(path):
    normalized = str(path).replace("\\", "/").strip()
    marker = "/runs/"
    if marker in normalized:
        return normalized.split(marker, 1)[1]
    if normalized.startswith("runs/"):
        return normalized[5:]
    return normalized


def trainer_name(protocol):
    return {
        "finetune": "MIM_finetune",
        "MIM_finetune": "MIM_finetune",
        "linear": "MIM_linear",
        "MIM_linear": "MIM_linear",
    }.get(protocol, protocol)


def matches_configuration(text, lr, epochs, batch_size):
    lr_values = [float(value) for value in LR_RE.findall(text)]
    epoch_values = [int(value) for value in EPOCH_RE.findall(text)]
    batch_values = [int(value) for value in BATCH_RE.findall(text)]
    sfafm_values = [value.strip().lower() for value in SFAFM_RE.findall(text)]
    pool_values = [value.strip().lower() for value in POOL_RE.findall(text)]
    if lr_values and not any(abs(value - lr) <= 1e-12 for value in lr_values):
        return False
    if epoch_values and epochs not in epoch_values:
        return False
    if batch_values and batch_size not in batch_values:
        return False
    if sfafm_values and sfafm_values[-1] not in {"false", "0"}:
        return False
    if pool_values and pool_values[-1] != "cls":
        return False
    return True


def load_candidates(search_root, output_root, lr, epochs, batch_size):
    index = {}
    output_root = output_root.resolve()
    for log_path in search_root.rglob("log.txt"):
        resolved = log_path.resolve()
        if resolved == output_root or output_root in resolved.parents:
            continue
        parts = log_path.parts
        if len(parts) < 5 or not parts[-2].startswith("seed"):
            continue
        dataset, protocol, config, seed = parts[-5:-1]
        if not config.startswith("vit_b16_") or not config.endswith("shots"):
            continue
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if not ACCURACY_RE.search(text):
            continue
        if not matches_configuration(text, lr, epochs, batch_size):
            continue
        checkpoint_matches = CHECKPOINT_RE.findall(text)
        if not checkpoint_matches:
            continue
        key = (
            checkpoint_key(checkpoint_matches[-1]),
            dataset,
            protocol,
            config,
            seed,
        )
        index.setdefault(key, []).append(log_path)
    return index


def main():
    args = parse_args()
    search_root = args.search_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        specs = list(csv.DictReader(handle, delimiter="\t"))

    candidates = load_candidates(
        search_root, output_root, args.lr, args.epochs, args.batch_size
    )
    reused_rows = []
    for spec in specs:
        checkpoint = checkpoint_key(spec["checkpoint"])
        for dataset in spec["datasets"].split():
            for protocol in spec["protocols"].split():
                trainer = trainer_name(protocol)
                for shots in spec["shots"].split():
                    config = f"vit_b16_{shots}shots"
                    for seed_number in spec["seeds"].split():
                        seed = f"seed{seed_number}"
                        destination = (
                            output_root
                            / spec["model"]
                            / dataset
                            / trainer
                            / config
                            / seed
                            / "log.txt"
                        )
                        if destination.is_file() and ACCURACY_RE.search(
                            destination.read_text(encoding="utf-8", errors="replace")
                        ):
                            continue
                        matches = candidates.get(
                            (checkpoint, dataset, trainer, config, seed), []
                        )
                        if not matches:
                            continue
                        source = max(matches, key=lambda path: path.stat().st_mtime)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, destination)
                        reused_rows.append(
                            {
                                "model": spec["model"],
                                "dataset": dataset,
                                "protocol": trainer,
                                "shots": shots,
                                "seed": seed_number,
                                "source_log": str(source),
                                "destination_log": str(destination),
                            }
                        )

    provenance_path = output_root / "reused_results.csv"
    fields = (
        "model",
        "dataset",
        "protocol",
        "shots",
        "seed",
        "source_log",
        "destination_log",
    )
    with provenance_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(reused_rows)
    print(f"Reused completed downstream logs: {len(reused_rows)}")
    print(f"Reuse provenance: {provenance_path}")


if __name__ == "__main__":
    main()
