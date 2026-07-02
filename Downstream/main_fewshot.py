from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

REPO_ROOT = Path(__file__).resolve().parents[1]
PRETRAINING_DIR = REPO_ROOT / "Pretraining"
sys.path.insert(0, str(PRETRAINING_DIR))

import models_lomar  # noqa: E402
from fewshot_dataset import (  # noqa: E402
    SARClassificationDataset,
    build_fewshot_split,
    resolve_dataset_dir,
)


class PhyDClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, use_sfafm: bool = True):
        super().__init__()
        self.backbone = backbone
        self.use_sfafm = use_sfafm
        self.head = nn.Linear(backbone.cls_token.shape[-1], num_classes)

    def forward(self, x):
        features = self.backbone.forward_features(x, use_sfafm=self.use_sfafm)
        return self.head(features)


FEATURE_PARAM_PREFIXES = (
    "patch_embed.",
    "cls_token",
    "pos_embed",
    "blocks.",
    "norm.",
    "img_SFAFM_process.",
)

LATEST_CHECKPOINT_ALIASES = {"latest", "auto", "final"}
BEST_LOSS_CHECKPOINT_ALIASES = {"best", "best-loss", "best_loss", "best-train-loss", "best_train_loss"}


def checkpoint_epoch(path: Path) -> int | None:
    match = re.match(r"checkpoint-(\d+)\.pth$", path.name)
    if match is None:
        return None
    return int(match.group(1))


def list_epoch_checkpoints(checkpoint_root: Path) -> dict[int, Path]:
    checkpoint_root = Path(checkpoint_root)
    if not checkpoint_root.is_dir():
        return {}
    checkpoints: dict[int, Path] = {}
    for path in checkpoint_root.glob("checkpoint-*.pth"):
        epoch = checkpoint_epoch(path)
        if epoch is not None:
            checkpoints[epoch] = path
    return checkpoints


def read_pretrain_losses(checkpoint_root: Path) -> dict[int, float]:
    log_path = Path(checkpoint_root) / "log.txt"
    if not log_path.is_file():
        return {}

    losses: dict[int, float] = {}
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            epoch = record.get("epoch")
            loss = record.get("train_loss")
            if isinstance(epoch, int) and isinstance(loss, (float, int)):
                losses[epoch] = float(loss)
    return losses


def resolve_checkpoint(checkpoint: str | None, checkpoint_root: Path, allow_random_init: bool) -> Path | None:
    if checkpoint is None or checkpoint.strip() == "":
        if allow_random_init:
            return None
        raise ValueError("A pretrained checkpoint is required unless --allow_random_init is set.")

    checkpoint_value = checkpoint.strip()
    normalized = checkpoint_value.lower()
    if normalized in {"none", "random"}:
        if allow_random_init:
            return None
        raise ValueError("Random initialization requested, but --allow_random_init is not set.")

    checkpoint_path = Path(checkpoint_value)
    if checkpoint_path.is_file():
        return checkpoint_path

    checkpoints = list_epoch_checkpoints(checkpoint_root)
    if normalized in LATEST_CHECKPOINT_ALIASES:
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint-*.pth files found under: {checkpoint_root}")
        latest_epoch = max(checkpoints)
        return checkpoints[latest_epoch]

    if normalized in BEST_LOSS_CHECKPOINT_ALIASES:
        explicit_best = Path(checkpoint_root) / "checkpoint-best.pth"
        if explicit_best.is_file():
            return explicit_best

        losses = read_pretrain_losses(checkpoint_root)
        candidates = [
            (losses[epoch], epoch, path)
            for epoch, path in checkpoints.items()
            if epoch in losses
        ]
        if not candidates:
            raise FileNotFoundError(
                f"Could not resolve best-loss checkpoint. Need saved checkpoint-*.pth files "
                f"and train_loss entries in {Path(checkpoint_root) / 'log.txt'}."
            )
        loss, epoch, path = min(candidates, key=lambda item: (item[0], item[1]))
        print(
            f"Selected best-loss checkpoint from saved epochs: {path} "
            f"(epoch={epoch}, train_loss={loss:.6f})."
        )
        return path

    raise FileNotFoundError(
        f"Checkpoint does not exist: {checkpoint_path}. "
        f"Use a path, latest, or best-loss."
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "module"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def strip_prefixes(state_dict):
    prefixes = ("module.", "model.")
    stripped = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        stripped[new_key] = value
    return stripped


def load_backbone(args):
    backbone = models_lomar.__dict__[args.model](
        norm_pix_loss=False,
        lfst_cutoff=args.lfst_cutoff,
        grad_loss_weight=1.0,
        lfst_loss_weight=1.0,
    )
    if args.checkpoint is None:
        print("WARNING: no checkpoint passed; downstream model starts from random weights.")
        return backbone

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = strip_prefixes(checkpoint_state_dict(checkpoint))
    msg = backbone.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(msg)
    return backbone


def configure_trainable_params(model: PhyDClassifier, protocol: str):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.head.parameters():
        param.requires_grad = True

    if protocol == "finetune":
        for name, param in model.backbone.named_parameters():
            is_feature_param = name.startswith(FEATURE_PARAM_PREFIXES)
            if name.startswith("img_SFAFM_process.") and not model.use_sfafm:
                is_feature_param = False
            param.requires_grad = is_feature_param


def count_trainable_params(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def build_transforms(input_size: int):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            input_size, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(contrast=0.5),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    return train_transform, test_transform


def accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item() * 100.0


def train_one_epoch(model, loader, optimizer, criterion, device, protocol):
    if protocol == "linear":
        model.backbone.eval()
        model.head.train()
    else:
        model.train()

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.numel()
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += batch_size

    return total_loss / total_count, total_correct * 100.0 / total_count


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        batch_size = labels.numel()
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += batch_size
    return total_loss / total_count, total_correct * 100.0 / total_count


def run_single(args, dataset_name: str, shots: int, seed: int, protocol: str):
    set_seed(seed)
    dataset_dir = resolve_dataset_dir(args.data_root, dataset_name, shots=shots)
    train_samples, test_samples, classes = build_fewshot_split(dataset_dir, shots, seed)
    train_transform, test_transform = build_transforms(args.input_size)

    train_set = SARClassificationDataset(train_samples, transform=train_transform)
    test_set = SARClassificationDataset(test_samples, transform=test_transform)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        generator=generator,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    backbone = load_backbone(args)
    model = PhyDClassifier(backbone, len(classes), use_sfafm=not args.no_sfafm_features)
    configure_trainable_params(model, protocol)

    device = torch.device(args.device)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    run_dir = args.output_dir / dataset_name / protocol / f"{shots}shot" / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"
    print(
        f"Running {dataset_name} {protocol} {shots}-shot seed={seed}: "
        f"{len(train_samples)} train, {len(test_samples)} test, "
        f"{len(classes)} classes, {count_trainable_params(model)} trainable params"
    )

    best_acc = 0.0
    best_epoch = -1
    final_acc = 0.0
    final_loss = 0.0
    start_time = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device, protocol
            )
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
            final_acc = test_acc
            final_loss = test_loss
            record = {
                "dataset": dataset_name,
                "protocol": protocol,
                "shots": shots,
                "seed": seed,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "best_acc": best_acc,
                "best_epoch": best_epoch,
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(record)

    return {
        "dataset": dataset_name,
        "protocol": protocol,
        "shots": shots,
        "seed": seed,
        "num_classes": len(classes),
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "final_acc": final_acc,
        "final_loss": final_loss,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "elapsed_sec": time.time() - start_time,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Few-shot downstream SAR classification.")
    parser.add_argument("--data_root", type=Path, default=Path("dataset/modelscope/extracted/classification_dataset"))
    parser.add_argument("--checkpoint", default="latest", help="Checkpoint path, latest, or best-loss.")
    parser.add_argument("--checkpoint_root", type=Path, default=Path("runs/pretrain_2xh100"))
    parser.add_argument("--output_dir", type=Path, default=Path("runs/downstream_fewshot"))
    parser.add_argument("--datasets", nargs="+", default=["mstar", "fusar_ship", "sar_acd"])
    parser.add_argument("--shots", nargs="+", type=int, default=[10, 20, 40])
    parser.add_argument("--protocols", nargs="+", choices=("finetune", "linear"), default=["finetune", "linear"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--model", default="mae_vit_base_patch16")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lfst_cutoff", type=int, default=30)
    parser.add_argument("--summary_metric", choices=("final_acc", "best_acc"), default="final_acc")
    parser.add_argument("--allow_random_init", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.add_argument("--no_sfafm_features", action="store_true")
    parser.set_defaults(pin_mem=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.checkpoint = resolve_checkpoint(args.checkpoint, args.checkpoint_root, args.allow_random_init)
    if args.checkpoint is not None:
        print(f"Using checkpoint: {args.checkpoint}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump({key: str(value) for key, value in vars(args).items()}, handle, indent=2)

    all_results = []
    for dataset_name in args.datasets:
        for protocol in args.protocols:
            for shots in args.shots:
                for seed in args.seeds:
                    result = run_single(args, dataset_name, shots, seed, protocol)
                    all_results.append(result)
                    summary_path = args.output_dir / "results.csv"
                    write_header = not summary_path.exists()
                    with summary_path.open("a", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(result))
                        if write_header:
                            writer.writeheader()
                        writer.writerow(result)

    grouped = defaultdict(list)
    for result in all_results:
        key = (result["dataset"], result["protocol"], result["shots"])
        grouped[key].append(result[args.summary_metric])
    print("\nFew-shot summary:")
    for (dataset_name, protocol, shots), values in sorted(grouped.items()):
        mean = float(np.mean(values))
        std = float(np.std(values))
        print(f"{dataset_name} {protocol} {shots}-shot {args.summary_metric}: {mean:.2f} +/- {std:.2f}")


if __name__ == "__main__":
    main()
