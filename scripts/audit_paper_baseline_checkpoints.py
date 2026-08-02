#!/usr/bin/env python3
"""Load every paper baseline checkpoint and verify full encoder coverage."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
FINETUNE_DIR = ROOT / "few_shot_classification" / "finetune"
sys.path.insert(0, str(FINETUNE_DIR))

from trainers.mim_sar_encoder import SARBaselineClassifier


SPECS = {
    "mae": ("MAE/checkpoint-200.pth", "weights/mae/checkpoint-200.pth"),
    "lomar": ("LoMaR/checkpoint-200.pth", "weights/lomar/checkpoint-200.pth"),
    "fg_mae": ("FG-MAE/checkpoint-200.pth", "weights/fg_mae/checkpoint-200.pth"),
    "i_jepa": ("ijepa/jepa-ep200.pth.tar", "weights/i_jepa/jepa-ep200.pth.tar"),
    "sar_jepa": ("SAR-JEPA/checkpoint-200.pth", "weights/sar_jepa/checkpoint-200.pth"),
}


def resolve(weights_root, candidates):
    for relative in candidates:
        candidate = weights_root / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "No checkpoint found among: "
        + ", ".join(str(weights_root / item) for item in candidates)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--forward", action="store_true")
    args = parser.parse_args()

    for family, candidates in SPECS.items():
        checkpoint = resolve(args.weights_root, candidates)
        print(f"\nAuditing {family}: {checkpoint}")
        model = SARBaselineClassifier(
            num_classes=10,
            checkpoint_path=str(checkpoint),
            family=family,
            linear_probe=False,
        )
        if args.forward:
            with torch.inference_mode():
                output = model(torch.zeros(1, 1, 224, 224))
            if tuple(output.shape) != (1, 10):
                raise RuntimeError(f"Unexpected output shape: {tuple(output.shape)}")
            print(f"Forward output: {tuple(output.shape)}")
        del model
        gc.collect()

    print("\nAll five paper baseline checkpoints passed the encoder audit.")


if __name__ == "__main__":
    main()
