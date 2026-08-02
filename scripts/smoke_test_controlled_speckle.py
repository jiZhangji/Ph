#!/usr/bin/env python3
"""Small deterministic/statistical check for controlled speckle injection."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
FINETUNE_DIR = ROOT / "few_shot_classification" / "finetune"
sys.path.insert(0, str(FINETUNE_DIR))

from trainers.controlled_speckle import add_amplitude_speckle


def main():
    looks = 4.0
    images = torch.full((8, 1, 256, 256), 0.1)
    paths = [f"sample-{index}.png" for index in range(images.shape[0])]

    first = add_amplitude_speckle(images, paths, looks, noise_seed=7)
    repeat = add_amplitude_speckle(images, paths, looks, noise_seed=7)
    changed = add_amplitude_speckle(images, paths, looks, noise_seed=8)
    if not torch.equal(first, repeat):
        raise RuntimeError("Same path and noise seed did not reproduce identical noise")
    if torch.equal(first, changed):
        raise RuntimeError("Different noise seeds produced identical noise")

    recovered_intensity_noise = (first / images).square()
    empirical_mean = recovered_intensity_noise.mean().item()
    empirical_variance = recovered_intensity_noise.var(unbiased=False).item()
    expected_variance = 1.0 / looks
    if abs(empirical_mean - 1.0) > 0.015:
        raise RuntimeError(f"Unexpected noise mean: {empirical_mean}")
    if abs(empirical_variance - expected_variance) > 0.02:
        raise RuntimeError(f"Unexpected noise variance: {empirical_variance}")

    clean = add_amplitude_speckle(images, paths, None, noise_seed=7)
    if clean is not images:
        raise RuntimeError("Clean mode must return the original tensor unchanged")

    print(f"Deterministic repeat: OK")
    print(f"Empirical E[N]: {empirical_mean:.5f} (expected 1.0)")
    print(
        f"Empirical Var[N]: {empirical_variance:.5f} "
        f"(expected {expected_variance:.5f})"
    )
    print("Controlled speckle smoke test passed.")


if __name__ == "__main__":
    main()
