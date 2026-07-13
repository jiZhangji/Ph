#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PRETRAINING = ROOT / "Pretraining_sarjepa_official_phyd"
sys.path.insert(0, str(PRETRAINING))

import models_lomar


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", default=1, type=int)
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device(args.device)
    model = models_lomar.mae_vit_base_patch16(
        grad_loss_weight=1.0,
        lfst_loss_weight=1.0,
        target_norm="image",
        use_sfafm=True,
        sfafm_reduction=4,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_model = checkpoint.get("model", checkpoint)
    checkpoint_model = {
        key.removeprefix("module."): value
        for key, value in checkpoint_model.items()
    }
    incompatible = model.load_state_dict(checkpoint_model, strict=False)
    disallowed_missing = [
        key for key in incompatible.missing_keys
        if not key.startswith("img_SFAFM_process.")
    ]
    if disallowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"disallowed missing={disallowed_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )

    feature_map = torch.randn(2, 768, 7, 7)
    with torch.no_grad():
        identity_error = (
            model.img_SFAFM_process(feature_map) - feature_map
        ).abs().max().item()
    if identity_error != 0.0:
        raise RuntimeError(f"SFAFM is not identity-initialized: {identity_error}")

    model.to(device).eval()
    images = torch.randn(args.batch_size, 1, 224, 224, device=device)
    with torch.no_grad():
        features = model.forward_features(images)

    print(f"checkpoint={args.checkpoint}")
    print(f"missing_sfafm_keys={len(incompatible.missing_keys)}")
    print(f"identity_max_error={identity_error}")
    print(f"features_shape={tuple(features.shape)}")
    print("SFAFM smoke test passed")


if __name__ == "__main__":
    main()
