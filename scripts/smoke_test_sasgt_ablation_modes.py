#!/usr/bin/env python3
"""Check SASGT ablation target shapes and numerical validity."""

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PRETRAINING = ROOT / "Pretraining_sarjepa_official_phyd"
sys.path.insert(0, str(PRETRAINING))

from models_lomar import MaskedAutoencoderViT, SASGTTarget  # noqa: E402


def main():
    torch.manual_seed(0)
    inputs = torch.rand(2, 1, 64, 64) * 25.0
    expected_channels = {
        "uniform": 1,
        "adaptive": 1,
        "no_log": 2,
        "complete": 2,
    }
    outputs = {}

    for mode, channels in expected_channels.items():
        target = SASGTTarget(mode=mode)
        output = target(inputs)
        expected_shape = (inputs.shape[0], channels, *inputs.shape[-2:])
        if target.out_channels != channels:
            raise AssertionError(
                f"{mode}: out_channels={target.out_channels}, expected {channels}"
            )
        if tuple(output.shape) != expected_shape:
            raise AssertionError(
                f"{mode}: output shape={tuple(output.shape)}, expected {expected_shape}"
            )
        if not torch.isfinite(output).all():
            raise AssertionError(f"{mode}: target contains non-finite values")
        outputs[mode] = output
        print(f"{mode:>8}: shape={tuple(output.shape)} finite=true")

        model = MaskedAutoencoderViT(
            img_size=32,
            patch_size=16,
            embed_dim=32,
            depth=1,
            num_heads=4,
            decoder_embed_dim=16,
            decoder_depth=1,
            decoder_num_heads=4,
            mlp_ratio=2,
            lfst_loss_weight=0.0,
            sasgt_mode=mode,
        )
        expected_head_dim = 16 * 16 * channels
        if model.decoder_pred.out_features != expected_head_dim:
            raise AssertionError(
                f"{mode}: decoder output={model.decoder_pred.out_features}, "
                f"expected {expected_head_dim}"
            )
        loss, predictions, _ = model(
            inputs[:1, :, :32, :32], window_size=2, num_window=1, mask_ratio=0.5
        )
        if not torch.isfinite(loss):
            raise AssertionError(f"{mode}: model smoke loss is not finite")
        if predictions[0].shape[-1] != expected_head_dim:
            raise AssertionError(
                f"{mode}: prediction dim={predictions[0].shape[-1]}, "
                f"expected {expected_head_dim}"
            )
        if predictions[1] is not None:
            raise AssertionError(f"{mode}: LFST prediction should be disabled")
        loss.backward()
        if model.decoder_pred.weight.grad is None:
            raise AssertionError(f"{mode}: spatial prediction head received no gradient")
        if model.decoder_pred_lfst.weight.grad is not None:
            raise AssertionError(f"{mode}: disabled LFST head unexpectedly received a gradient")

    if torch.allclose(outputs["complete"], outputs["no_log"], atol=1e-6, rtol=1e-5):
        raise AssertionError("no_log target unexpectedly matches complete SASGT")

    print("SASGT ablation smoke test passed.")


if __name__ == "__main__":
    main()
