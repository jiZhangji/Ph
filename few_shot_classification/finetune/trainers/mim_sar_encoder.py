import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


_ROOT = Path(__file__).resolve().parents[3]
_PRETRAINING = _ROOT / "Pretraining_sarjepa_official_phyd"
if str(_PRETRAINING) not in sys.path:
    sys.path.insert(0, str(_PRETRAINING))

import models_lomar


PRETRAIN_ONLY_PREFIXES = (
    "encoder_pred.",
    "decoder_blocks.",
    "decoder_norm.",
    "decoder_pred.",
    "decoder_pred_lfst.",
    "lfst_builder.",
    "sasgt_builder.",
)
PRETRAIN_ONLY_KEYS = {"mask_token"}


def _checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "module"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def _strip_prefixes(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "backbone.", "image_encoder."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value
    return cleaned


def _is_pretrain_only_key(key):
    return key in PRETRAIN_ONLY_KEYS or key.startswith(PRETRAIN_ONLY_PREFIXES)


def load_pretrained_backbone(backbone, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_model = _strip_prefixes(_checkpoint_state_dict(checkpoint))

    backbone_state = backbone.state_dict()
    loadable = {}
    skipped_shape = []
    skipped_head = []
    for key, value in checkpoint_model.items():
        if key.startswith("head."):
            skipped_head.append(key)
            continue
        if key in backbone_state and value.shape == backbone_state[key].shape:
            loadable[key] = value
        elif key in backbone_state:
            skipped_shape.append((key, tuple(value.shape), tuple(backbone_state[key].shape)))

    missing, unexpected = backbone.load_state_dict(loadable, strict=False)
    unexpected = [key for key in unexpected if not _is_pretrain_only_key(key)]
    missing = [key for key in missing if not _is_pretrain_only_key(key)]

    matched = len(loadable)
    sfafm_matched = sum(key.startswith("img_SFAFM_process.") for key in loadable)
    sfafm_expected = sum(
        key.startswith("img_SFAFM_process.") for key in backbone_state
    )
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Matched backbone keys: {matched}")
    print(f"Matched SFAFM keys: {sfafm_matched}/{sfafm_expected}")
    if skipped_shape:
        print(f"Skipped shape-mismatched keys: {skipped_shape[:8]}")
    if skipped_head:
        print(f"Skipped classifier head keys: {skipped_head}")
    if missing:
        print(f"WARNING missing non-pretrain keys: {missing[:40]}")
    if unexpected:
        print(f"WARNING unexpected non-pretrain keys: {unexpected[:40]}")
    if getattr(backbone, "use_sfafm", False) and sfafm_matched != sfafm_expected:
        raise RuntimeError(
            "SFAFM checkpoint mismatch: "
            f"loaded {sfafm_matched}/{sfafm_expected} SFAFM tensors"
        )


class SARPretrainClassifier(nn.Module):
    def __init__(self, num_classes, checkpoint_path=None, linear_probe=False):
        super().__init__()
        self.use_sfafm = os.environ.get("MIM_USE_SFAFM", "1") != "0"
        self.feature_pool = os.environ.get("MIM_FEATURE_POOL", "cls")
        if self.feature_pool not in {"cls", "patch_mean"}:
            raise ValueError(
                "MIM_FEATURE_POOL must be either 'cls' or 'patch_mean', "
                f"got {self.feature_pool!r}"
            )
        print(f"Use downstream SFAFM: {self.use_sfafm}")
        print(f"Downstream feature pool: {self.feature_pool}")
        self.backbone = models_lomar.mae_vit_base_patch16(use_sfafm=self.use_sfafm)
        self.head = nn.Linear(768, num_classes)

        if checkpoint_path:
            load_pretrained_backbone(self.backbone, checkpoint_path)

        from timm.models.layers import trunc_normal_
        trunc_normal_(self.head.weight, std=2e-5 if not linear_probe else 0.01)
        nn.init.constant_(self.head.bias, 0)
        if linear_probe:
            self.head = nn.Sequential(
                nn.BatchNorm1d(768, affine=False, eps=1e-6),
                self.head,
            )

        self._disable_pretrain_only_params()
        if linear_probe:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _disable_pretrain_only_params(self):
        for name, param in self.backbone.named_parameters():
            if name in PRETRAIN_ONLY_KEYS or name.startswith(PRETRAIN_ONLY_PREFIXES):
                param.requires_grad = False

    def forward(self, image):
        features = self.backbone.forward_features(
            image,
            use_sfafm=self.use_sfafm,
            feature_pool=self.feature_pool,
        )
        return self.head(features)
