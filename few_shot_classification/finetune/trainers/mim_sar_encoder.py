import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


_ROOT = Path(__file__).resolve().parents[3]
_PRETRAINING = _ROOT / "Pretraining_sarjepa_official_phyd"

PRETRAIN_ONLY_PREFIXES = (
    "encoder_pred.",
    "decoder_",
    "decoder_blocks.",
    "decoder_norm.",
    "decoder_pred.",
    "decoder_pred_lfst.",
    "lfst_builder.",
    "sasgt_builder.",
)
PRETRAIN_ONLY_KEYS = {"mask_token"}
MODEL_FAMILY_ALIASES = {
    "mae": "mae",
    "lomar": "lomar",
    "fg-mae": "fg_mae",
    "fg_mae": "fg_mae",
    "ijepa": "i_jepa",
    "i-jepa": "i_jepa",
    "i_jepa": "i_jepa",
    "sar-jepa": "sar_jepa",
    "sar_jepa": "sar_jepa",
    "phyd-mae": "phyd_mae",
    "phyd_mae": "phyd_mae",
    "phyd": "phyd_mae",
}


def normalize_model_family(value):
    key = value.strip().lower()
    if key not in MODEL_FAMILY_ALIASES:
        choices = ", ".join(sorted(set(MODEL_FAMILY_ALIASES.values())))
        raise ValueError(f"Unsupported MIM_MODEL_FAMILY={value!r}; choose from {choices}")
    return MODEL_FAMILY_ALIASES[key]


def _checkpoint_state_dict(checkpoint, family=None):
    if not isinstance(checkpoint, dict):
        return checkpoint

    preferred = ("encoder",) if family == "i_jepa" else ("model", "state_dict", "module")
    for key in preferred:
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    if family == "i_jepa":
        raise KeyError("I-JEPA checkpoint does not contain an 'encoder' state dict")
    raise KeyError("Checkpoint does not contain a model/state_dict/module state dict")


def _strip_prefixes(state_dict):
    cleaned = {}
    prefixes = ("module.", "backbone.", "image_encoder.")
    for original_key, value in state_dict.items():
        key = original_key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        cleaned[key] = value
    return cleaned


def _is_pretrain_only_key(key):
    return key in PRETRAIN_ONLY_KEYS or key.startswith(PRETRAIN_ONLY_PREFIXES)


def _is_classifier_key(key):
    return key.startswith("head.")


def _load_checkpoint_file(checkpoint_path):
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def load_baseline_backbone(model, checkpoint_path, family):
    family = normalize_model_family(family)
    if family == "phyd_mae":
        raise ValueError("Use load_pretrained_backbone for PhyD-MAE checkpoints")

    checkpoint = _load_checkpoint_file(checkpoint_path)
    checkpoint_model = _strip_prefixes(_checkpoint_state_dict(checkpoint, family))
    model_state = model.state_dict()

    expected_pos_tokens = 196 if family == "i_jepa" else 197
    pos_embed = checkpoint_model.get("pos_embed")
    if pos_embed is None or tuple(pos_embed.shape) != (1, expected_pos_tokens, 768):
        shape = None if pos_embed is None else tuple(pos_embed.shape)
        raise RuntimeError(
            f"{family} positional embedding mismatch: expected "
            f"(1, {expected_pos_tokens}, 768), got {shape}"
        )
    has_cls_token = "cls_token" in checkpoint_model
    if has_cls_token == (family == "i_jepa"):
        expectation = "without" if family == "i_jepa" else "with"
        raise RuntimeError(f"{family} must be loaded {expectation} a class token")

    patch_weight = checkpoint_model.get("patch_embed.proj.weight")
    if patch_weight is None or tuple(patch_weight.shape) != (768, 1, 16, 16):
        shape = None if patch_weight is None else tuple(patch_weight.shape)
        raise RuntimeError(
            "Expected a single-channel ViT-B/16 patch projection with shape "
            f"(768, 1, 16, 16), got {shape}"
        )

    required_keys = {
        key for key in model_state if not _is_classifier_key(key)
    }
    loadable = {}
    shape_mismatches = []
    for key, value in checkpoint_model.items():
        if _is_classifier_key(key) or _is_pretrain_only_key(key):
            continue
        if key in model_state:
            if value.shape == model_state[key].shape:
                loadable[key] = value
            else:
                shape_mismatches.append(
                    (key, tuple(value.shape), tuple(model_state[key].shape))
                )

    missing_required = sorted(required_keys - loadable.keys())
    matched_blocks = {
        int(key.split(".")[1])
        for key in loadable
        if key.startswith("blocks.") and key.split(".")[1].isdigit()
    }
    if shape_mismatches or missing_required or matched_blocks != set(range(12)):
        raise RuntimeError(
            "Incomplete baseline encoder load: "
            f"shape_mismatches={shape_mismatches[:8]}, "
            f"missing={missing_required[:20]}, "
            f"matched_blocks={sorted(matched_blocks)}"
        )

    missing, unexpected = model.load_state_dict(loadable, strict=False)
    unexpected = [key for key in unexpected if not _is_pretrain_only_key(key)]
    missing = [key for key in missing if not _is_classifier_key(key)]
    if missing or unexpected:
        raise RuntimeError(
            f"Unexpected load result: missing={missing[:20]}, unexpected={unexpected[:20]}"
        )

    required_numel = sum(model_state[key].numel() for key in required_keys)
    matched_numel = sum(model_state[key].numel() for key in required_keys & loadable.keys())
    ignored = [
        key for key in checkpoint_model
        if key not in loadable and not _is_classifier_key(key)
    ]
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Model family: {family}")
    print(f"Matched encoder tensors: {len(loadable)}/{len(required_keys)}")
    print(f"Matched encoder parameters: {matched_numel}/{required_numel} (100.00%)")
    print(f"Ignored pre-training or architecture-specific tensors: {len(ignored)}")


class IJEPAVisionTransformer:
    """Factory namespace to avoid importing timm until the baseline is used."""

    @staticmethod
    def build(num_classes):
        from timm.models.vision_transformer import VisionTransformer

        class PatchMeanVisionTransformer(VisionTransformer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.cls_token = None
                self.pos_embed = nn.Parameter(
                    torch.zeros(1, self.patch_embed.num_patches, kwargs["embed_dim"])
                )

            def forward_features(self, x):
                x = self.patch_embed(x)
                x = x + self.pos_embed
                x = self.pos_drop(x)
                for block in self.blocks:
                    x = block(x)
                x = self.norm(x)
                return x.mean(dim=1)

            def forward(self, x):
                # Newer timm versions expect forward_features() to return a
                # token sequence and apply their own pooling in forward_head.
                # I-JEPA has no CLS token, so keep patch-mean pooling here and
                # call the classifier head directly on every timm version.
                return self.head(self.forward_features(x))

        return PatchMeanVisionTransformer(
            img_size=224,
            patch_size=16,
            in_chans=1,
            num_classes=num_classes,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=lambda dim: nn.LayerNorm(dim, eps=1e-6),
        )


def _build_plain_vit(num_classes):
    from timm.models.vision_transformer import VisionTransformer

    return VisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=1,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=lambda dim: nn.LayerNorm(dim, eps=1e-6),
    )


class SARBaselineClassifier(nn.Module):
    def __init__(self, num_classes, checkpoint_path, family, linear_probe=False):
        super().__init__()
        family = normalize_model_family(family)
        self.family = family
        self.backbone = (
            IJEPAVisionTransformer.build(num_classes)
            if family == "i_jepa"
            else _build_plain_vit(num_classes)
        )
        load_baseline_backbone(self.backbone, checkpoint_path, family)

        from timm.models.layers import trunc_normal_

        trunc_normal_(
            self.backbone.head.weight,
            std=0.01 if linear_probe else 2e-5,
        )
        nn.init.constant_(self.backbone.head.bias, 0)
        if linear_probe:
            linear_head = self.backbone.head
            self.backbone.head = nn.Sequential(
                nn.BatchNorm1d(768, affine=False, eps=1e-6),
                linear_head,
            )
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            for parameter in self.backbone.head.parameters():
                parameter.requires_grad = True

    def forward(self, image):
        return self.backbone(image)


def load_pretrained_backbone(backbone, checkpoint_path):
    checkpoint = _load_checkpoint_file(checkpoint_path)
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
    sfafm_matched = sum(key.startswith("img_SFAFM_process") for key in loadable)
    sfafm_expected = sum(
        key.startswith("img_SFAFM_process") for key in backbone_state
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
        if str(_PRETRAINING) not in sys.path:
            sys.path.insert(0, str(_PRETRAINING))
        import models_lomar

        self.use_sfafm = os.environ.get("MIM_USE_SFAFM", "0") != "0"
        self.sfafm_layout = os.environ.get("MIM_SFAFM_LAYOUT", "late")
        self.feature_pool = os.environ.get("MIM_FEATURE_POOL", "cls")
        if self.feature_pool not in {"cls", "patch_mean"}:
            raise ValueError(
                "MIM_FEATURE_POOL must be either 'cls' or 'patch_mean', "
                f"got {self.feature_pool!r}"
            )
        print(f"Use downstream SFAFM: {self.use_sfafm}")
        print(f"Downstream SFAFM layout: {self.sfafm_layout}")
        print(f"Downstream feature pool: {self.feature_pool}")
        self.backbone = models_lomar.mae_vit_base_patch16(
            use_sfafm=self.use_sfafm,
            sfafm_layout=self.sfafm_layout,
        )
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


def build_sar_classifier(num_classes, checkpoint_path, family, linear_probe=False):
    family = normalize_model_family(family)
    if family == "phyd_mae":
        return SARPretrainClassifier(
            num_classes=num_classes,
            checkpoint_path=checkpoint_path,
            linear_probe=linear_probe,
        )
    return SARBaselineClassifier(
        num_classes=num_classes,
        checkpoint_path=checkpoint_path,
        family=family,
        linear_probe=linear_probe,
    )
