"""Deterministic test-only multiplicative speckle evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


CLEAN_NAMES = {"clean", "none", "inf", "infinity"}


def parse_looks(value):
    text = str(value).strip().lower()
    if text in CLEAN_NAMES:
        return None
    looks = float(text)
    if not np.isfinite(looks) or looks <= 0:
        raise ValueError(f"Speckle looks must be positive or 'clean', got {value!r}")
    return looks


def format_looks(looks):
    return "clean" if looks is None else f"{looks:g}"


def _sample_seed(impath, noise_seed):
    payload = f"{noise_seed}\0{impath}".encode("utf-8", errors="surrogatepass")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def add_amplitude_speckle(images, impaths, looks, noise_seed):
    """Apply A_noisy = clamp(A * sqrt(N), 0, 1), N ~ Gamma(L, 1/L)."""
    if looks is None:
        return images
    if images.ndim != 4:
        raise ValueError(f"Expected BCHW images, got shape {tuple(images.shape)}")
    if len(impaths) != images.shape[0]:
        raise ValueError(
            f"Expected {images.shape[0]} image paths, got {len(impaths)}"
        )

    source_device = images.device
    source_dtype = images.dtype
    output = images.detach().to(device="cpu", dtype=torch.float32).clone()
    height, width = output.shape[-2:]
    for index, impath in enumerate(impaths):
        rng = np.random.default_rng(_sample_seed(str(impath), noise_seed))
        intensity_noise = rng.gamma(
            shape=looks,
            scale=1.0 / looks,
            size=(1, height, width),
        ).astype(np.float32, copy=False)
        amplitude_noise = torch.from_numpy(np.sqrt(intensity_noise))
        output[index].mul_(amplitude_noise).clamp_(0.0, 1.0)
    return output.to(device=source_device, dtype=source_dtype)


def _split_values(raw):
    return [value for value in re.split(r"[\s,]+", raw.strip()) if value]


def requested_evaluations():
    looks_raw = os.environ.get("MIM_TEST_SPECKLE_LOOKS_LIST", "").strip()
    if not looks_raw:
        single = os.environ.get("MIM_TEST_SPECKLE_LOOKS", "clean")
        return [(parse_looks(single), int(os.environ.get("MIM_TEST_SPECKLE_SEED", "0")))], False

    looks_values = [parse_looks(value) for value in _split_values(looks_raw)]
    if not looks_values:
        raise ValueError("MIM_TEST_SPECKLE_LOOKS_LIST is empty")
    noise_seed_values = _split_values(
        os.environ.get("MIM_TEST_SPECKLE_NOISE_SEEDS", "0")
    )
    noise_seeds = [int(value) for value in noise_seed_values]
    if not noise_seeds:
        raise ValueError("MIM_TEST_SPECKLE_NOISE_SEEDS is empty")

    evaluations = []
    for looks in looks_values:
        if looks is None:
            evaluations.append((None, noise_seeds[0]))
        else:
            evaluations.extend((looks, seed) for seed in noise_seeds)
    return evaluations, True


def _feature_head(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    image_encoder = getattr(model, "image_encoder", None)
    if image_encoder is None:
        raise AttributeError("Feature export requires model.image_encoder")
    if hasattr(image_encoder, "head"):
        return image_encoder.head
    backbone = getattr(image_encoder, "backbone", None)
    if backbone is not None and hasattr(backbone, "head"):
        return backbone.head
    raise AttributeError("Unable to locate the downstream classifier head")


class ControlledSpeckleEvaluationMixin:
    """Adds repeatable clean/corrupted test passes without retraining."""

    _active_speckle_looks = None
    _active_speckle_seed = 0

    def parse_batch_test(self, batch):
        inputs = batch["img"]
        labels = batch["label"]
        if self._active_speckle_looks is not None:
            impaths = batch.get("impath")
            if impaths is None:
                raise KeyError(
                    "Controlled speckle evaluation requires batch['impath'] "
                    "for model-independent deterministic noise"
                )
            inputs = add_amplitude_speckle(
                inputs,
                impaths,
                self._active_speckle_looks,
                self._active_speckle_seed,
            )
        return inputs.to(self.device), labels.to(self.device)

    @torch.no_grad()
    def _test_once(self, split=None):
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT
        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"
            data_loader = self.test_loader

        feature_output = os.environ.get("MIM_FEATURE_OUTPUT", "").strip()
        export_features = bool(feature_output) and self._active_speckle_looks is None
        captured_features = []
        captured_labels = []
        captured_paths = []
        hook = None
        if export_features:
            def capture_head_input(_module, inputs):
                features = inputs[0]
                if features.ndim != 2:
                    raise RuntimeError(
                        "Expected BxD penultimate features, got "
                        f"{tuple(features.shape)}"
                    )
                captured_features.append(features.detach().cpu())

            hook = _feature_head(self.model).register_forward_pre_hook(
                capture_head_input
            )

        print(f"Evaluate on the *{split}* set")
        try:
            for batch in tqdm(data_loader):
                inputs, labels = self.parse_batch_test(batch)
                outputs = self.model(inputs)
                self.evaluator.process(outputs, labels)
                if export_features:
                    captured_labels.append(labels.detach().cpu())
                    paths = batch.get("impath", [])
                    captured_paths.extend(str(path) for path in paths)
        finally:
            if hook is not None:
                hook.remove()

        if export_features:
            output_path = Path(feature_output).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            features = torch.cat(captured_features).numpy()
            labels = torch.cat(captured_labels).numpy()
            if captured_paths and len(captured_paths) != len(labels):
                raise RuntimeError(
                    "Feature export path count mismatch: "
                    f"paths={len(captured_paths)} labels={len(labels)}"
                )
            np.savez_compressed(
                output_path,
                features=features,
                labels=labels,
                paths=np.asarray(captured_paths, dtype=str),
                classnames=np.asarray(self.dm.dataset.classnames, dtype=str),
                dataset=str(self.cfg.DATASET.NAME),
                protocol=self.__class__.__name__,
                shots=int(self.cfg.DATASET.NUM_SHOTS),
                seed=int(self.cfg.SEED),
                method=os.environ.get("MIM_MODEL_FAMILY", "unspecified"),
            )
            print(
                f"FEATURE_EXPORT path={output_path} "
                f"samples={len(labels)} dim={features.shape[1]}"
            )

        results = self.evaluator.evaluate()
        looks_label = format_looks(self._active_speckle_looks)
        for key, value in results.items():
            tag = f"{split}_speckle_{looks_label}_seed{self._active_speckle_seed}/{key}"
            self.write_scalar(tag, value, self.epoch)
        return results

    @torch.no_grad()
    def test(self, split=None):
        evaluations, is_sweep = requested_evaluations()
        first_metric = None
        completed = 0
        for looks, noise_seed in evaluations:
            self._active_speckle_looks = looks
            self._active_speckle_seed = noise_seed
            looks_label = format_looks(looks)
            print(
                "CONTROLLED_SPECKLE "
                f"L_add={looks_label} noise_seed={noise_seed} "
                "input=amplitude model=A*sqrt(Gamma(L,1/L))"
            )
            results = self._test_once(split)
            if first_metric is None:
                first_metric = next(iter(results.values()))
            payload = {
                "method": os.environ.get("MIM_MODEL_FAMILY", "unspecified"),
                "L_add": looks_label,
                "noise_seed": noise_seed if looks is not None else None,
                "downstream_seed": int(self.cfg.SEED),
                "dataset": self.cfg.DATASET.NAME,
                "shots": int(self.cfg.DATASET.NUM_SHOTS),
                "protocol": self.__class__.__name__,
                "learning_rate": os.environ.get("MIM_DOWNSTREAM_LR"),
                "metrics": {key: float(value) for key, value in results.items()},
            }
            print("SPECKLE_RESULT " + json.dumps(payload, sort_keys=True))
            completed += 1

        self._active_speckle_looks = None
        if is_sweep:
            print(
                f"SPECKLE_ROBUSTNESS_COMPLETE completed={completed} "
                f"expected={len(evaluations)}"
            )
        return first_metric
