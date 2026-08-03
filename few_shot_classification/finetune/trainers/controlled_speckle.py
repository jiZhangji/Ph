"""Deterministic test-only multiplicative speckle evaluation."""

from __future__ import annotations

import hashlib
import csv
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
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


def _attention_target_layer(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    image_encoder = getattr(model, "image_encoder", model)
    backbone = getattr(image_encoder, "backbone", image_encoder)
    blocks = getattr(backbone, "blocks", None)
    if blocks is None or len(blocks) == 0:
        raise AttributeError("Unable to locate ViT transformer blocks")
    layer = getattr(blocks[-1], "norm1", None)
    if layer is None:
        raise AttributeError("Unable to locate the final ViT block norm1")
    return layer


def _gradcam_from_tokens(activations, gradients):
    if activations.ndim != 3 or gradients.ndim != 3:
        raise RuntimeError(
            "Expected BxTxC token activations and gradients, got "
            f"{tuple(activations.shape)} and {tuple(gradients.shape)}"
        )
    if activations.shape != gradients.shape:
        raise RuntimeError(
            f"Activation/gradient shape mismatch: "
            f"{tuple(activations.shape)} vs {tuple(gradients.shape)}"
        )

    token_count = activations.shape[1]
    patch_count = token_count - 1
    grid_size = math.isqrt(patch_count)
    if grid_size * grid_size == patch_count:
        activations = activations[:, 1:]
        gradients = gradients[:, 1:]
    else:
        patch_count = token_count
        grid_size = math.isqrt(patch_count)
        if grid_size * grid_size != patch_count:
            raise RuntimeError(
                f"Cannot reshape {token_count} tokens into a square patch grid"
            )

    weights = gradients.mean(dim=1, keepdim=True)
    cams = torch.relu((activations * weights).sum(dim=-1))
    cams = cams.reshape(-1, grid_size, grid_size)
    flat = cams.flatten(1)
    minimum = flat.min(dim=1).values[:, None, None]
    maximum = flat.max(dim=1).values[:, None, None]
    return (cams - minimum) / (maximum - minimum).clamp_min(1e-12)


def _heat_colors(values):
    anchors = np.asarray(
        [
            [0, 0, 4],
            [87, 16, 110],
            [188, 55, 84],
            [249, 142, 8],
            [252, 255, 164],
        ],
        dtype=np.float32,
    )
    positions = np.linspace(0.0, 1.0, len(anchors))
    output = np.empty((*values.shape, 3), dtype=np.float32)
    for channel in range(3):
        output[..., channel] = np.interp(
            values, positions, anchors[:, channel]
        )
    return output


def _save_rgb_jpeg(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(
        path,
        format="JPEG",
        quality=92,
        subsampling=0,
        optimize=True,
    )


def _save_attention_images(original_path, overlay_path, image, cam):
    image = image.detach().float().cpu()
    if image.ndim != 3:
        raise RuntimeError(f"Expected CxHxW image, got {tuple(image.shape)}")
    gray = image.mean(dim=0).clamp(0, 1).numpy()
    height, width = gray.shape
    cam_image = Image.fromarray(np.uint8(np.clip(cam, 0, 1) * 255), mode="L")
    resampling = getattr(Image, "Resampling", Image)
    cam_image = cam_image.resize(
        (width, height), resample=resampling.BICUBIC
    )
    cam_resized = np.asarray(cam_image, dtype=np.float32) / 255.0

    original = np.repeat(gray[..., None] * 255.0, 3, axis=2)
    heat = _heat_colors(cam_resized)
    alpha = (0.68 * cam_resized)[..., None]
    overlay = original * (1.0 - alpha) + heat * alpha
    original = np.uint8(np.clip(original, 0, 255))
    overlay = np.uint8(np.clip(overlay, 0, 255))

    if not original_path.exists():
        temporary = original_path.with_name(
            f".{original_path.name}.{os.getpid()}.tmp.jpg"
        )
        _save_rgb_jpeg(temporary, original)
        original_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(temporary, original_path)
        finally:
            if temporary.exists():
                temporary.unlink()
    _save_rgb_jpeg(overlay_path, overlay)


class _GradCamCapture:
    def __init__(self, layer):
        self.activations = None
        self.gradients = None
        self.handle = layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, _module, _inputs, output):
        if not torch.is_tensor(output):
            raise RuntimeError("Attention target layer did not return a tensor")
        if not output.requires_grad:
            return
        self.activations = output
        self.gradients = None
        output.register_hook(self._gradient_hook)

    def _gradient_hook(self, gradient):
        self.gradients = gradient

    def cam(self):
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Attention activations or gradients were not captured")
        return _gradcam_from_tokens(
            self.activations.detach(), self.gradients.detach()
        )

    def close(self):
        self.handle.remove()


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
        attention_output = os.environ.get(
            "MIM_ATTENTION_OUTPUT_DIR", ""
        ).strip()
        export_attention = (
            bool(attention_output) and self._active_speckle_looks is None
        )
        if export_features and export_attention:
            raise ValueError(
                "MIM_FEATURE_OUTPUT and MIM_ATTENTION_OUTPUT_DIR "
                "cannot be enabled together"
            )
        captured_features = []
        captured_labels = []
        captured_paths = []
        hook = None
        gradcam = None
        attention_cams = []
        attention_labels = []
        attention_predictions = []
        attention_confidences = []
        attention_paths = []
        attention_rows = []
        attention_index = 0
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
        if export_attention:
            gradcam = _GradCamCapture(_attention_target_layer(self.model))
            attention_target = os.environ.get(
                "MIM_ATTENTION_TARGET", "ground_truth"
            ).strip().lower()
            if attention_target not in {"ground_truth", "predicted"}:
                raise ValueError(
                    "MIM_ATTENTION_TARGET must be ground_truth or predicted"
                )
            attention_dir = Path(attention_output).expanduser().resolve()
            attention_dir.mkdir(parents=True, exist_ok=True)
            attention_original_dir = Path(
                os.environ.get(
                    "MIM_ATTENTION_ORIGINAL_DIR",
                    str(attention_dir / "originals"),
                )
            ).expanduser().resolve()
            classnames = list(self.dm.dataset.classnames)
            sample_limit = int(
                os.environ.get("MIM_ATTENTION_SAMPLES_PER_DATASET", "0")
            )
            selection_seed = int(
                os.environ.get("MIM_ATTENTION_SAMPLE_SEED", "20260804")
            )
            selected_paths = None
            if sample_limit > 0:
                data_source = getattr(data_loader.dataset, "data_source", None)
                if data_source is None:
                    raise AttributeError(
                        "Unable to enumerate test paths for attention sampling"
                    )
                available_paths = sorted(
                    {str(item.impath) for item in data_source}
                )
                ranked_paths = sorted(
                    available_paths,
                    key=lambda path: hashlib.sha256(
                        f"{selection_seed}\0{path}".encode(
                            "utf-8", errors="surrogatepass"
                        )
                    ).digest(),
                )
                selected_paths = set(ranked_paths[:sample_limit])
                if len(selected_paths) != min(
                    sample_limit, len(available_paths)
                ):
                    raise RuntimeError("Attention sample selection mismatch")
                print(
                    "ATTENTION_SELECTION "
                    f"selected={len(selected_paths)} "
                    f"available={len(available_paths)} seed={selection_seed}"
                )

        print(f"Evaluate on the *{split}* set")
        try:
            for batch in tqdm(data_loader):
                inputs, labels = self.parse_batch_test(batch)
                if export_attention:
                    paths = [str(path) for path in batch.get("impath", [])]
                    if len(paths) != len(labels):
                        raise RuntimeError(
                            "Attention export requires one image path per sample"
                        )
                    with torch.no_grad():
                        outputs = self.model(inputs)
                    self.evaluator.process(outputs, labels)

                    selected_offsets = [
                        index
                        for index, path in enumerate(paths)
                        if selected_paths is None or path in selected_paths
                    ]
                    if not selected_offsets:
                        continue
                    selected_inputs = inputs[selected_offsets]
                    selected_labels = labels[selected_offsets]

                    self.model.zero_grad(set_to_none=True)
                    with torch.enable_grad():
                        attention_outputs = self.model(selected_inputs)
                        predictions = attention_outputs.argmax(dim=1)
                        targets = (
                            selected_labels
                            if attention_target == "ground_truth"
                            else predictions
                        )
                        target_scores = attention_outputs.gather(
                            1, targets[:, None]
                        ).sum()
                        target_scores.backward()
                    cams = gradcam.cam().cpu()
                    probabilities = attention_outputs.detach().softmax(dim=1)
                    confidences = probabilities.gather(
                        1, predictions[:, None]
                    ).squeeze(1).cpu()
                    predictions_cpu = predictions.detach().cpu()
                    labels_cpu = selected_labels.detach().cpu()

                    for offset, batch_offset in enumerate(selected_offsets):
                        path_text = paths[batch_offset]
                        label = int(labels_cpu[offset])
                        prediction = int(predictions_cpu[offset])
                        digest = hashlib.sha1(
                            path_text.encode(
                                "utf-8", errors="surrogatepass"
                            )
                        ).hexdigest()[:10]
                        stem = re.sub(
                            r"[^A-Za-z0-9_.-]+",
                            "_",
                            Path(path_text).stem,
                        )[:80]
                        classname = classnames[label]
                        safe_class = re.sub(
                            r"[^A-Za-z0-9_.-]+", "_", classname
                        )
                        sample_id = f"{attention_index:05d}_{digest}"
                        filename = f"{sample_id}_{stem}.jpg"
                        relative_overlay = (
                            Path("overlays")
                            / safe_class
                            / filename
                        )
                        relative_original = Path(safe_class) / filename
                        _save_attention_images(
                            attention_original_dir / relative_original,
                            attention_dir / relative_overlay,
                            inputs[batch_offset],
                            cams[offset].numpy(),
                        )
                        attention_rows.append(
                            {
                                "sample_index": attention_index,
                                "sample_id": sample_id,
                                "source_path": path_text,
                                "label": label,
                                "classname": classname,
                                "prediction": prediction,
                                "predicted_class": classnames[prediction],
                                "confidence": f"{float(confidences[offset]):.8f}",
                                "correct": int(label == prediction),
                                "target": attention_target,
                                "original_path": relative_original.as_posix(),
                                "overlay_path": relative_overlay.as_posix(),
                            }
                        )
                        attention_index += 1

                    attention_cams.append(cams.to(dtype=torch.float16))
                    attention_labels.append(labels_cpu)
                    attention_predictions.append(predictions_cpu)
                    attention_confidences.append(confidences)
                else:
                    with torch.no_grad():
                        outputs = self.model(inputs)
                    self.evaluator.process(outputs, labels)
                if export_features:
                    captured_labels.append(labels.detach().cpu())
                    paths = batch.get("impath", [])
                    captured_paths.extend(str(path) for path in paths)
        finally:
            if hook is not None:
                hook.remove()
            if gradcam is not None:
                gradcam.close()

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

        if export_attention:
            if not attention_rows:
                raise RuntimeError("Attention export produced no samples")
            if selected_paths is not None and len(attention_rows) != len(
                selected_paths
            ):
                raise RuntimeError(
                    "Attention export did not cover the selected sample set: "
                    f"exported={len(attention_rows)} "
                    f"selected={len(selected_paths)}"
                )
            fields = (
                "sample_index",
                "sample_id",
                "source_path",
                "label",
                "classname",
                "prediction",
                "predicted_class",
                "confidence",
                "correct",
                "target",
                "original_path",
                "overlay_path",
            )
            with (attention_dir / "index.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(attention_rows)
            np.savez_compressed(
                attention_dir / "attention_maps.npz",
                cams=torch.cat(attention_cams).numpy(),
                labels=torch.cat(attention_labels).numpy(),
                predictions=torch.cat(attention_predictions).numpy(),
                confidences=torch.cat(attention_confidences).numpy(),
                paths=np.asarray(
                    [row["source_path"] for row in attention_rows], dtype=str
                ),
                classnames=np.asarray(classnames, dtype=str),
                dataset=str(self.cfg.DATASET.NAME),
                protocol=self.__class__.__name__,
                shots=int(self.cfg.DATASET.NUM_SHOTS),
                seed=int(self.cfg.SEED),
                method=os.environ.get("MIM_MODEL_FAMILY", "unspecified"),
                target=attention_target,
                sample_selection_seed=selection_seed,
            )
            marker = {
                "samples": len(attention_rows),
                "method": os.environ.get(
                    "MIM_MODEL_FAMILY", "unspecified"
                ),
                "dataset": str(self.cfg.DATASET.NAME),
                "protocol": self.__class__.__name__,
                "shots": int(self.cfg.DATASET.NUM_SHOTS),
                "seed": int(self.cfg.SEED),
                "target": attention_target,
                "sample_selection_seed": selection_seed,
            }
            (attention_dir / "ATTENTION_EXPORT_COMPLETE.json").write_text(
                json.dumps(marker, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                f"ATTENTION_EXPORT path={attention_dir} "
                f"samples={len(attention_rows)} target={attention_target}"
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
