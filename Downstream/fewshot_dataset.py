from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

DATASET_ALIASES = {
    "mstar": ("mstar",),
    "fusar_ship": ("fusarship", "fusar_ship", "fusar-ship", "fusar"),
    "sar_acd": ("saracd", "sar_acd", "sar-acd", "acd"),
}

TRAIN_NAMES = ("train", "training", "train_set", "trainset")
TEST_NAMES = ("test", "testing", "val", "valid", "validation", "query")


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def resolve_dataset_dir(data_root: Path, dataset: str) -> Path:
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Downstream data root does not exist: {data_root}")

    aliases = DATASET_ALIASES.get(dataset, (dataset,))
    normalized_aliases = {normalize_name(alias) for alias in aliases}
    direct = data_root / dataset
    if direct.is_dir():
        return direct

    for path in data_root.iterdir():
        if path.is_dir() and normalize_name(path.name) in normalized_aliases:
            return path

    available = ", ".join(sorted(path.name for path in data_root.iterdir() if path.is_dir()))
    raise FileNotFoundError(
        f"Could not find dataset '{dataset}' under {data_root}. "
        f"Available directories: {available}"
    )


def find_split_dir(dataset_dir: Path, names: tuple[str, ...]) -> Path | None:
    normalized_names = {normalize_name(name) for name in names}
    for child in dataset_dir.iterdir():
        if child.is_dir() and normalize_name(child.name) in normalized_names:
            return child
    return None


def collect_class_images(root: Path) -> dict[str, list[Path]]:
    class_to_images: dict[str, list[Path]] = {}
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        images = sorted(
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if images:
            class_to_images[class_dir.name] = images
    if not class_to_images:
        raise RuntimeError(f"No class folders with supported images found under: {root}")
    return class_to_images


def build_fewshot_split(dataset_dir: Path, shots: int, seed: int):
    train_dir = find_split_dir(dataset_dir, TRAIN_NAMES)
    test_dir = find_split_dir(dataset_dir, TEST_NAMES)

    if train_dir is not None and test_dir is not None:
        train_pool = collect_class_images(train_dir)
        test_pool = collect_class_images(test_dir)
    else:
        train_pool = collect_class_images(dataset_dir)
        test_pool = None

    classes = sorted(train_pool)
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    rng = random.Random(seed)

    train_samples = []
    remaining_by_class: dict[str, list[Path]] = defaultdict(list)
    for class_name in classes:
        images = list(train_pool[class_name])
        rng.shuffle(images)
        if len(images) < shots:
            raise RuntimeError(
                f"Class '{class_name}' has only {len(images)} train images, "
                f"but {shots}-shot was requested."
            )
        selected = images[:shots]
        train_samples.extend((path, class_to_idx[class_name]) for path in selected)
        remaining_by_class[class_name].extend(images[shots:])

    if test_pool is not None:
        test_samples = [
            (path, class_to_idx[class_name])
            for class_name in classes
            for path in test_pool.get(class_name, [])
        ]
    else:
        test_samples = [
            (path, class_to_idx[class_name])
            for class_name in classes
            for path in remaining_by_class[class_name]
        ]

    if not test_samples:
        raise RuntimeError(f"No test/query samples found for dataset: {dataset_dir}")

    rng.shuffle(train_samples)
    return train_samples, test_samples, classes


class SARClassificationDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("L")
            if self.transform is not None:
                image = self.transform(image)
        return image, label
