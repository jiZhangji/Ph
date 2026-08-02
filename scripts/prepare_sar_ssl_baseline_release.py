#!/usr/bin/env python3
"""Prepare the paper baseline checkpoints for a Hugging Face release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


MODELS = {
    "mae": {
        "display_name": "MAE",
        "source": Path("MAE/checkpoint-200.pth"),
        "target": Path("weights/mae/checkpoint-200.pth"),
        "checkpoint_key": "model",
        "pooling": "cls",
    },
    "lomar": {
        "display_name": "LoMaR",
        "source": Path("LoMaR/checkpoint-200.pth"),
        "target": Path("weights/lomar/checkpoint-200.pth"),
        "checkpoint_key": "model",
        "pooling": "cls",
    },
    "fg_mae": {
        "display_name": "FG-MAE",
        "source": Path("FG-MAE/checkpoint-200.pth"),
        "target": Path("weights/fg_mae/checkpoint-200.pth"),
        "checkpoint_key": "model",
        "pooling": "cls",
    },
    "i_jepa": {
        "display_name": "I-JEPA",
        "source": Path("ijepa/jepa-ep200.pth.tar"),
        "target": Path("weights/i_jepa/jepa-ep200.pth.tar"),
        "checkpoint_key": "encoder",
        "pooling": "patch_mean",
    },
    "sar_jepa": {
        "display_name": "SAR-JEPA",
        "source": Path("SAR-JEPA/checkpoint-200.pth"),
        "target": Path("weights/sar_jepa/checkpoint-200.pth"),
        "checkpoint_key": "model",
        "pooling": "cls",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--materialize",
        choices=("auto", "hardlink", "copy"),
        default="auto",
        help="Use hard links when possible to avoid a second multi-GB copy.",
    )
    return parser.parse_args()


def sha256sum(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def materialize(source: Path, target: Path, mode: str) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"Existing target has the wrong size: {target}")
        return "existing"

    if mode in {"auto", "hardlink"}:
        try:
            os.link(source, target)
            return "hardlink"
        except OSError:
            if mode == "hardlink":
                raise

    shutil.copy2(source, target)
    return "copy"


def write_readme(output_dir: Path, records: list[dict]) -> None:
    rows = "\n".join(
        f"| {item['display_name']} | `{item['path']}` | "
        f"`{item['checkpoint_key']}` | `{item['pooling']}` | "
        f"{item['size_bytes'] / 1024**3:.2f} GiB |"
        for item in records
    )
    text = f"""---
license: other
tags:
- synthetic-aperture-radar
- self-supervised-learning
- vision-transformer
---

# SAR SSL paper baseline checkpoints

This private research release contains the five external baseline checkpoints
used by the PhyD-MAE paper comparison. The files are redistributed for
reproducibility from the authors' local SAR-JEPA weight bundle. Consult the
original method repositories and publications for their applicable terms
before any redistribution or non-research use.

| Method | File | State-dict key | Downstream pooling | Size |
|---|---|---|---|---:|
{rows}

The corresponding evaluator is in the PhyD-MAE GitHub repository. It performs
strict encoder coverage checks and handles I-JEPA's no-class-token checkpoint
with mean patch-token pooling.

Verify all files from this directory with:

```bash
sha256sum -c SHA256SUMS
```
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    weights_root = args.weights_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    missing = [
        str(weights_root / spec["source"])
        for spec in MODELS.values()
        if not (weights_root / spec["source"]).is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing required checkpoints:\n" + "\n".join(missing))

    records = []
    checksum_lines = []
    for model_id, spec in MODELS.items():
        source = weights_root / spec["source"]
        target = output_dir / spec["target"]
        link_mode = materialize(source, target, args.materialize)
        checksum = sha256sum(target)
        relative = target.relative_to(output_dir).as_posix()
        record = {
            "model_id": model_id,
            "display_name": spec["display_name"],
            "path": relative,
            "source_relative_path": spec["source"].as_posix(),
            "checkpoint_key": spec["checkpoint_key"],
            "pooling": spec["pooling"],
            "architecture": "ViT-B/16, single-channel SAR input",
            "size_bytes": target.stat().st_size,
            "sha256": checksum,
            "materialization": link_mode,
        }
        records.append(record)
        checksum_lines.append(f"{checksum}  {relative}")
        print(f"Prepared {spec['display_name']}: {relative} ({link_mode})")

    manifest = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "models": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n", encoding="ascii"
    )
    write_readme(output_dir, records)

    total = sum(record["size_bytes"] for record in records)
    print(f"Package ready: {output_dir}")
    print(f"Models: {len(records)}")
    print(f"Checkpoint bytes: {total}")


if __name__ == "__main__":
    main()
