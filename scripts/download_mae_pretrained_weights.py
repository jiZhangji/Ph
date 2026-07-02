#!/usr/bin/env python3
"""Download official MAE ImageNet pretraining checkpoints.

The default checkpoint matches the repository default model
`mae_vit_base_patch16`. It contains MAE pre-trained ViT-Base encoder weights
from the official facebookresearch/mae release. `main_pretrain.py` loads these
weights with `--init_ckpt_scope encoder` and skips task-specific heads.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


CHECKPOINTS = {
    "base": {
        "filename": "mae_pretrain_vit_base.pth",
        "url": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth",
        "md5_prefix": "8cad7c",
    },
    "large": {
        "filename": "mae_pretrain_vit_large.pth",
        "url": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth",
        "md5_prefix": "b8b06e",
    },
    "huge": {
        "filename": "mae_pretrain_vit_huge.pth",
        "url": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth",
        "md5_prefix": "9bdbb0",
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def human_size(num_bytes: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, force: bool) -> None:
    if target.exists() and not force and target.stat().st_size > 0:
        print(f"{target.name}: exists, skip download ({human_size(target.stat().st_size)})")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"{target.name}: downloading from {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "sar-pretrain-downloader"})

    with tempfile.NamedTemporaryFile(delete=False, dir=target.parent, suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                total = int(response.headers.get("Content-Length", "0") or 0)
                copied = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    tmp.write(chunk)
                    copied += len(chunk)
                    if total > 0:
                        percent = copied * 100 / total
                        print(
                            f"\r  downloaded {human_size(copied)} / {human_size(total)} ({percent:5.1f}%)",
                            end="",
                            flush=True,
                        )
                    else:
                        print(f"\r  downloaded {human_size(copied)}", end="", flush=True)
            print()
            shutil.move(str(tmp_path), target)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=CHECKPOINTS.keys(), default="base")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / "weights",
        help="directory to save downloaded checkpoints",
    )
    parser.add_argument("--force", action="store_true", help="re-download existing file")
    parser.add_argument(
        "--no-md5-check",
        action="store_true",
        help="skip the official md5-prefix verification",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    meta = CHECKPOINTS[args.model]
    target = args.output_dir.resolve() / meta["filename"]

    try:
        download(meta["url"], target, args.force)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"ERROR: failed to download {meta['url']}: {exc}", file=sys.stderr)
        return 1

    digest = md5sum(target)
    print(f"{target.name}: md5 {digest}")
    if not args.no_md5_check and not digest.startswith(meta["md5_prefix"]):
        print(
            f"ERROR: md5 prefix mismatch, expected {meta['md5_prefix']}..., got {digest}",
            file=sys.stderr,
        )
        return 1

    print(f"Saved: {target}")
    print("\nUse it for 2xH100 training with:")
    print(f"  export INIT_CKPT={target}")
    print("  export INIT_CKPT_SCOPE=encoder")
    print("  bash scripts/run_pretrain_2xh100_nohup.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
