#!/usr/bin/env python3
"""Download the SAR pretraining and classification datasets from ModelScope.

The pretraining code reads images recursively from --data_path, so this script
downloads the official zip files and extracts them under dataset/modelscope by
default. After extraction, point DATA_PATH at the extracted pretraining folder.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


DATASET_FILES = {
    "Pretraining_dataset.zip": [
        "https://modelscope.cn/datasets/shimian123/sar-pretrain/file/view/master/Pretraining_dataset.zip?id=203554&status=2",
    ],
    "classification_dataset.zip": [
        "https://modelscope.cn/datasets/shimian123/sar-pretrain/file/view/master/classification_dataset.zip",
    ],
}


class DownloadError(RuntimeError):
    pass


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


def reporthook(blocks: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        downloaded = blocks * block_size
        print(f"\r  downloaded {human_size(downloaded)}", end="", flush=True)
        return

    downloaded = min(blocks * block_size, total_size)
    percent = downloaded * 100 / total_size
    print(
        f"\r  downloaded {human_size(downloaded)} / {human_size(total_size)} ({percent:5.1f}%)",
        end="",
        flush=True,
    )


def looks_like_zip(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(4) == b"PK\x03\x04"


def download_one(filename: str, urls: list[str], output_dir: Path, force: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / filename
    partial = target.with_suffix(target.suffix + ".part")

    if target.exists() and not force:
        if looks_like_zip(target):
            print(f"{filename}: exists, skip download ({human_size(target.stat().st_size)})")
            return target
        print(f"{filename}: existing file is not a zip, re-downloading")

    errors: list[str] = []
    for url in urls:
        print(f"{filename}: downloading from {url}")
        if partial.exists():
            partial.unlink()
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 sar-pretrain-downloader"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                with partial.open("wb") as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            print()
            if not looks_like_zip(partial):
                errors.append(f"{url}: response was not a zip file")
                continue
            partial.replace(target)
            print(f"{filename}: saved to {target} ({human_size(target.stat().st_size)})")
            return target
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print()
            errors.append(f"{url}: {exc}")

    raise DownloadError(f"failed to download {filename}\n" + "\n".join(f"  - {e}" for e in errors))


def download_one_with_modelscope_cli(filename: str, output_dir: Path, force: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / filename

    if target.exists() and not force and looks_like_zip(target):
        print(f"{filename}: exists, skip download ({human_size(target.stat().st_size)})")
        return target

    modelscope_bin = shutil.which("modelscope")
    if modelscope_bin:
        command = [
            modelscope_bin,
            "download",
            "--dataset",
            "shimian123/sar-pretrain",
            filename,
            "--local_dir",
            str(output_dir),
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "modelscope.cli.cli",
            "download",
            "--dataset",
            "shimian123/sar-pretrain",
            filename,
            "--local_dir",
            str(output_dir),
        ]
    print(f"{filename}: direct URL failed, trying ModelScope CLI")
    try:
        subprocess.run(command, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise DownloadError(
            f"ModelScope CLI failed for {filename}: {exc}\n"
            "Install it with `pip install modelscope -U`, then retry."
        ) from exc

    if not target.exists():
        matches = list(output_dir.rglob(filename))
        if matches:
            shutil.move(str(matches[0]), target)

    if not target.exists() or not looks_like_zip(target):
        raise DownloadError(f"ModelScope CLI did not create a valid zip file: {target}")

    print(f"{filename}: saved to {target} ({human_size(target.stat().st_size)})")
    return target


def extract_one(zip_path: Path, extract_root: Path, force: bool) -> Path:
    destination = extract_root / zip_path.stem
    marker = destination / ".extracted_from_zip"

    if marker.exists() and not force:
        print(f"{zip_path.name}: already extracted to {destination}")
        return destination

    destination.mkdir(parents=True, exist_ok=True)
    print(f"{zip_path.name}: extracting to {destination}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)
    marker.write_text(zip_path.name + "\n", encoding="utf-8")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root() / "dataset" / "modelscope",
        help="directory used for downloaded zip files and extracted folders",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="only download zip files; do not extract them",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download and re-extract even if files already exist",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    zip_dir = output_dir / "zips"
    extract_root = output_dir / "extracted"

    try:
        zip_paths = []
        for filename, urls in DATASET_FILES.items():
            try:
                zip_paths.append(download_one(filename, urls, zip_dir, args.force))
            except DownloadError as direct_error:
                print(f"{filename}: {direct_error}", file=sys.stderr)
                zip_paths.append(download_one_with_modelscope_cli(filename, zip_dir, args.force))

        extracted: list[Path] = []
        if not args.no_extract:
            extracted = [extract_one(path, extract_root, args.force) for path in zip_paths]

        print("\nDone.")
        print(f"Zip files: {zip_dir}")
        if extracted:
            print("Extracted folders:")
            for path in extracted:
                print(f"  {path}")
            print("\nFor pretraining, for example:")
            print(f"  DATA_PATH={extract_root / 'Pretraining_dataset'} bash scripts/pretrain_4xa100.sh")
        return 0
    except DownloadError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print(
            "\nIf ModelScope requires authentication in your environment, run "
            "`modelscope login`, then retry this script.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
