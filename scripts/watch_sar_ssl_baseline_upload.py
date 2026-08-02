#!/usr/bin/env python3
"""Poll and verify the paper-baseline Hugging Face weight release."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi


METADATA_FILES = ("README.md", "manifest.json", "SHA256SUMS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--completion-file", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def lfs_sha256(sibling: object) -> str | None:
    lfs = getattr(sibling, "lfs", None)
    if lfs is None:
        return None
    if isinstance(lfs, dict):
        return lfs.get("sha256")
    return getattr(lfs, "sha256", None)


def remote_files(api: HfApi, repo_id: str) -> dict[str, object]:
    info = api.repo_info(
        repo_id=repo_id,
        repo_type="model",
        files_metadata=True,
    )
    return {item.rfilename: item for item in info.siblings}


def verify_weights(
    expected: list[dict[str, object]], remote: dict[str, object]
) -> tuple[bool, list[dict[str, object]]]:
    rows = []
    complete = True
    for model in expected:
        path = str(model["path"])
        item = remote.get(path)
        actual_size = getattr(item, "size", None) if item else None
        actual_sha = lfs_sha256(item) if item else None
        size_ok = actual_size == model["size_bytes"]
        sha_ok = actual_sha == model["sha256"]
        ok = size_ok and sha_ok
        complete &= ok
        rows.append(
            {
                "path": path,
                "present": item is not None,
                "size_ok": size_ok,
                "sha256_ok": sha_ok,
                "remote_size": actual_size,
                "remote_sha256": actual_sha,
            }
        )
    return complete, rows


def upload_metadata(api: HfApi, repo_id: str, package_dir: Path) -> str | None:
    commit_oid = None
    for name in METADATA_FILES:
        result = api.upload_file(
            path_or_fileobj=str(package_dir / name),
            path_in_repo=name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Refresh {name} after verified weight upload",
        )
        commit_oid = getattr(result, "oid", commit_oid)
    return commit_oid


def main() -> int:
    args = parse_args()
    package_dir = args.package_dir.resolve()
    manifest = json.loads((package_dir / "manifest.json").read_text("utf-8"))
    expected = manifest["models"]
    api = HfApi()

    while True:
        try:
            remote = remote_files(api, args.repo_id)
            complete, rows = verify_weights(expected, remote)
            print(
                json.dumps(
                    {
                        "checked_utc": utc_now(),
                        "complete": complete,
                        "weights": rows,
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
            if complete:
                commit_oid = upload_metadata(api, args.repo_id, package_dir)
                remote = remote_files(api, args.repo_id)
                verified, rows = verify_weights(expected, remote)
                metadata_ok = all(name in remote for name in METADATA_FILES)
                result = {
                    "completed_utc": utc_now(),
                    "repo_id": args.repo_id,
                    "commit_oid": commit_oid,
                    "weights_verified": verified,
                    "metadata_verified": metadata_ok,
                    "weights": rows,
                }
                args.completion_file.parent.mkdir(parents=True, exist_ok=True)
                args.completion_file.write_text(
                    json.dumps(result, indent=2, ensure_ascii=True) + "\n",
                    encoding="utf-8",
                )
                print(json.dumps(result, ensure_ascii=True), flush=True)
                return 0 if verified and metadata_ok else 1
        except Exception as exc:  # Keep transient network failures resumable.
            print(
                json.dumps(
                    {"checked_utc": utc_now(), "error": repr(exc)},
                    ensure_ascii=True,
                ),
                flush=True,
            )

        if args.once:
            return 2
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
