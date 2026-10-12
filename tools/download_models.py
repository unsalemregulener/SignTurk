"""Download the model weights listed in model-assets.json and verify them.

    python tools/download_models.py          # download missing or changed files
    python tools/download_models.py --check  # only check that every URL answers
                                             # with the expected size (no download)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "model-assets.json"
USER_AGENT = "SignTurk-model-downloader/1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(asset: dict, force: bool = False) -> None:
    target = ROOT / asset["path"]
    expected_hash = asset["sha256"]
    expected_size = int(asset["bytes"])

    if target.exists() and not force:
        if target.stat().st_size == expected_size and sha256(target) == expected_hash:
            print(f"verified  {asset['path']}")
            return
        print(f"replacing {asset['path']} (checksum mismatch)")

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(asset["url"], headers={"User-Agent": USER_AGENT})
    print(f"download  {asset['name']} -> {asset['path']}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        size = partial.stat().st_size
        if size != expected_size:
            raise RuntimeError(f"size mismatch for {asset['name']}: expected {expected_size}, got {size}")
        actual_hash = sha256(partial)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"checksum mismatch for {asset['name']}: expected {expected_hash}, got {actual_hash}"
            )
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)


def check(asset: dict) -> bool:
    """HEAD the URL (following redirects) and compare Content-Length."""
    request = urllib.request.Request(asset["url"], method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            length = int(response.headers.get("Content-Length", "-1"))
    except Exception as exc:  # noqa: BLE001 - report every failure the same way
        print(f"missing   {asset['name']}: {exc}")
        return False
    if length != int(asset["bytes"]):
        print(f"mismatch  {asset['name']}: expected {asset['bytes']} bytes, server reports {length}")
        return False
    print(f"ok        {asset['name']} ({length} bytes)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="verify the URLs without downloading")
    parser.add_argument("--force", action="store_true", help="download even if a local file verifies")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()

    assets = json.loads(args.manifest.read_text(encoding="utf-8"))["assets"]
    if args.check:
        return 0 if all([check(asset) for asset in assets]) else 1
    for asset in assets:
        download(asset, force=args.force)
    print(f"ready: {len(assets)} verified asset(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
