#!/usr/bin/env python3
"""Provision one Hugging Face model file into the local hub cache via hf-mirror.

The imgutils bridge is offline-first (HF_HUB_OFFLINE=1). When a model is missing, fetch it
once with plain GETs against hf-mirror.com (the HEAD handshake used by hf_hub_download does
not survive the mirror's redirects), then write the snapshot layout huggingface_hub expects.

Usage:
    python3 scripts/provision_hf_model.py skytnt/anime-seg isnetis.onnx

Requires Python 3.11+ (older urllib does not follow 308 redirects). Never sends a token.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

MIRROR = "https://hf-mirror.com"
CACHE = Path.home() / ".cache" / "huggingface" / "hub"
CHUNK = 1 << 20


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "loramake-provision/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def download(url: str, target: Path) -> int:
    request = urllib.request.Request(url, headers={"User-Agent": "loramake-provision/1.0"})
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".part")
    total = 0
    with urllib.request.urlopen(request, timeout=300) as response, temp.open("wb") as handle:
        while True:
            block = response.read(CHUNK)
            if not block:
                break
            handle.write(block)
            total += len(block)
            print(f"\r  {total / (1 << 20):.1f} MiB", end="", flush=True)
    print()
    temp.replace(target)
    return total


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    if sys.version_info < (3, 11):
        print("python 3.11+ required (308 redirects)")
        return 2
    repo, filename = argv[1], argv[2]
    meta = json.loads(fetch(f"{MIRROR}/api/models/{repo}"))
    sha = meta["sha"]
    root = CACHE / f"models--{repo.replace('/', '--')}"
    target = root / "snapshots" / sha / filename
    if target.is_file():
        print(f"already present: {target}")
        return 0
    print(f"{repo}@{sha[:12]} -> {target}")
    size = download(f"{MIRROR}/{repo}/resolve/main/{filename}", target)
    refs = root / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(sha, encoding="utf-8")
    print(f"done: {size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
