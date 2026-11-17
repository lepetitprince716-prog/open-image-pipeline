#!/usr/bin/env python3
"""Release the GPU on a box: POST /free to its ComfyUI instances and report VRAM before and after.

    freeboxa            # BOXA: the Qwen sidecar (:18189 loopback) and production (:18188)
    freeboxa --boxb      # BOXB instead (:18388)
    freeboxa --mode vram   # unload models only; default "full" also drops the cache

Renders leave models loaded on purpose; run this when switching models or before gaming.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from loramake.comfy import FREE_MODES, _request_json, free_vram  # noqa: E402
from loramake.qwen21_endpoints import resolve_qwen21_host  # noqa: E402

BOX_HOSTS = {
    "boxa": ("http://127.0.0.1:18188",),
    "boxb": ("http://127.0.0.1:18388",),
}


def _vram_mb(host: str) -> tuple[int, int] | None:
    try:
        device = (_request_json(host, "/system_stats", timeout=15.0).get("devices") or [{}])[0]
    except Exception:
        return None
    total, free = device.get("vram_total") or 0, device.get("vram_free") or 0
    return (free // 1048576, total // 1048576) if total else None


def free_host(host: str, mode: str) -> str:
    before = _vram_mb(host)
    if before is None:
        return f"{host:<24} unreachable"
    os.environ["LORAMAKE_FREE_MODE"] = mode  # free_vram reads the mode from the environment
    if not free_vram(host):
        return f"{host:<24} /free FAILED"
    after = _vram_mb(host) or before
    return f"{host:<24} freed: {before[0]} -> {after[0]} MB free of {before[1]}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--boxb", action="store_true", help="free BOXB instead of BOXA")
    ap.add_argument("--mode", choices=sorted(FREE_MODES), default="full")
    args = ap.parse_args(argv)
    hosts = BOX_HOSTS["boxb"] if args.boxb else (resolve_qwen21_host("boxa"), *BOX_HOSTS["boxa"])
    rc = 0
    for host in hosts:
        line = free_host(host, args.mode)
        print(line)
        if "FAILED" in line:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
