#!/usr/bin/env python3
"""Resolve ``z3:``/``zzz:`` media references to local files, reusing what is already here.

Most rendered output lives on the Windows boxes, not on this Mac, so showing it in the
gallery means having a local copy. This keeps one content-addressed cache under
``outputs/_gallery/remote_cache/`` and **only pulls when the cache misses** — a reference
that has been fetched before costs nothing and touches the network not at all.

Transport is the managed wrapper at ``/usr/local/bin/claude-scp`` and nothing else. Bare
``scp``, ad-hoc aliases and proxy overrides are all out of bounds here; a failure is
reported with its original error rather than retried down some other path.

Usage:
    python3 scripts/remote_media.py z3:E:/path/to/image.png [...] [--json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from config import OUTPUTS

SCP = "/usr/local/bin/claude-scp"
CACHE_PARTS = ("_gallery", "remote_cache")
# Each box is reachable over Tailscale or the Cloudflare tunnel. Only these four aliases
# exist; anything else is a new network path and is not this script's call to make.
ALIASES: dict[str, tuple[str, ...]] = {
    "z3": ("z3-ts", "z3-cf"),
    "zzz": ("zzz-ts", "zzz-cf"),
}
REF_RE = re.compile(r"\A(?P<box>z3|zzz):(?P<path>[^\r\n]+)\Z", re.IGNORECASE)
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".webm"}
PULL_TIMEOUT_S = 300


class RemoteMediaError(RuntimeError):
    """A user-facing resolution error."""


def is_remote_ref(raw: str) -> bool:
    """True when a spec entry names a box rather than a local path."""
    return REF_RE.match(raw.strip()) is not None


def _parse(raw: str) -> tuple[str, str]:
    match = REF_RE.match(raw.strip())
    if match is None:
        raise RemoteMediaError(f"not a remote reference: {raw}")
    box = match.group("box").lower()
    remote_path = match.group("path").strip()
    if not remote_path:
        raise RemoteMediaError(f"remote reference has no path: {raw}")
    # The reference becomes one scp argument. Anything that could end a token and start
    # another is refused outright rather than quoted and hoped for.
    if any(ch in remote_path for ch in ('"', "'", "`", "$", ";", "|", "&", "\\")):
        raise RemoteMediaError(f"remote path contains unsupported characters: {raw}")
    if Path(remote_path).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise RemoteMediaError(f"unsupported media extension: {raw}")
    return box, remote_path


def cache_path(box: str, remote_path: str) -> Path:
    """Where one reference caches. Keyed by the full remote path, so two files with the
    same basename on different drives never collide."""
    root = (OUTPUTS / CACHE_PARTS[0] / CACHE_PARTS[1] / box).resolve()
    digest = hashlib.sha1(remote_path.encode("utf-8")).hexdigest()[:12]
    name = Path(remote_path.replace("\\", "/")).name
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", name) or "media"
    return root / f"{digest}-{safe}"


def _pull(box: str, remote_path: str, destination: Path, alias: str | None) -> str:
    """Copy one remote file into the cache. Returns the alias that worked."""
    if not os.path.isfile(SCP) or not os.access(SCP, os.X_OK):
        raise RemoteMediaError(
            f"managed transport {SCP} is not available; remote media cannot be fetched"
        )
    candidates = (alias,) if alias else ALIASES[box]
    if alias and alias not in ALIASES[box]:
        raise RemoteMediaError(
            f"alias {alias} is not one of {', '.join(ALIASES[box])}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for candidate in candidates:
        fd, raw_tmp = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
        os.close(fd)
        tmp = Path(raw_tmp)
        try:
            proc = subprocess.run(
                [SCP, f"{candidate}:{remote_path}", str(tmp)],
                capture_output=True, text=True, timeout=PULL_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            tmp.unlink(missing_ok=True)
            errors.append(f"{candidate}: timed out after {PULL_TIMEOUT_S}s")
            continue
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise RemoteMediaError(f"cannot run {SCP}: {exc}") from None

        if proc.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 0:
            os.replace(tmp, destination)
            return candidate
        tmp.unlink(missing_ok=True)
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        errors.append(f"{candidate}: {detail[-1] if detail else f'exit {proc.returncode}'}")

    raise RemoteMediaError(
        f"could not fetch {box}:{remote_path} — " + "; ".join(errors)
        + ". Check `claude-net status` and the box's reachability; do not change the "
          "network configuration to work around this."
    )


def resolve(raw: str, *, alias: str | None = None, refresh: bool = False) -> dict[str, object]:
    """Return the local path for one reference, fetching only on a cache miss."""
    box, remote_path = _parse(raw)
    destination = cache_path(box, remote_path)
    if destination.is_file() and destination.stat().st_size > 0 and not refresh:
        return {
            "ref": raw, "box": box, "remote_path": remote_path,
            "path": str(destination), "cached": True, "alias": None,
        }
    used = _pull(box, remote_path, destination, alias)
    return {
        "ref": raw, "box": box, "remote_path": remote_path,
        "path": str(destination), "cached": False, "alias": used,
    }


def resolve_all(refs: Sequence[str], *, alias: str | None = None,
                refresh: bool = False) -> list[dict[str, object]]:
    """Resolve a mixed list, leaving non-remote entries untouched."""
    resolved: list[dict[str, object]] = []
    for raw in refs:
        if is_remote_ref(raw):
            resolved.append(resolve(raw, alias=alias, refresh=refresh))
        else:
            resolved.append({"ref": raw, "box": None, "remote_path": None,
                             "path": raw, "cached": True, "alias": None})
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Resolve z3:/zzz: media references into local cached files"
    )
    parser.add_argument("refs", nargs="+", metavar="REF")
    parser.add_argument("--alias", help="force one SSH alias (default: try each for the box)")
    parser.add_argument("--refresh", action="store_true", help="re-fetch even on a cache hit")
    parser.add_argument("--json", action="store_true", help="print one JSON array")
    args = parser.parse_args(argv)

    try:
        records = resolve_all(args.refs, alias=args.alias, refresh=args.refresh)
    except RemoteMediaError as exc:
        parser.exit(2, f"remote_media.py: error: {exc}\n")

    if args.json:
        print(json.dumps(records, ensure_ascii=False, separators=(",", ":")))
    else:
        for record in records:
            marker = "cached" if record["cached"] else f"fetched via {record['alias']}"
            print(f"{record['path']}  ({marker})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
