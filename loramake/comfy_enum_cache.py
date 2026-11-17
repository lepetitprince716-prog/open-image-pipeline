#!/usr/bin/env python3
"""Cross-process L2 disk cache for ComfyUI ``/object_info`` enumerations.

Extracted from ``loramake.comfy`` (C3). Every gen is a fresh process, so the driver's
in-process cache never survives an invocation; this layer makes a warm preflight free of
network. Safety: preflight re-probes LIVE before failing on a missing name, so a stale entry
can delay a rejection by one round-trip but can never wrongly reject a fresh model.

Leaf module: stdlib only. The two dependencies that would tie it back to the driver — the
default cache location (``loramake.config.ROOT``) and the atomic writer — are parameters, so
``loramake.comfy`` supplies both and keeps ``atomic_write_text`` as its own attribute.

The L1 state (``_ENUM_CACHE``, ``_ENUM_FROM_DISK``) deliberately stays in ``loramake.comfy``:
tests clear it through that module, and a second copy here would silently diverge.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

_ENUM_DISK_SCHEMA = "loramake.object-info-cache/v1"
_ENUM_DISK_TTL_DEFAULT = 900.0
_ENUM_DISK_PRUNE_AGE = 7 * 24 * 3600.0
_ENUM_DISK_LOCK = threading.Lock()


def _enum_disk_enabled() -> bool:
    return os.environ.get("LORAMAKE_ENUM_CACHE", "1").strip().lower() not in {"0", "false", "no", "off"}


def _enum_disk_ttl() -> float:
    try:
        return float(os.environ.get("LORAMAKE_ENUM_CACHE_TTL", str(_ENUM_DISK_TTL_DEFAULT)))
    except ValueError:
        return _ENUM_DISK_TTL_DEFAULT


def enum_disk_path(default: Path) -> Path:
    """The cache file: the ``LORAMAKE_ENUM_CACHE_PATH`` override, else the caller's default."""
    override = os.environ.get("LORAMAKE_ENUM_CACHE_PATH", "").strip()
    return Path(override).expanduser() if override else default


def enum_disk_load(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict) or data.get("schema") != _ENUM_DISK_SCHEMA:
        return {}
    entries = data.get("entries")
    return entries if isinstance(entries, dict) else {}


def enum_disk_read(key: tuple[str, str, str], *, path: Path) -> set[str] | None:
    if not _enum_disk_enabled():
        return None
    entry = enum_disk_load(path).get("|".join(key))
    if not isinstance(entry, dict):
        return None
    names, observed_at = entry.get("names"), entry.get("observed_at")
    if not isinstance(names, list) or not isinstance(observed_at, (int, float)):
        return None
    if not (0.0 <= time.time() - observed_at <= _enum_disk_ttl()):
        return None
    return {n for n in names if isinstance(n, str)}


def enum_disk_write(key: tuple[str, str, str], names: set[str], *,
                    path: Path, write_text) -> None:
    """Best-effort read-modify-write with expired-entry pruning; never fails the caller.

    ``write_text(path, text)`` is the driver's atomic writer, injected so this module keeps no
    dependency on the transport layer.
    """
    if not _enum_disk_enabled():
        return
    now = time.time()
    with _ENUM_DISK_LOCK:
        entries = enum_disk_load(path)
        entries = {k: v for k, v in entries.items()
                   if isinstance(v, dict) and isinstance(v.get("observed_at"), (int, float))
                   and now - v["observed_at"] < _ENUM_DISK_PRUNE_AGE}
        entries["|".join(key)] = {"names": sorted(names), "observed_at": now}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_text(path, json.dumps(
                {"schema": _ENUM_DISK_SCHEMA, "entries": entries}, ensure_ascii=False))
            os.chmod(path, 0o600)
        except OSError:
            pass
