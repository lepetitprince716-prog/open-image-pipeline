#!/usr/bin/env python3
"""A per-box VRAM lease: keep the models resident between renders, release when idle.

Before this, ``gen.py`` POSTed ``/free`` after every single ``--run``. On a burst of renders
that reloads the checkpoint and its LoRAs from disk once per image for no reason — the box is
about to be asked for the same weights again. The Studio worker already knew better and kept a
box resident across queued jobs (``app/jobs.py``'s ``keep_resident``); the CLI had no equivalent
because a ``gen.py`` process exits long before "idle" can be observed.

The lease is that missing memory. Every finished render stamps a small file; a separate reaper
watches the stamp and releases the box once it has gone quiet. Three conditions must all hold
before anything is released, and the middle one is the important one:

1. the lease has been idle for at least ``idle_s`` (default 300 — the user's call, 2026-09-17);
2. the box's queue is empty — somebody else's render must never be freed out from under them;
3. the lease has not already been released.

Leaf module: stdlib only, and no path authority. The lease location and the atomic writer are
parameters, exactly as ``comfy_enum_cache`` takes them, so this file cannot decide on its own
that ``outputs/`` lives anywhere in particular. Callers supply both.

Nothing here performs I/O against a box. Deciding is this module's job; releasing is the
reaper's (``scripts/vram_reaper.py``).
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path

LEASE_SCHEMA = "loramake.vram-lease/v1"
IDLE_DEFAULT_S = 300.0
# A reaper that dies without cleaning up must not keep the box leased forever. Any lock older
# than this is stale and may be taken over; it is generously larger than one poll cycle.
LOCK_TTL_S = 120.0
# Poll cadence while a reaper waits out the idle window. The lease is minutes long, so there is
# nothing to gain from checking more often than this.
POLL_S = 15.0


def lease_path(data_dir: Path, box: str) -> Path:
    """Where this box's lease lives. ``data_dir`` is injected; this module owns only the name."""
    return Path(data_dir) / f"vram_lease_{box}.json"


def lock_path(data_dir: Path, box: str) -> Path:
    return Path(data_dir) / f"vram_lease_{box}.lock"


def idle_seconds(default: float = IDLE_DEFAULT_S) -> float:
    """``LORAMAKE_VRAM_IDLE_S``, else the default. A bad value falls back rather than raising:
    a finished render must not fail at the release step over a typo in an env var."""
    raw = os.environ.get("LORAMAKE_VRAM_IDLE_S", "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def enabled() -> bool:
    """``LORAMAKE_VRAM_LEASE=0`` restores the historical release-after-every-render behaviour."""
    return os.environ.get("LORAMAKE_VRAM_LEASE", "1").strip().lower() not in {"0", "false", "no", "off"}


def read(path: Path) -> dict | None:
    """The lease, or None when it is absent or unreadable.

    An unparseable lease reads as absent on purpose: the consequence is one extra release, which
    is the safe direction on a box shared with a game.
    """
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or doc.get("schema") != LEASE_SCHEMA:
        return None
    return doc


def touch(path: Path, *, box: str, host: str, write_text: Callable[[Path, str], None],
          now: float | None = None, owner: str | None = None) -> dict:
    """Record that this box was just used, and clear any earlier release."""
    doc = {
        "schema": LEASE_SCHEMA,
        "box": box,
        "host": host,
        "touched_at": time.time() if now is None else now,
        "released": False,
        "owner": owner or f"pid:{os.getpid()}",
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_text(Path(path), json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    return doc


def mark_released(path: Path, *, write_text: Callable[[Path, str], None], ok: bool,
                  now: float | None = None, note: str = "") -> dict | None:
    """Close the lease after a release attempt. ``ok=False`` records a release that did not land
    so the next reader can see it rather than assuming the box is free."""
    doc = read(path)
    if doc is None:
        return None
    doc["released"] = True
    doc["released_at"] = time.time() if now is None else now
    doc["release_ok"] = bool(ok)
    if note:
        doc["release_note"] = note
    write_text(Path(path), json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    return doc


def release_decision(doc: dict | None, *, now: float, idle_s: float,
                     queue_empty: bool | None) -> tuple[bool, str]:
    """Whether the box may be released now, and why not when it may not.

    ``queue_empty=None`` means the queue could not be read. That is a refusal, not a default:
    an unreachable box tells us nothing about whose work is on it.
    """
    if doc is None:
        return False, "no lease"
    if doc.get("released"):
        return False, "already released"
    touched = doc.get("touched_at")
    if not isinstance(touched, (int, float)):
        return False, "lease has no usable timestamp"
    idle = now - float(touched)
    if idle < idle_s:
        return False, f"idle {idle:.0f}s of {idle_s:.0f}s"
    if queue_empty is None:
        return False, "queue unreadable"
    if not queue_empty:
        return False, "queue busy"
    return True, f"idle {idle:.0f}s"


def acquire_lock(path: Path, *, now: float | None = None, ttl_s: float = LOCK_TTL_S,
                 pid: int | None = None) -> bool:
    """Single-flight guard so two reapers never watch the same box.

    O_EXCL rather than a lock library: the holder writes its pid and the time, and a lock older
    than ``ttl_s`` is taken over. A reaper killed mid-wait therefore delays the next one by at
    most one TTL instead of blocking it forever.
    """
    now = time.time() if now is None else now
    pid = os.getpid() if pid is None else pid
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"pid": pid, "acquired_at": now}) + "\n"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        held = _lock_holder(path)
        if held is not None and now - held < ttl_s:
            return False
        # Stale. Replace it in place; the loser of a race just sees a fresher holder next poll.
        try:
            path.write_text(payload, encoding="utf-8")
        except OSError:
            return False
        return True
    except OSError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(payload)
    return True


def release_lock(path: Path, *, pid: int | None = None) -> None:
    """Drop the lock.

    With ``pid`` the file is removed only while it still names that pid — a lock that was
    taken over after going stale belongs to the new holder and must survive our exit.
    """
    path = Path(path)
    if pid is not None:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if doc.get("pid") != pid:
            return
    try:
        path.unlink()
    except OSError:
        pass


def touch_lock(path: Path, *, now: float | None = None, pid: int | None = None) -> bool:
    """Refresh the held lock so a long watch never reads as stale.

    The holder rewrites ``acquired_at`` once per poll; ``poll_s`` is far under ``ttl_s``, so
    a live reaper's lock never ages out. False means the file is gone or names another pid —
    the lock was taken over while this process was suspended past the TTL — and the caller
    must stop watching rather than run as a second reaper on the same box.
    """
    now = time.time() if now is None else now
    pid = os.getpid() if pid is None else pid
    path = Path(path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if doc.get("pid") != pid:
        return False
    try:
        path.write_text(json.dumps({"pid": pid, "acquired_at": now}) + "\n",
                        encoding="utf-8")
    except OSError:
        return False
    return True


def _lock_holder(path: Path) -> float | None:
    """When the current lock was taken, or None if that cannot be read (treated as stale)."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        return float(doc["acquired_at"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
