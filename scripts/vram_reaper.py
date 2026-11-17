#!/usr/bin/env python3
"""Release a ComfyUI box's VRAM once its lease has gone quiet.

``gen.py`` used to POST ``/free`` the instant a render finished, which reloaded the checkpoint
and its LoRAs from disk on the very next image of a burst. It now stamps a lease instead
(``loramake.vram_lease``) and leaves this reaper watching it: the box stays warm between
renders and is released after ``--idle`` seconds of silence — 300 by default, the user's call
on 2026-09-17.

Run detached; ``gen.py`` starts one automatically and the lock makes that idempotent.

    python3 scripts/vram_reaper.py --box boxa                 # watch until idle, then release
    python3 scripts/vram_reaper.py --box boxa --once --json   # evaluate once and report

Safety, in the order it matters:

* the queue is checked immediately before every release, so a render another session queued is
  never freed out from under it — and a queue that cannot be read is a refusal, not a default;
* the lock means two reapers never watch one box;
* after a release the box is read back. ComfyUI 0.33 has wedged on ``/free`` before
  (docs/handoffs/2026-09-01-comfyui-perf-opt/), and the release now happens when nobody is
  watching, so a box that stops answering is written into the lease and reported rather than
  retried.

Exit: 0 released / nothing to do, 1 the release did not land, 3 gave up at ``--max-wait``,
4 the reaper lock was taken over.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from loramake import vram_lease as lease  # noqa: E402
from loramake.comfy import atomic_write_text, free_vram, observe_queue  # noqa: E402
from loramake.comfy_endpoints import DIRECT_COMFY_HOSTS, resolve_comfy_host  # noqa: E402

BOXES = tuple(DIRECT_COMFY_HOSTS)
# A detached reaper must not outlive its usefulness if the lease file is never touched again.
MAX_WAIT_DEFAULT_S = 3600.0
VERIFY_TIMEOUT_S = 10.0


def data_dir(root: Path = REPO) -> Path:
    """Where leases live. Mirrors the app/data convention the box-reachability cache uses."""
    override = os.environ.get("LORAMAKE_VRAM_LEASE_DIR", "").strip()
    return Path(override).expanduser() if override else root / "app" / "data"


def queue_is_empty(host: str) -> bool | None:
    """True/False, or None when the queue could not be read at all."""
    try:
        observed = observe_queue(host)
    except Exception:
        # QueueStateError for a refused/ambiguous /queue, transport errors for everything else.
        # Both mean the same thing here: we do not know whose work is on the box.
        return None
    return not observed["running"] and not observed["pending"]


def verify_alive(host: str) -> bool:
    """Did the box survive the release? ComfyUI 0.33 has wedged on /free before."""
    try:
        from loramake.comfy import _request_json
        _request_json(host, "/system_stats", timeout=VERIFY_TIMEOUT_S)
        return True
    except Exception:
        return False


def release(path: Path, host: str) -> dict:
    """POST /free, read the box back, and close the lease with whichever happened."""
    freed = free_vram(host)
    alive = verify_alive(host)
    note = "" if alive else "box stopped answering /system_stats after /free"
    lease.mark_released(path, write_text=atomic_write_text, ok=freed and alive, note=note)
    return {"action": "released", "free_ok": freed, "alive_after": alive, "note": note}


def step(path: Path, *, idle_s: float, now: float | None = None) -> dict:
    """One evaluation. Releases when all three conditions hold, otherwise says why not."""
    now = time.time() if now is None else now
    doc = lease.read(path)
    if doc is None:
        return {"action": "none", "reason": "no lease"}
    if doc.get("released"):
        return {"action": "none", "reason": "already released"}
    host = doc.get("host") or resolve_comfy_host(doc.get("box", "boxa"))

    # Cheap check first: the queue costs a request, the clock costs nothing.
    ok, why = lease.release_decision(doc, now=now, idle_s=idle_s, queue_empty=True)
    if not ok:
        return {"action": "wait", "reason": why, "host": host}
    ok, why = lease.release_decision(doc, now=now, idle_s=idle_s, queue_empty=queue_is_empty(host))
    if not ok:
        return {"action": "wait", "reason": why, "host": host}
    return {**release(path, host), "reason": why, "host": host}


def watch(path: Path, *, idle_s: float, poll_s: float, max_wait_s: float,
          sleep=time.sleep, clock=time.monotonic, lock: Path | None = None) -> dict:
    """Poll until the box is released, the lease closes, or ``max_wait_s`` runs out.

    ``lock`` is the held single-flight file: each poll refreshes it so a watch longer than
    ``LOCK_TTL_S`` never reads as stale, and a refresh that finds another holder ends the
    watch with ``action: lock_lost`` instead of running two reapers on one box.
    """
    deadline = clock() + max_wait_s
    last: dict = {"action": "none", "reason": "no lease"}
    while True:
        if lock is not None and not lease.touch_lock(lock):
            return {"action": "lock_lost", "reason": "reaper lock lost to another reaper"}
        last = step(path, idle_s=idle_s)
        if last["action"] in {"released", "none"}:
            return last
        if clock() >= deadline:
            return {**last, "action": "gave_up", "reason": f"{last['reason']}; max-wait reached"}
        sleep(poll_s)


def run(box: str, *, idle_s: float, poll_s: float, max_wait_s: float, once: bool,
        root: Path = REPO) -> dict:
    path = lease.lease_path(data_dir(root), box)
    lock = lease.lock_path(data_dir(root), box)
    if not lease.acquire_lock(lock):
        return {"box": box, "action": "none", "reason": "another reaper holds this box"}
    try:
        result = step(path, idle_s=idle_s) if once else watch(
            path, idle_s=idle_s, poll_s=poll_s, max_wait_s=max_wait_s, lock=lock)
    finally:
        lease.release_lock(lock, pid=os.getpid())
    return {"box": box, **result}


def exit_code(result: dict) -> int:
    if result["action"] == "gave_up":
        return 3
    if result["action"] == "lock_lost":
        return 4
    if result["action"] == "released" and not result.get("free_ok", False):
        return 1
    if result["action"] == "released" and not result.get("alive_after", True):
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog=("Exit: 0 released / nothing to do, 1 the release did not land, "
                "3 gave up at --max-wait, 4 the reaper lock was taken over."),
    )
    parser.add_argument("--box", choices=BOXES, required=True)
    parser.add_argument("--idle", type=float, default=None,
                        help=f"seconds of silence before releasing (default {lease.IDLE_DEFAULT_S:g}, "
                             "or LORAMAKE_VRAM_IDLE_S)")
    parser.add_argument("--poll", type=float, default=lease.POLL_S,
                        help=f"seconds between checks (default {lease.POLL_S:g})")
    parser.add_argument("--max-wait", type=float, default=MAX_WAIT_DEFAULT_S,
                        help=f"give up after this long (default {MAX_WAIT_DEFAULT_S:g}s)")
    parser.add_argument("--once", action="store_true", help="evaluate once and exit; do not watch")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    result = run(args.box, idle_s=lease.idle_seconds() if args.idle is None else args.idle,
                 poll_s=args.poll, max_wait_s=args.max_wait, once=args.once)
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(f"[{result['box']}] {result['action']}: {result['reason']}")
    return exit_code(result)


if __name__ == "__main__":
    sys.exit(main())
