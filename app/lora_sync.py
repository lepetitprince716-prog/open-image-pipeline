"""Manual LoRA-catalog refresh for the app.

Drift detection is one cheap request (Z3's live LoraLoader enum vs the on-disk index);
an actual rebuild (lora_catalog build + embed, always as a pair so vectors never lag the
index) runs as subprocesses so the server process never imports sentence-transformers.
Single-flight: a refresh call while one is running reports already_running instead of
stacking rebuilds. The catalog JSON is read per-request by app.meta, so a finished
rebuild is visible to the UI immediately.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import lora_catalog  # noqa: E402  (scripts/lora_catalog.py — _comfy_lora_names/INDEX_PATH)

CATALOG_SCRIPT = ROOT / "scripts" / "lora_catalog.py"
STEP_TIMEOUT_S = 600

_lock = threading.Lock()
_running = False
_last: dict | None = None  # outcome of the most recent refresh run


def _index_files() -> set[str]:
    try:
        return {r.get("file") for r in json.loads(lora_catalog.INDEX_PATH.read_text())}
    except (OSError, json.JSONDecodeError):
        return set()


def _live_names() -> set[str]:
    """Z3's current LoraLoader enum; empty set when the box is unreachable (fail open)."""
    return set(lora_catalog._comfy_lora_names(strict=False))


def _zzz_remap_delta() -> int:
    """How many catalog entries would change zzz_file if rematched against ZZZ's live enum.

    Rebuilding owns the zzz_file data, so ZZZ-side changes (a LoRA added, removed or
    renamed there) must count as drift even when Z3 is unchanged. The live enum can't be
    compared to the stored values directly — ZZZ carries names the Z3-canonical catalog
    never maps — so re-run the pure matcher and count differences. 0 when ZZZ is
    unreachable or the index predates zzz_file (unknown is not drift; fail open)."""
    try:
        records = json.loads(lora_catalog.INDEX_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return 0
    if not records or not any("zzz_file" in r for r in records):
        return 0
    live = set(lora_catalog._comfy_lora_names(host=lora_catalog.ZZZ_HOST, strict=False))
    if not live:
        return 0
    files = [r.get("file") for r in records]
    fresh = lora_catalog._match_zzz(files, live)
    return sum(1 for r in records if r.get("zzz_file") != fresh.get(r.get("file")))


def drift() -> dict:
    live = _live_names()
    if not live:
        return {"drift": False, "error": "Z3 LoraLoader enum unreachable"}
    idx = _index_files()
    out = {"drift": live != idx, "added": len(live - idx), "removed": len(idx - live)}
    if not out["drift"]:
        zzz_changed = _zzz_remap_delta()
        if zzz_changed:
            out.update(drift=True, zzz_changed=zzz_changed)
    return out


def _run(d: dict) -> None:
    global _running, _last
    t0 = time.monotonic()
    steps, ok = [], True
    for step in ("build", "embed"):
        try:
            proc = subprocess.run([sys.executable, str(CATALOG_SCRIPT), step],
                                  capture_output=True, text=True, cwd=ROOT,
                                  timeout=STEP_TIMEOUT_S)
            steps.append({"step": step, "rc": proc.returncode,
                          "tail": (proc.stdout + proc.stderr)[-400:]})
            if proc.returncode != 0:
                ok = False
                break
        except (subprocess.TimeoutExpired, OSError) as exc:
            steps.append({"step": step, "rc": -1, "tail": str(exc)[-400:]})
            ok = False
            break
    _last = {"ok": ok, "steps": steps, "duration_s": round(time.monotonic() - t0, 1),
             "finished_at": time.time(), "drift": d}
    with _lock:
        _running = False


def refresh(force: bool = False) -> dict:
    """Kick a rebuild in the background if the index drifted from Z3 (or force=True)."""
    global _running
    with _lock:
        if _running:
            return {"status": "already_running"}
        d = drift()
        # A failed last run (embed died, subprocess timeout) must stay repairable from
        # the UI: retry on the next manual refresh even when the index shows no drift.
        last_failed = _last is not None and not _last.get("ok")
        if not force and not d.get("drift") and not last_failed:
            return {"status": "no_drift", **d}
        _running = True
    threading.Thread(target=_run, args=(d,), daemon=True, name="lora-refresh").start()
    return {"status": "started", **d}


def status() -> dict:
    try:
        mtime = lora_catalog.INDEX_PATH.stat().st_mtime
        count = len(_index_files())
    except OSError:
        mtime, count = None, 0
    return {"running": _running, "last": _last, "index_count": count, "index_mtime": mtime}
