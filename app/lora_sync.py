"""Manual LoRA-catalog refresh for the app.

Drift detection is one cheap request (BOXA's live LoraLoader enum vs the on-disk index);
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

from loramake import model_resources as _mr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import lora_catalog  # noqa: E402  (scripts/lora_catalog.py — _comfy_lora_names/INDEX_PATH)

CATALOG_SCRIPT = ROOT / "scripts" / "lora_catalog.py"
INDEX_SCRIPT = ROOT / "scripts" / "boxa_model_index.py"
STEP_TIMEOUT_S = 600

_lock = threading.Lock()
_running = False
_last: dict | None = None  # outcome of the most recent refresh run


def _index_files() -> set[str]:
    try:
        return {r.get("file") for r in json.loads(lora_catalog.INDEX_PATH.read_text())}
    except (OSError, json.JSONDecodeError):
        return set()


def _unified_drift() -> dict:
    """Check whether the unified asset index (boxa_model_index/index.json) is missing,
    stale, or has a LoRA count that disagrees with the on-disk catalog."""
    idx = _mr.load_unified_index()
    fresh = _mr.inventory_freshness(idx)
    if not fresh["present"]:
        return {"inventory_drift": True, "inventory_status": "missing"}
    if fresh["status"] in ("stale", "missing"):
        return {"inventory_drift": True, "inventory_status": fresh["status"],
                "inventory_generated_at": fresh["generated_at"]}
    try:
        snap = _mr.load_snapshot(lora_catalog.INDEX_PATH)
        catalog_count = len(snap.records)
    except (OSError, json.JSONDecodeError):
        catalog_count = None
    inv_count = (fresh.get("counts") or {}).get("loras")
    if catalog_count is not None and inv_count is not None and catalog_count != inv_count:
        return {
            "inventory_drift": True,
            "inventory_status": "count_mismatch",
            "inventory_generated_at": fresh["generated_at"],
            "inventory_counts_loras": inv_count,
            "catalog_counts_loras": catalog_count,
        }
    return {
        "inventory_drift": False,
        "inventory_status": fresh["status"],
        "inventory_generated_at": fresh["generated_at"],
    }


def _live_names() -> set[str]:
    """BOXA's current LoraLoader enum; empty set when the box is unreachable (fail open)."""
    return set(lora_catalog._comfy_lora_names(strict=False))


def _boxb_remap_delta() -> int:
    """How many catalog entries would change boxb_file if rematched against BOXB's live enum.

    Rebuilding owns the boxb_file data, so BOXB-side changes (a LoRA added, removed or
    renamed there) must count as drift even when BOXA is unchanged. The live enum can't be
    compared to the stored values directly — BOXB carries names the BOXA-canonical catalog
    never maps — so re-run the pure matcher and count differences. 0 when BOXB is
    unreachable or the index predates boxb_file (unknown is not drift; fail open)."""
    try:
        records = json.loads(lora_catalog.INDEX_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return 0
    if not records or not any("boxb_file" in r for r in records):
        return 0
    live = set(lora_catalog._comfy_lora_names(host=lora_catalog.BOXB_HOST, strict=False))
    if not live:
        return 0
    files = [r.get("file") for r in records]
    fresh = lora_catalog._match_boxb(files, live)
    return sum(1 for r in records if r.get("boxb_file") != fresh.get(r.get("file")))


def drift() -> dict:
    live = _live_names()
    if not live:
        out = {"drift": False, "error": "BOXA LoraLoader enum unreachable"}
        out.update(_unified_drift())
        return out
    idx = _index_files()
    out = {"drift": live != idx, "added": len(live - idx), "removed": len(idx - live)}
    if out["drift"]:
        return out
    boxb_changed = _boxb_remap_delta()
    if boxb_changed:
        out.update(drift=True, boxb_changed=boxb_changed)
        return out
    out.update(_unified_drift())
    if out.get("inventory_drift"):
        out["drift"] = True
    return out


def _run(d: dict) -> None:
    global _running, _last
    t0 = time.monotonic()
    steps, ok = [], True
    for step in d.get("steps", ("build", "embed", "index")):
        if step in ("build", "embed"):
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
        elif step == "index":
            try:
                proc = subprocess.run([
                    sys.executable, str(INDEX_SCRIPT), "refresh",
                    "--out", str(_mr.UNIFIED_INDEX_PATH),
                    "--skip-loras",
                ], capture_output=True, text=True, cwd=ROOT, timeout=STEP_TIMEOUT_S)
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


def _last_failed_steps() -> list[str]:
    """Infer the scope of a previous failed run so we can retry the same scope."""
    if _last is None:
        return []
    planned = _last.get("drift", {}).get("steps") or []
    if planned:
        return planned
    executed = _last.get("steps") or []
    return [s["step"] for s in executed if isinstance(s, dict) and "step" in s]


def _choose_steps(d: dict, force: bool, last_failed: bool) -> list[str]:
    """Decide which refresh steps to run.

    - force => full rebuild + index.
    - BOXA unreachable and not force => do nothing (fail open).
    - LoRA drift (added/removed/boxb_changed) => full.
    - Previous failed run => retry the same scope (index-only or full).
    - Only unified inventory drift => index-only.
    """
    if force:
        return ["build", "embed", "index"]
    if d.get("error"):
        return []
    loras_drift = d.get("added") or d.get("removed") or d.get("boxb_changed")
    if loras_drift:
        return ["build", "embed", "index"]
    if last_failed:
        last_steps = _last_failed_steps()
        if "index" in last_steps and "build" not in last_steps and "embed" not in last_steps:
            return ["index"]
        return ["build", "embed", "index"]
    if d.get("inventory_drift"):
        return ["index"]
    return []


def refresh(force: bool = False) -> dict:
    """Kick a rebuild in the background if the index drifted from BOXA (or force=True)."""
    global _running
    with _lock:
        if _running:
            return {"status": "already_running"}
        d = drift()
        last_failed = _last is not None and not _last.get("ok")
        if d.get("error") and not force:
            # Fail open while BOXA is unreachable; do not spin inventory refresh either.
            return {"status": "no_drift", **d}
        steps = _choose_steps(d, force, last_failed)
        if not steps:
            return {"status": "no_drift", **d}
        d["steps"] = steps
        _running = True
    threading.Thread(target=_run, args=(d,), daemon=True, name="lora-refresh").start()
    return {"status": "started", **d}


def status() -> dict:
    try:
        mtime = lora_catalog.INDEX_PATH.stat().st_mtime
        count = len(_index_files())
    except OSError:
        mtime, count = None, 0
    inv = _mr.inventory_freshness()
    return {
        "running": _running,
        "last": _last,
        "index_count": count,
        "index_mtime": mtime,
        "inventory_path": inv["path"],
        "inventory_generated_at": inv["generated_at"],
        "inventory_status": inv["status"],
        "inventory_counts": inv["counts"],
    }
