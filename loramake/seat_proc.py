"""Run one external model seat as a subprocess: own process group, hard timeout, owned cleanup.

Shared by the critic seat (and, later, other single-turn model calls). Copied from the
grok runner in ``scripts/defect_scan.py``: the child gets its own session so a timeout
kills exactly the tree we started and never enumerates global processes.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import Any, Mapping


def process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def kill_process_group(proc: subprocess.Popen[Any], pgid: int, grace_seconds: float = 2.0) -> None:
    """Terminate only the process group created for ``proc``."""
    if pgid <= 0 or not process_group_exists(pgid):
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        pass
    deadline = time.monotonic() + max(grace_seconds, 0.0)
    while time.monotonic() < deadline:
        if not process_group_exists(pgid):
            break
        time.sleep(0.05)
    if process_group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def run_seat(argv: list[str], *, timeout_s: float, env: Mapping[str, str] | None = None,
             cwd: str | None = None) -> dict[str, Any]:
    """One single-turn call. Returns {ok, raw, stderr, error, returncode}; never raises."""
    proc: subprocess.Popen[str] | None = None
    pgid: int | None = None
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                env=dict(env) if env is not None else None, cwd=cwd, preexec_fn=os.setsid)
        pgid = proc.pid
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            kill_process_group(proc, pgid)
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except Exception:
                stdout, stderr = "", ""
            return {"ok": False, "raw": stdout or "", "stderr": (stderr or "")[-2000:],
                    "error": f"seat timed out after {timeout_s:g}s", "returncode": None}
        diagnostic = (stderr or stdout or "no diagnostic").strip()[-2000:]
        error = None if proc.returncode == 0 else f"seat exited {proc.returncode}: {diagnostic}"
        return {"ok": proc.returncode == 0, "raw": stdout or "", "stderr": (stderr or "")[-2000:],
                "error": error, "returncode": proc.returncode}
    except Exception as exc:
        return {"ok": False, "raw": "", "stderr": "", "returncode": None,
                "error": f"seat launch failed: {type(exc).__name__}: {exc}"}
    finally:
        if proc is not None and pgid is not None:
            kill_process_group(proc, pgid)
