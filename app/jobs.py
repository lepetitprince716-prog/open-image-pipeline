"""Persistent, attempt-aware job queue for the Studio backend.

The jobs row is the stable public contract.  Every engine launch gets a separate
``job_attempts`` record so process ownership, remote prompt identity, timings and recovery
decisions survive an app restart.  stdout is diagnostic only; ``--json-out`` and the private
``GEN_EVENTS_OUT`` JSONL stream are the machine interfaces.
"""
from __future__ import annotations

import json
import os
import queue
import signal
import sqlite3
import subprocess
import threading
import time
import uuid
from pathlib import Path

from loramake import comfy

from . import brief as brief_mod
from . import job_experiments
from . import notify

try:  # the progress relay is best-effort; the queue works without the dependency
    from websockets.sync.client import connect as _ws_connect
except ImportError:  # pragma: no cover
    _ws_connect = None

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = ROOT / "app" / "data" / "jobs.sqlite3"
DB_PATH = DEFAULT_DB_PATH
RESULTS_DIR = ROOT / "app" / "data" / "results"
ATTEMPTS_DIR = ROOT / "app" / "data" / "attempts"
SCHEMA_VERSION = 3  # v3: job_reviews (lightweight per-job verdict layer)
JOB_TIMEOUT_S = 3600
PROCESS_GROUP_EXIT_S = 2.0
REMOTE_CANCEL_TIMEOUT_S = 20.0
RECOVERY_MONITOR_TIMEOUT_S = 3600.0
RECOVERY_POLL_S = 2.0
PROGRESS_MIN_INTERVAL_S = 0.5

# Kept injectable for tests and for the app server's transport host map.
to_argv = brief_mod.to_argv
# Optional fire-and-forget hook the server installs after a job is published as done.
PREWARM_RESULT_MEDIA = None
HOSTS: dict[str, str] = {}
# Injected by the server (gen.resolve_box): maps box "auto" to a concrete box at launch
# (ZZZ-first policy). None (e.g. unit tests) keeps the historical zzz default.
RESOLVE_BOX = None
# Injected by the server (app.network.enforcement.launch_gate): launch-time backstop for
# jobs queued before a network toggle flipped. None (unit tests, CLI) means no gating.
NETWORK_GATE = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id          TEXT PRIMARY KEY,
  client_id   TEXT UNIQUE,
  brief       TEXT NOT NULL,
  status      TEXT NOT NULL DEFAULT 'queued',
  created_at  REAL NOT NULL,
  started_at  REAL,
  finished_at REAL,
  result      TEXT,
  error       TEXT
);
CREATE TABLE IF NOT EXISTS job_attempts (
  attempt_id       TEXT PRIMARY KEY,
  job_id           TEXT NOT NULL REFERENCES jobs(id),
  attempt_no       INTEGER NOT NULL,
  status           TEXT NOT NULL,
  stage            TEXT NOT NULL,
  worker_pid       INTEGER,
  process_pid      INTEGER,
  process_group    INTEGER,
  box              TEXT,
  remote_prompt_id TEXT,
  created_at       REAL NOT NULL,
  started_at       REAL,
  heartbeat_at     REAL,
  finished_at      REAL,
  result_path      TEXT,
  event_path       TEXT,
  timings          TEXT,
  error_class      TEXT,
  error             TEXT,
  UNIQUE(job_id, attempt_no)
);
CREATE TABLE IF NOT EXISTS job_reviews (
  job_id     TEXT PRIMARY KEY REFERENCES jobs(id),
  verdict    TEXT NOT NULL CHECK(verdict IN ('up','down')),
  note       TEXT NOT NULL DEFAULT '',
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_status_created_idx ON jobs(status, created_at);
CREATE INDEX IF NOT EXISTS attempts_job_no_idx ON job_attempts(job_id, attempt_no DESC);
CREATE INDEX IF NOT EXISTS attempts_remote_prompt_idx ON job_attempts(remote_prompt_id);
"""

_subscribers: list[queue.Queue] = []
_subscribers_lock = threading.Lock()
_worker_started = False
_worker_lock = threading.Lock()
_wakeup = threading.Event()
# Serializes the one-time WAL switch across threads (see _ensure_wal).
_wal_lock = threading.Lock()
_WAL_SWITCH_ATTEMPTS = 5
_WAL_SWITCH_BACKOFF_S = 0.05
_current_lock = threading.Lock()
_current: dict = {
    "job_id": None, "attempt_id": None, "box": None, "proc": None,
    "process_group": None, "event_path": None,
    # Outlives the job: last skip-free box. Cleared by a later job on another box or last-of-burst /free.
    "resident_box": None,
}


def _chmod_private(path: Path) -> None:
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _migrate(conn: sqlite3.Connection) -> None:
    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if version > SCHEMA_VERSION:
        raise RuntimeError(f"jobs DB schema {version} is newer than supported {SCHEMA_VERSION}")
    if version < SCHEMA_VERSION:
        conn.executescript(_SCHEMA)
        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        conn.commit()


def _ensure_wal(conn: sqlite3.Connection) -> None:
    """Put the database in WAL mode, tolerating a concurrent opener.

    `PRAGMA journal_mode=WAL` takes an exclusive lock and — unlike ordinary statements —
    does NOT honour busy_timeout: when another connection holds the database it fails on
    the spot with "database is locked" instead of waiting. Two threads opening the same
    *fresh* database at once therefore make one of them raise, which is exactly what the
    API worker, the lifecycle sweeper and a request handler do on startup.

    The mode is a persistent property, so re-check before switching: the loser of the race
    finds `wal` already set and does nothing. The retry covers a second process (rather
    than a second thread), which the in-process lock cannot serialize.
    """
    if conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal":
        return
    with _wal_lock:
        if conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal":
            return
        for attempt in range(_WAL_SWITCH_ATTEMPTS):
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                return
            except sqlite3.OperationalError:
                if attempt == _WAL_SWITCH_ATTEMPTS - 1:
                    raise
                time.sleep(_WAL_SWITCH_BACKOFF_S)


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("PRAGMA foreign_keys=ON")
    _ensure_wal(conn)
    conn.execute("PRAGMA synchronous=NORMAL")
    _migrate(conn)
    _chmod_private(path)
    _chmod_private(Path(f"{path}-wal"))
    _chmod_private(Path(f"{path}-shm"))
    return conn


def _runtime_paths(db_path: Path | None) -> tuple[Path, Path]:
    path = Path(db_path or DB_PATH)
    if path.resolve(strict=False) == DEFAULT_DB_PATH.resolve(strict=False):
        return Path(RESULTS_DIR), Path(ATTEMPTS_DIR)
    return path.parent / "results", path.parent / "attempts"


def _latest_attempt(conn: sqlite3.Connection, job_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM job_attempts WHERE job_id=? ORDER BY attempt_no DESC LIMIT 1", (job_id,)
    ).fetchone()


def _review_row(conn: sqlite3.Connection, job_id: str) -> dict | None:
    row = conn.execute(
        "SELECT verdict,note,updated_at FROM job_reviews WHERE job_id=?", (job_id,)
    ).fetchone()
    return dict(row) if row else None


def _row_to_dict(row: sqlite3.Row, conn: sqlite3.Connection | None = None) -> dict:
    d = dict(row)
    d["brief"] = json.loads(d["brief"])
    if d.get("result"):
        d["result"] = json.loads(d["result"])
    attempt = _latest_attempt(conn, d["id"]) if conn is not None else None
    d.update({
        "attempt_id": attempt["attempt_id"] if attempt else None,
        "attempt": attempt["attempt_no"] if attempt else None,
        "stage": attempt["stage"] if attempt else None,
        "remote_prompt_id": attempt["remote_prompt_id"] if attempt else None,
        "timings": json.loads(attempt["timings"]) if attempt and attempt["timings"] else None,
        "error_class": attempt["error_class"] if attempt else None,
        "review": _review_row(conn, d["id"]) if conn is not None else None,
    })
    return d


def _job(conn: sqlite3.Connection, job_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return _row_to_dict(row, conn) if row else None


def _provider_name(job_brief: dict) -> str:
    provider = job_brief.get("provider", "local")
    return provider if provider in brief_mod.PROVIDERS else "local"


def _queued_local_ahead(conn: sqlite3.Connection, job_id: str) -> bool:
    """True when at least one other queued job is a local ComfyUI job."""
    rows = conn.execute("SELECT brief FROM jobs WHERE status='queued' AND id!=?", (job_id,)).fetchall()
    for row in rows:
        try:
            brief = json.loads(row["brief"])
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(brief, dict) and brief.get("provider", "local") == "local":
            return True
    return False


def _release_resident_box(box: str | None) -> bool:
    host = HOSTS.get(box) if box else None
    if not host:
        return False
    try:
        return comfy.free_vram(host)
    except Exception:
        return False


def _result_failure(result: dict) -> tuple[str, str]:
    """Stable error class/message from a normalized gen result (provider-aware)."""
    error = result.get("error")
    if isinstance(error, dict):
        error_class = str(error.get("class") or result.get("error_class") or "engine_error")
        message = str(error.get("message") or f"engine rc={result.get('rc')}")
        return error_class[:120], message[:800]
    return str(result.get("error_class") or "engine_error")[:120], f"engine rc={result.get('rc')}"


def _publish(event: dict) -> None:
    with _subscribers_lock:
        for subscriber in list(_subscribers):
            try:
                subscriber.put_nowait(event)
            except queue.Full:
                pass


def subscribe() -> queue.Queue:
    subscriber: queue.Queue = queue.Queue(maxsize=256)
    with _subscribers_lock:
        _subscribers.append(subscriber)
    return subscriber


def unsubscribe(subscriber: queue.Queue) -> None:
    with _subscribers_lock:
        if subscriber in _subscribers:
            _subscribers.remove(subscriber)


def submit(job_brief: dict, client_id: str | None = None, db_path: Path | None = None) -> dict:
    """Queue a validated brief. The same client id returns the original job."""
    path = Path(db_path or DB_PATH)
    problems = brief_mod.validate(job_brief)
    if problems:
        raise ValueError("; ".join(problems))
    conn = _connect(path)
    try:
        if client_id:
            row = conn.execute("SELECT * FROM jobs WHERE client_id=?", (client_id,)).fetchone()
            if row:
                return _row_to_dict(row, conn)
        job_id = uuid.uuid4().hex[:12]
        conn.execute(
            "INSERT INTO jobs (id,client_id,brief,status,created_at) VALUES (?,?,?,?,?)",
            (job_id, client_id, json.dumps(job_brief, ensure_ascii=False), "queued", time.time()),
        )
        conn.commit()
        job = _job(conn, job_id)
    finally:
        conn.close()
    _publish({"type": "queued", "job": job})
    _wakeup.set()
    return job


def get_job(job_id: str, db_path: Path | None = None) -> dict | None:
    conn = _connect(db_path)
    try:
        return _job(conn, job_id)
    finally:
        conn.close()


def list_jobs(limit: int = 50, db_path: Path | None = None) -> list[dict]:
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 500)),)
        ).fetchall()
        return [_row_to_dict(row, conn) for row in rows]
    finally:
        conn.close()


def _like_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def search_jobs(q: str = "", status: str = "", limit: int = 50, offset: int = 0,
                db_path: Path | None = None) -> dict:
    """Paged job history: q substring-matches the brief JSON (prompt, LoRA names,
    intent), status filters exactly. Returns {"total": N, "jobs": [...]}."""
    conn = _connect(db_path)
    try:
        where: list[str] = []
        params: list = []
        if q:
            where.append("brief LIKE ? ESCAPE '\\'")
            params.append("%" + _like_escape(q) + "%")
        if status:
            where.append("status=?")
            params.append(status)
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        total = conn.execute(f"SELECT COUNT(*) FROM jobs{clause}", params).fetchone()[0]
        rows = conn.execute(
            f"SELECT * FROM jobs{clause} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (*params, max(1, min(limit, 200)), max(0, offset)),
        ).fetchall()
        return {"total": total, "jobs": [_row_to_dict(row, conn) for row in rows]}
    finally:
        conn.close()


def find_job_by_file(path: str, db_path: Path | None = None) -> dict | None:
    """Reverse lookup: newest job whose result claims this output file.

    The LIKE needle is the JSON-encoded path (quotes included) so a prefix of a
    longer path can't match; new_files membership is re-checked to kill any
    remaining false positives from LIKE."""
    conn = _connect(db_path)
    try:
        needle = json.dumps(path, ensure_ascii=False)
        rows = conn.execute(
            "SELECT * FROM jobs WHERE result LIKE ? ESCAPE '\\' ORDER BY created_at DESC LIMIT 8",
            ("%" + _like_escape(needle) + "%",),
        ).fetchall()
        for row in rows:
            try:
                result = json.loads(row["result"]) if row["result"] else {}
            except json.JSONDecodeError:
                continue
            if path in (result.get("new_files") or []):
                return _row_to_dict(row, conn)
        return None
    finally:
        conn.close()


def set_review(job_id: str, verdict: str | None, note: str = "",
               db_path: Path | None = None) -> dict | None:
    """Upsert (or clear, verdict=None) the single review row for a job.
    Returns the refreshed job dict, or None when the job does not exist."""
    if verdict not in (None, "up", "down"):
        raise ValueError("verdict must be 'up', 'down' or null")
    conn = _connect(db_path)
    try:
        if conn.execute("SELECT 1 FROM jobs WHERE id=?", (job_id,)).fetchone() is None:
            return None
        if verdict is None:
            conn.execute("DELETE FROM job_reviews WHERE job_id=?", (job_id,))
        else:
            conn.execute(
                "INSERT INTO job_reviews (job_id,verdict,note,updated_at) VALUES (?,?,?,?) "
                "ON CONFLICT(job_id) DO UPDATE SET verdict=excluded.verdict,"
                "note=excluded.note,updated_at=excluded.updated_at",
                (job_id, verdict, note or "", time.time()),
            )
        conn.commit()
        job = _job(conn, job_id)
    finally:
        conn.close()
    _publish({"type": "review", "job": job})
    return job


def list_attempts(job_id: str, db_path: Path | None = None) -> list[dict]:
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM job_attempts WHERE job_id=? ORDER BY attempt_no", (job_id,)
        ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["timings"] = json.loads(item["timings"]) if item.get("timings") else None
            out.append(item)
        return out
    finally:
        conn.close()


def _merge_timings(existing: str | None, extra: dict | None) -> str | None:
    if not extra and not existing:
        return None
    merged = json.loads(existing) if existing else {}
    merged.update(extra or {})
    return json.dumps(merged, ensure_ascii=False, sort_keys=True)


def _finish(conn: sqlite3.Connection, job_id: str, *, status: str,
            result: dict | None = None, error: str | None = None,
            attempt_id: str | None = None, error_class: str | None = None,
            timings: dict | None = None) -> dict:
    finished = time.time()
    conn.execute(
        "UPDATE jobs SET status=?,finished_at=?,result=?,error=? WHERE id=?",
        (status, finished,
         json.dumps(result, ensure_ascii=False) if result is not None else None, error, job_id),
    )
    if attempt_id:
        current = conn.execute(
            "SELECT timings FROM job_attempts WHERE attempt_id=?", (attempt_id,)
        ).fetchone()
        conn.execute(
            "UPDATE job_attempts SET status=?,stage=?,heartbeat_at=?,finished_at=?,timings=?,"
            "error_class=?,error=? WHERE attempt_id=?",
            (status, status, finished, finished,
             _merge_timings(current["timings"] if current else None, timings),
             error_class, error, attempt_id),
        )
    conn.commit()
    job = _job(conn, job_id)
    _publish({"type": status, "job": job})
    notify.job_finished(job)
    if status == "done":
        prewarm = PREWARM_RESULT_MEDIA
        if prewarm is not None:
            threading.Thread(
                target=prewarm, args=(job,), daemon=True,
                name=f"media-prewarm-{job['id'][:8]}",
            ).start()
        # 2026-08-10 recording duty: every finished job becomes/updates an
        # experiment manifest under outputs/_analysis/studio_jobs/. Best-effort,
        # never raises, so recording can never change job state.
        job_experiments.record_finished_job(job)
    return job


def _new_attempt(conn: sqlite3.Connection, job: dict, db_path: Path | None) -> tuple[str, Path, Path]:
    results_dir, attempts_dir = _runtime_paths(db_path)
    results_dir.mkdir(parents=True, exist_ok=True)
    attempts_dir.mkdir(parents=True, exist_ok=True)
    try:
        attempts_dir.chmod(0o700)
    except OSError:
        pass
    attempt_no = int(conn.execute(
        "SELECT COALESCE(MAX(attempt_no),0)+1 FROM job_attempts WHERE job_id=?", (job["id"],)
    ).fetchone()[0])
    attempt_id = uuid.uuid4().hex
    result_path = results_dir / f"{job['id']}.json"
    event_path = attempts_dir / f"{attempt_id}.jsonl"
    now = time.time()
    provider = _provider_name(job["brief"])
    box = job["brief"].get("box", "zzz") if provider == "local" else None
    conn.execute(
        "INSERT INTO job_attempts (attempt_id,job_id,attempt_no,status,stage,worker_pid,box,"
        "created_at,started_at,heartbeat_at,result_path,event_path) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (attempt_id, job["id"], attempt_no, "running", "launching", os.getpid(),
         box, now, now, now, str(result_path), str(event_path)),
    )
    conn.commit()
    return attempt_id, result_path, event_path


def _event_error_fields(event: dict) -> tuple[str | None, str | None]:
    """(error_class, error summary) carried by a prompt_error / prompt_submit_failed
    event (loramake.comfy structured-diagnostics fields), else (None, None)."""
    if event.get("event") not in {"prompt_error", "prompt_submit_failed"}:
        return None, None
    error_class = event.get("error_class") if isinstance(event.get("error_class"), str) else None
    message = str(event.get("exception_message") or "").strip()
    if message:
        node_id, node_type = event.get("node_id"), event.get("node_type")
        where = f" at node {node_id} ({node_type})" if node_id and node_type else ""
        exc_type = str(event.get("exception_type") or "").strip()
        prefix = f"{exc_type}: " if exc_type else ""
        return error_class, f"{prefix}{message}{where}"[:1200]
    return error_class, None


def _apply_attempt_event(attempt_id: str, event: dict, db_path: Path | None) -> None:
    kind = event.get("event")
    raw_prompt_id = event.get("provider_request_id") or event.get("prompt_id")
    prompt_id = raw_prompt_id if isinstance(raw_prompt_id, str) else None
    stage = {
        "prompt_submitted": "remote_running",
        "prompt_completed": "fetching",
        "prompt_error": "remote_error",
        "prompt_timeout": "remote_timeout",
        "prompt_submit_failed": "submit_failed",
        "provider_request_started": "provider_starting",
        "provider_cost_previewed": "provider_cost_previewed",
        "provider_request_submitted": "provider_submitted",
        "provider_succeeded": "provider_succeeded",
        "provider_failed": "provider_failed",
    }.get(kind, "running")
    if kind == "provider_status":
        raw_status = str(event.get("provider_status") or "running").lower()
        safe_status = "".join(ch for ch in raw_status if ch.isalnum() or ch in "_-")[:48]
        stage = f"provider_{safe_status or 'running'}"
    timing = None
    if isinstance(event.get("elapsed_s"), (int, float)):
        timing = {"remote_elapsed_s": round(float(event["elapsed_s"]), 3)}
    error_class, error = _event_error_fields(event)
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT job_id,timings FROM job_attempts WHERE attempt_id=?", (attempt_id,)
        ).fetchone()
        if row is None:
            return
        conn.execute(
            "UPDATE job_attempts SET stage=?,heartbeat_at=?,remote_prompt_id=COALESCE(?,remote_prompt_id),"
            "timings=?,error_class=COALESCE(?,error_class),error=COALESCE(?,error) WHERE attempt_id=?",
            (stage, time.time(), prompt_id, _merge_timings(row["timings"], timing),
             error_class, error, attempt_id),
        )
        conn.commit()
        if isinstance(kind, str) and kind.startswith("provider_"):
            job = _job(conn, row["job_id"])
            if job:
                _publish({"type": "running", "job": job})
    finally:
        conn.close()


def _attempt_event_relay(attempt_id: str, event_path: Path, db_path: Path | None,
                         stop: threading.Event) -> None:
    offset = 0
    while not stop.wait(0.1):
        if not event_path.is_file():
            continue
        try:
            with event_path.open(encoding="utf-8") as handle:
                handle.seek(offset)
                lines = handle.readlines()
                offset = handle.tell()
        except OSError:
            continue
        for line in lines:
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(event, dict):
                _apply_attempt_event(attempt_id, event, db_path)
    # One final pass catches the event written immediately before process exit.
    if event_path.is_file():
        try:
            with event_path.open(encoding="utf-8") as handle:
                handle.seek(offset)
                lines = handle.readlines()
        except OSError:
            lines = []
        for line in lines:
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(event, dict):
                _apply_attempt_event(attempt_id, event, db_path)


def _prompt_from_events(event_path: str | Path | None) -> str | None:
    if not event_path or not Path(event_path).is_file():
        return None
    prompt_id = None
    try:
        lines = Path(event_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(event, dict):
            candidate = event.get("provider_request_id") or event.get("prompt_id")
            if isinstance(candidate, str):
                prompt_id = candidate
    return prompt_id


def _is_windows() -> bool:
    return os.name == "nt"


def _engine_popen_group_kwargs() -> dict:
    """Create an owned cancellation boundary without invoking a shell."""
    if _is_windows():
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}


def _process_group_id(proc: subprocess.Popen) -> int:
    # Windows has no os.getpgid().  The process id is the stable root consumed by
    # taskkill /T; POSIX keeps the dedicated session/process-group id.
    return proc.pid if _is_windows() else os.getpgid(proc.pid)


def _posix_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _terminate_windows_process_tree(
    proc: subprocess.Popen, timeout: float = PROCESS_GROUP_EXIT_S,
) -> dict:
    """Bounded termination of exactly one Windows engine tree rooted at proc.pid."""
    started = time.monotonic()
    if proc.poll() is not None:
        return {"ok": True, "process_group": proc.pid, "elapsed_s": 0.0,
                "escalated": False}
    deadline = started + max(0.1, timeout)
    escalated = True
    remaining = max(0.1, deadline - time.monotonic())
    try:
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            capture_output=True,
            text=True,
            timeout=remaining,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        pass
    remaining = max(0.01, deadline - time.monotonic())
    try:
        proc.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
            proc.wait(timeout=max(0.01, deadline - time.monotonic()))
        except (OSError, subprocess.SubprocessError):
            pass
    return {"ok": proc.poll() is not None, "process_group": proc.pid,
            "elapsed_s": round(time.monotonic() - started, 3), "escalated": escalated}


def _terminate_process_group(proc: subprocess.Popen, timeout: float = PROCESS_GROUP_EXIT_S) -> dict:
    """Terminate the complete session created for one engine attempt within ``timeout`` seconds."""
    if _is_windows():
        return _terminate_windows_process_tree(proc, timeout)
    started = time.monotonic()
    try:
        process_group = os.getpgid(proc.pid)
    except (OSError, ProcessLookupError):
        return {"ok": True, "process_group": None, "elapsed_s": 0.0, "escalated": False}
    escalated = False
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return {"ok": True, "process_group": process_group, "elapsed_s": 0.0,
                "escalated": False}
    soft_deadline = started + min(1.0, max(0.05, timeout * 0.5))
    deadline = started + max(0.1, timeout)
    while time.monotonic() < soft_deadline and _posix_group_exists(process_group):
        time.sleep(0.02)
    if _posix_group_exists(process_group):
        escalated = True
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    while time.monotonic() < deadline and _posix_group_exists(process_group):
        time.sleep(0.02)
    return {"ok": not _posix_group_exists(process_group), "process_group": process_group,
            "elapsed_s": round(time.monotonic() - started, 3), "escalated": escalated}


def _remote_cleanup(provider: str, box: str | None, prompt_id: str | None) -> dict:
    if provider != "local":
        if prompt_id:
            return {
                "ok": False,
                "action": "provider_cancel_unsupported",
                "reason": "external provider request may continue; automatic cancellation is unavailable",
            }
        return {"ok": True, "action": "no_provider_request", "reason": ""}
    host = HOSTS.get(box)
    if not host or not prompt_id:
        return {"ok": True, "action": "no_remote_prompt", "reason": ""}
    try:
        return comfy.cancel_prompt(host, prompt_id, timeout=REMOTE_CANCEL_TIMEOUT_S)
    except Exception as exc:
        return {"ok": False, "action": "queue_unknown",
                "reason": f"{type(exc).__name__}: {exc}"}


def _run_one(job: dict, db_path: Path | None) -> None:
    conn = _connect(db_path)
    attempt_id = None
    proc = None
    event_stop = threading.Event()
    progress_stop = threading.Event()
    started_mono = time.monotonic()
    try:
        claimed = conn.execute(
            "UPDATE jobs SET status='running',started_at=?,finished_at=NULL,error=NULL "
            "WHERE id=? AND status='queued'", (time.time(), job["id"]),
        )
        conn.commit()
        if claimed.rowcount == 0:
            return
        provider = _provider_name(job["brief"])
        # Resolve a local box "auto" to a concrete box before anything records or
        # uses it (attempt row, argv, progress relay, remote cleanup) so every
        # consumer sees the same box this attempt actually targets. External
        # providers must not probe either GPU as part of their launch path.
        brief = dict(job["brief"])
        forced_box = None
        if NETWORK_GATE is not None:
            gate_class, gate_error, forced_box = NETWORK_GATE(brief)
            if gate_class is not None:
                _finish(conn, job["id"], status="error", error=gate_error,
                        error_class=gate_class)
                return
        if provider == "local":
            requested = brief.get("box") or "auto"
            if requested in HOSTS:
                box = requested
            elif forced_box is not None:
                # box=auto with exactly one enabled box: use it directly so the
                # launch path never probes the disabled one.
                box = forced_box
            elif RESOLVE_BOX is not None:
                box, _ = RESOLVE_BOX(requested, brief.get("tier"))
            else:
                box = "zzz"
            brief["box"] = box
            if brief.get("keep_resident") is not True and _queued_local_ahead(conn, job["id"]):
                brief["keep_resident"] = True
        else:
            box = None
        job = {**job, "brief": brief}
        attempt_id, result_path, event_path = _new_attempt(conn, job, db_path)
        _publish({"type": "running", "job": _job(conn, job["id"])})
        argv = to_argv(job["brief"], result_path)
        env = {**os.environ, "LORAMAKE_ATTEMPT_ID": attempt_id,
               "GEN_EVENTS_OUT": str(event_path)}

        resident_box = _current.get("resident_box")
        if resident_box and resident_box != box:
            _release_resident_box(resident_box)

        # Serializes the narrow claim->launch window with cancel().
        with _current_lock:
            state = conn.execute("SELECT status FROM jobs WHERE id=?", (job["id"],)).fetchone()
            if state is None or state["status"] != "running":
                _finish(conn, job["id"], status="cancelled", error="cancelled before launch",
                        attempt_id=attempt_id, error_class="cancelled_before_launch")
                return
            proc = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env=env, **_engine_popen_group_kwargs(),
            )
            process_group = _process_group_id(proc)
            _current.update(job_id=job["id"], attempt_id=attempt_id, box=box, proc=proc,
                            process_group=process_group, event_path=str(event_path),
                            resident_box=box if brief.get("keep_resident") else None)
            launch_stage = "local_running" if provider == "local" else "provider_launching"
            conn.execute(
                "UPDATE job_attempts SET stage=?,process_pid=?,process_group=?,heartbeat_at=? "
                "WHERE attempt_id=?", (launch_stage, proc.pid, process_group, time.time(), attempt_id),
            )
            conn.commit()

        threading.Thread(
            target=_attempt_event_relay,
            args=(attempt_id, event_path, db_path, event_stop),
            daemon=True,
            name=f"attempt-events-{attempt_id[:8]}",
        ).start()
        if provider == "local":
            threading.Thread(
                target=_progress_relay,
                args=(job["id"], attempt_id, box, progress_stop, db_path),
                daemon=True,
                name=f"progress-relay-{attempt_id[:8]}",
            ).start()
        try:
            out, err = proc.communicate(timeout=JOB_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            local = _terminate_process_group(proc)
            out, err = proc.communicate()
            event_stop.set()
            prompt_id = _prompt_from_events(event_path)
            remote = _remote_cleanup(provider, box, prompt_id)
            status = "error" if local["ok"] and remote["ok"] else "interrupted"
            _finish(
                conn, job["id"], status=status, attempt_id=attempt_id,
                error=f"timed out after {JOB_TIMEOUT_S}s; remote={remote.get('action')}",
                error_class="timeout" if status == "error" else "timeout_cleanup_incomplete",
                timings={"wall_s": round(time.monotonic() - started_mono, 3),
                         "local_cancel_s": local["elapsed_s"]},
            )
            return
        finally:
            event_stop.set()
            progress_stop.set()
            with _current_lock:
                if _current["job_id"] == job["id"]:
                    _current.update(job_id=None, attempt_id=None, box=None, proc=None,
                                    process_group=None, event_path=None)

        # Ensure the relay's final event pass has completed before finalizing the attempt row.
        time.sleep(0.02)
        state = conn.execute("SELECT status FROM jobs WHERE id=?", (job["id"],)).fetchone()
        if state and state["status"] in {"cancelling", "cancelled", "interrupted"}:
            return
        wall = round(time.monotonic() - started_mono, 3)
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                _finish(conn, job["id"], status="error", attempt_id=attempt_id,
                        error="json-out file is not valid JSON", error_class="invalid_result",
                        timings={"wall_s": wall})
                return
            if result.get("rc") == 0:
                status = "done"
            elif provider != "local" and result.get("provider_status") == "outcome_unknown":
                status = "interrupted"
            else:
                status = "error"
            error_class = error_message = None
            if status != "done":
                fallback_class, fallback_message = _result_failure(result)
                if provider == "local":
                    # The event relay may have parked structured remote diagnostics
                    # (node/exception detail from loramake.comfy) on the attempt row;
                    # surface those over the generic result/exit-code fallback.
                    att = conn.execute(
                        "SELECT error,error_class FROM job_attempts WHERE attempt_id=?",
                        (attempt_id,),
                    ).fetchone()
                    error_message = (att["error"] if att else None) or fallback_message
                    error_class = (att["error_class"] if att else None) or fallback_class
                else:
                    error_class, error_message = fallback_class, fallback_message
            _finish(conn, job["id"], status=status, result=result, attempt_id=attempt_id,
                    error=error_message, error_class=error_class,
                    timings={"wall_s": wall})
        else:
            tail = (err or out or "").strip()[-500:]
            _finish(conn, job["id"], status="error", attempt_id=attempt_id,
                    error=f"gen.py exited {proc.returncode} without a json-out file: {tail}",
                    error_class="missing_result", timings={"wall_s": wall})
    finally:
        conn.close()


def cancel(job_id: str, db_path: Path | None = None) -> dict | None:
    """Cancel queued/running work without touching another prompt on the same box."""
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            return None
        if row["status"] not in {"queued", "running"}:
            raise ValueError(
                f"job is {row['status']}; only queued/running jobs can be cancelled"
            )
        if row["status"] == "queued":
            return _finish(conn, job_id, status="cancelled", error="cancelled by user",
                           error_class="cancelled")
        attempt = _latest_attempt(conn, job_id)
        conn.execute("UPDATE jobs SET status='cancelling',error=? WHERE id=? AND status='running'",
                     ("cancellation requested", job_id))
        if attempt:
            conn.execute(
                "UPDATE job_attempts SET stage='cancelling',heartbeat_at=? WHERE attempt_id=?",
                (time.time(), attempt["attempt_id"]),
            )
        conn.commit()
        attempt_id = attempt["attempt_id"] if attempt else None
        event_path = attempt["event_path"] if attempt else None
        prompt_id = attempt["remote_prompt_id"] if attempt else None
        job_brief = json.loads(row["brief"])
        provider = _provider_name(job_brief)
        # The attempt row carries the concrete box (post "auto" resolution); the raw
        # brief may still say "auto", which HOSTS.get() maps to no remote host.
        box = (
            ((attempt["box"] if attempt else None) or job_brief.get("box", "zzz"))
            if provider == "local" else None
        )
    finally:
        conn.close()

    with _current_lock:
        proc = _current["proc"] if _current["job_id"] == job_id else None
        local = (_terminate_process_group(proc) if proc is not None else
                 {"ok": True, "elapsed_s": 0.0, "escalated": False})
    prompt_id = prompt_id or _prompt_from_events(event_path)
    remote = _remote_cleanup(provider, box, prompt_id)
    status = "cancelled" if local["ok"] and remote["ok"] else "interrupted"
    error_class = "cancelled" if status == "cancelled" else "cancel_cleanup_incomplete"
    error = "cancelled by user" if status == "cancelled" else (
        f"cancel incomplete: local_ok={local['ok']}; remote={remote.get('reason') or remote.get('action')}"
    )
    conn = _connect(db_path)
    try:
        return _finish(
            conn, job_id, status=status, error=error, attempt_id=attempt_id,
            error_class=error_class,
            timings={"local_cancel_s": local["elapsed_s"], "remote_cancel": remote.get("action")},
        )
    finally:
        conn.close()


def _ws_message(raw) -> dict | None:
    """Parse one ComfyUI /ws text frame; binary latent previews stay ignored."""
    if isinstance(raw, (bytes, bytearray)):
        return None
    try:
        msg = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return msg if isinstance(msg, dict) else None


_PROGRESS_EXECUTION_START_TYPES = {"execution_start", "execution_cached"}
_UNFILTERED = object()  # omitted keyword = no prompt filter; the relay always passes an explicit value


def _progress_event(job_id: str, box: str, raw, *, expected_prompt_id=_UNFILTERED) -> dict | None:
    """Parse a progress frame; without ``expected_prompt_id`` it stays unfiltered (legacy helper API)."""
    msg = _ws_message(raw)
    if msg is None:
        return None
    return _progress_event_from_message(job_id, box, msg, expected_prompt_id=expected_prompt_id)


def _progress_event_from_message(job_id: str, box: str, msg: dict, *,
                                 expected_prompt_id, executing_prompt_id: str | None = None) -> dict | None:
    if msg.get("type") != "progress":
        return None
    data = msg.get("data")
    if not isinstance(data, dict):
        return None
    prompt_id = data.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        prompt_id = executing_prompt_id
    if expected_prompt_id is _UNFILTERED:
        pass
    elif expected_prompt_id is None or prompt_id != expected_prompt_id:
        return None
    return {"type": "progress", "job_id": job_id, "box": box,
            "value": data.get("value"), "max": data.get("max"), "node": data.get("node")}


def _attempt_remote_prompt_id(conn: sqlite3.Connection, attempt_id: str) -> str | None:
    row = conn.execute("SELECT remote_prompt_id FROM job_attempts WHERE attempt_id=?",
                       (attempt_id,)).fetchone()
    value = row["remote_prompt_id"] if row is not None else None
    return value if isinstance(value, str) and value else None


def _progress_relay(job_id: str, attempt_id: str, box: str, stop: threading.Event,
                    db_path: Path | None = None) -> None:
    """Forward progress frames belonging to this attempt's ComfyUI prompt only.

    ComfyUI frames carry ``prompt_id`` on progress and on execution_start/cached; the
    relay tracks both so progress frames without an embedded id still resolve to the
    currently executing prompt.  Frames from external prompts on the same box are
    deliberately not published, which closes the previous cross-source progress crosstalk.
    """
    host = HOSTS.get(box)
    if not host or _ws_connect is None:
        return
    try:
        conn = _connect(db_path)
    except Exception:
        return
    try:
        url = host.replace("http://", "ws://", 1) + f"/ws?clientId=loramake-{uuid.uuid4().hex[:8]}"
        last = 0.0
        executing_prompt_id = None
        with _ws_connect(url, open_timeout=5) as websocket:
            while not stop.is_set():
                try:
                    raw = websocket.recv(timeout=2)
                except TimeoutError:
                    continue
                msg = _ws_message(raw)
                if msg is None:
                    continue
                if msg.get("type") in _PROGRESS_EXECUTION_START_TYPES:
                    data = msg.get("data")
                    if not isinstance(data, dict):
                        continue
                    candidate = data.get("prompt_id")
                    if isinstance(candidate, str) and candidate:
                        executing_prompt_id = candidate
                    continue
                event = _progress_event_from_message(
                    job_id, box, msg,
                    expected_prompt_id=_attempt_remote_prompt_id(conn, attempt_id),
                    executing_prompt_id=executing_prompt_id,
                )
                now = time.monotonic()
                if event and now - last >= PROGRESS_MIN_INTERVAL_S:
                    last = now
                    _publish(event)
    except Exception:
        return
    finally:
        conn.close()


def _finish_recovered_result(conn: sqlite3.Connection, job_id: str, attempt_id: str,
                             result_path: Path) -> bool:
    if not result_path.is_file():
        return False
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if result.get("rc") == 0:
        status = "done"
    elif result.get("provider") in brief_mod.EXTERNAL_PROVIDERS \
            and result.get("provider_status") == "outcome_unknown":
        status = "interrupted"
    else:
        status = "error"
    error_class, error_message = _result_failure(result)
    _finish(conn, job_id, status=status, result=result, attempt_id=attempt_id,
            error=None if status == "done" else error_message,
            error_class=None if status == "done" else error_class)
    return True


def _monitor_recovered(job_id: str, attempt_id: str, host: str, prompt_id: str,
                       result_path: Path, db_path: Path | None) -> None:
    deadline = time.monotonic() + RECOVERY_MONITOR_TIMEOUT_S
    last_error = None
    while time.monotonic() < deadline:
        conn = _connect(db_path)
        try:
            if _finish_recovered_result(conn, job_id, attempt_id, result_path):
                _wakeup.set()
                return
        finally:
            conn.close()
        try:
            observation = comfy.observe_queue(host, timeout=min(10.0, RECOVERY_POLL_S))
            active = prompt_id in set(observation["running"]) | set(observation["pending"])
            last_error = None
        except Exception as exc:
            active = True
            last_error = f"{type(exc).__name__}: {exc}"
        if not active:
            conn = _connect(db_path)
            try:
                _finish(conn, job_id, status="interrupted", attempt_id=attempt_id,
                        error="remote prompt left the queue but no local result was sealed; explicit rerun required",
                        error_class="recovered_prompt_without_result")
            finally:
                conn.close()
            _wakeup.set()
            return
        time.sleep(RECOVERY_POLL_S)
    conn = _connect(db_path)
    try:
        _finish(conn, job_id, status="interrupted", attempt_id=attempt_id,
                error=f"recovery monitor timed out; explicit rerun required ({last_error or 'prompt still active'})",
                error_class="recovery_timeout")
    finally:
        conn.close()
    _wakeup.set()


def _recover_inflight(db_path: Path | None) -> None:
    """Converge stale in-flight rows without ever turning them back into queued work."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE status IN ('running','cancelling') ORDER BY created_at"
        ).fetchall()
        for row in rows:
            job_id = row["id"]
            attempt = _latest_attempt(conn, job_id)
            if attempt is None:
                _finish(conn, job_id, status="interrupted",
                        error="backend restarted without an attempt record; explicit rerun required",
                        error_class="restart_without_attempt")
                continue
            attempt_id = attempt["attempt_id"]
            result_path = Path(attempt["result_path"])
            if _finish_recovered_result(conn, job_id, attempt_id, result_path):
                continue
            job_brief = json.loads(row["brief"])
            provider = _provider_name(job_brief)
            if provider != "local":
                _finish(
                    conn, job_id, status="interrupted", attempt_id=attempt_id,
                    error="backend restarted during an external provider request; no automatic resubmit",
                    error_class="provider_recovery_unsupported",
                )
                continue
            prompt_id = attempt["remote_prompt_id"] or _prompt_from_events(attempt["event_path"])
            # Prefer the attempt's concrete post-resolution box over a raw "auto"
            # value retained in the immutable job brief.
            box = attempt["box"] or job_brief.get("box", "zzz")
            host = HOSTS.get(box)
            if not prompt_id or not host:
                _finish(conn, job_id, status="interrupted", attempt_id=attempt_id,
                        error="backend restarted before a recoverable prompt/result was recorded; explicit rerun required",
                        error_class="restart_without_prompt")
                continue
            try:
                observation = comfy.observe_queue(host, timeout=10.0)
            except Exception as exc:
                _finish(conn, job_id, status="interrupted", attempt_id=attempt_id,
                        error=f"remote queue state unknown after restart: {type(exc).__name__}; explicit rerun required",
                        error_class="recovery_queue_unknown")
                continue
            active = prompt_id in set(observation["running"]) | set(observation["pending"])
            if not active:
                _finish(conn, job_id, status="interrupted", attempt_id=attempt_id,
                        error="recorded prompt is no longer active and no result exists; explicit rerun required",
                        error_class="recovered_prompt_absent")
                continue
            conn.execute(
                "UPDATE job_attempts SET stage='recovering',heartbeat_at=?,remote_prompt_id=? "
                "WHERE attempt_id=?", (time.time(), prompt_id, attempt_id),
            )
            conn.commit()
            threading.Thread(
                target=_monitor_recovered,
                args=(job_id, attempt_id, host, prompt_id, result_path, db_path),
                daemon=True, name=f"recover-{job_id}",
            ).start()
    finally:
        conn.close()


def _worker_loop(db_path: Path | None) -> None:
    _recover_inflight(db_path)
    while True:
        conn = _connect(db_path)
        active = conn.execute(
            "SELECT 1 FROM jobs WHERE status IN ('running','cancelling') LIMIT 1"
        ).fetchone()
        row = None if active else conn.execute(
            "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1"
        ).fetchone()
        job = _row_to_dict(row, conn) if row else None
        conn.close()
        if job is None:
            _wakeup.wait(timeout=1.0)
            _wakeup.clear()
            continue
        _run_one(job, db_path)


def start_worker(db_path: Path | None = None) -> None:
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        # Resolve the path ONCE and hand the thread a concrete Path. The worker is a daemon
        # that outlives whoever started it, and every helper it calls reads DB_PATH afresh
        # when handed None — so a worker started with None follows a later rebinding of that
        # global and claims jobs out of a database it was never meant to serve. Only tests
        # rebind DB_PATH, and there it means one test's leftover worker running the next
        # test's queue against whatever to_argv is patched in by then.
        resolved = Path(db_path or DB_PATH)
        # Initialize/migrate synchronously before background maintenance can inspect the DB.
        # This also makes schema/version failures visible during app startup instead of leaving
        # the worker and lifecycle threads racing on a fresh runtime root.
        conn = _connect(resolved)
        conn.close()
        threading.Thread(
            target=_worker_loop, args=(resolved,), daemon=True, name="gen-worker"
        ).start()
        _worker_started = True
