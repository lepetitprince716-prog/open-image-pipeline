"""Attempt ledger, process ownership and crash-recovery contracts (offline)."""
from __future__ import annotations

import json
import os
import sqlite3
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

from app import jobs


def _wait(job_id: str, db: Path, states: set[str], timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = jobs.get_job(job_id, db)
        if current and current["status"] in states:
            return current
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not reach {sorted(states)}")


def _pid_running(pid: int) -> bool:
    proc = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    state = proc.stdout.strip()
    return proc.returncode == 0 and bool(state) and not state.startswith("Z")


def _wait_for_pid(marker: Path, timeout: float = 3.0) -> int:
    """Wait until the child marker exists *and* contains a valid PID."""
    deadline = time.monotonic() + timeout
    last_value: str | None = None
    while time.monotonic() < deadline:
        try:
            last_value = marker.read_text(encoding="utf-8").strip()
            pid = int(last_value)
            if pid > 0:
                return pid
        except (FileNotFoundError, ValueError):
            pass
        time.sleep(0.02)
    raise AssertionError(
        f"child PID marker was not populated within {timeout:.1f}s "
        f"(last value: {last_value!r})"
    )


def test_wait_for_pid_ignores_transient_unpopulated_marker(monkeypatch):
    values: list[BaseException | str] = [FileNotFoundError(), "", "partial", "4321"]

    class Marker:
        def read_text(self, *, encoding: str) -> str:
            assert encoding == "utf-8"
            value = values.pop(0)
            if isinstance(value, BaseException):
                raise value
            return value

    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    assert _wait_for_pid(Marker()) == 4321  # type: ignore[arg-type]


def test_db_migrates_once_to_wal_and_private_mode(tmp_path):
    db = tmp_path / "jobs.sqlite3"
    conn = jobs._connect(db)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA user_version").fetchone()[0] == jobs.SCHEMA_VERSION
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert {"jobs", "job_attempts"} <= tables
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        conn.close()
    assert stat.S_IMODE(db.stat().st_mode) == 0o600
    # Re-open proves the idempotent migration path rather than rebuilding the tables.
    conn = jobs._connect(db)
    assert conn.execute("SELECT COUNT(*) FROM job_attempts").fetchone()[0] == 0
    conn.close()


def test_cancel_terminates_the_whole_local_process_group_within_two_seconds(tmp_path, monkeypatch):
    db = tmp_path / "jobs.sqlite3"
    marker = tmp_path / "child.pid"

    def fake_to_argv(_brief: dict, _result: Path) -> list[str]:
        child = "import time; time.sleep(60)"
        parent = (
            "import pathlib,subprocess,sys,time; "
            f"p=subprocess.Popen([sys.executable,'-c',{child!r}]); "
            f"pathlib.Path({str(marker)!r}).write_text(str(p.pid)); time.sleep(60)"
        )
        return [sys.executable, "-c", parent]

    monkeypatch.setattr(jobs, "to_argv", fake_to_argv)
    monkeypatch.setattr(jobs, "HOSTS", {})
    # app.server injects gen.resolve_box into this module at import time.  Keep
    # this process-cancellation test offline and deterministic: its subject is
    # the launched process group, not live GPU routing.
    monkeypatch.setattr(jobs, "RESOLVE_BOX", lambda *_args: ("zzz", "test fixture"))
    job = jobs.submit({"tier": "best", "prompt": "1girl"}, db_path=db)
    runner = threading.Thread(target=jobs._run_one, args=(job, db), daemon=True)
    runner.start()
    _wait(job["id"], db, {"running"})
    child_pid = _wait_for_pid(marker)
    started = time.monotonic()
    cancelled = jobs.cancel(job["id"], db)
    elapsed = time.monotonic() - started
    runner.join(timeout=3)
    assert cancelled["status"] == "cancelled"
    assert elapsed < 2.0
    assert not runner.is_alive()
    assert not _pid_running(child_pid)
    attempt = jobs.list_attempts(job["id"], db)[0]
    assert attempt["process_group"]
    assert attempt["error_class"] == "cancelled"
    assert attempt["timings"]["local_cancel_s"] < 2.0


def test_windows_engine_processes_use_a_pid_rooted_group(monkeypatch):
    monkeypatch.setattr(jobs, "_is_windows", lambda: True)
    monkeypatch.setattr(jobs.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, raising=False)

    class Proc:
        pid = 4321

    assert jobs._engine_popen_group_kwargs() == {"creationflags": 0x200}
    assert jobs._process_group_id(Proc()) == 4321


def test_windows_cancel_uses_bounded_exact_process_tree(monkeypatch):
    calls = []
    monkeypatch.setattr(jobs, "_is_windows", lambda: True)
    monkeypatch.setattr(jobs.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)

    class Proc:
        pid = 8765
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            assert timeout > 0
            self.returncode = 1
            return self.returncode

        def kill(self):
            self.returncode = 1

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(jobs.subprocess, "run", fake_run)
    result = jobs._terminate_process_group(Proc(), timeout=2.0)

    assert calls[0][0] == ["taskkill", "/PID", "8765", "/T", "/F"]
    assert "shell" not in calls[0][1]
    assert calls[0][1]["creationflags"] == 0x08000000
    assert result["ok"] is True
    assert result["process_group"] == 8765
    assert result["escalated"] is True


def _stale_attempt(db: Path, *, result: dict | None = None) -> tuple[dict, str, Path]:
    job = jobs.submit({"tier": "best", "prompt": "1girl"}, db_path=db)
    conn = jobs._connect(db)
    conn.execute("UPDATE jobs SET status='running',started_at=? WHERE id=?", (time.time(), job["id"]))
    attempt_id, result_path, _event_path = jobs._new_attempt(conn, job, db)
    conn.commit()
    conn.close()
    if result is not None:
        result_path.write_text(json.dumps(result), encoding="utf-8")
    return job, attempt_id, result_path


def test_recovery_seals_existing_result_without_resubmitting(tmp_path, monkeypatch):
    db = tmp_path / "jobs.sqlite3"
    job, attempt_id, _ = _stale_attempt(db, result={"rc": 0, "new_files": ["kept.png"]})
    monkeypatch.setattr(jobs, "to_argv", lambda *_: (_ for _ in ()).throw(AssertionError("resubmitted")))
    jobs._recover_inflight(db)
    recovered = jobs.get_job(job["id"], db)
    assert recovered["status"] == "done"
    assert recovered["result"]["new_files"] == ["kept.png"]
    assert recovered["attempt_id"] == attempt_id
    assert len(jobs.list_attempts(job["id"], db)) == 1


def test_recovery_without_prompt_or_result_becomes_interrupted_not_queued(tmp_path, monkeypatch):
    db = tmp_path / "jobs.sqlite3"
    job, attempt_id, _ = _stale_attempt(db)
    monkeypatch.setattr(jobs, "to_argv", lambda *_: (_ for _ in ()).throw(AssertionError("resubmitted")))
    jobs._recover_inflight(db)
    recovered = jobs.get_job(job["id"], db)
    assert recovered["status"] == "interrupted"
    assert recovered["error_class"] == "restart_without_prompt"
    assert recovered["attempt_id"] == attempt_id
    assert len(jobs.list_attempts(job["id"], db)) == 1


def test_recovery_with_active_known_prompt_monitors_without_resubmitting(tmp_path, monkeypatch):
    db = tmp_path / "jobs.sqlite3"
    job, attempt_id, _ = _stale_attempt(db)
    conn = jobs._connect(db)
    conn.execute(
        "UPDATE job_attempts SET remote_prompt_id='ours',box='zzz' WHERE attempt_id=?", (attempt_id,)
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(jobs, "HOSTS", {"zzz": "http://box"})
    monkeypatch.setattr(
        jobs.comfy, "observe_queue",
        lambda *_args, **_kwargs: {"running": ["ours"], "pending": ["foreign"]},
    )
    monkeypatch.setattr(jobs, "to_argv", lambda *_: (_ for _ in ()).throw(AssertionError("resubmitted")))
    started = []

    class FakeThread:
        def __init__(self, *, target, args, **_kwargs):
            self.target = target
            self.args = args

        def start(self):
            started.append((self.target, self.args))

    monkeypatch.setattr(jobs.threading, "Thread", FakeThread)
    jobs._recover_inflight(db)
    recovered = jobs.get_job(job["id"], db)
    assert recovered["status"] == "running"
    assert recovered["stage"] == "recovering"
    assert recovered["remote_prompt_id"] == "ours"
    assert started and started[0][0] is jobs._monitor_recovered


def test_existing_delete_mode_db_migrates_without_losing_jobs(tmp_path):
    db = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(db)
    conn.executescript(jobs._SCHEMA.split("CREATE TABLE IF NOT EXISTS job_attempts", 1)[0])
    conn.execute(
        "INSERT INTO jobs (id,brief,status,created_at) VALUES (?,?,?,?)",
        ("legacy", '{"tier":"best","prompt":"1girl"}', "done", 1.0),
    )
    conn.commit()
    conn.close()
    migrated = jobs.get_job("legacy", db)
    assert migrated["status"] == "done"
    assert migrated["attempt_id"] is None


def test_event_error_fields_builds_summary_from_structured_diagnostics():
    error_class, error = jobs._event_error_fields({
        "event": "prompt_error", "error_class": "oom",
        "node_id": "12", "node_type": "KSampler",
        "exception_type": "torch.OutOfMemoryError",
        "exception_message": "CUDA out of memory",
    })
    assert error_class == "oom"
    assert error == "torch.OutOfMemoryError: CUDA out of memory at node 12 (KSampler)"
    # events without structured detail carry nothing
    assert jobs._event_error_fields({"event": "prompt_completed"}) == (None, None)
    assert jobs._event_error_fields({"event": "prompt_error"}) == (None, None)


def test_apply_attempt_event_parks_error_detail_on_attempt_row(tmp_path):
    db = tmp_path / "jobs.sqlite3"
    _job, attempt_id, _ = _stale_attempt(db)
    jobs._apply_attempt_event(attempt_id, {
        "event": "prompt_error", "prompt_id": "p1", "elapsed_s": 4.2,
        "error_class": "model_missing", "exception_type": "ValueError",
        "exception_message": "'nope.safetensors' not in list",
        "node_id": "4", "node_type": "CheckpointLoaderSimple",
    }, db)
    conn = jobs._connect(db)
    row = conn.execute(
        "SELECT stage,error,error_class FROM job_attempts WHERE attempt_id=?", (attempt_id,)
    ).fetchone()
    conn.close()
    assert row["stage"] == "remote_error"
    assert row["error_class"] == "model_missing"
    assert "not in list" in row["error"] and "CheckpointLoaderSimple" in row["error"]
